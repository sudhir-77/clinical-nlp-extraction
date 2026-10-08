"""CPU fine-tuning of a biomedical transformer for disease NER."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BertConfig,
    BertForTokenClassification,
    BertTokenizerFast,
    PreTrainedTokenizerBase,
)

from clinical_nlp.metrics import entity_scores, repair_bio

LABELS = ["O", "B-DISEASE", "I-DISEASE"]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"


def limit_threads() -> None:
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _enter_train(model: torch.nn.Module, n_unfreeze: int | None) -> None:
    model.train()
    if n_unfreeze is None:
        return
    layers = model.bert.encoder.layer
    frozen = max(0, len(layers) - n_unfreeze)
    model.bert.embeddings.eval()
    for layer in layers[:frozen]:
        layer.eval()


def set_trainable(model: torch.nn.Module, n_unfreeze: int | None) -> None:
    if n_unfreeze is None:
        for parameter in model.parameters():
            parameter.requires_grad = True
        return
    for parameter in model.parameters():
        parameter.requires_grad = False
    layers = model.bert.encoder.layer
    for layer in layers[max(0, len(layers) - n_unfreeze) :]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    _enter_train(model, n_unfreeze)


def encode_example(
    tokens: list[str],
    tags: list[str] | None,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> tuple[list[int], list[int]]:
    encoded = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    word_ids = encoded.word_ids()
    labels: list[int] = []
    previous = None
    for word_id in word_ids:
        if tags is None or word_id is None:
            labels.append(-100)
        elif word_id != previous:
            labels.append(LABEL2ID[tags[word_id]])
        else:
            labels.append(-100)
        previous = word_id
    return list(encoded["input_ids"]), labels


def predict_tokens(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    tokens: list[str],
    *,
    max_length: int,
    device: torch.device,
) -> list[str]:
    if not tokens:
        return []
    model.eval()
    encoded = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        logits = model(**encoded).logits
    pred_ids = logits.argmax(dim=-1)[0].tolist()
    word_ids = encoded_word_ids(tokens, tokenizer, max_length)
    tags = ["O"] * len(tokens)
    seen: set[int] = set()
    for position, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen or word_id >= len(tokens):
            continue
        seen.add(word_id)
        tags[word_id] = LABELS[pred_ids[position]]
    return repair_bio(tags)


def encoded_word_ids(
    tokens: list[str], tokenizer: PreTrainedTokenizerBase, max_length: int
) -> list[int | None]:
    encoded = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    return list(encoded.word_ids())


def _collate(
    batch: list[tuple[list[int], list[int]]], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(item[0]) for item in batch)
    input_ids, attention, labels = [], [], []
    for ids, labs in batch:
        pad = width - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attention.append([1] * len(ids) + [0] * pad)
        labels.append(labs + [-100] * pad)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def predict_sentences(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    sentences: list[dict],
    *,
    max_length: int,
    device: torch.device,
    batch_size: int,
) -> list[list[str]]:
    model.eval()
    predicted: list[list[str]] = []
    for start in range(0, len(sentences), batch_size):
        chunk = sentences[start : start + batch_size]
        encoded_batch = []
        word_id_rows = []
        for row in chunk:
            ids, labels = encode_example(row["tokens"], row["tags"], tokenizer, max_length)
            encoded_batch.append((ids, labels))
            word_id_rows.append(encoded_word_ids(row["tokens"], tokenizer, max_length))
        input_ids, attention, _labels = _collate(encoded_batch, tokenizer.pad_token_id)
        input_ids = input_ids.to(device)
        attention = attention.to(device)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention).logits
        pred_ids = logits.argmax(dim=-1).tolist()
        for row, ids, words in zip(chunk, pred_ids, word_id_rows):
            tags = ["O"] * len(row["tokens"])
            seen: set[int] = set()
            for position, word_id in enumerate(words):
                if word_id is None or word_id in seen or word_id >= len(tags):
                    continue
                seen.add(word_id)
                tags[word_id] = LABELS[ids[position]]
            predicted.append(repair_bio(tags))
    return predicted


def train_transformer(
    train: list[dict],
    dev: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    model: torch.nn.Module,
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    max_length: int,
    learning_rate: float,
    n_unfreeze: int | None,
    checkpoint: Path,
    min_epochs: int = 2,
    patience: int = 1,
) -> dict[str, float]:
    limit_threads()
    random.seed(seed)
    torch.manual_seed(seed)
    set_trainable(model, n_unfreeze)
    model.to(device)
    examples = [
        encode_example(row["tokens"], row["tags"], tokenizer, max_length) for row in train
    ]
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(parameters, lr=learning_rate, weight_decay=0.01)
    steps_per_epoch = max(1, (len(examples) + batch_size - 1) // batch_size)
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(0.1 * total_steps))
    best_f1 = -1.0
    stall = 0
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    global_step = 0
    history: dict[str, float] = {}
    for epoch in range(epochs):
        _enter_train(model, n_unfreeze)
        order = list(range(len(examples)))
        random.Random(seed + epoch).shuffle(order)
        total_loss = 0.0
        seen = 0
        for start in range(0, len(order), batch_size):
            batch_index = order[start : start + batch_size]
            batch = [examples[index] for index in batch_index]
            input_ids, attention, labels = _collate(batch, tokenizer.pad_token_id)
            input_ids = input_ids.to(device)
            attention = attention.to(device)
            labels = labels.to(device)
            if global_step < warmup:
                scale = learning_rate * float(global_step + 1) / float(warmup)
            else:
                progress = (global_step - warmup) / max(1, total_steps - warmup)
                scale = learning_rate * max(0.0, 1.0 - progress)
            for group in optimizer.param_groups:
                group["lr"] = scale
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=input_ids, attention_mask=attention, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch)
            seen += len(batch)
            global_step += 1
        dev_pred = predict_sentences(
            model,
            tokenizer,
            dev,
            max_length=max_length,
            device=device,
            batch_size=batch_size,
        )
        dev_f1 = entity_scores([row["tags"] for row in dev], dev_pred)["f1"] if dev else 0.0
        history[f"epoch_{epoch + 1}_dev_f1"] = dev_f1
        history[f"epoch_{epoch + 1}_loss"] = total_loss / max(1, seen)
        if dev_f1 > best_f1 + 1e-4:
            best_f1 = dev_f1
            torch.save(model.state_dict(), checkpoint)
            stall = 0
        else:
            stall += 1
        print(
            f"epoch {epoch + 1} loss={history[f'epoch_{epoch + 1}_loss']:.4f} "
            f"dev_f1={dev_f1:.4f}",
            flush=True,
        )
        if epoch + 1 >= min_epochs and stall >= patience:
            break
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    history["best_dev_f1"] = best_f1
    model.eval()
    return history


def load_pretrained(cache_dir: Path, n_unfreeze: int) -> tuple[PreTrainedTokenizerBase, torch.nn.Module]:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=str(cache_dir), use_fast=True
    )
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        cache_dir=str(cache_dir),
        num_labels=len(LABELS),
        id2label={index: label for index, label in enumerate(LABELS)},
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    set_trainable(model, n_unfreeze)
    return tokenizer, model


def build_quick_tokenizer(sentences: list[dict], vocab_path: Path) -> BertTokenizerFast:
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    words = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}
    for sentence in sentences:
        for token in sentence["tokens"]:
            words.add(token.lower())
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    rest = sorted(word for word in words if word not in specials)
    vocab_path.write_text("\n".join(specials + rest) + "\n", encoding="utf-8")
    return BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=True)


def build_quick_model(tokenizer: BertTokenizerFast) -> BertForTokenClassification:
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=128,
        pad_token_id=tokenizer.pad_token_id,
        num_labels=len(LABELS),
        id2label={index: label for index, label in enumerate(LABELS)},
        label2id=LABEL2ID,
    )
    return BertForTokenClassification(config)


class TransformerTagger:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def predict_tags(self, tokens: list[str]) -> list[str]:
        return predict_tokens(
            self.model,
            self.tokenizer,
            tokens,
            max_length=self.max_length,
            device=self.device,
        )
