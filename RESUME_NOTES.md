# Resume notes

## Bullets

- Fine-tuned a biomedical transformer for disease NER on 793 expert-annotated abstracts, reaching 0.876 entity-level F1 vs. 0.699 for a CRF baseline
- Added NegEx-style negation and uncertainty detection (0.830 F1) and served extraction as a FastAPI endpoint with 100% schema-valid JSON
- Analyzed 50 test errors by boundary, miss and false-positive type to guide the next labeling pass

## Skills and tools

Python, PyTorch, Hugging Face Transformers, NCBI Disease corpus, BioScope, sklearn-crfsuite, seqeval, FastAPI, pydantic, pytest

## Summary

This project fine-tunes a biomedical transformer to find disease mentions in PubMed abstracts and compares it with a dictionary-plus-CRF baseline (0.876 vs. 0.699 entity F1 on the official NCBI Disease test split). A NegEx-style module labels negation (0.830 F1 on held-out BioScope abstract sentences) and the same pipeline is served as JSON from FastAPI.

## GitHub description

Fine-tune a biomedical transformer for disease NER, compare it with a dictionary CRF, and serve NegEx-style extraction as a FastAPI endpoint.
