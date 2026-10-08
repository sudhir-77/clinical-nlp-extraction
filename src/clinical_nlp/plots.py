"""Plots written from computed metrics, not from hand-entered numbers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_plots(results_dir: Path, payload: dict, error_counts: dict[str, int]) -> list[str]:
    results_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    ner = payload["metrics"]["ncbi_disease"]
    figure, axis = plt.subplots(figsize=(6, 4))
    labels = ["CRF baseline", "Transformer"]
    values = [ner["crf"]["f1"], ner["transformer"]["f1"]]
    axis.bar(labels, values, color=["#7f8c8d", "#1f4e79"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Entity F1")
    axis.set_title("NCBI Disease official test split")
    figure.tight_layout()
    f1_path = results_dir / "f1_comparison.png"
    figure.savefig(f1_path, dpi=120)
    plt.close(figure)
    names.append(f1_path.name)

    figure, axis = plt.subplots(figsize=(6, 4))
    cats = ["boundary", "missed", "false_positive"]
    axis.bar(cats, [error_counts.get(cat, 0) for cat in cats], color="#8c4a32")
    axis.set_ylabel("Test errors")
    axis.set_title("Exact-span error categories")
    figure.tight_layout()
    err_path = results_dir / "error_categories.png"
    figure.savefig(err_path, dpi=120)
    plt.close(figure)
    names.append(err_path.name)
    return names
