"""Thread caps before NumPy or PyTorch are imported."""

from __future__ import annotations

from pathlib import Path

from clinical_nlp.env import configure_runtime

ROOT = Path(__file__).resolve().parents[1]
configure_runtime(ROOT)
(ROOT / "results").mkdir(exist_ok=True)
