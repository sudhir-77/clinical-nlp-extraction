"""Reproduce every result: python run.py

Optional flags: --quick, --device cpu, --skip-tests, --results-dir PATH
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from clinical_nlp.env import configure_runtime

configure_runtime(ROOT)

from clinical_nlp.hardware import PeakMemory
from clinical_nlp.report import criterion, update_readme, write_json, write_resume_notes
from clinical_nlp.runner import run_pipeline


def _coverage_fraction(coverage_path: Path) -> float | None:
    if not coverage_path.exists():
        return None
    payload = json.loads(coverage_path.read_text(encoding="utf-8"))
    totals = payload.get("totals") or {}
    percent = totals.get("percent_covered")
    if percent is None:
        return None
    return float(percent) / 100.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate clinical NLP extraction")
    parser.add_argument("--quick", action="store_true", help="Small subset, tiny model, under 3 minutes")
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    parser.add_argument("--skip-tests", action="store_true", help="Do not measure pytest coverage")
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    monitor = PeakMemory()
    monitor.start()
    results = args.results_dir or (ROOT / "results")
    results.mkdir(parents=True, exist_ok=True)
    # Coverage runs before training so the pytest process is gone before the
    # biomedical checkpoint is loaded. Peak RSS is the max of the two, not the sum.
    if not args.quick and not args.skip_tests and args.results_dir is None:
        print("running pytest for coverage", flush=True)
        completed = subprocess.run([sys.executable, "-m", "pytest"], cwd=ROOT, check=False)
        if completed.returncode != 0:
            monitor.stop()
            print(f"pytest failed with exit code {completed.returncode}", flush=True)
            return completed.returncode
    payload = run_pipeline(
        ROOT,
        quick=args.quick,
        device_name=args.device,
        results_dir=args.results_dir,
    )
    if not args.quick and not args.skip_tests and args.results_dir is None:
        covered = _coverage_fraction(results / "coverage.json")
        if covered is None:
            covered = 0.0
        payload["success_criteria"].append(
            criterion("Test coverage of src/", covered, ">=", 0.75)
        )
        payload["metrics"]["coverage"] = {"line_rate": covered}
    monitor.stop()
    payload["hardware"]["peak_rss_gb"] = monitor.peak_gb
    payload["runtime_seconds"] = round(time.perf_counter() - started, 3)
    write_json(results / "metrics.json", payload)
    if not args.quick and args.results_dir is None:
        update_readme(ROOT / "README.md", payload)
        write_resume_notes(ROOT / "RESUME_NOTES.md", payload)
    for row in payload["success_criteria"]:
        print(f"{row['passed']!s:5} {row['value']:.4f} {row['name']}", flush=True)
    print(f"runtime_seconds {payload['runtime_seconds']}", flush=True)
    if any(not row["passed"] for row in payload["success_criteria"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
