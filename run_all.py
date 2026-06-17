"""Run the full analysis pipeline for all sections."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

SECTIONS = [
    ("01_control_stationarity_cv", "run.py", ["--ensemble-samples", "{ensemble_samples}", "--bootstrap-reps", "{bootstrap_reps}"]),
    ("02_drought_control_flux_entropy", "run.py", ["--ensemble-samples", "{ensemble_samples}"]),
    ("03_frr_moisture_respiration", "run.py", []),
    ("04_entropy_component_regression", "run.py", []),
    ("SI_mfinale_sensitivity", "run.py", []),
]


def run_section(folder: str, script: str, args: list[str]) -> int:
    script_path = PROJECT_ROOT / folder / script
    print(f"\n{'=' * 60}")
    print(f"Running {folder}/{script} {' '.join(args)}")
    print("=" * 60)
    result = subprocess.run([sys.executable, str(script_path)] + args, cwd=PROJECT_ROOT / folder)
    if result.returncode != 0:
        print(f"ERROR: {folder}/{script} exited with code {result.returncode}")
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-samples", type=int, default=150, help="Number of CREMA ensemble samples (used in sections 01 and 02)")
    parser.add_argument("--bootstrap-reps", type=int, default=1000, help="Number of bootstrap replicates for CV (used in section 01)")
    args = parser.parse_args()

    fmt = {"ensemble_samples": str(args.ensemble_samples), "bootstrap_reps": str(args.bootstrap_reps)}
    overall_success = True
    for folder, script, extra_args in SECTIONS:
        rendered = [a.format(**fmt) for a in extra_args]
        code = run_section(folder, script, rendered)
        if code != 0:
            overall_success = False
            break

    print("\n" + "=" * 60)
    if overall_success:
        print("All sections completed successfully.")
    else:
        print("Pipeline stopped due to an error.")
    print("=" * 60)


if __name__ == "__main__":
    main()
