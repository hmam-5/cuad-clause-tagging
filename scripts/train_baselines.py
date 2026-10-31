"""Train the CUAD majority and feature-based CRF baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuad_tagging.baselines import run_baselines
from cuad_tagging.config import PROCESSED_DATA, RESULTS, SEED


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=PROCESSED_DATA / "contracts.jsonl.xz"
    )
    parser.add_argument("--model", choices=("all", "o-only", "most-frequent-word", "crf"), default="all")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-train-contracts", type=int)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    results = run_baselines(
        args.data,
        args.model,
        args.seed,
        args.max_train_contracts,
        args.bootstrap_iterations,
        args.output_dir,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
