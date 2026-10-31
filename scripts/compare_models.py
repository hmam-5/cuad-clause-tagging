"""Paired contract bootstrap comparison of two saved prediction files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuad_tagging.config import RESULTS, SEED
from cuad_tagging.metrics import (
    paired_bootstrap_difference,
    span_metrics,
    token_metrics,
)


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            contract_id = row["contract_id"]
            if contract_id in predictions:
                raise ValueError(f"Duplicate contract in {path}: {contract_id}")
            if len(row["gold"]) != len(row["predicted"]):
                raise ValueError(f"Token count mismatch for {contract_id} in {path}")
            predictions[contract_id] = row
    return predictions


def compare(
    first_path: Path,
    second_path: Path,
    iterations: int = 1000,
    seed: int = SEED,
) -> dict[str, Any]:
    first = load_predictions(first_path)
    second = load_predictions(second_path)
    if not first or first.keys() != second.keys():
        raise ValueError("Prediction files must contain the same non-empty contracts.")
    contract_ids = sorted(first)
    gold = [first[key]["gold"] for key in contract_ids]
    if any(first[key]["gold"] != second[key]["gold"] for key in contract_ids):
        raise ValueError("Gold labels differ between prediction files.")
    first_predicted = [first[key]["predicted"] for key in contract_ids]
    second_predicted = [second[key]["predicted"] for key in contract_ids]
    result = {
        "first_predictions": str(first_path),
        "second_predictions": str(second_path),
        "contracts": len(contract_ids),
        "first_token_metrics": token_metrics(gold, first_predicted),
        "second_token_metrics": token_metrics(gold, second_predicted),
        "first_span_metrics": span_metrics(gold, first_predicted),
        "second_span_metrics": span_metrics(gold, second_predicted),
        "paired_contract_bootstrap_macro_f1_difference_first_minus_second": (
            paired_bootstrap_difference(
                gold,
                first_predicted,
                second_predicted,
                iterations=iterations,
                seed=seed,
            )
        ),
        "interpretation": (
            "A 95% difference interval containing zero is not evidence that either "
            "model is better; report the paired test and uncertainty."
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=RESULTS / "paired_comparison.json")
    args = parser.parse_args()
    result = compare(args.first, args.second, args.iterations, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(result["paired_contract_bootstrap_macro_f1_difference_first_minus_second"], indent=2))


if __name__ == "__main__":
    main()
