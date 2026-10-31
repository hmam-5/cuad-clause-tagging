"""Classify sampled span errors and audit exact/near-duplicate contracts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import lzma
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuad_tagging.config import BIO_LABELS, PROCESSED_DATA, RAW_DATA, RESULTS
from cuad_tagging.metrics import decode_bio

WORD_PATTERN = re.compile(r"\w+", re.UNICODE)


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["contract_id"] in result:
                raise ValueError(f"Duplicate contract in predictions: {row['contract_id']}")
            result[row["contract_id"]] = row
    return result


def _simhash(text: str) -> int:
    words = [word.lower() for word in WORD_PATTERN.findall(text)]
    if not words:
        return 0
    bits = [0] * 64
    shingle_count = max(1, len(words) - 4)
    sample_step = max(1, (shingle_count + 511) // 512)
    for index in range(0, shingle_count, sample_step):
        shingle = " ".join(words[index : index + 5]) if len(words) >= 5 else " ".join(words)
        fingerprint = int.from_bytes(
            hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(),
            "big",
        )
        for bit in range(64):
            bits[bit] += 1 if fingerprint & (1 << bit) else -1
    return sum(1 << bit for bit, value in enumerate(bits) if value >= 0)


def _duplicate_audit(
    raw_path: Path, processed_path: Path, hamming_threshold: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    split_by_id: dict[str, str] = {}
    with lzma.open(processed_path, "rt", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            split_by_id[record["contract_id"]] = record["split"]
    with raw_path.open(encoding="utf-8") as source:
        payload = json.load(source)
    fingerprints: list[tuple[str, str, str, int]] = []
    exact_hashes: dict[str, list[str]] = defaultdict(list)
    for document in payload["data"]:
        contract_id = document["title"].strip()
        context = document["paragraphs"][0]["context"]
        normalized = " ".join(context.split()).casefold()
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        exact_hashes[digest].append(contract_id)
        fingerprints.append(
            (contract_id, split_by_id[contract_id], digest, _simhash(context))
        )

    exact_groups = [
        ids for ids in exact_hashes.values() if len(ids) > 1
    ]
    near_pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(fingerprints):
        for right in fingerprints[left_index + 1 :]:
            distance = (left[3] ^ right[3]).bit_count()
            is_exact = left[2] == right[2]
            if is_exact or distance <= hamming_threshold:
                near_pairs.append(
                    {
                        "contract_a": left[0],
                        "split_a": left[1],
                        "contract_b": right[0],
                        "split_b": right[1],
                        "exact_text_match": is_exact,
                        "simhash_hamming_distance": distance,
                        "cross_split": left[1] != right[1],
                    }
                )
    report = {
        "contracts": len(fingerprints),
        "exact_duplicate_groups": exact_groups,
        "exact_duplicate_contract_pairs": sum(len(ids) * (len(ids) - 1) // 2 for ids in exact_groups),
        "near_duplicate_pairs_at_or_below_hamming_threshold": len(near_pairs),
        "cross_split_near_duplicate_pairs": sum(pair["cross_split"] for pair in near_pairs),
        "simhash_hamming_threshold": hamming_threshold,
        "method": (
            "64-bit SimHash over lowercased five-word shingles, using up to 512 "
            "uniformly spaced shingles per contract. It is approximate and may "
            "miss shared boilerplate within otherwise different contracts."
        ),
    }
    return report, near_pairs


def _classify_errors(
    contract_id: str, gold_tags: list[str], predicted_tags: list[str], tokens: list[str]
) -> list[dict[str, Any]]:
    gold_spans = decode_bio(gold_tags)
    predicted_spans = decode_bio(predicted_tags)
    candidates: list[tuple[float, int, int]] = []
    for predicted_index, predicted in enumerate(predicted_spans):
        for gold_index, gold in enumerate(gold_spans):
            intersection = max(0, min(predicted[2], gold[2]) - max(predicted[1], gold[1]))
            if intersection:
                union = max(predicted[2], gold[2]) - min(predicted[1], gold[1])
                candidates.append((intersection / union, predicted_index, gold_index))
    unmatched_gold = set(range(len(gold_spans)))
    unmatched_predicted = set(range(len(predicted_spans)))
    errors: list[dict[str, Any]] = []
    for _, predicted_index, gold_index in sorted(candidates, reverse=True):
        if predicted_index not in unmatched_predicted or gold_index not in unmatched_gold:
            continue
        predicted, gold = predicted_spans[predicted_index], gold_spans[gold_index]
        unmatched_predicted.remove(predicted_index)
        unmatched_gold.remove(gold_index)
        if predicted[0] != gold[0]:
            error_type = "wrong_category"
        elif predicted[1:] != gold[1:]:
            error_type = "boundary_error"
        else:
            continue
        errors.append(_error_row(contract_id, error_type, gold, predicted, tokens))
    for gold_index in unmatched_gold:
        errors.append(
            _error_row(contract_id, "missed_span", gold_spans[gold_index], None, tokens)
        )
    for predicted_index in unmatched_predicted:
        errors.append(
            _error_row(
                contract_id,
                "hallucinated_span",
                None,
                predicted_spans[predicted_index],
                tokens,
            )
        )
    return errors


def _error_row(
    contract_id: str,
    error_type: str,
    gold: tuple[str, int, int] | None,
    predicted: tuple[str, int, int] | None,
    tokens: list[str],
) -> dict[str, Any]:
    spans = [span for span in (gold, predicted) if span is not None]
    start = min(span[1] for span in spans)
    end = max(span[2] for span in spans)
    context_start = max(0, start - 12)
    context_end = min(len(tokens), end + 12)
    return {
        "contract_id": contract_id,
        "error_type": error_type,
        "gold_category": gold[0] if gold else "",
        "gold_start_token": gold[1] if gold else "",
        "gold_end_token": gold[2] if gold else "",
        "predicted_category": predicted[0] if predicted else "",
        "predicted_start_token": predicted[1] if predicted else "",
        "predicted_end_token": predicted[2] if predicted else "",
        "token_context": " ".join(tokens[context_start:context_end]),
    }


def audit(
    predictions_path: Path,
    data_path: Path = PROCESSED_DATA / "contracts.jsonl.xz",
    raw_path: Path = RAW_DATA,
    output_dir: Path = RESULTS / "audit",
    seed: int = 42,
    sample_size: int = 30,
    hamming_threshold: int = 4,
) -> dict[str, Any]:
    predictions = load_predictions(predictions_path)
    test_records: dict[str, dict[str, Any]] = {}
    with lzma.open(data_path, "rt", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record["split"] == "test" and record["contract_id"] in predictions:
                test_records[record["contract_id"]] = record
    if set(test_records) != set(predictions):
        raise ValueError("Prediction contracts do not match the preprocessed test split.")

    errors: list[dict[str, Any]] = []
    for contract_id, row in predictions.items():
        record = test_records[contract_id]
        if row["gold"] != [BIO_LABELS[index] for index in record["labels"]]:
            raise ValueError(f"Prediction gold labels differ for {contract_id}.")
        errors.extend(
            _classify_errors(
                contract_id, row["gold"], row["predicted"], record["tokens"]
            )
        )
    counts: dict[str, int] = defaultdict(int)
    for error in errors:
        counts[error["error_type"]] += 1
    rng = random.Random(seed)
    sample = rng.sample(errors, min(sample_size, len(errors)))
    sampled_counts: dict[str, int] = defaultdict(int)
    for error in sample:
        sampled_counts[error["error_type"]] += 1
    output_dir.mkdir(parents=True, exist_ok=True)
    error_path = output_dir / "sampled_errors.csv"
    with error_path.open("w", encoding="utf-8-sig", newline="") as output:
        fields = (
            "contract_id",
            "error_type",
            "gold_category",
            "gold_start_token",
            "gold_end_token",
            "predicted_category",
            "predicted_start_token",
            "predicted_end_token",
            "token_context",
        )
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sample)

    duplicate_report, duplicate_pairs = _duplicate_audit(
        raw_path, data_path, hamming_threshold
    )
    with (output_dir / "near_duplicate_pairs.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as output:
        fields = (
            "contract_a",
            "split_a",
            "contract_b",
            "split_b",
            "exact_text_match",
            "simhash_hamming_distance",
            "cross_split",
        )
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(duplicate_pairs)
    report = {
        "prediction_file": predictions_path.name,
        "test_contracts": len(predictions),
        "error_counts": dict(sorted(counts.items())),
        "total_span_errors": len(errors),
        "sampled_errors": len(sample),
        "sampled_error_counts": dict(sorted(sampled_counts.items())),
        "error_sample": error_path.name,
        "duplicate_audit": duplicate_report,
        "near_duplicate_pairs_file": "near_duplicate_pairs.csv",
    }
    with (output_dir / "audit_summary.json").open("w", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--data", type=Path, default=PROCESSED_DATA / "contracts.jsonl.xz")
    parser.add_argument("--raw", type=Path, default=RAW_DATA)
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "audit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--hamming-threshold", type=int, default=4)
    args = parser.parse_args()
    report = audit(
        args.predictions,
        args.data,
        args.raw,
        args.output_dir,
        args.seed,
        args.sample_size,
        args.hamming_threshold,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
