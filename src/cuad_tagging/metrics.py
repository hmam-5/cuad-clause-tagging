"""Token, span, and contract-bootstrap evaluation helpers."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable, Sequence

from .config import CATEGORIES


def decode_bio(tags: Sequence[str]) -> list[tuple[str, int, int]]:
    """Decode BIO tags into category and half-open token-index spans."""
    spans: list[tuple[str, int, int]] = []
    category: str | None = None
    start = 0
    for index, tag in enumerate((*tags, "O")):
        if tag == "O" or "-" not in tag:
            next_category = None
            prefix = "O"
        else:
            prefix, next_category = tag.split("-", 1)
            if prefix not in {"B", "I"}:
                next_category = None
                prefix = "O"
        if category is not None and (
            next_category != category or prefix == "B"
        ):
            spans.append((category, start, index))
            category = None
        if next_category is not None and (prefix == "B" or category is None):
            category = next_category
            start = index
    return spans


def _category(tag: str) -> str | None:
    if tag == "O" or "-" not in tag:
        return None
    prefix, category = tag.split("-", 1)
    return category if prefix in {"B", "I"} else None


def _prf(counts: Counter[str], categories: Iterable[str]) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for category in categories:
        tp = counts[f"tp:{category}"]
        fp = counts[f"fp:{category}"]
        fn = counts[f"fn:{category}"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        scores[category] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(tp + fn),
        }
    return scores


def token_metrics(
    gold_sequences: Sequence[Sequence[str]],
    predicted_sequences: Sequence[Sequence[str]],
) -> dict[str, object]:
    """Calculate macro F1 over all 41 clause classes, excluding outside tokens."""
    if len(gold_sequences) != len(predicted_sequences):
        raise ValueError("Gold and predicted sequence counts differ.")
    counts = _token_counts(gold_sequences, predicted_sequences)
    per_category = _prf(counts, CATEGORIES)
    return {
        "macro_f1": sum(score["f1"] for score in per_category.values())
        / len(CATEGORIES),
        "per_category": per_category,
        "metric": "token-level macro F1 over 41 categories (O excluded)",
    }


def _token_counts(
    gold_sequences: Sequence[Sequence[str]],
    predicted_sequences: Sequence[Sequence[str]],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for gold, predicted in zip(gold_sequences, predicted_sequences):
        if len(gold) != len(predicted):
            raise ValueError("Gold and predicted token counts differ.")
        for gold_tag, predicted_tag in zip(gold, predicted):
            gold_category, predicted_category = _category(gold_tag), _category(
                predicted_tag
            )
            for category in {gold_category, predicted_category} - {None}:
                if gold_category == predicted_category == category:
                    counts[f"tp:{category}"] += 1
                elif predicted_category == category:
                    counts[f"fp:{category}"] += 1
                elif gold_category == category:
                    counts[f"fn:{category}"] += 1
    return counts


def _sum_counts(
    per_contract: Sequence[Counter[str]], sample: Sequence[int]
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for index in sample:
        counts.update(per_contract[index])
    return counts


def _macro_f1(counts: Counter[str]) -> float:
    return sum(score["f1"] for score in _prf(counts, CATEGORIES).values()) / len(
        CATEGORIES
    )


def _span_counts(
    gold_sequences: Sequence[Sequence[str]],
    predicted_sequences: Sequence[Sequence[str]],
    partial: bool,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for contract_index, (gold_tags, predicted_tags) in enumerate(
        zip(gold_sequences, predicted_sequences)
    ):
        gold = decode_bio(gold_tags)
        predicted = decode_bio(predicted_tags)
        unmatched_gold = set(range(len(gold)))
        unmatched_predicted = set(range(len(predicted)))
        candidates: list[tuple[float, int, int]] = []
        for predicted_index, predicted_span in enumerate(predicted):
            for gold_index, gold_span in enumerate(gold):
                if predicted_span[0] != gold_span[0]:
                    continue
                intersection = max(
                    0, min(predicted_span[2], gold_span[2]) - max(predicted_span[1], gold_span[1])
                )
                if not intersection:
                    continue
                if not partial and predicted_span[1:] != gold_span[1:]:
                    continue
                union = max(predicted_span[2], gold_span[2]) - min(
                    predicted_span[1], gold_span[1]
                )
                candidates.append((intersection / union, predicted_index, gold_index))
        for _, predicted_index, gold_index in sorted(candidates, reverse=True):
            if (
                predicted_index in unmatched_predicted
                and gold_index in unmatched_gold
            ):
                unmatched_predicted.remove(predicted_index)
                unmatched_gold.remove(gold_index)
                counts[f"tp:{gold[gold_index][0]}"] += 1
        for predicted_index in unmatched_predicted:
            counts[f"fp:{predicted[predicted_index][0]}"] += 1
        for gold_index in unmatched_gold:
            counts[f"fn:{gold[gold_index][0]}"] += 1
    return counts


def span_metrics(
    gold_sequences: Sequence[Sequence[str]],
    predicted_sequences: Sequence[Sequence[str]],
) -> dict[str, object]:
    """Report strict and one-to-one partial-overlap span F1."""
    if len(gold_sequences) != len(predicted_sequences):
        raise ValueError("Gold and predicted sequence counts differ.")
    for gold, predicted in zip(gold_sequences, predicted_sequences):
        if len(gold) != len(predicted):
            raise ValueError("Gold and predicted token counts differ.")
    result: dict[str, object] = {}
    for name, partial in (("strict", False), ("partial", True)):
        counts = _span_counts(gold_sequences, predicted_sequences, partial)
        per_category = _prf(counts, CATEGORIES)
        micro_tp = sum(counts[f"tp:{category}"] for category in CATEGORIES)
        micro_fp = sum(counts[f"fp:{category}"] for category in CATEGORIES)
        micro_fn = sum(counts[f"fn:{category}"] for category in CATEGORIES)
        precision = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
        recall = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
        micro_f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        result[name] = {
            "micro_f1": micro_f1,
            "macro_f1": sum(score["f1"] for score in per_category.values())
            / len(CATEGORIES),
            "per_category": per_category,
        }
    return result


def bootstrap_macro_f1(
    gold_sequences: Sequence[Sequence[str]],
    predicted_sequences: Sequence[Sequence[str]],
    iterations: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Percentile confidence interval by resampling whole contracts."""
    if len(gold_sequences) != len(predicted_sequences):
        raise ValueError("Gold and predicted sequence counts differ.")
    if not gold_sequences:
        raise ValueError("At least one contract is required for bootstrap.")
    if iterations < 1:
        raise ValueError("Bootstrap iterations must be positive.")
    per_contract = [
        _token_counts([gold], [predicted])
        for gold, predicted in zip(gold_sequences, predicted_sequences)
    ]
    rng = random.Random(seed)
    scores: list[float] = []
    for _ in range(iterations):
        sample = [rng.randrange(len(gold_sequences)) for _ in gold_sequences]
        scores.append(_macro_f1(_sum_counts(per_contract, sample)))
    scores.sort()
    return {
        "lower": scores[int(0.025 * (iterations - 1))],
        "upper": scores[int(0.975 * (iterations - 1))],
        "iterations": float(iterations),
        "seed": float(seed),
    }


def paired_bootstrap_difference(
    gold_sequences: Sequence[Sequence[str]],
    first_predictions: Sequence[Sequence[str]],
    second_predictions: Sequence[Sequence[str]],
    iterations: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired contract bootstrap for the macro-F1 difference (first minus second)."""
    if not (
        len(gold_sequences)
        == len(first_predictions)
        == len(second_predictions)
    ):
        raise ValueError("Gold and both prediction sequence counts must match.")
    if not gold_sequences or iterations < 1:
        raise ValueError("Contracts and positive bootstrap iterations are required.")
    first_counts = [
        _token_counts([gold], [predicted])
        for gold, predicted in zip(gold_sequences, first_predictions)
    ]
    second_counts = [
        _token_counts([gold], [predicted])
        for gold, predicted in zip(gold_sequences, second_predictions)
    ]
    rng = random.Random(seed)
    differences: list[float] = []
    for _ in range(iterations):
        sample = [rng.randrange(len(gold_sequences)) for _ in gold_sequences]
        differences.append(
            _macro_f1(_sum_counts(first_counts, sample))
            - _macro_f1(_sum_counts(second_counts, sample))
        )
    differences.sort()
    lower_tail = sum(value <= 0 for value in differences)
    upper_tail = sum(value >= 0 for value in differences)
    return {
        "mean_difference": sum(differences) / len(differences),
        "lower": differences[int(0.025 * (iterations - 1))],
        "upper": differences[int(0.975 * (iterations - 1))],
        "two_sided_p": min(
            1.0,
            2 * (1 + min(lower_tail, upper_tail)) / (iterations + 1),
        ),
        "iterations": float(iterations),
        "seed": float(seed),
    }
