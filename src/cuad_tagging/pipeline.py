"""CUAD JSON loading, offset-preserving cleaning, tagging, and split creation."""

from __future__ import annotations

import csv
import json
import lzma
import random
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import BIO_LABELS, CATEGORIES, RAW_DATA, ROOT, SEED

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}


@dataclass(frozen=True)
class Annotation:
    category: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int
    raw_start: int
    raw_end: int


def load_contracts(path: Path = RAW_DATA) -> list[dict[str, Any]]:
    """Load the original SQuAD-style CUAD release and validate its structure."""
    if not path.is_file():
        raise FileNotFoundError(
            f"CUAD annotations were not found at {path}. "
            "Run `python scripts/download_data.py` first."
        )
    with path.open(encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload.get("data"), list):
        raise TypeError(f"{path} does not contain a CUAD `data` list.")

    contracts: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    known_categories = set(CATEGORIES)
    for document in payload["data"]:
        contract_id = str(document.get("title", "")).strip()
        paragraphs = document.get("paragraphs", [])
        if not contract_id or len(paragraphs) != 1:
            raise ValueError(f"Unexpected CUAD document structure: {contract_id!r}")
        if contract_id in seen_ids:
            raise ValueError(f"Duplicate contract identifier: {contract_id}")
        seen_ids.add(contract_id)

        paragraph = paragraphs[0]
        context = paragraph.get("context")
        if not isinstance(context, str):
            raise TypeError(f"Missing contract text for {contract_id}")
        annotations: list[Annotation] = []
        for question in paragraph.get("qas", []):
            question_id = str(question.get("id", ""))
            if "__" not in question_id:
                raise ValueError(f"Cannot determine CUAD category from {question_id!r}")
            category = question_id.rsplit("__", 1)[1]
            if category not in known_categories:
                raise ValueError(f"Unknown CUAD category {category!r} in {contract_id}")
            for answer in question.get("answers", []):
                answer_text = answer.get("text", "")
                start = answer.get("answer_start")
                if not answer_text or not isinstance(start, int):
                    continue
                annotations.append(
                    Annotation(category, start, start + len(answer_text), answer_text)
                )
        contracts.append(
            {"contract_id": contract_id, "text": context, "annotations": annotations}
        )
    return contracts


def normalize_text(text: str) -> tuple[str, list[int], Counter[str]]:
    """Normalize whitespace while mapping each output character to its source index."""
    output: list[str] = []
    source_offsets: list[int] = []
    changes: Counter[str] = Counter()
    index = 0
    while index < len(text):
        character = text[index]
        if character == "\r":
            changes["line_endings_normalized"] += 1
            if index + 1 < len(text) and text[index + 1] == "\n":
                index += 1
        elif character == "\f":
            character = "\n"
            changes["form_feeds_normalized"] += 1
        elif character == "\u00a0":
            character = " "
            changes["nonbreaking_spaces_normalized"] += 1
        elif character == "\ufffd":
            changes["replacement_characters_retained"] += 1
        elif character == "\x00":
            changes["null_characters_removed"] += 1
            index += 1
            continue
        elif unicodedata.category(character) == "Cf":
            changes["format_characters_removed"] += 1
            index += 1
            continue
        elif unicodedata.category(character) == "Cc" and not character.isspace():
            changes["control_characters_removed"] += 1
            index += 1
            continue

        if character.isspace():
            if not output or output[-1] != " ":
                output.append(" ")
                source_offsets.append(index)
            else:
                changes["repeated_whitespace_collapsed"] += 1
        else:
            output.append(character)
            source_offsets.append(index)
        index += 1
    left = 0
    right = len(output)
    while left < right and output[left] == " ":
        left += 1
    while right > left and output[right - 1] == " ":
        right -= 1
    return "".join(output[left:right]), source_offsets[left:right], changes


def _find_repair(text: str, annotation: Annotation, radius: int = 32) -> int | None:
    """Return a nearby exact match only when the repair is unambiguous."""
    lower = max(0, annotation.start - radius)
    upper = min(len(text), annotation.start + radius + len(annotation.text))
    matches: list[int] = []
    position = text.find(annotation.text, lower, upper)
    while position != -1:
        matches.append(position)
        position = text.find(annotation.text, position + 1, upper)
    return matches[0] if len(matches) == 1 else None


def validate_annotations(
    text: str, annotations: Iterable[Annotation], contract_id: str
) -> tuple[list[Annotation], list[dict[str, Any]]]:
    """Keep exact spans, repair unique nearby offsets, and report every drop."""
    kept: list[Annotation] = []
    changes: list[dict[str, Any]] = []
    for annotation in annotations:
        start, end = annotation.start, annotation.end
        if start >= 0 and end <= len(text) and text[start:end] == annotation.text:
            kept.append(annotation)
            continue
        repaired = _find_repair(text, annotation)
        if repaired is not None:
            fixed = Annotation(
                annotation.category,
                repaired,
                repaired + len(annotation.text),
                annotation.text,
            )
            kept.append(fixed)
            changes.append(
                {
                    "contract_id": contract_id,
                    "category": annotation.category,
                    "reported_start": start,
                    "reported_end": end,
                    "action": "corrected",
                    "reason": "unique exact match within 32 characters",
                    "text": annotation.text,
                }
            )
        else:
            changes.append(
                {
                    "contract_id": contract_id,
                    "category": annotation.category,
                    "reported_start": start,
                    "reported_end": end,
                    "action": "dropped",
                    "reason": "text mismatch or ambiguous nearby match",
                    "text": annotation.text,
                }
            )
    return kept, changes


def tokenize(text: str, source_offsets: list[int]) -> list[Token]:
    """Tokenize words and punctuation and retain normalized and raw offsets."""
    tokens: list[Token] = []
    for match in TOKEN_PATTERN.finditer(text):
        start, end = match.span()
        raw_start = source_offsets[start]
        raw_end = source_offsets[end - 1] + 1
        tokens.append(Token(match.group(), start, end, raw_start, raw_end))
    return tokens


def _normalized_span(
    annotation: Annotation, normalized_offsets: list[int], text_length: int
) -> tuple[int, int]:
    start = _lower_bound(normalized_offsets, annotation.start)
    end = _lower_bound(normalized_offsets, annotation.end)
    return min(start, text_length), min(max(start, end), text_length)


def _lower_bound(values: list[int], target: int) -> int:
    low, high = 0, len(values)
    while low < high:
        middle = (low + high) // 2
        if values[middle] < target:
            low = middle + 1
        else:
            high = middle
    return low


def _merge_intervals(
    intervals: Iterable[tuple[int, int]],
) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def make_windows(
    tags: list[str], max_tokens: int = 512, stride: int = 128
) -> list[dict[str, int]]:
    """Create overlapping windows, shifting boundaries past complete BIO spans."""
    if max_tokens < 1 or stride < 0 or stride >= max_tokens:
        raise ValueError("Require max_tokens > stride >= 0.")
    if not tags:
        return []

    windows: list[dict[str, int]] = []
    start = 0
    while start < len(tags):
        end = min(start + max_tokens, len(tags))
        if end < len(tags):
            span_start = end - 1
            while span_start > start and tags[span_start].startswith("I-"):
                span_start -= 1
            span_end = end
            while span_end < len(tags) and tags[span_end].startswith("I-"):
                span_end += 1
            if span_end - start <= max_tokens:
                end = span_end
            elif span_start > start:
                end = span_start
        windows.append({"start": start, "end": end})
        if end == len(tags):
            break
        if end - start < max_tokens:
            start = end
            continue
        next_start = max(start + 1, end - stride)
        adjusted_start = next_start
        while next_start > start and tags[next_start].startswith("I-"):
            next_start -= 1
        if next_start <= start:
            span_end = adjusted_start
            while span_end < len(tags) and tags[span_end].startswith("I-"):
                span_end += 1
            next_start = span_end if span_end - start <= max_tokens else adjusted_start
        start = next_start
    return windows


def stratified_group_split(
    contract_categories: dict[str, set[str]],
    seed: int = SEED,
    ratios: dict[str, float] | None = None,
) -> dict[str, str]:
    """Greedily balance rare categories while assigning whole contracts."""
    ratios = ratios or SPLIT_RATIOS
    if not ratios or any(value <= 0 for value in ratios.values()):
        raise ValueError("Split ratios must be positive.")
    if abs(sum(ratios.values()) - 1.0) > 1e-8:
        raise ValueError("Split ratios must sum to 1.")

    rng = random.Random(seed)
    names = list(ratios)
    totals = Counter(
        category
        for categories in contract_categories.values()
        for category in categories
    )
    randomized_ids = list(contract_categories)
    rng.shuffle(randomized_ids)
    randomized_ids.sort(
        key=lambda contract_id: (
            min(
                (totals[category] for category in contract_categories[contract_id]),
                default=len(contract_categories) + 1,
            ),
            -len(contract_categories[contract_id]),
        )
    )

    target_sizes = {name: len(contract_categories) * ratios[name] for name in names}
    target_categories = {
        name: {category: count * ratios[name] for category, count in totals.items()}
        for name in names
    }
    sizes = Counter()
    assigned: dict[str, Counter[str]] = {name: Counter() for name in names}
    output: dict[str, str] = {}
    for contract_id in randomized_ids:
        labels = contract_categories[contract_id]
        def score(split: str, category_labels: set[str]) -> tuple[float, float]:
            category_deficit = sum(
                (target_categories[split][label] - assigned[split][label])
                / max(target_categories[split][label], 1.0)
                for label in category_labels
            )
            size_deficit = (target_sizes[split] - sizes[split]) / max(
                target_sizes[split], 1.0
            )
            return category_deficit, size_deficit

        selected = max(names, key=lambda split: score(split, labels))
        output[contract_id] = selected
        sizes[selected] += 1
        assigned[selected].update(labels)
    return output


def _overlap_summary(annotations: list[Annotation]) -> tuple[int, int]:
    ordered = sorted(annotations, key=lambda span: (span.start, span.end))
    overlaps = 0
    overlapping_characters = 0
    active: list[Annotation] = []
    for span in ordered:
        active = [previous for previous in active if previous.end > span.start]
        for previous in active:
            if previous.category != span.category:
                overlaps += 1
                overlapping_characters += max(
                    0, min(previous.end, span.end) - max(previous.start, span.start)
                )
        active.append(span)
    return overlaps, overlapping_characters


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with lzma.open(path, "wt", encoding="utf-8", preset=9) as destination:
        for record in records:
            compact = {
                "contract_id": record["contract_id"],
                "split": record["split"],
                "tokens": [token["text"] for token in record["tokens"]],
                "offsets": [
                    [
                        token["start"],
                        token["end"],
                        token["raw_start"],
                        token["raw_end"],
                    ]
                    for token in record["tokens"]
                ],
                "labels": [BIO_LABELS.index(token["label"]) for token in record["tokens"]],
                "categories": [
                    [CATEGORIES.index(category) for category in token["categories"]]
                    for token in record["tokens"]
                ],
                "spans": [
                    [
                        CATEGORIES.index(span["category"]),
                        span["start"],
                        span["end"],
                        span["raw_start"],
                        span["raw_end"],
                        span["token_start"],
                        span["token_end"],
                        span["text"],
                    ]
                    for span in record["spans"]
                ],
                "windows": [[window["start"], window["end"]] for window in record["windows"]],
            }
            destination.write(
                json.dumps(compact, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def preprocess(
    input_path: Path = RAW_DATA,
    output_dir: Path | None = None,
    seed: int = SEED,
    max_tokens: int = 512,
    stride: int = 128,
) -> dict[str, Any]:
    """Build token labels, contract-level splits, windows, and auditable reports."""
    from .metrics import decode_bio

    destination = output_dir or input_path.parents[1] / "processed"
    destination.mkdir(parents=True, exist_ok=True)
    raw_contracts = load_contracts(input_path)

    category_span_counts: Counter[str] = Counter(
        annotation.category
        for contract in raw_contracts
        for annotation in contract["annotations"]
    )
    prepared: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    cleaning_counts: Counter[str] = Counter()
    all_annotations: list[tuple[str, Annotation]] = []
    raw_spans = 0
    kept_spans = 0
    dropped_spans = 0
    corrected_spans = 0
    overlap_pairs = 0
    overlap_chars = 0
    token_multi_label_count = 0
    token_count = 0
    single_label_o_count = 0
    roundtrip_matches = 0
    roundtrip_total = 0

    for raw_contract in raw_contracts:
        contract_id = raw_contract["contract_id"]
        source_text = raw_contract["text"]
        annotations = raw_contract["annotations"]
        raw_spans += len(annotations)
        valid, changes = validate_annotations(source_text, annotations, contract_id)
        corrections.extend(changes)
        corrected_spans += sum(change["action"] == "corrected" for change in changes)
        dropped_spans += sum(change["action"] == "dropped" for change in changes)
        kept_spans += len(valid)
        normalized_text, offset_map, changes_by_type = normalize_text(source_text)
        cleaning_counts.update(changes_by_type)
        tokens = tokenize(normalized_text, offset_map)

        projected: dict[str, list[tuple[int, int]]] = defaultdict(list)
        span_rows: list[dict[str, Any]] = []
        for annotation in valid:
            norm_start, norm_end = _normalized_span(
                annotation, offset_map, len(normalized_text)
            )
            token_indices = [
                i
                for i, token in enumerate(tokens)
                if token.start < norm_end and token.end > norm_start
            ]
            if not token_indices:
                corrections.append(
                    {
                        "contract_id": contract_id,
                        "category": annotation.category,
                        "reported_start": annotation.start,
                        "reported_end": annotation.end,
                        "action": "dropped",
                        "reason": "annotation did not overlap a token after cleaning",
                        "text": annotation.text,
                    }
                )
                dropped_spans += 1
                kept_spans -= 1
                continue
            token_start, token_end = token_indices[0], token_indices[-1] + 1
            projected[annotation.category].append((token_start, token_end))
            category_span_counts[annotation.category] += 1
            all_annotations.append((contract_id, annotation))
            span_rows.append(
                {
                    "category": annotation.category,
                    "start": norm_start,
                    "end": norm_end,
                    "raw_start": annotation.start,
                    "raw_end": annotation.end,
                    "token_start": token_start,
                    "token_end": token_end,
                    "text": annotation.text,
                }
            )

        merged = {
            category: _merge_intervals(intervals)
            for category, intervals in projected.items()
        }
        token_categories: list[list[str]] = [[] for _ in tokens]
        category_tags: dict[str, list[str]] = {
            category: ["O"] * len(tokens) for category in CATEGORIES
        }
        for category, intervals in merged.items():
            tags = category_tags[category]
            for start, end in intervals:
                tags[start] = f"B-{category}"
                for index in range(start + 1, end):
                    tags[index] = f"I-{category}"
                for index in range(start, end):
                    token_categories[index].append(category)
        single_tags: list[str] = []
        for index, categories in enumerate(token_categories):
            if len(set(categories)) > 1:
                token_multi_label_count += 1
            if not categories:
                single_tags.append("O")
                single_label_o_count += 1
            else:
                winner = min(
                    set(categories),
                    key=lambda category: (
                        category_span_counts[category],
                        CATEGORIES.index(category),
                    ),
                )
                single_tags.append(category_tags[winner][index])
        token_count += len(tokens)

        for category, intervals in merged.items():
            decoded = decode_bio(category_tags[category])
            expected = intervals
            actual = [(start, end) for _, start, end in decoded]
            roundtrip_total += len(expected)
            roundtrip_matches += sum(span in actual for span in expected)

        contract_overlap_pairs, contract_overlap_chars = _overlap_summary(valid)
        overlap_pairs += contract_overlap_pairs
        overlap_chars += contract_overlap_chars
        prepared.append(
            {
                "contract_id": contract_id,
                "text": normalized_text,
                "tokens": [
                    {
                        "text": token.text,
                        "start": token.start,
                        "end": token.end,
                        "raw_start": token.raw_start,
                        "raw_end": token.raw_end,
                        "categories": sorted(set(token_categories[index])),
                        "label": single_tags[index],
                        "multilabel": [
                            int(category in token_categories[index])
                            for category in CATEGORIES
                        ],
                    }
                    for index, token in enumerate(tokens)
                ],
                "spans": span_rows,
            }
        )

    splits = stratified_group_split(
        {
            record["contract_id"]: {
                category for span in record["spans"] for category in [span["category"]]
            }
            for record in prepared
        },
        seed,
    )
    split_counts: dict[str, Counter[str]] = {
        split: Counter() for split in SPLIT_RATIOS
    }
    split_contract_counts = Counter()
    total_windows = 0
    for record in prepared:
        record["split"] = splits[record["contract_id"]]
        labels = [token["label"] for token in record["tokens"]]
        record["windows"] = make_windows(labels, max_tokens, stride)
        total_windows += len(record["windows"])
        split_contract_counts[record["split"]] += 1
        for span in record["spans"]:
            split_counts[record["split"]][span["category"]] += 1

    minimum_test_spans = 5
    rare_test_categories = [
        category
        for category in CATEGORIES
        if split_counts["test"][category] < minimum_test_spans
    ]
    report_path = destination / "span_corrections.csv"
    with report_path.open("w", encoding="utf-8-sig", newline="") as report:
        fields = (
            "contract_id",
            "category",
            "reported_start",
            "reported_end",
            "action",
            "reason",
            "text",
        )
        writer = csv.DictWriter(report, fieldnames=fields)
        writer.writeheader()
        writer.writerows(corrections)

    _write_jsonl(destination / "contracts.jsonl.xz", prepared)
    span_lengths = [span.end - span.start for _, span in all_annotations]
    token_span_lengths = [
        span["token_end"] - span["token_start"]
        for record in prepared
        for span in record["spans"]
    ]
    stats: dict[str, Any] = {
        "seed": seed,
        "contracts": len(prepared),
        "raw_spans": raw_spans,
        "kept_spans": kept_spans,
        "corrected_spans": corrected_spans,
        "dropped_spans": dropped_spans,
        "average_span_characters": (
            sum(span_lengths) / len(span_lengths) if span_lengths else 0
        ),
        "p95_span_characters": _percentile(span_lengths, 0.95),
        "average_span_tokens": (
            sum(token_span_lengths) / len(token_span_lengths)
            if token_span_lengths
            else 0
        ),
        "p95_span_tokens": _percentile(token_span_lengths, 0.95),
        "average_contract_tokens": token_count / max(len(prepared), 1),
        "token_count": token_count,
        "o_token_share": single_label_o_count / max(token_count, 1),
        "tokens_with_multiple_categories": token_multi_label_count,
        "multi_category_token_share": token_multi_label_count / max(token_count, 1),
        "overlapping_annotation_pairs": overlap_pairs,
        "overlapping_annotation_characters": overlap_chars,
        "windows": total_windows,
        "window_size_words": max_tokens,
        "window_stride_words": stride,
        "roundtrip_span_agreement": (
            roundtrip_matches / roundtrip_total if roundtrip_total else 1.0
        ),
        "cleaning_changes": dict(cleaning_counts),
        "split_contract_counts": dict(split_contract_counts),
        "per_category_spans": {
            category: {
                split: split_counts[split][category] for split in SPLIT_RATIOS
            }
            for category in CATEGORIES
        },
        "categories_below_minimum_test_spans": rare_test_categories,
        "minimum_test_spans": minimum_test_spans,
        "source_file": _display_path(input_path),
        "correction_report": _display_path(report_path),
    }
    with (destination / "stats.json").open("w", encoding="utf-8") as output:
        json.dump(stats, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return stats


def _percentile(values: list[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * quantile)))
    return float(ordered[index])


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name
