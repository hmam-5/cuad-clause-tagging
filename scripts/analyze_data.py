"""Generate CSV/LaTeX analysis tables and readable dataset figures."""

from __future__ import annotations

import argparse
import csv
import json
import lzma
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuad_tagging.config import (
    BIO_LABELS,
    CATEGORIES,
    PROCESSED_DATA,
    RAW_DATA,
    RESULTS,
)


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Prepared data not found: {path}")
    opener = lzma.open if path.suffix == ".xz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def write_table(
    directory: Path, stem: str, headers: list[str], rows: list[list[object]]
) -> None:
    with (directory / f"{stem}.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as output:
        writer = csv.writer(output)
        writer.writerow(headers)
        writer.writerows(rows)
    alignment = "l" + "r" * (len(headers) - 1)
    with (directory / f"{stem}.tex").open("w", encoding="utf-8") as output:
        output.write("\\begin{tabular}{" + alignment + "}\n\\toprule\n")
        output.write(" & ".join(_escape_latex(value) for value in headers) + " \\\\\n")
        output.write("\\midrule\n")
        for row in rows:
            output.write(
                " & ".join(_escape_latex(value) for value in row) + " \\\\\n"
            )
        output.write("\\bottomrule\n\\end{tabular}\n")


def _escape_latex(value: object) -> str:
    text = str(value)
    for character, replacement in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    ):
        text = text.replace(character, replacement)
    return text


def audit_source_annotations(raw_path: Path) -> dict[str, Any]:
    with raw_path.open(encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload.get("data"), list):
        raise TypeError(f"{raw_path} does not contain a CUAD `data` list.")
    counts: Counter[str] = Counter()
    category_spans: Counter[str] = Counter()
    for document in payload["data"]:
        for question in document["paragraphs"][0]["qas"]:
            counts["questions"] += 1
            answers = question.get("answers")
            if "answers" not in question or answers is None:
                counts["missing_answer_lists"] += 1
            if not answers:
                counts["empty_answer_questions"] += 1
                continue
            for answer in answers:
                counts["answer_entries"] += 1
                text = answer.get("text", "")
                if not text:
                    counts["empty_answer_entries"] += 1
                elif not isinstance(answer.get("answer_start"), int):
                    counts["nonempty_answers_missing_offsets"] += 1
                else:
                    counts["valid_nonempty_spans"] += 1
                    category = str(question["id"]).rsplit("__", 1)[-1]
                    category_spans[category] += 1
    for key in (
        "questions",
        "missing_answer_lists",
        "empty_answer_questions",
        "answer_entries",
        "empty_answer_entries",
        "nonempty_answers_missing_offsets",
        "valid_nonempty_spans",
    ):
        counts.setdefault(key, 0)
    return {
        **dict(counts),
        "nonempty_span_counts_by_category": dict(sorted(category_spans.items())),
    }


def analyze(
    data_path: Path,
    stats_path: Path,
    results_dir: Path,
    raw_path: Path = RAW_DATA,
) -> dict[str, Any]:
    if not stats_path.is_file():
        raise FileNotFoundError(f"Preprocessing statistics not found: {stats_path}")
    with stats_path.open(encoding="utf-8") as source:
        stats = json.load(source)
    results_dir.mkdir(parents=True, exist_ok=True)
    source_audit = audit_source_annotations(raw_path)
    with (results_dir / "source_annotation_audit.json").open(
        "w", encoding="utf-8"
    ) as output:
        json.dump(source_audit, output, ensure_ascii=False, indent=2)
        output.write("\n")
    write_table(
        results_dir,
        "source_annotation_audit",
        ["metric", "count"],
        [
            [key, value]
            for key, value in source_audit.items()
            if key != "nonempty_span_counts_by_category"
        ],
    )
    write_table(
        results_dir,
        "source_spans_by_category",
        ["category", "nonempty_spans"],
        [
            [category, count]
            for category, count in source_audit[
                "nonempty_span_counts_by_category"
            ].items()
        ],
    )

    token_counts: Counter[str] = Counter()
    bio_counts: Counter[str] = Counter()
    span_counts: Counter[str] = Counter()
    span_lengths: dict[str, list[int]] = defaultdict(list)
    contract_lengths: list[int] = []
    window_counts: list[int] = []
    split_contracts: Counter[str] = Counter()
    split_windows: Counter[str] = Counter()
    split_spans: dict[str, Counter[str]] = defaultdict(Counter)
    overlap_pairs: Counter[tuple[str, str]] = Counter()
    total_tokens = 0
    o_tokens = 0
    contract_total = 0
    for record in iter_records(data_path):
        contract_total += 1
        tokens = record["tokens"]
        total_tokens += len(tokens)
        contract_lengths.append(len(tokens))
        window_counts.append(len(record.get("windows", [])))
        split = record["split"]
        split_contracts[split] += 1
        split_windows[split] += len(record.get("windows", []))
        for label_id, categories in zip(record["labels"], record["categories"]):
            bio_label = BIO_LABELS[label_id]
            bio_counts[bio_label] += 1
            if bio_label == "O":
                o_tokens += 1
            for category_id in categories:
                token_counts[CATEGORIES[category_id]] += 1
        for span in record["spans"]:
            category = CATEGORIES[span[0]]
            span_counts[category] += 1
            split_spans[split][category] += 1
            span_lengths[category].append(span[6] - span[5])
        ordered = sorted(record["spans"], key=lambda span: span[3])
        active: list[list[Any]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for span in ordered:
            active = [
                previous
                for previous in active
                if previous[4] > span[3]
            ]
            for previous in active:
                if previous[0] != span[0]:
                    pair = tuple(
                        sorted((CATEGORIES[previous[0]], CATEGORIES[span[0]]))
                    )
                    seen_pairs.add(pair)
            active.append(span)
        for pair in seen_pairs:
            overlap_pairs[pair] += 1

    rows = [
        [
            category,
            token_counts[category],
            span_counts[category],
            *(split_spans[split][category] for split in ("train", "validation", "test")),
            len(span_lengths[category]),
        ]
        for category in CATEGORIES
    ]
    write_table(
        results_dir,
        "class_distribution",
        [
            "category",
            "positive_tokens",
            "spans",
            "train_spans",
            "validation_spans",
            "test_spans",
            "span_count",
        ],
        rows,
    )
    write_table(
        results_dir,
        "bio_label_distribution",
        ["bio_label", "token_count"],
        [[label, bio_counts[label]] for label in BIO_LABELS],
    )
    overlap_rows = [
        [left, right, count]
        for (left, right), count in sorted(
            overlap_pairs.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    write_table(
        results_dir,
        "overlap_pairs",
        ["category_a", "category_b", "contract_count"],
        overlap_rows,
    )
    summary_rows = [
        ["contracts", contract_total],
        ["tokens", total_tokens],
        ["outside_token_share", f"{o_tokens / max(total_tokens, 1):.6f}"],
        ["average_tokens_per_contract", f"{sum(contract_lengths) / max(contract_total, 1):.2f}"],
        ["average_windows_per_contract", f"{sum(window_counts) / max(contract_total, 1):.2f}"],
        ["raw_spans", stats.get("raw_spans", 0)],
        ["corrected_spans", stats.get("corrected_spans", 0)],
        ["dropped_spans", stats.get("dropped_spans", 0)],
        ["kept_spans", stats.get("kept_spans", 0)],
        *[
            [f"{split}_contracts", split_contracts[split]]
            for split in ("train", "validation", "test")
        ],
        *[
            [f"{split}_windows", split_windows[split]]
            for split in ("train", "validation", "test")
        ],
        ["overlapping_category_pairs", sum(overlap_pairs.values())],
    ]
    write_table(results_dir, "dataset_summary", ["metric", "value"], summary_rows)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Dataset figures require matplotlib; install requirements.txt."
        ) from error

    categories = list(CATEGORIES)
    figure_height = max(10, len(categories) * 0.28)
    fig, axis = plt.subplots(figsize=(12, figure_height))
    values = [token_counts[category] for category in categories]
    axis.barh(categories, [max(value, 1) for value in values])
    axis.set_xscale("log")
    axis.set_xlabel("Positive token count (log scale)")
    axis.set_title("CUAD positive token counts by clause category")
    axis.invert_yaxis()
    fig.tight_layout()
    fig.savefig(results_dir / "token_distribution.png", dpi=220)
    fig.savefig(results_dir / "token_distribution.pdf")
    plt.close(fig)

    bio_figure_height = max(12, len(BIO_LABELS) * 0.19)
    fig, axis = plt.subplots(figsize=(12, bio_figure_height))
    axis.barh(
        BIO_LABELS,
        [max(bio_counts[label], 1) for label in BIO_LABELS],
        color="#3273b9",
    )
    axis.set_xscale("log")
    axis.set_xlabel("Priority-resolved token count (log scale)")
    axis.set_title("BIO label distribution (O plus 82 clause tags)")
    axis.invert_yaxis()
    fig.tight_layout()
    fig.savefig(results_dir / "bio_label_distribution.png", dpi=220)
    fig.savefig(results_dir / "bio_label_distribution.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(14, 8))
    box_data = [span_lengths[category] for category in categories]
    axis.boxplot(box_data, tick_labels=categories, showfliers=False)
    axis.set_ylabel("Span length (word tokens)")
    axis.set_title("CUAD span lengths by clause category")
    axis.tick_params(axis="x", labelrotation=75, labelsize=8)
    fig.tight_layout()
    fig.savefig(results_dir / "span_lengths.png", dpi=220)
    fig.savefig(results_dir / "span_lengths.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 6))
    axis.hist(contract_lengths, bins=30)
    axis.set_xlabel("Word tokens per contract")
    axis.set_ylabel("Number of contracts")
    axis.set_title("CUAD contract length distribution")
    fig.tight_layout()
    fig.savefig(results_dir / "contract_lengths.png", dpi=220)
    fig.savefig(results_dir / "contract_lengths.pdf")
    plt.close(fig)

    stats["analysis_outputs"] = [
        "class_distribution.csv",
        "class_distribution.tex",
        "source_annotation_audit.csv",
        "source_annotation_audit.tex",
        "source_spans_by_category.csv",
        "source_spans_by_category.tex",
        "bio_label_distribution.csv",
        "bio_label_distribution.tex",
        "overlap_pairs.csv",
        "overlap_pairs.tex",
        "dataset_summary.csv",
        "dataset_summary.tex",
        "token_distribution.png",
        "token_distribution.pdf",
        "bio_label_distribution.png",
        "bio_label_distribution.pdf",
        "span_lengths.png",
        "span_lengths.pdf",
        "contract_lengths.png",
        "contract_lengths.pdf",
    ]
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=PROCESSED_DATA / "contracts.jsonl.xz"
    )
    parser.add_argument("--stats", type=Path, default=PROCESSED_DATA / "stats.json")
    parser.add_argument("--results", type=Path, default=RESULTS / "analysis")
    parser.add_argument("--raw", type=Path, default=RAW_DATA)
    args = parser.parse_args()
    analyze(args.data, args.stats, args.results, args.raw)
    print(f"Analysis tables and figures saved in {args.results}")


if __name__ == "__main__":
    main()
