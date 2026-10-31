"""Aggregate measured runs and create the Stage 6 report and figures."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_models import compare

from cuad_tagging.config import PROCESSED_DATA, RESULTS


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _run_group(run: dict[str, Any]) -> str:
    return f"{run['architecture']} | {run['checkpoint']}"


def _write_category_report(
    results_dir: Path,
    best: dict[str, Any] | None,
    stats: dict[str, Any],
) -> list[dict[str, Any]]:
    if best is None:
        return []
    category_scores = best.get("per_category", {})
    span_counts = stats.get("per_category_spans", {})
    rows = []
    for category, score in category_scores.items():
        support = int(score.get("support", 0))
        test_spans = int(span_counts.get(category, {}).get("test", 0))
        f1 = float(score.get("f1", 0.0))
        if test_spans < 5:
            explanation = f"Only {test_spans} gold test spans; per-category estimate is unstable."
        elif f1 == 0:
            explanation = "No correctly labeled test tokens; class imbalance and overlap are likely factors."
        elif score.get("recall", 0.0) < score.get("precision", 0.0):
            explanation = "Recall trails precision; the model misses some annotated tokens."
        else:
            explanation = "Test support and annotation boundary ambiguity affect this estimate."
        rows.append(
            {
                "category": category,
                "token_f1": f1,
                "precision": float(score.get("precision", 0.0)),
                "recall": float(score.get("recall", 0.0)),
                "test_token_support": support,
                "test_spans": test_spans,
                "interpretation": explanation,
            }
        )
    rows.sort(key=lambda row: (row["token_f1"], row["category"]))
    csv_path = results_dir / "per_category_f1.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Stage 6 figures require matplotlib.") from error
    ordered = list(reversed(rows))
    height = max(8, len(ordered) * 0.25)
    figure, axis = plt.subplots(figsize=(12, height))
    axis.barh(
        [row["category"] for row in ordered],
        [row["token_f1"] for row in ordered],
        color="#3273b9",
    )
    axis.set_xlim(0, max(0.05, max(row["token_f1"] for row in ordered) * 1.1))
    axis.set_xlabel("Token-level F1")
    axis.set_title(f"Per-category token F1: {_run_group(best)}")
    figure.tight_layout()
    figure.savefig(results_dir / "per_category_f1.png", dpi=220)
    figure.savefig(results_dir / "per_category_f1.pdf")
    plt.close(figure)
    return rows


def _write_loss_figure(results_dir: Path, runs: list[dict[str, Any]]) -> list[str]:
    curves = []
    for run in runs:
        curve_path = ROOT / run.get("loss_curve", "")
        if not curve_path.is_file():
            curve_path = results_dir / "loss" / f"{run['run_id']}.csv"
        if not curve_path.is_file():
            continue
        with curve_path.open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        if rows:
            curves.append((run, rows))
    if not curves:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Stage 6 figures require matplotlib.") from error

    columns = 2
    rows_count = math.ceil(len(curves) / columns)
    figure, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(14, 4.2 * rows_count),
        squeeze=False,
    )
    for index, (run, history) in enumerate(curves):
        axis = axes[index // columns][index % columns]
        epochs = [int(float(row["epoch"])) for row in history]
        axis.plot(epochs, [float(row["train_loss"]) for row in history], marker="o", label="Train")
        axis.plot(epochs, [float(row["validation_loss"]) for row in history], marker="s", label="Validation")
        axis.set_title(_run_group(run))
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.legend()
        axis.grid(alpha=0.2)
    for index in range(len(curves), rows_count * columns):
        axes[index // columns][index % columns].set_visible(False)
    figure.suptitle("Training and validation loss (recorded runs)", fontsize=16)
    figure.tight_layout()
    figure.savefig(results_dir / "loss_curves.png", dpi=220)
    figure.savefig(results_dir / "loss_curves.pdf")
    plt.close(figure)
    return [run["run_id"] for run, _ in curves]


def summarize(results_dir: Path = RESULTS) -> dict[str, Any]:
    stats = _read_json(PROCESSED_DATA / "stats.json")
    runs = []
    for path in sorted(results_dir.glob("*-seed*.json")):
        run = _read_json(path)
        if not {"run_id", "token_macro_f1", "model"}.issubset(run):
            continue
        if "training_scope" not in run:
            run["training_scope"] = (
                "smoke" if run["model"].lower() == "crf" else "full"
            )
        runs.append(run)
    full_runs = [run for run in runs if run["training_scope"] == "full"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for run in full_runs:
        groups.setdefault(_run_group(run), []).append(run)
    aggregates = []
    for name, group in sorted(groups.items()):
        row: dict[str, Any] = {
            "model": name,
            "seeds": len(group),
            "token_macro_f1_mean": statistics.mean(float(run["token_macro_f1"]) for run in group),
            "token_macro_f1_sd": _std([float(run["token_macro_f1"]) for run in group]),
            "strict_span_micro_f1_mean": statistics.mean(float(run["span_strict_micro_f1"]) for run in group),
            "partial_span_micro_f1_mean": statistics.mean(float(run["span_partial_micro_f1"]) for run in group),
            "training_minutes_mean": statistics.mean(float(run["train_seconds"]) for run in group) / 60,
            "bootstrap_ci_per_run": [
                [float(run["bootstrap_95_lower"]), float(run["bootstrap_95_upper"])]
                for run in group
            ],
        }
        aggregates.append(row)
    aggregates.sort(key=lambda row: row["token_macro_f1_mean"], reverse=True)

    best_group = None
    best_run = None
    if aggregates:
        best_group = aggregates[0]["model"]
        best_run = max(
            groups[best_group],
            key=lambda run: float(run.get("best_validation_macro_f1", run["token_macro_f1"])),
        )
    categories = _write_category_report(results_dir, best_run, stats)
    loss_runs = _write_loss_figure(results_dir, full_runs)

    significance = []
    for first, second in combinations(aggregates[:2], 2):
        first_run = max(
            groups[first["model"]],
            key=lambda run: float(run.get("best_validation_macro_f1", run["token_macro_f1"])),
        )
        second_run = max(
            groups[second["model"]],
            key=lambda run: float(run.get("best_validation_macro_f1", run["token_macro_f1"])),
        )
        first_predictions = results_dir / f"{first_run['run_id']}_predictions.jsonl"
        second_predictions = results_dir / f"{second_run['run_id']}_predictions.jsonl"
        if first_predictions.is_file() and second_predictions.is_file():
            result = compare(first_predictions, second_predictions, iterations=1000)
            significance.append(
                {
                    "first": first["model"],
                    "second": second["model"],
                    "paired_difference": result[
                        "paired_contract_bootstrap_macro_f1_difference_first_minus_second"
                    ],
                    "selection_note": (
                        "Runs are selected using validation score; paired test "
                        "comparison is descriptive and is not used for tuning."
                    ),
                }
            )

    report = {
        "dataset_contracts": stats["contracts"],
        "dataset_spans": stats["kept_spans"],
        "offset_roundtrip_agreement": stats["roundtrip_span_agreement"],
        "full_run_count": len(full_runs),
        "model_aggregates": aggregates,
        "top_model_group": best_group,
        "best_validation_selected_run": best_run["run_id"] if best_run else None,
        "per_category_report": "per_category_f1.csv" if categories else None,
        "loss_figure_runs": loss_runs,
        "paired_comparisons": significance,
        "caveats": [
            (
                f"{len(stats['categories_below_minimum_test_spans'])} "
                "categories have fewer than five test spans."
            ),
            (
                "Outside tokens account for "
                f"{stats['o_token_share']:.1%} of priority-resolved tokens."
            ),
            (
                "Runs with fewer than three seeds (and the 10-contract CRF smoke run) "
                "are exploratory; seed variance and robust model significance remain unestablished."
            ),
        ],
    }
    with (results_dir / "experiment_summary.json").open("w", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")

    lines = [
        "# CUAD project: findings summary",
        "",
        "## Three key numbers",
        f"- **{stats['contracts']:,} contracts** and **{stats['kept_spans']:,} validated spans**.",
        f"- **{stats['roundtrip_span_agreement']:.0%}** token-to-span round-trip agreement.",
        (
            f"- Best full-data run so far: **{best_group}**, token macro F1 "
            f"**{aggregates[0]['token_macro_f1_mean']:.4f}** "
            f"(n={aggregates[0]['seeds']} seed(s))."
            if aggregates
            else "- No full-data model run is present; no performance conclusion is available."
        ),
        "",
        "## Full-data model results",
        "",
        "| Model | Seeds | Token macro F1 mean (SD) | Strict span micro F1 | Partial span micro F1 | Mean train minutes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        display_model = row["model"].replace("|", r"\|")
        lines.append(
            f"| {display_model} | {row['seeds']} | "
            f"{row['token_macro_f1_mean']:.4f} ({row['token_macro_f1_sd']:.4f}) | "
            f"{row['strict_span_micro_f1_mean']:.4f} | "
            f"{row['partial_span_micro_f1_mean']:.4f} | "
            f"{row['training_minutes_mean']:.2f} |"
        )
    lines += ["", "## Top and bottom five categories"]
    if categories:
        ranked = sorted(categories, key=lambda row: row["token_f1"], reverse=True)
        lines += ["", "### Highest F1", ""]
        lines += [
            f"- **{row['category']}** — F1 {row['token_f1']:.3f}; "
            f"{row['test_spans']} gold test spans; {row['interpretation']}"
            for row in ranked[:5]
        ]
        lines += ["", "### Lowest F1", ""]
        lines += [
            f"- **{row['category']}** — F1 {row['token_f1']:.3f}; "
            f"{row['test_spans']} gold test spans; {row['interpretation']}"
            for row in ranked[-5:]
        ]
    else:
        lines.append("\nNo full-data model scores are available for category analysis.")
    lines += [
        "",
        "## Three caveats",
        *[f"- {caveat}" for caveat in report["caveats"]],
        "",
    ]
    (results_dir / "findings_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(summarize(), ensure_ascii=False, indent=2))
