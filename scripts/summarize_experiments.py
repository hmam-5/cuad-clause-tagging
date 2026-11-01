"""Summarize measured runs by individual seed and create result figures."""

from __future__ import annotations

import csv
import json
import lzma
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cuad_tagging.baselines import _write_summary
from cuad_tagging.config import CATEGORIES, PROCESSED_DATA, RESULTS


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _per_seed_results(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "seed": int(run["seed"]),
            "token_macro_f1": float(run["token_macro_f1"]),
            "strict_span_micro_f1": float(run["span_strict_micro_f1"]),
            "partial_span_micro_f1": float(run["span_partial_micro_f1"]),
            "training_minutes": float(run["train_seconds"]) / 60,
            "bootstrap_95": [
                float(run["bootstrap_95_lower"]),
                float(run["bootstrap_95_upper"]),
            ],
        }
        for run in sorted(group, key=lambda item: int(item["seed"]))
    ]


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = ((start + 1) + end) / 2
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def _spearman(first: list[float], second: list[float]) -> float | None:
    if len(first) != len(second) or len(first) < 2:
        raise ValueError(
            "Spearman correlation requires equally sized vectors of length >= 2."
        )
    first_ranks = _average_ranks(first)
    second_ranks = _average_ranks(second)
    first_mean = statistics.mean(first_ranks)
    second_mean = statistics.mean(second_ranks)
    covariance = sum(
        (left - first_mean) * (right - second_mean)
        for left, right in zip(first_ranks, second_ranks)
    )
    first_variance = sum((rank - first_mean) ** 2 for rank in first_ranks)
    second_variance = sum((rank - second_mean) ** 2 for rank in second_ranks)
    if first_variance == 0 or second_variance == 0:
        return None
    return covariance / math.sqrt(first_variance * second_variance)


def _write_crf_category_analysis(
    results_dir: Path, stats: dict[str, Any]
) -> dict[str, Any] | None:
    metrics_path = ROOT / "artifacts" / "crf_full_data" / "test_metrics.json"
    if metrics_path.is_file():
        metrics = _read_json(metrics_path)
        strict_scores = metrics["span"]["strict"]["per_category"]
    else:
        result_path = results_dir / "crf-full-seed42-test.json"
        if not result_path.is_file():
            return None
        strict_scores = _read_json(result_path).get("span_strict_per_category")
        if not isinstance(strict_scores, dict):
            return None
    training_lengths: dict[str, list[int]] = {category: [] for category in CATEGORIES}
    with lzma.open(
        PROCESSED_DATA / "contracts.jsonl.xz", "rt", encoding="utf-8"
    ) as source:
        for line in source:
            record = json.loads(line)
            if record["split"] != "train":
                continue
            for span in record["spans"]:
                category = CATEGORIES[span[0]]
                training_lengths[category].append(span[6] - span[5])

    span_counts = stats["per_category_spans"]
    rows = []
    for category in CATEGORIES:
        test_spans = int(span_counts[category]["test"])
        train_spans = int(span_counts[category]["train"])
        score = strict_scores[category]
        lengths = training_lengths[category]
        rows.append(
            {
                "category": category,
                "strict_span_f1": float(score["f1"]),
                "precision": float(score["precision"]),
                "recall": float(score["recall"]),
                "gold_test_spans": test_spans,
                "training_spans": train_spans,
                "mean_training_span_tokens": (
                    statistics.mean(lengths) if lengths else 0.0
                ),
                "fewer_than_5_test_spans": test_spans < 5,
            }
        )
    rows.sort(key=lambda row: (-row["strict_span_f1"], row["category"]))
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    csv_path = analysis_dir / "crf_full_data_per_category.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    tex_path = analysis_dir / "crf_full_data_per_category.tex"
    headers = [
        "Category",
        "Strict span F1",
        "Gold test spans",
        "Training spans",
        "Mean training span tokens",
        "Test support < 5",
    ]
    with tex_path.open("w", encoding="utf-8") as output:
        output.write("\\begin{tabular}{lrrrrr}\n\\toprule\n")
        output.write(" & ".join(headers) + " \\\\\n\\midrule\n")
        for row in rows:
            category = row["category"].replace("&", r"\&").replace("_", r"\_")
            output.write(
                f"{category} & {row['strict_span_f1']:.6f} "
                f"& {row['gold_test_spans']} & {row['training_spans']} "
                f"& {row['mean_training_span_tokens']:.3f} "
                f"& {str(row['fewer_than_5_test_spans']).lower()} \\\\\n"
            )
        output.write("\\bottomrule\n\\end{tabular}\n")

    correlations = {
        "model": "full-data CRF seed 42",
        "evaluation_split": "test",
        "category_metric": "strict span F1",
        "categories": len(rows),
        "strict_span_f1_vs_training_span_count": _spearman(
            [row["strict_span_f1"] for row in rows],
            [float(row["training_spans"]) for row in rows],
        ),
        "strict_span_f1_vs_mean_training_span_tokens": _spearman(
            [row["strict_span_f1"] for row in rows],
            [row["mean_training_span_tokens"] for row in rows],
        ),
        "top_five": rows[:5],
        "bottom_five": rows[-5:],
        "categories_with_fewer_than_five_test_spans": [
            row["category"] for row in rows if row["fewer_than_5_test_spans"]
        ],
    }
    (analysis_dir / "crf_category_correlations.json").write_text(
        json.dumps(correlations, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return correlations


def _test_inference_ms_per_contract(
    artifacts_dir: Path, contract_count: int
) -> float | str:
    log_path = artifacts_dir / "test_evaluation.stdout.log"
    if not log_path.is_file() or contract_count < 1:
        return ""
    elapsed_seconds = None
    with log_path.open(encoding="utf-8") as source:
        for line in source:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                event.get("event") == "evaluation_contracts_completed"
                and event.get("split") == "test"
                and event.get("contracts_completed") == contract_count
            ):
                elapsed_seconds = float(event["elapsed_seconds"])
    if elapsed_seconds is None:
        return ""
    return 1000 * elapsed_seconds / contract_count


def _import_full_crf_results(results_dir: Path) -> list[dict[str, Any]]:
    artifacts_dir = ROOT / "artifacts" / "crf_full_data"
    metadata_path = artifacts_dir / "run_metadata.json"
    if not metadata_path.is_file():
        return []
    metadata = _read_json(metadata_path)
    if metadata.get("status") != "test_evaluation_complete":
        return []

    imported = []
    for split in ("validation", "test"):
        metrics_path = artifacts_dir / f"{split}_metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Completed CRF run is missing {metrics_path}.")
        metrics = _read_json(metrics_path)
        contract_count = int(metrics["contracts"])
        bootstrap = metrics["token_macro_f1_bootstrap_95"]
        span = metrics["span"]
        inference_ms = (
            _test_inference_ms_per_contract(artifacts_dir, contract_count)
            if split == "test"
            else ""
        )
        run_id = f"crf-full-seed{metadata['seed']}-{split}"
        record: dict[str, Any] = {
            "run_id": run_id,
            "model": "crf",
            "architecture": "linear-chain CRF",
            "checkpoint": Path(metadata["model_file"]).name,
            "seed": metadata["seed"],
            "evaluation_split": split,
            "token_macro_f1": metrics["token_macro_f1"],
            "span_strict_micro_f1": span["strict"]["micro_f1"],
            "span_strict_macro_f1": span["strict"]["macro_f1"],
            "span_partial_micro_f1": span["partial"]["micro_f1"],
            "span_partial_macro_f1": span["partial"]["macro_f1"],
            "bootstrap_95_lower": bootstrap["lower"],
            "bootstrap_95_upper": bootstrap["upper"],
            "bootstrap_iterations": bootstrap["iterations"],
            "train_seconds": metadata["fit_seconds"],
            "inference_ms_per_contract": inference_ms,
            "training_contracts": metadata["train_contracts"],
            "evaluation_contracts": contract_count,
            "test_contracts": metadata["test_contracts"],
            "training_scope": "full",
            "evaluation_scope": "full",
            "max_iterations": metadata["max_iterations"],
            "optimizer_converged": metadata["optimizer_converged"],
            "iteration_cap_reached": metadata["iteration_cap_reached"],
            "c1": metadata["c1"],
            "c2": metadata["c2"],
            "git_commit_hash": metadata.get("git_commit_hash", "not recorded"),
            "epochs": "",
            "learning_rate": "",
            "best_validation_macro_f1": (
                metrics["token_macro_f1"] if split == "validation" else None
            ),
            "loss_curve": "",
            "per_category": metrics["token_per_category"],
            "span_strict_per_category": metrics["span"]["strict"]["per_category"],
            "predictions_path": metrics["predictions_file"] if split == "test" else "",
            "final_iteration_loss": metadata["final_iteration_loss"],
            "final_iteration_error_norm": metadata["final_iteration_error_norm"],
        }
        destination = results_dir / f"{run_id}.json"
        destination.write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        imported.append(record)
    return imported


def _import_transformer_smoke(results_dir: Path) -> dict[str, Any] | None:
    source = ROOT / "artifacts" / "smoke_minimal" / "bert-base-cased-seed42.json"
    if not source.is_file():
        return None
    result = _read_json(source)
    result["evaluation_split"] = "smoke"
    destination = results_dir / "bert-base-cased-seed42.json"
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _import_full_transformer_results(
    results_dir: Path, artifacts_dir: Path | None = None
) -> list[dict[str, Any]]:
    artifacts_root = artifacts_dir or ROOT / "artifacts"
    if not artifacts_root.is_dir():
        return []
    candidates: dict[str, tuple[Path, dict[str, Any], dict[str, Any]]] = {}
    for run_dir in sorted(artifacts_root.iterdir()):
        if not run_dir.is_dir():
            continue
        for test_path in sorted(run_dir.glob("*-seed*.json")):
            if test_path.stem.endswith("-validation"):
                continue
            test_run = _read_json(test_path)
            if (
                test_run.get("training_scope") != "full"
                or test_run.get("evaluation_scope") != "full"
                or test_run.get("test_contracts") is None
            ):
                continue
            run_id = test_run["run_id"]
            validation_path = run_dir / f"{run_id}-validation.json"
            if not validation_path.is_file():
                raise FileNotFoundError(
                    f"Completed transformer run is missing {validation_path}."
                )
            validation_run = _read_json(validation_path)
            if (
                validation_run.get("run_id") != f"{run_id}-validation"
                or validation_run.get("evaluation_split") != "validation"
            ):
                raise ValueError(
                    f"Invalid validation record for transformer run {run_id}."
                )
            current = candidates.get(run_id)
            if (
                current is None
                or test_path.stat().st_mtime_ns > current[0].stat().st_mtime_ns
            ):
                candidates[run_id] = (test_path, test_run, validation_run)

    imported = []
    for run_id, (test_path, test_run, validation_run) in sorted(candidates.items()):
        loss_filename = Path(test_run["loss_curve"]).name
        loss_source = test_path.parent / "loss" / loss_filename
        if not loss_source.is_file():
            raise FileNotFoundError(
                f"Completed transformer run is missing {loss_source}."
            )
        loss_destination = results_dir / "loss" / loss_filename
        loss_destination.parent.mkdir(parents=True, exist_ok=True)
        loss_destination.write_bytes(loss_source.read_bytes())
        loss_curve = f"loss/{loss_filename}"

        test_record = {
            key: test_run[key]
            for key in (
                "model",
                "architecture",
                "checkpoint",
                "seed",
                "token_macro_f1",
                "span_strict_micro_f1",
                "span_strict_macro_f1",
                "span_partial_micro_f1",
                "span_partial_macro_f1",
                "bootstrap_95_lower",
                "bootstrap_95_upper",
                "bootstrap_iterations",
                "train_seconds",
                "inference_ms_per_contract",
                "training_contracts",
                "test_contracts",
                "training_scope",
                "evaluation_scope",
                "best_validation_macro_f1",
                "best_epoch",
                "epochs",
                "learning_rate",
                "effective_batch_size",
                "per_category",
                "training_hyperparameters",
                "predictions_path",
            )
            if key in test_run
        }
        test_record.update(
            {
                "run_id": run_id,
                "evaluation_split": "test",
                "evaluation_contracts": test_run["test_contracts"],
                "loss_curve": loss_curve,
            }
        )
        validation_record = {key: value for key, value in validation_run.items()}
        validation_record["loss_curve"] = loss_curve

        for record in (test_record, validation_record):
            destination = results_dir / f"{record['run_id']}.json"
            destination.write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            imported.append(record)
    return imported


def _saved_paired_comparisons(
    results_dir: Path, runs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    comparisons = []
    for path in sorted(results_dir.glob("paired_comparison*.json")):
        result = _read_json(path)
        difference = result.get(
            "paired_contract_bootstrap_macro_f1_difference_first_minus_second"
        )
        if not difference:
            continue
        first_path = Path(result["first_predictions"]).name
        second_path = Path(result["second_predictions"]).name
        first_run_id = first_path.removesuffix("_predictions.jsonl")
        second_run_id = second_path.removesuffix("_predictions.jsonl")
        first_run = next(
            (
                run
                for run in runs
                if run["run_id"] == first_run_id
                or Path(run.get("predictions_path") or "").name == first_path
            ),
            None,
        )
        second_run = next(
            (
                run
                for run in runs
                if run["run_id"] == second_run_id
                or Path(run.get("predictions_path") or "").name == second_path
            ),
            None,
        )
        comparisons.append(
            {
                "first": _run_group(first_run) if first_run else first_run_id,
                "second": _run_group(second_run) if second_run else second_run_id,
                "paired_difference": difference,
                "source_file": path.name,
                "selection_note": (
                    "Descriptive paired test comparison only; not used for model "
                    "selection or tuning."
                ),
            }
        )
    return comparisons


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
            explanation = (
                f"Only {test_spans} gold test spans; per-category estimate is unstable."
            )
        elif f1 == 0:
            explanation = "No correctly labeled test tokens; class imbalance and overlap are likely factors."
        elif score.get("recall", 0.0) < score.get("precision", 0.0):
            explanation = (
                "Recall trails precision; the model misses some annotated tokens."
            )
        else:
            explanation = (
                "Test support and annotation boundary ambiguity affect this estimate."
            )
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
        with curve_path.open(encoding="utf-8-sig", newline="") as source:
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
        axis.plot(
            epochs,
            [float(row["train_loss"]) for row in history],
            marker="o",
            label="Train",
        )
        axis.plot(
            epochs,
            [float(row["validation_loss"]) for row in history],
            marker="s",
            label="Validation",
        )
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
    _import_full_crf_results(results_dir)
    _import_full_transformer_results(results_dir)
    _import_transformer_smoke(results_dir)
    runs = []
    for path in sorted(results_dir.glob("*-seed*.json")):
        run = _read_json(path)
        if not {"run_id", "token_macro_f1", "model"}.issubset(run):
            continue
        if "training_scope" not in run:
            run["training_scope"] = "smoke" if run["model"].lower() == "crf" else "full"
        run.setdefault(
            "evaluation_split",
            "smoke" if run.get("evaluation_scope") == "smoke" else "test",
        )
        runs.append(run)
    _write_summary(results_dir, runs)
    full_runs = [
        run
        for run in runs
        if run["training_scope"] == "full" and run["evaluation_split"] == "test"
    ]
    validation_runs = [
        run
        for run in runs
        if run["training_scope"] == "full" and run["evaluation_split"] == "validation"
    ]
    smoke_runs = [run for run in runs if run["training_scope"] == "smoke"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for run in full_runs:
        groups.setdefault(_run_group(run), []).append(run)
    aggregates = []
    for name, group in sorted(groups.items()):
        row: dict[str, Any] = {
            "model": name,
            "seeds": len(group),
            "per_seed_results": _per_seed_results(group),
            "max_iterations": group[0].get("max_iterations"),
            "optimizer_converged": group[0].get("optimizer_converged"),
            "regularization": {
                "c1": group[0].get("c1"),
                "c2": group[0].get("c2"),
            },
            "bootstrap_ci_per_run": [
                [float(run["bootstrap_95_lower"]), float(run["bootstrap_95_upper"])]
                for run in group
            ],
        }
        aggregates.append(row)
    aggregates.sort(
        key=lambda row: max(
            result["token_macro_f1"] for result in row["per_seed_results"]
        ),
        reverse=True,
    )

    best_group = None
    best_validation_run = None
    best_test_run = None
    validation_groups: dict[str, list[dict[str, Any]]] = {}
    for run in validation_runs:
        validation_groups.setdefault(_run_group(run), []).append(run)
    if validation_groups:
        best_group, selected_runs = max(
            validation_groups.items(),
            key=lambda item: max(float(run["token_macro_f1"]) for run in item[1]),
        )
        best_validation_run = max(
            selected_runs, key=lambda run: float(run["token_macro_f1"])
        )
        matching_test_runs = groups.get(best_group, [])
        if matching_test_runs:
            best_test_run = min(
                matching_test_runs,
                key=lambda run: abs(
                    int(run["seed"]) - int(best_validation_run["seed"])
                ),
            )
    categories = _write_category_report(results_dir, best_test_run, stats)
    crf_category_analysis = _write_crf_category_analysis(results_dir, stats)
    loss_runs = _write_loss_figure(results_dir, full_runs)

    significance = _saved_paired_comparisons(results_dir, runs)
    report = {
        "dataset_contracts": stats["contracts"],
        "dataset_spans": stats["kept_spans"],
        "offset_roundtrip_agreement": stats["roundtrip_span_agreement"],
        "full_run_count": len(full_runs),
        "model_aggregates": aggregates,
        "full_data_run_metadata": [
            {
                "run_id": run["run_id"],
                "max_iterations": run.get("max_iterations"),
                "iterations_completed": run.get("max_iterations"),
                "iteration_cap_reached": run.get("iteration_cap_reached"),
                "optimizer_converged": run.get("optimizer_converged"),
                "c1": run.get("c1"),
                "c2": run.get("c2"),
                "train_seconds": run.get("train_seconds"),
                "final_iteration_loss": run.get("final_iteration_loss"),
                "final_iteration_error_norm": run.get("final_iteration_error_norm"),
                "git_commit_hash": run.get("git_commit_hash", "not recorded"),
            }
            for run in full_runs
            if run["model"] == "crf"
        ],
        "validation_results": [
            {
                "run_id": run["run_id"],
                "model": _run_group(run),
                "token_macro_f1": run["token_macro_f1"],
                "bootstrap_95": [
                    run["bootstrap_95_lower"],
                    run["bootstrap_95_upper"],
                ],
                "strict_span_micro_f1": run["span_strict_micro_f1"],
                "strict_span_macro_f1": run["span_strict_macro_f1"],
                "partial_span_micro_f1": run["span_partial_micro_f1"],
                "partial_span_macro_f1": run["span_partial_macro_f1"],
            }
            for run in validation_runs
        ],
        "smoke_run_ids": [run["run_id"] for run in smoke_runs],
        "best_validation_selected_group": best_group,
        "best_validation_selected_run": (
            best_validation_run["run_id"] if best_validation_run else None
        ),
        "best_validation_selected_test_run": (
            best_test_run["run_id"] if best_test_run else None
        ),
        "per_category_report": "per_category_f1.csv" if categories else None,
        "crf_category_analysis": (
            "analysis/crf_full_data_per_category.csv" if crf_category_analysis else None
        ),
        "crf_category_correlations": (
            {
                "strict_span_f1_vs_training_span_count": (
                    crf_category_analysis["strict_span_f1_vs_training_span_count"]
                ),
                "strict_span_f1_vs_mean_training_span_tokens": (
                    crf_category_analysis["strict_span_f1_vs_mean_training_span_tokens"]
                ),
            }
            if crf_category_analysis
            else None
        ),
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
            (
                "Test results are descriptive only; model selection is not performed "
                "from test scores."
            ),
        ],
    }
    with (results_dir / "experiment_summary.json").open(
        "w", encoding="utf-8"
    ) as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")

    lines = [
        "# CUAD project: findings summary",
        "",
        "## Three key numbers",
        f"- **{stats['contracts']:,} contracts** and **{stats['kept_spans']:,} validated spans**.",
        f"- **{stats['roundtrip_span_agreement']:.0%}** token-to-span round-trip agreement.",
        (
            f"- Highest validation-selected group: **{best_group}** "
            f"(validation token macro F1 "
            f"**{best_validation_run['token_macro_f1']:.4f}**)."
            if best_validation_run
            else "- No validation-scored full-data model run is present."
        ),
        "",
        "## Full-data test results (descriptive; not used for selection)",
        "",
        "| Model | Seed | Token macro F1 (contract-bootstrap 95% CI) | Strict span micro F1 | Partial span micro F1 | Train minutes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        display_model = row["model"].replace("|", r"\|")
        for index, result in enumerate(row["per_seed_results"]):
            ci = result["bootstrap_95"]
            lines.append(
                f"| {display_model if index == 0 else ''} | {result['seed']} | "
                f"{result['token_macro_f1']:.4f} "
                f"({ci[0]:.4f}–{ci[1]:.4f}) | "
                f"{result['strict_span_micro_f1']:.4f} | "
                f"{result['partial_span_micro_f1']:.4f} | "
                f"{result['training_minutes']:.2f} |"
            )
    lines += ["", "## Validation results", ""]
    lines += [
        f"- **{run['model']}** ({run['run_id']}): token macro F1 "
        f"{run['token_macro_f1']:.4f}; strict span micro F1 "
        f"{run['strict_span_micro_f1']:.4f}; partial span micro F1 "
        f"{run['partial_span_micro_f1']:.4f}."
        for run in report["validation_results"]
    ] or ["- No full-data validation metrics are present."]
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
