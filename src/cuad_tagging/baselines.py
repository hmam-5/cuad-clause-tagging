"""Reproducible majority and feature-based CRF baselines."""

from __future__ import annotations

import csv
import json
import lzma
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import BIO_LABELS, PROCESSED_DATA, RESULTS, SEED
from .metrics import bootstrap_macro_f1, span_metrics, token_metrics


def load_records(path: Path = PROCESSED_DATA / "contracts.jsonl.xz") -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Prepared data not found: {path}")
    opener = lzma.open if path.suffix == ".xz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        records: list[dict[str, Any]] = []
        for line in source:
            if not line.strip():
                continue
            compact = json.loads(line)
            records.append(
                {
                    "contract_id": compact["contract_id"],
                    "split": compact["split"],
                    "tokens": compact["tokens"],
                    "labels": compact["labels"],
                    "windows": compact["windows"],
                }
            )
        return records


def _word_features(tokens: list[str], index: int) -> dict[str, Any]:
    word = tokens[index]
    features: dict[str, Any] = {
        "bias": 1.0,
        "word.lower": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word[:3]": word[:3],
        "word[:2]": word[:2],
        "word.isupper": word.isupper(),
        "word.istitle": word.istitle(),
        "word.isdigit": word.isdigit(),
        "word.length": min(len(word), 20),
        "word.shape": re.sub(r"[a-z]", "a", re.sub(r"[A-Z]", "A", word)),
        "is_first": index == 0,
        "is_last": index == len(tokens) - 1,
    }
    for offset, prefix in ((-1, "-1"), (1, "+1")):
        neighbor_index = index + offset
        if 0 <= neighbor_index < len(tokens):
            neighbor = tokens[neighbor_index]
            features[f"word{prefix}.lower"] = neighbor.lower()
            features[f"word{prefix}.istitle"] = neighbor.istitle()
            features[f"word{prefix}.isupper"] = neighbor.isupper()
            features[f"word{prefix}.shape"] = re.sub(
                r"[a-z]", "a", re.sub(r"[A-Z]", "A", neighbor)
            )
        else:
            features[f"word{prefix}.boundary"] = True
    return features


def _contract_windows(
    record: dict[str, Any],
) -> list[tuple[list[str], list[str], int, int]]:
    windows = record.get("windows") or [[0, len(record["tokens"])]]
    return [
        (
            record["tokens"][window[0] : window[1]],
            [
                BIO_LABELS[label_id]
                for label_id in record["labels"][window[0] : window[1]]
            ],
            window[0],
            window[1],
        )
        for window in windows
    ]


def _merge_window_predictions(
    record: dict[str, Any], window_predictions: list[list[str]]
) -> list[str]:
    votes: list[Counter[str]] = [Counter() for _ in record["tokens"]]
    for window, predictions in zip(record["windows"], window_predictions):
        for index, prediction in enumerate(predictions, start=window[0]):
            votes[index][prediction] += 1
    return [
        max(
            counter,
            key=lambda label: (counter[label], -BIO_LABELS.index(label)),
        )
        if counter
        else "O"
        for counter in votes
    ]


def _metrics_for(
    records: list[dict[str, Any]], predictions: list[list[str]], iterations: int, seed: int
) -> dict[str, Any]:
    gold = [
        [BIO_LABELS[label_id] for label_id in record["labels"]]
        for record in records
    ]
    return {
        "token": token_metrics(gold, predictions),
        "span": span_metrics(gold, predictions),
        "bootstrap_95": bootstrap_macro_f1(gold, predictions, iterations, seed),
    }


def _save_run(
    name: str,
    seed: int,
    test_records: list[dict[str, Any]],
    predictions: list[list[str]],
    metrics: dict[str, Any],
    fit_seconds: float,
    inference_seconds: float,
    output_dir: Path,
    iterations: int,
    architecture: str | None = None,
    checkpoint: str = "none",
    model_name: str | None = None,
    run_id: str | None = None,
    training_contracts: int | None = None,
    training_scope: str = "full",
    evaluation_scope: str = "full",
) -> dict[str, Any]:
    token = metrics["token"]
    strict = metrics["span"]["strict"]
    partial = metrics["span"]["partial"]
    result = {
        "run_id": run_id or f"{name}-seed{seed}",
        "model": model_name or name,
        "architecture": architecture or {
            "o-only": "constant classifier",
            "most-frequent-word": "word lookup",
            "crf": "linear-chain CRF",
        }.get(name, name),
        "checkpoint": checkpoint,
        "seed": seed,
        "token_macro_f1": token["macro_f1"],
        "span_strict_micro_f1": strict["micro_f1"],
        "span_strict_macro_f1": strict["macro_f1"],
        "span_partial_micro_f1": partial["micro_f1"],
        "span_partial_macro_f1": partial["macro_f1"],
        "bootstrap_95_lower": metrics["bootstrap_95"]["lower"],
        "bootstrap_95_upper": metrics["bootstrap_95"]["upper"],
        "train_seconds": fit_seconds,
        "inference_ms_per_contract": 1000 * inference_seconds / max(len(test_records), 1),
        "bootstrap_iterations": iterations,
        "training_contracts": training_contracts,
        "test_contracts": len(test_records),
        "training_scope": training_scope,
        "evaluation_scope": evaluation_scope,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{result['run_id']}.json").open("w", encoding="utf-8") as output:
        json.dump({**result, "per_category": token["per_category"]}, output, indent=2)
        output.write("\n")
    with (output_dir / f"{result['run_id']}_predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as output:
        for record, predicted in zip(test_records, predictions):
            output.write(
                json.dumps(
                    {
                        "contract_id": record["contract_id"],
                        "gold": [
                            BIO_LABELS[label_id] for label_id in record["labels"]
                        ],
                        "predicted": predicted,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return result


def _write_summary(output_dir: Path, results: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "metrics.csv"
    rows_by_id: dict[str, dict[str, Any]] = {}
    if path.is_file():
        with path.open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                rows_by_id[row["run_id"]] = row
    for result in results:
        rows_by_id[result["run_id"]] = result
    fields = [
        "run_id",
        "model",
        "architecture",
        "checkpoint",
        "seed",
        "evaluation_split",
        "token_macro_f1",
        "span_strict_micro_f1",
        "span_strict_macro_f1",
        "span_partial_micro_f1",
        "span_partial_macro_f1",
        "bootstrap_95_lower",
        "bootstrap_95_upper",
        "train_seconds",
        "inference_ms_per_contract",
        "bootstrap_iterations",
        "training_contracts",
        "evaluation_contracts",
        "test_contracts",
        "training_scope",
        "evaluation_scope",
        "max_iterations",
        "iteration_cap_reached",
        "optimizer_converged",
        "c1",
        "c2",
        "final_iteration_loss",
        "final_iteration_error_norm",
        "git_commit_hash",
        "epochs",
        "learning_rate",
        "best_validation_macro_f1",
        "loss_curve",
    ]
    for row in rows_by_id.values():
        row.setdefault(
            "evaluation_split",
            "smoke" if row.get("evaluation_scope") == "smoke" else "test",
        )
    ordered = [rows_by_id[key] for key in sorted(rows_by_id)]
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    latex_path = output_dir / "metrics.tex"
    with latex_path.open("w", encoding="utf-8") as output:
        output.write("\\begin{tabular}{llllrrrrrrr}\n\\toprule\n")
        output.write(
            "Architecture & Checkpoint & Training contracts (scope) "
            "& Split "
            "& Token macro F1 (95\\% CI) "
            "& Strict span micro F1 & Strict span macro F1 "
            "& Partial span micro F1 & Train time (s) & Inference (ms/contract) "
            "& Iterations (converged) \\\\\n"
        )
        output.write("\\midrule\n")
        for row in ordered:
            converged = row.get("optimizer_converged")
            convergence_text = "n/a" if converged is None else str(converged)
            output.write(
                f"{row['architecture']} & {row['checkpoint']} "
                f"& {row.get('training_contracts', '')}/"
                f"{row.get('evaluation_contracts', row.get('test_contracts', ''))} "
                f"({row.get('training_scope', '')}) "
                f"& {row.get('evaluation_split', '')} "
                f"& {float(row['token_macro_f1']):.4f} "
                f"({float(row['bootstrap_95_lower']):.4f}--"
                f"{float(row['bootstrap_95_upper']):.4f}) "
                f"& {float(row['span_strict_micro_f1']):.4f} "
                f"& {float(row['span_strict_macro_f1']):.4f} "
                f"& {float(row['span_partial_micro_f1']):.4f} "
                f"& {float(row['train_seconds']):.1f} "
                f"& {float(row['inference_ms_per_contract'] or 0):.1f} "
                f"& {row.get('max_iterations', '')} "
                f"({convergence_text}) \\\\\n"
            )
        output.write("\\bottomrule\n\\end{tabular}\n")


def run_baselines(
    data_path: Path,
    model: str = "all",
    seed: int = SEED,
    max_train_contracts: int | None = None,
    bootstrap_iterations: int = 1000,
    output_dir: Path = RESULTS,
) -> list[dict[str, Any]]:
    records = load_records(data_path)
    train_records = [record for record in records if record["split"] == "train"]
    test_records = [record for record in records if record["split"] == "test"]
    if not train_records or not test_records:
        raise ValueError("Both train and test splits must contain contracts.")
    if max_train_contracts is not None:
        rng = random.Random(seed)
        train_records = rng.sample(
            train_records, min(max_train_contracts, len(train_records))
        )
    models = ("o-only", "most-frequent-word", "crf") if model == "all" else (model,)
    results: list[dict[str, Any]] = []

    for name in models:
        if name not in {"o-only", "most-frequent-word", "crf"}:
            raise ValueError(f"Unsupported baseline: {name}")
        start = time.perf_counter()
        fitted: Any = None
        if name == "most-frequent-word":
            word_labels: dict[str, Counter[str]] = defaultdict(Counter)
            for record in train_records:
                for word, label_id in zip(record["tokens"], record["labels"]):
                    word_labels[word.lower()][BIO_LABELS[label_id]] += 1
            fitted = {
                word: max(
                    counts,
                    key=lambda label: (counts[label], -BIO_LABELS.index(label)),
                )
                for word, counts in word_labels.items()
            }
        elif name == "crf":
            try:
                import sklearn_crfsuite
            except ImportError as error:
                raise RuntimeError(
                    "The CRF baseline requires sklearn-crfsuite; install requirements.txt."
                ) from error
            feature_sequences: list[list[dict[str, Any]]] = []
            label_sequences: list[list[str]] = []
            for record in train_records:
                for tokens, labels, _, _ in _contract_windows(record):
                    if tokens:
                        feature_sequences.append(
                            [_word_features(tokens, index) for index in range(len(tokens))]
                        )
                        label_sequences.append(labels)
            fitted = sklearn_crfsuite.CRF(
                algorithm="lbfgs",
                c1=0.1,
                c2=0.1,
                max_iterations=50,
                all_possible_transitions=True,
                verbose=False,
            )
            fitted.fit(feature_sequences, label_sequences)
        fit_seconds = time.perf_counter() - start

        start = time.perf_counter()
        test_predictions: list[list[str]] = []
        for record in test_records:
            if name == "o-only":
                predicted = ["O"] * len(record["tokens"])
            elif name == "most-frequent-word":
                predicted = [
                    fitted.get(word.lower(), "O") for word in record["tokens"]
                ]
            else:
                windows = _contract_windows(record)
                predictions = fitted.predict(
                    [
                        [
                            _word_features(tokens, index)
                            for index in range(len(tokens))
                        ]
                        for tokens, _, _, _ in windows
                    ]
                )
                predicted = _merge_window_predictions(record, predictions)
            test_predictions.append(predicted)
        inference_seconds = time.perf_counter() - start

        metrics = _metrics_for(
            test_records, test_predictions, bootstrap_iterations, seed
        )
        result = _save_run(
            name,
            seed,
            test_records,
            test_predictions,
            metrics,
            fit_seconds,
            inference_seconds,
            output_dir,
            bootstrap_iterations,
            training_contracts=len(train_records),
            training_scope="smoke" if max_train_contracts is not None else "full",
        )
        results.append(result)
    _write_summary(output_dir, results)
    return results
