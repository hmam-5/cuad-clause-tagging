"""Memory-batched Hugging Face token classification with word-level stitching."""

from __future__ import annotations

import csv
import json
import lzma
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .baselines import _save_run, _write_summary
from .config import BIO_LABELS, PROCESSED_DATA, RESULTS, SEED
from .metrics import span_metrics, token_metrics
from .pipeline import make_windows


@dataclass(frozen=True)
class Window:
    contract_id: str
    start: int
    end: int


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Prepared data not found: {path}")
    records: list[dict[str, Any]] = []
    with lzma.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            compact = json.loads(line)
            records.append(
                {
                    "contract_id": compact["contract_id"],
                    "split": compact["split"],
                    "tokens": compact["tokens"],
                    "labels": compact["labels"],
                }
            )
    return records


def _windows(
    records: list[dict[str, Any]],
    max_words: int,
    stride_words: int,
    tokenizer: Any | None = None,
    max_length: int = 512,
) -> list[Window]:
    if max_words < 1 or stride_words < 0 or stride_words >= max_words:
        raise ValueError("Require max_words > stride_words >= 0.")
    windows: list[Window] = []
    for record in records:
        tags = [BIO_LABELS[label_id] for label_id in record["labels"]]
        if tokenizer is None:
            boundaries = make_windows(tags, max_words, stride_words)
        else:
            encoded = tokenizer(
                [record["tokens"]],
                is_split_into_words=True,
                truncation=False,
                add_special_tokens=False,
            )
            piece_counts = [0] * len(record["tokens"])
            for word_id in encoded.word_ids(batch_index=0):
                if word_id is not None:
                    piece_counts[word_id] += 1
            if any(count == 0 for count in piece_counts):
                raise ValueError(f"Tokenizer omitted a word in {record['contract_id']}.")
            boundaries = _subword_windows(
                tags,
                piece_counts,
                max_words,
                stride_words,
                max_length - 2,
                record["contract_id"],
            )
        windows.extend(
            Window(record["contract_id"], window["start"], window["end"])
            for window in boundaries
        )
    return windows


def _subword_windows(
    tags: list[str],
    piece_counts: list[int],
    max_words: int,
    overlap_words: int,
    max_content_pieces: int,
    contract_id: str,
) -> list[dict[str, int]]:
    if len(tags) != len(piece_counts):
        raise ValueError("Tag and subword-count sequences differ in length.")
    if max_content_pieces < 1:
        raise ValueError("The maximum sequence length leaves no room for content.")
    prefix = [0]
    for count in piece_counts:
        if count < 1:
            raise ValueError(f"Word tokenizes to no pieces in {contract_id}.")
        if count > max_content_pieces:
            raise ValueError(
                f"A single word in {contract_id} exceeds the tokenizer's "
                f"{max_content_pieces}-piece window limit."
            )
        prefix.append(prefix[-1] + count)

    windows: list[dict[str, int]] = []
    start = 0
    while start < len(tags):
        end = min(start + max_words, len(tags))
        while end > start and prefix[end] - prefix[start] > max_content_pieces:
            end -= 1
        if end == start:
            raise ValueError(f"Could not fit a word into a window in {contract_id}.")
        if end < len(tags) and tags[end].startswith("I-"):
            category = tags[end].split("-", 1)[1]
            span_start = end - 1
            while (
                span_start > start
                and tags[span_start].startswith("I-")
                and tags[span_start].split("-", 1)[1] == category
            ):
                span_start -= 1
            if tags[span_start] == f"B-{category}" and span_start > start:
                end = span_start
        if end == start:
            end = min(start + 1, len(tags))
        windows.append({"start": start, "end": end})
        if end == len(tags):
            break
        overlap = min(overlap_words, max(0, (end - start - 1) // 2))
        start = max(start + 1, end - overlap)
    return windows


def _encode_batch(
    windows: list[Window],
    records_by_id: dict[str, dict[str, Any]],
    tokenizer: Any,
    max_length: int,
) -> list[dict[str, Any]]:
    texts = [
        records_by_id[window.contract_id]["tokens"][window.start : window.end]
        for window in windows
    ]
    batch = tokenizer(
        texts,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )
    encoded: list[dict[str, Any]] = []
    for batch_index, window in enumerate(windows):
        word_ids = batch.word_ids(batch_index=batch_index)
        represented = {word_id for word_id in word_ids if word_id is not None}
        expected = window.end - window.start
        if len(represented) != expected:
            raise ValueError(
                f"Tokenizer truncated words in {window.contract_id} "
                f"({window.start}:{window.end}); reduce --window-words."
            )
        labels = [-100] * len(word_ids)
        global_word_ids = [-1] * len(word_ids)
        seen: set[int] = set()
        record = records_by_id[window.contract_id]
        for subword_index, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            global_index = window.start + word_id
            global_word_ids[subword_index] = global_index
            if word_id not in seen:
                labels[subword_index] = record["labels"][global_index]
                seen.add(word_id)
        model_inputs = {
            key: batch[key][batch_index]
            for key in ("input_ids", "attention_mask", "token_type_ids")
            if key in batch
        }
        encoded.append(
            {
                "inputs": model_inputs,
                "labels": labels,
                "global_word_ids": global_word_ids,
                "contract_id": window.contract_id,
            }
        )
    return encoded


def _collate(samples: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    import torch

    length = max(len(sample["inputs"]["input_ids"]) for sample in samples)
    batch: dict[str, Any] = {}
    for key in samples[0]["inputs"]:
        pad_value = pad_token_id if key == "input_ids" else 0
        values = [
            sample["inputs"][key] + [pad_value] * (length - len(sample["inputs"][key]))
            for sample in samples
        ]
        batch[key] = torch.tensor(values, dtype=torch.long)
    batch["labels"] = torch.tensor(
        [
            sample["labels"] + [-100] * (length - len(sample["labels"]))
            for sample in samples
        ],
        dtype=torch.long,
    )
    return batch


def _class_weights(records: list[dict[str, Any]], torch: Any, device: Any) -> Any:
    counts = np.zeros(len(BIO_LABELS), dtype=np.int64)
    for record in records:
        counts += np.bincount(record["labels"], minlength=len(BIO_LABELS))
    present = counts > 0
    weights = np.zeros(len(BIO_LABELS), dtype=np.float32)
    weights[present] = np.sqrt(counts.sum() / counts[present])
    weights[present] = np.minimum(weights[present], 10.0)
    weights[present] /= weights[present].mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _run_epoch(
    model: Any,
    windows: list[Window],
    records_by_id: dict[str, dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    device: Any,
    weights: Any,
    optimizer: Any | None,
    rng: random.Random,
) -> tuple[float, dict[str, Any] | None]:
    import torch
    from torch.nn import functional

    training = optimizer is not None
    model.train(training)
    order = list(range(len(windows)))
    if training:
        rng.shuffle(order)
    loss_total = 0.0
    token_total = 0
    sums: dict[str, np.ndarray] = {}
    vote_counts: dict[str, np.ndarray] = {}
    gold_sequences: dict[str, list[str]] = {}

    for batch_start in range(0, len(order), batch_size):
        selected = [windows[index] for index in order[batch_start : batch_start + batch_size]]
        samples = _encode_batch(selected, records_by_id, tokenizer, max_length)
        batch = _collate(samples, tokenizer.pad_token_id)
        labels = batch.pop("labels").to(device)
        model_inputs = {key: value.to(device) for key, value in batch.items()}
        with torch.set_grad_enabled(training):
            logits = model(**model_inputs).logits
            valid = labels != -100
            loss_sum = functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                weight=weights,
                ignore_index=-100,
                reduction="sum",
            )
            token_count = int(valid.sum().item())
            loss = loss_sum / max(token_count, 1)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        loss_total += float(loss_sum.detach().item())
        token_total += token_count

        if not training:
            logits_cpu = logits.detach().float().cpu().numpy()
            for sample_index, sample in enumerate(samples):
                contract_id = sample["contract_id"]
                record = records_by_id[contract_id]
                if contract_id not in sums:
                    sums[contract_id] = np.zeros(
                        (len(record["tokens"]), len(BIO_LABELS)), dtype=np.float32
                    )
                    vote_counts[contract_id] = np.zeros(
                        len(record["tokens"]), dtype=np.int32
                    )
                    gold_sequences[contract_id] = [
                        BIO_LABELS[label_id] for label_id in record["labels"]
                    ]
                for position, global_index in enumerate(sample["global_word_ids"]):
                    if global_index >= 0 and sample["labels"][position] != -100:
                        sums[contract_id][global_index] += logits_cpu[sample_index, position]
                        vote_counts[contract_id][global_index] += 1

    average_loss = loss_total / max(token_total, 1)
    if training:
        return average_loss, None
    predicted: list[list[str]] = []
    gold: list[list[str]] = []
    for contract_id in sorted(gold_sequences):
        logits = sums[contract_id]
        covered = vote_counts[contract_id] > 0
        predicted_ids = np.zeros(len(logits), dtype=np.int64)
        predicted_ids[covered] = logits[covered].argmax(axis=1)
        predicted.append([BIO_LABELS[label_id] for label_id in predicted_ids])
        gold.append(gold_sequences[contract_id])
    return average_loss, {
        "gold": gold,
        "predicted": predicted,
        "contract_ids": sorted(gold_sequences),
        "token": token_metrics(gold, predicted),
        "span": span_metrics(gold, predicted),
    }


def _write_history(
    path: Path,
    history: list[dict[str, float]],
    loss_label: str = "Weighted cross-entropy loss",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=("epoch", "train_loss", "validation_loss", "validation_macro_f1"),
        )
        writer.writeheader()
        writer.writerows(history)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Loss plots require matplotlib; install requirements.txt.") from error
    epochs = [row["epoch"] for row in history]
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(epochs, [row["train_loss"] for row in history], label="Train")
    axis.plot(epochs, [row["validation_loss"] for row in history], label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel(loss_label)
    axis.set_title("Training and validation loss")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def train_transformer(
    data_path: Path = PROCESSED_DATA / "contracts.jsonl.xz",
    checkpoint: str = "bert-base-cased",
    seed: int = SEED,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 3e-5,
    max_length: int = 512,
    window_words: int = 128,
    stride_words: int = 32,
    patience: int = 2,
    max_train_contracts: int | None = None,
    max_eval_contracts: int | None = None,
    bootstrap_iterations: int = 1000,
    output_dir: Path = RESULTS,
    device_name: str = "auto",
) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Transformer training requires torch and transformers; install requirements.txt."
        ) from error
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("Epochs, batch size, and patience must be positive.")
    if max_length < 3:
        raise ValueError("--max-length must allow special tokens and at least one word.")
    if learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if max_train_contracts is not None and max_train_contracts < 1:
        raise ValueError("--max-train-contracts must be positive.")
    if max_eval_contracts is not None and max_eval_contracts < 1:
        raise ValueError("--max-eval-contracts must be positive.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_name == "auto"
        else device_name
    )

    records = load_records(data_path)
    train_records = [record for record in records if record["split"] == "train"]
    validation_records = [
        record for record in records if record["split"] == "validation"
    ]
    test_records = [record for record in records if record["split"] == "test"]
    if max_eval_contracts is not None:
        eval_sampler = random.Random(seed + 1)
        validation_records = eval_sampler.sample(
            validation_records, min(max_eval_contracts, len(validation_records))
        )
        test_records = eval_sampler.sample(
            test_records, min(max_eval_contracts, len(test_records))
        )
    if max_train_contracts is not None:
        sampler = random.Random(seed)
        train_records = sampler.sample(
            train_records, min(max_train_contracts, len(train_records))
        )
    if not train_records or not validation_records or not test_records:
        raise ValueError("Train, validation, and test splits must all be non-empty.")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for subword-to-word alignment.")
    if tokenizer.pad_token_id is None:
        raise ValueError(f"Checkpoint {checkpoint} has no tokenizer pad token.")
    # Full-contract tokenization below only counts pieces; actual model windows stay bounded.
    tokenizer.model_max_length = 1_000_000_000
    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint,
        num_labels=len(BIO_LABELS),
        id2label={index: label for index, label in enumerate(BIO_LABELS)},
        label2id={label: index for index, label in enumerate(BIO_LABELS)},
        ignore_mismatched_sizes=True,
    ).to(device)
    records_by_id = {record["contract_id"]: record for record in records}
    train_windows = _windows(
        train_records, window_words, stride_words, tokenizer, max_length
    )
    validation_windows = _windows(
        validation_records, window_words, stride_words, tokenizer, max_length
    )
    test_windows = _windows(
        test_records, window_words, stride_words, tokenizer, max_length
    )
    class_weights = _class_weights(train_records, torch, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    architecture = (
        "RoBERTa"
        if "roberta" in checkpoint.lower()
        else "DeBERTa"
        if "deberta" in checkpoint.lower()
        else "BERT"
    )
    run_id = f"{checkpoint.replace('/', '_')}-seed{seed}"
    checkpoint_dir = Path("artifacts") / "checkpoints" / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best_f1 = -1.0
    epochs_without_improvement = 0
    epoch_rng = random.Random(seed)
    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_loss, _ = _run_epoch(
            model,
            train_windows,
            records_by_id,
            tokenizer,
            max_length,
            batch_size,
            device,
            class_weights,
            optimizer,
            epoch_rng,
        )
        validation_loss, validation = _run_epoch(
            model,
            validation_windows,
            records_by_id,
            tokenizer,
            max_length,
            batch_size,
            device,
            class_weights,
            None,
            epoch_rng,
        )
        validation_f1 = float(validation["token"]["macro_f1"])
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_macro_f1": validation_f1,
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"validation_loss={validation_loss:.4f} "
            f"validation_macro_f1={validation_f1:.4f}"
        )
        if validation_f1 > best_f1:
            best_f1 = validation_f1
            epochs_without_improvement = 0
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    train_seconds = time.perf_counter() - train_start
    loss_path = output_dir / "loss" / f"{run_id}.csv"
    _write_history(loss_path, history)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir).to(device)
    inference_start = time.perf_counter()
    _, test_result = _run_epoch(
        model,
        test_windows,
        records_by_id,
        tokenizer,
        max_length,
        batch_size,
        device,
        class_weights,
        None,
        epoch_rng,
    )
    inference_seconds = time.perf_counter() - inference_start
    test_records = sorted(test_records, key=lambda record: record["contract_id"])
    result_by_id = dict(zip(test_result["contract_ids"], zip(test_result["gold"], test_result["predicted"])))
    gold = [result_by_id[record["contract_id"]][0] for record in test_records]
    predicted = [result_by_id[record["contract_id"]][1] for record in test_records]
    metrics = {
        "token": token_metrics(gold, predicted),
        "span": span_metrics(gold, predicted),
    }
    from .metrics import bootstrap_macro_f1

    metrics["bootstrap_95"] = bootstrap_macro_f1(
        gold, predicted, bootstrap_iterations, seed
    )
    result = _save_run(
        architecture,
        seed,
        test_records,
        predicted,
        metrics,
        train_seconds,
        inference_seconds,
        output_dir,
        bootstrap_iterations,
        architecture=architecture,
        checkpoint=checkpoint,
        model_name=architecture,
        run_id=run_id,
        training_contracts=len(train_records),
        training_scope=(
            "smoke"
            if max_train_contracts is not None or max_eval_contracts is not None
            else "full"
        ),
        evaluation_scope="smoke" if max_eval_contracts is not None else "full",
    )
    result["best_validation_macro_f1"] = best_f1
    result["checkpoint_path"] = str(checkpoint_dir)
    result["loss_curve"] = str(loss_path)
    detail_path = output_dir / f"{run_id}.json"
    with detail_path.open(encoding="utf-8") as source:
        details = json.load(source)
    with detail_path.open("w", encoding="utf-8") as output:
        json.dump({**details, **result}, output, indent=2)
        output.write("\n")
    _write_summary(output_dir, [result])
    return result
