"""Memory-batched Hugging Face token classification with word-level stitching."""

from __future__ import annotations

import csv
import json
import lzma
import random
import time
from contextlib import nullcontext
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


def _encode_windows(
    windows: list[Window],
    records_by_id: dict[str, dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    batch_size: int = 32,
    split_name: str = "windows",
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for offset in range(0, len(windows), batch_size):
        samples.extend(
            _encode_batch(
                windows[offset : offset + batch_size],
                records_by_id,
                tokenizer,
                max_length,
            )
        )
        completed = min(offset + batch_size, len(windows))
        if completed == batch_size or completed == len(windows) or completed % 8192 == 0:
            print(
                f"tokenized {split_name} windows={completed}/{len(windows)}",
                flush=True,
            )
    return samples


def _subsample_o_only_windows(
    windows: list[Window],
    records_by_id: dict[str, dict[str, Any]],
    fraction: float,
    rng: random.Random,
) -> tuple[list[Window], int, int]:
    if not 0 <= fraction <= 1:
        raise ValueError("--o-only-fraction must be between 0 and 1.")
    labeled: list[Window] = []
    outside: list[Window] = []
    for window in windows:
        labels = records_by_id[window.contract_id]["labels"][window.start : window.end]
        (labeled if any(label_id != 0 for label_id in labels) else outside).append(window)
    selected_outside = rng.sample(
        outside, min(len(outside), round(len(outside) * fraction))
    )
    selected = set(labeled + selected_outside)
    return (
        [window for window in windows if window in selected],
        len(labeled),
        len(selected_outside),
    )


def _resolve_precision(mode: str, device: Any, torch: Any) -> Any | None:
    if mode == "no":
        return None
    if mode == "auto":
        if device.type != "cuda":
            return None
        mode = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if device.type != "cuda":
        raise ValueError("Mixed precision is supported only on CUDA devices.")
    if mode == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("The selected CUDA device does not support bf16.")
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported mixed precision mode: {mode}")


def _repair_bio_sequence(tags: list[str]) -> list[str]:
    repaired: list[str] = []
    active_category: str | None = None
    for tag in tags:
        if tag == "O":
            repaired.append(tag)
            active_category = None
            continue
        prefix, category = tag.split("-", 1)
        if prefix == "B" or active_category != category:
            repaired.append(f"B-{category}")
        else:
            repaired.append(f"I-{category}")
        active_category = category
    return repaired


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
    samples: list[dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]],
    batch_size: int,
    pad_token_id: int,
    device: Any,
    weights: Any,
    optimizer: Any | None,
    rng: random.Random,
    gradient_accumulation_steps: int = 1,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    autocast_dtype: Any | None = None,
) -> tuple[float, dict[str, Any] | None]:
    import torch
    from torch.nn import functional

    if gradient_accumulation_steps < 1:
        raise ValueError("Gradient accumulation steps must be positive.")
    training = optimizer is not None
    model.train(training)
    order = list(range(len(samples)))
    if training:
        rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
    sample_token_counts = [
        sum(label != -100 for label in sample["labels"]) for sample in samples
    ]
    total_batches = (len(order) + batch_size - 1) // batch_size
    loss_total = 0.0
    token_total = 0
    sums: dict[str, np.ndarray] = {}
    vote_counts: dict[str, np.ndarray] = {}
    gold_sequences: dict[str, list[str]] = {}

    for batch_start in range(0, len(order), batch_size):
        batch_number = batch_start // batch_size
        selected = [samples[index] for index in order[batch_start : batch_start + batch_size]]
        batch = _collate(selected, pad_token_id)
        labels = batch.pop("labels").to(device)
        model_inputs = {key: value.to(device) for key, value in batch.items()}
        with torch.set_grad_enabled(training):
            autocast = (
                nullcontext()
                if autocast_dtype is None
                else torch.autocast(device_type=device.type, dtype=autocast_dtype)
            )
            with autocast:
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
            if training:
                group_start = (
                    batch_number // gradient_accumulation_steps
                ) * gradient_accumulation_steps
                group_size = min(
                    gradient_accumulation_steps, total_batches - group_start
                )
                group_end = min(
                    len(order),
                    (group_start + group_size) * batch_size,
                )
                group_token_count = sum(
                    sample_token_counts[index]
                    for index in order[group_start * batch_size : group_end]
                )
                scaled_loss = loss_sum / max(group_token_count, 1)
                if scaler is None:
                    scaled_loss.backward()
                else:
                    scaler.scale(scaled_loss).backward()
                final_batch = batch_number + 1 == total_batches
                if (
                    (batch_number + 1) % gradient_accumulation_steps == 0
                    or final_batch
                ):
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if scaler is None:
                        optimizer.step()
                    else:
                        scaler.step(optimizer)
                        scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()
        loss_total += float(loss_sum.detach().item())
        token_total += token_count

        if not training:
            logits_cpu = logits.detach().float().cpu().numpy()
            for sample_index, sample in enumerate(selected):
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
    raw_predicted: list[list[str]] = []
    predicted: list[list[str]] = []
    gold: list[list[str]] = []
    for contract_id in sorted(gold_sequences):
        logits = sums[contract_id]
        covered = vote_counts[contract_id] > 0
        predicted_ids = np.zeros(len(logits), dtype=np.int64)
        predicted_ids[covered] = logits[covered].argmax(axis=1)
        raw_predicted.append([BIO_LABELS[label_id] for label_id in predicted_ids])
        predicted.append(
            _repair_bio_sequence(raw_predicted[-1])
        )
        gold.append(gold_sequences[contract_id])
    return average_loss, {
        "gold": gold,
        "predicted": predicted,
        "contract_ids": sorted(gold_sequences),
        "token": token_metrics(gold, predicted),
        "span": span_metrics(gold, predicted),
        "predicted_b_labels": sum(
            label.startswith("B-") for sequence in predicted for label in sequence
        ),
        "raw_predicted_b_labels": sum(
            label.startswith("B-") for sequence in raw_predicted for label in sequence
        ),
    }


def _write_history(
    path: Path,
    history: list[dict[str, float]],
    loss_label: str = "Weighted cross-entropy loss",
    best_epoch: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "epoch",
                "train_loss",
                "validation_loss",
                "validation_macro_f1",
                "train_macro_f1",
                "learning_rate",
                "elapsed_seconds",
                "is_best_epoch",
            ),
        )
        writer.writeheader()
        writer.writerows(
            [
                {**row, "is_best_epoch": int(row["epoch"]) == best_epoch}
                for row in history
            ]
        )
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
    if best_epoch is not None:
        axis.axvline(best_epoch, color="gray", linestyle="--", label="Best epoch")
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
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 3e-5,
    max_length: int = 512,
    window_words: int = 128,
    stride_words: int = 32,
    patience: int = 2,
    o_only_fraction: float = 0.25,
    mixed_precision: str = "auto",
    max_train_contracts: int | None = None,
    max_eval_contracts: int | None = None,
    bootstrap_iterations: int = 1000,
    output_dir: Path = RESULTS,
    device_name: str = "auto",
) -> dict[str, Any]:
    try:
        import torch
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
    except ImportError as error:
        raise RuntimeError(
            "Transformer training requires torch and transformers; install requirements.txt."
        ) from error
    if epochs < 1 or batch_size < 1 or gradient_accumulation_steps < 1 or patience < 1:
        raise ValueError(
            "Epochs, batch size, gradient accumulation, and patience must be positive."
        )
    if max_length < 3:
        raise ValueError("--max-length must allow special tokens and at least one word.")
    if learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if not 0 <= o_only_fraction <= 1:
        raise ValueError("--o-only-fraction must be between 0 and 1.")
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
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    precision_dtype = _resolve_precision(mixed_precision, device, torch)
    precision_name = (
        "bf16"
        if precision_dtype == torch.bfloat16
        else "fp16"
        if precision_dtype == torch.float16
        else "no"
    )
    print(
        f"device={device} mixed_precision={precision_name} "
        f"batch_size={batch_size} accumulation_steps={gradient_accumulation_steps}",
        flush=True,
    )
    scaler_enabled = precision_dtype == torch.float16 and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

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

    print(
        f"loaded contracts train={len(train_records)} "
        f"validation={len(validation_records)} test={len(test_records)}",
        flush=True,
    )
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
    print("building subword-safe windows", flush=True)
    train_windows = _windows(
        train_records, window_words, stride_words, tokenizer, max_length
    )
    validation_windows = _windows(
        validation_records, window_words, stride_words, tokenizer, max_length
    )
    test_windows = _windows(
        test_records, window_words, stride_words, tokenizer, max_length
    )
    print(
        f"window counts train={len(train_windows)} "
        f"validation={len(validation_windows)} test={len(test_windows)}",
        flush=True,
    )
    train_windows, labeled_window_count, o_only_window_count = (
        _subsample_o_only_windows(
            train_windows,
            records_by_id,
            o_only_fraction,
            random.Random(seed + 2),
        )
    )
    if not train_windows:
        raise ValueError("O-only subsampling left no training windows.")
    print(
        f"training windows labeled={labeled_window_count} "
        f"retained_o_only={o_only_window_count}",
        flush=True,
    )
    encoding_start = time.perf_counter()
    train_samples = _encode_windows(
        train_windows, records_by_id, tokenizer, max_length, split_name="train"
    )
    validation_samples = _encode_windows(
        validation_windows,
        records_by_id,
        tokenizer,
        max_length,
        split_name="validation",
    )
    test_samples = _encode_windows(
        test_windows,
        records_by_id,
        tokenizer,
        max_length,
        split_name="test",
    )
    print(
        f"tokenization cache ready seconds={time.perf_counter() - encoding_start:.1f}",
        flush=True,
    )
    class_weights = _class_weights(train_records, torch, device)
    weight_decay = 0.01
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_batches = (len(train_samples) + batch_size - 1) // batch_size
    updates_per_epoch = (
        train_batches + gradient_accumulation_steps - 1
    ) // gradient_accumulation_steps
    total_updates = updates_per_epoch * epochs
    warmup_steps = int(total_updates * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    architecture = (
        "RoBERTa"
        if "roberta" in checkpoint.lower()
        else "DeBERTa"
        if "deberta" in checkpoint.lower()
        else "BERT"
    )
    run_scope = (
        "smoke"
        if max_train_contracts is not None or max_eval_contracts is not None
        else "full"
    )
    run_id = f"{checkpoint.replace('/', '_')}-{run_scope}-seed{seed}"
    checkpoint_dir = (
        Path("artifacts") / "checkpoints" / f"{run_id}-{output_dir.name}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best_f1 = -1.0
    best_validation_loss = float("inf")
    best_epoch = 0
    best_validation: dict[str, Any] | None = None
    epochs_without_improvement = 0
    epoch_rng = random.Random(seed)
    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, _ = _run_epoch(
            model,
            train_samples,
            records_by_id,
            batch_size,
            tokenizer.pad_token_id,
            device,
            class_weights,
            optimizer,
            epoch_rng,
            gradient_accumulation_steps,
            scheduler,
            scaler,
            precision_dtype,
        )
        validation_loss, validation = _run_epoch(
            model,
            validation_samples,
            records_by_id,
            batch_size,
            tokenizer.pad_token_id,
            device,
            class_weights,
            None,
            epoch_rng,
            autocast_dtype=precision_dtype,
        )
        validation_f1 = float(validation["token"]["macro_f1"])
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_macro_f1": validation_f1,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "elapsed_seconds": time.perf_counter() - epoch_start,
        }
        train_result = None
        if max_train_contracts is not None:
            _, train_result = _run_epoch(
                model,
                train_samples,
                records_by_id,
                batch_size,
                tokenizer.pad_token_id,
                device,
                class_weights,
                None,
                epoch_rng,
                autocast_dtype=precision_dtype,
            )
            row["train_macro_f1"] = float(train_result["token"]["macro_f1"])
        row["elapsed_seconds"] = time.perf_counter() - epoch_start
        history.append(row)
        progress = (
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"validation_loss={validation_loss:.4f} "
            f"validation_macro_f1={validation_f1:.4f}"
        )
        if train_result is not None:
            progress += (
                f" train_macro_f1={row['train_macro_f1']:.4f}"
                f" train_predicted_b_labels={train_result['predicted_b_labels']}"
                f" train_raw_predicted_b_labels="
                f"{train_result['raw_predicted_b_labels']}"
            )
        print(progress, flush=True)
        if validation_f1 > best_f1 or (
            validation_f1 == best_f1 and validation_loss < best_validation_loss
        ):
            best_f1 = validation_f1
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_validation = validation
            epochs_without_improvement = 0
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    train_seconds = time.perf_counter() - train_start
    loss_path = output_dir / "loss" / f"{run_id}.csv"
    _write_history(loss_path, history, best_epoch=best_epoch)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir).to(device)
    inference_start = time.perf_counter()
    _, test_result = _run_epoch(
        model,
        test_samples,
        records_by_id,
        batch_size,
        tokenizer.pad_token_id,
        device,
        class_weights,
        None,
        epoch_rng,
        autocast_dtype=precision_dtype,
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
    if best_validation is None:
        raise RuntimeError("Training completed without a best validation checkpoint.")
    from .metrics import bootstrap_macro_f1

    metrics["bootstrap_95"] = bootstrap_macro_f1(
        gold, predicted, bootstrap_iterations, seed
    )
    validation_metrics = {
        "token": best_validation["token"],
        "span": best_validation["span"],
        "bootstrap_95": bootstrap_macro_f1(
            best_validation["gold"],
            best_validation["predicted"],
            bootstrap_iterations,
            seed,
        ),
    }
    training_scope = run_scope
    evaluation_scope = "smoke" if max_eval_contracts is not None else "full"
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
        training_scope=training_scope,
        evaluation_scope=evaluation_scope,
    )
    result["best_validation_macro_f1"] = best_f1
    result["best_epoch"] = best_epoch
    result["epochs"] = len(history)
    result["learning_rate"] = learning_rate
    result["effective_batch_size"] = batch_size * gradient_accumulation_steps
    result["checkpoint_path"] = str(checkpoint_dir)
    result["predictions_path"] = str(
        output_dir / f"{run_id}_predictions.jsonl"
    )
    result["loss_curve"] = str(loss_path)
    result["validation_metrics"] = validation_metrics
    result["training_hyperparameters"] = {
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "scheduler": "linear warmup then linear decay",
        "warmup_fraction": 0.1,
        "warmup_steps": warmup_steps,
        "max_length": max_length,
        "window_words": window_words,
        "stride_words": stride_words,
        "patience": patience,
        "o_only_fraction": o_only_fraction,
        "labeled_training_windows": labeled_window_count,
        "retained_o_only_training_windows": o_only_window_count,
        "mixed_precision": precision_name,
        "class_weight_formula": "sqrt(total label count / label count), capped at 10 and mean-normalized",
        "gradient_clip_norm": 1.0,
    }
    if max_train_contracts is not None and train_result is not None:
        result["smoke_train_macro_f1"] = train_result["token"]["macro_f1"]
        result["smoke_train_predicted_b_labels"] = train_result["predicted_b_labels"]
        result["smoke_train_raw_predicted_b_labels"] = train_result[
            "raw_predicted_b_labels"
        ]
    validation_record = {
        **result,
        "run_id": f"{run_id}-validation",
        "evaluation_split": "validation",
        "token_macro_f1": validation_metrics["token"]["macro_f1"],
        "span_strict_micro_f1": validation_metrics["span"]["strict"]["micro_f1"],
        "span_strict_macro_f1": validation_metrics["span"]["strict"]["macro_f1"],
        "span_partial_micro_f1": validation_metrics["span"]["partial"]["micro_f1"],
        "span_partial_macro_f1": validation_metrics["span"]["partial"]["macro_f1"],
        "bootstrap_95_lower": validation_metrics["bootstrap_95"]["lower"],
        "bootstrap_95_upper": validation_metrics["bootstrap_95"]["upper"],
        "evaluation_contracts": len(best_validation["contract_ids"]),
        "test_contracts": None,
        "inference_ms_per_contract": None,
        "predictions_path": None,
    }
    validation_path = output_dir / f"{run_id}-validation.json"
    validation_path.write_text(
        json.dumps(
            {
                **validation_record,
                "per_category": validation_metrics["token"]["per_category"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    result["validation_metrics_file"] = str(validation_path)
    detail_path = output_dir / f"{run_id}.json"
    with detail_path.open(encoding="utf-8") as source:
        details = json.load(source)
    with detail_path.open("w", encoding="utf-8") as output:
        json.dump({**details, **result}, output, indent=2)
        output.write("\n")
    _write_summary(output_dir, [validation_record, result])
    return result
