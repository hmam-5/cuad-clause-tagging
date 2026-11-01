"""Random-embedding BiLSTM-CRF sequence tagger and training loop."""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .baselines import _save_run, _write_summary, load_records
from .config import BIO_LABELS, RESULTS, SEED
from .metrics import bootstrap_macro_f1, span_metrics, token_metrics
from .transformer import _write_history


def _build_vocabulary(records: list[dict[str, Any]], limit: int) -> dict[str, int]:
    counts: Counter[str] = Counter(
        word for record in records for word in record["tokens"]
    )
    vocabulary = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counts.most_common(max(0, limit - len(vocabulary))):
        vocabulary[word] = len(vocabulary)
    return vocabulary


def _window_samples(
    records: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], int, int]]:
    return [
        (record, start, end)
        for record in records
        for start, end in record["windows"]
        if end > start
    ]


def _collate(
    samples: list[tuple[dict[str, Any], int, int]],
    vocabulary: dict[str, int],
    torch: Any,
) -> tuple[Any, Any, Any, list[int]]:
    max_length = max(end - start for _, start, end in samples)
    input_ids = []
    labels = []
    lengths = []
    for record, start, end in samples:
        words = record["tokens"][start:end]
        label_ids = record["labels"][start:end]
        lengths.append(len(words))
        input_ids.append(
            [vocabulary.get(word, 1) for word in words]
            + [0] * (max_length - len(words))
        )
        labels.append(
            label_ids + [-100] * (max_length - len(label_ids))
        )
    ids = torch.tensor(input_ids, dtype=torch.long)
    tag_ids = torch.tensor(labels, dtype=torch.long)
    mask = torch.arange(max_length).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    return ids, tag_ids, mask, lengths


def _score(
    model: Any,
    records: list[dict[str, Any]],
    windows: list[tuple[dict[str, Any], int, int]],
    vocabulary: dict[str, int],
    batch_size: int,
    device: Any,
    torch: Any,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    loss_total = 0.0
    sequence_total = 0
    votes: dict[str, list[Counter[int]]] = {
        record["contract_id"]: [Counter() for _ in record["tokens"]]
        for record in records
    }
    with torch.no_grad():
        for offset in range(0, len(windows), batch_size):
            batch_windows = windows[offset : offset + batch_size]
            ids, labels, mask, _ = _collate(batch_windows, vocabulary, torch)
            ids, labels, mask = ids.to(device), labels.to(device), mask.to(device)
            loss = model.neg_log_likelihood(ids, labels, mask)
            loss_total += float(loss.item()) * len(batch_windows)
            sequence_total += len(batch_windows)
            emissions = model(ids, mask)
            decoded = model.decode(emissions, mask)
            for sample, prediction in zip(batch_windows, decoded):
                record, start, _ = sample
                for index, tag_id in enumerate(prediction, start=start):
                    votes[record["contract_id"]][index][tag_id] += 1
    gold_sequences: list[list[str]] = []
    predicted_sequences: list[list[str]] = []
    for record in records:
        contract_id = record["contract_id"]
        gold_sequences.append([BIO_LABELS[label_id] for label_id in record["labels"]])
        predicted_sequences.append(
            [
                BIO_LABELS[
                    max(counter, key=lambda tag: (counter[tag], -tag))
                ]
                if counter
                else "O"
                for counter in votes[contract_id]
            ]
        )
    return loss_total / max(sequence_total, 1), {
        "gold": gold_sequences,
        "predicted": predicted_sequences,
        "token": token_metrics(gold_sequences, predicted_sequences),
        "span": span_metrics(gold_sequences, predicted_sequences),
    }


def train_bilstm_crf(
    data_path: Path,
    seed: int = SEED,
    epochs: int = 8,
    batch_size: int = 16,
    embedding_dim: int = 100,
    hidden_dim: int = 128,
    max_vocab: int = 50_000,
    patience: int = 2,
    o_only_fraction: float = 0.25,
    max_train_contracts: int | None = None,
    max_eval_contracts: int | None = None,
    bootstrap_iterations: int = 1000,
    output_dir: Path = RESULTS,
    device_name: str = "auto",
) -> dict[str, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError("BiLSTM-CRF training requires PyTorch.") from error
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("Epochs, batch size, and patience must be positive.")
    if not 0 <= o_only_fraction <= 1:
        raise ValueError("--o-only-fraction must be between 0 and 1.")
    if embedding_dim < 1 or hidden_dim < 2 or hidden_dim % 2:
        raise ValueError("Embedding size must be positive and hidden size must be even.")
    if max_vocab < 2:
        raise ValueError("--max-vocab must be at least 2.")
    if max_train_contracts is not None and max_train_contracts < 1:
        raise ValueError("--max-train-contracts must be positive.")
    if max_eval_contracts is not None and max_eval_contracts < 1:
        raise ValueError("--max-eval-contracts must be positive.")

    class Model(nn.Module):
        def __init__(self, vocabulary_size: int) -> None:
            super().__init__()
            self.embedding = nn.Embedding(
                vocabulary_size, embedding_dim, padding_idx=0
            )
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim // 2,
                batch_first=True,
                bidirectional=True,
            )
            self.projection = nn.Linear(hidden_dim, len(BIO_LABELS))
            self.transitions = nn.Parameter(
                torch.zeros(len(BIO_LABELS), len(BIO_LABELS))
            )
            self.start_transitions = nn.Parameter(torch.zeros(len(BIO_LABELS)))
            self.end_transitions = nn.Parameter(torch.zeros(len(BIO_LABELS)))

        def forward(self, ids: Any, mask: Any) -> Any:
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

            lengths = mask.sum(dim=1).cpu()
            embedded = self.embedding(ids)
            packed = pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.lstm(packed)
            outputs, _ = pad_packed_sequence(
                packed_output, batch_first=True, total_length=ids.shape[1]
            )
            return self.projection(outputs)

        def neg_log_likelihood(self, ids: Any, labels: Any, mask: Any) -> Any:
            emissions = self(ids, mask)
            safe_labels = labels.masked_fill(labels < 0, 0)
            batch_size, sequence_length = labels.shape
            rows = torch.arange(batch_size, device=labels.device)
            gold_score = self.start_transitions[safe_labels[:, 0]]
            gold_score = gold_score + emissions[rows, 0, safe_labels[:, 0]]
            for step in range(1, sequence_length):
                transition = self.transitions[
                    safe_labels[:, step - 1], safe_labels[:, step]
                ]
                emission = emissions[rows, step, safe_labels[:, step]]
                gold_score = gold_score + (transition + emission) * mask[:, step]
            last_indices = mask.long().sum(dim=1) - 1
            last_tags = safe_labels[rows, last_indices]
            gold_score = gold_score + self.end_transitions[last_tags]

            alpha = self.start_transitions + emissions[:, 0]
            for step in range(1, sequence_length):
                candidates = (
                    alpha.unsqueeze(2)
                    + self.transitions.unsqueeze(0)
                    + emissions[:, step].unsqueeze(1)
                )
                next_alpha = torch.logsumexp(candidates, dim=1)
                alpha = torch.where(mask[:, step].unsqueeze(1), next_alpha, alpha)
            partition = torch.logsumexp(alpha + self.end_transitions, dim=1)
            return (partition - gold_score).mean()

        def decode(self, emissions: Any, mask: Any) -> list[list[int]]:
            decoded: list[list[int]] = []
            lengths = mask.long().sum(dim=1).tolist()
            for row, length in enumerate(lengths):
                score = self.start_transitions + emissions[row, 0]
                backpointers = []
                for step in range(1, length):
                    candidates = score.unsqueeze(1) + self.transitions
                    best_score, best_path = candidates.max(dim=0)
                    score = best_score + emissions[row, step]
                    backpointers.append(best_path)
                last = int((score + self.end_transitions).argmax().item())
                path = [last]
                for pointer in reversed(backpointers):
                    last = int(pointer[last].item())
                    path.append(last)
                decoded.append(list(reversed(path)))
            return decoded

    random.seed(seed)
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
        train_records = random.Random(seed).sample(
            train_records, min(max_train_contracts, len(train_records))
        )
    vocabulary = _build_vocabulary(train_records, max_vocab)
    train_windows = _window_samples(train_records)
    rng = random.Random(seed)
    labeled_windows = [
        sample
        for sample in train_windows
        if any(label_id != 0 for label_id in sample[0]["labels"][sample[1] : sample[2]])
    ]
    outside_windows = [
        sample
        for sample in train_windows
        if not any(label_id != 0 for label_id in sample[0]["labels"][sample[1] : sample[2]])
    ]
    kept_outside = rng.sample(
        outside_windows, min(len(outside_windows), round(len(outside_windows) * o_only_fraction))
    )
    train_windows = labeled_windows + kept_outside
    rng.shuffle(train_windows)
    validation_windows = _window_samples(validation_records)
    test_windows = _window_samples(test_records)
    if not train_windows or not validation_windows or not test_windows:
        raise ValueError("Train, validation, and test splits must all be non-empty.")

    model = Model(len(vocabulary)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    run_id = f"bilstm-crf-seed{seed}"
    checkpoint_path = (
        Path("artifacts")
        / "checkpoints"
        / f"{run_id}-{output_dir.name}.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    wait = 0
    history: list[dict[str, float]] = []
    train_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(train_windows)
        loss_total = 0.0
        sequence_total = 0
        for offset in range(0, len(train_windows), batch_size):
            batch_samples = train_windows[offset : offset + batch_size]
            ids, labels, mask, _ = _collate(batch_samples, vocabulary, torch)
            ids, labels, mask = ids.to(device), labels.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.neg_log_likelihood(ids, labels, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_total += float(loss.detach().item()) * len(batch_samples)
            sequence_total += len(batch_samples)
        train_loss = loss_total / max(sequence_total, 1)
        validation_loss, validation = _score(
            model,
            validation_records,
            validation_windows,
            vocabulary,
            batch_size,
            device,
            torch,
        )
        validation_f1 = float(validation["token"]["macro_f1"])
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_macro_f1": validation_f1,
            }
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"validation_loss={validation_loss:.4f} "
            f"validation_macro_f1={validation_f1:.4f}"
        )
        if validation_f1 > best_f1:
            best_f1 = validation_f1
            wait = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "vocabulary": vocabulary,
                    "embedding_dim": embedding_dim,
                    "hidden_dim": hidden_dim,
                    "seed": seed,
                },
                checkpoint_path,
            )
        else:
            wait += 1
            if wait >= patience:
                break

    train_seconds = time.perf_counter() - train_start
    history_path = output_dir / "loss" / f"{run_id}.csv"
    _write_history(history_path, history, "CRF negative log-likelihood")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    inference_start = time.perf_counter()
    _, test_result = _score(
        model,
        test_records,
        test_windows,
        vocabulary,
        batch_size,
        device,
        torch,
    )
    inference_seconds = time.perf_counter() - inference_start
    metrics = {
        "token": token_metrics(test_result["gold"], test_result["predicted"]),
        "span": span_metrics(test_result["gold"], test_result["predicted"]),
        "bootstrap_95": bootstrap_macro_f1(
            test_result["gold"],
            test_result["predicted"],
            bootstrap_iterations,
            seed,
        ),
    }
    result = _save_run(
        "BiLSTM-CRF",
        seed,
        test_records,
        test_result["predicted"],
        metrics,
        train_seconds,
        inference_seconds,
        output_dir,
        bootstrap_iterations,
        architecture="BiLSTM-CRF",
        checkpoint=f"random embeddings ({embedding_dim}d)",
        run_id=run_id,
        training_contracts=len(train_records),
        training_scope=(
            "smoke"
            if max_train_contracts is not None or max_eval_contracts is not None
            else "full"
        ),
        evaluation_scope="smoke" if max_eval_contracts is not None else "full",
    )
    result.update(
        {
            "best_validation_macro_f1": best_f1,
            "checkpoint_path": str(checkpoint_path),
            "loss_curve": str(history_path),
            "o_only_window_fraction": o_only_fraction,
            "training_windows": len(train_windows),
        }
    )
    detail_path = output_dir / f"{run_id}.json"
    with detail_path.open(encoding="utf-8") as source:
        details = json.load(source)
    with detail_path.open("w", encoding="utf-8") as output:
        json.dump({**details, **result}, output, indent=2)
        output.write("\n")
    _write_summary(output_dir, [result])
    return result
