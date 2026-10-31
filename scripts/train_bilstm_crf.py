"""Train a BiLSTM-CRF using random embeddings and the fixed contract splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuad_tagging.bilstm_crf import train_bilstm_crf
from cuad_tagging.config import PROCESSED_DATA, RESULTS, SEED


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROCESSED_DATA / "contracts.jsonl.xz")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-vocab", type=int, default=50_000)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--o-only-fraction", type=float, default=0.25)
    parser.add_argument("--max-train-contracts", type=int)
    parser.add_argument("--max-eval-contracts", type=int)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = train_bilstm_crf(
        data_path=args.data,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        max_vocab=args.max_vocab,
        patience=args.patience,
        o_only_fraction=args.o_only_fraction,
        max_train_contracts=args.max_train_contracts,
        max_eval_contracts=args.max_eval_contracts,
        bootstrap_iterations=args.bootstrap_iterations,
        output_dir=args.output_dir,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
