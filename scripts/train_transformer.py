"""Fine-tune a Hugging Face token classifier with contract-level evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuad_tagging.config import PROCESSED_DATA, RESULTS, SEED
from cuad_tagging.transformer import train_transformer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROCESSED_DATA / "contracts.jsonl.xz")
    parser.add_argument("--checkpoint", default="bert-base-cased")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--window-words", type=int, default=128)
    parser.add_argument("--stride-words", type=int, default=32)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max-train-contracts", type=int)
    parser.add_argument("--max-eval-contracts", type=int)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = train_transformer(
        data_path=args.data,
        checkpoint=args.checkpoint,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        window_words=args.window_words,
        stride_words=args.stride_words,
        patience=args.patience,
        max_train_contracts=args.max_train_contracts,
        max_eval_contracts=args.max_eval_contracts,
        bootstrap_iterations=args.bootstrap_iterations,
        output_dir=args.output_dir,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
