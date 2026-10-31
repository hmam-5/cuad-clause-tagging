"""Run offset-safe CUAD preprocessing and write cleaned data and statistics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuad_tagging.config import RAW_DATA, SEED
from cuad_tagging.pipeline import preprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_DATA)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128)
    args = parser.parse_args()
    stats = preprocess(
        args.input, args.output_dir, args.seed, args.window_size, args.stride
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
