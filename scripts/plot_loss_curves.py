"""Plot training and validation losses from saved experiment histories."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuad_tagging.config import RESULTS


def plot_loss_curves(
    loss_dir: Path = RESULTS / "loss",
    output_dir: Path = RESULTS / "figures",
) -> list[str]:
    histories = []
    for path in sorted(loss_dir.glob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        if rows and {"epoch", "train_loss", "validation_loss"}.issubset(rows[0]):
            histories.append((path.stem, rows))
    if not histories:
        raise FileNotFoundError(f"No usable loss histories found in {loss_dir}.")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Loss plots require matplotlib; install requirements.txt.") from error

    columns = 2
    row_count = (len(histories) + columns - 1) // columns
    figure, axes = plt.subplots(
        row_count, columns, figsize=(14, 4.5 * row_count), squeeze=False
    )
    for index, (run_id, history) in enumerate(histories):
        axis = axes[index // columns][index % columns]
        epochs = [int(float(row["epoch"])) for row in history]
        axis.plot(epochs, [float(row["train_loss"]) for row in history], marker="o", label="Train")
        axis.plot(
            epochs,
            [float(row["validation_loss"]) for row in history],
            marker="s",
            label="Validation",
        )
        best_epochs = [
            epoch
            for epoch, row in zip(epochs, history)
            if row.get("is_best_epoch", "").lower() == "true"
        ]
        if best_epochs:
            axis.axvline(
                best_epochs[-1], color="gray", linestyle="--", label="Best epoch"
            )
        axis.set_title(run_id)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(alpha=0.2)
        axis.legend()
    for index in range(len(histories), row_count * columns):
        axes[index // columns][index % columns].set_visible(False)

    output_dir.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_dir / "loss_curves.png", dpi=220)
    figure.savefig(output_dir / "loss_curves.pdf")
    plt.close(figure)
    return [run_id for run_id, _ in histories]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss-dir", type=Path, default=RESULTS / "loss")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "figures")
    args = parser.parse_args()
    print({"plotted_runs": plot_loss_curves(args.loss_dir, args.output_dir)})


if __name__ == "__main__":
    main()
