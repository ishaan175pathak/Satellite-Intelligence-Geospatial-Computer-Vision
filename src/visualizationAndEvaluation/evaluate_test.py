"""Pixel-level evaluation of the trained models on a WHU split.

Computes IoU, Dice, precision, recall, F1 and pixel accuracy from pixel
counts accumulated over the whole split (the same global-count definition
the trainer uses for validation Dice / IoU). This fills the empty
precision / recall / F1 cells in the README tables.

Note: for binary segmentation with global counts, F1 equals Dice exactly.
Both are printed because the README tables list both.

Usage (from the repository root):

    # Sanity check first: should reproduce the README validation numbers
    # (U-Net Dice 0.9416 / IoU 0.8896) up to small numerical differences.
    python -m src.visualizationAndEvaluation.evaluate_test --split val

    # Held-out test numbers for both models and their ensemble
    python -m src.visualizationAndEvaluation.evaluate_test --split test
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict

import numpy as np

from ..code.dataset import WHUBuildingDataset
from ..inference.predictor import (
    ARCHS,
    DEFAULT_CHECKPOINTS,
    SegmentationPredictor,
    iter_split_probabilities,
)


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


@dataclass
class PixelCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    tiles: int = 0

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        """pred, target: boolean arrays of shape [B, H, W]."""

        self.tp += int(np.count_nonzero(pred & target))
        self.fp += int(np.count_nonzero(pred & ~target))
        self.fn += int(np.count_nonzero(~pred & target))
        self.tn += int(np.count_nonzero(~pred & ~target))
        self.tiles += int(pred.shape[0])

    def metrics(self) -> Dict[str, float]:
        precision = safe_div(self.tp, self.tp + self.fp)
        recall = safe_div(self.tp, self.tp + self.fn)

        return {
            "iou": safe_div(self.tp, self.tp + self.fp + self.fn),
            "dice": safe_div(2 * self.tp, 2 * self.tp + self.fp + self.fn),
            "precision": precision,
            "recall": recall,
            "f1": safe_div(2 * precision * recall, precision + recall),
            "pixel_accuracy": safe_div(
                self.tp + self.tn, self.tp + self.fp + self.fn + self.tn
            ),
            "tiles": self.tiles,
        }


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Pixel metrics for the U-Net, SegFormer and their ensemble."
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--archs", nargs="+", choices=ARCHS, default=list(ARCHS)
    )
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Skip the averaged-probability ensemble row.",
    )
    parser.add_argument(
        "--unet-checkpoint",
        default=os.path.join(os.getcwd(), DEFAULT_CHECKPOINTS["unet"]),
    )
    parser.add_argument(
        "--segformer-checkpoint",
        default=os.path.join(os.getcwd(), DEFAULT_CHECKPOINTS["segformer"]),
    )
    parser.add_argument("--segformer-name", default="nvidia/mit-b0")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    predictor = SegmentationPredictor.from_checkpoints(
        args.archs,
        {
            "unet": args.unet_checkpoint,
            "segformer": args.segformer_checkpoint,
        },
        segformer_name=args.segformer_name,
    )

    names = [
        n for n in predictor.names if not (n == "ensemble" and args.no_ensemble)
    ]

    dataset = WHUBuildingDataset(split=args.split, tile_size=512)

    counts = {name: PixelCounts() for name in names}

    for _, gt, probs in iter_split_probabilities(
        predictor, dataset, args.batch_size, desc=f"Evaluating {args.split}"
    ):
        for name in names:
            counts[name].update(probs[name] > args.threshold, gt)

    rows = []

    for name in names:
        rows.append({"model": name, "split": args.split, **counts[name].metrics()})

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"pixel_metrics_{args.split}.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\nPixel metrics on the {args.split} split "
        f"(threshold {args.threshold}, {rows[0]['tiles']} tiles)\n"
    )
    print("| Model | IoU | Dice | Precision | Recall | F1 | Pixel acc. |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")

    for r in rows:
        print(
            f"| {r['model']} | {r['iou']:.4f} | {r['dice']:.4f} | "
            f"{r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | "
            f"{r['pixel_accuracy']:.4f} |"
        )

    print(f"\n[INFO] Saved: {csv_path}")


if __name__ == "__main__":
    main()
