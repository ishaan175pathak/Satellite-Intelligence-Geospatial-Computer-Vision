"""Object-level (building-by-building) evaluation of extracted polygons.

Predicted polygons are matched one-to-one with ground-truth polygons at
``IoU >= iou_threshold`` (default 0.5). GT polygons come from the *same*
extraction applied to the GT masks (exact pixel-edge outlines, no
simplification, no area filter), because the ground truth is raster only.

Reported: building precision / recall / F1, mean IoU of matched buildings,
false-positive and missed-building counts, and false-positive polygons on
tiles that contain no buildings.

Usage (from the repository root):

    # 1. Tune the post-processing on the VALIDATION split
    python -m src.geospatial.evaluate_polygons --arch unet --split val --sweep

    # 2. Report once on the TEST split with the chosen settings
    python -m src.geospatial.evaluate_polygons --arch unet --split test \
        --threshold 0.5 --min-area 30 --min-score 0.0 --simplify 1.0

    # Ceiling check: use the GT mask as the "prediction" (no model needed).
    # Shows how much the extraction settings alone cost (min-area filter,
    # simplification).
    python -m src.geospatial.evaluate_polygons --arch oracle --split val
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from .polygons import PolygonConfig, extract_buildings, mask_to_buildings


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


# --------------------------------------------------------------------------
# Matching and metrics
# --------------------------------------------------------------------------


@dataclass
class MatchResult:
    matches: List[Tuple[int, int, float]]  # (pred index, gt index, IoU)
    unmatched_pred: List[int]  # false positives
    unmatched_gt: List[int]  # missed buildings


def match_polygons(
    preds: Sequence[Polygon],
    gts: Sequence[Polygon],
    iou_threshold: float = 0.5,
) -> MatchResult:
    """One-to-one greedy matching by descending IoU.

    Only pairs with ``IoU >= iou_threshold`` can match. Predicted polygons
    are disjoint, so for thresholds >= 0.5 no ground-truth building can
    match two predictions and the greedy result is the unique matching.
    """

    if not preds or not gts:
        return MatchResult([], list(range(len(preds))), list(range(len(gts))))

    tree = STRtree(list(gts))

    candidates: List[Tuple[float, int, int]] = []

    for pi, pred in enumerate(preds):
        for gi in tree.query(pred, predicate="intersects"):
            gi = int(gi)
            intersection = pred.intersection(gts[gi]).area

            if intersection <= 0:
                continue

            union = pred.area + gts[gi].area - intersection
            iou = intersection / union if union > 0 else 0.0

            if iou >= iou_threshold:
                candidates.append((iou, pi, gi))

    candidates.sort(reverse=True)

    used_pred, used_gt = set(), set()
    matches: List[Tuple[int, int, float]] = []

    for iou, pi, gi in candidates:
        if pi in used_pred or gi in used_gt:
            continue

        used_pred.add(pi)
        used_gt.add(gi)
        matches.append((pi, gi, iou))

    return MatchResult(
        matches,
        [i for i in range(len(preds)) if i not in used_pred],
        [i for i in range(len(gts)) if i not in used_gt],
    )


@dataclass
class ObjectMetrics:
    """Counts that add up across tiles; ratios are derived from them."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    matched_iou_sum: float = 0.0
    n_tiles: int = 0
    n_empty_tiles: int = 0  # tiles whose GT has no building
    fp_on_empty_tiles: int = 0  # predicted polygons on those tiles

    def add(self, other: "ObjectMetrics") -> "ObjectMetrics":
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.matched_iou_sum += other.matched_iou_sum
        self.n_tiles += other.n_tiles
        self.n_empty_tiles += other.n_empty_tiles
        self.fp_on_empty_tiles += other.fp_on_empty_tiles
        return self

    @property
    def n_pred(self) -> int:
        return self.tp + self.fp

    @property
    def n_gt(self) -> int:
        return self.tp + self.fn

    @property
    def precision(self) -> float:
        return safe_div(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return safe_div(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        return safe_div(
            2 * self.precision * self.recall,
            self.precision + self.recall,
        )

    @property
    def mean_matched_iou(self) -> float:
        return safe_div(self.matched_iou_sum, self.tp)

    def to_dict(self) -> Dict[str, float]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "n_pred": self.n_pred,
            "n_gt": self.n_gt,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mean_matched_iou": self.mean_matched_iou,
            "n_tiles": self.n_tiles,
            "n_empty_tiles": self.n_empty_tiles,
            "fp_on_empty_tiles": self.fp_on_empty_tiles,
        }


def score_tile(
    pred_polygons: Sequence[Polygon],
    gt_polygons: Sequence[Polygon],
    iou_threshold: float = 0.5,
) -> ObjectMetrics:
    """Object metrics for a single tile."""

    result = match_polygons(pred_polygons, gt_polygons, iou_threshold)

    metrics = ObjectMetrics(
        tp=len(result.matches),
        fp=len(result.unmatched_pred),
        fn=len(result.unmatched_gt),
        matched_iou_sum=sum(iou for _, _, iou in result.matches),
        n_tiles=1,
    )

    if not gt_polygons:
        metrics.n_empty_tiles = 1
        metrics.fp_on_empty_tiles = len(pred_polygons)

    return metrics


# --------------------------------------------------------------------------
# Evaluating a split
# --------------------------------------------------------------------------


@dataclass
class TileRecord:
    file: str
    probs: np.ndarray  # [H, W] float16 (halves memory; error < 0.001)
    gt_polygons: List[Polygon]


def collect_tiles(
    batches: Iterable[Tuple[List[str], np.ndarray, Dict[str, np.ndarray]]],
    name: str,
    gt_min_area_px: int = 1,
) -> List[TileRecord]:
    """Cache probabilities and GT polygons for every tile once, so many
    post-processing settings can be scored without re-running the model."""

    tiles: List[TileRecord] = []

    for files, gt_masks, probs in batches:
        for i, file in enumerate(files):
            gt = mask_to_buildings(
                gt_masks[i],
                min_area_px=gt_min_area_px,
                simplify_tolerance=0.0,
            )

            tiles.append(
                TileRecord(
                    file=file,
                    probs=probs[name][i].astype(np.float16),
                    gt_polygons=[b.polygon for b in gt],
                )
            )

    return tiles


def evaluate_tiles(
    tiles: Sequence[TileRecord],
    config: PolygonConfig,
    iou_threshold: float = 0.5,
) -> ObjectMetrics:

    total = ObjectMetrics()

    for tile in tiles:
        buildings = extract_buildings(tile.probs.astype(np.float32), config)

        total.add(
            score_tile(
                [b.polygon for b in buildings],
                tile.gt_polygons,
                iou_threshold,
            )
        )

    return total


def run_sweep(
    tiles: Sequence[TileRecord],
    thresholds: Sequence[float],
    min_areas: Sequence[int],
    min_scores: Sequence[float],
    simplifies: Sequence[float],
    iou_threshold: float = 0.5,
) -> List[Dict[str, float]]:
    """Score every combination; rows are sorted by object-level F1."""

    grid = list(itertools.product(thresholds, min_areas, min_scores, simplifies))
    rows: List[Dict[str, float]] = []

    for n, (thr, area, score, simp) in enumerate(grid, start=1):

        config = PolygonConfig(
            threshold=thr,
            min_area_px=area,
            min_score=score,
            simplify_tolerance=simp,
        )

        metrics = evaluate_tiles(tiles, config, iou_threshold)

        rows.append(
            {
                "threshold": thr,
                "min_area_px": area,
                "min_score": score,
                "simplify": simp,
                **metrics.to_dict(),
            }
        )

        print(
            f"[{n}/{len(grid)}] thr={thr} area={area} score={score} "
            f"simp={simp} -> F1 {metrics.f1:.4f} "
            f"(P {metrics.precision:.4f}, R {metrics.recall:.4f})"
        )

    rows.sort(key=lambda r: r["f1"], reverse=True)

    return rows


def oracle_batches(
    dataset,
    batch_size: int,
) -> Iterator[Tuple[List[str], np.ndarray, Dict[str, np.ndarray]]]:
    """Use the GT mask as the prediction (needs no model)."""

    from ..inference.predictor import iter_split_batches

    for files, _, masks in iter_split_batches(dataset, batch_size):
        mask_array = masks.numpy()

        yield (
            files,
            mask_array > 0.5,
            {"oracle": mask_array.astype(np.float32)},
        )


def print_metrics(metrics: ObjectMetrics, title: str, iou_threshold: float) -> None:
    print(f"\n{title} (match at IoU >= {iou_threshold:.2f})")
    print(f"  GT buildings:            {metrics.n_gt}")
    print(f"  Predicted polygons:      {metrics.n_pred}")
    print(f"  True positives:          {metrics.tp}")
    print(f"  False positives:         {metrics.fp}")
    print(f"  Missed buildings:        {metrics.fn}")
    print(f"  Precision:               {metrics.precision:.4f}")
    print(f"  Recall:                  {metrics.recall:.4f}")
    print(f"  F1:                      {metrics.f1:.4f}")
    print(f"  Mean IoU (matched):      {metrics.mean_matched_iou:.4f}")
    print(
        f"  FP polygons on empty tiles: {metrics.fp_on_empty_tiles} "
        f"({metrics.n_empty_tiles} empty tiles of {metrics.n_tiles})"
    )


def main() -> None:

    from ..code.dataset import WHUBuildingDataset
    from ..inference.predictor import (
        add_predictor_args,
        iter_split_probabilities,
        predictor_from_args,
    )

    parser = argparse.ArgumentParser(
        description="Object-level polygon evaluation on a WHU split."
    )
    add_predictor_args(parser, extra_archs=("oracle",))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--gt-min-area",
        type=int,
        default=1,
        help="Drop GT components smaller than this many pixels (1 = keep all).",
    )

    defaults = PolygonConfig()
    parser.add_argument("--threshold", type=float, default=defaults.threshold)
    parser.add_argument("--min-area", type=int, default=defaults.min_area_px)
    parser.add_argument("--min-score", type=float, default=defaults.min_score)
    parser.add_argument("--simplify", type=float, default=defaults.simplify_tolerance)

    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Grid-search the post-processing settings below instead of "
        "scoring a single configuration. Do this on --split val.",
    )
    parser.add_argument(
        "--sweep-thresholds", type=float, nargs="+", default=[0.4, 0.5, 0.6]
    )
    parser.add_argument(
        "--sweep-min-areas", type=int, nargs="+", default=[10, 30, 60, 100]
    )
    parser.add_argument(
        "--sweep-min-scores", type=float, nargs="+", default=[0.0, 0.7]
    )
    parser.add_argument(
        "--sweep-simplify", type=float, nargs="+", default=[1.0]
    )
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    if args.sweep and args.split == "test":
        print(
            "[WARNING] Sweeping settings on the test split makes the reported "
            "numbers optimistic. Tune on --split val, then score test once."
        )

    dataset = WHUBuildingDataset(split=args.split, tile_size=512)

    if args.arch == "oracle":
        batches = oracle_batches(dataset, args.batch_size)
    else:
        predictor = predictor_from_args(args)
        batches = iter_split_probabilities(
            predictor, dataset, args.batch_size, desc=f"Predicting {args.split}"
        )

    tiles = collect_tiles(batches, args.arch, gt_min_area_px=args.gt_min_area)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.sweep:

        rows = run_sweep(
            tiles,
            args.sweep_thresholds,
            args.sweep_min_areas,
            args.sweep_min_scores,
            args.sweep_simplify,
            args.iou_threshold,
        )

        csv_path = os.path.join(
            args.output_dir, f"polygon_sweep_{args.arch}_{args.split}.csv"
        )

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        best = rows[0]

        print(f"\n[INFO] Saved sweep: {csv_path}")
        print(
            "[INFO] Best by object F1: "
            f"--threshold {best['threshold']} --min-area {best['min_area_px']} "
            f"--min-score {best['min_score']} --simplify {best['simplify']} "
            f"(F1 {best['f1']:.4f}, P {best['precision']:.4f}, "
            f"R {best['recall']:.4f})"
        )
        return

    config = PolygonConfig(
        threshold=args.threshold,
        min_area_px=args.min_area,
        min_score=args.min_score,
        simplify_tolerance=args.simplify,
    )

    metrics = evaluate_tiles(tiles, config, args.iou_threshold)

    print_metrics(
        metrics,
        f"Object-level metrics: {args.arch}, {args.split} split, {config}",
        args.iou_threshold,
    )

    json_path = os.path.join(
        args.output_dir, f"polygon_metrics_{args.arch}_{args.split}.json"
    )

    with open(json_path, "w") as f:
        json.dump(
            {
                "arch": args.arch,
                "split": args.split,
                "iou_threshold": args.iou_threshold,
                "gt_min_area_px": args.gt_min_area,
                "config": {
                    "threshold": config.threshold,
                    "min_area_px": config.min_area_px,
                    "min_score": config.min_score,
                    "simplify_tolerance": config.simplify_tolerance,
                },
                "metrics": metrics.to_dict(),
            },
            f,
            indent=2,
        )

    print(f"\n[INFO] Saved metrics: {json_path}")


if __name__ == "__main__":
    main()
