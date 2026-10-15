"""Overlay ground-truth and predicted building polygons on test tiles.

Tiles are chosen by per-tile pixel IoU (best / median / worst / empty), the
same way as the existing prediction figures, so the figure shows the range of
behaviour rather than a random sample.

Columns: input | ground-truth polygons | predicted polygons | object matching
(green = matched building, red = false alarm, blue = missed building).

Usage (from the repository root):

    python -m src.geospatial.visualize_polygons --arch unet --split test
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from ..visualizationAndEvaluation.visualize_unet import (
    BLUE,
    GREEN,
    RED,
    select_tiles,
)
from .evaluate_polygons import match_polygons
from .polygons import PolygonConfig, extract_buildings, mask_to_buildings

GT_COLOR = "#f2c200"
PRED_COLOR = "#00b7eb"


def draw_polygon(
    ax,
    polygon: Polygon,
    *,
    edgecolor,
    facecolor=None,
    alpha: float = 0.35,
    linewidth: float = 1.2,
) -> None:
    """Draw a Shapely polygon (holes included) on a Matplotlib axis."""

    polygon = orient(polygon, sign=1.0)  # exterior CCW, holes CW

    vertices: List[np.ndarray] = []
    codes: List[int] = []

    for ring in [polygon.exterior, *polygon.interiors]:
        coords = np.asarray(ring.coords)
        vertices.extend(coords)
        codes.extend(
            [Path.MOVETO] + [Path.LINETO] * (len(coords) - 2) + [Path.CLOSEPOLY]
        )

    ax.add_patch(
        PathPatch(
            Path(vertices, codes),
            facecolor=(
                to_rgba(facecolor, alpha) if facecolor is not None else "none"
            ),
            edgecolor=to_rgba(edgecolor, 1.0),
            linewidth=linewidth,
        )
    )


def score_tiles(
    predictor,
    dataset,
    name: str,
    threshold: float,
    batch_size: int,
) -> List[Dict]:
    """Per-tile pixel counts / IoU, in the row format select_tiles expects."""

    from ..inference.predictor import iter_split_probabilities

    rows: List[Dict] = []
    index = 0

    for files, gt, probs in iter_split_probabilities(
        predictor, dataset, batch_size, desc="Scoring tiles"
    ):
        pred = probs[name] > threshold

        for i, file in enumerate(files):
            tp = int(np.count_nonzero(pred[i] & gt[i]))
            fp = int(np.count_nonzero(pred[i] & ~gt[i]))
            fn = int(np.count_nonzero(~pred[i] & gt[i]))

            gt_pixels = tp + fn

            rows.append(
                {
                    "index": index,
                    "file": file,
                    "gt_pixels": gt_pixels,
                    "pred_pixels": tp + fp,
                    "iou": (tp / (tp + fp + fn)) if gt_pixels > 0 else None,
                }
            )
            index += 1

    return rows


def main() -> None:

    from ..code.dataset import WHUBuildingDataset
    from ..inference.predictor import (
        add_predictor_args,
        denormalize,
        predictor_from_args,
    )

    parser = argparse.ArgumentParser(
        description="Visualize GT vs predicted building polygons."
    )
    add_predictor_args(parser)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--iou-threshold", type=float, default=0.5)

    defaults = PolygonConfig()
    parser.add_argument("--threshold", type=float, default=defaults.threshold)
    parser.add_argument("--min-area", type=int, default=defaults.min_area_px)
    parser.add_argument("--min-score", type=float, default=defaults.min_score)
    parser.add_argument("--simplify", type=float, default=defaults.simplify_tolerance)

    parser.add_argument("--n-best", type=int, default=2)
    parser.add_argument("--n-median", type=int, default=2)
    parser.add_argument("--n-worst", type=int, default=2)
    parser.add_argument("--n-empty", type=int, default=1)
    parser.add_argument("--output-dir", default="results/visualizations")
    args = parser.parse_args()

    config = PolygonConfig(
        threshold=args.threshold,
        min_area_px=args.min_area,
        min_score=args.min_score,
        simplify_tolerance=args.simplify,
    )

    predictor = predictor_from_args(args)
    dataset = WHUBuildingDataset(split=args.split, tile_size=512)

    rows = score_tiles(
        predictor, dataset, args.arch, config.threshold, args.batch_size
    )

    picks = select_tiles(
        rows, args.n_best, args.n_median, args.n_worst, args.n_empty
    )

    if not picks:
        raise RuntimeError("No tiles selected for visualization.")

    fig, axes = plt.subplots(
        len(picks), 4, figsize=(16, len(picks) * 4), squeeze=False
    )

    for row_idx, (label, r) in enumerate(picks):

        image, mask = dataset[r["index"]]
        display = denormalize(image)
        height, width = display.shape[:2]
        extent = (0, width, height, 0)  # pixel-edge coordinates, y down

        probs = predictor.predict_each(image.unsqueeze(0))[args.arch][0].numpy()

        gt_polys = [
            b.polygon for b in mask_to_buildings(mask.numpy() > 0.5)
        ]
        pred_polys = [b.polygon for b in extract_buildings(probs, config)]

        result = match_polygons(pred_polys, gt_polys, args.iou_threshold)

        counts = (
            f"TP {len(result.matches)} | FP {len(result.unmatched_pred)} | "
            f"FN {len(result.unmatched_gt)}"
        )

        row_axes = axes[row_idx]

        for ax in row_axes[:3]:
            ax.imshow(display, extent=extent)

        row_axes[3].imshow(display * 0.35, extent=extent)

        for polygon in gt_polys:
            draw_polygon(row_axes[1], polygon, edgecolor=GT_COLOR, facecolor=GT_COLOR)

        for polygon in pred_polys:
            draw_polygon(row_axes[2], polygon, edgecolor=PRED_COLOR, facecolor=PRED_COLOR)

        for pi, _, _ in result.matches:
            draw_polygon(row_axes[3], pred_polys[pi], edgecolor=GREEN, facecolor=GREEN, alpha=0.6)

        for pi in result.unmatched_pred:
            draw_polygon(row_axes[3], pred_polys[pi], edgecolor=RED, facecolor=RED, alpha=0.6)

        for gi in result.unmatched_gt:
            draw_polygon(row_axes[3], gt_polys[gi], edgecolor=BLUE, facecolor=BLUE, alpha=0.6)

        score = (
            f"IoU {r['iou']:.3f}" if r["iou"] is not None else "no GT buildings"
        )

        row_axes[0].set_title(f"{label}: {r['file'][:-3]} ({score})", fontsize=9)
        row_axes[1].set_title(f"Ground truth ({len(gt_polys)} buildings)", fontsize=9)
        row_axes[2].set_title(f"Predicted ({len(pred_polys)} polygons)", fontsize=9)
        row_axes[3].set_title(f"Object matching: {counts}", fontsize=9)

        for ax in row_axes:
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)
            ax.axis("off")

        print(f"{label:<10} {r['file'][:-3]} | {score} | {counts}")

    fig.legend(
        handles=[
            Patch(color=GREEN, label=f"Matched building (IoU >= {args.iou_threshold:.2f})"),
            Patch(color=RED, label="False alarm"),
            Patch(color=BLUE, label="Missed building"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
    )

    plt.tight_layout(rect=(0, 0.03, 1, 1))

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir, f"whu_{args.split}_polygons_{args.arch}.png"
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved visualization: {output_path}")


if __name__ == "__main__":
    main()
