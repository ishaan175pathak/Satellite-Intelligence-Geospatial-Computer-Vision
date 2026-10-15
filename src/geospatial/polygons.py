"""Convert building masks / probability maps into vector polygons.

Coordinate convention
---------------------
Polygons are in image ("pixel-edge") coordinates: ``x`` is the column,
``y`` is the row, and the origin is the top-left *corner* of the top-left
pixel. Pixel (row r, col c) therefore covers [c, c+1] x [r, r+1], a
W x H tile spans [0, W] x [0, H], and ``y`` points down.

Because the polygons follow pixel edges, an unsimplified polygon has exactly
the pixel count of its component for axis-aligned shapes, and is within about
1-3% of it for diagonal or curved outlines (the half-pixel growth step below
slightly over-fills staircase edges). Ground truth and predictions go through
the same extraction, so polygon IoU is directly comparable to pixel IoU.

Pipeline (per tile)
-------------------
1. Threshold the probability map.
2. Label 4-connected components. Buildings that share an edge merge into one
   polygon (a limit of semantic masks); buildings that only touch at a corner
   stay separate.
3. Drop components smaller than ``min_area_px`` or with a mean probability
   below ``min_score`` (false-positive filtering).
4. Trace each component with ``cv2.findContours`` (``RETR_CCOMP`` so
   courtyards become holes) and build a Shapely polygon.
5. Optionally simplify with Douglas-Peucker (``simplify_tolerance`` pixels).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np
from shapely.geometry import LineString, Point, Polygon

# Shapely buffer styles (integers work on Shapely 1.8 and 2.x).
_JOIN_MITRE = 2
_CAP_SQUARE = 3


@dataclass(frozen=True)
class PolygonConfig:
    """Post-processing settings for turning probabilities into polygons.

    Defaults are reasonable starting points, not tuned values. Tune them on
    the validation split (``evaluate_polygons.py --sweep``) and report the
    test split once.
    """

    threshold: float = 0.5
    min_area_px: int = 30
    min_score: float = 0.0
    simplify_tolerance: float = 1.0


@dataclass(frozen=True)
class Building:
    """One extracted building footprint."""

    polygon: Polygon
    area_px: int  # pixel count of the component (before simplification)
    score: float  # mean model probability inside the component

    @property
    def area(self) -> float:
        """Polygon area in squared pixel units (after simplification)."""
        return float(self.polygon.area)


def _largest_polygon(geometry) -> Optional[Polygon]:
    """Return the largest Polygon in ``geometry`` (None if there is none)."""

    if geometry is None or geometry.is_empty:
        return None

    if isinstance(geometry, Polygon):
        return geometry

    parts = [
        g
        for g in getattr(geometry, "geoms", [])
        if isinstance(g, Polygon) and not g.is_empty
    ]

    return max(parts, key=lambda g: g.area) if parts else None


def _ring(contour: np.ndarray) -> np.ndarray:
    """Contour -> (N, 2) float array of pixel *centres* in edge coordinates."""

    return contour.reshape(-1, 2).astype(np.float64) + 0.5


def _thin_shape(points: np.ndarray) -> Optional[Polygon]:
    """Fallback for components that are one pixel wide (zero-area contour).

    Buffering the traced pixel centres with square caps gives exactly the
    pixel-edge outline of a 1-pixel-wide line or a single pixel.
    """

    if len(points) >= 2:
        shape = LineString(points).buffer(
            0.5, cap_style=_CAP_SQUARE, join_style=_JOIN_MITRE
        )
    else:
        shape = Point(points[0]).buffer(0.5, cap_style=_CAP_SQUARE)

    return _largest_polygon(shape)


def _component_polygon(contours, hierarchy) -> Optional[Polygon]:
    """Build one pixel-edge polygon (with holes) from a single component."""

    if hierarchy is None or len(contours) == 0:
        return None

    # hierarchy[0][i] = [next, previous, first_child, parent]
    info = hierarchy[0]
    outer = [i for i in range(len(contours)) if info[i][3] == -1]
    holes = [i for i in range(len(contours)) if info[i][3] != -1]

    if not outer:
        return None

    # A single connected component has one outer contour. Be defensive
    # anyway and keep the longest one.
    shell_idx = max(outer, key=lambda i: len(contours[i]))
    shell = _ring(contours[shell_idx])

    if len(shell) < 3:
        return _thin_shape(shell)

    hole_rings = [
        _ring(contours[i]) for i in holes if len(contours[i]) >= 3
    ]

    polygon: Optional[Polygon] = Polygon(shell, hole_rings)

    if not polygon.is_valid:
        polygon = _largest_polygon(polygon.buffer(0))

    if polygon is None or polygon.area == 0:
        return _thin_shape(shell)

    # Contours run through boundary pixel centres. Growing the polygon by
    # half a pixel moves it onto the pixel edges (and shrinks the holes by
    # the same amount), so area matches the pixel count.
    return _largest_polygon(
        polygon.buffer(0.5, join_style=_JOIN_MITRE)
    )


def mask_to_buildings(
    mask: np.ndarray,
    *,
    min_area_px: int = 1,
    simplify_tolerance: float = 0.0,
    probs: Optional[np.ndarray] = None,
    min_score: float = 0.0,
) -> List[Building]:
    """Convert a binary mask into building polygons.

    Args:
        mask: 2-D array; non-zero pixels are building.
        min_area_px: drop components with fewer pixels than this.
        simplify_tolerance: Douglas-Peucker tolerance in pixels
            (0 keeps the exact pixel-edge outline).
        probs: optional 2-D probability map used to score each component
            (mean probability inside it). Without it every score is 1.0.
        min_score: drop components whose mean probability is below this.

    Returns:
        A list of ``Building`` (empty for an empty mask). Coordinates follow
        the convention described in the module docstring.
    """

    mask_u8 = np.ascontiguousarray(np.asarray(mask) > 0, dtype=np.uint8)

    if mask_u8.ndim != 2:
        raise ValueError(
            f"mask must be 2-D [H, W], got shape {mask_u8.shape}"
        )

    score_map: Optional[np.ndarray] = None

    if probs is not None:
        score_map = np.asarray(probs, dtype=np.float32)

        if score_map.shape != mask_u8.shape:
            raise ValueError(
                f"probs shape {score_map.shape} does not match "
                f"mask shape {mask_u8.shape}"
            )

    if not mask_u8.any():
        return []

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=4
    )

    buildings: List[Building] = []

    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])

        if area < min_area_px:
            continue

        component = labels[y : y + h, x : x + w] == label

        score = 1.0
        if score_map is not None:
            score = float(score_map[y : y + h, x : x + w][component].mean())

            if score < min_score:
                continue

        contours, hierarchy = cv2.findContours(
            component.astype(np.uint8),
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE,
            offset=(x, y),
        )

        polygon = _component_polygon(contours, hierarchy)

        if polygon is None:
            continue

        if simplify_tolerance > 0:
            simplified = _largest_polygon(
                polygon.simplify(simplify_tolerance, preserve_topology=True)
            )
            if simplified is not None:
                polygon = simplified

        buildings.append(Building(polygon, area, score))

    return buildings


def extract_buildings(
    probs: np.ndarray,
    config: PolygonConfig = PolygonConfig(),
) -> List[Building]:
    """Probability map [H, W] -> building polygons using ``config``."""

    probs = np.asarray(probs, dtype=np.float32)

    return mask_to_buildings(
        probs > config.threshold,
        min_area_px=config.min_area_px,
        simplify_tolerance=config.simplify_tolerance,
        probs=probs,
        min_score=config.min_score,
    )


def polygons_of(buildings: Sequence[Building]) -> List[Polygon]:
    """Convenience: the Shapely polygons of a list of buildings."""

    return [b.polygon for b in buildings]
