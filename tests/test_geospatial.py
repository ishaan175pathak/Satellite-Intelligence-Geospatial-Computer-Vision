"""Tests for polygon extraction, object-level matching and GeoJSON export.

These need only numpy, OpenCV and Shapely (no model, no data, no GPU):

    python -m pytest tests/test_geospatial.py
"""

import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon, box

from src.geospatial.evaluate_polygons import (
    ObjectMetrics,
    match_polygons,
    score_tile,
)
from src.geospatial.export import buildings_to_feature_collection
from src.geospatial.polygons import (
    PolygonConfig,
    extract_buildings,
    mask_to_buildings,
)

SIZE = 512


def blank() -> np.ndarray:
    return np.zeros((SIZE, SIZE), dtype=np.uint8)


# ---------------------------------------------------------------- polygons


def test_empty_mask_gives_no_polygons():
    assert mask_to_buildings(blank()) == []
    assert extract_buildings(np.zeros((SIZE, SIZE), dtype=np.float32)) == []


def test_rectangle_is_exact_pixel_edge_polygon():
    mask = blank()
    mask[10:50, 20:80] = 1  # 40 rows x 60 cols

    (building,) = mask_to_buildings(mask)

    assert building.polygon.area == pytest.approx(2400)
    assert building.polygon.bounds == pytest.approx((20, 10, 80, 50))
    assert building.area_px == 2400
    assert building.score == 1.0


def test_courtyard_becomes_a_hole():
    mask = blank()
    mask[100:150, 100:150] = 1
    mask[120:130, 120:130] = 0  # 10 x 10 courtyard

    (building,) = mask_to_buildings(mask)

    assert len(building.polygon.interiors) == 1
    # Hole corners are clipped by a fraction of a pixel (<0.1% of the area).
    assert building.polygon.area == pytest.approx(2500 - 100, rel=0.005)
    assert building.area_px == 2400


def test_single_pixel_and_one_pixel_wide_line():
    mask = blank()
    mask[5, 5] = 1
    mask[100, 200:220] = 1  # 1 x 20 line

    buildings = sorted(mask_to_buildings(mask), key=lambda b: b.area_px)

    assert [b.area_px for b in buildings] == [1, 20]
    assert buildings[0].polygon.area == pytest.approx(1)
    assert buildings[1].polygon.area == pytest.approx(20)


def test_polygon_area_matches_pixel_count_for_irregular_shape():
    mask = blank()
    cv2.circle(mask, (256, 256), 40, 1, thickness=-1)

    (building,) = mask_to_buildings(mask)

    assert building.polygon.area == pytest.approx(building.area_px, rel=0.05)


def test_separate_buildings_are_separate_polygons():
    mask = blank()
    mask[10:30, 10:30] = 1
    mask[10:30, 40:60] = 1  # 10 px gap

    assert len(mask_to_buildings(mask)) == 2


def test_touching_buildings_merge_into_one_polygon():
    # Documented limitation of semantic masks.
    mask = blank()
    mask[10:30, 10:30] = 1
    mask[10:30, 30:50] = 1  # shares an edge

    assert len(mask_to_buildings(mask)) == 1


def test_corner_touching_buildings_stay_separate():
    mask = blank()
    mask[10:30, 10:30] = 1
    mask[30:50, 30:50] = 1  # touches the first only at a corner

    buildings = mask_to_buildings(mask)

    assert len(buildings) == 2
    assert all(b.polygon.is_valid for b in buildings)
    assert sorted(b.polygon.area for b in buildings) == pytest.approx([400, 400])


def test_min_area_filters_small_components():
    mask = blank()
    mask[10:12, 10:12] = 1  # 4 px
    mask[100:120, 100:120] = 1  # 400 px

    assert len(mask_to_buildings(mask, min_area_px=1)) == 2

    kept = mask_to_buildings(mask, min_area_px=30)
    assert [b.area_px for b in kept] == [400]


def test_min_score_filters_low_confidence_components():
    probs = np.zeros((SIZE, SIZE), dtype=np.float32)
    probs[10:40, 10:40] = 0.95
    probs[100:130, 100:130] = 0.55  # above threshold but unsure

    config = PolygonConfig(threshold=0.5, min_area_px=1, min_score=0.0)
    assert len(extract_buildings(probs, config)) == 2

    config = PolygonConfig(threshold=0.5, min_area_px=1, min_score=0.7)
    kept = extract_buildings(probs, config)
    assert len(kept) == 1
    assert kept[0].score == pytest.approx(0.95, abs=1e-4)


def test_threshold_is_strictly_greater_than():
    probs = np.full((SIZE, SIZE), 0.5, dtype=np.float32)

    assert extract_buildings(probs, PolygonConfig(threshold=0.5)) == []


def test_simplification_reduces_vertices_and_keeps_shape():
    mask = blank()
    cv2.circle(mask, (256, 256), 60, 1, thickness=-1)

    (exact,) = mask_to_buildings(mask, simplify_tolerance=0.0)
    (simple,) = mask_to_buildings(mask, simplify_tolerance=2.0)

    assert len(simple.polygon.exterior.coords) < len(exact.polygon.exterior.coords)

    inter = exact.polygon.intersection(simple.polygon).area
    union = exact.polygon.union(simple.polygon).area
    assert inter / union > 0.95
    assert simple.polygon.is_valid


def test_full_tile_mask_covers_the_tile():
    mask = np.ones((SIZE, SIZE), dtype=np.uint8)

    (building,) = mask_to_buildings(mask)

    assert building.polygon.area == pytest.approx(SIZE * SIZE)
    assert building.polygon.bounds == pytest.approx((0, 0, SIZE, SIZE))


def test_shape_mismatch_and_bad_ndim_raise():
    with pytest.raises(ValueError):
        mask_to_buildings(np.zeros((4, 4, 1)))

    with pytest.raises(ValueError):
        mask_to_buildings(np.ones((4, 4)), probs=np.ones((5, 5)))


# ---------------------------------------------------------------- matching


def test_identical_polygons_match_perfectly():
    gts = [box(0, 0, 10, 10), box(50, 50, 70, 70)]

    result = match_polygons(gts, gts)

    assert len(result.matches) == 2
    assert result.unmatched_pred == [] and result.unmatched_gt == []
    assert all(iou == pytest.approx(1.0) for _, _, iou in result.matches)


def test_iou_threshold_decides_the_match():
    gt = [box(0, 0, 10, 10)]
    shifted = [box(5, 0, 15, 10)]  # IoU = 50 / 150 = 1/3

    assert len(match_polygons(shifted, gt, 0.3).matches) == 1
    assert len(match_polygons(shifted, gt, 0.5).matches) == 0


def test_false_positive_and_missed_building():
    gts = [box(0, 0, 10, 10), box(100, 100, 120, 120)]
    preds = [box(0, 0, 10, 10), box(300, 300, 320, 320)]

    result = match_polygons(preds, gts)

    assert len(result.matches) == 1
    assert result.unmatched_pred == [1]
    assert result.unmatched_gt == [1]


def test_matching_is_one_to_one():
    gt = [box(0, 0, 10, 10)]
    preds = [box(0, 0, 10, 10), box(0, 0, 10, 10)]  # duplicate prediction

    result = match_polygons(preds, gt)

    assert len(result.matches) == 1
    assert len(result.unmatched_pred) == 1


def test_matching_with_empty_sides():
    poly = [box(0, 0, 5, 5)]

    assert match_polygons([], poly).unmatched_gt == [0]
    assert match_polygons(poly, []).unmatched_pred == [0]
    assert match_polygons([], []).matches == []


def test_touching_but_not_overlapping_polygons_do_not_match():
    result = match_polygons([box(0, 0, 10, 10)], [box(10, 0, 20, 10)], 0.0)

    assert result.matches == []


def test_metrics_arithmetic_and_empty_tile_accounting():
    total = ObjectMetrics()

    # Tile 1: 2 GT buildings, 1 found, 1 false alarm.
    total.add(
        score_tile(
            [box(0, 0, 10, 10), box(200, 200, 210, 210)],
            [box(0, 0, 10, 10), box(100, 100, 120, 120)],
        )
    )
    # Tile 2: no GT buildings, 3 false alarms.
    total.add(
        score_tile(
            [box(0, 0, 4, 4), box(10, 10, 14, 14), box(20, 20, 24, 24)], []
        )
    )

    assert (total.tp, total.fp, total.fn) == (1, 4, 1)
    assert total.precision == pytest.approx(1 / 5)
    assert total.recall == pytest.approx(1 / 2)
    assert total.f1 == pytest.approx(2 * 0.2 * 0.5 / 0.7)
    assert total.mean_matched_iou == pytest.approx(1.0)
    assert total.n_empty_tiles == 1
    assert total.fp_on_empty_tiles == 3


def test_metrics_with_no_predictions_or_no_gt_do_not_divide_by_zero():
    metrics = ObjectMetrics()

    assert metrics.precision == metrics.recall == metrics.f1 == 0.0
    assert metrics.mean_matched_iou == 0.0


def test_oracle_round_trip_recovers_every_building():
    """GT mask -> polygons -> match against itself: perfect, even with
    simplification (a rectangle stays a rectangle)."""

    mask = blank()
    mask[10:50, 10:50] = 1
    mask[100:160, 200:230] = 1
    cv2.circle(mask, (400, 400), 30, 1, thickness=-1)

    gt = [b.polygon for b in mask_to_buildings(mask)]
    pred = [
        b.polygon
        for b in extract_buildings(
            mask.astype(np.float32), PolygonConfig(simplify_tolerance=1.0)
        )
    ]

    metrics = score_tile(pred, gt)

    assert (metrics.tp, metrics.fp, metrics.fn) == (3, 0, 0)
    assert metrics.mean_matched_iou > 0.95


# ------------------------------------------------------------------ export


def test_pixel_coordinate_geojson_structure():
    mask = blank()
    mask[10:20, 30:50] = 1
    buildings = mask_to_buildings(mask)

    fc = buildings_to_feature_collection(
        buildings, tile="train_0001", image_size=(SIZE, SIZE)
    )

    assert fc["type"] == "FeatureCollection"
    assert fc["coordinate_system"] == "pixel"
    assert fc["y_axis"] == "down"
    assert fc["image_size"] == [SIZE, SIZE]
    assert len(fc["features"]) == 1

    feature = fc["features"][0]
    assert feature["geometry"]["type"] == "Polygon"
    assert feature["properties"]["area_px"] == 200
    assert Polygon(feature["geometry"]["coordinates"][0]).area == pytest.approx(200)


def test_geotransform_maps_pixels_to_map_coordinates():
    mask = blank()
    mask[0:10, 0:10] = 1  # 10 x 10 pixel building at the top-left corner
    buildings = mask_to_buildings(mask)

    # origin (1000, 2000), 0.3 m pixels, north-up (negative dy)
    transform = (1000.0, 0.3, 0.0, 2000.0, 0.0, -0.3)

    fc = buildings_to_feature_collection(
        buildings, geotransform=transform, crs="EPSG:2193"
    )

    ring = Polygon(fc["features"][0]["geometry"]["coordinates"][0])

    assert fc["coordinate_system"] == "geographic"
    assert fc["crs"]["properties"]["name"] == "EPSG:2193"
    assert ring.bounds == pytest.approx((1000.0, 1997.0, 1003.0, 2000.0))
    assert ring.area == pytest.approx(9.0)
