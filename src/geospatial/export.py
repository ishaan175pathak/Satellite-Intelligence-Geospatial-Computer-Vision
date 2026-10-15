"""GeoJSON export for extracted building polygons.

The WHU tiles in this project were preprocessed to plain tensors and carry no
georeferencing, so by default polygons are written in tile *pixel*
coordinates (x = column, y = row, origin top-left, y down). That is not valid
WGS84 GeoJSON, and the file says so in its ``coordinate_system`` member.

If you later recover a geotransform for a tile, pass it as ``geotransform``
(GDAL order: ``(x0, dx, rx, y0, ry, dy)``) and the polygons are written in
that coordinate system instead; pass ``crs`` (e.g. "EPSG:2193") to record it.

CLI (writes one FeatureCollection per tile):

    python -m src.geospatial.export --arch unet --split test \
        --tiles train_0896 train_2461 --output-dir results/geojson
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from shapely.affinity import affine_transform
from shapely.geometry import mapping

from .polygons import Building, PolygonConfig, extract_buildings

GeoTransform = Tuple[float, float, float, float, float, float]


def _round_coords(obj: Any, ndigits: int) -> Any:
    """Recursively round every number in a GeoJSON coordinates structure."""

    if isinstance(obj, (list, tuple)):
        return [_round_coords(item, ndigits) for item in obj]

    return round(float(obj), ndigits)


def buildings_to_feature_collection(
    buildings: Sequence[Building],
    *,
    tile: Optional[str] = None,
    image_size: Optional[Tuple[int, int]] = None,
    geotransform: Optional[GeoTransform] = None,
    crs: Optional[str] = None,
    extra_properties: Optional[Dict[str, Any]] = None,
    ndigits: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a GeoJSON FeatureCollection (as a dict) from buildings.

    Without ``geotransform`` the coordinates are tile pixel coordinates.
    """

    georeferenced = geotransform is not None

    if ndigits is None:
        ndigits = 7 if georeferenced else 2

    features: List[Dict[str, Any]] = []

    for index, building in enumerate(buildings):

        polygon = building.polygon

        if georeferenced:
            x0, dx, rx, y0, ry, dy = geotransform
            # GDAL: X = x0 + col*dx + row*rx ; Y = y0 + col*ry + row*dy
            polygon = affine_transform(polygon, [dx, rx, ry, dy, x0, y0])

        geometry = mapping(polygon)
        geometry = {
            "type": geometry["type"],
            "coordinates": _round_coords(geometry["coordinates"], ndigits),
        }

        properties: Dict[str, Any] = {
            "id": index,
            "area_px": building.area_px,
            "score": round(building.score, 4),
        }

        if extra_properties:
            properties.update(extra_properties)

        features.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": properties,
            }
        )

    collection: Dict[str, Any] = {
        "type": "FeatureCollection",
        "coordinate_system": "geographic" if georeferenced else "pixel",
        "features": features,
    }

    if tile is not None:
        collection["name"] = tile

    if not georeferenced:
        collection["y_axis"] = "down"
        if image_size is not None:
            collection["image_size"] = [int(image_size[0]), int(image_size[1])]

    if crs is not None:
        collection["crs"] = {
            "type": "name",
            "properties": {"name": crs},
        }

    return collection


def write_geojson(path: str, collection: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(collection, f)


def _find_tile(files: Sequence[str], name: str) -> int:
    """Accept 'train_0896', 'train_0896.png' or 'train_0896.png.pt'."""

    for candidate in (name, f"{name}.pt", f"{name}.png.pt"):
        if candidate in files:
            return list(files).index(candidate)

    raise FileNotFoundError(f"Tile not found in split: {name}")


def main() -> None:

    from ..code.dataset import WHUBuildingDataset
    from ..inference.predictor import add_predictor_args, predictor_from_args

    parser = argparse.ArgumentParser(
        description="Export predicted building polygons as GeoJSON."
    )
    add_predictor_args(parser)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--tiles",
        nargs="*",
        default=None,
        help="Tile names to export (e.g. train_0896). Default: the first "
        "--limit tiles of the split.",
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=30)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--simplify", type=float, default=1.0)
    parser.add_argument("--output-dir", default="results/geojson")
    args = parser.parse_args()

    config = PolygonConfig(
        threshold=args.threshold,
        min_area_px=args.min_area,
        min_score=args.min_score,
        simplify_tolerance=args.simplify,
    )

    predictor = predictor_from_args(args)
    dataset = WHUBuildingDataset(split=args.split, tile_size=512)

    if args.tiles:
        indices = [_find_tile(dataset.files, t) for t in args.tiles]
    else:
        indices = list(range(min(args.limit, len(dataset))))

    for idx in indices:

        image, _ = dataset[idx]
        probs = predictor.predict_each(image.unsqueeze(0))[args.arch][0].numpy()

        buildings = extract_buildings(probs, config)

        name = dataset.files[idx][: -len(".pt")]

        collection = buildings_to_feature_collection(
            buildings,
            tile=name,
            image_size=(probs.shape[1], probs.shape[0]),
            extra_properties={"model": args.arch},
        )

        out_path = os.path.join(args.output_dir, f"{name}_{args.arch}.geojson")
        write_geojson(out_path, collection)

        print(f"[INFO] {name}: {len(buildings)} polygons -> {out_path}")


if __name__ == "__main__":
    main()
