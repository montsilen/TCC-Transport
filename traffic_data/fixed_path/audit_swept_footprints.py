"""Audit fixed-path swept footprints using segment-aligned rectangles."""

import argparse
from collections import Counter
import csv
import json
from math import isfinite
from pathlib import Path

import numpy as np
import shapely


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / "TCC-Driving/Result/real_traffic_expanded"
OUTPUT = ROOT / "Result/fixed_path_audit"
NOMINAL = (4.5, 1.8)
SMALL = (4.2, 1.7)
LARGE = (5.0, 2.0)


def swept_area(points, length, width):
    """Translate a rectangle along each polyline segment and union the resulting strips."""
    return shapely.union_all(swept_strips(points, length, width))


def swept_strips(points, length, width):
    displacement = np.diff(points, axis=0)
    segment_length = np.linalg.norm(displacement, axis=1)
    moving = segment_length > 0
    direction = displacement[moving] / segment_length[moving, None]
    lateral = np.stack((-direction[:, 1], direction[:, 0]), axis=1) * (width / 2)
    rear = points[:-1][moving] - direction * (length / 2)
    front = points[1:][moving] + direction * (length / 2)
    corners = np.stack((rear - lateral, front - lateral, front + lateral, rear + lateral), axis=1)
    return shapely.polygons(corners)


def dimensions(row):
    try:
        length, width = float(row["length"]), float(row["width"])
    except (KeyError, TypeError, ValueError):
        return None
    return (length, width) if all(isfinite(value) and value > 0 for value in (length, width)) else None


def audit_case(case, ref_ids):
    window = case["observed_window"]
    samples = [
        {point["t_s"]: (point["x_m"], point["y_m"]) for point in route["samples"]}
        for route in case["observed_routes"]
    ]
    times = sorted(t for t in samples[0].keys() & samples[1].keys() if t <= window["duration_s"])
    assert len(times) == window["common_sample_count"]
    paths = [np.array([samples_by_time[t] for t in times]) for samples_by_time in samples]
    body_by_track = {str(row["track_id"]): dimensions(row) for row in case["raw_rows"]}
    body_sizes = [body_by_track[str(route["track_id"])] for route in case["observed_routes"]]
    sizing_options = (NOMINAL, SMALL, LARGE) if None in body_sizes else (NOMINAL,)
    overlaps = []
    for default in sizing_options:
        first, second = [swept_area(path, *(size or default)) for path, size in zip(paths, body_sizes)]
        overlaps.append(shapely.intersection(first, second).area if shapely.intersects(first, second) else None)
    nominal, small, large = overlaps if len(overlaps) == 3 else overlaps * 3
    areas = [0.0 if area is None else area for area in (small, nominal, large)]
    assert areas[0] <= areas[1] + 1e-8 and areas[1] <= areas[2] + 1e-8
    case_id = case["case_id"]
    center_crosses = bool(shapely.crosses(*(shapely.LineString(path) for path in paths)))
    assert not center_crosses or nominal is not None and nominal > 0, case_id
    result = {key: case[key] for key in ("case_id", "dataset", "site", "road_type", "family")}
    result.update(
        in_696_reference_subset=case_id in ref_ids,
        center_path_intersects=center_crosses,
        swept_areas_intersect=nominal is not None,
        overlap_area_m2=round(areas[1], 8),
        overlap_with_small_fallback=small is not None,
        overlap_with_large_fallback=large is not None,
        has_assumed_dimensions=None in body_sizes,
    )
    for index, size in enumerate(body_sizes, 1):
        result[f"vehicle_{index}_length_m"], result[f"vehicle_{index}_width_m"] = size or NOMINAL
        result[f"vehicle_{index}_dimensions_measured"] = size is not None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DATA / "observed_endpoints_5278.json")
    parser.add_argument("--shortlist", type=Path, default=DATA / "observed_crossings_696.json")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_bytes())["cases"]
    priority = json.loads(args.shortlist.read_bytes())["cases"]
    ref_ids = {row["case_id"] for row in priority}
    assert manifest and len({case["case_id"] for case in manifest}) == len(manifest)
    results = [audit_case(case, ref_ids) for case in manifest]

    with (args.output / "swept_footprint_intersections.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    overlap = [row for row in results if row["swept_areas_intersect"]]
    extra = [row for row in overlap if not row["center_path_intersects"]]
    print("All cases", len(results), "Measured dimensions", sum(not r["has_assumed_dimensions"] for r in results))
    print("Overlap", len(overlap), "Sweep only", len(extra),
          "Boundary only", sum(row["overlap_area_m2"] == 0 for row in overlap))
    print("Size sensitivity",
          sum(row["overlap_with_small_fallback"] for row in results),
          sum(row["overlap_with_large_fallback"] for row in results))
    print("Datasets", Counter(row["dataset"] for row in overlap))
    print("Nonfollowing", sum(row["family"] != "following" for row in overlap))
    for road_type in sorted({row["road_type"] for row in results}):
        print(road_type,
              sum(row["road_type"] == road_type for row in results),
              sum(row["road_type"] == road_type for row in overlap),
              sum(row["road_type"] == road_type for row in extra),
              sum(row["road_type"] == road_type and row["family"] != "following" for row in overlap),
              sum(row["road_type"] == road_type and row["in_696_reference_subset"] for row in overlap),
              sum(row["road_type"] == road_type and row["in_696_reference_subset"] for row in extra))


if __name__ == "__main__":
    main()
