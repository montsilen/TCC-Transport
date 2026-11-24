"""Select prepared paths whose swept bodies can come within the clearance threshold."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
import shapely

from .audit_swept_footprints import DATA, LARGE, NOMINAL, OUTPUT, swept_area


CLEARANCE_M = 1.0


def screen(case, vehicle_audit):
    window = case["observed_window"]
    samples = [
        {point["t_s"]: (point["x_m"], point["y_m"]) for point in route["samples"]}
        for route in case["observed_routes"]
    ]
    times = sorted(t for t in samples[0].keys() & samples[1].keys() if t <= window["duration_s"])
    assert len(times) == window["common_sample_count"] and times[-1] == window["duration_s"]
    paths = [np.array([points[t] for t in times]) for points in samples]
    areas = []
    larger_areas = []
    for index, path in enumerate(paths, 1):
        prefix = f"vehicle_{index}"
        size = (float(vehicle_audit[f"{prefix}_length_m"]), float(vehicle_audit[f"{prefix}_width_m"]))
        area = swept_area(path, *size)
        areas.append(area)
        larger_areas.append(area if vehicle_audit[f"{prefix}_dimensions_measured"] == "True"
                            else swept_area(path, *LARGE))
    clearance = float(shapely.distance(*areas))
    larger_clearance = float(shapely.distance(*larger_areas))
    assert np.isfinite([clearance, larger_clearance]).all() and 0 <= larger_clearance <= clearance + 1e-8
    assert (clearance == 0) == (vehicle_audit["swept_areas_intersect"] == "True")

    if clearance == 0:
        category = "swept_overlap"
    elif clearance <= CLEARANCE_M:
        category = "within_1m"
    elif larger_clearance <= CLEARANCE_M:
        category = "within_1m_with_larger_assumed_body"
    else:
        category = "excluded"

    return {
        "category": category,
        "clearance_m": clearance,
        "clearance_with_larger_assumed_body_m": larger_clearance,
        "has_assumed_dimensions": vehicle_audit["has_assumed_dimensions"] == "True",
        "in_previous_696_shortlist": vehicle_audit["in_696_reference_subset"] == "True",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DATA / "observed_endpoints_5278.json")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    original = args.manifest
    manifest = json.loads(original.read_bytes())
    with (args.output / "swept_footprint_intersections.csv").open(newline="", encoding="utf-8-sig") as stream:
        audit = {row["case_id"]: row for row in csv.DictReader(stream)}
    assert len(manifest["cases"]) == len(audit)
    selected = []
    index = []
    for case in manifest["cases"]:
        status = screen(case, audit[case["case_id"]])
        index.append({**{key: case[key] for key in ("case_id", "dataset", "site", "road_type", "family")}, **status})
        if status["category"] != "excluded":
            selected.append({**case, "near_miss_screen": status})

    categories = Counter(row["category"] for row in index)
    map_ids = {case["map_id"] for case in selected if "map_id" in case}
    source_names = {case["source_file"] for case in selected}
    result = {
        **manifest,
        "maps": {key: manifest["maps"][key] for key in sorted(map_ids)},
        "recordings_sha256": {key: manifest["recordings_sha256"][key] for key in sorted(source_names)},
        "near_miss_screen": {
            "source_manifest": original.name,
            "screened_case_count": len(index),
            "clearance_threshold_m": CLEARANCE_M,
            "nominal_assumed_dimensions_m": dict(zip(("length", "width"), NOMINAL)),
            "larger_assumed_dimensions_m": dict(zip(("length", "width"), LARGE)),
            "method": "Minimum distance between segment-aligned rectangular sweeps over the common observation window.",
            "category_counts": dict(categories),
        },
        "cases": selected,
    }
    destination = args.output / "near_miss_candidates.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n",
                           encoding="utf-8")
    with (args.output / "near_miss_screening.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(index[0]))
        writer.writeheader()
        writer.writerows(index)
    print("Categories", categories)
    print("Selected", len(selected))
    print("Non-following", sum(row["family"] != "following" for row in selected))
    print("Previously shortlisted at labeled intersections",
          Counter(row["road_type"] for row in index if row["in_previous_696_shortlist"]
                  and row["category"] != "excluded" and row["road_type"] in ("intersection", "signalized_intersection")))
    print("By dataset", Counter(row["dataset"] for row in selected))


if __name__ == "__main__":
    main()
