"""Select synchronized measured segments and record explicit observation endpoints."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from .build_real_traffic_cases import load_cases


MAX_STEP_S = {"INTERACTION": 0.2, "SinD": 0.2, "Waymo Open Motion": 0.21,
              "Songdo Traffic": 1.0, "NGSIM": 1.0}
MIN_DURATION_S = 3.0
MIN_DISPLACEMENT_M = 5.0


def passes_reference(samples):
    """Find whether the observed polyline approaches and leaves the local origin."""
    nearest = math.inf
    nearest_progress = 0.0
    length = 0.0
    for a, b in zip(samples, samples[1:]):
        ax, ay = a["x_m"], a["y_m"]
        dx, dy = b["x_m"] - ax, b["y_m"] - ay
        segment_length = math.hypot(dx, dy)
        if segment_length == 0:
            continue
        fraction = min(1.0, max(0.0, -(ax * dx + ay * dy) / segment_length ** 2))
        distance = math.hypot(ax + fraction * dx, ay + fraction * dy)
        if distance < nearest:
            nearest, nearest_progress = distance, length + fraction * segment_length
        length += segment_length
    first, last = samples[0], samples[-1]
    return (nearest <= 3.0 and nearest_progress >= 3.0 and length - nearest_progress >= 4.0
            and math.hypot(first["x_m"], first["y_m"]) >= 4.0
            and math.hypot(last["x_m"], last["y_m"]) >= 4.0)


def observed_window(case):
    routes = case["observed_routes"]
    samples_by_time = [{sample["t_s"]: sample for sample in route["samples"]} for route in routes]
    synchronized = sorted(samples_by_time[0].keys() & samples_by_time[1].keys())
    if not synchronized or synchronized[0] != 0:
        return None, "missing_common_start"
    for i, (a, b) in enumerate(zip(synchronized, synchronized[1:]), 1):
        if b - a > MAX_STEP_S[case["dataset"]] + 1e-8:
            synchronized = synchronized[:i]
            break
    end = synchronized[-1]
    if end < MIN_DURATION_S:
        return None, "short_common_track"

    origin = case["conflict_xy"]
    endpoints, polyline = [], []
    for route, time_index in zip(routes, samples_by_time, strict=True):
        points = [time_index[t] for t in synchronized]
        polyline.append(points)
        start, finish = points[0], points[-1]
        displacement = math.hypot(finish["x_m"] - start["x_m"], finish["y_m"] - start["y_m"])
        if not math.isfinite(displacement) or displacement < MIN_DISPLACEMENT_M:
            return None, "near_stationary_endpoints"
        anchors = {}
        for label, sample in (("start", start), ("end", finish)):
            anchors[label] = {
                **sample,
                "recording_xy_m": [origin[0] + sample["x_m"], origin[1] + sample["y_m"]],
                "recording_timestamp_ms": case["timestamp_ms"] + round(sample["t_s"] * 1000),
            }
        endpoints.append({"track_id": route["track_id"], **anchors,
                          "displacement_m": displacement,
                          "distance_traveled_m": sum(math.hypot(b["x_m"] - a["x_m"],
                              b["y_m"] - a["y_m"]) for a, b in zip(points, points[1:]))})
    priority = case["family"] != "following" and all(passes_reference(points) for points in polyline)
    return {"duration_s": end, "common_sample_count": len(synchronized), "vehicles": endpoints,
            "both_approached_and_left_reference": priority,
            "endpoint_kind": "measured_segment"}, None


def select(manifest):
    chosen, rejected = [], Counter()
    for row in manifest["cases"]:
        window, reason = observed_window(row)
        if window is None:
            rejected[reason] += 1
        else:
            chosen.append({**row, "observed_window": window})
    return chosen, rejected


def write_manifest(input_manifest, rows, folder, name):
    recording_names = {r["source_file"] for r in rows}
    map_ids = {r["map_id"] for r in rows if "map_id" in r}
    manifest = {**input_manifest, "cases": rows,
                "maps": {key: input_manifest["maps"][key] for key in sorted(map_ids)},
                "recordings_sha256": {key: input_manifest["recordings_sha256"][key]
                                      for key in sorted(recording_names)}}
    path = folder / name
    path.write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n",
                    encoding="utf-8")
    assert len(load_cases(path)[0]) == len(rows)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.input.read_text(encoding="utf-8"))
    cases, rejected = select(manifest)
    priority = [row for row in cases if row["observed_window"]["both_approached_and_left_reference"]]
    manifest["endpoint_filter"] = {
        "source_manifest": args.input.name,
        "minimum_joint_duration_s": MIN_DURATION_S,
        "minimum_vehicle_displacement_m": MIN_DISPLACEMENT_M,
        "maximum_step_s_by_dataset": MAX_STEP_S,
        "rejected_case_counts": dict(rejected),
    }
    path = write_manifest(manifest, cases, args.input.parent, f"observed_endpoints_{len(cases)}.json")
    priority_path = write_manifest(manifest, priority, args.input.parent,
                                   f"observed_crossings_{len(priority)}.json")
    print(path, Counter(r["dataset"] for r in cases))
    print(priority_path, Counter(r["dataset"] for r in priority))
    print("Rejected:", rejected)
