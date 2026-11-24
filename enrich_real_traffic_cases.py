"""Attach measured route snippets and road-site labels to a simulation manifest."""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def road_type(site):
    if site.startswith("SinD_"):
        return "signalized_intersection"
    if site == "NGSIM_Emeryville_I80":
        return "highway"
    if site.startswith("NGSIM_"):
        return "urban_arterial"
    if "Roundabout" in site:
        return "roundabout"
    if "Merging" in site:
        return "merge_site"
    return "intersection"


def maneuver(points):
    moving = [p for p in points if math.hypot(p["vx_mps"], p["vy_mps"]) >= 1.5]
    if len(moving) < 4 or moving[-1]["t_s"] - moving[0]["t_s"] < 3:
        return {"label": "insufficient_motion", "heading_change_deg": None}
    headings = [math.atan2(p["vy_mps"], p["vx_mps"]) for p in moving]
    changes = [math.remainder(b - a, math.tau) for a, b in zip(headings, headings[1:])]
    angle = math.degrees(sum(changes))
    if abs(angle) <= 20 and sum(abs(change) for change in changes) <= math.radians(35):
        label = "straight"
    elif abs(angle) >= 45:
        label = "leftward" if angle > 0 else "rightward"
    else:
        label = "mild_or_mixed_curve"
    return {"label": label, "heading_change_deg": round(angle, 2)}


def enrich(manifest, roots, *, route_step_ms=1000):
    requests = defaultdict(list)
    for case in manifest["cases"]:
        requests[case["source_file"]].append(case)
    files = {f"{path.parent.name}/{path.name}": path
             for root in roots for path in root.rglob("vehicle_tracks_*.csv")}
    for recording, cases in requests.items():
        tracks = {r["track_id"]: {} for case in cases for r in case["raw_rows"]}
        with files[recording].open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                track = tracks.get(row["track_id"])
                if track is not None:
                    timestamp = int(row["timestamp_ms"])
                    track[timestamp] = row
        for case in cases:
            reference = case["conflict_xy"]
            timestamp = case["timestamp_ms"]
            route = []
            for raw in case["raw_rows"]:
                samples = []
                for offset in range(0, 6001, route_step_ms):
                    row = tracks[raw["track_id"]].get(timestamp + offset)
                    if row is None:
                        break
                    samples.append({
                        "t_s": offset / 1000, "x_m": round(float(row["x"]) - reference[0], 5),
                        "y_m": round(float(row["y"]) - reference[1], 5),
                        "vx_mps": float(row["vx"]), "vy_mps": float(row["vy"]),
                    })
                if not samples:
                    raise ValueError(f"Missing tracked initial frame: {case['case_id']}")
                route.append({"track_id": raw["track_id"], **maneuver(samples), "samples": samples})
            case["road_type"] = road_type(case["site"])
            case["observed_routes"] = route
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--tracks", nargs="+", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--route-step-ms", type=int, default=1000)
    args = parser.parse_args()
    if not 1 <= args.route_step_ms <= 1000:
        parser.error("--route-step-ms must be between 1 and 1000")
    manifest = enrich(json.loads(args.input.read_text(encoding="utf-8")), args.tracks,
                      route_step_ms=args.route_step_ms)
    args.out.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(Counter(case["road_type"] for case in manifest["cases"]))
    print(Counter(route["label"] for case in manifest["cases"] for route in case["observed_routes"]))


if __name__ == "__main__":
    main()
