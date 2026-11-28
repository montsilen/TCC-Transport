"""Find co-observed car pairs with at most 1 m between their swept bodies."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
import shapely

from .audit_swept_footprints import DATA, LARGE, NOMINAL, OUTPUT, ROOT, dimensions, swept_strips


SOURCE = ROOT.parent / "TCC-Driving/Result/real_traffic_sources"
MIN_DURATION_MS = 3000
MIN_DISPLACEMENT_M = 5.0
MAX_CLEARANCE_M = 1.0


@dataclass
class Track:
    identifier: str
    times: np.ndarray
    xy: np.ndarray
    size: tuple | None
    clock_ms: np.ndarray | None = None
    velocity_xy: np.ndarray | None = None
    heading_rad: np.ndarray | None = None


def recordings(dataset, source, tc_root):
    if dataset == "INTERACTION":
        for path in sorted((source / "DR").rglob("vehicle_tracks_*.csv")):
            yield path, path.parent.name, f"{path.parent.name}/{path.name}"
    elif dataset == "INTERACTION-TC":
        for path in sorted(tc_root.rglob("vehicle_tracks_*.csv")):
            yield path, path.parent.name, f"{path.parent.name}/{path.name}"
    elif dataset == "SinD":
        for path in sorted((source / "SinD").rglob("vehicle_tracks_*.csv")):
            yield path, path.parent.name, f"{path.parent.name}/{path.name}"
    else:
        for path in sorted((source / "Songdo_expansion/tracks").glob("*.csv")):
            date, site, session = path.stem.split("_")
            site = f"Songdo_{site}"
            yield path, site, f"{site}/vehicle_tracks_{date}_{session}.csv"


def milliseconds(value):
    hour, minute, second = value.split(":")
    return (int(hour) * 3600 + int(minute) * 60) * 1000 + round(float(second) * 1000)


def road_type(site):
    if site.startswith("SinD"):
        return "signalized_intersection"
    if "Roundabout" in site:
        return "roundabout"
    if "Merging" in site:
        return "merge_site"
    return "intersection"


def optional_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def read_tracks(path, dataset, *, include_kinematics=False):
    by_car = defaultdict(dict)
    songdo = dataset == "Songdo"
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if songdo:
                if row["Vehicle_Class"] != "0" or row["Visibility"] != "1":
                    continue
                identifier = row["Vehicle_ID"]
                time = milliseconds(row["Local_Time"])
                x, y = float(row["Local_X"]), float(row["Local_Y"])
                size = dimensions({"length": row["Vehicle_Length"], "width": row["Vehicle_Width"]})
            else:
                if row["agent_type"].lower() != "car":
                    continue
                identifier = row["track_id"]
                time = int(row["frame_id"]) * 100 if dataset == "SinD" else int(row["timestamp_ms"])
                x, y = float(row["x"]), float(row["y"])
                size = dimensions(row)
            if np.isfinite(x) and np.isfinite(y):
                row_data = (x, y, size)
                if include_kinematics:
                    clock = optional_number(row["timestamp_ms"]) if dataset == "SinD" else time
                    row_data += (clock, optional_number(row.get("vx")), optional_number(row.get("vy")),
                                 optional_number(row.get("psi_rad", row.get("yaw_rad"))))
                by_car[identifier][time] = row_data

    for identifier, by_time in by_car.items():
        times = np.array(sorted(by_time))
        points = [by_time[time] for time in times]
        xy = np.array([point[:2] for point in points])
        kinematics = np.array([point[3:] for point in points]) if include_kinematics else None
        sizes = [point[2] for point in points if point[2] is not None]
        size = tuple(np.median(sizes, axis=0)) if sizes else None
        for run in continuous_runs(times, 150 if songdo else 100):
            t, positions = times[run], xy[run]
            if t[-1] - t[0] < MIN_DURATION_MS:
                continue
            if songdo:
                target = np.arange((t[0] + 99) // 100 * 100, t[-1] + 1, 100)
                positions = np.column_stack([np.interp(target, t, axis) for axis in positions.T])
                t = target
            if t[-1] - t[0] >= MIN_DURATION_MS and np.linalg.norm(np.ptp(positions, axis=0)) >= MIN_DISPLACEMENT_M:
                if kinematics is not None and not songdo:
                    clock, vx, vy, heading = kinematics[run].T
                    yield Track(identifier, t, positions, size, clock, np.column_stack((vx, vy)), heading)
                else:
                    yield Track(identifier, t, positions, size)


def continuous_runs(times, max_gap=100):
    cuts = np.r_[0, np.flatnonzero(np.diff(times) > max_gap) + 1, len(times)]
    return (slice(a, b) for a, b in zip(cuts[:-1], cuts[1:]))


def pair_windows(a, b):
    times, a_idx, b_idx = np.intersect1d(a.times, b.times, assume_unique=True, return_indices=True)
    for run in continuous_runs(times):
        t = times[run]
        if not len(t) or t[-1] - t[0] < MIN_DURATION_MS:
            continue
        paths = a.xy[a_idx[run]], b.xy[b_idx[run]]
        if min(np.linalg.norm(path[-1] - path[0]) for path in paths) >= MIN_DISPLACEMENT_M:
            yield int(t[0]), int(t[-1]), paths


def nearby_clearance(strips, other):
    if len(strips) > len(other):
        strips, other = other, strips
    tree = shapely.STRtree(other)
    _, distances = tree.query_nearest(strips, max_distance=MAX_CLEARANCE_M,
                                      all_matches=False, return_distance=True)
    return float(distances.min(initial=float("inf")))


def scan(path, dataset):
    return scan_tracks(list(read_tracks(path, dataset)))


def scan_tracks(tracks):
    if not tracks:
        return {}, 0
    starts = np.array([track.times[0] for track in tracks])
    ends = np.array([track.times[-1] for track in tracks])
    radii = np.array([np.linalg.norm(track.size or LARGE) / 2 for track in tracks]).reshape(-1, 1)
    minima = np.array([track.xy.min(axis=0) for track in tracks]) - radii
    maxima = np.array([track.xy.max(axis=0) for track in tracks]) + radii
    bounding = shapely.box(minima[:, 0], minima[:, 1], maxima[:, 0], maxima[:, 1])
    tree = shapely.STRtree(bounding)
    pairs = {}
    for i, a in enumerate(tracks):
        candidates = tree.query(shapely.box(*(minima[i] - MAX_CLEARANCE_M), *(maxima[i] + MAX_CLEARANCE_M)))
        co_observed = np.minimum(ends[i], ends[candidates]) - np.maximum(starts[i], starts[candidates])
        for j in candidates[(candidates > i) & (co_observed >= MIN_DURATION_MS)]:
            b = tracks[j]
            if a.identifier == b.identifier:
                continue
            for start, end, paths in pair_windows(a, b):
                bigger = [swept_strips(points, *(track.size or LARGE)) for track, points in zip((a, b), paths)]
                clearance = nearby_clearance(*bigger)
                if clearance > MAX_CLEARANCE_M:
                    continue
                nominal_areas = [region if track.size else swept_strips(points, *NOMINAL)
                                 for track, points, region in zip((a, b), paths, bigger)]
                nominal = clearance if a.size and b.size else float(
                    shapely.distance(*(shapely.union_all(strips) for strips in nominal_areas)))
                key = tuple(sorted((a.identifier, b.identifier)))
                order = (nominal > MAX_CLEARANCE_M, clearance, -(end - start), start)
                if key not in pairs or order < pairs[key][0]:
                    pairs[key] = order, {
                        "start_ms": start, "end_ms": end, "duration_s": (end - start) / 1000,
                        "nominal_clearance_m": nominal, "larger_assumed_body_clearance_m": clearance,
                        "track_1_length_m": (a.size or NOMINAL)[0], "track_1_width_m": (a.size or NOMINAL)[1],
                        "track_2_length_m": (b.size or NOMINAL)[0], "track_2_width_m": (b.size or NOMINAL)[1],
                        "track_1_measured_size": a.size is not None, "track_2_measured_size": b.size is not None,
                        "track_1": a.identifier, "track_2": b.identifier,
                    }
    return pairs, len(tracks)


def previous_pairs(path):
    cases = json.loads(path.read_bytes())["cases"]
    return {(case["source_file"], tuple(sorted(str(row["track_id"]) for row in case["raw_rows"])))
            for case in cases}


def previous_waymo_pairs(path):
    cases = json.loads(path.read_bytes())["cases"]
    return {(case["scenario_id"], tuple(sorted(str(row["track_id"]) for row in case["raw_rows"])))
            for case in cases if case["dataset"] == "Waymo Open Motion"}


def waymo_footprint_message_types():
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

    descriptor = descriptor_pb2.FileDescriptorProto(name="raw_waymo_footprints.proto", syntax="proto2")
    field = descriptor_pb2.FieldDescriptorProto
    for name, fields in {
        "Scenario": [("times", 1, field.TYPE_DOUBLE, True), ("tracks", 2, field.TYPE_BYTES, True),
                     ("pair", 4, field.TYPE_INT32, True), ("id", 5, field.TYPE_STRING, False)],
        "Track": [("id", 1, field.TYPE_INT32, False), ("object_type", 2, field.TYPE_INT32, False),
                  ("states", 3, field.TYPE_BYTES, True)],
        "State": [("x", 2, field.TYPE_DOUBLE, False), ("y", 3, field.TYPE_DOUBLE, False),
                  ("length", 5, field.TYPE_FLOAT, False), ("width", 6, field.TYPE_FLOAT, False),
                     ("heading", 8, field.TYPE_FLOAT, False),
                     ("vx", 9, field.TYPE_FLOAT, False), ("vy", 10, field.TYPE_FLOAT, False),
                  ("valid", 11, field.TYPE_BOOL, False)],
    }.items():
        message = descriptor.message_type.add(name=name)
        for key, number, kind, repeated in fields:
            message.field.add(name=key, number=number, type=kind,
                              label=field.LABEL_REPEATED if repeated else field.LABEL_OPTIONAL)
    pool = descriptor_pool.DescriptorPool()
    pool.Add(descriptor)
    return tuple(
        message_factory.GetMessageClass(pool.FindMessageTypeByName(name))
        for name in ("Scenario", "Track", "State")
    )


def scan_waymo(baseline, previous_near, downloads):
    from traffic_data.sources.extract_waymo_two_car_cases import records

    scenario_type, track_type, state_type = waymo_footprint_message_types()
    rows, coverage, seen = [], [], set()
    for path in sorted(downloads.glob("*validation_interactive*.tfrecord-*")):
        count = accepted = 0
        for record_index, payload in enumerate(records(path)):
            count += 1
            scenario = scenario_type.FromString(payload)
            if scenario.id in seen or len(set(scenario.pair)) != 2:
                continue
            seen.add(scenario.id)
            raw_tracks = [t for raw in scenario.tracks if (t := track_type.FromString(raw)).id in scenario.pair
                          and t.object_type == 1 and len(t.states) == len(scenario.times)]
            if len(raw_tracks) != 2:
                continue
            times = np.rint(np.asarray(scenario.times) * 1000).astype(int)
            tracks = []
            for track in raw_tracks:
                states = [(t, s) for t, raw in zip(times, track.states)
                          if (s := state_type.FromString(raw)).valid and s.HasField("x") and s.HasField("y")
                          and np.isfinite(s.x) and np.isfinite(s.y)]
                track_times = np.array([t for t, _ in states], dtype=int)
                xy = np.array([(s.x, s.y) for _, s in states])
                sizes = [(s.length, s.width) for _, s in states if s.length > 0 and s.width > 0]
                size = tuple(np.median(sizes, axis=0)) if sizes else None
                for run in continuous_runs(track_times):
                    t = track_times[run]
                    if len(t) and t[-1] - t[0] >= MIN_DURATION_MS:
                        tracks.append(Track(str(track.id), t, xy[run], size))
            found, _ = scan_tracks(tracks)
            for pair, (_, details) in found.items():
                key = (scenario.id, pair)
                rows.append({
                    "dataset": "Waymo Open Motion", "site": "Waymo_validation_interactive", "road_type": "unknown",
                    "raw_file": path.name, "previous_source_file": path.name, "scenario_id": scenario.id,
                    "source_record_index": record_index, "in_previous_6583": key in baseline,
                    "in_previous_3138": key in previous_near, **details,
                })
                accepted += 1
        coverage.append({"raw_file": path.name, "scenarios_read": count, "qualifying_pairs": accepted})
        print(path.name[-14:], count, "scenes", accepted, "pairs", flush=True)
    return rows, coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("INTERACTION", "INTERACTION-TC", "SinD", "Songdo", "Waymo"))
    parser.add_argument("--baseline", type=Path, default=DATA / "expanded_6583_v3.json")
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--tc-root", type=Path, default=SOURCE.parent / "interaction_extension/source/TC")
    parser.add_argument("--waymo-root", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    dataset = args.dataset
    original = args.baseline
    near = args.output / "near_miss_candidates.json"
    if dataset == "Waymo":
        rows, scanned = scan_waymo(previous_waymo_pairs(original), previous_waymo_pairs(near), args.waymo_root)
    else:
        rows, scanned = scan_recordings(dataset, previous_pairs(original), previous_pairs(near),
                                        args.source_dir, args.tc_root)
    if not scanned:
        parser.error(f"No raw recordings found for {dataset}")
    nickname = re.sub(r"[^a-z]", "", dataset.lower())
    with (args.output / f"raw_near_miss_{nickname}.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output / f"raw_near_miss_{nickname}_coverage.json").write_text(json.dumps(scanned, indent=2), encoding="utf-8")
    print(dataset, "recordings", len(scanned), "pairs", len(rows),
          "new", sum(not row["in_previous_6583"] for row in rows), "road types",
          Counter(row["road_type"] for row in rows if not row["in_previous_6583"]))


def scan_recordings(dataset, baseline, previous_near, source, tc_root):
    rows = []
    scanned = []
    for index, (path, site, source_file) in enumerate(recordings(dataset, source, tc_root), 1):
        found, tracks = scan(path, dataset)
        relative = path.relative_to(tc_root if dataset == "INTERACTION-TC" else source).as_posix()
        scanned.append({"raw_file": relative, "eligible_track_segments": tracks, "qualifying_pairs": len(found)})
        for pair, (_, details) in found.items():
            pair_key = source_file, pair
            rows.append({"dataset": "INTERACTION" if dataset == "INTERACTION-TC" else dataset,
                         "site": site, "road_type": road_type(site), "raw_file": relative,
                         "previous_source_file": source_file, "in_previous_6583": pair_key in baseline,
                         "in_previous_3138": pair_key in previous_near, **details})
        if index % 25 == 0:
            print(dataset, index, "recordings", len(rows), "pairs so far", flush=True)
    return rows, scanned


if __name__ == "__main__":
    main()
