"""Rebuild synchronized fixed-path clips from the existing vehicle-pair census."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import shapely

from .audit_swept_footprints import LARGE, NOMINAL, OUTPUT, dimensions, swept_strips
from .find_raw_fixed_path_pairs import (
    SOURCE, Track, continuous_runs, nearby_clearance, read_tracks, recordings, waymo_footprint_message_types,
)
from ..sources.extract_waymo_two_car_cases import records


DEST = OUTPUT.parent / "fixed_path_common_windows"
DATASETS = ("interaction", "interactiontc", "sind", "songdo", "waymo")
WAYMO = "Waymo Open Motion"
MIN_WINDOW_S = 2.95  # Preserve native Waymo clocks for 30 nominal 0.1 s steps.
MAX_GAP_S = 0.205  # Retain source windows with one skipped 10 Hz frame.
THRESHOLD_M = 1.0
CONTACT_EPS_M = 1e-9
INDEX_FIELDS = (
    "scene_id", "candidate_id", "dataset", "site", "road_type", "source_file", "raw_file", "source_record_index",
    "track_a", "track_b", "window_source", "status", "archive_case_id", "raw_index_start_ms", "raw_index_end_ms",
    "clock_start_ms", "clock_end_ms", "duration_s", "sample_count", "max_sample_step_s",
    "array_file", "array_row", "start_a_x_m", "start_a_y_m", "start_b_x_m", "start_b_y_m",
    "end_a_x_m", "end_a_y_m", "end_b_x_m", "end_b_y_m", "initial_speed_a_mps", "initial_speed_b_mps",
    "initial_velocity_a_recorded", "initial_velocity_b_recorded", "progress_a_m", "progress_b_m",
    "size_a_recorded", "size_b_recorded", "swept_gap_m", "larger_box_swept_gap_m", "potential_class",
    "initial_body_gap_m", "sampled_min_body_gap_m", "sampled_min_gap_t_s", "observed_geometry_class",
)


def archive_key(case):
    tracks = sorted(str(row["track_id"]) for row in case["raw_rows"])
    return (case["dataset"].replace("Songdo Traffic", "Songdo"), case.get("scenario_id") or case["source_file"], *tracks)


def raw_key(row):
    tracks = sorted((row["track_1"], row["track_2"]))
    return (row["dataset"], row.get("scenario_id") or row["previous_source_file"], *tracks)


def index(dest, shards):
    inventory = {}
    for dataset in DATASETS:
        with (OUTPUT / f"raw_near_miss_{dataset}.csv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                key = raw_key(row)
                if key in inventory:
                    raise ValueError(f"Duplicate raw candidate: {key}")
                inventory[key] = (row, None)
    archive = json.loads((OUTPUT / "near_miss_candidates.json").read_bytes())["cases"]
    for case in archive:
        inventory.setdefault(archive_key(case), (None, case))
    assert len(inventory) == 125_932
    groups = defaultdict(list)
    for key, (raw, archived) in sorted(inventory.items()):
        if key[0] == WAYMO:
            source = raw["raw_file"] if raw else archived["source_file"]
        else:
            source = key[1]
        groups[(key[0], source)].append((key, raw, archived))
    assignments = [[] for _ in range(shards)]
    workload = [0] * shards
    # Keep whole recordings together so only one worker decodes each source.
    for group, entries in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        shard = min(range(shards), key=workload.__getitem__)
        assignments[shard].append((group, entries))
        workload[shard] += len(entries)
    for shard, groups_to_read in enumerate(assignments):
        with (dest / f"jobs_{shard:02}.jsonl").open("w", encoding="utf-8") as stream:
            for group in groups_to_read:
                stream.write(json.dumps(group, separators=(",", ":")) + "\n")
    print(f"Indexed {len(inventory):,} unique pairs in {len(groups)} recordings: {workload}", flush=True)


def recording_files():
    tc_root = SOURCE.parent / "interaction_extension/source/TC"
    files = {}
    for kind in ("INTERACTION", "INTERACTION-TC", "SinD", "Songdo"):
        label = "INTERACTION" if kind == "INTERACTION-TC" else kind
        for path, _, key in recordings(kind, SOURCE, tc_root):
            index_key = (label, key)
            if index_key in files:
                raise ValueError(f"Duplicate recording path: {index_key}")
            files[index_key] = path
    return files


def archive_window(archived, track_ids):
    routes = {str(r["track_id"]): r["samples"] for r in archived["observed_routes"]}
    body_rows = {str(r["track_id"]): r for r in archived["raw_rows"]}
    source = []
    duration = archived["observed_window"]["duration_s"]
    origin = np.asarray(archived["conflict_xy"])
    for identifier in track_ids:
        samples = [p for p in routes[identifier] if p["t_s"] <= duration + 1e-8]
        clock = archived["timestamp_ms"] + 1000 * np.array([s["t_s"] for s in samples])
        xy = origin + np.array([(s["x_m"], s["y_m"]) for s in samples])
        velocity = np.array([(s.get("vx_mps", np.nan), s.get("vy_mps", np.nan)) for s in samples])
        source.append(Track(identifier, np.rint(clock).astype(np.int64), xy, dimensions(body_rows[identifier]),
                            clock, velocity))
    return source


def protocol_window(raw, archived):
    if raw:
        return int(raw["start_ms"]), int(raw["end_ms"])
    start = int(archived["timestamp_ms"])
    return start, start + round(1000 * archived["observed_window"]["duration_s"])


def read_waymo_tracks(scene, track_type, state_type, ids):
    clock = np.asarray(scene.times) * 1000
    tracks = defaultdict(list)
    if not len(clock):
        return tracks
    # Rounded milliseconds preserve the geometry audit's indexing. Export the original clock.
    ticks = np.rint(clock).astype(np.int64)
    for payload in scene.tracks:
        track = track_type.FromString(payload)
        identifier = str(track.id)
        if identifier not in ids or track.object_type != 1 or len(track.states) != len(ticks):
            continue
        states = [state_type.FromString(payload) for payload in track.states]
        usable = np.array([s.valid and s.HasField("x") and s.HasField("y") and np.isfinite((s.x, s.y)).all()
                           for s in states])
        t = ticks[usable]
        if not len(t):
            continue
        xy = np.array([(s.x, s.y) for s, valid in zip(states, usable) if valid])
        velocity = np.array([(s.vx if s.HasField("vx") else np.nan, s.vy if s.HasField("vy") else np.nan)
                             for s, valid in zip(states, usable) if valid])
        heading = np.array([s.heading if s.HasField("heading") else np.nan
                            for s, valid in zip(states, usable) if valid])
        sizes = [(s.length, s.width) for s, valid in zip(states, usable) if valid
                 and np.isfinite((s.length, s.width)).all() and s.length > 0 and s.width > 0]
        size = tuple(np.median(sizes, axis=0)) if sizes else None
        for run in continuous_runs(t, max_gap=round(1000 * MAX_GAP_S)):
            tracks[identifier].append(Track(identifier, t[run], xy[run], size, clock[usable][run],
                                            velocity[run], heading[run]))
    return tracks


def waymo_entries(path, entries):
    scenario_type, track_type, state_type = waymo_footprint_message_types()
    by_record = defaultdict(list)
    for key, raw, archived in entries:
        descriptor = raw or archived
        by_record[int(descriptor["source_record_index"])].append((key, raw, archived))
    unresolved = set(by_record)
    if path.is_file():
        for record_number, payload in enumerate(records(path)):
            if record_number not in unresolved:
                continue
            unresolved.remove(record_number)
            scene = scenario_type.FromString(payload)
            for key, raw, archived in by_record[record_number]:
                tracks = read_waymo_tracks(scene, track_type, state_type, set(key[2:])) if scene.id == key[1] else {}
                yield key, raw, archived, tracks
            if not unresolved:
                break
    for record_number in sorted(unresolved):
        for key, raw, archived in by_record[record_number]:
            yield key, raw, archived, {}


def recording_entries(group, entries, files, waymo_root):
    dataset, source = group
    if dataset == WAYMO:
        yield from waymo_entries(waymo_root / source, entries)
        return
    tracks = defaultdict(list)
    if path := files.get((dataset, source)):
        for track in read_tracks(path, dataset, include_kinematics=True):
            tracks[track.identifier].append(track)
    for key, raw, archived in entries:
        yield key, raw, archived, tracks


def shared_samples(tracks, start, end, *, boundary_tolerance_ms=0):
    chosen = []
    for options in tracks:
        good = [t for t in options if t.times[0] <= start + boundary_tolerance_ms
                and t.times[-1] >= end - boundary_tolerance_ms]
        if len(good) != 1:
            return None
        chosen.append(good[0])
    ticks, ia, ib = np.intersect1d(chosen[0].times, chosen[1].times, assume_unique=True, return_indices=True)
    mask = (ticks >= start - boundary_tolerance_ms) & (ticks <= end + boundary_tolerance_ms)
    ticks, ia, ib = ticks[mask], ia[mask], ib[mask]
    if len(ticks) < 2 or max(abs(ticks[0] - start), abs(ticks[-1] - end)) > boundary_tolerance_ms:
        return None
    clock = [track.times[ix] if track.clock_ms is None else track.clock_ms[ix]
             for track, ix in zip(chosen, (ia, ib))]
    if not np.isfinite(clock).all() or not np.allclose(*clock, atol=1e-5, rtol=0):
        return None
    step = np.diff(clock[0]) / 1000
    if np.any(step <= 0) or step.sum() < MIN_WINDOW_S or step.max() > MAX_GAP_S:
        return None
    return np.asarray(clock[0]), chosen, (ia, ib)


def route_heading(xy, t, fallback):
    velocity = np.gradient(xy, t, axis=0)
    moving = np.linalg.norm(velocity, axis=1) > 0.02
    if moving.any():
        values = np.arctan2(velocity[moving, 1], velocity[moving, 0])
        return np.interp(t, t[moving], np.unwrap(values))
    available = fallback[np.isfinite(fallback)]
    return np.full(len(t), available[0] if len(available) else 0.0)


def car_samples(track, ix, elapsed):
    xy = track.xy[ix]
    velocity = np.full_like(xy, np.nan) if track.velocity_xy is None else track.velocity_xy[ix]
    observed_velocity = np.isfinite(velocity).all(axis=1)
    velocity = np.where(observed_velocity[:, None], velocity, np.gradient(xy, elapsed, axis=0))
    heading = np.full(len(ix), np.nan) if track.heading_rad is None else track.heading_rad[ix]
    observed_heading = np.isfinite(heading)
    heading = np.where(observed_heading, heading, route_heading(xy, elapsed, heading))
    travel = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    return xy, velocity, heading, travel, observed_velocity, observed_heading


def footprints(xy, heading, size):
    sine, cosine = np.sin(heading), np.cos(heading)
    front = np.stack((cosine, sine), axis=-1) * size[0] / 2
    side = np.stack((-sine, cosine), axis=-1) * size[1] / 2
    return shapely.polygons(np.stack((xy + front + side, xy - front + side,
                                      xy - front - side, xy + front - side), axis=-2))


def swept_gap(points, heading, sizes):
    strips = [swept_strips(path, *size) for path, size in zip(points, sizes)]
    # Retain a real occupied area if all recorded positions are identical.
    for i, strip in enumerate(strips):
        if not len(strip):
            strips[i] = footprints(points[i][:1], heading[i][:1], sizes[i])
    gap = nearby_clearance(*strips)
    if np.isfinite(gap):
        return gap
    _, distances = shapely.STRtree(strips[1]).query_nearest(strips[0], return_distance=True, all_matches=False)
    return float(distances.min())


def describe_episode(sample, raw):
    clock, tracks, ix = sample
    elapsed = (clock - clock[0]) / 1000
    cars = [car_samples(track, t, elapsed) for track, t in zip(tracks, ix)]
    xy, velocity, heading, progress, v_recorded, yaw_recorded = (
        np.stack(col, axis=1) for col in zip(*cars)
    )
    raw_slot = {raw[f"track_{i}"]: i for i in (1, 2)} if raw else {}
    sizes, measured = [], []
    for track in tracks:
        slot = raw_slot.get(track.identifier)
        if slot:
            size = (float(raw[f"track_{slot}_length_m"]), float(raw[f"track_{slot}_width_m"]))
            known = raw[f"track_{slot}_measured_size"] == "True"
        else:
            size, known = track.size or NOMINAL, track.size is not None
        sizes.append(size)
        measured.append(known)
    origin = xy[0].mean(axis=0)
    local = xy - origin
    nominal_gap = swept_gap(local.transpose(1, 0, 2), heading.T, sizes)
    larger = [size if known else LARGE for size, known in zip(sizes, measured)]
    large_gap = nominal_gap if all(measured) else swept_gap(local.transpose(1, 0, 2), heading.T, larger)
    simultaneous_gap = shapely.distance(*(footprints(local[:, i], heading[:, i], sizes[i]) for i in (0, 1)))
    if not np.isfinite(simultaneous_gap).all():
        raise ValueError("Non-finite synchronized vehicle clearance")
    nearest = int(np.argmin(simultaneous_gap))
    d = float(simultaneous_gap[nearest])
    geometry = "initial_contact" if simultaneous_gap[0] <= CONTACT_EPS_M else (
        "later_contact" if d <= CONTACT_EPS_M else "near_miss_sampled" if d <= THRESHOLD_M else "clear_sampled")
    potential = ("swept_overlap" if nominal_gap <= CONTACT_EPS_M else
                 "near_with_nominal_body" if nominal_gap <= THRESHOLD_M else
                 "near_only_with_larger_assumed_body" if large_gap <= THRESHOLD_M else "no_swept_near_miss")
    metadata = {
        "clock_start_ms": float(clock[0]), "clock_end_ms": float(clock[-1]), "duration_s": float(elapsed[-1]),
        "sample_count": len(elapsed), "max_sample_step_s": float(np.diff(elapsed).max()),
        **{f"{label}_{car}_{axis}_m": float(xy[at, i, j])
           for label, at in (("start", 0), ("end", -1))
           for i, car in enumerate("ab") for j, axis in enumerate("xy")},
        **{f"initial_speed_{car}_mps": float(np.linalg.norm(velocity[0, i])) for i, car in enumerate("ab")},
        **{f"initial_velocity_{car}_recorded": bool(v_recorded[0, i]) for i, car in enumerate("ab")},
        **{f"progress_{car}_m": float(progress[-1, i]) for i, car in enumerate("ab")},
        **{f"size_{car}_recorded": known for car, known in zip("ab", measured)},
        "swept_gap_m": nominal_gap, "larger_box_swept_gap_m": large_gap, "potential_class": potential,
        "initial_body_gap_m": float(simultaneous_gap[0]), "sampled_min_body_gap_m": d,
        "sampled_min_gap_t_s": float(elapsed[nearest]), "observed_geometry_class": geometry,
    }
    arrays = {
        "t_s": elapsed, "xy_m": local.astype(np.float32), "velocity_mps": velocity.astype(np.float32),
        "heading_rad": heading.astype(np.float32), "progress_m": progress.astype(np.float32),
        "velocity_recorded": v_recorded, "heading_recorded": yaw_recorded,
    }
    constants = {"origin_xy_m": origin, "body_size_m": np.asarray(sizes, dtype=np.float32),
                 "body_size_recorded": np.asarray(measured, dtype=bool)}
    return metadata, arrays, constants


def rebuild(key, raw, archived, tracks):
    dataset, source, a, b = key
    descriptor = raw or archived
    start, end = protocol_window(raw, archived)
    candidate = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
    scene_id = hashlib.blake2b(candidate.encode("utf-8"), digest_size=12).hexdigest()
    row = {
        "scene_id": scene_id, "candidate_id": candidate, "dataset": dataset,
        "site": descriptor["site"], "road_type": descriptor["road_type"],
        "source_file": descriptor["previous_source_file"] if raw else archived["source_file"],
        "raw_file": descriptor.get("raw_file", archived["source_file"] if archived else ""),
        "source_record_index": descriptor.get("source_record_index", ""),
        "track_a": a, "track_b": b, "archive_case_id": archived["case_id"] if archived else "",
        "raw_index_start_ms": start, "raw_index_end_ms": end,
    }
    # An archive can independently round its absolute start and relative duration.
    tolerance = 1 if archived else 0
    sample = shared_samples([tracks.get(a, []), tracks.get(b, [])], start, end, boundary_tolerance_ms=tolerance)
    window_source = "raw_indexed" if raw else "raw_recovered_archive"
    if sample is None and archived:
        aa, bb = archive_window(archived, (a, b))
        sample = shared_samples([[aa], [bb]], start, end, boundary_tolerance_ms=tolerance)
        window_source = "archive_samples"
    if sample is None:
        return {**row, "status": "missing_synchronized_source"}, None, None
    attributes, arrays, constants = describe_episode(sample, raw)
    return {**row, "status": "aligned", "window_source": window_source, **attributes}, arrays, constants


def build(dest, shard, waymo_root):
    files = recording_files()
    chunk = f"windows_{shard:02}"
    frame_data, scene_data = defaultdict(list), defaultdict(list)
    offsets, ids, histogram = [0], [], Counter()
    index = dest / f"index_{shard:02}.csv"
    jobs = dest / f"jobs_{shard:02}.jsonl"
    with (jobs.open(encoding="utf-8") as stream,
          index.with_suffix(".tmp").open("w", encoding="utf-8", newline="") as results):
        writer = csv.DictWriter(results, INDEX_FIELDS)
        writer.writeheader()
        for group_number, line in enumerate(stream, 1):
            group, entries = json.loads(line)
            for key, raw, archived, tracks in recording_entries(group, entries, files, waymo_root):
                row, arrays, constants = rebuild(key, raw, archived, tracks)
                histogram[row["status"]] += 1
                if arrays is not None:
                    row.update(array_file=f"{chunk}.npz", array_row=len(ids))
                    ids.append(row["scene_id"])
                    offsets.append(offsets[-1] + row["sample_count"])
                    for name, data in arrays.items():
                        frame_data[name].append(data)
                    for name, data in constants.items():
                        scene_data[name].append(data)
                writer.writerow(row)
            if group_number % 5 == 0:
                print(shard, group_number, dict(histogram), flush=True)
    target = dest / f"{chunk}.npz"
    with target.with_suffix(".tmp").open("wb") as stream:
        np.savez_compressed(stream, offsets=np.asarray(offsets, dtype=np.int64), scene_id=np.asarray(ids, dtype="U24"),
                            **{key: np.concatenate(items) for key, items in frame_data.items()},
                            **{key: np.stack(items) for key, items in scene_data.items()})
    target.with_suffix(".tmp").replace(target)
    index.with_suffix(".tmp").replace(index)
    print(shard, "complete", dict(histogram), flush=True)


def combine(dest, shards):
    rows = []
    for shard in range(shards):
        with (dest / f"index_{shard:02}.csv").open(encoding="utf-8", newline="") as stream:
            portion = list(csv.DictReader(stream))
        rows.extend(portion)
        aligned = [row for row in portion if row["status"] == "aligned"]
        with np.load(dest / f"windows_{shard:02}.npz", allow_pickle=False) as archive:
            offsets, identifiers, elapsed = (archive[name] for name in ("offsets", "scene_id", "t_s"))
            assert list(identifiers) == [row["scene_id"] for row in aligned]
            assert list(np.diff(offsets)) == [int(row["sample_count"]) for row in aligned]
            assert np.all(elapsed[offsets[:-1]] == 0)
            assert np.allclose(elapsed[offsets[1:] - 1], [float(row["duration_s"]) for row in aligned])
            for name in ("xy_m", "velocity_mps", "heading_rad", "progress_m"):
                signal = archive[name]
                assert len(signal) == offsets[-1] and np.isfinite(signal).all()
            progress, position, origin = (archive[name] for name in ("progress_m", "xy_m", "origin_xy_m"))
            assert np.all(progress[offsets[:-1]] == 0)
            for at, label in ((offsets[:-1], "start"), (offsets[1:] - 1, "end")):
                world = position[at] + origin[:, None, :]
                expected = np.array([[[float(r[f"{label}_{car}_{axis}_m"]) for axis in "xy"] for car in "ab"]
                                     for r in aligned])
                assert np.allclose(world, expected, atol=0.001, rtol=0)
    assert len(rows) == len({row["candidate_id"] for row in rows}) == len({row["scene_id"] for row in rows}) == 125_932
    output = dest / "windows.csv"
    with output.with_suffix(".tmp").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".tmp").replace(output)
    aligned = [row for row in rows if row["status"] == "aligned"]
    durations = np.array([float(row["duration_s"]) for row in aligned])
    by_dataset = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        group = [row for row in rows if row["dataset"] == dataset]
        by_dataset[dataset] = {
            "candidates": len(group), "aligned": sum(r["status"] == "aligned" for r in group),
            "potential_class": dict(Counter(r["potential_class"] for r in group if r["potential_class"])),
        }
    summary = {
        "candidates": len(rows), "aligned": len(aligned), "status": dict(Counter(r["status"] for r in rows)),
        "by_dataset": by_dataset, "road_type": dict(Counter(r["road_type"] for r in aligned)),
        "window_source": dict(Counter(r["window_source"] for r in aligned)),
        "potential_class": dict(Counter(r["potential_class"] for r in aligned)),
        "observed_geometry_class": dict(Counter(r["observed_geometry_class"] for r in aligned)),
        "duration_s": {str(q): float(np.quantile(durations, q)) for q in (0, .25, .5, .75, .95, 1)},
        "complete_seconds_available": {str(t): int(np.count_nonzero(durations >= t)) for t in (3, 5, 10)},
        "recorded_initial_velocity_both": sum(r["initial_velocity_a_recorded"] == r["initial_velocity_b_recorded"] == "True"
                                              for r in aligned),
        "sampling_note": "Both vehicles share one source clock per clip. Original sample times are retained, except "
                         "Songdo tracks on the existing 100 ms interpolated grid. Observed clearances use recorded "
                         "knots only. A skipped source frame can leave an approximately 200 ms interval.",
        "potential_note": "Minimum distance between swept boxes at any locations along the two observed paths; "
                          "it does not require simultaneous arrival.",
        "endpoint_note": "The two starts and ends are matched-time sampled positions, not imposed finishing goals.",
        "array_schema": {
            "scene_id": "24-hex-character ID matching windows.csv",
            "offsets": "frame slice for row i is offsets[i]:offsets[i+1]",
            "t_s": "shared elapsed seconds from the source window start",
            "xy_m": "frame × ordered vehicle (a,b) × xy; add origin_xy_m[i] for source coordinates",
            "velocity_mps": "frame × ordered vehicle × xy; velocity_recorded flags source values vs derivatives",
            "heading_rad": "frame × ordered vehicle; heading_recorded flags source values vs trajectory tangents",
            "progress_m": "frame × ordered vehicle; traveled arc length relative to each start",
            "body_size_m": "scene × ordered vehicle × (length, width); body_size_recorded flags source dimensions",
        },
    }
    (dest / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("index", "build", "combine"))
    parser.add_argument("--out", type=Path, default=DEST)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--waymo-root", type=Path, default=Path.home() / "Downloads")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "index":
        index(args.out, args.shards)
    elif args.stage == "build":
        if args.shard is None or not 0 <= args.shard < args.shards:
            parser.error("build requires a shard number in [0, shards)")
        build(args.out, args.shard, args.waymo_root)
    else:
        combine(args.out, args.shards)


if __name__ == "__main__":
    main()
