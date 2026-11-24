"""Extract annotated Waymo vehicle pairs into the shared kinematic manifest."""

import argparse
from collections import defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from enrich_real_traffic_cases import maneuver


def message_types():
    # Field numbers and wire types follow Waymo's scenario.proto. Reading tracks
    # and states as bytes lets us decode state arrays only for the annotated pair.
    file = descriptor_pb2.FileDescriptorProto(name="waymo_motion_subset.proto", syntax="proto2")
    field = descriptor_pb2.FieldDescriptorProto
    for name, fields in {
        "Scenario": [
            ("timestamps", 1, field.TYPE_DOUBLE, True), ("tracks", 2, field.TYPE_BYTES, True),
            ("objects_of_interest", 4, field.TYPE_INT32, True), ("scenario_id", 5, field.TYPE_STRING, False),
            ("current_time_index", 10, field.TYPE_INT32, False),
        ],
        "Track": [("id", 1, field.TYPE_INT32, False), ("object_type", 2, field.TYPE_INT32, False),
                  ("states", 3, field.TYPE_BYTES, True)],
        "State": [
            ("center_x", 2, field.TYPE_DOUBLE, False), ("center_y", 3, field.TYPE_DOUBLE, False),
            ("heading", 8, field.TYPE_FLOAT, False), ("velocity_x", 9, field.TYPE_FLOAT, False),
            ("velocity_y", 10, field.TYPE_FLOAT, False), ("valid", 11, field.TYPE_BOOL, False),
        ],
    }.items():
        descriptor = file.message_type.add(name=name)
        for key, number, kind, repeated in fields:
            descriptor.field.add(name=key, number=number, type=kind,
                                 label=field.LABEL_REPEATED if repeated else field.LABEL_OPTIONAL)
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file)
    factory = message_factory.MessageFactory(pool)
    return tuple(factory.GetPrototype(pool.FindMessageTypeByName(name))
                 for name in ("Scenario", "Track", "State"))


def records(path):
    with path.open("rb") as stream:
        while header := stream.read(12):
            if len(header) != 12:
                raise ValueError(f"Truncated TFRecord header in {path}")
            size, _ = struct.unpack("<QI", header)
            if size > 64 * 1024 * 1024:
                raise ValueError(f"Unexpected TFRecord size in {path}: {size}")
            payload = stream.read(size)
            if len(payload) != size or len(stream.read(4)) != 4:
                raise ValueError(f"Truncated TFRecord payload in {path}")
            yield payload


def geometry(positions, velocities, *, expanded=False):
    speed = np.linalg.norm(velocities, axis=1)
    separation = np.linalg.norm(positions[1] - positions[0])
    min_speed, min_spacing, max_spacing = (0.8, 3.5, 65) if expanded else (2, 5, 50)
    if not (speed.min() >= min_speed and speed.max() <= 15
            and min_spacing <= separation <= max_spacing):
        return None
    directions = velocities / speed[:, None]
    angle = math.degrees(math.acos(np.clip(directions[0] @ directions[1], -1, 1)))
    order = [0, 1]
    if angle <= 8:
        if (positions[1] - positions[0]) @ directions[0] < 0:
            order.reverse()
        a, b = order
        delta = positions[b] - positions[a]
        gap = float(delta @ directions[a])
        lateral = np.linalg.norm(delta - gap * directions[a])
        closing = speed[a] - speed[b]
        min_closing, max_catch = (.35, 15) if expanded else (1, 8)
        if gap <= 0 or lateral > 1.8 or closing < min_closing or gap / closing > max_catch:
            return None
        reference = positions[b]
        starts = [reference - gap * directions[a], reference]
        return ("following", order, reference, starts, [directions[a]] * 2, speed[order],
                [gap, 0.0], angle)
    if angle > 165:
        return None
    distances = np.linalg.solve(np.stack((directions[0], -directions[1]), axis=1),
                                positions[1] - positions[0])
    arrivals = distances / speed
    earliest, latest, max_gap = (.35, 8, 4) if expanded else (.6, 5.5, 2.5)
    if arrivals.min() < earliest or arrivals.max() > latest or abs(arrivals[1] - arrivals[0]) > max_gap:
        return None
    family = next(label for upper, label in (
        (25, "shallow_convergence"), (65, "oblique"), (115, "crossing"), (166, "obtuse")
    ) if angle < upper)
    return (family, order, positions[0] + distances[0] * directions[0],
            positions, directions, speed, distances, angle)


def case_from_scenario(scene, track_type, state_type, source, record_index, *, expanded=False):
    ids = set(scene.objects_of_interest)
    if len(ids) != 2 or not scene.scenario_id or not scene.HasField("current_time_index"):
        return None
    tracks = [track for raw in scene.tracks if (track := track_type.FromString(raw)).id in ids]
    if len(tracks) != 2 or any(track.object_type != 1 for track in tracks):
        return None
    timestamps = scene.timestamps
    if any(len(track.states) != len(timestamps) for track in tracks):
        return None
    states = [[state_type.FromString(raw) for raw in track.states] for track in tracks]
    first = scene.current_time_index
    # Search closest to the official current frame first.
    before, after = (10, 40) if expanded else (0, 20)
    indices = range(max(0, first - before), min(first + after + 1, len(timestamps) - 30))
    for index in sorted(indices, key=lambda i: (abs(i - first), i < first)):
        step = [s[index] for s in states]
        fields = ("center_x", "center_y", "velocity_x", "velocity_y")
        if any(not s.valid or not all(s.HasField(k) and math.isfinite(getattr(s, k))
                                     for k in fields) for s in step):
            continue
        result = geometry(np.array([[s.center_x, s.center_y] for s in step]),
                          np.array([[s.velocity_x, s.velocity_y] for s in step]), expanded=expanded)
        if result is None:
            continue
        family, order, reference, starts, directions, speeds, distances, angle = result
        routes, raw_rows = [], []
        for agent in order:
            track_id = str(tracks[agent].id)
            samples = []
            for j in range(index, min(index + 61, len(timestamps))):
                s = states[agent][j]
                if not s.valid or not all(s.HasField(k) and math.isfinite(getattr(s, k)) for k in fields):
                    break
                samples.append({"t_s": round(timestamps[j] - timestamps[index], 4),
                                "x_m": s.center_x - reference[0], "y_m": s.center_y - reference[1],
                                "vx_mps": s.velocity_x, "vy_mps": s.velocity_y})
            if len(samples) < 31:
                break
            s = step[agent]
            raw_rows.append({"track_id": track_id, "frame_id": index,
                             "timestamp_ms": round(timestamps[index] * 1000),
                             "x": s.center_x, "y": s.center_y, "vx": s.velocity_x,
                             "vy": s.velocity_y, "heading_rad": s.heading})
            routes.append({"track_id": track_id, **maneuver(samples), "samples": samples})
        if len(routes) != 2:
            continue
        return {
            "case_id": f"waymo_{scene.scenario_id}", "dataset": "Waymo Open Motion",
            "scenario_id": scene.scenario_id, "site": "Waymo_validation_interactive",
            "source_file": source, "source_record_index": record_index,
            "timestamp_ms": round(timestamps[index] * 1000), "initial_frame_index": index,
            "current_frame_index": first, "family": family, "road_type": "unknown",
            "screening": "expanded" if expanded else "standard",
            "raw_rows": raw_rows, "observed_routes": routes, "observed_heading_angle_deg": angle,
            "conflict_xy": reference.tolist(), "simulated_start_xy": np.asarray(starts).tolist(),
            "simulated_heading_rad": [math.atan2(d[1], d[0]) for d in directions],
            "initial_speed_mps": speeds.tolist(), "distance_to_reference_m": list(distances),
            "gap_s": float(distances[1] / speeds[1] - distances[0] / speeds[0]), "max_time_s": 9.0,
        }
    return None


def extract(paths, limit):
    scene_type, track_type, state_type = message_types()
    groups = defaultdict(deque)
    counts, hashes = {}, {}
    seen = set()
    for path in paths:
        scanned, eligible, expanded = 0, 0, 0
        for index, payload in enumerate(records(path)):
            scanned += 1
            scene = scene_type.FromString(payload)
            if scene.scenario_id in seen:
                continue
            case = case_from_scenario(scene, track_type, state_type, path.name, index)
            if case is None:
                case = case_from_scenario(scene, track_type, state_type, path.name, index, expanded=True)
            if case is not None:
                seen.add(scene.scenario_id)
                eligible += 1
                expanded += case["screening"] == "expanded"
                groups[case["screening"] == "expanded", case["family"], path.name].append(case)
        counts[path.name] = {"scanned": scanned, "eligible": eligible, "expanded": expanded}
        with path.open("rb") as stream:
            hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    selected = []
    for expanded in (False, True):
        queues = deque(groups[key] for key in sorted(groups) if key[0] == expanded)
        while queues and (limit is None or len(selected) < limit):
            queue = queues.popleft()
            selected.append(queue.popleft())
            if queue:
                queues.append(queue)
    if not selected or (limit is not None and len(selected) != limit):
        raise ValueError(f"Found {len(selected)} eligible distinct scenarios, expected {limit}. {counts}")
    return {"schema_version": 1, "recordings_sha256": hashes, "source_counts": counts,
            "case_selection": "Waymo objects_of_interest, two vehicles, kinematic feasibility",
            "cases": selected}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", type=Path, nargs="+")
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--limit", type=int, default=142)
    size.add_argument("--all-eligible", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    manifest = extract(args.shards, None if args.all_eligible else args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f'{len(manifest["cases"])} cases -> {args.out}')
