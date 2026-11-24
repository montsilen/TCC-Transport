"""Extract auditable two-car snapshots from INTERACTION camera tracks."""

import csv
import itertools
import math
from collections import defaultdict

import numpy as np

from run_mechanism_ablation_v2 import ScenarioCase, Vehicle


EXTRA_TARGETS = (
    ("crossing", -1.0),
    ("crossing", 1.0),
    ("following", 2.5),
    ("following", 6.0),
    ("oblique", 1.0),
    ("obtuse", -1.0),
)


def scan_cases(root, *, extended=False, broad=False, sample_ms=1000):
    candidates = defaultdict(list)
    for path in sorted(root.rglob("vehicle_tracks_*.csv")):
        frames = defaultdict(list)
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                time = int(row["timestamp_ms"])
                if row["agent_type"] == "car" and time % sample_ms == 0:
                    frames[time].append(row)
        for time, rows in frames.items():
            for a, b in itertools.combinations(rows, 2):
                pos = np.array([[float(r[k]) for k in ("x", "y")] for r in (a, b)])
                vel = np.array([[float(r[k]) for k in ("vx", "vy")] for r in (a, b)])
                speed = np.linalg.norm(vel, axis=1)
                separation = np.linalg.norm(pos[0] - pos[1])
                original = speed.min() >= 3 and speed.max() <= 13 and 6 <= separation <= 38
                if not original and not (extended and speed.min() >= 2 and speed.max() <= 15
                                         and 5 <= separation <= 50):
                    continue
                direction = vel / speed[:, None]
                angle = math.degrees(math.acos(np.clip(direction[0] @ direction[1], -1, 1)))
                if angle <= 8:
                    # Reorder so that A is the faster follower.
                    if (pos[1] - pos[0]) @ direction[0] < 0:
                        a, b = b, a
                        pos, speed, direction = pos[::-1], speed[::-1], direction[::-1]
                    recorded_leader = "preceding_id" in a
                    if recorded_leader and (
                        a["preceding_id"] != b["track_id"] or a["lane_id"] != b["lane_id"]
                    ):
                        continue
                    along = (pos[1] - pos[0]) @ direction[0]
                    lateral = abs(np.linalg.det(np.stack((direction[0], pos[1] - pos[0]))))
                    closing = speed[0] - speed[1]
                    # An explicit same-lane leader link also supports milder approaches.
                    min_closing, max_catch = (0.5, 12) if recorded_leader else (1, 8)
                    if lateral > 1.8 or closing < min_closing or along / closing > max_catch:
                        continue
                    family = "following"
                    conflict = pos[1].copy()
                    distances = np.array([along, 0.0])
                    direction = np.stack((direction[0], direction[0]))
                    pos = conflict - distances[:, None] * direction
                    score = abs(along / closing - 4)
                elif (8 if broad else 25) <= angle <= 165:
                    distances = np.linalg.solve(np.stack((direction[0], -direction[1]), axis=1),
                                                pos[1] - pos[0])
                    arrival = distances / speed
                    gap_size = abs(arrival[1] - arrival[0])
                    original = original and arrival.min() >= 1 and arrival.max() <= 4 and gap_size <= 1.5
                    if not original and not (extended and arrival.min() >= 0.6
                                             and arrival.max() <= 5.5 and gap_size <= 2.5):
                        continue
                    conflict = pos[0] + distances[0] * direction[0]
                    if angle < 25:
                        family = "shallow_convergence"
                    elif angle < 65:
                        family = "oblique"
                    else:
                        family = "crossing" if angle < 115 else "obtuse"
                    score = abs(arrival.mean() - 2.5) + abs(arrival[1] - arrival[0])
                else:
                    continue
                gap = float(distances[1] / speed[1] - distances[0] / speed[0])
                selection_value = along / closing if family == "following" else gap
                case_id = f"{family}_{path.stem[-3:]}_{time}_{a['track_id']}_{b['track_id']}"
                if broad:
                    case_id = f"{path.parent.name}_{case_id}"
                vehicles = tuple(Vehicle(p, math.atan2(d[1], d[0]), float(s), name)
                                 for p, d, s, name in zip(pos, direction, speed, ("a", "b")))
                case = ScenarioCase(family, case_id, vehicles, conflict, 9.0, {"gap_s": gap})
                source = {
                    "case_id": case_id, "family": family, "source_file": f"{path.parent.name}/{path.name}",
                    "timestamp_ms": time, "raw_rows": [a, b], "observed_heading_angle_deg": angle,
                    "conflict_xy": conflict.tolist(),
                    "simulated_start_xy": pos.tolist(), "initial_speed_mps": speed.tolist(),
                    "distance_to_reference_m": distances.tolist(),
                    "simulated_heading_rad": [car.heading for car in vehicles],
                }
                candidates[family].append((score, case, source, selection_value, original))
    if not candidates:
        raise ValueError(f"No qualifying simultaneous vehicle pairs found in {root}")
    return candidates


def extract_cases(root, *, extended=False):
    candidates = scan_cases(root, extended=extended)
    ordered = []
    for family in sorted(candidates):
        original_pool = [item for item in candidates[family] if item[4]]
        if original_pool:
            ordered.append(min(original_pool, key=lambda item: item[0]))
    if extended:
        used = set()

        def track_keys(item):
            source = item[2]
            return {(source["source_file"], row["track_id"]) for row in source["raw_rows"]}

        for item in ordered:
            used.update(track_keys(item))
        for family, target in EXTRA_TARGETS:
            pool = [item for item in candidates[family] if not (track_keys(item) & used)]
            if not pool:
                raise ValueError(f"No additional independent {family} pair near {target} s")
            selected = min(pool, key=lambda item: (abs(item[3] - target), item[0]))
            ordered.append(selected)
            used.update(track_keys(selected))
            selected[2]["selection_target_s"] = target
            selected[2]["selection_actual_s"] = selected[3]
    return [item[1] for item in ordered], [item[2] for item in ordered]
