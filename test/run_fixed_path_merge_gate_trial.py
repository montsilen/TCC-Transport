"""Evaluate fixed paths at the two mapped lane drops in DR_CHN_Merging_ZS."""

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import sys
import xml.etree.ElementTree as ET

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_fixed_path_ttc_trial as trial
from fixed_path_merge_policy import FORECAST_STRIDE, MergePolicy
from fixed_path_trial_scene import DATA, Route, Scene, traffic_files
from traffic_data.fixed_path.find_raw_fixed_path_pairs import read_tracks


SITE = "DR_CHN_Merging_ZS"
RESULTS = ROOT / "Result"
PRIOR = RESULTS / "fixed_path_trial_1000" / "full_candidate_sweep_15mps" / "screening.csv"
RAW = RESULTS / "fixed_path_audit" / "raw_near_miss_interaction.csv"
DEST = RESULTS / "fixed_path_merge_gate_zs_15mps"
GATE_AFTER_M = 5.0
APPROACH_M = 45.0
SPEED_LIMIT = 15.0
ARMS = tuple((mode, method) for mode in ("discrete", "continuous")
             for method in ("ttc_bargaining", "fixed_discount_bargaining")) + (
    ("baseline", "two_layer_mpc"), ("baseline", "fixed_release"), ("baseline", "human"))


def projection(points, axis):
    return np.sum(points * axis, axis=-1)


@dataclass
class Merge:
    name: str
    taper_lanelet: str
    receiving_lanelet: str
    tip: np.ndarray
    direction: np.ndarray
    center: np.ndarray
    gate: np.ndarray
    ramp: object
    main: object

    def gate_crossing(self, track):
        signed = projection(track.xy - self.center, self.direction)
        candidates = np.flatnonzero((signed[:-1] < 0) & (signed[1:] >= 0))
        for i in candidates:
            fraction = -signed[i] / (signed[i + 1] - signed[i])
            xy = track.xy[i] + fraction * (track.xy[i + 1] - track.xy[i])
            along = self.gate[1] - self.gate[0]
            relative = projection(xy - self.gate[0], along) / projection(along, along)
            if 0 <= relative <= 1:
                return i + 1
        return None


def mapped_merges():
    root = ET.parse(DATA / "maps" / "maps" / f"{SITE}.osm_xy").getroot()
    nodes = {n.get("id"): (float(n.get("x")), float(n.get("y"))) for n in root.findall("node")}
    ways = {w.get("id"): np.array([nodes[n.get("ref")] for n in w.findall("nd")]) for w in root.findall("way")}
    bounds = {
        r.get("id"): {m.get("role"): ways[m.get("ref")] for m in r.findall("member")
                      if m.get("role") in ("left", "right")}
        for r in root.findall("relation")
    }

    def lane(lanelet, direction):
        left, right = (bounds[lanelet][side] for side in ("left", "right"))
        return tuple(x if projection(x[-1] - x[0], direction) > 0 else x[::-1] for x in (left, right))

    def polygon(lanelet, direction):
        left, right = lane(lanelet, direction)
        return Polygon(np.concatenate((left, right[::-1])))

    specifications = (
        ("eastbound", 1, "30028", "30037",
         ("30001", "30017", "30021", "30028"), ("30004", "30040", "30016", "30022", "30023")),
        ("westbound", -1, "30036", "30035",
         ("30041", "30044", "30025", "30036"), ("30043", "30032", "30024", "30031", "30035")),
    )
    merges = []
    for name, sign, taper, receiver, ramps, mains in specifications:
        heading = np.array([sign, 0])
        taper_left, taper_right = lane(taper, heading)
        tip = taper_left[-1]
        assert np.allclose(tip, taper_right[-1])
        left, right = lane(receiver, heading)
        fractions = np.linspace(0, 1, 101)
        centers = (shapely.get_coordinates(shapely.line_interpolate_point(LineString(left), fractions, normalized=True))
                   + shapely.get_coordinates(shapely.line_interpolate_point(LineString(right), fractions, normalized=True))) / 2
        centerline = LineString(centers)
        station = centerline.project(Point(tip)) + GATE_AFTER_M
        assert station < centerline.length
        center = shapely.get_coordinates(centerline.interpolate(station))[0]
        tangent = shapely.get_coordinates(centerline.interpolate(station + 0.05))[0] - center
        direction = tangent / np.linalg.norm(tangent)
        perpendicular = np.array([-direction[1], direction[0]])
        gate = shapely.get_coordinates(polygon(receiver, heading).intersection(
            LineString([center - 15 * perpendicular, center + 15 * perpendicular])))
        assert gate.shape == (2, 2)
        merges.append(Merge(name, taper, receiver, tip, direction, center, gate,
                            shapely.union_all([polygon(k, heading) for k in ramps]),
                            shapely.union_all([polygon(k, heading) for k in mains])))
    return merges


def pair_key(row, source_field):
    return (row[source_field], *sorted((row["track_1"], row["track_2"])))


def candidates_for_site():
    recordings = defaultdict(list)
    with PRIOR.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["site"] == SITE:
                recordings[row["source"]].append(row)
    with RAW.open(encoding="utf-8-sig", newline="") as stream:
        raw = {pair_key(row, "previous_source_file"): row for row in csv.DictReader(stream) if row["site"] == SITE}
    return recordings, raw


def choose_run(runs, indexed):
    if indexed is not None:
        first, last = (int(indexed[key]) for key in ("start_ms", "end_ms"))
        runs = [track for track in runs if track.times[0] <= first and track.times[-1] >= last]
    return runs[0] if len(runs) == 1 else None


def prepare_merge_pair(merge, tracks, crosses, file):
    ta, tb = tracks
    times, ia, ib = np.intersect1d(ta.times, tb.times, assume_unique=True, return_indices=True)
    common = (ia, ib)
    start_options = []
    in_ramp = [shapely.contains_xy(merge.ramp, *(track.xy[ix].T)) for track, ix in zip(tracks, common)]
    in_main = [shapely.contains_xy(merge.main, *(track.xy[ix].T)) for track, ix in zip(tracks, common)]
    stations = [projection(track.xy[ix] - merge.tip, merge.direction) for track, ix in zip(tracks, common)]
    approaching = ((ia < crosses[0]) & (ib < crosses[1])
                   & (stations[0] >= -APPROACH_M) & (stations[1] >= -APPROACH_M))
    for ramp_idx in (0, 1):
        frames = np.flatnonzero(approaching & in_ramp[ramp_idx] & in_main[1 - ramp_idx])
        if len(frames):
            start_options.append((int(frames[0]), ramp_idx))
    if not start_options:
        return "no_coobserved_ramp_and_receiving_approach", None, {}
    start, ramp_idx = min(start_options)
    # Both recordings must extend through the later finish so safety can be scored until then.
    if not len(times) or times[-1] < max(track.times[k] for track, k in zip(tracks, crosses)):
        return "coobservation_ends_before_later_finish", None, {}
    if any(track.size is None for track in tracks):
        return "missing_vehicle_dimensions", None, {}
    routes, goals, entry, exit, speeds, measured = [], [], [], [], [], []
    for track, ix, hit in zip(tracks, common, crosses):
        route, s = Route.build(track.xy, track.size)
        signed = projection(track.xy[hit-1:hit+1] - merge.center, merge.direction)
        goal = float(np.interp(0, signed, s[hit-1:hit+1]))
        # Sufficient observed road after the line keeps a leading car's path boundary
        # beyond the last possible contact with a car that has not yet finished.
        downstream = projection(track.xy[hit:] - merge.center, merge.direction)
        if downstream.max() < 10.0:
            return "less_than_10m_observed_after_finish", None, {}
        stations = projection(track.xy[:hit+1] - merge.tip, merge.direction)
        entrance = np.flatnonzero(stations >= -15.0)
        end = np.flatnonzero(stations >= 0)
        if not len(entrance) or not len(end):
            return "merge_reference_not_on_recorded_path", None, {}
        routes.append(route)
        goals.append(goal)
        entry.append(s[entrance[0]])
        exit.append(s[end[0]])
        measured.append(s[ix[start:]])
        k = ix[start]
        before, after = max(0, k - 1), min(len(s) - 1, k + 1)
        speeds.append((s[after] - s[before]) / ((track.times[after] - track.times[before]) / 1000))
    progress = np.array(measured).T
    if shapely.intersects(*(route.polygons(position) for route, position in zip(routes, progress[0]))):
        return "contact_already_at_start", None, {}
    if not np.all(progress[-1] >= goals):
        return "coobservation_ends_before_later_finish", None, {}
    case = Scene(f"{Path(file).stem}_{ta.identifier}_{tb.identifier}", file, SITE, tuple(routes), np.array(entry),
                 np.array(exit), np.array(goals), (times[start:] - times[start]) / 1000,
                 progress, np.array(speeds), True, bool(max(speeds) < 2), int(times[start]))
    return "eligible", case, {"direction": merge.name, "ramp_track": tracks[ramp_idx].identifier,
                               "gate_goal_a_m": goals[0], "gate_goal_b_m": goals[1], "start_ms": case.start_ms}


def screen():
    merges, files = mapped_merges(), traffic_files()
    recordings, raw = candidates_for_site()
    stats = Counter()
    with (DEST / "screening.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ("candidate_id", "source", "track_1", "track_2", "previous_outcome", "previous_reason",
                  "shared_gate_observed", "reason", "direction", "ramp_track", "gate_goal_a_m", "gate_goal_b_m", "start_ms")
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for file, rows in sorted(recordings.items()):
            tracks = defaultdict(list)
            for track in read_tracks(files[file], "INTERACTION"):
                tracks[track.identifier].append(track)
            crossings = [{id(track): merge.gate_crossing(track) for runs in tracks.values() for track in runs}
                         for merge in merges]
            prepared = []
            checkpoint = Counter()
            for row in rows:
                indexed = raw.get(pair_key(row, "source"))
                pair = tuple(choose_run(tracks[row[field]], indexed) for field in ("track_1", "track_2"))
                shared = False
                reason, scene, metadata = "recorded_runs_not_reconstructed", None, {}
                if all(track is not None for track in pair):
                    reason = "no_shared_gate_crossing_by_both_vehicles"
                    for merge, times in zip(merges, crossings):
                        hits = tuple(times[id(track)] for track in pair)
                        if all(hit is not None for hit in hits):
                            shared = True
                            reason, scene, metadata = prepare_merge_pair(merge, pair, hits, file)
                            metadata.setdefault("direction", merge.name)
                            break
                stats[reason] += 1
                stats["both_observed_at_a_shared_gate"] += shared
                checkpoint[reason] += 1
                if scene is not None:
                    prepared.append((row["candidate_id"], scene, metadata))
                writer.writerow({**{k: row[k] for k in fields[:4]}, "previous_outcome": row["outcome"],
                                 "previous_reason": row["reason"], "shared_gate_observed": int(shared),
                                 "reason": reason, **metadata})
            with (DEST / f"scenes_{Path(file).stem}.pkl").open("wb") as target:
                pickle.dump(prepared, target)
            print(file, dict(checkpoint), flush=True)
    config = {"screening": dict(stats), "candidates": sum(len(rows) for rows in recordings.values()),
              "finish_distance_after_taper_m": GATE_AFTER_M, "approach_distance_m": APPROACH_M,
              "gates": [{"direction": m.name, "taper_lanelet": m.taper_lanelet, "receiving_lanelet": m.receiving_lanelet,
                         "taper_tip_xy_m": m.tip.tolist(), "gate_endpoints_xy_m": m.gate.tolist()} for m in merges]}
    (DEST / "gates.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(config, indent=2), flush=True)


def run_case(scene, policy):
    original_offers = trial.offers
    cache = {}

    def cached_offers(current, vehicle, state, candidates, speed_limit):
        key = (vehicle, *state, candidates.tobytes(), speed_limit)
        if key not in cache:
            cache[key] = original_offers(current, vehicle, state, candidates, speed_limit)
        return cache[key]

    trial.offers = cached_offers
    try:
        for mode, method in ARMS:
            stalled = False
            try:
                if method == "human":
                    times, progress, decisions = scene.elapsed, scene.measured, []
                elif method == "fixed_release":
                    times, progress, decisions = trial.simulate_fixed_release(
                        scene, evaluation_limit=None, speed_limit=SPEED_LIMIT)
                else:
                    speed_range = (0.0, SPEED_LIMIT) if mode == "continuous" else None
                    times, progress, decisions = trial.simulate(
                        scene, method, evaluation_limit=None, speed_limit=SPEED_LIMIT,
                        continuous_speed_range=speed_range, pair_policy=policy)
            except trial.SimulationStalled as error:
                stalled = True
                times, progress, decisions = error.times, error.progress, error.decisions
            result = trial.assess(scene, times, progress, evaluation_limit=None)
            assert result["complete"] != stalled, (scene.case_id, mode, method)
            if method != "human":
                speeds = np.diff(progress, axis=0) / np.diff(times)[:, None]
                speed_envelope = np.maximum(SPEED_LIMIT, scene.initial_speed[None, :] - trial.DECEL * times[:-1, None])
                assert np.all(speeds <= speed_envelope + 1e-7)
            yield {"mode": mode, "method": method, "stalled": stalled, **result,
                   "decision_count": len(decisions),
                   "braking_fallback_count": sum(d["offers"] is None for d in decisions)}
    finally:
        trial.offers = original_offers


def run_recording(recording):
    with (DEST / f"scenes_{recording}.pkl").open("rb") as stream:
        scenes = pickle.load(stream)
    policy = MergePolicy()
    target = DEST / f"trial_{recording}.jsonl"
    with target.with_suffix(".tmp").open("w", encoding="utf-8") as stream:
        for i, (candidate, scene, metadata) in enumerate(scenes, 1):
            for result in run_case(scene, policy):
                row = {"candidate_id": candidate, "case_id": scene.case_id, "source": scene.source,
                       **metadata, **result}
                stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            if i % 100 == 0:
                print(recording, i, "of", len(scenes), flush=True)
    target.with_suffix(".tmp").replace(target)
    print(recording, len(scenes), "complete", flush=True)


def aggregate(rows):
    arms = defaultdict(dict)
    for row in rows:
        index = (row["mode"], row["method"])
        key = row["candidate_id"]
        assert key not in arms[index]
        arms[index][key] = row
    overview = {}
    for (mode, method), episodes in arms.items():
        times = [r["joint_finish_s"] for r in episodes.values() if r["complete"]]
        overview[f"{mode}/{method}"] = {
            "cases": len(episodes),
            **{k: sum(r[k] for r in episodes.values()) for k in ("complete", "stalled", "near_miss", "collision")},
            "mean_finish_s": float(np.mean(times)) if times else None,
            "p95_finish_s": float(np.quantile(times, 0.95)) if times else None,
        }
    paired = {}
    for first in (("continuous", "ttc_bargaining"), ("discrete", "ttc_bargaining")):
        for other in ARMS:
            if other == first:
                continue
            a, b = arms[first], arms[other]
            assert a.keys() == b.keys()
            differences = [r["joint_finish_s"] - b[key]["joint_finish_s"] for key, r in a.items()
                           if r["complete"] and b[key]["complete"]]
            paired[f"{'/'.join(first)} minus {'/'.join(other)}"] = {
                "jointly_completed": len(differences),
                "mean_paired_finish_difference_s": float(np.mean(differences)) if differences else None,
                "near_miss_count_difference": sum(int(r["near_miss"]) - int(b[key]["near_miss"]) for key, r in a.items()),
                "collision_count_difference": sum(int(r["collision"]) - int(b[key]["collision"]) for key, r in a.items()),
            }
    return {"overview": overview, "paired": paired}


def combine():
    with (DEST / "screening.csv").open(encoding="utf-8-sig", newline="") as stream:
        screening = list(csv.DictReader(stream))
    by_id = {r["candidate_id"]: r for r in screening}
    assert len(screening) == len(by_id) == 45676
    rows = []
    for path in sorted({Path(row["source"]).stem for row in screening}):
        with (DEST / f"trial_{path}.jsonl").open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream)
    included = {r["candidate_id"] for r in screening if r["reason"] == "eligible"}
    previous = {r["candidate_id"] for r in screening if r["previous_outcome"] == "evaluated"}
    expected = {(key, *arm) for key in included for arm in ARMS}
    assert len(rows) == len(expected)
    assert {(r["candidate_id"], r["mode"], r["method"]) for r in rows} == expected
    with (DEST / "episodes.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, [*rows[0], "previous_outcome", "previous_reason"])
        writer.writeheader()
        for row in rows:
            screen_row = by_id[row["candidate_id"]]
            writer.writerow({**row, **{k: screen_row[k] for k in ("previous_outcome", "previous_reason")}})
    result = json.loads((DEST / "gates.json").read_bytes())
    result.update(evaluated=len(included), evaluations=len(rows), speed_limit_mps=SPEED_LIMIT,
                  forecast_geometry_step_s=trial.DT * FORECAST_STRIDE, scoring_step_s=trial.EVALUATION_STEP,
                  near_miss_m=trial.NEAR_MISS_M, replanning_s=trial.DT * trial.DEFAULT_BARGAINING_STEPS,
                  previously_eligible=len(previous), previously_eligible_also_included=len(included & previous),
                  newly_admitted=len(included - previous),
                  newly_admitted_by_previous_reason=dict(Counter(by_id[k]["previous_reason"] for k in included - previous)),
                  eligible_by_direction=dict(Counter(by_id[k]["direction"] for k in included)),
                  all=aggregate(rows),
                  by_direction={d: aggregate([r for r in rows if r["direction"] == d])
                                for d in sorted({r["direction"] for r in rows})})
    (DEST / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("all", "by_direction")}, indent=2))
    print(json.dumps(result["all"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("screen", "run", "combine"), default="screen")
    parser.add_argument("--recording", help="File stem, such as vehicle_tracks_000")
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    if args.stage == "screen":
        screen()
    elif args.stage == "run":
        if args.recording is None:
            parser.error("--stage run requires --recording")
        run_recording(args.recording)
    else:
        combine()
