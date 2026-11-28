"""Run and combine the full fixed-path common-window comparison."""

import argparse
from collections import Counter, defaultdict
import csv
import ctypes
from functools import lru_cache
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fixed_path_trial_scene import Route
from traffic_data.fixed_path.audit_swept_footprints import swept_strips


RESULTS = ROOT / "Result"
SOURCE = RESULTS / "fixed_path_common_windows"
DEST = RESULTS / "common_window_ttc_15mps"
BLOCK = 250
ARMS = ("continuous_ttc", "discrete_ttc", "continuous_fixed_discount",
        "discrete_fixed_discount", "two_layer_mpc", "human", "fixed_release_5m")
META = ("scene_id", "candidate_id", "dataset", "site", "road_type", "raw_window_s",
        "effective_window_s", "tail_a_m", "tail_b_m", "truncated",
        "observed_min_after_effective_window", "verified_completion_points", "fixed_release_applicable")
FIELDS = ("min_clearance_m", "progress_a_m", "progress_b_m", "new_contact", "near_miss", "initial_contact",
          "any_contact", "decision_count", "fallback_count", "continuous_trials",
          "finish_a_s", "finish_b_s", "joint_finish_s")


@lru_cache(maxsize=1)
def kernel():
    library = ctypes.CDLL(str(DEST / "common_window_ttc_kernel.dll"))
    arr = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    integer = ctypes.c_int
    evaluate = library.evaluate_window
    evaluate.argtypes = [arr, integer, arr, integer, arr, arr, arr, arr, arr, arr, integer, arr, arr, arr]
    evaluate.restype = integer
    score = library.score_window
    score.argtypes = [arr, integer, arr, integer, arr, arr, arr, integer, ctypes.c_double, arr]
    score.restype = None
    return evaluate, score


@lru_cache(maxsize=1)
def previous_coverage():
    old = RESULTS / "fixed_path_trial_1000" / "full_candidate_sweep_15mps" / "screening.csv"
    with old.open(encoding="utf-8-sig", newline="") as stream:
        release_keys = {r["candidate_id"] for r in csv.DictReader(stream) if r["outcome"] == "evaluated"}
    verified = {}
    merge = RESULTS / "fixed_path_merge_gate_zs_15mps"
    for file in sorted(merge.glob("scenes_*.pkl")):
        with file.open("rb") as stream:
            for key, scene, _ in pickle.load(stream):
                verified[key] = [np.array([np.interp(goal, route.progress, route.points[:, i]) for i in (0, 1)])
                                 for route, goal in zip(scene.routes, scene.goal)]
    return release_keys, verified


def observed_region(routes, obs_progress, distance):
    """Locate the potential interaction along each route's observed portion."""
    observed_routes = [route.points[route.progress <= t+1e-8] for route, t in zip(routes, obs_progress)]
    strips = [swept_strips(points, *route.size) if len(points) >= 2 else route.polygons([0])
              for route, points in zip(routes, observed_routes)]
    entries, exits = [], []
    for i, (route, limit) in enumerate(zip(routes, obs_progress)):
        # The 0.5 m search bins only gate waits and proposer order. The forecast and
        # outcome checks use interpolated positions, not these bins.
        grid = np.linspace(0, limit, max(2, int(np.ceil(limit/.5)) + 1))
        polys = route.polygons(grid)
        tree = shapely.STRtree(strips[1-i])
        cells = tree.query(polys, predicate="dwithin", distance=distance)[0]
        if len(cells):
            entries.append(max(0., grid[cells.min()] - .25))
            exits.append(min(limit, grid[cells.max()] + .25))
        elif distance:
            # A negative control still receives a time-ordered offer if needed.
            nearest, gaps = tree.query_nearest(polys, return_distance=True, all_matches=False)
            t = grid[nearest[0, np.argmin(gaps)]]
            entries.append(t)
            exits.append(t)
        else:
            return None, None
    return np.array(entries), np.array(exits)


def case_inputs(row, arrays, tails, priors):
    i = int(row["array_row"])
    lo, hi = arrays["offsets"][i:i+2]
    clock = np.ascontiguousarray(arrays["t_s"][lo:hi], dtype=np.float64)
    xy = arrays["xy_m"][lo:hi].astype(np.float64)
    sizes = arrays["body_size_m"][i].astype(np.float64)
    velocity = arrays["velocity_mps"][lo].astype(np.float64)
    paths, observed, ext = [], [], []
    for car in (0, 1):
        q = 2*i + car
        begin, end = tails["offsets"][q:q+2]
        path = np.concatenate((xy[:, car], tails["xy_m"][begin:end]))
        route, progress = Route.build(path, sizes[car])
        paths.append(route)
        observed.append(progress[:len(clock)])
        ext.append(float(route.progress[-1]-progress[len(clock)-1]))
    progress = np.ascontiguousarray(np.column_stack(observed), dtype=np.float64)
    directions = np.array([route.angles[0] for route in paths])
    initial = np.maximum(0., velocity[:, 0]*np.cos(directions) + velocity[:, 1]*np.sin(directions))
    entry, leave = observed_region(paths, progress[-1], 1.)
    raw_key = row["candidate_id"]
    release = np.full(4, np.inf)
    if raw_key in priors[0]:
        starts, stops = observed_region(paths, progress[-1], 0.)
        if starts is not None and np.all(starts > .25) and np.all(stops + 2 < progress[-1]):
            priority = int(np.argmin(starts/np.maximum(.5, initial)))
            if (starts[priority]+stops[priority])/2 + 5 < paths[priority].progress[-1]:
                release = np.r_[starts, stops]
    goals = np.full(2, np.inf)
    if points := priors[1].get(raw_key):
        local_goals = np.asarray(points) - arrays["origin_xy_m"][i]
        stations = []
        for car in (0, 1):
            line = shapely.LineString(xy[:, car])
            point = shapely.Point(local_goals[car])
            station = line.project(point)
            if line.distance(point) > .02 or not .02 < station < progress[-1, car]:
                break
            stations.append(station)
        if len(stations) == 2:
            goals[:] = stations
    packed = [np.ascontiguousarray(np.column_stack((r.progress, r.points, r.angles))) for r in paths]
    return packed, entry, leave, sizes, initial, clock, progress, release, goals, ext


def evaluate_case(row, arrays, tails, priors):
    paths, entry, leave, sizes, initial, clock, progress, release, goals, ext = case_inputs(row, arrays, tails, priors)
    out = np.full(1+13*len(ARMS), np.nan)
    a, b = paths
    run, _ = kernel()
    if run(a, len(a), b, len(b), entry, leave, sizes, initial, clock, progress, len(clock), release, goals, out):
        raise ValueError(f"Invalid packed trajectory for {row['scene_id']}")
    window = float(out[0])
    assert 0 < window <= clock[-1]+1e-9 and np.isfinite(out[1:11]).all(), row["scene_id"]
    metadata = {
        **{name: row[name] for name in META[:5]},
        "raw_window_s": float(clock[-1]), "effective_window_s": window,
        "tail_a_m": ext[0], "tail_b_m": ext[1], "truncated": int(window < clock[-1]-1e-9),
        "observed_min_after_effective_window": int(float(row["sampled_min_gap_t_s"]) > window+1e-9),
        "verified_completion_points": int(np.isfinite(goals).all()),
        "fixed_release_applicable": int(np.isfinite(release[0])),
    }
    for k, arm in enumerate(ARMS):
        if arm == ARMS[-1] and not metadata["fixed_release_applicable"]:
            continue
        data = out[1+13*k:1+13*(k+1)]
        assert np.isfinite(data[:10]).all() and not (data[4] and data[6]), (row["scene_id"], arm)
        values = {name: float(value) if np.isfinite(value) else None for name, value in zip(FIELDS, data)}
        for name in FIELDS[3:10]:
            values[name] = int(values[name])
        yield {**metadata, "arm": arm, **values}


def read_arrays(folder, name):
    with np.load(folder / name, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def run_shard(shard, start=0, stop=None):
    with (SOURCE / f"index_{shard:02}.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert all(r["status"] == "aligned" for r in rows)
    priors = previous_coverage()
    arrays = read_arrays(SOURCE, f"windows_{shard:02}.npz")
    tails = read_arrays(DEST, f"tails_{shard:02}.npz")
    stop = len(rows) if stop is None else min(stop, len(rows))
    for beginning in range(start, stop, BLOCK):
        finish = min(beginning+BLOCK, stop)
        path = DEST / f"episodes_{shard:02}_{beginning:05}_{finish:05}.csv"
        if path.exists():
            continue
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, (*META, "arm", *FIELDS))
            writer.writeheader()
            for r in rows[beginning:finish]:
                writer.writerows(evaluate_case(r, arrays, tails, priors))
        temporary.replace(path)
        if (beginning//BLOCK) % 4 == 0 or finish == stop:
            print(shard, beginning, finish, len(rows), flush=True)


def aggregate():
    class Totals:
        def __init__(self):
            self.n = Counter()
            self.distance = self.speed = self.time = self.joint_finish = 0.

        def add(self, row):
            self.n["episodes"] += 1
            for name in ("new_contact", "near_miss", "initial_contact", "any_contact"):
                self.n[name] += int(row[name])
            self.n["fallback_steps"] += int(row["fallback_count"])
            self.n["continuous_trials"] += int(row["continuous_trials"])
            self.n["verified_completion_points"] += int(row["verified_completion_points"])
            horizon = float(row["effective_window_s"])
            travelled = (float(row["progress_a_m"])+float(row["progress_b_m"]))/2
            self.distance += travelled
            self.speed += travelled/horizon
            self.time += horizon
            if row["joint_finish_s"]:
                self.n["completed_within_window"] += 1
                self.joint_finish += float(row["joint_finish_s"])

        def summary(self):
            n = self.n["episodes"]
            completed = self.n["completed_within_window"]
            return {
                **self.n, "near_miss_rate": self.n["near_miss"]/n, "collision_rate": self.n["new_contact"]/n,
                "any_contact_rate": self.n["any_contact"]/n, "mean_progress_m_per_car": self.distance/n,
                "mean_progress_rate_mps_per_car": self.speed/n, "mean_effective_window_s": self.time/n,
                "completed_within_window": completed,
                "mean_joint_finish_s_when_completed": self.joint_finish/completed if completed else None,
            }

    counts = Counter()
    all_arms, subset = defaultdict(Totals), defaultdict(Totals)
    by_dataset = defaultdict(lambda: defaultdict(Totals))
    by_road = defaultdict(lambda: defaultdict(Totals))
    ids = defaultdict(set)
    horizons = []
    paired_horizons = {}
    completion_times = defaultdict(dict)
    with (SOURCE / "windows.csv").open(encoding="utf-8-sig", newline="") as raw:
        original = {r["scene_id"]: r["observed_geometry_class"] for r in csv.DictReader(raw)}
    output = DEST / "episodes.csv"
    with output.with_suffix(".tmp").open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, (*META, "arm", *FIELDS))
        writer.writeheader()
        for path in sorted(DEST.glob("episodes_??_?????_?????.csv")):
            with path.open(encoding="utf-8", newline="") as stream:
                for r in csv.DictReader(stream):
                    key, arm = r["scene_id"], r["arm"]
                    assert key not in ids[arm], (key, arm)
                    assert paired_horizons.setdefault(key, r["effective_window_s"]) == r["effective_window_s"], key
                    ids[arm].add(key)
                    all_arms[arm].add(r)
                    by_dataset[r["dataset"]][arm].add(r)
                    by_road[r["road_type"]][arm].add(r)
                    if int(r["fixed_release_applicable"]):
                        subset[arm].add(r)
                    if r["joint_finish_s"]:
                        completion_times[arm][key] = float(r["joint_finish_s"])
                    if arm == "human":
                        horizons.append(float(r["effective_window_s"]))
                        for name in ("truncated", "observed_min_after_effective_window", "fixed_release_applicable",
                                     "verified_completion_points"):
                            counts[name] += int(r[name])
                        counts["either_path_extended"] += (float(r["tail_a_m"]) + float(r["tail_b_m"]) > .001)
                        counts["both_paths_extended"] += min(float(r["tail_a_m"]),float(r["tail_b_m"])) > .001
                        counts["recorded_close_minimum_outside_window"] += (
                            original[key] in ("near_miss_sampled", "later_contact") and
                            int(r["observed_min_after_effective_window"]))
                    writer.writerow(r)
    assert len(ids["human"]) == 125_932 and ids["human"] == original.keys()
    assert all(ids[arm] == ids["human"] for arm in ARMS[:-1])
    assert len(ids[ARMS[-1]]) == counts["fixed_release_applicable"]
    assert all(arm.n["episodes"] == counts["fixed_release_applicable"] for arm in subset.values())
    output.with_suffix(".tmp").replace(output)
    summarize = lambda group: {arm: stats.summary() for arm, stats in group.items()}
    joint_time_differences = {}
    for comparison in ("discrete_ttc", "continuous_fixed_discount", "discrete_fixed_discount", "two_layer_mpc", "human"):
        first, other = completion_times["continuous_ttc"], completion_times[comparison]
        keys = first.keys() & other.keys()
        joint_time_differences[comparison] = {
            "cases": len(keys),
            "mean_continuous_ttc_minus_comparison_s": sum(first[k]-other[k] for k in keys)/len(keys) if keys else None,
        }
    result = {
        "scenes": len(horizons), "global_evaluations": len(horizons)*len(ARMS[:-1]),
        "fixed_release_additional_evaluations": counts["fixed_release_applicable"], **counts,
        "window_s": {str(q): float(np.quantile(horizons, q)) for q in (0,.1,.25,.5,.75,.9,1)},
        "all": summarize(all_arms), "fixed_release_matched_subset": summarize(subset),
        "continuous_ttc_finish_time_paired": joint_time_differences,
        "by_dataset": {k: summarize(group) for k, group in by_dataset.items()},
        "by_road_type": {k: summarize(group) for k, group in by_road.items()},
        "settings": {
            "speed_limit_mps": 15, "replanning_s": .4, "safety_scoring_s": .02,
            "forecast_geometry_s": .2, "near_miss_m": 1,
            "continuous_search": "Bounded coordinate refinement of continuous target speeds and acceleration fractions",
            "collision_rate": "New rectangle contact after start, including recontact after separation",
            "near_miss_rate": "Positive clearance at most 1m and no rectangle contact in the window",
            "effective_window": "Original common recording, shortened if either vehicle could physically reach its extended route end first",
            "efficiency": "Observed arc-length progress for each vehicle over its matched window",
            "background": "Two-vehicle fixed-path comparison without other road users or signal-phase enforcement",
        },
    }
    (DEST / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for key in ("scenes", "global_evaluations", "fixed_release_additional_evaluations", "all", "window_s"):
        print(key, result[key], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "combine"))
    parser.add_argument("--shard", type=int, choices=range(8))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    args = parser.parse_args()
    if args.stage == "run":
        if args.shard is None:
            parser.error("run requires --shard")
        run_shard(args.shard, args.start, args.stop)
    else:
        aggregate()
