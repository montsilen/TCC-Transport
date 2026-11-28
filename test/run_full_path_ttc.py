"""Evaluate each pair to the recorded endpoints, releasing vehicles on first arrival."""

import argparse
from collections import Counter, defaultdict
import csv
import ctypes
from functools import lru_cache
import json

import numpy as np

from run_common_window_ttc import ARMS, BLOCK, FIELDS, RESULTS, SOURCE, Route, observed_region, read_arrays


DEST = RESULTS / "full_path_ttc_15mps"
WITHOUT_SHIELD = RESULTS / "full_path_ttc_no_shield_15mps"
UNFILTERED = False
PRIOR = RESULTS / "common_window_ttc_15mps" / "episodes.csv"
METADATA = ("scene_id", "candidate_id", "dataset", "site", "road_type", "raw_window_s",
            "path_a_m", "path_b_m", "fixed_release_applicable")
MEASURED = (*FIELDS[:1], *FIELDS[3:], "safety_exposure_s", "elapsed_s", "stalled")
OUTPUT_FIELDS = (*METADATA, "arm", *MEASURED, "complete", "safe_complete")
SCENES = 125_932
RELEASE_SCENES = 6_107


def initialize(without_fixed_release=False):
    DEST.mkdir(parents=True, exist_ok=True)
    if without_fixed_release:
        eligible = set()
    elif UNFILTERED:
        eligible = set(json.loads((RESULTS / "full_path_ttc_15mps" / "fixed_release_ids.json").read_bytes()))
    else:
        with PRIOR.open(encoding="utf-8-sig", newline="") as stream:
            eligible = {r["scene_id"] for r in csv.DictReader(stream) if r["arm"] == ARMS[-1]}
    assert len(eligible) in (0, RELEASE_SCENES)
    manifest = DEST / "fixed_release_ids.json"
    if manifest.exists() and any(DEST.glob("episodes_*.csv")):
        if eligible != set(json.loads(manifest.read_bytes())):
            raise ValueError(f"Fixed-release membership differs from the existing run in {DEST}")
    manifest.write_text(json.dumps(sorted(eligible)) + "\n", encoding="utf-8")
    print("Selected fixed-release pairs:", len(eligible), flush=True)


@lru_cache(maxsize=1)
def kernel():
    library = ctypes.CDLL(str(DEST / "full_path_ttc_kernel.dll"))
    array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    integer = ctypes.c_int
    evaluate = library.evaluate_unfiltered_endpoints if UNFILTERED else library.evaluate_endpoints
    evaluate.argtypes = [array, integer, array, integer, array, array, array, array, array, array, integer, array, array]
    evaluate.restype = integer
    score = library.score_endpoints
    score.argtypes = [array, integer, array, integer, array, array, array, integer, array]
    score.restype = None
    return evaluate, score


def output_fields():
    return (*OUTPUT_FIELDS, "collision_time_s") if UNFILTERED else OUTPUT_FIELDS


def inputs(row, arrays, include_release):
    i = int(row["array_row"])
    lo, hi = arrays["offsets"][i:i+2]
    times = np.ascontiguousarray(arrays["t_s"][lo:hi], dtype=np.float64)
    xy = arrays["xy_m"][lo:hi].astype(np.float64)
    sizes = arrays["body_size_m"][i].astype(np.float64)
    velocity = arrays["velocity_mps"][lo].astype(np.float64)
    geometry = [Route.build(xy[:, j], sizes[j]) for j in (0, 1)]
    routes = [item[0] for item in geometry]
    progress = np.ascontiguousarray(np.column_stack([item[1] for item in geometry]))
    directions = np.array([route.angles[0] for route in routes])
    initial = np.maximum(0., velocity[:, 0]*np.cos(directions) + velocity[:, 1]*np.sin(directions))
    entry, leave = observed_region(routes, progress[-1], 1.)
    release = np.full(4, np.inf)
    if include_release:
        starts, stops = observed_region(routes, progress[-1], 0.)
        assert starts is not None and np.all(starts > .25) and np.all(stops + 2 < progress[-1]), row["scene_id"]
        release[:] = np.r_[starts, stops]
    packed = [np.ascontiguousarray(np.column_stack((route.progress, route.points, route.angles))) for route in routes]
    return packed, entry, leave, sizes, initial, times, progress, release


def evaluate(row, arrays, eligible):
    include_release = row["scene_id"] in eligible
    (a, b), entry, leave, sizes, initial, times, progress, release = inputs(row, arrays, include_release)
    output = np.full((len(ARMS), 17 if UNFILTERED else 16), np.nan)
    simulate, _ = kernel()
    if simulate(a, len(a), b, len(b), entry, leave, sizes, initial, times, progress, len(times), release, output):
        raise ValueError(f"Invalid input paths for {row['scene_id']}")
    meta = {**{key: row[key] for key in METADATA[:5]},
            "raw_window_s": float(times[-1]), "path_a_m": float(a[-1, 0]), "path_b_m": float(b[-1, 0]),
            "fixed_release_applicable": int(include_release)}
    for arm, data in zip(ARMS[:6+include_release], output):
        assert np.isfinite(data[:10]).all() and np.isfinite(data[13:16]).all(), (row["scene_id"], arm)
        assert not (data[4] and data[6]), row["scene_id"]
        complete = bool(np.isfinite(data[10:12]).all())
        assert complete == bool(np.isfinite(data[12])), (row["scene_id"], arm)
        assert not complete or (data[12] == max(data[10:12]) == data[14]), (row["scene_id"], arm)
        assert data[13] == min(data[10], data[11], data[14]), (row["scene_id"], arm)
        assert not (complete and data[15]), (row["scene_id"], arm)
        if UNFILTERED:
            contact = bool(data[6])
            assert np.isfinite(data[16]) == contact
            assert not contact or (not complete and data[13] == data[14] == data[16])
            assert not complete or not contact
            if arm in ARMS[:4]:
                assert data[8] == 0 and bool(complete) != contact
        assert arm != "human" or ((complete or (UNFILTERED and data[6])) and data[14] <= times[-1]+1e-9)
        values = {key: float(value) if np.isfinite(value) else None
                  for key, value in zip(MEASURED, np.r_[data[0], data[3:16]])}
        for key in (*FIELDS[3:10], "stalled"):
            values[key] = int(values[key])
        if UNFILTERED:
            values["collision_time_s"] = float(data[16]) if np.isfinite(data[16]) else None
        yield {**meta, "arm": arm, **values, "complete": int(complete),
               "safe_complete": int(complete and not data[6])}


def run_shard(shard):
    with (SOURCE / f"index_{shard:02}.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert all(r["status"] == "aligned" for r in rows)
    eligible = set(json.loads((DEST / "fixed_release_ids.json").read_bytes()))
    arrays = read_arrays(SOURCE, f"windows_{shard:02}.npz")
    for start in range(0, len(rows), BLOCK):
        stop = min(start+BLOCK, len(rows))
        path = DEST / f"episodes_{shard:02}_{start:05}_{stop:05}.csv"
        if path.exists():
            continue
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, output_fields())
            writer.writeheader()
            for row in rows[start:stop]:
                writer.writerows(evaluate(row, arrays, eligible))
        temporary.replace(path)
        if (start//BLOCK) % 4 == 0 or stop == len(rows):
            print(shard, "completed", stop, "of", len(rows), flush=True)


class Stats:
    def __init__(self):
        self.count = Counter()
        self.times = defaultdict(list)
        self.seconds = 0.

    def add(self, row):
        self.count["scenes"] += 1
        for key in ("complete", "safe_complete", "stalled", "new_contact", "near_miss", "initial_contact", "any_contact"):
            self.count[key] += int(row[key])
        self.seconds += float(row["safety_exposure_s"])
        for key in ("finish_a_s", "finish_b_s", "joint_finish_s"):
            if row[key] != "":
                self.times[key].append(float(row[key]))

    def summary(self):
        n = self.count["scenes"]
        joint = self.times["joint_finish_s"]
        return {
            **self.count, "completion_rate": self.count["complete"]/n,
            "safe_completion_rate": self.count["safe_complete"]/n,
            "near_miss_rate": self.count["near_miss"]/n,
            "collision_rate": self.count["any_contact" if UNFILTERED else "new_contact"]/n,
            "any_contact_rate": self.count["any_contact"]/n,
            "mean_safety_exposure_s": self.seconds/n,
            "completion_a": len(self.times["finish_a_s"]), "completion_b": len(self.times["finish_b_s"]),
            **{f"mean_{key}": float(np.mean(value)) if value else None for key, value in self.times.items()},
            "p50_joint_finish_s": float(np.median(joint)) if joint else None,
            "p95_joint_finish_s": float(np.quantile(joint, .95)) if joint else None,
        }


def aggregate():
    with (SOURCE / "windows.csv").open(encoding="utf-8-sig", newline="") as stream:
        expected = {r["scene_id"] for r in csv.DictReader(stream)}
    eligible = set(json.loads((DEST / "fixed_release_ids.json").read_bytes()))
    by_method, matched = defaultdict(Stats), defaultdict(Stats)
    by_dataset = defaultdict(lambda: defaultdict(Stats))
    by_road_type = defaultdict(lambda: defaultdict(Stats))
    paired = defaultdict(list)
    seen = set()
    output = DEST / "episodes.csv"

    def consume(group):
        if not group:
            return
        scene = group[0]["scene_id"]
        assert scene not in seen and scene in expected, scene
        seen.add(scene)
        assert [r["arm"] for r in group] == list(ARMS[:6+(scene in eligible)]), scene
        assert len({r["initial_contact"] for r in group}) == 1, scene
        if UNFILTERED:
            for i, j in ((0, 2), (1, 3)):
                assert all(group[i][key] == group[j][key] for key in output_fields() if key != "arm"), scene
        continuous_time = group[0]["joint_finish_s"]
        for row in group:
            arm = row["arm"]
            by_method[arm].add(row)
            by_dataset[row["dataset"]][arm].add(row)
            by_road_type[row["road_type"]][arm].add(row)
            if scene in eligible:
                matched[arm].add(row)
            if arm != ARMS[0] and continuous_time != "" and row["joint_finish_s"] != "":
                paired[arm].append(float(continuous_time)-float(row["joint_finish_s"]))

    with output.with_suffix(".tmp").open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, output_fields())
        writer.writeheader()
        for block in sorted(DEST.glob("episodes_??_?????_?????.csv")):
            group = []
            with block.open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    if group and group[0]["scene_id"] != row["scene_id"]:
                        consume(group)
                        group = []
                    group.append(row)
                    writer.writerow(row)
            consume(group)
    assert len(seen) == SCENES and seen == expected
    assert len(eligible) in (0, RELEASE_SCENES)
    assert set(matched) == (set(ARMS) if eligible else set())
    assert all(s.count["scenes"] == len(eligible) for s in matched.values())
    human = by_method["human"].count
    assert human["complete"] + (human["any_contact"] if UNFILTERED else 0) == SCENES
    output.with_suffix(".tmp").replace(output)
    def arm_summary(group):
        return {arm: group[arm].summary() for arm in ARMS if arm in group}
    result = {
        "unique_pairs": len(seen), "total_evaluations": sum(s.count["scenes"] for s in by_method.values()),
        "all_pairs": {arm: by_method[arm].summary() for arm in ARMS[:-1]},
        "fixed_release_matched_subset": arm_summary(matched),
        "continuous_ttc_joint_finish_time_paired": {
            arm: {"joint_completions": len(t), "mean_difference_s": float(np.mean(t)),
                  "median_difference_s": float(np.median(t))} for arm, t in paired.items()
        },
        "by_dataset": {key: arm_summary(group) for key, group in by_dataset.items()},
        "by_road_type": {key: arm_summary(group) for key, group in by_road_type.items()},
        "settings": {
            "task_goal": "Each observed route's own final position; earlier data window is not a simulation deadline.",
            "arrival": "First interpolation crossing of each route's terminal arc length.",
            "safety_exposure": "Until the first terminal arrival, or until the repeated-state stop if neither arrives.",
            "collision_rate": "New rectangular-body contact after the common start; any_contact_rate also includes initial contact.",
            "near_miss_rate": "Minimum body clearance at most 1 m with no contact during the safety exposure.",
            "completion": "Both terminal arrival times are finite. Safe completion additionally requires no contact.",
            "turn_limit": "End at joint completion or an unchanged pair of positions and velocities over a full deterministic control update.",
            "speed_limit_mps": 15, "discrete_targets_mps": [7, 10, 13], "continuous_target_bounds_mps": [0, 15],
            "replan_s": .4, "forecast_geometry_s": .2, "scoring_s": .02,
            "scenario_scope": "Fixed-path, two-vehicle geometry; additional traffic participants and signal phases are outside the simulator.",
        },
    }
    if UNFILTERED:
        result["settings"].update({
            "safety_exposure": "Through first contact, first terminal arrival, or a stationary-state stop, whichever comes first.",
            "collision_rate": "Any detected rectangle contact before or at the first terminal arrival; initial contact counts as failure.",
            "completion": "Both terminal arrivals with no contact. Arrival times are recorded only for successful episodes.",
            "turn_limit": "End scoring at first contact, successful joint completion, or a repeated unchanged state.",
            "bargaining": "No joint safety filter or emergency fallback. Own travel time is each vehicle's utility; the unrestricted fastest joint offer is used.",
            "reference_controllers": "MPC and fixed-release retain the policies of the earlier endpoint run.",
        })
    (DEST / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("unique_pairs", "total_evaluations", "all_pairs",
                                          "continuous_ttc_joint_finish_time_paired")}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("init", "run", "combine"))
    parser.add_argument("--shard", type=int, choices=range(8))
    parser.add_argument("--unfiltered", action="store_true", help="Remove the bargaining safety shield and mark first contact as failure.")
    parser.add_argument("--without-fixed-release", action="store_true",
                        help="Initialize a six-method run without the archived fixed-release cohort.")
    args = parser.parse_args()
    if args.without_fixed_release and args.stage != "init":
        parser.error("--without-fixed-release is only used during init")
    if args.unfiltered:
        UNFILTERED = True
        DEST = WITHOUT_SHIELD
    if args.stage == "init":
        initialize(args.without_fixed_release)
    elif args.stage == "run":
        if args.shard is None:
            parser.error("run requires --shard")
        run_shard(args.shard)
    else:
        aggregate()
