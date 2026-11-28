"""Diagnose the incomplete TTC episodes from the endpoint experiment."""

from collections import Counter, defaultdict
import csv
import ctypes
import json

import numpy as np

from run_full_path_ttc import DEST as EXPERIMENT, SOURCE, inputs

DEST = EXPERIMENT / "stall_diagnostics"
ARMS = ("continuous_ttc", "discrete_ttc")
TRACE_FIELDS = (
    "time_s", "step_s", "fallback", "enumerated_feasible_pairs",
    "start_a_m", "start_a_mps", "start_b_m", "start_b_mps", "end_a_m", "end_a_mps", "end_b_m", "end_b_mps",
    "wait_a_s", "target_a_mps", "ramp_a", "wait_b_s", "target_b_mps", "ramp_b",
)
BASE_PROBES = ("tested", "safe_0p4s", "safe_12s", "moving_first_update", "one_moves_first_update", "can_exit_in_12s")
PROBE_FIELDS = ("entry_a_m", "entry_b_m", "path_a_m", "path_b_m", "initial_gap_m", "terminal_gap_m", "terminal_ttc_s",
                *(f"original_{name}" for name in BASE_PROBES), *(f"add_stop_{name}" for name in BASE_PROBES),
                "a_can_finish_with_b_parked", "b_can_finish_with_a_parked",
                "finer_grid_safe_moving_pairs", "finer_grid_safe_moving_pairs_with_one_arrival")


def native():
    library = ctypes.CDLL(str(DEST / "stall_diagnostics.dll"))
    diagnose = library.diagnose_stall
    a = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    i = ctypes.c_int
    diagnose.argtypes = [a, i, a, i, a, a, a, a, i, a, a, i, a]
    diagnose.restype = i
    return diagnose


def published_stalls():
    stalls = defaultdict(dict)
    stalled_sets = defaultdict(set)
    with (EXPERIMENT / "episodes.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["stalled"] == "1":
                stalled_sets[row["arm"]].add(row["scene_id"])
                if row["arm"] in ARMS:
                    stalls[row["scene_id"]][row["arm"]] = row
    assert len(stalled_sets[ARMS[0]]) == 123 and len(stalled_sets[ARMS[1]]) == 221
    return stalls, stalled_sets


def number(value):
    return float(value) if np.isfinite(value) else None


def reload_stall(diagnose, reference, row, arrays):
    (a, b), entry, leave, sizes, initial, _, _, _ = inputs(row, arrays, False)
    metrics = np.empty(16)
    trace = np.empty((8192, len(TRACE_FIELDS)))
    terminal = np.empty(len(PROBE_FIELDS))
    arm = reference["arm"]
    count = diagnose(a, len(a), b, len(b), entry, leave, sizes, initial, ARMS.index(arm),
                     metrics, trace, len(trace), terminal)
    assert count > 0 and metrics[15] == 1, (reference["scene_id"], arm, count)
    assert count == int(metrics[7]) == int(reference["decision_count"])
    for name, i in (("min_clearance_m", 0), ("new_contact", 3), ("near_miss", 4), ("initial_contact", 5),
                    ("any_contact", 6), ("fallback_count", 8), ("continuous_trials", 9), ("elapsed_s", 14)):
        assert abs(float(reference[name])-metrics[i]) < 1e-9, (reference["scene_id"], arm, name)
    trace = trace[:count]
    assert np.array_equal(trace[-1,4:8], trace[-1,8:12]), reference["scene_id"]
    info = dict(zip(PROBE_FIELDS, map(number, terminal)))
    last = dict(zip(TRACE_FIELDS, map(number, trace[-1])))
    fallback = bool(last["fallback"])
    if fallback:
        mechanism = "infeasible_joint_offer"
    else:
        holds = [last[f"wait_{car}_s"] >= last["step_s"]-1e-9 and
                 last[f"start_{car}_m"] < info[f"entry_{car}_m"] for car in ("a", "b")]
        stopped = [last[f"target_{car}_mps"] == 0 for car in ("a", "b")]
        if all(holds):
            mechanism = "both_reissue_wait"
        elif all(stopped):
            mechanism = "both_select_zero"
        elif all(h or z for h, z in zip(holds, stopped)):
            mechanism = "mixed_wait_and_zero"
        else:
            mechanism = "other_selected_offer"
    last_gap_case = "contact" if info["terminal_gap_m"] <= 1e-9 else "separated"
    metadata = {k: reference[k] for k in ("scene_id", "candidate_id", "dataset", "site", "road_type", "arm",
                                         "elapsed_s", "new_contact", "near_miss", "initial_contact", "any_contact",
                                         "decision_count", "fallback_count")}
    joined = {**metadata, "last_mechanism": mechanism, "terminal_state": last_gap_case,
              "moved_from_initial": int(max(last["end_a_m"], last["end_b_m"]) > 1e-7),
              **info, **{f"last_{field}": value for field, value in last.items()}}
    return joined, trace


def main():
    diagnose = native()
    stalls, by_method = published_stalls()
    records = []
    with (DEST / "decisions.csv").open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(("scene_id", "arm", *TRACE_FIELDS))
        for shard in range(8):
            with (SOURCE / f"index_{shard:02}.csv").open(encoding="utf-8", newline="") as source:
                rows = [row for row in csv.DictReader(source) if row["scene_id"] in stalls]
            if not rows:
                continue
            with np.load(SOURCE / f"windows_{shard:02}.npz", allow_pickle=False) as source:
                arrays = {key: source[key] for key in ("offsets", "t_s", "xy_m", "body_size_m", "velocity_mps")}
            for row in rows:
                for arm, published in stalls[row["scene_id"]].items():
                    report, trace = reload_stall(diagnose, published, row, arrays)
                    records.append(report)
                    writer.writerows([row["scene_id"], arm, *map(number, step)] for step in trace)
    assert len(records) == 344 and len({(r["scene_id"], r["arm"]) for r in records}) == 344
    with (DEST / "cases.csv").open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    ids_a, ids_b = (by_method[arm] for arm in ARMS)
    summary = {
        "unique_scenes": len(stalls), "episodes": len(records), "both_ttc_arms_stalled": len(ids_a & ids_b),
        "only_continuous_ttc": len(ids_a - ids_b), "only_discrete_ttc": len(ids_b - ids_a),
        "same_stall_sets": {arm: sorted(other for other, ids in by_method.items() if ids == by_method[arm])
                            for arm in ARMS},
        "arms": {},
    }
    for arm in ARMS:
        own = [r for r in records if r["arm"] == arm]
        counters = Counter()
        for r in own:
            counters[r["last_mechanism"], r["terminal_state"]] += 1
        free = [r for r in own if r["terminal_state"] == "separated"]
        summary["arms"][arm] = {
            "cases": len(own), "dataset": dict(Counter(r["dataset"] for r in own)),
            "last_mechanism_and_geometry": {" / ".join(k): v for k, v in sorted(counters.items())},
            "initial_contact": sum(int(r["initial_contact"]) for r in own),
            "contact_only_after_start_among_terminal_contact": sum(
                r["initial_contact"] == "0" and r["terminal_state"] == "contact" for r in own),
            "fallback_from_first_to_last": sum(r["fallback_count"] == r["decision_count"] for r in own),
            "neither_moved_from_initial": sum(not r["moved_from_initial"] for r in own),
            "stationary_separated": len(free),
            "separated_with_one_static_parked_safe_to_other_endpoint": sum(
                bool(r["a_can_finish_with_b_parked"] or r["b_can_finish_with_a_parked"]) for r in free),
            "separated_with_any_original_safe_short_offer": sum(r["original_safe_0p4s"] > 0 for r in free),
            "separated_with_original_safe_moving_full_forecast": sum(r["original_moving_first_update"] > 0 for r in free),
            "separated_with_stop_added_and_a_safe_moving_full_forecast": sum(
                r["add_stop_moving_first_update"] > 0 for r in free),
            "separated_with_stop_added_and_unilateral_safe_move_full_forecast": sum(
                r["add_stop_one_moves_first_update"] > 0 for r in free),
            "separated_with_finer_grid_safe_move_full_forecast": sum(
                r["finer_grid_safe_moving_pairs"] > 0 for r in free),
            "separated_with_finer_grid_safe_move_and_one_arrival_full_forecast": sum(
                r["finer_grid_safe_moving_pairs_with_one_arrival"] > 0 for r in free),
        }
    (DEST / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
