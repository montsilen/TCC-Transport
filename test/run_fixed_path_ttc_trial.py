"""Run paired fixed-path TTC bargaining, two-layer MPC and human replays."""

import argparse
from collections import Counter
import csv
from itertools import count
import json
from math import isclose, isfinite
from pathlib import Path

import numpy as np
import shapely

from traffic_data.fixed_path.audit_swept_footprints import DATA, OUTPUT, ROOT
from fixed_path_trial_scene import EVALUATION_STEP, prepare_pilot, road_speed_limit


DT = 0.1
MPC_REPLAN_STEPS = 4
DEFAULT_BARGAINING_STEPS = 4
DEFAULT_SPEED_LIMIT_MPS = 15.0
HORIZON = 12.0
TTC_HORIZON = 5.0
ACCEL, DECEL = 2.0, 4.0
TIME_GAP = 0.2
NEAR_MISS_M = 1.0
FIXED_DELTA = 0.8
TTC_ALPHA = 0.3
CANDIDATES = np.array([(delay, speed) for delay in (0, 0.5, 1, 1.5, 2, 3, 4, 5)
                       for speed in (7, 10, 13)])
FORECAST_TIMES = np.arange(round(HORIZON/DT)+1) * DT
METHODS = ("human", "two_layer_mpc", "fixed_release", "ttc_bargaining", "fixed_discount_bargaining")


class SimulationStalled(RuntimeError):
    """An uncapped controller has repeated an unfinished state."""

    def __init__(self, message, progress, decisions):
        super().__init__(message)
        self.progress = np.asarray(progress)
        self.times = np.arange(len(self.progress))*DT
        self.decisions = decisions


def first_crossing(times, positions, levels):
    """Return the interpolated time each path reaches its level."""
    positions, levels = np.broadcast_arrays(positions, levels)
    hit = positions >= levels
    i = np.argmax(hit, axis=-1)
    last = np.take_along_axis(positions, i[..., None], axis=-1)[..., 0]
    old = np.take_along_axis(positions, np.maximum(0, i-1)[..., None], axis=-1)[..., 0]
    level = levels[..., 0]
    fraction = np.divide(level-old, last-old, out=np.zeros_like(last), where=last > old)
    return np.where(hit.any(axis=-1), np.maximum(0, times[i-1]+(times[i]-times[i-1])*fraction)
                    * (i > 0), np.inf)


def advance(position, velocity, target, end):
    acceleration = np.clip((target - velocity)/DT, -DECEL, ACCEL)
    next_speed = np.maximum(0, velocity + acceleration*DT)
    next_position = np.minimum(end, position + DT*(velocity+next_speed)/2)
    return next_position, next_speed, acceleration


def offers(scene, i, state, candidates=CANDIDATES, speed_limit=np.inf):
    delay, speed = candidates.T
    position = np.full(len(delay), state[0], dtype=float)
    velocity = np.full(len(delay), state[1], dtype=float)
    positions = np.empty((len(delay), len(FORECAST_TIMES)))
    velocities = np.empty_like(positions)
    positions[:, 0], velocities[:, 0] = position, velocity
    effort = np.zeros(len(delay))
    for k, t in enumerate(FORECAST_TIMES[:-1], 1):
        hold = (t < delay) & (position < scene.entry[i])
        braking_speed = np.sqrt(2*DECEL*np.maximum(0, scene.entry[i] - 0.7 - position))
        target = np.minimum(np.where(hold, np.minimum(velocity, braking_speed), speed), speed_limit)
        position, velocity, acceleration = advance(position, velocity, target, scene.routes[i].progress[-1])
        effort += acceleration**2 * DT * (positions[:, k-1] < scene.goal[i])
        positions[:, k], velocities[:, k] = position, velocity

    enter = first_crossing(FORECAST_TIMES, positions, scene.entry[i])
    leave = first_crossing(FORECAST_TIMES, positions, scene.exit[i])
    finish = first_crossing(FORECAST_TIMES, positions, scene.goal[i])
    moving_end = np.where(np.isfinite(finish), finish, HORIZON + np.maximum(0, scene.goal[i]-position)/2)
    # Each distinct slot retains the cheapest feasible single-vehicle trajectory for MPC.
    cost = moving_end + 0.02 * effort
    return positions, velocities, enter, leave, moving_end, cost


def continuous_candidates(speed_range):
    if speed_range is None:
        return CANDIDATES
    lower, upper = speed_range
    if not isfinite(lower) or not isfinite(upper) or not 0 <= lower < upper:
        raise ValueError("The continuous target speed range must satisfy 0 <= MIN < MAX")
    if (lower, upper) == (7.0, 13.0):
        return CANDIDATES
    waits, speeds = (np.unique(column) for column in CANDIDATES.T)
    interior = speeds[(lower < speeds) & (speeds < upper)]
    return np.array([(wait, speed) for wait in waits for speed in (lower, *interior, upper)])


def safe_slots(scene, state, banks):
    first, second = banks
    entry_a, exit_a = first[2][:, None], first[3][:, None]
    entry_b, exit_b = second[2][None, :], second[3][None, :]
    if state[0, 0] >= scene.exit[0] or state[1, 0] >= scene.exit[1]:
        return np.ones((len(entry_a), entry_b.shape[1]), dtype=bool)
    return ((exit_a+TIME_GAP <= entry_b) & np.isfinite(exit_a)) \
        | ((exit_b+TIME_GAP <= entry_a) & np.isfinite(exit_b)) \
        | (~np.isfinite(entry_a) & ~np.isfinite(entry_b))


def cheapest_by_slot(bank):
    _, _, entry, exit, _, cost = bank
    cheapest = {}
    for index, (start, end) in enumerate(zip(entry, exit)):
        key = (start, end)
        if key not in cheapest or cost[index] < cost[cheapest[key]]:
            cheapest[key] = index
    return np.array(list(cheapest.values()))


def choose_mpc(banks, safe):
    """Enumerate compatible slot pairs using the lower-level optimal profile per slot."""
    a, b = map(cheapest_by_slot, banks)
    objective = banks[0][5][a, None] + banks[1][5][None, b]
    selected = np.argmin(np.where(safe[np.ix_(a, b)], objective, np.inf))
    i, j = np.unravel_index(selected, objective.shape)
    return int(a[i]), int(b[j])


def predicted_ttc(scene, state):
    """Estimate TTC from the vehicles' current speeds along their fixed paths."""
    times = FORECAST_TIMES[:round(TTC_HORIZON/DT) + 1]
    progress = state[:, :1] + state[:, 1:2]*times
    path_ends = np.array([route.progress[-1] for route in scene.routes])
    within_paths = np.all(progress <= path_ends[:, None], axis=0)
    shapes = [route.polygons(path[within_paths]) for route, path in zip(scene.routes, progress)]
    overlap = np.flatnonzero(shapely.intersects(*shapes))
    return float(times[within_paths][overlap[0]]) if len(overlap) else np.inf


def first_proposer(scene, state):
    """Choose the earlier predicted entrant at the current along-path speeds."""
    distance = np.maximum(0.0, scene.entry - state[:, 0])
    speed = state[:, 1]
    arrival = np.where(distance == 0, 0.0, np.inf)
    np.divide(distance, speed, out=arrival, where=speed > 0)
    return int(arrival[1] < arrival[0])


def bargain(banks, safe, delta, first):
    ids = np.argwhere(safe)
    finish = np.stack([banks[i][4][ids[:, i]] for i in range(2)], axis=1)
    utilities = np.exp(-(finish-finish.min(axis=0))/2.0)
    continuation = np.zeros(2)
    next_index = 0
    # Solve the three-round alternating-offer game backwards, with zero surplus after rejection.
    for actor in (first, 1-first, first):
        responder = 1 - actor
        feasible = np.flatnonzero(utilities[:, responder] >= delta*continuation[responder] - 1e-12)
        ranks = np.lexsort((finish[feasible].sum(axis=1), -utilities[feasible, actor]))
        offer = feasible[ranks[0]]
        if utilities[offer, actor] >= delta*continuation[actor] - 1e-12:
            continuation = utilities[offer].copy()
            next_index = offer
        else:
            continuation *= delta
    return tuple(ids[next_index])


def fallback(scene, state, steps, speed_limit=np.inf):
    position, speed = state.T.copy()
    history = []
    for _ in range(steps):
        target = np.where((position >= scene.entry) & (position < scene.exit), min(7, speed_limit), 0)
        position, speed, _ = advance(position, speed, target, [route.progress[-1] for route in scene.routes])
        history.append(np.column_stack((position, speed)))
    return np.array(history)


def simulate_fixed_release(scene, *, evaluation_limit=HORIZON, speed_limit=np.inf):
    """Run fixed release until the evaluation limit, or until completion if it is None."""
    ticks = (count() if evaluation_limit is None else
             range(int(np.ceil(min(evaluation_limit, scene.elapsed[-1])/DT))))
    position, speed = scene.measured[0].copy(), scene.initial_speed.copy()
    arrival = (scene.entry-position)/np.maximum(speed, 0.5)
    priority = int(np.argmin(arrival))
    yielding = 1-priority
    release = (scene.entry[priority]+scene.exit[priority])/2 + 5
    end = [route.progress[-1] for route in scene.routes]
    history = [position.copy()]
    for _ in ticks:
        if np.all(position >= scene.goal):
            break
        target = np.full(2, min(10.0, speed_limit))
        if position[priority] < release - 1e-6:
            distance = max(0, scene.entry[yielding] - 0.7 - position[yielding])
            target[yielding] = min(speed_limit, speed[yielding], np.sqrt(2*DECEL*distance))
        next_position, next_speed, _ = advance(position, speed, target, end)
        if (evaluation_limit is None and np.array_equal(next_position, position)
                and np.array_equal(next_speed, speed)):
            raise SimulationStalled(f"Fixed release stalled before completion in {scene.case_id}", history, [])
        position, speed = next_position, next_speed
        history.append(position)
    return np.arange(len(history))*DT, np.array(history), []


def simulate(scene, method, bargaining_steps=DEFAULT_BARGAINING_STEPS, *,
             evaluation_limit=HORIZON, continuous_offers=False, continuous_speed_range=None,
             speed_limit=np.inf, pair_policy=None):
    """Run the controller until the evaluation limit, or until completion if it is None."""
    use_continuous = (continuous_offers or continuous_speed_range is not None) and method.endswith("_bargaining")
    candidates = continuous_candidates(continuous_speed_range) if use_continuous else CANDIDATES
    if speed_limit < candidates[:, 1].max():
        candidates = np.unique(np.column_stack((candidates[:, 0],
                                                np.minimum(candidates[:, 1], speed_limit))), axis=0)
    if use_continuous:
        if pair_policy is None:
            from fixed_path_continuous_offers import ContinuousOfferSearch
        else:
            ContinuousOfferSearch = pair_policy.continuous_search

    feasible = safe_slots if pair_policy is None else pair_policy.feasible
    mpc = choose_mpc if pair_policy is None else pair_policy.mpc
    emergency = fallback if pair_policy is None else pair_policy.emergency

    replan_steps = MPC_REPLAN_STEPS if method == "two_layer_mpc" else bargaining_steps
    steps = None if evaluation_limit is None else int(np.ceil(min(evaluation_limit, scene.elapsed[-1])/DT))
    ticks = count(0, replan_steps) if steps is None else range(0, steps, replan_steps)
    state = np.column_stack((scene.measured[0], scene.initial_speed))
    history = [state.copy()]
    decisions = []
    seen = set()
    for tick in ticks:
        if np.all(state[:, 0] >= scene.goal):
            break
        if steps is None:
            key = tuple(state.flat)
            if key in seen:
                raise SimulationStalled(f"{method} repeated an unfinished state in {scene.case_id}",
                                        np.asarray(history)[:, :, 0], decisions)
            seen.add(key)
        execute = replan_steps if steps is None else min(replan_steps, steps-tick)
        banks = ([offers(scene, i, state[i]) for i in range(2)]
                 if candidates is CANDIDATES and np.isinf(speed_limit) else
                 [offers(scene, i, state[i], candidates, speed_limit) for i in range(2)])
        safe = feasible(scene, state, banks)
        ttc = predicted_ttc(scene, state)
        delta = (1.0 - np.exp(-TTC_ALPHA*ttc)) if method == "ttc_bargaining" else FIXED_DELTA
        first = first_proposer(scene, state) if method.endswith("_bargaining") else None
        if not safe.any():
            selected = None
            block = emergency(scene, state, execute, speed_limit)
        elif use_continuous:
            search = ContinuousOfferSearch(scene, state, banks, safe, first=first, candidates=candidates,
                                           times=FORECAST_TIMES, step=DT, accel=ACCEL, decel=DECEL,
                                           gap=TIME_GAP, advance=advance, crossing_time=first_crossing,
                                           speed_limit=speed_limit)
            proposal = search.select(delta)
            if proposal is None:
                selected = None
                block = emergency(scene, state, execute, speed_limit)
            else:
                block = proposal.execute(execute)
                selected = [{
                    "wait_s": float(proposal.delays[i]),
                    "target_speed_mps": float(proposal.parameters[2*i]),
                    "ramp_acceleration_mps2": float(proposal.ramp_acceleration[i]),
                } for i in range(2)]
        else:
            selected = mpc(banks, safe) if method == "two_layer_mpc" else bargain(banks, safe, delta, first)
            block = np.stack([np.stack((banks[i][0][selected[i], 1:execute+1],
                                        banks[i][1][selected[i], 1:execute+1]), axis=-1)
                              for i in range(2)], axis=1)
            selected = [int(i) for i in selected]
        decision = {"t_s": tick*DT, "ttc_s": None if np.isinf(ttc) else ttc, "delta": float(delta),
                    "offers": selected}
        if first is not None:
            decision["first_proposer"] = "AB"[first]
        decisions.append(decision)
        history.extend(block)
        state = block[-1]
    positions = np.asarray(history)[:, :, 0]
    times = np.arange(len(positions))*DT
    return times, positions, decisions


def assess(scene, times, progress, step=EVALUATION_STEP, *, evaluation_limit=HORIZON):
    limit = times[-1] if evaluation_limit is None else min(evaluation_limit, scene.elapsed[-1])
    finish = first_crossing(times, progress.T, scene.goal[:, None])
    finish = np.where(finish <= limit+1e-9, finish, np.inf)
    end = min(limit, float(finish.max()))
    evaluation_times = np.r_[np.arange(0, end, step), end]
    bodies = [route.polygons(np.interp(evaluation_times, times, progress[:, i]))
              for i, route in enumerate(scene.routes)]
    distance = float(np.min(shapely.distance(*bodies)))
    completed = bool(np.isfinite(finish).all())
    return {"collision": distance <= 1e-9, "near_miss": 1e-9 < distance <= NEAR_MISS_M,
            "min_clearance_m": distance, "complete": completed,
            "safe_complete": completed and distance > 1e-9,
            "finish_a_s": None if not np.isfinite(finish[0]) else float(finish[0]),
            "finish_b_s": None if not np.isfinite(finish[1]) else float(finish[1]),
            "joint_finish_s": float(finish.max()) if completed else None,
            "evaluation_limit_s": float(limit)}


def paired_summary(by_method, target, seed):
    first = by_method["ttc_bargaining"]
    other = by_method[target]
    paired = list(zip(first, other))
    if not paired:
        return {}
    rng = np.random.default_rng(seed)
    recordings = sorted({row["source"] for row in first})
    groups = {source: np.flatnonzero([row["source"] == source for row in first]) for source in recordings}
    resamples = [np.concatenate([groups[source] for source in rng.choice(recordings, len(recordings))])
                 for _ in range(1000)]
    stats = {}
    for key in ("collision", "near_miss", "safe_complete", "joint_finish_s"):
        changes = np.array([np.nan if a[key] is None or b[key] is None else float(a[key])-float(b[key])
                            for a, b in paired])
        valid = changes[np.isfinite(changes)]
        if not len(valid):
            continue
        boot = [float(np.nanmean(changes[indices])) for indices in resamples if np.isfinite(changes[indices]).any()]
        stats[key] = {"pairs": len(valid), "mean_difference": float(valid.mean()),
                      "recording_bootstrap_95_ci": list(map(float, np.quantile(boot, [0.025, 0.975])))}
    return stats


def summarize(results, seed):
    by_method = {method: [row for row in results if row["method"] == method] for method in METHODS}
    overview = {}
    for method, rows in by_method.items():
        completed = [row["joint_finish_s"] for row in rows if row["joint_finish_s"] is not None]
        overview[method] = {"cases": len(rows),
                            **{key: sum(row[key] for row in rows)
                               for key in ("collision", "near_miss", "complete", "safe_complete")},
                            "mean_finish_s_completed": float(np.mean(completed)) if completed else None,
                            "median_min_clearance_m": float(np.median([row["min_clearance_m"] for row in rows]))
                            if rows else None,
                            "fallback_steps": sum(row["fallback_steps"] for row in rows)}
    return {"overview": overview, "ttc_minus_comparator": {
        method: paired_summary(by_method, method, seed) for method in METHODS if method != "ttc_bargaining"}}


def run(scenes, output, seed, census, bargaining_steps=DEFAULT_BARGAINING_STEPS, *,
        continuous_offers=False, continuous_speed_range=None, map_speed_limits=False,
        speed_limit_mps=DEFAULT_SPEED_LIMIT_MPS):
    validate_speed_limit(speed_limit_mps)
    continuous_offers = continuous_offers or continuous_speed_range is not None
    candidates = continuous_candidates(continuous_speed_range) if continuous_offers else CANDIDATES
    results = []
    separate = {"no_recorded_third_party": [], "recorded_third_party": []}
    difference = {"changed_trace": 0, "ttc_finite_cases": 0, "changed_first_offer": 0}
    definitions = [{
        "case_id": scene.case_id, "source": scene.source, "site": scene.site, "start_ms": scene.start_ms,
        "third_party": scene.third_party, "both_slow_at_start": scene.stopped_start,
        "conflict_entry_m": scene.entry.tolist(), "conflict_exit_m": scene.exit.tolist(),
        "finish_progress_m": scene.goal.tolist(), "initial_speed_mps": scene.initial_speed.tolist(),
        "routes": [{"xy_m": route.points.tolist(), "progress_m": route.progress.tolist(),
                    "body_length_width_m": route.size} for route in scene.routes],
    } for scene in scenes]
    (output / "case_geometry.json").write_text(json.dumps(definitions, allow_nan=False), encoding="utf-8")
    with (output / "decision_traces.jsonl").open("w", encoding="utf-8") as stream:
        for index, scene in enumerate(scenes, 1):
            metadata = {"case_id": scene.case_id, "source": scene.source, "site": scene.site,
                        "third_party": scene.third_party, "both_slow_at_start": scene.stopped_start,
                        "start_ms": scene.start_ms}
            limit, limit_source = road_speed_limit(scene.site) if map_speed_limits else (speed_limit_mps, "uniform")
            metadata["road_speed_limit_mps"] = limit
            metadata["road_speed_limit_source"] = limit_source
            trajectories = {}
            logs = {}
            for method in METHODS:
                if method == "human":
                    time, progress, decisions = scene.elapsed, scene.measured, []
                elif method == "fixed_release":
                    time, progress, decisions = simulate_fixed_release(scene, speed_limit=limit)
                else:
                    time, progress, decisions = simulate(scene, method, bargaining_steps,
                                                         continuous_offers=continuous_offers,
                                                         continuous_speed_range=continuous_speed_range,
                                                         speed_limit=limit)
                trajectories[method] = progress
                logs[method] = decisions
                row = {**metadata, "method": method, **assess(scene, time, progress),
                       "fallback_steps": sum(entry["offers"] is None for entry in decisions)}
                row["end_reason"] = ("completed" if row["complete"] else
                                     "human_record_ended" if scene.elapsed[-1] < HORIZON else "time_limit")
                results.append(row)
                separate["recorded_third_party" if scene.third_party else "no_recorded_third_party"].append(row)
                stream.write(json.dumps({**metadata, "method": method, "times_s": time.tolist(),
                                         "progress_m": progress.tolist(), "decisions": decisions})+"\n")
            a, b = (trajectories[name] for name in METHODS[-2:])
            x, y = (logs[name] for name in METHODS[-2:])
            difference["changed_trace"] += int(a.shape != b.shape or not np.allclose(a, b))
            difference["ttc_finite_cases"] += int(any(row["ttc_s"] is not None for row in x))
            difference["changed_first_offer"] += int(bool(x and y) and x[0]["offers"] != y[0]["offers"])
            if index % 10 == 0:
                print(f"{index}/{len(scenes)} paired cases", flush=True)
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = {"case_count": len(scenes), "seed": seed, "screening": dict(census),
               "sites": dict(Counter(scene.site for scene in scenes)), "discount_effect": difference,
               "configuration": {"time_step_s": DT, "replan_s": bargaining_steps*DT,
                                 "mpc_replan_s": MPC_REPLAN_STEPS*DT, "continuous_offers": continuous_offers,
                                 "map_speed_limits": map_speed_limits,
                                 "uniform_speed_limit_mps": None if map_speed_limits else speed_limit_mps,
                                 "continuous_speed_range_mps": [float(candidates[:, 1].min()),
                                                               float(candidates[:, 1].max())] if continuous_offers else None,
                                 "evaluation_step_s": EVALUATION_STEP, "horizon_s": HORIZON,
                                 "ttc_horizon_s": TTC_HORIZON, "ttc_prediction": "constant_speed_on_fixed_paths",
                                 "first_proposer_prediction": "constant_speed_on_fixed_paths",
                                 "near_miss_m": NEAR_MISS_M,
                                 "fixed_discount": FIXED_DELTA, "ttc_alpha_per_s": TTC_ALPHA},
               "all": summarize(results, seed),
               **{name: summarize(rows, seed) for name, rows in separate.items() if rows},
               "both_slow_at_start": summarize([row for row in results if row["both_slow_at_start"]], seed)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(summary["all"]["overview"], indent=2), flush=True)


def bargaining_interval_steps(seconds):
    if not isfinite(seconds) or seconds < DT:
        raise ValueError(f"The bargaining interval must be at least {DT:g} s")
    steps = round(seconds/DT)
    if not isclose(seconds, steps*DT, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"The bargaining interval must be a multiple of {DT:g} s")
    return steps


def validate_speed_limit(limit):
    if not isfinite(limit) or limit <= 0:
        raise ValueError("The speed limit must be a positive finite value in m/s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DATA/"observed_crossings_696.json")
    parser.add_argument("--raw-index", type=Path, default=OUTPUT/"new_raw_near_miss_pairs.csv")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cases", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--bargaining-interval-s", type=float, default=DEFAULT_BARGAINING_STEPS*DT,
                        help="Seconds between offers for both bargaining methods (default: %(default)g)")
    parser.add_argument("--continuous-offers", action="store_true",
                        help="Refine bargaining offers with continuous target speeds and accelerations (requires SciPy)")
    parser.add_argument("--continuous-speed-range", type=float, nargs=2, metavar=("MIN", "MAX"),
                        help="Enable continuous offers with target speeds in the given m/s range")
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument("--speed-limit-mps", type=float, default=DEFAULT_SPEED_LIMIT_MPS,
                        help="Uniform speed cap for all simulated methods (default: %(default)g m/s)")
    limits.add_argument("--map-speed-limits", action="store_true",
                        help="Use the site's mapped or experimental limit")
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be positive")
    try:
        bargaining_steps = bargaining_interval_steps(args.bargaining_interval_s)
        continuous_candidates(args.continuous_speed_range)
        validate_speed_limit(args.speed_limit_mps)
    except ValueError as error:
        parser.error(str(error))
    suffix = "" if bargaining_steps == DEFAULT_BARGAINING_STEPS else f"_{bargaining_steps*DT:g}s"
    if args.continuous_speed_range is not None:
        lower, upper = args.continuous_speed_range
        suffix += f"_continuous_{lower:g}-{upper:g}mps"
    elif args.continuous_offers:
        suffix += "_continuous"
    if args.map_speed_limits:
        suffix += "_speed_limits"
    else:
        suffix += f"_speed_limit_{args.speed_limit_mps:g}mps"
    output = args.output or ROOT / "Result" / f"fixed_path_trial{suffix}"
    output.mkdir(parents=True, exist_ok=True)
    clean, contextual, census = prepare_pilot(args.manifest, args.raw_index, args.cases, HORIZON, args.seed)
    print(f"Prepared {len(clean)} screened cases and {len(contextual)} with recorded third parties", flush=True)
    scenes = (clean + contextual)[:args.cases]
    if not scenes:
        parser.error("No cases passed the approach and finish-line checks")
    run(scenes, output, args.seed, census, bargaining_steps, continuous_offers=args.continuous_offers,
        continuous_speed_range=args.continuous_speed_range, map_speed_limits=args.map_speed_limits,
        speed_limit_mps=args.speed_limit_mps)


if __name__ == "__main__":
    main()
