"""Paired, deterministic discount experiment at a two-car right-angle crossing."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from opencood.models.comm_modules.rubinstein_bargaining import (
    TTCBasedDiscount,
    compute_pairwise_ttc,
)
from run_mechanism_ablation_v2 import (
    RESULT_ROOT,
    clamp,
    make_crossing_case,
    progress_past_conflict,
    write_csv,
)


DT = 0.05
CRUISE_SPEED = 8.0
CLEARANCE = 5.0
STOP_LINE = 4.0
COLLISION_DISTANCE = 2.2
MAX_ROUNDS = 3
CALIBRATION_TTC = 2.5
GAPS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
DISCOUNT = TTCBasedDiscount()
CALIBRATED_DELTA = DISCOUNT(torch.tensor(CALIBRATION_TTC, dtype=torch.float64)).item()
VARIANTS = {"fixed_0.9": 0.9, "fixed_at_2.5s": CALIBRATED_DELTA, "ttc": None}


def negotiate(first, arrivals, delta):
    """Each proposer offers to go first. A refusal hands the turn to the other car."""
    rounds = []
    fallback = min(range(2), key=arrivals.__getitem__)
    for round_index in range(MAX_ROUNDS):
        proposer = (first + round_index) % 2
        responder = 1 - proposer
        # Nominal extra wait before the responder can enter after the proposer.
        wait = max(0.0, arrivals[proposer] + 1.5 - arrivals[responder])
        utility = math.exp(-0.35 * wait)
        # Reference utility for the shared right-of-way fallback.
        threshold = delta ** (MAX_ROUNDS - round_index) * 0.85
        accepted = utility >= threshold
        rounds.append({
            "proposer": proposer, "responder": responder, "wait_s": wait,
            "utility": utility, "threshold": threshold, "accepted": accepted,
        })
        if accepted:
            return proposer, rounds
    return fallback, rounds


def run_case(gap, first, variant, fixed_delta):
    case = make_crossing_case("intersection", gap)
    cars = case.vehicles
    trace = []
    passed = [None, None]
    min_distance = math.inf
    priority = None

    for step in range(round(case.max_time / DT) + 1):
        time = step * DT
        progress = [progress_past_conflict(car, case.conflict) for car in cars]
        distance = float(np.linalg.norm(cars[0].pos - cars[1].pos))
        min_distance = min(min_distance, distance)
        for index, position in enumerate(progress):
            if position >= 9.0 - 1e-9 and passed[index] is None:
                passed[index] = time

        status = "collision" if distance < COLLISION_DISTANCE else None
        if status is None and all(value is not None for value in passed):
            status = "passed"
        if status is None and time >= case.max_time:
            status = "timeout"

        ttc = compute_pairwise_ttc(
            torch.from_numpy(np.stack([car.pos for car in cars])),
            torch.from_numpy(np.stack([car.vel for car in cars])),
            torch.tensor([car.heading for car in cars], dtype=torch.float64),
        )[0, 1].item()
        delta = DISCOUNT(torch.tensor(ttc, dtype=torch.float64)).item()
        if fixed_delta is not None:
            delta = fixed_delta

        # Agree once before reaching the conflict zone, then honor that order
        # until the first car clears it. Re-evaluate risk from the new state
        # at every step, including steps before the conflict is close enough.
        arrivals = [max(0.0, -position) / CRUISE_SPEED for position in progress]
        rounds = []
        if (status is None and priority is None and max(progress) < 0
                and min(arrivals) <= 2.0 + 1e-9 and abs(arrivals[0] - arrivals[1]) < 1.5):
            priority, rounds = negotiate(first, arrivals, delta)

        yielding = (1 - priority if priority is not None
                    and progress[priority] < CLEARANCE else None)
        accelerations = []
        if status is None:
            for index, car in enumerate(cars):
                target = CRUISE_SPEED
                if index == yielding:
                    remaining = max(0.0, -progress[index] - STOP_LINE)
                    target = min(CRUISE_SPEED, math.sqrt(2.0 * 3.5 * remaining))
                accelerations.append(clamp((target - car.speed) / DT, -5.5, 2.0))

        trace.append({
            "t_s": time, "progress_m": progress, "speed_mps": [car.speed for car in cars],
            "distance_m": distance, "ttc_s": ttc if math.isfinite(ttc) else None,
            "delta": delta, "rounds": rounds, "priority": priority, "yielding": yielding,
            "acceleration_mps2": accelerations, "status": status,
        })
        if status is not None:
            break
        for car, acceleration in zip(cars, accelerations):
            car.speed = clamp(car.speed + acceleration * DT, 0.0, 16.0)
            car.pos = car.pos + car.dir * car.speed * DT

    negotiation = next((row for row in trace if row["rounds"]), None)
    free_times = [math.ceil((arrival + 9.0 / CRUISE_SPEED) / DT) * DT
                  for arrival in (4.0, 4.0 + gap)]
    summary = {
        "gap_s": gap, "first_proposer": first, "variant": variant, "status": status,
        "min_distance_m": min_distance, "end_time_s": time,
        "pass_a_s": passed[0], "pass_b_s": passed[1],
        "total_delay_s": round(sum(passed) - sum(free_times), 6) if status == "passed" else None,
        "priority": priority,
        "agreement": negotiation["rounds"][-1]["accepted"] if negotiation else None,
        "rounds": len(negotiation["rounds"]) if negotiation else 0,
        "negotiation_ttc_s": negotiation["ttc_s"] if negotiation else None,
        "negotiation_delta": negotiation["delta"] if negotiation else None,
    }
    return summary, trace


def compare(fixed, adaptive, fixed_trace, adaptive_trace):
    first_decision = next((a["t_s"] for f, a in zip(fixed_trace, adaptive_trace)
                           if (f["priority"], [r["accepted"] for r in f["rounds"]])
                           != (a["priority"], [r["accepted"] for r in a["rounds"]])), None)
    first_action = next((a["t_s"] for f, a in zip(fixed_trace, adaptive_trace)
                         if f["acceleration_mps2"] != a["acceleration_mps2"]), None)
    both_pass = fixed["status"] == adaptive["status"] == "passed"
    return {
        "gap_s": fixed["gap_s"], "first_proposer": fixed["first_proposer"],
        "fixed_variant": fixed["variant"], "fixed_status": fixed["status"],
        "ttc_status": adaptive["status"], "fixed_priority": fixed["priority"],
        "ttc_priority": adaptive["priority"], "fixed_rounds": fixed["rounds"],
        "ttc_rounds": adaptive["rounds"], "first_decision_diff_s": first_decision,
        "first_action_diff_s": first_action,
        "ttc_minus_fixed_min_distance_m": adaptive["min_distance_m"] - fixed["min_distance_m"],
        "ttc_minus_fixed_total_delay_s": (adaptive["total_delay_s"] - fixed["total_delay_s"]
                                         if both_pass else None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULT_ROOT / "two_car_discount")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summaries, pairs = [], []
    with (args.out / "trace.jsonl").open("w", encoding="utf-8") as stream:
        for gap in GAPS:
            for first in range(2):
                runs = {name: run_case(gap, first, name, delta)
                        for name, delta in VARIANTS.items()}
                for name, (summary, trace) in runs.items():
                    summaries.append(summary)
                    stream.write(json.dumps({"gap_s": gap, "first_proposer": first,
                                             "variant": name, "steps": trace},
                                            allow_nan=False) + "\n")
                adaptive, adaptive_trace = runs["ttc"]
                for name in ("fixed_0.9", "fixed_at_2.5s"):
                    fixed, fixed_trace = runs[name]
                    pairs.append(compare(fixed, adaptive, fixed_trace, adaptive_trace))
    write_csv(args.out / "episodes.csv", summaries)
    write_csv(args.out / "pairs.csv", pairs)
    print(f"{len(summaries)} episodes, {len(pairs)} paired comparisons -> {args.out}")


if __name__ == "__main__":
    main()
