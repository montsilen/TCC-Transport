"""Replay the paired two-car crossing experiment with full PPO checkpoints."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from opencood.models.comm_modules.rubinstein_bargaining import (
    RubinsteinBargaining,
    alternating_offers_protocol,
    compute_pairwise_ttc,
)
from run_two_car_discount_experiment import (
    COLLISION_DISTANCE,
    DT,
    GAPS,
    MAX_ROUNDS,
    RESULT_ROOT,
    VARIANTS,
    clamp,
    make_crossing_case,
    progress_past_conflict,
    write_csv,
)
from run_mechanism_ablation_v2 import clone_case
from train_bargaining_ppo import StochasticPolicy


@torch.inference_mode()
def run_episode(model, policy, base_case, first, fixed_delta, seed, *, recover_after_clear=False):
    case = clone_case(base_case)
    cars = case.vehicles
    reference = sum(math.ceil(
        (9.0 - progress_past_conflict(car, case.conflict)) / car.speed / DT
    ) * DT for car in cars)
    order = [first, 1 - first]
    generator = torch.Generator().manual_seed(seed)
    trace, passed = [], [None, None]
    minimum = math.inf
    for tick in range(round(case.max_time / DT) + 1):
        time = tick * DT
        progress = [progress_past_conflict(car, case.conflict) for car in cars]
        for index, position in enumerate(progress):
            if position >= 9.0 - 1e-9 and passed[index] is None:
                passed[index] = time
        distance = float(np.linalg.norm(cars[0].pos - cars[1].pos))
        minimum = min(minimum, distance)
        row = {"t_s": time, "progress_m": progress, "distance_m": distance,
               "speed_mps": [car.speed for car in cars]}
        trace.append(row)
        if distance < COLLISION_DISTANCE:
            status = "collision"
            break
        if all(value is not None for value in passed):
            status = "passed"
            break
        if tick == round(case.max_time / DT):
            status = "timeout"
            break

        # Keep the same input stream per physical vehicle across variants and first proposers.
        features = torch.randn((2, 256, 48, 48), generator=generator)
        states = model.state_encoder(features)[order]
        offers, _ = policy(states, deterministic=True)
        positions = torch.tensor(np.stack([cars[i].pos for i in order]), dtype=torch.float32)
        velocities = torch.tensor(np.stack([cars[i].vel for i in order]), dtype=torch.float32)
        headings = torch.tensor([cars[i].heading for i in order])
        ttc = compute_pairwise_ttc(positions, velocities, headings)
        delta = (model.discount(ttc).min(dim=1).values if fixed_delta is None
                 else torch.full((2,), fixed_delta, dtype=offers.dtype))
        agreed, accepted, metadata = alternating_offers_protocol(
            offers, delta, model.utility_estimator, states, MAX_ROUNDS, return_all_offers=True,
        )
        # Retain the historical evaluator's A offset and B counter-offset.
        # Scale its 0.1 s speed impulse to the crossing simulator's 0.05 s tick.
        offset = model.offer_to_waypoint(agreed[order.index(0)])[:2].numpy()
        offset *= model.waypoint_offset_scale
        row.update({
            "ttc_s": ttc[0, 1].item() if torch.isfinite(ttc[0, 1]) else None,
            "delta": delta.tolist(), "order": order, "initial_offers": offers.tolist(),
            "offers_by_round": metadata["all_offers"].tolist(),
            "initial_utilities": model.utility_estimator(states, offers).squeeze(-1).tolist(),
            "final_offers": agreed.tolist(), "accepted": accepted.tolist(),
            "rounds": metadata["n_rounds"], "converged": metadata["converged"],
            "ego_offset_xy": offset.tolist(),
        })
        # Same-lane following has no point after which the leader ceases to matter.
        released = (recover_after_clear and case.family != "following"
                    and max(progress) >= 5.0)
        row["cruise_recovery"] = released
        for index, (car, adjustment) in enumerate(zip(cars, (offset, -0.5 * offset))):
            if released:
                target = base_case.vehicles[index].speed
                speed_change = clamp(target - car.speed, -2.0 * DT, 2.0 * DT)
            else:
                speed_change = float(np.dot(adjustment, car.dir)) * 2.0 * DT / 0.1
            car.speed = clamp(car.speed + speed_change, 0.3, 15.0)
            car.pos = car.pos + car.dir * car.speed * DT

    decisions = trace[:-1]
    return {
        "status": status, "min_distance_m": minimum, "end_time_s": time,
        "pass_a_s": passed[0], "pass_b_s": passed[1],
        "total_delay_s": round(sum(passed) - reference, 6) if status == "passed" else None,
        "decision_steps": len(decisions),
        "converged_steps": sum(row["converged"] for row in decisions),
        "multi_round_steps": sum(row["rounds"] > 1 for row in decisions),
    }, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", default=20260608, type=int)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--tracks", type=Path, help="Extract cases from an INTERACTION track directory.")
    inputs.add_argument("--case-manifest", type=Path, help="Replay an exported real-traffic case manifest.")
    parser.add_argument("--extended-tracks", action="store_true",
                        help="Keep the four original track cases and add six distinct pairs.")
    parser.add_argument("--recover-after-clear", action="store_true",
                        help="Restore cruise speed after a car clears a crossing by 5 m.")
    args = parser.parse_args()
    if args.extended_tracks and not args.tracks:
        parser.error("--extended-tracks requires --tracks")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    numpy_metadata = [np._core.multiarray.scalar, np.dtype, np.dtypes.Float64DType,
                      np.dtypes.Float32DType]
    with torch.serialization.safe_globals(numpy_metadata):
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model, policy = RubinsteinBargaining({}).eval(), StochasticPolicy().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    policy.load_state_dict(checkpoint["policy"], strict=True)
    args.out.mkdir(parents=True, exist_ok=True)
    episodes, pairs = [], []
    if args.case_manifest:
        from build_real_traffic_cases import load_cases

        cases, sources = load_cases(args.case_manifest)
    elif args.tracks:
        from interaction_two_car_cases import extract_cases

        cases, sources = extract_cases(args.tracks, extended=args.extended_tracks)
    else:
        cases = [make_crossing_case("intersection", gap) for gap in GAPS]
    if args.tracks or args.case_manifest:
        (args.out / "scenarios.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    with (args.out / "trace.jsonl").open("w", encoding="utf-8") as stream:
        for index, case in enumerate(cases):
            seed = args.seed + index
            for first in range(2):
                runs = {}
                for variant, fixed_delta in VARIANTS.items():
                    summary, trace = run_episode(model, policy, case, first, fixed_delta, seed,
                                                 recover_after_clear=args.recover_after_clear)
                    identity = {"gap_s": case.meta.get("gap_s"), "first_proposer": first, "seed": seed}
                    if args.tracks or args.case_manifest:
                        identity.update(case_id=case.case_id, family=case.family)
                    episodes.append({**identity, "variant": variant, **summary})
                    stream.write(json.dumps({**identity, "variant": variant, "steps": trace},
                                            allow_nan=False) + "\n")
                    runs[variant] = summary, trace
                adaptive, adaptive_trace = runs["ttc"]
                for variant in VARIANTS.keys() - {"ttc"}:
                    fixed, fixed_trace = runs[variant]
                    paired_steps = list(zip(fixed_trace[:-1], adaptive_trace[:-1]))
                    decision = next((a["t_s"] for f, a in paired_steps
                                     if (f["accepted"], f["rounds"], f["final_offers"])
                                     != (a["accepted"], a["rounds"], a["final_offers"])), None)
                    action = next((a["t_s"] for f, a in paired_steps
                                   if f["ego_offset_xy"] != a["ego_offset_xy"]), None)
                    state = next((a["t_s"] for f, a in zip(fixed_trace, adaptive_trace)
                                  if f["progress_m"] != a["progress_m"]
                                  or f["speed_mps"] != a["speed_mps"]), None)
                    delay = (adaptive["total_delay_s"] - fixed["total_delay_s"]
                             if fixed["status"] == adaptive["status"] == "passed" else None)
                    pairs.append({
                        **identity, "fixed_variant": variant, "fixed_status": fixed["status"],
                        "ttc_status": adaptive["status"], "first_decision_diff_s": decision,
                        "first_offset_diff_s": action, "first_state_diff_s": state,
                        "ttc_minus_fixed_min_distance_m": adaptive["min_distance_m"] - fixed["min_distance_m"],
                        "ttc_minus_fixed_total_delay_s": delay,
                    })
    write_csv(args.out / "episodes.csv", episodes)
    write_csv(args.out / "pairs.csv", sorted(pairs, key=lambda row: (
        row.get("case_id", ""), row["gap_s"], row["first_proposer"], row["fixed_variant"])))
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint_epoch_zero_based": checkpoint["epoch"],
        "checkpoint_logged_best_reward": float(checkpoint["best_reward"]),
        "seed": args.seed, "variants": VARIANTS, "dt_s": DT,
        "recover_after_clear": args.recover_after_clear,
        "torch_version": torch.__version__,
    }
    if args.case_manifest:
        metadata["case_manifest_sha256"] = hashlib.sha256(args.case_manifest.read_bytes()).hexdigest()
    (args.out / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"{len(episodes)} episodes, {len(pairs)} pairs -> {args.out}")


if __name__ == "__main__":
    main()
