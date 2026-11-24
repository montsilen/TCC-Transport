# TCC-Transport

When two vehicles approach a crossing, deciding who goes first solves only part of the problem. The yielding vehicle also needs to know when it can accelerate again. TCC-Transport studies both decisions in paired, two-car simulations. The experiments change one bargaining or control rule at a time and trace its effect from the first offer to the time both vehicles clear the crossing.

## How the experiments work

### Does time to collision change who goes first?

Two vehicles approach a right-angle crossing at 8 m/s. The experiment varies their nominal arrival-time difference across seven values from −2 to +2 seconds and gives each vehicle the first offer in turn. Each of these 14 starting conditions runs with three bargaining discounts: a fixed 0.9, a discount that varies with time to collision (TTC), and a fixed value of about 0.528 equal to the adaptive discount at a TTC of 2.5 seconds. The calibrated fixed value helps isolate the effect of adapting the discount to the encounter.

Each proposer asks to go first. The other vehicle weighs the resulting wait against an acceptance threshold set by the discount and the number of rounds remaining. After three rejected offers, the vehicle with the earlier predicted arrival goes first. The controller holds the agreed order and releases the yielding vehicle once the priority vehicle has passed the conflict point by 5 m. [Rule-based experiment](test/run_two_car_discount_experiment.py)

### Does the learned policy respond to the same change?

The second experiment replays a policy trained with proximal policy optimization (PPO). Every 0.05 seconds, the policy produces offers, the bargaining protocol evaluates them, and the controller converts the agreed offer into a waypoint adjustment. Each discount variant receives the same initial state, checkpoint, proposer order, and sequence of random bird’s-eye-view features. Comparing offers, waypoint adjustments, and vehicle states step by step shows whether a change in discount reaches the controller. [PPO experiment](test/run_two_car_ppo_discount_experiment.py)

An optional fixed-release rule tests a separate decision. In crossing encounters, once either vehicle has passed the conflict point by 5 m, both vehicles accelerate or decelerate toward their own initial speeds. Following encounters keep their original controller. The next research question is whether the cars can safely resume earlier by evaluating candidate acceleration trajectories, their predicted minimum separation, and their conflict-zone occupancy. That comparison is planned.

## How measured tracks enter the simulation

The data adapters read vehicle tracks from INTERACTION, Waymo, NGSIM, SinD, and Songdo. The pipeline converts source-specific records into two-car initial states, groups encounters by geometry, and can attach the observed route segments and recording identifiers. The endpoint filter requires at least 3 seconds of synchronized observations and at least 5 m of displacement for each vehicle. The measured states initialize each replay. Subsequent vehicle motion and collision outcomes come from the simulator. [Source adapters](traffic_data/sources/) · [Scenario construction](traffic_data/build_real_traffic_cases.py) · [Endpoint filter](traffic_data/filter_observed_traffic_cases.py)

## What the paired runs showed

In the controlled crossing, a run succeeds when both vehicle centers have traveled 9 m beyond the conflict point within 9 seconds. A center-to-center distance below 2.2 m counts as a collision. Completion time is measured when the second vehicle crosses its finish line and is averaged over successful runs only.

| Discount in the rule-based crossing | Both vehicles completed | Collisions | Mean completion time |
| --- | ---: | ---: | ---: |
| Fixed 0.9 | 14/14 | 0 | 6.91 s |
| Fixed at the 2.5 s calibration point | 13/14 | 0 | 7.24 s |
| TTC-adaptive | 13/14 | 0 | 7.24 s |

The calibrated fixed discount and the adaptive discount produced identical trajectories in all 14 conditions. For PPO replay, ten measured vehicle pairs from one INTERACTION intersection, two proposer orders, and two checkpoints gave 40 runs per discount. Across 80 adaptive-versus-fixed comparisons, the offers and vehicle trajectories matched step by step. Enabling fixed release increased completions from **14/40 to 22/40** per discount, with **8/40 collisions** in both settings. In these encounters, changing when vehicles resumed cruising affected completion, while varying the discount produced no observed change in the PPO trajectory.

## Reproduce the experiments

Place the [TCC-Driving baseline](https://github.com/montsilen/TCC-Driving/tree/8e9891e20f68d563476c5e1dae40f4e42c14d954) beside this checkout. It supplies the bargaining protocol, vehicle model, and PPO policy. From the TCC-Transport root, use a Python environment with NumPy and PyTorch and run the following PowerShell commands.

```powershell
$env:PYTHONPATH = (Resolve-Path ., ../TCC-Driving).Path -join [IO.Path]::PathSeparator
python test/run_two_car_discount_experiment.py --out Result/two_car_discount
python test/run_two_car_ppo_discount_experiment.py --help
```

The first command writes episode summaries, paired comparisons, and step-by-step traces. For a measured PPO replay, supply a local full checkpoint with `--checkpoint`, a scenario manifest with `--case-manifest`, and an output directory with `--out`. Add `--recover-after-clear` to test fixed release using the same inputs and a second output directory.

Run data tools as Python modules, for example `python -m traffic_data.build_real_traffic_cases --help`. Waymo extraction also requires protobuf. Datasets, manifests, weights, and run outputs are stored locally. The kinematic experiments run independently of CARLA. [CARLA setup and smoke test](tools/carla/README.md)
