# Fixed-path two-vehicle replay

Each of the 125,932 vehicle pairs starts at a common observed timestamp. The recorded paths and their individual endpoints define the task. The simulator changes the timing along each path through longitudinal control. The joint completion time is the later of the two first-arrival times. A vehicle leaves the interaction upon reaching its endpoint, so collision and near-miss scoring stops at the first arrival.

Vehicle footprints are oriented rectangles evaluated at the same time. Safety scoring samples them every 0.02 s and at the first arrival. A near miss occurs when an episode's minimum clearance is greater than zero and at most 1 m. A repeated unchanged state during execution counts as a stall.

## Methods

- **Bargaining.** Discrete and continuous offers use either a discount calculated from constant-speed TTC extrapolation or a fixed discount. Both variants replan every 0.4 s. The speed cap is 15 m/s, the discrete target speeds are 7, 10, and 13 m/s, and continuous target speeds span 0–15 m/s.
- **Predictive screening.** The screened run tests pairs of candidate profiles for overlapping footprints at 0.2 s intervals over a forecast of up to 12 s. Rejection of every offer triggers braking for both cars. Outcome scoring uses the finer 0.02 s grid. These two sampling rates allow a contact between forecast samples to appear in the outcomes.
- **Screening ablation.** The four bargaining methods select from the unrestricted action pairs and execute without a braking fallback. The utility represents each vehicle's own predicted arrival time. When both vehicles can select their individual fastest profiles together, both discount rules select the same pair. First contact ends the episode as a failure. Arrival times and their averages use the pairs that both finish with no contact.
- **References.** The two-layer MPC-style controller enumerates and scores paired profiles. Human motion is replayed from the source timestamps. The 5 m fixed-release rule uses the same 6,107 paired scenes from the earlier common-window comparison. The ablation applies the same contact and timing criteria to all reference methods. The simulator represents the two selected vehicles and uses their recorded paths, while the human recordings reflect signals and surrounding traffic.

## Results

The screening ablation includes one evaluation per pair for each of the six full-cohort methods, plus 6,107 fixed-release evaluations. Contact rates include the 139 pairs that have contact at initialization.

| Method | Contact-free completion | Contact | Near miss | Mean joint time on contact-free completions |
| --- | ---: | ---: | ---: | ---: |
| Continuous TTC | 92.21% | 7.79% | 3.96% | 8.092 s |
| Continuous fixed discount | 92.21% | 7.79% | 3.96% | 8.092 s |
| Discrete TTC | 92.95% | 7.05% | 3.57% | 8.697 s |
| Discrete fixed discount | 92.95% | 7.05% | 3.57% | 8.697 s |
| Two-layer MPC-style baseline | 98.77% | 1.15% | 6.82% | 8.721 s |
| Human replay | 98.22% | 1.78% | 5.09% | 21.299 s |

The TTC and fixed-discount results matched for every pair at the same action resolution in the ablation. Compared with the screened run, removing screening raised the continuous TTC contact rate from 1.56% to 7.79%, and the discrete TTC rate from 1.06% to 7.05%. Among pairs that completed without contact in both conditions, completion times matched scene by scene. The screened run recorded 123 stalls with continuous TTC and 221 with discrete TTC. All 221 discrete stalls ended with no feasible candidate, and 18 continuous stalls selected zero target speeds for both cars.

## Reproduce locally

Use Python with NumPy, Shapely 2, and protobuf, together with a Windows C++17 compiler such as MinGW-w64. For the same cohort, prepare the four sets of original recordings and the fixed-path candidate indexes described in the [repository README](../README.md#screen-fixed-paths-for-close-encounters). The Waymo reader uses the selected interactive validation shards in the user's Downloads folder by default. Keep source files, synchronized paths, and episode outputs under their local data and results directories.

From the repository root, build the synchronized windows:

```powershell
python -m traffic_data.fixed_path.build_common_windows index
0..7 | ForEach-Object { python -m traffic_data.fixed_path.build_common_windows build --shard $_ }
python -m traffic_data.fixed_path.build_common_windows combine
```

To run the two full-path comparisons with the six common methods, use the following commands. The two modes use separate output directories.

```powershell
foreach ($mode in 'screened', 'unfiltered') {
    $flags = @()
    $out = 'Result/full_path_ttc_15mps'
    if ($mode -eq 'unfiltered') {
        $flags = @('--unfiltered')
        $out = 'Result/full_path_ttc_no_shield_15mps'
    }
    python test/run_full_path_ttc.py init @flags --without-fixed-release
    g++ -std=c++17 -O3 -shared -static-libgcc -static-libstdc++ test/common_window_ttc_kernel.cpp -o "$out/full_path_ttc_kernel.dll"
    0..7 | ForEach-Object { python test/run_full_path_ttc.py run @flags --shard $_ }
    python test/run_full_path_ttc.py combine @flags
}
```

For the archived 6,107-pair reference, run the same commands in fresh output directories and omit `--without-fixed-release`. The screened initialization reads the eligible pair IDs from the locally archived `Result/common_window_ttc_15mps/episodes.csv`. The ablation then uses the same ID list. Each full run writes `episodes.csv` and `summary.json` with pairwise comparisons and breakdowns by dataset and road type. Completed shards resume from their saved blocks.

To inspect the screened TTC stalls, build the diagnostic library and run the replay:

```powershell
$out = 'Result/full_path_ttc_15mps/stall_diagnostics'
New-Item -ItemType Directory -Force $out | Out-Null
g++ -std=c++17 -O3 -shared -static-libgcc -static-libstdc++ test/diagnose_full_path_ttc_stalls.cpp -o "$out/stall_diagnostics.dll"
python test/diagnose_full_path_ttc_stalls.py
```

The diagnostic records every control update for the 344 incomplete TTC method–pair combinations, verifies the records against the endpoint experiment, and tests alternate offers at each stopped state.
