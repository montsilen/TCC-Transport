"""Build a stratified manifest of measured two-vehicle initial states."""

import argparse
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path

import numpy as np

from interaction_two_car_cases import scan_cases
from run_mechanism_ablation_v2 import ScenarioCase, Vehicle


def build(roots, limit, *, sample_ms=1000):
    strata = defaultdict(list)
    files = {}
    for root in roots:
        for path in root.rglob("vehicle_tracks_*.csv"):
            name = f"{path.parent.name}/{path.name}"
            if name in files:
                raise ValueError(f"Duplicate recording key: {name}")
            files[name] = path
        for entries in scan_cases(root, extended=True, broad=True, sample_ms=sample_ms).values():
            for score, case, source, value, _ in entries:
                recording = source["source_file"]
                band = (("short" if value < 3 else "medium" if value < 5 else "long")
                        if case.family == "following"
                        else ("near_simultaneous" if abs(value) <= 0.5 else
                              "b_first" if value < 0 else "a_first"))
                site = recording.split("/")[0]
                source.update(site=site, timing_band=band, max_time_s=case.max_time,
                              gap_s=case.meta["gap_s"])
                if case.family == "following":
                    source["nominal_catch_time_s"] = float(value)
                strata[case.family, band, site].append((score, source))

    queues = deque(deque(source for _, source in sorted(pool, key=lambda item: item[0]))
                   for _, pool in sorted(strata.items()))
    selected, used = [], set()
    while queues and len(selected) < limit:
        queue = queues.popleft()
        while queue:
            item = queue.popleft()
            keys = {(item["source_file"], row["track_id"]) for row in item["raw_rows"]}
            if keys.isdisjoint(used):
                used.update(keys)
                selected.append(item)
                break
        if queue:
            queues.append(queue)
    recordings = {item["source_file"] for item in selected}
    return {
        "schema_version": 1,
        "recordings_sha256": {name: hashlib.sha256(files[name].read_bytes()).hexdigest()
                              for name in sorted(recordings)},
        "cases": selected,
    }


def load_cases(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1 or not manifest["cases"]:
        raise ValueError(f"Unsupported or empty case manifest: {path}")
    cases = []
    for row in manifest["cases"]:
        reference = np.asarray(row["conflict_xy"], dtype=float)
        vehicles = tuple(
            Vehicle(np.asarray(position, dtype=float) - reference, heading, speed, name)
            for position, heading, speed, name in zip(
                row["simulated_start_xy"], row["simulated_heading_rad"],
                row["initial_speed_mps"], ("a", "b"), strict=True)
        )
        cases.append(ScenarioCase(row["family"], row["case_id"], vehicles,
                                  np.zeros(2, dtype=float), row["max_time_s"],
                                  {"gap_s": row["gap_s"]}))
    return cases, manifest["cases"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--sample-ms", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.limit, args.sample_ms) < 1:
        parser.error("--limit and --sample-ms must be positive")
    manifest = build(args.tracks, args.limit, sample_ms=args.sample_ms)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f'{len(manifest["cases"])} cases -> {args.out}')


if __name__ == "__main__":
    main()
