"""Recover independent post-window path geometry from the original recordings."""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from traffic_data.fixed_path import build_common_windows as windows

DEST = ROOT / "Result" / "common_window_ttc_15mps"


def tail(track, last_clock_ms, origin, last_xy):
    clock = track.times if track.clock_ms is None else track.clock_ms
    candidates = np.flatnonzero(np.abs(clock - last_clock_ms) <= 1.01)
    if len(candidates) != 1:
        return None
    i = int(candidates[0])
    if np.linalg.norm((track.xy[i] - origin) - last_xy) > 0.01:
        return None
    return (track.xy[i+1:] - origin).astype(np.float32)


def extend(shard, waymo_root):
    DEST.mkdir(parents=True, exist_ok=True)
    with (windows.DEST / f"index_{shard:02}.csv").open(encoding="utf-8", newline="") as stream:
        rows = {r["candidate_id"]: r for r in csv.DictReader(stream)}
    with np.load(windows.DEST / f"windows_{shard:02}.npz", allow_pickle=False) as archive:
        origins, offsets, xy = (archive[k] for k in ("origin_xy_m", "offsets", "xy_m"))
    tails = [None] * (2 * len(rows))
    status = Counter()
    files = windows.recording_files()
    with (windows.DEST / f"jobs_{shard:02}.jsonl").open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            group, entries = json.loads(line)
            for key, _, _, tracks in windows.recording_entries(group, entries, files, waymo_root):
                candidate = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
                row = rows[candidate]
                i, clock = int(row["array_row"]), float(row["clock_end_ms"])
                endpoints = xy[offsets[i+1]-1]
                for car, identifier in enumerate(key[2:]):
                    matched = [result for track in tracks.get(identifier, [])
                               if (result := tail(track, clock, origins[i], endpoints[car])) is not None]
                    if len(matched) != 1:
                        raise ValueError(f"Could not locate the unique recorded endpoint for {candidate}, car {car}")
                    tails[2*i+car] = matched[0]
                    status["vehicles_with_source_tail"] += bool(len(matched[0]))
                    status["tail_source_samples"] += len(matched[0])
            if number % 10 == 0:
                print(shard, number, dict(status), flush=True)
    assert all(t is not None for t in tails), shard
    ends = np.r_[0, np.cumsum([len(t) for t in tails], dtype=np.int64)]
    path = DEST / f"tails_{shard:02}.npz"
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, offsets=ends, xy_m=np.concatenate(tails))
    temporary.replace(path)
    print(shard, len(rows), dict(status), flush=True)


if __name__ == "__main__":
    extend(int(sys.argv[1]), Path.home() / "Downloads")
