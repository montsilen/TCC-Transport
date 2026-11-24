"""Combine newly found raw-source vehicle pairs and verify the source counts."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from .audit_swept_footprints import OUTPUT


DATASETS = ("interaction", "interactiontc", "sind", "songdo", "waymo")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    args = parser.parse_args()
    rows, coverage = [], {}
    for name in dict.fromkeys(args.datasets):
        with (args.output / f"raw_near_miss_{name}.csv").open(newline="", encoding="utf-8-sig") as stream:
            selected = list(csv.DictReader(stream))
        recordings = json.loads((args.output / f"raw_near_miss_{name}_coverage.json").read_bytes())
        assert len(selected) == sum(recording["qualifying_pairs"] for recording in recordings)
        coverage[name] = recordings
        rows.extend(selected)

    keys = {(row["dataset"], row.get("scenario_id") or row["raw_file"],
             tuple(sorted((row["track_1"], row["track_2"])))) for row in rows}
    assert len(keys) == len(rows)
    assert all(float(row["duration_s"]) >= 3 and float(row["larger_assumed_body_clearance_m"]) <= 1 for row in rows)
    assert all(row["in_previous_6583"] == "True" or row["in_previous_3138"] == "False" for row in rows)
    new = [row for row in rows if row["in_previous_6583"] == "False"]
    new.sort(key=lambda row: (row["dataset"], row["site"], row["raw_file"], int(row["start_ms"])))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output / "new_raw_near_miss_pairs.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(new)

    summary = {
        "by_dataset_all": dict(Counter(row["dataset"] for row in rows)),
        "by_dataset_new": dict(Counter(row["dataset"] for row in new)),
        "by_road_type_new": dict(Counter(row["road_type"] for row in new)),
        "new_body_sweeps_overlap": sum(float(row["nominal_clearance_m"]) == 0 for row in new),
        "new_require_larger_assumed_body": sum(float(row["nominal_clearance_m"]) > 1 for row in new),
        "recordings": coverage,
    }
    (args.output / "new_raw_near_miss_coverage.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print({key: value for key, value in summary.items() if key != "recordings"})
    print("Total", len(rows), "new to expanded 6583", len(new),
          "new to 3138", sum(row["in_previous_3138"] == "False" for row in rows))


if __name__ == "__main__":
    main()
