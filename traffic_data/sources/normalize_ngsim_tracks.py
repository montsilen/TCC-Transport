"""Convert NGSIM I-80 snapshots to the shared longitudinal simulation schema."""

import argparse
import csv
from pathlib import Path


FEET_TO_METERS = 0.3048


def convert(source, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with (source.open(newline="", encoding="utf-8-sig") as stream,
          output.open("w", newline="", encoding="utf-8") as normalized):
        writer = csv.writer(normalized)
        writer.writerow(("track_id", "timestamp_ms", "agent_type", "x", "y", "vx", "vy",
                         "original_file", "frame_id", "lane_id", "preceding_id"))
        for row in csv.DictReader(stream):
            if row["Location"].lower() != "i-80" or float(row["v_Class"]) != 2:
                continue
            writer.writerow((
                row["Vehicle_ID"], row["Global_Time"], "car",
                float(row["Local_X"]) * FEET_TO_METERS,
                (float(row["Local_Y"]) - float(row["v_length"]) / 2) * FEET_TO_METERS,
                0, float(row["v_Vel"]) * FEET_TO_METERS, source.name, row["Frame_ID"],
                row["Lane_ID"], str(int(float(row["Preceding"]))) if row["Preceding"] else "",
            ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    convert(args.source, args.out / "NGSIM_Emeryville_I80" / "vehicle_tracks_000.csv")
