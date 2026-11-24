"""Index SinD vehicle observations on a common nominal 10 Hz frame clock."""

import argparse
import csv
from pathlib import Path


def convert(source, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with (source.open(newline="", encoding="utf-8-sig") as stream,
          output.open("w", newline="", encoding="utf-8") as normalized):
        reader = csv.DictReader(stream)
        writer = csv.DictWriter(normalized, fieldnames=[*reader.fieldnames, "source_timestamp_ms"])
        writer.writeheader()
        for row in reader:
            if row["agent_type"].lower() != "car":
                continue
            row["source_timestamp_ms"] = row["timestamp_ms"]
            row["timestamp_ms"] = str(int(row["frame_id"]) * 100)
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    convert(args.source, args.out)
