"""Resample Songdo passenger-vehicle tracks to the scenario extractor's CSV schema."""

import argparse
from bisect import bisect_left
from collections import defaultdict
import csv
from pathlib import Path


def milliseconds(local_time):
    hour, minute, second = local_time.split(":")
    return (int(hour) * 3600 + int(minute) * 60) * 1000 + round(float(second) * 1000)


def interpolate(samples, times, time):
    index = bisect_left(times, time)
    if index == 0 or index == len(times):
        return None
    before, after = samples[index - 1:index + 1]
    if after[0] - before[0] > 150:
        return None
    fraction = (time - before[0]) / (after[0] - before[0])
    xy = tuple(a + fraction * (b - a) for a, b in zip(before[1:3], after[1:3]))
    return xy, f"{before[3]}|{after[3]}"


def convert(source, destination):
    tracks = defaultdict(dict)
    with source.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["Vehicle_Class"] == "0" and row["Visibility"] == "1":
                timestamp = row["Local_Time"]
                time = milliseconds(timestamp)
                tracks[row["Vehicle_ID"]][time] = (
                    time, float(row["Local_X"]), float(row["Local_Y"]), timestamp
                )
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("track_id", "timestamp_ms", "agent_type", "x", "y", "vx", "vy",
                         "original_file", "position_bracket", "velocity_start_bracket",
                         "velocity_end_bracket"))
        for track, by_time in sorted(tracks.items()):
            times = sorted(by_time)
            samples = [by_time[time] for time in times]
            for time in range((times[0] // 1000 + 1) * 1000, times[-1], 1000):
                previous, center, following = (
                    interpolate(samples, times, time + offset) for offset in (-250, 0, 250)
                )
                if previous is None or center is None or following is None:
                    continue
                velocity = [(b - a) / 0.5 for a, b in zip(previous[0], following[0])]
                writer.writerow((track, time, "car", *center[0], *velocity, source.name,
                                 center[1], previous[1], following[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for source in args.sources:
        date, site, session = source.stem.split("_")
        folder = args.out / f"Songdo_{site}"
        folder.mkdir(parents=True, exist_ok=True)
        convert(source, folder / f"vehicle_tracks_{date}_{session}.csv")


if __name__ == "__main__":
    main()
