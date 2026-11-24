"""Plot closest spatial distances for all prepared fixed-path case pairs."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fonts
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import shapely

from .audit_swept_footprints import DATA, OUTPUT, swept_area


def case_distances(case, audit):
    window = case["observed_window"]
    samples = [
        {point["t_s"]: (point["x_m"], point["y_m"]) for point in route["samples"]}
        for route in case["observed_routes"]
    ]
    times = sorted(t for t in samples[0].keys() & samples[1].keys() if t <= window["duration_s"])
    assert len(times) == window["common_sample_count"]
    paths = [np.array([points[t] for t in times]) for points in samples]
    areas = [
        swept_area(path, float(audit[f"vehicle_{i}_length_m"]), float(audit[f"vehicle_{i}_width_m"]))
        for i, path in enumerate(paths, 1)
    ]
    center_distance = shapely.distance(shapely.LineString(paths[0]), shapely.LineString(paths[1]))
    footprint_distance = shapely.distance(*areas)
    assert (footprint_distance == 0) == (audit["swept_areas_intersect"] == "True")
    return center_distance, footprint_distance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DATA / "observed_endpoints_5278.json")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    cases = json.loads(args.manifest.read_bytes())["cases"]
    with (args.output / "swept_footprint_intersections.csv").open(newline="", encoding="utf-8-sig") as stream:
        audit = {row["case_id"]: row for row in csv.DictReader(stream)}
    assert cases and len(cases) == len(audit)
    distances = np.array([case_distances(case, audit[case["case_id"]]) for case in cases])
    assert np.isfinite(distances).all() and (distances >= 0).all()
    upper_limits = [0, 0.25, 0.5, 1, 2, 5]
    counts = [np.bincount(np.searchsorted(upper_limits, column), minlength=7) for column in distances.T]
    assert all(series.sum() == len(cases) for series in counts)

    yahei = Path("C:/Windows/Fonts/msyh.ttc")
    chinese = yahei.is_file()
    if chinese:
        fonts.fontManager.addfont(yahei)
        plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["font.size"] = 11
    navy, gray = "#192A3A", "#4F6474"
    center_color, vehicle_color = "#325EAA", "#0A7D70"
    fig, ax = plt.subplots(figsize=(10.8, 6.3), facecolor="#FAFCFE")
    fig.subplots_adjust(left=0.23, right=0.92, top=0.73, bottom=0.19)
    ax.set_facecolor(fig.get_facecolor())
    labels = ["0", "0 < d ≤ 0.25", "0.25 < d ≤ 0.5", "0.5 < d ≤ 1",
              "1 < d ≤ 2", "2 < d ≤ 5", "d > 5"]
    y = np.arange(len(labels))
    for series, offset, color in zip(counts, (-0.18, 0.18), (center_color, vehicle_color)):
        bars = ax.barh(y + offset, series, height=0.30, color=color, zorder=3)
        ax.bar_label(bars, labels=[f"{value:,}" for value in series], padding=5,
                     fontsize=10, color=navy)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(series.max() for series in counts) * 1.14)
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:,.0f}")
    ax.set_xlabel("双车案例数" if chinese else "Number of vehicle pairs", color=gray, labelpad=10)
    ax.xaxis.grid(color="#DCE6EF", linewidth=0.8)
    ax.tick_params(axis="both", colors=gray, length=0, pad=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(0.08, 0.935, "两车轨迹的最小空间距离" if chinese else "Minimum distance between two vehicle paths",
             fontsize=25, weight="bold", color=navy)
    subtitle = f"{len(cases):,} 组实测双车路径 · 按最近距离统计案例数" if chinese else f"{len(cases):,} measured vehicle pairs"
    fig.text(0.08, 0.879, subtitle, fontsize=12, color=gray)
    fig.legend(handles=[Patch(color=center_color, label="车心轨迹" if chinese else "Center paths"),
                        Patch(color=vehicle_color, label="车身扫掠区域" if chinese else "Swept vehicle bodies")],
               loc="upper left", bbox_to_anchor=(0.22, 0.834), ncol=2, frameon=False,
               labelcolor=navy, handlelength=1.0)
    fig.text(0.08, 0.759, "距离 d（米）" if chinese else "Distance d (m)", color=gray)
    note = ("距离为两条完整路径之间的空间最小值，最近点可以出现在不同时间。缺少尺寸的车辆按 4.5 × 1.8 米建模。"
            if chinese else "Closest points may occur at different times. Missing vehicle dimensions use a 4.5 × 1.8 m rectangle.")
    fig.text(0.08, 0.050, note, fontsize=9.2, color=gray)
    destination = args.output / "fixed_path_minimum_distance_counts.png"
    fig.savefig(destination, dpi=180, facecolor=fig.get_facecolor())
    print(list(zip(labels, *counts)))
    print(destination)


if __name__ == "__main__":
    main()
