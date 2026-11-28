"""Build paired two-car experiments from synchronized measured paths."""

from dataclasses import dataclass
from functools import lru_cache
from collections import Counter, defaultdict
from pathlib import Path
import csv
import json
import random
import re
import xml.etree.ElementTree as ET

import numpy as np
import shapely

from traffic_data.fixed_path.audit_swept_footprints import DATA, dimensions, swept_area, swept_strips
from traffic_data.fixed_path.find_raw_fixed_path_pairs import pair_windows, read_tracks


SOURCE = DATA.parent / "real_traffic_sources"
SPATIAL_STEP = 0.15
FINISH_BUFFER = 2.0
EVALUATION_STEP = 0.02
DEFAULT_ROAD_LIMIT_MPS = 13.0


@dataclass
class Route:
    points: np.ndarray
    progress: np.ndarray
    angles: np.ndarray
    size: tuple

    @classmethod
    def build(cls, points, size):
        length = np.linalg.norm(np.diff(points, axis=0), axis=1)
        observed_progress = np.r_[0, length.cumsum()]
        moving = np.r_[True, length > 1e-6]
        points = points[moving]
        progress = observed_progress[moving]
        edges = np.diff(points, axis=0)
        tangents = np.unwrap(np.arctan2(edges[:, 1], edges[:, 0]))
        angles = np.r_[tangents[0], (tangents[:-1] + tangents[1:]) / 2, tangents[-1]]
        return cls(points, progress, angles, size), observed_progress

    def polygons(self, progress):
        x = np.interp(progress, self.progress, self.points[:, 0])
        y = np.interp(progress, self.progress, self.points[:, 1])
        angle = np.interp(progress, self.progress, self.angles)
        front = np.stack((np.cos(angle), np.sin(angle)), axis=-1) * (self.size[0] / 2)
        side = np.stack((-np.sin(angle), np.cos(angle)), axis=-1) * (self.size[1] / 2)
        center = np.stack((x, y), axis=-1)
        corners = np.stack((center + front + side, center - front + side,
                            center - front - side, center + front - side), axis=-2)
        return shapely.polygons(corners)


@dataclass
class Scene:
    case_id: str
    source: str
    site: str
    routes: tuple
    entry: np.ndarray
    exit: np.ndarray
    goal: np.ndarray
    elapsed: np.ndarray
    measured: np.ndarray
    initial_speed: np.ndarray
    third_party: bool
    stopped_start: bool
    start_ms: int


@lru_cache(maxsize=None)
def mapped_stop_lines(site):
    path = DATA / "maps" / "maps" / f"{site}.osm_xy"
    if not path.exists():
        return None
    root = ET.parse(path).getroot()
    nodes = {node.get("id"): (float(node.get("x")), float(node.get("y"))) for node in root.findall("node")}
    lines = [shapely.LineString([nodes[child.get("ref")] for child in way.findall("nd")])
             for way in root.findall("way")
             if any(tag.get("k") == "type" and tag.get("v") == "stop_line" for tag in way.findall("tag"))]
    return shapely.union_all(lines)


@lru_cache(maxsize=None)
def road_speed_limit(site):
    """Use a published site-map cap when all recorded driving lanelets agree."""
    path = DATA / "maps" / "maps" / f"{site}.osm_xy"
    if not path.is_file():
        return DEFAULT_ROAD_LIMIT_MPS, "experiment_default"
    relations = ET.parse(path).getroot().findall("relation")
    regulations, lanes = {}, []
    units = {"mph": 0.44704, "kmh": 1/3.6, "km/h": 1/3.6, "kph": 1/3.6, "m/s": 1.0}
    for relation in relations:
        tags = {tag.get("k"): tag.get("v") for tag in relation.findall("tag")}
        if tags.get("type") == "lanelet":
            if tags.get("subtype", "road") in ("road", "highway"):
                lanes.append({member.get("ref") for member in relation.findall("member")
                              if member.get("role") == "regulatory_element"})
        elif tags.get("subtype") == "speed_limit":
            match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(mph|km/?h|kph|m/s)",
                                 tags.get("sign_type", "").lower())
            if match and float(match[1]) > 0:
                regulations[relation.get("id")] = float(match[1]) * units[match[2]]
    lane_limits = [{regulations[ref] for ref in lane if ref in regulations} for lane in lanes]
    if lane_limits and len(lane_limits[0]) == 1 and all(limit == lane_limits[0] for limit in lane_limits):
        return next(iter(lane_limits[0])), "map"
    return DEFAULT_ROAD_LIMIT_MPS, "experiment_default"


def traffic_files():
    folders = (SOURCE / "DR", DATA.parent / "interaction_extension/source/TC")
    return {f"{path.parent.name}/{path.name}": path for folder in folders
            for path in folder.rglob("vehicle_tracks_*.csv")}


@lru_cache(maxsize=16)
def traffic_rows(file):
    by_time = {}
    paths = (file, file.with_name(file.name.replace("vehicle_tracks_", "pedestrian_tracks_")))
    for index, path in enumerate(paths):
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                identifier = row["track_id"] if index == 0 else f"pedestrian:{row['track_id']}"
                size = dimensions(row) or (0.8, 0.8)
                by_time.setdefault(int(row["timestamp_ms"]), []).append(
                    (identifier, float(row["x"]), float(row["y"]), np.hypot(*size)/2))
    return by_time


def nearby_traffic(file, ids, first_ms, last_ms, region):
    bounds = region.bounds
    for time, rows in traffic_rows(file).items():
        if not first_ms <= time <= last_ms:
            continue
        for identifier, x, y, radius in rows:
            if identifier not in ids and bounds[0] - radius <= x <= bounds[2] + radius \
                    and bounds[1] - radius <= y <= bounds[3] + radius \
                    and shapely.distance(shapely.Point(x, y), region) <= radius:
                return True
    return False


def rectangle_conflict(routes):
    grids = [np.linspace(0, route.progress[-1], int(np.ceil(route.progress[-1]/SPATIAL_STEP)) + 1)
             for route in routes]
    envelopes = [swept_area(route.points, *route.size) for route in routes]
    if not shapely.intersects(*envelopes):
        return None
    entry, exit = [], []
    for index, (route, grid) in enumerate(zip(routes, grids)):
        interval = grid[shapely.intersects(route.polygons(grid), envelopes[1-index])]
        if not len(interval):
            return None
        entry.append(max(0, interval[0]-SPATIAL_STEP))
        exit.append(interval[-1]+SPATIAL_STEP)
    return np.array(entry), np.array(exit)


def prepare_case(case, files, observation_limit, *, check_finish_coverage_first=False):
    file = files.get(case["source_file"])
    stop_lines = mapped_stop_lines(case["site"])
    if file is None or stop_lines is None:
        return None, "missing_recording_or_map"
    observed = case["observed_window"]
    all_rows = case["raw_rows"]
    sizes = {str(row["track_id"]): dimensions(row) for row in all_rows}
    samples = [{sample["t_s"]: sample for sample in route["samples"]
                if sample["t_s"] <= observed["duration_s"]} for route in case["observed_routes"]]
    elapsed = np.array(sorted(samples[0].keys() & samples[1].keys()))
    offset = np.array(case["conflict_xy"])
    routes, progress, speeds = [], [], []
    for route, track in zip(case["observed_routes"], samples):
        size = sizes[str(route["track_id"])]
        if size is None:
            return None, "missing_dimensions"
        points = np.array([(track[t]["x_m"], track[t]["y_m"]) for t in elapsed]) + offset
        if np.linalg.norm(points[-1]-points[0]) < 5:
            return None, "short_route"
        path, observed_progress = Route.build(points, size)
        routes.append(path)
        progress.append(observed_progress)
        speeds.append(np.array([(track[t]["vx_mps"], track[t]["vy_mps"]) for t in elapsed]))
    if check_finish_coverage_first:
        for index, route in enumerate(routes):
            length = route.progress[-1]
            grid = np.linspace(0, length, int(np.ceil(length/SPATIAL_STEP)) + 1)
            too_late = grid[grid + SPATIAL_STEP + FINISH_BUFFER > length]
            other = routes[1-index]
            sweep = shapely.STRtree(swept_strips(other.points, *other.size))
            if sweep.query(route.polygons(too_late), predicate="intersects").size:
                return None, "finish_line_outside_recorded_path"
            if sweep.query(route.polygons(0.0), predicate="intersects").size:
                return None, "path_starts_in_conflict_zone"
    conflict = rectangle_conflict(routes)
    if conflict is None:
        return None, "no_footprint_conflict"
    entry, exit = conflict
    goal = exit + FINISH_BUFFER
    progress = np.array(progress).T
    if np.any(goal > progress[-1]):
        return None, "finish_line_outside_recorded_path"
    passed_lines = np.zeros(2)
    for index, route in enumerate(routes):
        line = shapely.LineString(route.points)
        contacts = shapely.intersection(line, stop_lines)
        points = shapely.get_coordinates(contacts)
        mapped = shapely.line_locate_point(line, shapely.points(points))
        before_entry = mapped[mapped <= entry[index]]
        if len(before_entry):
            passed_lines[index] = max(before_entry)
            near_line = (progress[:, index] >= passed_lines[index] - 5) & (progress[:, index] <= passed_lines[index])
            if not np.any(near_line & (np.linalg.norm(speeds[index], axis=1) <= 1.0)):
                return None, "mapped_stop_without_observed_stop"
    eligible = np.flatnonzero(np.all((progress >= passed_lines) & (progress < entry - 0.25), axis=1)
                              & (elapsed[-1]-elapsed >= 2.0))
    if not len(eligible):
        return None, "no_common_approach_after_mapped_stop_lines"
    start = int(eligible[0])
    at_start = progress[start]
    direction = [np.interp(at_start[i], route.progress, route.angles) for i, route in enumerate(routes)]
    speed = np.maximum(0.0, [np.dot(v[start], [np.cos(theta), np.sin(theta)])
                            for v, theta in zip(speeds, direction)])
    first_ms = case["timestamp_ms"] + int(round(elapsed[start]*1000))
    horizon = min(observation_limit, elapsed[-1]-elapsed[start])
    # Screen other recorded actors in the two remaining travel corridors.
    remaining = [route.points[(route.progress >= at_start[i]-SPATIAL_STEP)
                              & (route.progress <= goal[i]+route.size[0]/2)] for i, route in enumerate(routes)]
    corridor = shapely.union_all([swept_area(points, *route.size) if len(points)>1
                                  else route.polygons(np.array([at_start[i]]))[0]
                                  for i, (points, route) in enumerate(zip(remaining, routes))])
    other = nearby_traffic(file, sizes, first_ms, first_ms + round(horizon * 1000), shapely.buffer(corridor, 2))
    return Scene(case["case_id"], case["source_file"], case["site"], tuple(routes), entry, exit, goal,
                 elapsed[start:]-elapsed[start], progress[start:], speed, other,
                 bool(np.all(speed < 2.0)), first_ms), "eligible"


def load_cases(path):
    manifest = json.loads(Path(path).read_bytes())
    return [case for case in manifest["cases"] if case["road_type"] == "intersection"
            and case["dataset"] == "INTERACTION"]


def expand_case(row, tracks, *, require_centerline_crossing=True):
    start, end = int(row["start_ms"]), int(row["end_ms"])
    ids = (row["track_1"], row["track_2"])
    options = [(a, b, paths) for a in tracks[ids[0]] for b in tracks[ids[1]]
               for first, last, paths in pair_windows(a, b) if (first, last) == (start, end)]
    if len(options) != 1:
        return None
    a, b, paths = options[0]
    if require_centerline_crossing and not shapely.crosses(*(shapely.LineString(path) for path in paths)):
        return None
    relative_time = np.arange(len(paths[0])) * 0.1
    routes, bodies = [], []
    for index, (track, path) in enumerate(zip((a, b), paths), 1):
        velocity = np.gradient(path, 0.1, axis=0)
        routes.append({"track_id": track.identifier,
                       "samples": [{"t_s": float(t), "x_m": float(x), "y_m": float(y),
                                    "vx_mps": float(vx), "vy_mps": float(vy)}
                                   for t, (x, y), (vx, vy) in zip(relative_time, path, velocity)]})
        bodies.append({"track_id": track.identifier, "length": row[f"track_{index}_length_m"],
                       "width": row[f"track_{index}_width_m"]})
    return {"case_id": f"{row['site']}_{Path(row['raw_file']).stem}_{start}_{ids[0]}_{ids[1]}",
            "source_file": row["previous_source_file"], "site": row["site"], "timestamp_ms": start,
            "conflict_xy": [0, 0], "observed_routes": routes, "raw_rows": bodies,
            "observed_window": {"duration_s": float(relative_time[-1])}}


def prepare_pilot(manifest, raw_index, count, horizon, seed):
    files = traffic_files()
    census = Counter()
    clean, contextual = [], []
    for case in load_cases(manifest):
        scene, reason = prepare_case(case, files, horizon)
        census[reason] += 1
        if scene is not None:
            (contextual if scene.third_party else clean).append(scene)
    if len(clean) >= count or not raw_index.is_file():
        return clean[:count], contextual[:count], census

    by_file = defaultdict(list)
    with raw_index.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["dataset"] == "INTERACTION" and row["road_type"] == "intersection" \
                    and row["previous_source_file"] in files and mapped_stop_lines(row["site"]) is not None \
                    and 4 <= float(row["duration_s"]) <= 18 and float(row["nominal_clearance_m"]) == 0:
                by_file[row["previous_source_file"]].append(row)
    rng = random.Random(seed)
    items = sorted(by_file.items())
    rng.shuffle(items)
    # Limit cases from any recording so the pilot is not dominated by one traffic stream.
    cap = max(4, int(np.ceil(count/max(1, len(items))))*3)
    for source, rows in items:
        tracks = defaultdict(list)
        for track in read_tracks(files[source], "INTERACTION"):
            tracks[track.identifier].append(track)
        rng.shuffle(rows)
        added = 0
        for row in rows:
            case = expand_case(row, tracks)
            if case is None:
                census["raw_pair_without_centerline_crossing"] += 1
                continue
            scene, reason = prepare_case(case, files, horizon)
            census[reason] += 1
            if scene is None:
                continue
            if scene.third_party:
                if len(contextual) < count:
                    contextual.append(scene)
            else:
                clean.append(scene)
                added += 1
                if len(clean) >= count or added >= cap:
                    break
        if len(clean) >= count:
            break
    return clean, contextual, census
