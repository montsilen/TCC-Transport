"""Assemble the expanded, traceable simulation workload from extracted manifests."""

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .build_real_traffic_cases import load_cases
from .enrich_real_traffic_cases import enrich


RESULT = Path("Result")
SOURCE = RESULT / "real_traffic_sources"
OUTPUT = RESULT / "real_traffic_expanded"
INTERACTION_ROOTS = (RESULT / "interaction_extension/source/TC", SOURCE / "DR")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def track_keys(row):
    return {(row["source_file"], row.get("scenario_id"), str(track["track_id"]))
            for track in row["raw_rows"]}


def map_index(folder):
    locations = {}
    for path in sorted((folder / "maps/maps").glob("*.osm_xy")):
        nodes = ET.parse(path).getroot().findall("node")
        xs, ys = ([float(node.attrib[key]) for node in nodes] for key in ("x", "y"))
        locations[path.stem] = {
            "file": path.relative_to(folder).as_posix(), "coordinate_frame": "recording_xy_m",
            "bounds_xy_m": [min(xs), min(ys), max(xs), max(ys)],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    for site in ("C", "G"):
        path = folder / "maps" / f"Songdo_{site}_lanes.csv"
        locations[f"Songdo_{site}"] = {
            "file": path.relative_to(folder).as_posix(), "coordinate_frame": "ortho_pixels",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    for path in sorted((folder / "maps").glob("SinD_*.osm")):
        locations[path.stem] = {
            "file": path.relative_to(folder).as_posix(), "coordinate_frame": "lanelet2_source",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return locations


def assemble(limit):
    base = read(RESULT / "seven_source_traffic_cases/three_source_1000_balanced.json")
    rows = base["cases"]
    base_count = len(rows)
    if not base_count <= limit < 10000:
        raise ValueError("Limit must retain the base workload and be below 10,000")
    ids = {row["case_id"] for row in rows}
    tracks = set().union(*(track_keys(row) for row in rows))
    hashes = dict(base["recordings_sha256"])
    counts = Counter(row["family"] for row in rows)

    def include(row, dataset, source):
        keys = track_keys(row)
        if row["case_id"] in ids or not keys.isdisjoint(tracks):
            return
        row["dataset"] = dataset
        rows.append(row)
        ids.add(row["case_id"])
        tracks.update(keys)
        counts[row["family"]] += 1
        name = row["source_file"]
        digest = source["recordings_sha256"][name]
        if hashes.setdefault(name, digest) != digest:
            raise ValueError(f"Conflicting source hash for {name}")

    for dataset, path in (
        ("Waymo Open Motion", RESULT / "seven_source_traffic_cases/waymo_four_shards.json"),
        ("Songdo Traffic", OUTPUT / "songdo_routes.json"),
        ("SinD", OUTPUT / "sind_routes.json"),
    ):
        source = read(path)
        for row in source["cases"]:
            if len(rows) == limit:
                break
            include(row, dataset, source)

    interaction = read(OUTPUT / "interaction_candidates.json")
    strata = defaultdict(deque)
    for row in interaction["cases"]:
        strata[row["family"], row["site"], row["timing_band"]].append(row)
    families = defaultdict(deque)
    for (family, _, _), queue in sorted(strata.items()):
        families[family].append(queue)
    while families and len(rows) < limit:
        family = min(families, key=lambda key: (counts[key], key))
        queues = families[family]
        queue = queues.popleft()
        include(queue.popleft(), "INTERACTION", interaction)
        if queue:
            queues.append(queue)
        if not queues:
            del families[family]
    dense = {"cases": [row for row in rows if row["dataset"] == "INTERACTION"]}
    enrich(dense, INTERACTION_ROOTS, route_step_ms=200)

    locations = map_index(OUTPUT)
    for row in rows:
        asset = locations.get(row["site"])
        if asset:
            row["map_id"] = row["site"]
        routes = row["observed_routes"]
        row["joint_observed_duration_s"] = min(route["samples"][-1]["t_s"] for route in routes)
        if asset and "bounds_xy_m" in asset:
            min_x, min_y, max_x, max_y = asset["bounds_xy_m"]
            ox, oy = row["conflict_xy"]
            row["observed_inside_map_extent"] = all(
                min_x <= p["x_m"] + ox <= max_x and min_y <= p["y_m"] + oy <= max_y
                for route in routes for p in route["samples"]
            )
    map_ids = {row["map_id"] for row in rows if "map_id" in row}
    return {"schema_version": 1,
            "selection": {"requested_limit": limit, "base_cases_preserved": base_count,
                          "policy": "retain base; add non-INTERACTION sources; balance geometry across sites and temporal strata"},
            "maps": {key: locations[key] for key in sorted(map_ids)},
            "recordings_sha256": dict(sorted(hashes.items())), "cases": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=6000)
    args = parser.parse_args()
    manifest = assemble(args.limit)
    path = OUTPUT / f"expanded_{len(manifest['cases'])}.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n",
                    encoding="utf-8")
    cases, rows = load_cases(path)
    assert len(cases) == len({case.case_id for case in cases})
    print(f"{len(cases)} cases -> {path}")
    print(Counter(row["dataset"] for row in rows))
    print(Counter(row["family"] for row in rows))
