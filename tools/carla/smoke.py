"""Exercise real CARLA RPC, physics, RGB and LiDAR; no ML weights required."""
import argparse
import json
import math
import queue
import random
import sys
from datetime import datetime, timezone
from pathlib import Path


def frame_from(q, frame):
    while True:
        data = q.get(timeout=15)
        if data.frame == frame:
            return data
        if data.frame > frame:
            raise RuntimeError("Sensor frame passed world frame {}".format(frame))


def run(args):
    import carla
    import numpy as np
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    result = {"client_version": client.get_client_version(), "server_version": client.get_server_version()}
    if result["client_version"] != result["server_version"]:
        raise RuntimeError("CARLA client/server version mismatch: " + str(result))
    if not result["server_version"].startswith("0.9.15"):
        raise RuntimeError("This environment targets CARLA 0.9.15")
    world = client.get_world()
    if world.get_settings().synchronous_mode:
        raise RuntimeError("Server already in synchronous mode; use an idle server for smoke tests")
    if args.map and world.get_map().name.split("/")[-1] != args.map:
        world = client.load_world(args.map)
    result["map"] = world.get_map().name
    original = world.get_settings()
    actors = []
    sensors = []
    collisions = []
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        library = world.get_blueprint_library()
        blueprint = library.find("vehicle.tesla.model3")
        blueprint.set_attribute("role_name", "carla_environment_smoke")
        spawns = list(world.get_map().get_spawn_points())
        random.Random(args.seed).shuffle(spawns)
        vehicle = None
        for spawn in spawns:
            vehicle = world.try_spawn_actor(blueprint, spawn)
            if vehicle is not None:
                break
        if vehicle is None:
            raise RuntimeError("No free vehicle spawn point")
        actors.append(vehicle)
        streams = {}
        for name, identifier in [("rgb", "sensor.camera.rgb"), ("lidar", "sensor.lidar.ray_cast")]:
            bp = library.find(identifier)
            if name == "rgb":
                bp.set_attribute("image_size_x", "320")
                bp.set_attribute("image_size_y", "180")
            else:
                bp.set_attribute("channels", "16")
                bp.set_attribute("points_per_second", "20000")
                bp.set_attribute("rotation_frequency", "20")
            sensor = world.spawn_actor(bp, carla.Transform(carla.Location(z=2.2)), attach_to=vehicle)
            actors.append(sensor)
            sensors.append(sensor)
            q = queue.Queue()
            sensor.listen(q.put)
            streams[name] = q
        collision_sensor = world.spawn_actor(library.find("sensor.other.collision"), carla.Transform(), attach_to=vehicle)
        actors.append(collision_sensor)
        sensors.append(collision_sensor)
        collision_sensor.listen(lambda event: collisions.append(event.frame))
        # Newly spawned actors may not have reached the client's cached snapshot.
        # Settle physics and consume sensor frames before measuring displacement.
        vehicle.apply_control(carla.VehicleControl(brake=1.0))
        for _ in range(5):
            frame = world.tick(30.0)
            for q in streams.values():
                frame_from(q, frame)
        start_time = world.get_snapshot().timestamp.elapsed_seconds
        start = vehicle.get_location()
        counts = {name: 0 for name in streams}
        samples = {}
        max_speed = 0.0
        for step in range(args.steps):
            vehicle.apply_control(carla.VehicleControl(throttle=0.25, steer=0.0))
            frame = world.tick(30.0)
            for name, q in streams.items():
                data = frame_from(q, frame)
                if not data.raw_data:
                    raise RuntimeError("Empty {} sensor packet".format(name))
                counts[name] += 1
                samples[name] = data
            velocity = vehicle.get_velocity()
            max_speed = max(max_speed, math.hypot(velocity.x, velocity.y))
        end = vehicle.get_location()
        simulated_seconds = world.get_snapshot().timestamp.elapsed_seconds - start_time
        if abs(simulated_seconds - args.steps * 0.05) > 1e-4:
            raise RuntimeError("Unexpected simulation time; another client may be ticking the world")
        distance = math.hypot(end.x - start.x, end.y - start.y)
        if not math.isfinite(distance) or distance < 0.1:
            raise RuntimeError("Vehicle did not move: {} m".format(distance))
        if distance > max_speed * simulated_seconds + 0.5:
            raise RuntimeError("Displacement is inconsistent with observed vehicle speed")
        if collisions:
            raise RuntimeError("Smoke vehicle collided; choose another seed")
        rgb = np.frombuffer(samples["rgb"].raw_data, dtype=np.uint8).reshape(180, 320, 4)[:, :, :3]
        image_std = float(rgb.std())
        if image_std < 1.0:
            raise RuntimeError("RGB frame has no useful variation; check rendering")
        points = np.frombuffer(samples["lidar"].raw_data, dtype=np.float32).reshape(-1, 4)
        if len(points) == 0 or not np.isfinite(points).all():
            raise RuntimeError("LiDAR frame has no finite points")
        samples["rgb"].save_to_disk(str(args.output / "rgb.png"))
        samples["lidar"].save_to_disk(str(args.output / "lidar.ply"))
        result.update({"steps": args.steps, "seed": args.seed, "sensor_frames": counts,
                       "horizontal_distance_m": distance, "rgb_std": image_std,
                       "start_xy": [start.x, start.y], "end_xy": [end.x, end.y],
                       "simulated_seconds": simulated_seconds, "max_speed_mps": max_speed,
                       "lidar_points_last_frame": len(points), "collision_events": len(collisions), "ok": True})
        return result
    finally:
        for sensor in sensors:
            sensor.stop()
        for actor in reversed(actors):
            actor.destroy()
        world.apply_settings(original)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--map", default=None, help="Optional map change on an idle server")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=20261030)
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "Result" / "carla_environment_smoke")
    args = p.parse_args()
    if args.steps < 1:
        p.error("steps must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args)
    except Exception as exc:
        result = {"ok": False, "error": "{}: {}".format(type(exc).__name__, exc)}
    result["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    (args.output / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
