"""Report base CARLA readiness separately from the CoLMDriver integration."""
import argparse
import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--connect", action="store_true")
    p.add_argument("--full-stack", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "Result" / "carla_environment" / "preflight.json")
    args = p.parse_args()
    checks = []

    def check(group, name, callback):
        try:
            detail = callback()
            checks.append({"group": group, "name": name, "ok": True, "detail": str(detail)})
        except Exception as exc:
            checks.append({"group": group, "name": name, "ok": False, "detail": "{}: {}".format(type(exc).__name__, exc)})

    def require_file(relative):
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(str(path))
        return path

    def module_version(name):
        m = importlib.import_module(name)
        return getattr(m, "__version__", getattr(m, "__file__", "imported"))

    for name in ["carla", "numpy", "pygame"]:
        check("client", name, lambda name=name: module_version(name))
    carla_root = Path(os.environ.get("CARLA_ROOT") or ROOT / "simulation" / "carla" / "WindowsNoEditor")
    check("server", "CarlaUE4.exe", lambda: require_file(carla_root / "CarlaUE4.exe"))
    if args.connect:
        def connect():
            import carla
            client = carla.Client(args.host, args.port)
            client.set_timeout(10.0)
            if client.get_client_version() != client.get_server_version():
                raise RuntimeError("CARLA client/server versions differ")
            return "{} {}".format(client.get_server_version(), client.get_world().get_map().name)
        check("server", "RPC", connect)
    if args.full_stack:
        sys.path[:0] = [str(ROOT), str(ROOT / "simulation/leaderboard"),
                       str(ROOT / "simulation/scenario_runner"), str(carla_root / "PythonAPI/carla")]
        for name in ["torch", "torchvision", "yaml", "cv2", "py_trees", "networkx", "dictor",
                     "agents.navigation.local_planner", "srunner", "leaderboard", "team_code.pnp_agent_e2e"]:
            check("codriving", name, lambda name=name: module_version(name))
        for path in ["ckpt/codriving/perception/config.yaml", "ckpt/codriving/perception/net_epoch_bestval_at16.pth",
                     "ckpt/codriving/planning/epoch_26.ckpt", "simulation/leaderboard/data/routes_smoke_town10.xml",
                     "simulation/leaderboard/data/scenarios/no_scenarios.json"]:
            check("codriving", path, lambda path=path: require_file(path))
        def checkpoint():
            import yaml
            config = yaml.safe_load((ROOT / "simulation/leaderboard/team_code/agent_config/pnp_config_bargaining_5_10.yaml").read_text())
            path = config.get("bargaining", {}).get("checkpoint", "")
            if not path:
                raise RuntimeError("No bargaining checkpoint configured")
            return require_file(path)
        check("codriving", "configured bargaining checkpoint", checkpoint)
    report = {"python": sys.executable, "python_version": sys.version, "carla_root": str(carla_root),
              "checks": checks, "ok": all(c["ok"] for c in checks)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for c in checks:
        print("{} [{}] {}: {}".format("OK" if c["ok"] else "MISSING", c["group"], c["name"], c["detail"]))
    print("Report:", args.output)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
