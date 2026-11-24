# Local Windows CARLA environment

This installs CARLA **0.9.15** and a separate **Python 3.8** client under the
repository. It does not modify the existing DRAI training environment.
The selected version matches the Windows launcher already present in this project;
the older `simulation/setup_carla.sh` instead targets Linux CARLA 0.9.10.1.

## Install and verify

Run from the repository root in PowerShell. Conda and a working bootstrap Python
must already be available (use `-CondaExe` and `-BootstrapPython` to specify them).

```powershell
./tools/carla/setup_windows.ps1 -DownloadServer -StartServer -Smoke
```

If an existing proxy environment variable points to an unavailable proxy, add
`-DirectDownload`. This only changes the downloader's connection handling.
An optional `-WheelMirror https://mirrors.tuna.tsinghua.edu.cn/pypi/web/`
downloads Python wheels through that mirror while checking their SHA-256 against
official PyPI metadata. Server binaries always come from the CARLA release bucket.

The server download is approximately 7.76 GB. Allow at least 32 GB free for the
partial files, assembled ZIP and extracted distribution. Partial downloads resume
on rerun. An OS file lock prevents concurrent server installers. The installer
checks the ZIP's CRCs and validates extraction paths. The recorded archive SHA-256
is a local integrity record, not a publisher signature.
After successful extraction, range parts are removed and the complete ZIP is retained.

Paths:

- `.venv-carla/`: isolated Python and client dependencies.
- `simulation/carla/downloads/`: server ZIP and resumable parts.
- `simulation/carla/WindowsNoEditor/`: extracted server and PythonAPI helpers.
- `simulation/carla/wheels/`: verified Python client wheels.
- `Result/carla_environment/install_status.json`: latest installation stage.
- `Result/carla_environment/preflight.json`: client/server readiness report.
- `Result/carla_environment_smoke/`: actual physics/sensor report and RGB/LiDAR samples.

The server starts hidden, at low graphics quality, with offscreen rendering.
Offscreen rendering keeps cameras functional; it is different from disabling
rendering. The process ID is recorded in `simulation/carla/server.pid`.
The setup does not stop the server after the test. Stop the specific instance
when finished to release GPU memory.

## Subsequent use

```powershell
$env:CARLA_ROOT = "$PWD/simulation/carla/WindowsNoEditor"
$env:PATH = "$PWD/.venv-carla;$PWD/.venv-carla/Library/bin;$env:PATH"
./.venv-carla/python.exe tools/carla/doctor.py --connect
./.venv-carla/python.exe tools/carla/smoke.py
```

The smoke test uses an idle server, spawns one vehicle plus RGB, LiDAR and
collision sensors, and advances 40 synchronous 0.05-second ticks. It requires
matching client/server versions, nonempty sensor packets at matching frames,
vehicle movement and no collision. It saves `report.json`, `rgb.png`, and
`lidar.ply`, then destroys its actors and restores world settings. This tests the
simulator environment; it is not a negotiation performance experiment.
Five warm-up ticks allow actor snapshots and physics to settle before measurement.
The test also checks measured simulation time, image variation, finite LiDAR points,
and consistency between horizontal displacement and observed speed.

## CoLMDriver integration is a separate readiness level

```powershell
./.venv-carla/python.exe tools/carla/doctor.py --connect --full-stack
```

The base environment deliberately does not install the legacy perception/planning
ML stack. In this checkout, the initial audit found that the following resources
were missing:

- `ckpt/codriving/perception/config.yaml`
- `ckpt/codriving/perception/net_epoch_bestval_at16.pth`
- `ckpt/codriving/planning/epoch_26.ckpt`
- `simulation/leaderboard/data/routes_smoke_town10.xml`
- `simulation/leaderboard/data/scenarios/no_scenarios.json`
- the bargaining checkpoint referenced by `pnp_config_bargaining_5_10.yaml`.

The `Result/` directory does contain later reproduction checkpoints, but they
must be selected and their full module/feature compatibility checked before being
used for CARLA evaluation. Do not silently replace missing pretrained models with
random weights or interpret a simulator smoke test as CoLMDriver validation.

The upstream CoLMDriver README provides a separate CoDriving baseline checkpoint
archive matching the `ckpt/codriving/{perception,planning}` directory convention:
[official baseline instructions](https://github.com/cxliu0314/CoLMDriver#evaluation-of-baselines),
[linked checkpoint archive](https://drive.google.com/file/d/1Izg9wZ3ktR-mwn7J_ZqxrwBmtI1YJ6Xi/view).
The archive has not been installed or validated by this base-environment setup.
Upstream targets Linux, Python 3.7, PyTorch 1.10.1 and CARLA 0.9.10.1, so the
Windows 0.9.15 environment must be treated as a port requiring separate integration
validation, not an exact reproduction of the upstream environment.

After installing compatible ML dependencies, obtaining perception/planning weights,
preparing real map routes and selecting a compatible bargaining checkpoint, the
existing `launch_closed_loop_suite.py --variant suite` runs the baseline and
bargaining variants. `CLOSED_LOOP_EXPERIMENTS.md` describes their evaluation.

Official references:

- https://github.com/carla-simulator/carla/releases/tag/0.9.15
- https://carla.readthedocs.io/en/0.9.15/start_quickstart/
- https://pypi.org/project/carla/0.9.15/
