param(
    [string]$CondaExe = 'conda',
    [string]$BootstrapPython = 'python',
    [string]$CarlaRoot = '',
    [switch]$DownloadServer,
    [switch]$DirectDownload,
    [switch]$StartServer,
    [switch]$Smoke,
    [string]$WheelMirror = '',
    [int]$Port = 2000
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$environmentRoot = Join-Path $projectRoot '.venv-carla'
$pythonExe = Join-Path $environmentRoot 'python.exe'
$reportRoot = Join-Path $projectRoot 'Result/carla_environment'
New-Item -ItemType Directory -Force $reportRoot | Out-Null
function Write-InstallState([string]$Stage, [string]$Status, [string]$Detail = '') {
    @{ stage=$Stage; status=$Status; detail=$Detail; timestamp_utc=[DateTime]::UtcNow.ToString('o'); python=$pythonExe } |
        ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $reportRoot 'install_status.json')
    Write-Host "[$Status] $Stage $Detail"
}
trap {
    Write-InstallState 'installation' 'failed' $_.Exception.Message
    exit 1
}
Write-InstallState 'python_environment' 'running'
if (-not $CarlaRoot) { $CarlaRoot = Join-Path $projectRoot 'simulation/carla/WindowsNoEditor' }
if (-not (Test-Path -LiteralPath $pythonExe)) {
    & $CondaExe create --prefix $environmentRoot python=3.8 pip -y
    if ($LASTEXITCODE) { throw 'Conda environment creation failed' }
}
$env:PATH = "$environmentRoot;$(Join-Path $environmentRoot 'Library/bin');$env:PATH"
& $pythonExe -c "import sys; assert sys.version_info[:2] == (3, 8), sys.version"
if ($LASTEXITCODE) { throw 'Expected Python 3.8' }
$downloadArgs = @()
if ($DirectDownload) { $downloadArgs += '--direct' }
$wheelArgs = @()
if ($WheelMirror) { $wheelArgs += @('--wheel-mirror', $WheelMirror) }
Write-InstallState 'client_dependencies' 'running'
& $pythonExe -c "import importlib.metadata as m; assert all(m.version(n) == v for n, v in [('carla','0.9.15'),('numpy','1.24.4'),('pygame','2.5.2')])" 2>$null
if ($LASTEXITCODE -ne 0) {
    & $BootstrapPython (Join-Path $PSScriptRoot 'install_windows.py') --wheels-only @downloadArgs @wheelArgs
    if ($LASTEXITCODE) { throw 'Client wheel download failed' }
    & $pythonExe -m pip install --no-index --find-links (Join-Path $projectRoot 'simulation/carla/wheels') -r (Join-Path $PSScriptRoot 'requirements-client.txt')
    if ($LASTEXITCODE) { throw 'Client dependency installation failed' }
}
& $pythonExe -m pip check
if ($LASTEXITCODE) { throw 'Client dependencies are inconsistent' }
& $pythonExe -m pip freeze | Set-Content -Encoding UTF8 (Join-Path $reportRoot 'requirements-client-installed.txt')
if ($DownloadServer) {
    Write-InstallState 'server_download_and_extraction' 'running'
    & $BootstrapPython (Join-Path $PSScriptRoot 'install_windows.py') @downloadArgs
    if ($LASTEXITCODE) { throw 'Server download/install failed; rerun to resume' }
}
$env:CARLA_ROOT = $CarlaRoot
$serverExe = Join-Path $CarlaRoot 'CarlaUE4.exe'
if ($StartServer) {
    Write-InstallState 'server_start' 'running'
    if (-not (Test-Path -LiteralPath $serverExe)) { throw "CARLA executable missing: $serverExe" }
    $server = Start-Process -FilePath $serverExe -ArgumentList @('-RenderOffScreen', '-nosound', '-quality-level=Low', "-carla-rpc-port=$Port") -WorkingDirectory $CarlaRoot -WindowStyle Hidden -PassThru
    $server.Id | Set-Content (Join-Path $projectRoot 'simulation/carla/server.pid')
    $ready = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        if ($server.HasExited) { throw "CARLA exited with code $($server.ExitCode)" }
        & $pythonExe -c "import carla; c=carla.Client('127.0.0.1',$Port); c.set_timeout(2.0); print(c.get_server_version())" 2>$null
        if ($LASTEXITCODE -eq 0) { $ready = $true; break }
        Start-Sleep -Seconds 2
    }
    if (-not $ready) { throw 'CARLA did not become ready; inspect server logs' }
}
if ($Smoke) {
    Write-InstallState 'physics_and_sensor_smoke' 'running'
    & $pythonExe (Join-Path $projectRoot 'test/carla_smoke.py') --port $Port
    if ($LASTEXITCODE) { throw 'CARLA physics/sensor smoke test failed' }
}
$doctorArgs = @()
if ($Smoke) { $doctorArgs += '--connect' }
& $pythonExe (Join-Path $PSScriptRoot 'doctor.py') @doctorArgs
if ($LASTEXITCODE) { throw 'Final base CARLA readiness check failed' }
Write-InstallState 'base_carla_environment' 'complete' 'CoLMDriver model dependencies and weights are separate.'
Write-Host "Client Python: $pythonExe"
Write-Host "CARLA_ROOT: $CarlaRoot"
Write-Host 'Base simulator environment only. CoLMDriver model dependencies and weights are checked separately.'
