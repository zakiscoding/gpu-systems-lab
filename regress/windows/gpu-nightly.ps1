# Run the nightly GPU benchmark on the GPU PC's Windows boot, starting WSL only for the run.
# Scheduled by the "GPU Nightly Benchmark" task (install-gpu-nightly.ps1), 03:30 daily.
#
# If WSL was already running (a work session, started by scripts/gpu-pc-up.sh), it is left running.
# Otherwise WSL is shut down afterwards, so nothing sits in memory between runs.
$ErrorActionPreference = 'Continue'
$env:WSL_UTF8 = '1'
$log = Join-Path $env:LOCALAPPDATA 'gpu-nightly.log'
function Log($m) { "$(Get-Date -Format s) $m" | Add-Content -Path $log }

$wasRunning = [bool]((wsl.exe -l --running -q) -match '^Ubuntu$')
Log "start (WSL already running: $wasRunning)"

# Boots the distro if needed, waits for systemd, then runs the oneshot service. The service's
# drop-in starts the GopherGrid and DCGM containers first and stops them afterwards.
$out = wsl.exe -d Ubuntu -u root -- bash -c 'systemctl is-system-running --wait >/dev/null; systemctl start gg-nightly.service; rc=$?; journalctl -u gg-nightly.service -n 4 --no-pager -o cat; exit $rc' 2>&1
$rc = $LASTEXITCODE
$out | ForEach-Object { Log "  $_" }
Log "gg-nightly.service exit $rc"

if (-not $wasRunning) {
    wsl.exe --shutdown
    Log 'WSL shut down'
}
exit $rc
