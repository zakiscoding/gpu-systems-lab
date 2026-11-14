# Install the Windows side of the on-demand GPU PC (run once on the GPU PC, as the WSL user).
#
# - "GPU Nightly Benchmark": 03:30 daily, starts WSL, runs the benchmark, shuts WSL down again.
#   Runs whether or not anyone is logged in (S4U), and catches up after the PC was off.
# - "WSL Keep Alive" is disabled: WSL no longer runs all the time. scripts/gpu-pc-up.sh
#   enables it for a work session and scripts/gpu-pc-down.sh disables it again.
#
# Undo: Unregister-ScheduledTask 'GPU Nightly Benchmark'; Enable-ScheduledTask 'WSL Keep Alive'
param([string]$ScriptDir = "$env:USERPROFILE\homelab")

New-Item -ItemType Directory -Force $ScriptDir | Out-Null
Copy-Item -Force (Join-Path $PSScriptRoot 'gpu-nightly.ps1') $ScriptDir

$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$ScriptDir\gpu-nightly.ps1`""
$trigger = New-ScheduledTaskTrigger -Daily -At 3:30am
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 45) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
Register-ScheduledTask -TaskName 'GPU Nightly Benchmark' -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal -Force `
    -Description 'Starts WSL, runs the GopherGrid nightly GPU benchmark, then shuts WSL down. gpu-systems-lab regress/windows.' | Out-Null

Disable-ScheduledTask -TaskName 'WSL Keep Alive' | Out-Null
Get-ScheduledTask 'GPU Nightly Benchmark', 'WSL Keep Alive' | Format-Table TaskName, State
