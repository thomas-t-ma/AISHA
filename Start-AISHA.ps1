param(
    [string]$Profile = "mock",
    [switch]$Vision,
    [switch]$SkipInstall
)

$launcher = Join-Path $PSScriptRoot "infrastructure\scripts\start_windows_dev.ps1"
& $launcher -Profile $Profile -Vision:$Vision -SkipInstall:$SkipInstall
