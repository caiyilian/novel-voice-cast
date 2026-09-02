param(
    [string]$Endpoint = "http://172.31.102.189:8189",
    [string]$Config = "config\config.yaml",
    [int]$PollSeconds = 30,
    [int]$RestartDelaySeconds = 60,
    [int]$MaxNoProgressRestarts = 3,
    [string]$PipelineLog = "logs\h3_continuous_video.log",
    [string]$LauncherLog = "logs\h3_continuous_launcher.log"
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$env:PYTHONUTF8 = "1"

if ($PollSeconds -lt 5 -or $RestartDelaySeconds -lt 5) {
    throw "PollSeconds and RestartDelaySeconds must be at least 5"
}
if ($MaxNoProgressRestarts -lt 1) {
    throw "MaxNoProgressRestarts must be positive"
}

$launcherLogPath = Join-Path $root $LauncherLog
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $launcherLogPath) | Out-Null

function Write-LauncherLog([string]$Message) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $launcherLogPath -Encoding UTF8 -Value "$stamp $Message"
}

$healthUrl = $Endpoint.TrimEnd("/") + "/api/health"

function Wait-ForH3 {
    Write-LauncherLog "waiting_for=$healthUrl"
    $lastState = ""
    $checks = 0
    while ($true) {
        $checks++
        try {
            $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 15
            $state = "status=$($health.status) comfyui=$($health.comfyui)"
            if ($state -ne $lastState -or $checks % 20 -eq 0) {
                Write-LauncherLog "health $state"
                $lastState = $state
            }
            if ($health.status -eq "ok" -and [bool]$health.comfyui) {
                return
            }
        }
        catch {
            $state = "error=$($_.Exception.Message)"
            if ($state -ne $lastState -or $checks % 20 -eq 0) {
                Write-LauncherLog "health $state"
                $lastState = $state
            }
        }
        Start-Sleep -Seconds $PollSeconds
    }
}

function Get-CompletionMarker {
    $success = 0
    $total = 0
    $checkpointRoot = Join-Path $root "output\h3_video_continuous"
    $checkpoints = Get-ChildItem -LiteralPath $checkpointRoot `
        -Filter "h3_clips.checkpoint.json" -Recurse -File -ErrorAction SilentlyContinue
    foreach ($checkpoint in $checkpoints) {
        try {
            $payload = Get-Content -LiteralPath $checkpoint.FullName -Raw -Encoding UTF8 `
                | ConvertFrom-Json
            $clips = @($payload.clips)
            $total += $clips.Count
            $success += @($clips | Where-Object { $_.status -eq "success" }).Count
        }
        catch {
            Write-LauncherLog "checkpoint_read_error path=$($checkpoint.FullName) error=$($_.Exception.Message)"
        }
    }
    return "$success/$total"
}

$python = Join-Path $root ".venv\Scripts\python.exe"
$noProgressRestarts = 0
Write-LauncherLog "supervisor_started max_no_progress_restarts=$MaxNoProgressRestarts"

while ($true) {
    Wait-ForH3
    $duplicate = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match "python" -and $_.CommandLine -match "scripts[\\/]run_full\.py"
    } | Select-Object -First 1
    if ($duplicate) {
        Write-LauncherLog "pipeline_already_running pid=$($duplicate.ProcessId); supervisor_exiting"
        exit 0
    }

    $before = Get-CompletionMarker
    $started = Get-Date
    Write-LauncherLog "h3_ready starting_pipeline completion=$before"
    & $python -u "scripts\run_full.py" `
        --config $Config `
        --from-stage video `
        --to-stage video `
        --log $PipelineLog
    $exitCode = $LASTEXITCODE
    $elapsed = [Math]::Round(((Get-Date) - $started).TotalSeconds, 1)
    $after = Get-CompletionMarker
    Write-LauncherLog "pipeline_exited code=$exitCode elapsed_seconds=$elapsed completion=$after"
    if ($exitCode -eq 0) {
        exit 0
    }

    if ($after -ne $before -or $elapsed -ge 300) {
        $noProgressRestarts = 0
    }
    else {
        $noProgressRestarts++
    }
    if ($noProgressRestarts -ge $MaxNoProgressRestarts) {
        Write-LauncherLog "supervisor_stopped reason=repeated_no_progress count=$noProgressRestarts"
        exit $exitCode
    }
    Write-LauncherLog "restarting_after_seconds=$RestartDelaySeconds no_progress_count=$noProgressRestarts"
    Start-Sleep -Seconds $RestartDelaySeconds
}
