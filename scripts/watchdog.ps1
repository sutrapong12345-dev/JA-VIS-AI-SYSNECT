# JARVIS Watchdog
# Keeps the backend (uvicorn) and the Cloudflare quick tunnel alive.
# When the tunnel's ephemeral URL changes (every fresh start), it rewrites
# frontend/index.html's API_BASE and auto-commits + pushes so the public
# GitHub Pages frontend always points at the current tunnel.
# Meant to be launched once (at Windows logon, via the registered Scheduled
# Task) and left running forever in the background.

$ErrorActionPreference = 'Stop'

$RootDir      = Split-Path -Parent $PSScriptRoot
$BackendDir   = Join-Path $RootDir 'backend'
$FrontendFile = Join-Path $RootDir 'frontend\index.html'
$LogDir       = Join-Path $RootDir 'logs'
$WatchdogLog  = Join-Path $LogDir 'watchdog.log'
$BackendOut   = Join-Path $LogDir 'watchdog_backend.out.log'
$BackendErr   = Join-Path $LogDir 'watchdog_backend.err.log'
$TunnelOut    = Join-Path $LogDir 'watchdog_tunnel.out.log'
$TunnelErr    = Join-Path $LogDir 'watchdog_tunnel.err.log'
# Always the project venv — the system Python may miss packages installed
# only in the venv, and mixing the two led to orphan/ghost backend processes.
$PythonExe    = Join-Path $RootDir 'venv\Scripts\python.exe'
$CloudflaredExe = 'C:\Program Files (x86)\cloudflared\cloudflared.exe'

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Add-Content -Path $WatchdogLog -Value $line -Encoding utf8
}

# Single-instance guard: a named Mutex is process/session-independent, so this
# reliably blocks a second copy even if something (a re-fired logon trigger, a
# manual double-launch) tries to start another watchdog while one is already
# running — prevents duplicate tunnels/backends fighting each other and git
# lock-file collisions from two instances committing at once.
$mutex = New-Object System.Threading.Mutex($false, 'Global\JARVIS_Watchdog_Mutex')
if (-not $mutex.WaitOne(0)) {
    Log 'Another watchdog instance is already running — exiting.'
    exit 0
}

function Test-BackendHealthy {
    try {
        $null = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/health' -TimeoutSec 5
        return $true
    } catch {
        return $false
    }
}

# Cloudflare quick tunnels can die at Cloudflare's edge (hostname silently
# deregistered/DNS dropped) while the local cloudflared.exe process keeps
# running as a zombie that thinks it's still connected. A process-alive
# check alone can't see that — only an actual request through the public
# URL proves the tunnel still works end-to-end.
function Test-TunnelHealthy($url) {
    try {
        $null = Invoke-RestMethod -Uri "$url/api/health" -TimeoutSec 8
        return $true
    } catch {
        return $false
    }
}

function Start-Backend {
    Log 'Starting backend (uvicorn)...'
    $conns = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $conns) {
        try { Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue } catch {}
    }
    Start-Sleep -Seconds 1
    if (Test-Path $BackendOut) { Remove-Item $BackendOut -Force -ErrorAction SilentlyContinue }
    if (Test-Path $BackendErr) { Remove-Item $BackendErr -Force -ErrorAction SilentlyContinue }
    $p = Start-Process -FilePath $PythonExe `
        -ArgumentList '-m','uvicorn','main:app','--host','127.0.0.1','--port','8000' `
        -WorkingDirectory $BackendDir `
        -WindowStyle Hidden `
        -RedirectStandardOutput $BackendOut `
        -RedirectStandardError $BackendErr `
        -PassThru
    return $p.Id
}

function Get-TunnelUrlFromLog {
    $text = ''
    foreach ($f in @($TunnelOut, $TunnelErr)) {
        if (Test-Path $f) { $text += (Get-Content $f -Raw -ErrorAction SilentlyContinue) }
    }
    if (-not $text) { return $null }
    # cloudflared's own log lines mention https://api.trycloudflare.com (the
    # registration endpoint) while REQUESTING a tunnel — that is never a real
    # tunnel hostname, but the old first-match regex kept committing it to
    # GitHub Pages whenever tunnel creation was slow, killing the frontend.
    # Real quick-tunnel URLs are word-word-word-word subdomains; take the
    # LAST non-api match so we never return a stale URL from an old start.
    $all = [regex]::Matches($text, 'https://[a-zA-Z0-9-]+\.trycloudflare\.com') |
        ForEach-Object { $_.Value } |
        Where-Object { $_ -ne 'https://api.trycloudflare.com' }
    $all = @($all)
    if ($all.Count -gt 0) { return $all[-1] }
    return $null
}

function Start-Tunnel {
    Log 'Starting cloudflared tunnel...'
    Get-Process -Name 'cloudflared' -ErrorAction SilentlyContinue | ForEach-Object {
        try { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue } catch {}
    }
    Start-Sleep -Seconds 1
    foreach ($f in @($TunnelOut, $TunnelErr)) {
        if (Test-Path $f) { Remove-Item $f -Force -ErrorAction SilentlyContinue }
    }
    $p = Start-Process -FilePath $CloudflaredExe `
        -ArgumentList 'tunnel','--url','http://127.0.0.1:8000' `
        -WindowStyle Hidden `
        -RedirectStandardOutput $TunnelOut `
        -RedirectStandardError $TunnelErr `
        -PassThru
    $deadline = (Get-Date).AddSeconds(30)
    $url = $null
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 2
        $url = Get-TunnelUrlFromLog
        if ($url) { break }
    }
    return [PSCustomObject]@{ Pid = $p.Id; Url = $url }
}

function Update-FrontendApiBase($newUrl) {
    $content = Get-Content $FrontendFile -Raw -Encoding UTF8
    $pattern = "const API_BASE = window\.JARVIS_API_BASE \|\| '([^']*)'"
    $match = [regex]::Match($content, $pattern)
    if (-not $match.Success) {
        Log 'WARNING: could not find API_BASE line in frontend/index.html'
        return $false
    }
    $oldUrl = $match.Groups[1].Value
    if ($oldUrl -eq $newUrl) { return $false }
    $updated = [regex]::Replace($content, $pattern, "const API_BASE = window.JARVIS_API_BASE || '$newUrl'")
    Set-Content -Path $FrontendFile -Value $updated -Encoding UTF8 -NoNewline
    Log "frontend/index.html API_BASE updated: $oldUrl -> $newUrl"
    return $true
}

function Push-FrontendUpdate($newUrl) {
    # NOTE: git's own progress/summary lines go to stderr, which under the
    # script-wide $ErrorActionPreference='Stop' gets promoted into a
    # terminating error on ANY redirection (2>&1/*>&1) even on success —
    # a documented PowerShell 5.1 native-command quirk. Scope EAP down to
    # 'Continue' here and rely on $LASTEXITCODE for the real result.
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    Push-Location $RootDir
    try {
        # Never let the deployment watchdog commit feature work that happens to
        # be in frontend/index.html. Auto-push is allowed only when every changed
        # content line is the single API_BASE fallback URL.
        $diffLines = git diff --unified=0 -- frontend/index.html 2>$null
        $contentChanges = @($diffLines | Where-Object {
            ($_ -match '^[+-]') -and ($_ -notmatch '^(---|\+\+\+)')
        })
        $unsafeChanges = @($contentChanges | Where-Object {
            $_ -notmatch "^[+-]const API_BASE = window\.JARVIS_API_BASE \|\| 'https://[a-zA-Z0-9-]+\.trycloudflare\.com';$"
        })
        if ($unsafeChanges.Count -gt 0 -or $contentChanges.Count -gt 2) {
            Log 'SECURITY: skipped auto-commit because frontend/index.html contains non-URL changes.'
            return
        }
        git add frontend/index.html 2>$null | Out-Null
        $status = git status --porcelain frontend/index.html
        if ($status) {
            git commit -m "chore: auto-update tunnel URL ($newUrl)" 2>$null | Out-Null
            $pushOutput = git push 2>&1 | Out-String
            if ($LASTEXITCODE -eq 0) {
                Log 'Pushed updated API_BASE to GitHub.'
            } else {
                Log "Git push FAILED (exit $LASTEXITCODE): $pushOutput"
            }
        } else {
            Log 'Nothing to push (git already clean).'
        }
    } catch {
        Log "Git push threw an exception: $($_.Exception.Message)"
    } finally {
        Pop-Location
        $ErrorActionPreference = $prevEAP
    }
}

# ===== Main loop =====
Log '=== JARVIS Watchdog started ==='

$tunnelPid = $null

while ($true) {
    if (-not (Test-BackendHealthy)) {
        Log 'Backend unhealthy/down — restarting.'
        Start-Backend | Out-Null
        $tries = 0
        while (-not (Test-BackendHealthy) -and $tries -lt 10) {
            Start-Sleep -Seconds 2
            $tries++
        }
        if (Test-BackendHealthy) { Log 'Backend is back online.' }
        else { Log 'WARNING: backend still not responding after restart attempt.' }
    }

    $tunnelAlive = $false
    if ($tunnelPid -and (Get-Process -Id $tunnelPid -ErrorAction SilentlyContinue)) {
        $currentUrl = Get-TunnelUrlFromLog
        if ($currentUrl -and (Test-TunnelHealthy $currentUrl)) {
            $tunnelAlive = $true
        }
    }
    if (-not $tunnelAlive) {
        if ($tunnelPid) {
            Log 'Tunnel process alive but unresponsive (edge likely dropped it) — restarting.'
        } else {
            Log 'Tunnel not running — starting a new one.'
        }
        $result = Start-Tunnel
        $tunnelPid = $result.Pid
        if ($result.Url) {
            Log "Tunnel URL: $($result.Url)"
            $changed = Update-FrontendApiBase $result.Url
            if ($changed) { Push-FrontendUpdate $result.Url }
        } else {
            Log 'WARNING: could not detect tunnel URL from cloudflared output within 30s.'
        }
    }

    Start-Sleep -Seconds 30
}
