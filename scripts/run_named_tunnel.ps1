param(
    [string]$TunnelName = 'jarvis-sysnect',
    [string]$ConfigPath = "$env:USERPROFILE\.cloudflared\config.yml"
)

$ErrorActionPreference = 'Stop'
$CloudflaredExe = 'C:\Program Files (x86)\cloudflared\cloudflared.exe'

if (-not (Test-Path -LiteralPath $CloudflaredExe)) {
    throw "cloudflared not found: $CloudflaredExe"
}
if (-not (Test-Path -LiteralPath $ConfigPath)) {
    throw "Named Tunnel config not found: $ConfigPath"
}

& $CloudflaredExe tunnel --config $ConfigPath ingress validate
if ($LASTEXITCODE -ne 0) {
    throw 'Cloudflare ingress configuration is invalid'
}

& $CloudflaredExe tunnel --config $ConfigPath run $TunnelName
