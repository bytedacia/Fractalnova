# Avvio completo: IA su GPU + tunnel Cloudflare (per live / condividere la demo)
# Uso:  .\demo\run_live.ps1
#       .\demo\run_live.ps1 -SkipTunnel    # solo GPU, senza Cloudflare

param([switch]$SkipTunnel, [switch]$NamedTunnel)

$ErrorActionPreference = "Stop"
$DemoDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path $DemoDir -Parent
$Python = if ($env:FRACTALNOVA_PYTHON) { $env:FRACTALNOVA_PYTHON } else { "C:\Users\Utente\AppData\Local\Programs\Python\Python314\python.exe" }
$Port = if ($env:FRACTALNOVA_DEMO_PORT) { $env:FRACTALNOVA_DEMO_PORT } else { 8080 }

$env:CUDA_VISIBLE_DEVICES = if ($env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES } else { "0" }
$env:FRACTALNOVA_DEMO_PORT = "$Port"

Set-Location $Root

Write-Host "=== FractalNova LIVE (GPU + Cloudflare) ===" -ForegroundColor Cyan
& $Python -c @"
import torch
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
else:
    print('ATTENZIONE: CUDA non disponibile')
"@

# Libera porta
$conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($conn) {
    Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

# Avvia demo in background
$demoLog = Join-Path $DemoDir "demo_server.log"
Write-Host "[demo] avvio IA in background (log: $demoLog)"
$demoJob = Start-Process -FilePath $Python -ArgumentList (Join-Path $DemoDir "book_demo.py") `
    -WorkingDirectory $Root -RedirectStandardOutput $demoLog -RedirectStandardError $demoLog -PassThru -WindowStyle Hidden

# Attendi server
$deadline = (Get-Date).AddMinutes(3)
$ready = $false
while ((Get-Date) -lt $deadline) {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/api/info" -UseBasicParsing -TimeoutSec 3
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch {}
    Start-Sleep -Seconds 2
}

if (-not $ready) {
    Write-Host "[demo] server non risponde ancora; warmup in corso (modello GPU)..." -ForegroundColor Yellow
    try { Invoke-WebRequest -Uri "http://127.0.0.1:$Port/api/warmup" -Method POST -UseBasicParsing -TimeoutSec 600 | Out-Null } catch {}
}

try {
    $info = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/info" -TimeoutSec 10
    Write-Host "[demo] locale: http://localhost:$Port" -ForegroundColor Green
    Write-Host "[demo] modello: $($info.model) | GPU: $($info.gpu) | CUDA: $($info.cuda)" -ForegroundColor Green
} catch {
    Write-Host "[demo] errore avvio — controlla $demoLog" -ForegroundColor Red
    Get-Content $demoLog -Tail 30 -ErrorAction SilentlyContinue
    exit 1
}

if ($SkipTunnel) {
    Write-Host "[cloudflare] saltato (-SkipTunnel). PID demo: $($demoJob.Id)"
    exit 0
}

Write-Host ""
Write-Host "[cloudflare] dominio: https://demo.fractalnova.live -> http://127.0.0.1:$Port" -ForegroundColor Cyan
Write-Host "  cloudflared tunnel run --token `$env:CLOUDFLARE_TUNNEL_TOKEN"
Write-Host "  oppure quick: cloudflared tunnel --url http://127.0.0.1:$Port"
Write-Host "[demo] PID server: $($demoJob.Id) — log: $demoLog"
