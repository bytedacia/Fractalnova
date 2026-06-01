# Avvia demo IA + Cloudflare in background (sopravvive alla chiusura di Cursor)
$ErrorActionPreference = "SilentlyContinue"
$DemoDir = $PSScriptRoot
$Root = Split-Path $DemoDir -Parent
$Python = if ($env:FRACTALNOVA_PYTHON) { $env:FRACTALNOVA_PYTHON } else { "C:\Users\Utente\AppData\Local\Programs\Python\Python314\python.exe" }
$Cf = Join-Path $DemoDir "bin\cloudflared.exe"
$Port = if ($env:FRACTALNOVA_DEMO_PORT) { $env:FRACTALNOVA_DEMO_PORT } else { 8080 }

$env:CUDA_VISIBLE_DEVICES = if ($env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES } else { "0" }
$env:FRACTALNOVA_DEMO_PORT = "$Port"

# ferma vecchie istanze
Get-NetTCPConnection -LocalPort $Port -State Listen -EA SilentlyContinue | ForEach-Object {
    Stop-Process -Id $_.OwningProcess -Force -EA SilentlyContinue
}
Get-Process cloudflared -EA SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

$demoLog = Join-Path $DemoDir "demo_server.log"
$demoErr = Join-Path $DemoDir "demo_server.err.log"
$cfOut = Join-Path $DemoDir "cloudflare_url.log"
$cfErr = Join-Path $DemoDir "cloudflare_url.log.err"
$pidFile = Join-Path $DemoDir "running.pids.txt"

# demo IA (GPU)
$pDemo = Start-Process -FilePath $Python -ArgumentList (Join-Path $DemoDir "book_demo.py") `
    -WorkingDirectory $Root -WindowStyle Hidden `
    -RedirectStandardOutput $demoLog -RedirectStandardError $demoErr -PassThru

# cloudflare quick tunnel
$url = ""
if (Test-Path $Cf) {
    $pCf = Start-Process -FilePath $Cf -ArgumentList "tunnel", "--url", "http://127.0.0.1:$Port" `
        -WindowStyle Hidden -RedirectStandardOutput $cfOut -RedirectStandardError $cfErr -PassThru
    for ($i = 0; $i -lt 25; $i++) {
        Start-Sleep -Seconds 2
        if (Test-Path $cfErr) {
            $hit = Select-String -Path $cfErr -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" -AllMatches | Select-Object -Last 1
            if ($hit -and $hit.Matches.Count -gt 0) {
                $url = $hit.Matches[0].Value
                break
            }
        }
    }
} else {
    $pCf = $null
}

@"
FractalNova · servizi attivi ($(Get-Date -Format 'yyyy-MM-dd HH:mm'))
============================================================
Demo locale : http://localhost:$Port
GPU         : RTX 5060 Ti (CUDA_VISIBLE_DEVICES=$env:CUDA_VISIBLE_DEVICES)
Demo PID    : $($pDemo.Id)
Cloudflare  : $(if ($pCf) { $pCf.Id } else { 'non avviato' })
URL pubblico: $(if ($url) { $url } else { 'vedi cloudflare_url.log.err' })
Fisso (cfg) : https://demo.fractalnova.live
============================================================
Per fermare: Get-Process cloudflared | Stop-Process -Force
             Get-NetTCPConnection -LocalPort $Port | %% { Stop-Process -Id `$_.OwningProcess -Force }
"@ | Set-Content -Path $pidFile -Encoding UTF8

if ($url) {
    @"
# URL pubblico attivo
$url

# Dominio fisso (tunnel nominato)
https://demo.fractalnova.live
"@ | Set-Content -Path (Join-Path $DemoDir "public_url.txt") -Encoding UTF8
}
