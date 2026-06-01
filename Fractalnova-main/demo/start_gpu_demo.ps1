# Avvia FractalNova Demo IA su GPU (RTX 5060 Ti)
$ErrorActionPreference = "Stop"
$DemoDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path $DemoDir -Parent
$Python = if ($env:FRACTALNOVA_PYTHON) { $env:FRACTALNOVA_PYTHON } else { "C:\Users\Utente\AppData\Local\Programs\Python\Python314\python.exe" }
$Port = if ($env:FRACTALNOVA_DEMO_PORT) { $env:FRACTALNOVA_DEMO_PORT } else { "8080" }

$env:CUDA_VISIBLE_DEVICES = if ($env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES } else { "0" }
$env:FRACTALNOVA_DEMO_PORT = $Port

Set-Location $Root

Write-Host "[demo] CUDA_VISIBLE_DEVICES=$env:CUDA_VISIBLE_DEVICES"
& $Python -c "import torch; print('[demo] GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# Libera la porta se occupata
$conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($conn) {
    $pid = $conn.OwningProcess
    Write-Host "[demo] porta $Port occupata da PID $pid — termino..."
    Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

Write-Host "[demo] avvio server su http://localhost:$Port"
& $Python (Join-Path $DemoDir "book_demo.py")
