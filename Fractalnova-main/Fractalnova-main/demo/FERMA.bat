@echo off
echo Arresto FractalNova demo e Cloudflare...
powershell -NoProfile -Command "Get-Process cloudflared -EA SilentlyContinue | Stop-Process -Force; Get-NetTCPConnection -LocalPort 8080 -State Listen -EA SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -EA SilentlyContinue }"
echo Fatto.
pause
