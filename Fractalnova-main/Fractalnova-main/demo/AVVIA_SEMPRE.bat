@echo off
title FractalNova - avvio in background
powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0keep_alive.ps1"
timeout /t 3 /nobreak >nul
type "%~dp0running.pids.txt"
type "%~dp0public_url.txt" 2>nul
echo.
echo Puoi chiudere questa finestra. I servizi restano attivi.
pause
