# FractalNova - Installa Git hooks (pre-commit, pre-push) su Windows
# Esegui: powershell -ExecutionPolicy Bypass -File scripts/install_hooks.ps1

$ErrorActionPreference = "Stop"
$repoRoot = (Get-Item $PSScriptRoot).Parent.FullName
$hooksDir = Join-Path $repoRoot ".git\hooks"

if (-not (Test-Path (Join-Path $repoRoot ".git"))) {
  Write-Host "Non e' una repo git. Crea prima git init."
  exit 1
}

$preCommitSrc = Join-Path $repoRoot "scripts\git_hooks\pre-commit"
$prePushSrc   = Join-Path $repoRoot "scripts\git_hooks\pre-push"

# Crea hook che invoca il runner Python (cross-platform)
$preCommitHook = @"
#!/bin/sh
ROOT=`$(git rev-parse --show-toplevel)
cd "`$ROOT"
exec python -m security.pre_commit_runner --root "`$ROOT"
"@

$prePushHook = @"
#!/bin/sh
ROOT=`$(git rev-parse --show-toplevel)
cd "`$ROOT"
exec python -m security.pre_commit_runner --root "`$ROOT" --pre-push
"@

$hookPreCommitPath = Join-Path $hooksDir "pre-commit"
$hookPrePushPath   = Join-Path $hooksDir "pre-push"

# Su Windows .git/hooks può usare script bash (Git Bash) o abbiamo bisogno di uno wrapper
# Se esiste Git Bash, copia gli script; altrimenti crea uno stub che chiama Python
if (Test-Path $preCommitSrc) {
  Copy-Item $preCommitSrc $hookPreCommitPath -Force
  Copy-Item $prePushSrc   $hookPrePushPath   -Force
  Write-Host "Hook copiati in .git/hooks (usa Git Bash per eseguirli)."
} else {
  Set-Content -Path $hookPreCommitPath -Value $preCommitHook
  Set-Content -Path $hookPrePushPath   -Value $prePushHook
}

Write-Host "Per eseguire i check manualmente: python -m security.pre_commit_runner"
Write-Host "OK. Hooks in: $hooksDir"
