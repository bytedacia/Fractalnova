# Push su https://github.com/bytedacia/scribenova.git
# Un solo commit = un solo contributor su GitHub
# Esegui dalla cartella del progetto (parent di scripts/)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

Write-Host "Remote:" (git remote get-url origin)
Write-Host "Branch: main (1 commit)"
Write-Host ""
Write-Host "Eseguo: git push -u origin main"
Write-Host "Se il repo su GitHub ha gia' commit, usa: git push -u origin main --force"
Write-Host ""

$force = $args -contains "--force"
if ($force) {
    git push -u origin main --force
} else {
    git push -u origin main
}

Write-Host "Fatto. Su GitHub vedrai un solo contributor (tu)."
