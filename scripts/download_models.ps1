# requires -Version 5.1
param()

Write-Host "Downloading FractalNova models from Hugging Face..."

$env:HF_TOKEN = if ($env:HF_TOKEN) { $env:HF_TOKEN } elseif ($env:HUGGINGFACE_HUB_TOKEN) { $env:HUGGINGFACE_HUB_TOKEN } else { $null }

$models = @(
  @{ env = 'QWEN_LOCAL_MODEL_PATH'; id = $env:QWEN_REPO_ID; fallback='Qwen/Qwen3-8B'; rev=$env:QWEN_REVISION },
  @{ env = 'LLAMA_LOCAL_MODEL_PATH'; id = $env:LLAMA_REPO_ID; fallback='meta-llama/Meta-Llama-3-8B-Instruct'; rev=$env:LLAMA_REVISION },
  @{ env = 'GEMMA_LOCAL_MODEL_PATH'; id = $env:GEMMA_REPO_ID; fallback='google/gemma-7b-it'; rev=$env:GEMMA_REVISION },
  @{ env = 'DEEPSEEK_LOCAL_PATH'; id = $env:DEEPSEEK_REPO_ID; fallback='deepseek-ai/DeepSeek-V3'; rev=$env:DEEPSEEK_REVISION },
  @{ env = 'FLUX_LOCAL_MODEL_PATH'; id = $env:FLUX_REPO_ID; fallback=($env:FLUX_MODEL_ID ? $env:FLUX_MODEL_ID : 'black-forest-labs/FLUX.1-schnell'); rev=$env:FLUX_REVISION }
)

$baseDir = Join-Path $PSScriptRoot '..' | Resolve-Path
$modelsDir = Join-Path $baseDir 'models'
New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null

$ErrorActionPreference = 'Stop'

# Ensure huggingface-cli
pip show huggingface_hub > $null 2>&1
if ($LASTEXITCODE -ne 0) {
  Write-Host "Installing huggingface_hub..."
  pip install huggingface_hub
}

foreach ($m in $models) {
  $id = if ($m.id) { $m.id } else { $m.fallback }
  $local = Join-Path $modelsDir ($id.Split('/')[-1])
  Write-Host "Downloading $id -> $local"
  $py = @"
import os
from huggingface_hub import snapshot_download
repo_id = r'''$id'''
local_dir = r'''$local'''
revision = r'''$($m.rev)'''
revision = revision if revision else None
snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False, token=os.getenv('HF_TOKEN'), revision=revision)
print(local_dir)
"@
  python - <<PY
$py
PY
  if ($m.env) {
    [Environment]::SetEnvironmentVariable($m.env, $local, 'User')
    Write-Host "Set $($m.env) = $local"
  }
}

Write-Host "Done. Restart your shell to use the new environment variables."
