param([string[]]$PytestArgs = @('-q'))
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $projectRoot
$localRoot = Join-Path $projectRoot '.qualification'
foreach ($directory in @('tmp','cache','cache\nltk','cache\hf','cache\tiktoken','cache\matplotlib')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $localRoot $directory) | Out-Null
}
$env:APS_OFFLINE_TESTS = '1'
$env:OFFLINE_MODE = '1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = (Join-Path $PSScriptRoot 'offline') + [IO.Path]::PathSeparator + $projectRoot
$env:TEMP = Join-Path $localRoot 'tmp'
$env:TMP = $env:TEMP
$env:HF_HOME = Join-Path $localRoot 'cache\hf'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:NLTK_DATA = Join-Path $localRoot 'cache\nltk'
$env:TIKTOKEN_CACHE_DIR = Join-Path $localRoot 'cache\tiktoken'
$env:MPLCONFIGDIR = Join-Path $localRoot 'cache\matplotlib'
$env:XDG_CACHE_HOME = Join-Path $localRoot 'cache'
$env:PYTHONUTF8 = '1'
Get-ChildItem Env: | Where-Object { $_.Name -match '(API_KEY|ACCESS_TOKEN|AUTH_TOKEN)$|^(OPENAI_ADMIN_KEY|HF_TOKEN)$' } | Remove-Item
& (Join-Path $localRoot 'venv\Scripts\python.exe') -m pytest @PytestArgs
exit $LASTEXITCODE
