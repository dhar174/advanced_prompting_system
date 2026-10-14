param([ValidateSet('test','lint','build','guard')][string]$Action = 'test')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$localRoot = Join-Path $projectRoot '.qualification'
$env:TEMP = Join-Path $localRoot 'tmp'
$env:TMP = $env:TEMP
$env:npm_config_cache = Join-Path $localRoot 'cache\npm'
$env:npm_config_update_notifier = 'false'
$guardPath = (Join-Path $PSScriptRoot 'offline\node-network-guard.cjs').Replace('\', '/')
$env:NODE_OPTIONS = '--require="' + $guardPath + '"'
Get-ChildItem Env: | Where-Object { $_.Name -match '(API_KEY|ACCESS_TOKEN|AUTH_TOKEN)$|^(OPENAI_ADMIN_KEY|HF_TOKEN)$' } | Remove-Item
Set-Location -LiteralPath (Join-Path $projectRoot 'collaborative-assistant-frontend')
switch ($Action) {
    'test' { & node node_modules/vitest/vitest.mjs run }
    'lint' { & node node_modules/eslint/bin/eslint.js . }
    'build' {
        & node node_modules/typescript/bin/tsc -b
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & node node_modules/vite/bin/vite.js build
    }
    'guard' {
        & node -e 'const net=require("node:net"); if(process.env.APS_NODE_NETWORK_GUARD!=="1")process.exit(2); try {net.connect(443,"203.0.113.1");process.exit(3)} catch(e) {if(!e.message.includes("Offline test"))throw e; console.log("Node network denial verified")}'
    }
}
exit $LASTEXITCODE
