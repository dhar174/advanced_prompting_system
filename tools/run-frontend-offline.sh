#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
local_root="$project_root/.qualification"
action="${1:-test}"

mkdir -p "$local_root/tmp" "$local_root/cache/npm"
export TEMP="$local_root/tmp"
export TMP="$TEMP"
export npm_config_cache="$local_root/cache/npm"
export npm_config_update_notifier="false"
guard_path="$project_root/tools/offline/node-network-guard.cjs"
export NODE_OPTIONS="--require=$guard_path"

unset OPENAI_API_KEY OPENAI_ADMIN_KEY HF_TOKEN

cd "$project_root/collaborative-assistant-frontend"
case "$action" in
    test)
        exec node node_modules/vitest/vitest.mjs run
        ;;
    lint)
        exec node node_modules/eslint/bin/eslint.js .
        ;;
    build)
        node node_modules/typescript/bin/tsc -b
        exec node node_modules/vite/bin/vite.js build
        ;;
    guard)
        exec node "$project_root/tools/offline/guard-probe.cjs"
        ;;
    *)
        echo "Unknown action: $action" >&2
        exit 1
        ;;
esac
