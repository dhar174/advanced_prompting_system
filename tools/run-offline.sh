#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
local_root="$project_root/.qualification"
mkdir -p "$local_root/tmp" "$local_root/cache/nltk" "$local_root/cache/hf" "$local_root/cache/tiktoken" "$local_root/cache/matplotlib" "$local_root/pytest-tmp"

export APS_OFFLINE_TESTS="1"
export OFFLINE_MODE="1"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"
export PYTHONDONTWRITEBYTECODE="1"
export PYTHONPATH="$project_root/tools/offline:$project_root${PYTHONPATH:+:$PYTHONPATH}"
export TEMP="$local_root/tmp"
export TMP="$TEMP"
export HF_HOME="$local_root/cache/hf"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export NLTK_DATA="$local_root/cache/nltk"
export TIKTOKEN_CACHE_DIR="$local_root/cache/tiktoken"
export MPLCONFIGDIR="$local_root/cache/matplotlib"
export XDG_CACHE_HOME="$local_root/cache"
export PYTHONUTF8="1"

# Unset any API credentials
unset OPENAI_API_KEY OPENAI_ADMIN_KEY HF_TOKEN

python_exe="$local_root/venv/bin/python"
if [ ! -x "$python_exe" ]; then
    python_exe="python3"
    if ! command -v "$python_exe" &>/dev/null; then
        python_exe="python"
    fi
fi

exec "$python_exe" -m pytest "${@:- -q}"
