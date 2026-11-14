# Advanced Prompting System

A hybrid Python + React/TypeScript AI orchestration platform combining:

1. **Multi-Agent Collaborative Debate (`conversation_manager.py`)**: 26 distinct domain personas (`assistant_personalities.py`) performing problem definition consensus, round deliberation, clarifying question extraction, confidence/binary voting, mediator tie-breaking, and format-aware artifact delivery (`.txt`, `.json`, `.pdf`, `.html`, `.csv`, `.py`).
2. **Complexity-Driven Reasoning Pipeline (`advanced_prompting.py`, `complexity_measures.py`)**: Linguistic/AST/entropy/ML complexity assessment, structured `Plan` / `PlanStep` / `Subtask` decomposition, self-consistency, least-to-most, progressive-hint, step reflection/reward judging, and collaborative multi-agent step delegation.
3. **Unified GraphQL API & React 19 SPA (`app.py`, `schema.py`, `execution_modes.py`, `collaborative-assistant-frontend/`)**: Flask + Strawberry GraphQL backend serving `/graphql`, per-run isolated `.aps_artifacts/<run_id>/`, CORS/auth/rate-limit hardening, and a React 19 + TypeScript + Vite 6 + Apollo Client frontend.

---

## Execution Modes

| Mode ID | Backend Engine | Status | Notes |
| :--- | :--- | :--- | :--- |
| `collaborative` | `conversation_manager.run_conversation` | Enabled | Multi-agent debate, voting, consensus summary, artifact generation |
| `advanced_pipeline` | `advanced_prompting.AdvancedPromptEngineer.main` | Enabled | Solo (`agents=0`) or multi-agent (`agents>0`) complexity/step planning + diagnostics |
| `tree_of_thought` | — | **Gated (`enabled=False`)** | Explicitly gated pending [Issue #36](https://github.com/dhar174/advanced_prompting_system/issues/36); zero provider calls permitted |

---

## Quick Start (Local Development)

### Prerequisites

- Python 3.11+ (3.12 recommended)
- Node.js 22+ & npm (for frontend development/builds)
- Optional for live LLM calls (`OFFLINE_MODE=0`): OpenAI API credentials in `.env` (see [`.env.example`](.env.example))
- Optional for full NLP linguistic scorers: `python -m spacy download en_core_web_sm` (safely degrades offline when absent)

### 1. Clone & Install

```bash
git clone https://github.com/dhar174/advanced_prompting_system.git
cd advanced_prompting_system

python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -e ".[dev]"
```

For production WSGI serving (`waitress` on Windows / `gunicorn` on Linux) or Playwright browser E2E testing:

```bash
pip install -e ".[server]"
pip install -e ".[dev,e2e]"
python -m playwright install chromium
```

### 2. Configure Environment

Copy `.env.example` to `.env` (never commit `.env` or real API keys):

```bash
cp .env.example .env
```

All application modules use lazy provider initialization (`provider_client.LazyClient`). Importing Python modules never instantiates an OpenAI client or requires credentials at import time.

### 3. Build Frontend & Run Server

```bash
cd collaborative-assistant-frontend
npm ci
npm run build
cd ..

# Production-style WSGI entrypoint (binds 0.0.0.0:$PORT, default 8000)
python wsgi.py
```

On Linux / containers:

```bash
gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 8 --timeout 120 wsgi:app
```

In development, you can also run the Vite dev server (`npm run dev` on `http://localhost:5173`) alongside `python wsgi.py`, because `/graphql` allows local loopback origins (`localhost`/`127.0.0.1`) by default or configured `APS_ALLOWED_ORIGINS`.

---

## Python Programmatic Usage

```python
from advanced_prompting import AdvancedPromptEngineer, PromptEngineeringConfig
from conversation_manager import OutputType

config = PromptEngineeringConfig(
    output_type=OutputType(output_type="PDF File", file_extension=".pdf"),
    max_steps=5,
    initial_budget=5,
    agents=2,
)
engineer = AdvancedPromptEngineer(config)
interaction = engineer.main("Design an offline-first API reliability checklist.")

print("Final Answer:", interaction.answer)
print("Final Reward:", interaction.final_reward)
print("Plan Steps:", len(interaction.plan.steps) if interaction.plan else 0)
```

---

## Production Packaging & Docker

A multi-stage [`Dockerfile`](Dockerfile) builds the React bundle (`node:22-alpine`) and packages only runtime dependencies (`python:3.12-slim` + `pip install ".[server]"`) under a non-root `appuser`, binding `0.0.0.0:${PORT:-8080}` via `gunicorn wsgi:app`:

```bash
docker build -t advanced-prompting-system .
docker run --rm -p 8080:8080 -e PORT=8080 --env-file .env advanced-prompting-system
```

---

## Offline Qualification & Testing

`conftest.py` enforces **fail-closed offline network isolation** (`sys._aps_network_guard`). Running bare `pytest` without the wrapper intentionally stops with a usage error.

Always execute tests through the guarded offline runners:

### Backend & Integration Tests

**Windows PowerShell:**

```powershell
# Fast backend unit & GraphQL HTTP integration suite (without Playwright browser E2E)
./tools/run-offline.ps1 -PytestArgs '--ignore=test_browser_e2e.py','-q'

# Full backend + GraphQL integration + Playwright Chromium browser E2E suite
./tools/run-offline.ps1 -PytestArgs '-q'

# Playwright browser E2E matrix only (requires built dist/ and Playwright Chromium)
./tools/run-offline.ps1 -PytestArgs 'test_browser_e2e.py','-v'
```

**Linux / macOS:**

```bash
./tools/run-offline.sh --ignore=test_browser_e2e.py -q
./tools/run-offline.sh -q
./tools/run-offline.sh test_browser_e2e.py -v
```

### Frontend Offline Verification

**Windows PowerShell:**

```powershell
./tools/run-frontend-offline.ps1 -Action guard
./tools/run-frontend-offline.ps1 -Action lint
./tools/run-frontend-offline.ps1 -Action test
./tools/run-frontend-offline.ps1 -Action build
```

**Linux / macOS:**

```bash
./tools/run-frontend-offline.sh guard
./tools/run-frontend-offline.sh lint
./tools/run-frontend-offline.sh test
./tools/run-frontend-offline.sh build
```

See [`docs/issue35-qualification.md`](docs/issue35-qualification.md) and [`docs/offline-test-inventory.md`](docs/offline-test-inventory.md) for qualification evidence, bug fixes, and test baselines.
