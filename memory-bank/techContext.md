# Technical Context (`advanced_prompting_system`)

## Runtime & Tooling Stack
- **OS & Shell**: Windows 11 (`pwsh`) & Linux CI (`ubuntu-latest`).
- **Python**: CPython 3.11 / 3.12 (`pyproject.toml` requires `>=3.10`).
- **Core Python Dependencies**:
  - `openai>=1.0.0`, `pydantic>=2.0.0`, `flask>=3.0.0`, `strawberry-graphql[flask]>=0.200.0`
  - `numpy>=1.24.0`, `scipy>=1.10.0`, `scikit-learn>=1.3.0`, `pandas>=2.0.0`
  - `jarowinkler>=2.0.0`, `Levenshtein>=0.21.0`, `regex>=2023.0.0`, `tqdm>=4.65.0`
  - `spacy>=3.7.0`, `nltk>=3.8.0`, `textblob>=0.17.0`, `textstat>=0.7.0`, `tiktoken>=0.5.0`
- **Optional NLP Dependencies (`[project.optional-dependencies].nlp`)**:
  - `transformers>=4.30.0`; the unconditionally imported NLP packages above are required runtime dependencies.
- **Frontend Stack (`collaborative-assistant-frontend/`)**:
  - Node.js 22/24, React 19, TypeScript, Vite 6, Tailwind CSS, `@apollo/client`, Lucide React, Vitest 3, React Testing Library, ESLint.

## Authoritative Verification Wrappers
- **Backend (Windows)**: `./tools/run-offline.ps1 -PytestArgs '-q'`
- **Backend (Linux/CI)**: `./tools/run-offline.sh -q`
- **Frontend (Windows)**: `./tools/run-frontend-offline.ps1 -Action guard|test|build|lint`
- **Frontend (Linux/CI)**: `./tools/run-frontend-offline.sh guard|test|build|lint`

## Canonical Provider Policy
- `APS_PROVIDER_MAX_CALLS` defaults to 40 (canonical configured range 1–200); an explicit standalone constructor limit can be zero and can only tighten the environment ceiling.
- `APS_PROVIDER_TIMEOUT_SECONDS` defaults to 120 (positive finite value at most 600); constructor timeout can only tighten it. `APS_DEFAULT_MODEL`, `APS_ALLOWED_MODELS` and `APS_DEFAULT_TEMPERATURE` complete the canonical provider settings.
- Canonical policy parsing is strict. Broader aliases and `.env` startup resolution remain #56/#57; do not infer those repairs from #51.
