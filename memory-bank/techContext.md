# Technical Context (`advanced_prompting_system`)

## Runtime & Tooling Stack
- **OS & Shell**: Windows 11 (`pwsh`) & Linux CI (`ubuntu-latest`).
- **Python**: CPython 3.11 / 3.12 (`pyproject.toml` requires `>=3.10`).
- **Core Python Dependencies**:
  - `openai>=1.0.0`, `pydantic>=2.0.0`, `flask>=3.0.0`, `strawberry-graphql[flask]>=0.200.0`
  - `numpy>=1.24.0`, `scipy>=1.10.0`, `scikit-learn>=1.3.0`, `pandas>=2.0.0`
  - `jarowinkler>=2.0.0`, `Levenshtein>=0.21.0`, `regex>=2023.0.0`, `tqdm>=4.65.0`
- **Optional NLP Dependencies (`[project.optional-dependencies].nlp`)**:
  - `spacy>=3.7.0`, `nltk>=3.8.0`, `textblob>=0.17.0`, `textstat>=0.7.0`, `tiktoken>=0.5.0`, `transformers>=4.30.0`
- **Frontend Stack (`collaborative-assistant-frontend/`)**:
  - Node.js 22/24, React 19, TypeScript, Vite 6, Tailwind CSS, `@apollo/client`, Lucide React, Vitest 3, React Testing Library, ESLint.

## Authoritative Verification Wrappers
- **Backend (Windows)**: `./tools/run-offline.ps1 -PytestArgs '-q'`
- **Backend (Linux/CI)**: `./tools/run-offline.sh -q`
- **Frontend (Windows)**: `./tools/run-frontend-offline.ps1 -Action guard|test|build|lint`
- **Frontend (Linux/CI)**: `./tools/run-frontend-offline.sh guard|test|build|lint`
