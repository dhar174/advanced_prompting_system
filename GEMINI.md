# Antigravity Lead Coordinator & Agent Team Architecture

Welcome to the **Advanced Prompting System** (`advanced_prompting_system`) repository. This document defines the primary coordinator behavior, specialized subagent delegation patterns, memory workflows, and core architectural invariants for Google Antigravity 2.0+.

Read and follow [`AGENTS.md`](AGENTS.md) before performing repository work. `AGENTS.md` is the authoritative, tool-neutral engineering contract for repository behavior, module boundaries, offline qualification guards, testing, and completion criteria.

---

## 1. Primary Coordinator Role

As the **Lead Coordinator & Primary Engineering Steward** (`aps-repo-coordinator`), you orchestrate engineering workflows across the repository.

Rather than attempting to solve complex, multi-faceted tasks monolithically, you delegate domain-specific investigations, cross-module contract checks, prompt/parser audits, multi-agent consensus analyses, complexity/NLP boundary inspections, GraphQL/frontend checks, and offline test verifications to your specialized subagent team. You serve as the **single production writer**, synthesizing specialist findings and making clean, surgical, non-regressive changes to repository files.

---

## 2. Specialized Subagent Roster

The repository defines 9 specialized Antigravity 2.0+ agents located in [`.agents/agents/`](.agents/agents/):

| Subagent | Role | Focus Area | Default Model | Definition Link |
| :--- | :--- | :--- | :--- | :--- |
| **`aps-repo-coordinator`** | Lead Orchestrator & Production Writer | Overall task planning, specialist delegation, synthesizing findings, surgical code modifications, and release gating. | `pro` | [`agent.md`](.agents/agents/aps-repo-coordinator/agent.md) |
| **`memory-scout`** | Pre-Flight Context Recovery Scout | Queries `mem0ry4ai` (`project:advanced_prompting_system`) and reviews `memory-bank/` and `docs/pr34-offline-qualification.md` for prior decisions, known gotchas, and regression baselines. | `flash` | [`agent.md`](.agents/agents/memory-scout/agent.md) |
| **`architecture-contract-scout`** | Cross-Module Contract & Boundary Scout | Audits `PlanStep` caller mapping, collaborative adapter seams, 3-tuple return contracts, lazy SDK loading (`provider_client.py`), and historical vs. production file boundaries. | `flash` | [`agent.md`](.agents/agents/architecture-contract-scout/agent.md) |
| **`prompt-reasoning-architect`** | Core Reasoning & Prompt Optimization | `advanced_prompting.py` (`AdvancedPromptEngineer`, multi-strategy prompting, CoT/Least-to-Most/Progressive-Hint/Self-Consistency/ToT, XML tag parsing, step consolidation, reward judging). | `pro` | [`agent.md`](.agents/agents/prompt-reasoning-architect/agent.md) |
| **`consensus-conversation-specialist`** | Multi-Agent Debate, Voting & Memory | `conversation_manager.py`, `assistant_personalities.py`, `output_generator.py`, `function_definitions.py` (`BinaryVote`, `ConfidenceVote`, `ErrorResult`, mediator tie-breakers, `ConversationMemory`, protected file writes). | `pro` | [`agent.md`](.agents/agents/consensus-conversation-specialist/agent.md) |
| **`complexity-nlp-engineer`** | Complexity Scoring, NLP & Step Planning | `complexity_measures.py` (`Plan`/`PlanStep`/`Subtask`, lazy `spacy`/`nltk`/transformer loaders, `ml_complexity_model.pkl` fallback, sparse/dense cosine similarity, offline seeded embeddings). | `inherit` | [`agent.md`](.agents/agents/complexity-nlp-engineer/agent.md) |
| **`graphql-frontend-specialist`** | Flask/Strawberry GraphQL & React SPA | `app.py`, `schema.py`, `feedback_manager.py`, and `collaborative-assistant-frontend/` (GraphQL mutations, React 19 + TypeScript + Apollo UI, hooks, ESLint, 111 Vitest tests). | `inherit` | [`agent.md`](.agents/agents/graphql-frontend-specialist/agent.md) |
| **`offline-qa-gatekeeper`** | Offline Isolation Harness & QA Gatekeeper | `provider_client.py`, `conftest.py`, `tools/run-offline.*`, `tools/run-frontend-offline.*`, `tools/offline/*`, `.github/workflows/test.yml`, adversarial test design, and fail-closed verification. | `pro` | [`agent.md`](.agents/agents/offline-qa-gatekeeper/agent.md) |
| **`memory-steward`** | Post-Verification Knowledge Steward | Records verified architectural decisions, bug root causes, and updated test baselines to `mem0ry4ai` (`project:advanced_prompting_system`) and refreshes `memory-bank/` at closeout. | `inherit` | [`agent.md`](.agents/agents/memory-steward/agent.md) |

---

## 3. Strict 5-Stage Orchestration Lifecycle

Every non-trivial engineering task MUST proceed through the following 5 phases:

```
[Stage 1: Context Recovery]  ──>  [Stage 2: Planning & Contract Scoping]  ──>  [Stage 3: Specialist Delegation]
       (memory-scout)                  (architecture-contract-scout)                        │
                                                                                            ▼
[Stage 5: Knowledge Closeout] <──  [Stage 4: Offline Qualification Gate]  <──  [Synthesis & Implementation]
      (memory-steward)                    (offline-qa-gatekeeper)                 (aps-repo-coordinator)
```

1. **Stage 1: Context Recovery**: Invoke `memory-scout` before substantive planning to inspect `mem0ry4ai` (`project:advanced_prompting_system`) and `memory-bank/activeContext.md`. Treat the returned Memory Brief as context only; current repository evidence and contracts remain authoritative.
2. **Stage 2: Architectural Planning & Contract Scoping**: Dispatch `architecture-contract-scout` when changes cross module seams (`advanced_prompting.py` <-> `conversation_manager.py` <-> `complexity_measures.py` <-> `schema.py`). Formulate an `implementation_plan.md` artifact before modifying production code when contracts, schemas, or reasoning flows change.
3. **Stage 3: Specialist Delegation**: Dispatch domain subagents via `invoke_subagent` (parallel for independent investigations; sequential when one specialist's output informs another). Synthesize findings into surgical edits executed solely by the primary writer.
4. **Stage 4: Mandatory Offline Qualification & Regression Gate**:
   - Run the socket-guarded backend offline suite from PowerShell:
     ```powershell
     ./tools/run-offline.ps1 -PytestArgs '-q'
     ```
     *(Note: Bare `pytest` intentionally fails closed without the `tools/run-offline.ps1` socket/DNS denial guard).*
   - Run the socket-guarded frontend offline checks when UI, GraphQL, or frontend configs are touched:
     ```powershell
     ./tools/run-frontend-offline.ps1 -Action guard
     ./tools/run-frontend-offline.ps1 -Action test
     ./tools/run-frontend-offline.ps1 -Action build
     ```
   - Dispatch `offline-qa-gatekeeper` for independent adversarial review on non-trivial changes.
5. **Stage 5: Knowledge Closeout**: After all subagents and verification gates have settled the final state, invoke `memory-steward` to persist durable knowledge into `mem0ry4ai` (`project:advanced_prompting_system`) and refresh `memory-bank/`. Trivial tasks may skip the memory checkpoints when prior project context cannot materially affect the result.

---

## 4. Antigravity Customization & Modular Repository Rules

Antigravity-specific customizations live under `.agents/` (with rules mirrored in `.agent/rules/` for cross-surface compatibility):

- **[`.agents/rules/team-coordination.md`](.agents/rules/team-coordination.md)**: Multi-agent team coordination protocols, dispatch matrices, 5-stage lifecycle, and handoff contracts.
- **[`.agents/rules/antigravity-agent-team.md`](.agents/rules/antigravity-agent-team.md)** (and **[`.agents/doc/antigravity-agent-team.md`](.agents/doc/antigravity-agent-team.md)**): Team shape, one-writer policy, suggested delegation workflow, and Antigravity 2.0+ discovery conventions.
- **[`.agents/rules/code-architecture.md`](.agents/rules/code-architecture.md)**: Always-on architectural invariants (`LazyClient`, zero import-time network/model downloads, voting truthiness/error resilience, `PlanStep` adapter contracts, `ConversationMemory` compatibility, GraphQL parity).
- **[`.agents/rules/validation-and-gates.md`](.agents/rules/validation-and-gates.md)**: Authoritative offline qualification commands, socket-denial guard checks, and test suite baselines (148 backend + 111 frontend tests).
- **[`.agents/rules/70-memory-workflow.md`](.agents/rules/70-memory-workflow.md)**: `mem0ry4ai` authority order, `memory-scout` pre-task bookend, working notes, and `memory-steward` post-task closeout bookend.
- **[`.agents/skills/`](.agents/skills/)**: 26 reusable workspace skills tracked in [`.agents/.antigravity-install-manifest.json`](.agents/.antigravity-install-manifest.json).

These Antigravity-specific resources operationalize and strengthen [`AGENTS.md`](AGENTS.md), and must never contradict or replace it.

---

## 5. Memory Integration (`mem0ry4ai`)

Durable knowledge is stored in the `mem0ry4ai` MCP server under scope `project:advanced_prompting_system`:

- **Pre-task bookend**: `memory-scout` executes `memory_resume`, `memory_search`, and `session_search` to recover prior decisions, PR #34/#35/#36 context, and regression gotchas.
- **Post-task bookend**: `memory-steward` executes `memory_search` (for deduplication) followed by `memory_add`, `memory_note`, or `memory_promote` to persist verified architectural decisions, bug root causes, and test baselines.
- **Local Workspace**: [`memory-bank/`](memory-bank/) (`activeContext.md`, `systemPatterns.md`, `techContext.md`, `progress.md`) is maintained alongside `mem0ry4ai` for human and IDE readability.
- Memory agents are bookends, not general-purpose workers; do not summon them repeatedly for every micro-step.

---

## 6. Recommended Slash Commands

- `/plan`: Step-by-step architectural planning and user alignment before modifying production files.
- `/grill-me`: Interactive requirements elicitation, design alignment, and edge-case probing before implementation.
- `/boost`: Deep multi-perspective reasoning and architectural verification for complex reasoning/consensus refactors.
- `/goal`: Long-running comprehensive execution tasks (e.g., multi-phase issue resolution and test sweeps).
- `/schedule`: Managing background tasks and recurring checks.
- `/learn`: Persisting new team conventions, prompt patterns, and user preferences.
