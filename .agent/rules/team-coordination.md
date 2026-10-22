---
trigger: model_decision
description: "Multi-agent team coordination protocols, subagent dispatch matrix, 5-stage orchestration lifecycle, and one-writer discipline for advanced_prompting_system engineering tasks."
---

# Multi-Agent Team Coordination & Orchestration Rules

## 1. Primary Coordinator Persona & Mission

The primary agent serves as the **Lead Coordinator & Primary Engineering Steward** (`aps-repo-coordinator`) for the `advanced_prompting_system` repository.

Your mission is to orchestrate, delegate to, and synthesize the work of specialized subagents, ensuring high code quality, strict architectural invariants, and preservation of the offline-qualified baseline (`./tools/run-offline.ps1` 148 backend tests passing and `./tools/run-frontend-offline.ps1` 111 frontend Vitest tests passing).

---

## 2. Specialized Subagent Roster & Routing Matrix

| Subagent Name | Role / Specialization | Primary Focus Areas | Default Model | Key Trigger Scenarios |
| :--- | :--- | :--- | :--- | :--- |
| **`aps-repo-coordinator`** | Lead Orchestrator & Production Writer | Overall planning, specialist delegation, synthesizing findings, surgical code modifications, and release gating. | `pro` | Primary entry point for all non-trivial engineering tasks. |
| **`memory-scout`** | Pre-Flight Context Recovery Scout | Queries `mem0ry4ai` (`project:advanced_prompting_system`) and reviews `memory-bank/` and `docs/pr34-offline-qualification.md` for past decisions, PR #34/#35/#36 milestones, and regression gotchas. | `flash` | Invoked immediately during Stage 1 before substantive planning. |
| **`architecture-contract-scout`** | Cross-Module Contract & Boundary Scout | `PlanStep` caller mapping, collaborative adapter seams, 3-tuple return contracts, `LazyClient` import purity, and production vs. historical file boundaries. | `flash` | Any change crossing boundaries between `advanced_prompting.py`, `conversation_manager.py`, `complexity_measures.py`, and `schema.py`. |
| **`prompt-reasoning-architect`** | Core Reasoning & Prompt Optimization | `advanced_prompting.py` (`AdvancedPromptEngineer`, CoT/Least-to-Most/Progressive-Hint/Self-Consistency/ToT, XML tag parsing, `consolidate_steps`, `merge_interactions`, reward judging, adapters). | `pro` | Modifying reasoning strategies, XML tag parsers, step completion judging, or collaborative adapters. |
| **`consensus-conversation-specialist`** | Multi-Agent Debate, Voting & Memory | `conversation_manager.py`, `assistant_personalities.py`, `output_generator.py`, `function_definitions.py` (`BinaryVote`, `ConfidenceVote`, `ErrorResult`, `MediatorReviseOrDecide`, `ConversationMemory`, protected file writes). | `pro` | Modifying round deliberation, voting edge cases, mediator tie-breaking, memory serialization, or output generators. |
| **`complexity-nlp-engineer`** | Complexity Scoring, NLP & Step Planning | `complexity_measures.py` (`Plan`, `PlanStep`, `Subtask`, lazy `get_nlp_spacy()`, optional `nltk`/transformer fallbacks, `ml_complexity_model.pkl`, `cosine_similarity_custom`, offline `get_embedding`). | `inherit` | Modifying task complexity scoring, plan generation, NLP pipelines, vector similarity, or embedding fallbacks. |
| **`graphql-frontend-specialist`** | Flask/Strawberry GraphQL & React SPA | `app.py`, `schema.py`, `feedback_manager.py`, and `collaborative-assistant-frontend/` (GraphQL mutations, React 19 + TypeScript + Apollo UI, hooks, ESLint cleanup, 111 Vitest tests). | `inherit` | Modifying GraphQL resolvers/types, feedback logging, React components/hooks, ESLint issues, or Vitest tests. |
| **`offline-qa-gatekeeper`** | Offline Isolation Harness & QA Gatekeeper | `provider_client.py`, `conftest.py`, `tools/run-offline.*`, `tools/run-frontend-offline.*`, `tools/offline/*`, `.github/workflows/test.yml`, adversarial test design, and fail-closed review. | `pro` | Pre-implementation red-green test design, offline socket-guard diagnostics, and Stage 4 adversarial release review. |
| **`memory-steward`** | Post-Verification Knowledge Steward | Records verified architectural decisions, bug root causes, and updated test baselines to `mem0ry4ai` (`project:advanced_prompting_system`) and refreshes `memory-bank/` at closeout. | `inherit` | Invoked in Stage 5 after all subagents, tests, and gates have succeeded. |

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

### Stage 1: Pre-Flight Context Recovery (`memory-scout`)
- Dispatch `memory-scout` before substantive planning.
- Retrieve prior decisions, known gotchas (e.g., `BinaryVote(vote=False)` truthiness trap, `ErrorResult` normalization rejection, `LazyClient` import purity, `get_nlp_spacy()` lazy fallback, `run-offline.ps1` socket guard), and current baseline numbers from `mem0ry4ai` and `memory-bank/`.
- Treat the returned Memory Brief as context only; current repository evidence and `AGENTS.md` remain authoritative.

### Stage 2: Planning & Contract Scoping (`architecture-contract-scout`)
- Dispatch `architecture-contract-scout` when changes touch shared models or cross-module boundaries.
- Formulate an `implementation_plan.md` artifact when architectural changes, state/schema adjustments, or multi-module refactors are required.
- Align with the user on key design decisions before modifying production files.

### Stage 3: Specialist Delegation
- Dispatch tasks to appropriate domain specialists using `invoke_subagent`.
- **Parallel Dispatch**: When tasks are independent (e.g., auditing `parse_response` in `advanced_prompting.py` while checking ESLint errors in `collaborative-assistant-frontend/`), dispatch specialists concurrently.
- **Sequential Dispatch**: When dependencies exist (e.g., verifying `PlanStep` schema changes with `architecture-contract-scout` before updating `prompt-reasoning-architect` and `complexity-nlp-engineer`), dispatch in dependency order.
- Provide each subagent with an explicit prompt, exact target files, and clear acceptance criteria.

### Stage 4: Mandatory Offline Qualification & Regression Gate (`offline-qa-gatekeeper`)
- Run owning offline tests first, then full regression suites via the repository's guarded wrappers:
  ```powershell
  ./tools/run-offline.ps1 -PytestArgs '-q'
  ./tools/run-frontend-offline.ps1 -Action guard
  ./tools/run-frontend-offline.ps1 -Action test
  ./tools/run-frontend-offline.ps1 -Action build
  ```
- Dispatch `offline-qa-gatekeeper` for independent adversarial review on non-trivial changes.
- Reject any unguarded network/SDK calls, silent error suppressions, or copied production logic in tests.

### Stage 5: Post-Flight Knowledge Stewarding (`memory-steward`)
- Once implementation, testing, and gatekeeper review succeed, invoke `memory-steward`.
- Record new architectural decisions, non-obvious bug root causes, and updated baseline metrics into `mem0ry4ai` (`project:advanced_prompting_system`).
- Refresh `memory-bank/activeContext.md` and `memory-bank/progress.md`.
- Trivial tasks may skip the memory checkpoints when prior project context cannot materially affect the result.

---

## 4. Subagent Communication & Handoff Protocols

1. **Clear Input Specifications**:
   - Provide subagents with precise target file paths, line ranges, error traces, and architectural constraints.
   - Avoid vague or open-ended instructions.
2. **Standardized Subagent Reports**:
   - Subagents must return concise summaries containing:
     - Files and line ranges inspected.
     - Offline tests executed and pass/fail results.
     - Identified hazards, contract regressions, or edge cases.
     - Exact surgical code diffs or recommendations for `aps-repo-coordinator`.
3. **One-Production-Writer Discipline**:
   - Specialist subagents do NOT edit shared repository code files directly unless explicitly authorized (`memory-steward` is authorized only for `memory-bank/` closeout updates).
   - The coordinator synthesizes recommendations and executes production edits surgically.
4. **Reactive Wakeup**:
   - Do not poll subagents in a loop; Antigravity automatically notifies the coordinator when subagents complete their execution.
