# Antigravity Agent Team

This repository defines workspace-scoped Google Antigravity 2.0+ custom agents under [`.agents/agents/<name>/agent.md`](../agents/).

The team follows the repository's authoritative operating contract rather than creating a parallel source of truth. Every agent starts from [`AGENTS.md`](../../AGENTS.md); offline qualification boundaries and PR #34/#35/#36 milestones remain documented in [`docs/pr34-offline-qualification.md`](../../docs/pr34-offline-qualification.md) and [`docs/offline-test-inventory.md`](../../docs/offline-test-inventory.md), and cross-session context is maintained in `mem0ry4ai` (`project:advanced_prompting_system`) and [`memory-bank/`](../../memory-bank/).

## Team Shape

- **`aps-repo-coordinator`**: Primary selectable agent, single production writer, and subagent coordinator. Plans work, integrates changes, delegates investigation/review across the 5-stage lifecycle, and owns final synthesis.
- **`memory-scout`**: Read-only pre-task durable-context retrieval scout to be called immediately after coordinator intake (Stage 1 bookend).
- **`architecture-contract-scout`**: Read-only pre-implementation scout for cross-module seams (`PlanStep` caller mapping, collaborative adapters, 3-tuple return shapes, `LazyClient` import purity, and production vs. historical file boundaries).
- **`prompt-reasoning-architect`**: Read-only specialist for `advanced_prompting.py` (`AdvancedPromptEngineer`, CoT/Least-to-Most/Progressive-Hint/Self-Consistency/ToT, XML tag parsing, `consolidate_steps`, `merge_interactions`, reward judging, and collaborative adapters).
- **`consensus-conversation-specialist`**: Read-only specialist for `conversation_manager.py`, `assistant_personalities.py`, `output_generator.py`, and `function_definitions.py` (`BinaryVote`, `ConfidenceVote`, `ErrorResult`, `MediatorReviseOrDecide`, `ConversationMemory`, message normalization, and protected file writes).
- **`complexity-nlp-engineer`**: Read-only specialist for `complexity_measures.py` (`Plan`, `PlanStep`, `Subtask`, lazy `get_nlp_spacy()`, optional `nltk`/transformer fallbacks, `ml_complexity_model.pkl`, `cosine_similarity_custom`, and offline `get_embedding`).
- **`graphql-frontend-specialist`**: Read-only specialist for `app.py`, `schema.py`, `feedback_manager.py`, and `collaborative-assistant-frontend/` (Strawberry GraphQL schema contracts, React 19/TypeScript/Apollo UI components, hooks, ESLint cleanup, and 111 Vitest tests).
- **`offline-qa-gatekeeper`**: Read-only independent test designer, offline isolation diagnostician, and adversarial final reviewer governing `provider_client.py`, `conftest.py`, `tools/run-offline.*`, `tools/run-frontend-offline.*`, `tools/offline/*`, `.github/workflows/test.yml`, and all backend/frontend test suites.
- **`memory-steward`**: Post-verification durable-memory curation steward (Stage 5 bookend) that curates verified knowledge through `mem0ry4ai` MCP tools and refreshes `memory-bank/` after all other subagents and verification gates settle.

## One-Writer Rule

The custom team enforces a strict **one-production-writer policy**:

Specialist subagents are configured without production file-write tools (except `memory-steward`, which updates only `memory-bank/` at closeout). They inspect code, run sandboxed offline verification commands (`./tools/run-offline.ps1` and `./tools/run-frontend-offline.ps1`), and return findings and surgical diffs to `aps-repo-coordinator` rather than editing production modules directly.

This keeps parallelism focused on independent domain reasoning instead of concurrent edits to shared state.

## Suggested Workflow

For a non-trivial feature, bugfix, or refactor:

1. Invoke `memory-scout` before substantive planning. Treat the returned Memory Brief as context only; current repository evidence and contracts remain authoritative.
2. Start with `aps-repo-coordinator` as the primary agent.
3. Have the coordinator dispatch `architecture-contract-scout` before changing cross-module signatures or inventing new abstractions.
4. Dispatch the domain specialist(s) that own the affected subsystem (`prompt-reasoning-architect`, `consensus-conversation-specialist`, `complexity-nlp-engineer`, or `graphql-frontend-specialist`).
5. Ask `offline-qa-gatekeeper` for independent red-green contract and adversarial cases before or alongside implementation.
6. Keep all production edits with `aps-repo-coordinator`.
7. Run the guarded offline verification suites (`./tools/run-offline.ps1` and `./tools/run-frontend-offline.ps1`) and ask `offline-qa-gatekeeper` for an independent final review.
8. After all subagents and verification gates have settled the final state, invoke `memory-steward`. Give `memory-steward`:
   - original task;
   - implementation result;
   - reviewer/gatekeeper findings and resolutions;
   - offline test/validator results;
   - documentation changes;
   - known unfinished work.
9. Include the steward's memory actions in the final closeout when useful. Trivial tasks may skip the memory checkpoints when prior project context cannot materially affect the result.

Example delegation flow:

```text
User / Task
  -> aps-repo-coordinator
      -> memory-scout (Stage 1: Pre-Flight Context)
      -> architecture-contract-scout (Stage 2: Seam & Contract Audit)
      -> prompt-reasoning-architect / consensus-conversation-specialist /
         complexity-nlp-engineer / graphql-frontend-specialist (Stage 3: Domain Analysis)
      -> implementation by aps-repo-coordinator (Single Production Writer)
      -> offline-qa-gatekeeper (Stage 4: Offline Gate & Adversarial Review)
      -> memory-steward (Stage 5: Durable Memory Closeout)
```

## Antigravity Discovery

Antigravity 2.0+ discovers workspace agents from `.agents/agents/<name>/agent.md`. This repository uses the directory form so each agent can later gain role-specific supporting references without changing its public identifier.

`aps-repo-coordinator` is configured as a selectable main agent (`mainAgent: true`). The other 8 definitions are specialist subagents (`subagent: true`).

## Skills and MCP

Workspace skills reside in [`.agents/skills/<name>/SKILL.md`](../skills/) and are bound to relevant subagents via their YAML frontmatter `skills:` list. When `mem0ry4ai` or `aas-core` MCP tools are available, agents use them according to their prompts and [`.agents/rules/70-memory-workflow.md`](../rules/70-memory-workflow.md).

## Safety Notes

These agent files do not replace repository permissions or Antigravity's sandbox. The root [`AGENTS.md`](../../AGENTS.md) contract governs lazy SDK imports, offline socket-denial guards, artifact hygiene, secrets, merging, force-pushing, and destructive cleanup.
Remember that `memory-scout` and `memory-steward` are bookends, not general-purpose workers—call them once before and once after non-trivial tasks rather than repeatedly during micro-steps.
