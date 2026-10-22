---
trigger: model_decision
description: "mem0ry4ai durable cross-session memory workflow, authority order, and memory-scout / memory-steward bookend checkpoints."
---

# Memory Workflow (`mem0ry4ai` & `memory-bank/`)

`mem0ry4ai` (`project:advanced_prompting_system`) paired with local [`memory-bank/`](../../memory-bank/) files forms the repository's durable cross-session context layer. Memory is useful context, never a higher authority than current repository contracts or evidence.

---

## 1. Authority Order

When information conflicts, follow this strict precedence order (highest to lowest):

1. Current explicit user instruction;
2. [`AGENTS.md`](../../AGENTS.md) and [`GEMINI.md`](../../GEMINI.md);
3. Applicable [`.agents/rules/`](.) (and mirrored [`.agent/rules/`](../../.agent/rules/));
4. [`docs/pr34-offline-qualification.md`](../../docs/pr34-offline-qualification.md) and [`docs/offline-test-inventory.md`](../../docs/offline-test-inventory.md);
5. Active GitHub issue / PR task requirements;
6. Current production source code and offline test suites;
7. Local [`memory-bank/`](../../memory-bank/) documentation;
8. `mem0ry4ai` durable memories (`project:advanced_prompting_system`);
9. Tentative `mem0ry4ai` working-memory notes (`status="working"`).

Never implement or revert behavior merely because a recalled memory says it was once true. Always verify material claims against current files and tests.

---

## 2. Required Agent-Loop Bookend Checkpoints

For non-trivial repository work, bookend the specialist workflow with `memory-scout` and `memory-steward`:

```text
aps-repo-coordinator
  -> memory-scout (Stage 1: Pre-Flight Context Recovery)
  -> architecture-contract-scout (Stage 2: Seam & Contract Audit)
  -> Domain Specialists (Stage 3: Analysis & Proposed Diffs)
  -> Implementation by aps-repo-coordinator (Single Production Writer)
  -> offline-qa-gatekeeper (Stage 4: Offline Verification & Review)
  -> memory-steward (Stage 5: Durable Knowledge Closeout)
  -> Coordinator Closeout
```

### Start Checkpoint: `memory-scout` (Stage 1)
Before substantive planning or implementation, `aps-repo-coordinator` invokes `memory-scout` with:
- the current task / issue / PR number;
- repository scope `project:advanced_prompting_system`;
- important named components or files already known;
- a request for relevant status, todos, decisions, gotchas, facts, commands, and conflicts.

Pass the resulting **Memory Brief** to downstream specialists only where relevant. Trivial tasks may skip `memory-scout` when prior project context cannot materially affect the result.

### During-Work Checkpoint: Working Notes
Agents may call `memory_note(scope="project:advanced_prompting_system", ...)` for uncertain in-progress findings worth surviving context compaction, but only when the note has genuine future value during active work. Do not turn ordinary scratch reasoning into working memory.

### Closeout Checkpoint: `memory-steward` (Stage 5)
After implementation, offline test suites (`./tools/run-offline.ps1` and `./tools/run-frontend-offline.ps1`), and gatekeeper review have settled the final repository truth, invoke `memory-steward` with:
- the original task;
- the verified implementation result;
- reviewer/gatekeeper findings and resolutions;
- exact offline test/validator results;
- documentation changes;
- known unfinished work.

`memory-steward` must search `mem0ry4ai` before every durable write, write agent-neutrally in second person, avoid duplicate entries, and refresh `memory-bank/activeContext.md` and `memory-bank/progress.md`.
