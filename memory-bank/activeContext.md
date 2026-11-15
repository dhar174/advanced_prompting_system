# Active Context (`advanced_prompting_system`)

## Current Focus

- Issue #35 remains open. Follow-ups #44–#61 track its acceptance backlog; a green narrow slice does not establish whole-issue acceptance.
- #48 / [PR #62](https://github.com/dhar174/advanced_prompting_system/pull/62) and #49 / [PR #63](https://github.com/dhar174/advanced_prompting_system/pull/63) are merged and their issues are closed. Preserve their typed owned outputs, selected metadata, atomic revisions, measured scores and isolated candidate state.
- Current slice: **#51 finite standalone provider policy**, branch `fix/issue-51-standalone-provider-policy`, based on merged main `a85a913f673fe65afb22485b1e919f7fcaedebe1`, under the approved [plan](../docs/superpowers/plans/2026-11-14-issue51-standalone-policy.md).
- [PR #64](https://github.com/dhar174/advanced_prompting_system/pull/64) is open and unmerged. Current review-fix source: `684053885775b9b4e9e4b6888271b87883c0abcc`. Its local qualification is complete; push, current-head CI/runtime image, downloaded evidence verification and individual review replies/resolutions remain **pending** with the coordinator. Original head `247bc73` passed CI/evidence historically; those results do not qualify the review repairs.

## Verified #51 Contracts

- Use `AdvancedPromptEngineer.main(task) -> Interaction` as the supported direct boundary. It establishes one finite provider scope before retrieval/planning, composes with artifact ownership and restores owned scopes on every exit. Lower-level helpers do not establish that guarantee by themselves.
- Provider-only canonical configuration belongs to `provider_client.py`; `execution_modes.py` retains compatibility exports. Standalone defaults are 40 calls / 120 seconds. Constructor limits may tighten canonical environment ceilings; zero calls dispatch nothing. Invalid/nonfinite/excessive values fail before dispatch. Omitted models inherit the effective default; explicit selection must pass the allowlist.
- The user chose **caller policy wins**: reuse a valid finite parent context unchanged, including identity, consumed ledger, deadline, model policy and cancellation. Constructor call/time limits apply only to standalone roots. Reject incomplete/unbounded policies, malformed or rolled-back counts and conflicting explicit models before dispatch.
- Planning, embeddings, generation, revisions, selection, judges and nested collaboration share one ledger/deadline. Keep embedding identifiers distinct from chat-model allowlisting. SDK retries remain zero; explicit attempts and waits are bounded. Ambient single-call reservations must not exempt a whole run: bind reservation markers to their context and suspend/restore them at main entry.
- Policy stops before normal Interaction construction return fresh safe non-success results with unknown quality; later stops preserve accepted partial work and diagnostics. Check deadline/cancellation before returning, but allow success on the final permitted call. Deadlines are cooperative, not hard Python preemption. Use separate engineer instances for simultaneous runs.
- The README's literal example is exercised under the network guard with a deterministic SDK fixture. Legacy tuples, mutation shapes, application-token access and lazy imports remain supported.
- PR #64 review repairs use a reentrant context lock so synchronous cancellation callbacks can read execution stats safely; all termination-state writes use that lock and SDK work stays outside it. Scope allowances accept only `None` or exact nonnegative integers, preserving zero and trusted parent allowances such as 300. Filename-generation policy exceptions propagate immediately; ordinary failures retain their fallback.
- Install spaCy `en_core_web_sm` during setup before guarded multi-step qualification. The production missing-model fallback remains supported; it does not establish the two-step test contract. Never download assets during guarded execution or skip planning to hide this prerequisite.

## Verified Evidence and Limits

- Full guarded review-fix suites: **Python 3.12.14: 525 passed in 57.59s; Python 3.11.18: 525 passed in 70.47s**. Each includes ten browser-suite checks: eight actual Chromium scenarios, ownership/CORS and manifest verification.
- Latest focused policy checks: **97 passed in 7.84s**, including 21 new review regressions. Nine policy and three filename failures were witnessed before repair. Fresh independent review passed all 21 review cases and found no actionable findings. Earlier accounting/number-validation repairs remain enabled.
- Frontend's prior **147 tests across 17 files**, guard, lint and build remain historical evidence. No frontend/API/configuration files changed in this repair; local frontend checks were not repeated. Current-head CI will qualify them again. The existing Browserslist warning remains. CRLF-aware diff checks are clean.
- Local environments/assets were reused; do not infer clean dependency resolution or container success from local tests. Current-head CI and retained evidence are separate gates. No paid calls, deployment, merge or #35 closure occurred in this slice. mem0ry4ai was unavailable; no durable-tool writes were made.

## Next Priorities

1. Finish PR #64 review-fix publication, current-head CI/runtime-image qualification, retained evidence verification and per-comment replies/resolutions.
2. Address #50 whole-task finalized deliverables next. Broader configuration/setup remains #56/#57; acceptance correlation/disposition remains #59/#60/#61. Other API/UI/operations follow-ups remain separate.
3. Keep #36 ToT gated. New streaming, shared-engine concurrency, backend cancellation API, hard kill and durable resume remain outside this slice.
