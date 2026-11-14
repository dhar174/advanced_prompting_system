# Active Context (`advanced_prompting_system`)

## Current Focus

- Issue #35 remains open. Follow-ups #44–#61 track its acceptance backlog; a green narrow slice does not establish whole-issue acceptance.
- #48 / [PR #62](https://github.com/dhar174/advanced_prompting_system/pull/62) and #49 / [PR #63](https://github.com/dhar174/advanced_prompting_system/pull/63) are merged and their issues are closed. Preserve their typed owned outputs, selected metadata, atomic revisions, measured scores and isolated candidate state.
- Current slice: **#51 finite standalone provider policy**, branch `fix/issue-51-standalone-provider-policy`, based on merged main `a85a913f673fe65afb22485b1e919f7fcaedebe1`, under the approved [plan](../docs/superpowers/plans/2026-11-14-issue51-standalone-policy.md).
- Current qualified implementation source: `8f4300ac6c83d63f4ec50dbfeb90e79dc278cb83`. At this memory checkpoint, PR publication, exact-head CI, clean runtime-image qualification and downloaded evidence verification remain **pending** with the coordinator. Consult the final PR description for later outcomes.

## Verified #51 Contracts

- Use `AdvancedPromptEngineer.main(task) -> Interaction` as the supported direct boundary. It establishes one finite provider scope before retrieval/planning, composes with artifact ownership and restores owned scopes on every exit. Lower-level helpers do not establish that guarantee by themselves.
- Provider-only canonical configuration belongs to `provider_client.py`; `execution_modes.py` retains compatibility exports. Standalone defaults are 40 calls / 120 seconds. Constructor limits may tighten canonical environment ceilings; zero calls dispatch nothing. Invalid/nonfinite/excessive values fail before dispatch. Omitted models inherit the effective default; explicit selection must pass the allowlist.
- The user chose **caller policy wins**: reuse a valid finite parent context unchanged, including identity, consumed ledger, deadline, model policy and cancellation. Constructor call/time limits apply only to standalone roots. Reject incomplete/unbounded policies, malformed or rolled-back counts and conflicting explicit models before dispatch.
- Planning, embeddings, generation, revisions, selection, judges and nested collaboration share one ledger/deadline. Keep embedding identifiers distinct from chat-model allowlisting. SDK retries remain zero; explicit attempts and waits are bounded. Ambient single-call reservations must not exempt a whole run: bind reservation markers to their context and suspend/restore them at main entry.
- Policy stops before normal Interaction construction return fresh safe non-success results with unknown quality; later stops preserve accepted partial work and diagnostics. Check deadline/cancellation before returning, but allow success on the final permitted call. Deadlines are cooperative, not hard Python preemption. Use separate engineer instances for simultaneous runs.
- The README's literal example is exercised under the network guard with a deterministic SDK fixture. Legacy tuples, mutation shapes, application-token access and lazy imports remain supported.

## Verified Evidence and Limits

- Full guarded source suites: **Python 3.12.14: 504 passed in 55.51s; Python 3.11.18: 504 passed in 74.95s**. Each includes ten browser-suite checks: eight actual Chromium scenarios, ownership/CORS and manifest verification.
- Latest focused policy checks: **76 passed in 7.69s**. Two independent review accounting findings were reproduced RED→GREEN and repaired; final independent recheck found no remaining actionable findings. An additional huge-timeout validation regression also passed RED→GREEN.
- Frontend network guard, **147 tests across 17 files**, lint and production build pass. The existing Browserslist warning remains. CRLF-aware diff checks are clean.
- Local environments/assets were reused; do not infer clean dependency resolution or container success from local tests. Current-head CI and retained evidence are separate gates. No paid calls, deployment, merge or #35 closure occurred in this slice. mem0ry4ai was unavailable; no durable-tool writes were made.

## Next Priorities

1. Finish #51 PR publication, exact-head CI/runtime-image qualification and retained evidence verification.
2. Address #50 whole-task finalized deliverables next. Broader configuration/setup remains #56/#57; acceptance correlation/disposition remains #59/#60/#61. Other API/UI/operations follow-ups remain separate.
3. Keep #36 ToT gated. New streaming, shared-engine concurrency, backend cancellation API, hard kill and durable resume remain outside this slice.
