# Issue 51 implementation plan

Approved in the conversation. Base: merged main `a85a913f673fe65afb22485b1e919f7fcaedebe1`.

## Required contract

- Keep `AdvancedPromptEngineer.main(task) -> Interaction` supported. Establish one finite validated provider scope before retrieval/rephrasing/planning when standalone; preserve artifact ownership and lazy imports.
- Move provider-only configuration, environment resolution and model validation into provider_client; keep execution_modes compatibility imports. Canonical defaults: 40 calls / 120 seconds. Add optional constructor max_calls/timeout_seconds; only tighten environment ceilings. Zero calls dispatch nothing; timeout must be positive and finite. Reject invalid/excessive values.
- Distinguish omitted and explicit model. Validate explicit selection; do not infer omission by the default model string.
- User chose caller-policy-wins: reuse a valid finite parent scope unchanged, including ledger identity, consumed calls, deadline, model and cancellation. Constructor limits govern standalone roots only. Reject incomplete/unbounded parent policy and conflicting explicit model before dispatch.
- All reachable planning, embedding, generation, revision, selection, judging, complexity and nested calls share the ledger, charged once. Embeddings share counts without chat-model allowlisting. Preserve zero SDK retries and bounded explicit retries/waits.
- Normalize early policy termination to safe non-success Interaction with unknown quality; preserve accepted partials on later stops. Check deadline/cancellation before publishing a return, without rejecting exact-final-call success. Clean up owned contexts and preserve parent contexts.
- Cooperative deadlines enforce SDK remaining timeouts and reject late returns; no hard Python preemption claim.

## Delivery and qualification

- [x] Configuration contract with RED→GREEN regressions; commit.
- [x] Standalone/main composition, early/late termination and cleanup; RED→GREEN; commit.
- [x] Real provider-only direct, nested, revision, exact-cap, timeout/retry, cancellation, concurrent-run and HTTP compatibility qualification; literal README example.
- [x] Guarded full Python 3.11/3.12, frontend guard/tests/lint/build, Chromium, independent whole-branch review and CRLF diff checks.
- [ ] Clean production-image CI and exact-head browser artifact verification (PR description records final results).
- [ ] Focused PR, exact-source qualification/evidence and documented limitations.

## Boundaries

Preserve #48/#49 result, ownership, revision, quality and partial diagnostics. No paid calls, deployment, merges, #35 closure, new streaming, shared-engine concurrency, hard kill or durable resume. #50 deliverables, #56/#57 broader configuration/setup and #59/#60/#61 acceptance remain separate. Use existing outbound guards and deterministic fixtures; root is sole production writer, specialists are read-only.
