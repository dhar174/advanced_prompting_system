# Issue 48: Secure nested collaboration results

Approved plan: secure the typed producer/ownership boundary; preserve measured quality and selected per-step metadata; qualify the real HTTP/provider path. Base: 619bd150ebbbed94e763a258579228a05b735273.

## Task 1 — Typed producer and ownership
- Add failing regressions for owned text/PDF, unregistered/missing/foreign files and path-like text.
- Add ConversationOutput/ConversationResult and typed finalization/run entries; keep legacy projections.
- Resolve registered artifacts under the active owner; reuse scopes; propagate explicit safe failures.
- Verify focused checks and commit.

## Task 2 — Scores and selected metadata
- Evaluate actual nested output with judge_step under the shared ledger; never fabricate quality.
- Copy candidate-local step-linked histories/questions, committing only selected records.
- Add optional planStepNumber GraphQL output fields and preserve unknown aggregate quality.
- Verify focused regressions and commit.

## Task 3 — Production qualification
- Add distinguishable two-step provider fixtures and real HTTP integration, partial-failure, selection and finite-ledger tests.
- Run guarded Python 3.11/3.12, frontend guard/tests/lint/build, affected Chromium/runtime-image checks; inspect all results.
- Independent contract/QA review, qualification documentation, diff checks, focused PR.

## Boundaries
Only #48. Preserve legacy tuples/API shapes, token access, lazy imports and guards. #49/#50/#51/#52 remain separate. No paid calls, deployment, merge or automatic #35 closure.
