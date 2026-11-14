# Active Context (`advanced_prompting_system`)

## Current Focus

- Issue #35 remains open. Its original acceptance backlog is tracked by 18 follow-ups, #44–#61; green maintained suites or a narrow PR review do not establish whole-issue acceptance.
- The current slice is **#48: typed collaboration results**, implemented on `fix/issue-48-typed-collaboration-results` under the approved [plan](../docs/superpowers/plans/2026-11-14-issue48-typed-collaboration.md).
- [PR #62](https://github.com/dhar174/advanced_prompting_system/pull/62) is open and ready for review (not draft), not merged. #48 is not declared closed; human review and the PR's integration decision remain outstanding.
- Current review-fix source: `8d0a9065b83c68ec81ced8ec6218553d451552ab`. It addresses all 11 Antigravity/Codex inline concerns on PR #62. Current-head CI, replies and thread resolution are not yet verified; do not call them passed or resolved before the coordinator checks them.

## Verified #48 Contracts

- Internal `ConversationOutput` distinguishes text, an owned artifact and failure; `ConversationResult` retains actual copied history/questions and nullable measured quality. Path-like ordinary text is never opened.
- Typed producers establish output kind; legacy `run_conversation` three-tuple and finalizer projections remain compatible, including question `reply`/`asking_assistant` attribution.
- Ownership is enforced through conversion and direct adapter artifact-scope lifetime. Existing nested owner/provider ledger/deadline are reused; a standalone provider-policy boundary is still #51.
- Nested output uses the actual `judge_step`; unknown quality stays null and measured 1.0 remains valid. Accepted plan/step-linked metadata is copied and selected; rejected metadata/artifacts are excluded. Registered filename collisions are versioned to retain accepted bytes.
- Empty tool payloads, including empty PDF input, fail explicitly. Later typed failure preserves prior accepted metadata and emits a safe error. GraphQL adds nullable `planStepNumber` without changing input shapes.

- Solo exploration takeover now replaces the whole selected candidate atomically, including equal-length state; discarded debate history/questions/artifacts do not leak into the winner. Unevaluated candidate quality does not trigger speculative exploration.
- Final evaluation preserves a valid zero as well as 1.0. Invalid/unavailable evaluation remains null, preserves the answer and reports explicit non-success; policy stops are attempted once without a second final-judge dispatch.
- Fatal/partial collaboration failures retain ordered plan-linked history/questions and safe errors/events, without inventing finalization. Owner-positive controls now cover actual nested conversion/service paths, typed path-like text and foreign artifact read/list denial.

## Verified Evidence and Limits

- At current review-fix source: full guarded Python **3.11.18: 389 passed in 61.25s**. Python **3.12.14: 389 passed in 46.09s**. Full suites include eight browser-suite checks: six real Chromium scenarios, one non-browser ownership/CORS check and one manifest verifier.
- Frontend: **147 passed across 17 files**; network guard, lint and production build pass. Existing dependencies, NLP assets and browser binaries were reused; no fresh dependency resolution is claimed.
- Historical initial #48 source `4c328b215e4bb38f7412d27945f59d75995a2514` passed 342 tests on each Python version and [push CI 43567823628](https://github.com/dhar174/advanced_prompting_system/actions/runs/43567823628). Those results do not qualify current-head CI; inspect the actual revision independently.
- The initial four whole-branch blockers were reproduced RED→GREEN and fixed. A fresh independent review of the latest PR #62 repairs found no remaining blockers in the inspected slice; this is not broad #35 acceptance. See [#48 qualification](../docs/issue48-qualification.md) and the retained review regressions for evidence and scope.
- Two local Docker builds failed at npm setup, including a proxy retry; no local image/startup pass is claimed. No paid calls, deployment, merge or original-scope human acceptance occurred.

## Next Priorities

1. Finish review/integration of #48 without treating it as completion of #35.
2. Address #49 (revision/state isolation) and #51 (standalone provider policy), then #50 (whole-task finalized deliverables). Downloads/UI, follow-up/feedback, progress/configuration and broader evidence remain separate follow-ups.
3. Keep #36 Tree of Thoughts gated. Backend cancellation and durable resume remain unsupported; ToT is not the next priority while #35 blockers remain.
