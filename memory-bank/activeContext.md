# Active Context (`advanced_prompting_system`)

## Current Focus

- Issue #35 remains open. Its original acceptance backlog is tracked by 18 follow-ups, #44–#61; green maintained suites or a narrow PR review do not establish whole-issue acceptance.
- The current slice is **#48: typed collaboration results**, implemented on `fix/issue-48-typed-collaboration-results` under the approved [plan](../docs/superpowers/plans/2026-11-14-issue48-typed-collaboration.md).
- [PR #62](https://github.com/dhar174/advanced_prompting_system/pull/62) is draft/open, not merged. #48 is not declared closed; human review and the PR's integration decision remain outstanding.
- Qualified source: `4c328b215e4bb38f7412d27945f59d75995a2514`, based on `619bd150ebbbed94e763a258579228a05b735273`. Subsequent schema-indentation/documentation changes have separate exact-head checks; do not describe their final CI as passed before checking it.

## Verified #48 Contracts

- Internal `ConversationOutput` distinguishes text, an owned artifact and failure; `ConversationResult` retains actual copied history/questions and nullable measured quality. Path-like ordinary text is never opened.
- Typed producers establish output kind; legacy `run_conversation` three-tuple and finalizer projections remain compatible, including question `reply`/`asking_assistant` attribution.
- Ownership is enforced through conversion and direct adapter artifact-scope lifetime. Existing nested owner/provider ledger/deadline are reused; a standalone provider-policy boundary is still #51.
- Nested output uses the actual `judge_step`; unknown quality stays null and measured 1.0 remains valid. Accepted plan/step-linked metadata is copied and selected; rejected metadata/artifacts are excluded. Registered filename collisions are versioned to retain accepted bytes.
- Empty tool payloads, including empty PDF input, fail explicitly. Later typed failure preserves prior accepted metadata and emits a safe error. GraphQL adds nullable `planStepNumber` without changing input shapes.

## Verified Evidence and Limits

- Full guarded suites at the qualified source: **342 passed on Python 3.11.18 and 3.12.14**, including eight browser-suite checks: six real Chromium scenarios, one non-browser ownership/CORS check and one manifest verifier.
- Frontend: **147 passed across 17 files**; network guard, lint and production build pass. Existing dependencies, NLP assets and browser binaries were reused; no fresh dependency resolution is claimed.
- [Source push CI 43567823628](https://github.com/dhar174/advanced_prompting_system/actions/runs/43567823628) passed all five jobs, including production image and browser. Separate PR/final-documentation-head checks must be inspected independently.
- Four blocking findings from a fresh whole-branch reviewer were reproduced failing-before/passing-after and fixed. See [#48 qualification](../docs/issue48-qualification.md) for tests, exact commits and limits.
- Two local Docker builds failed at npm setup, including a proxy retry; no local image/startup pass is claimed. No paid calls, deployment, merge or original-scope human acceptance occurred.

## Next Priorities

1. Finish review/integration of #48 without treating it as completion of #35.
2. Address #49 (revision/state isolation) and #51 (standalone provider policy), then #50 (whole-task finalized deliverables). Downloads/UI, follow-up/feedback, progress/configuration and broader evidence remain separate follow-ups.
3. Keep #36 Tree of Thoughts gated. Backend cancellation and durable resume remain unsupported; ToT is not the next priority while #35 blockers remain.
