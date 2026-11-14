# Project Progress & Baselines (`advanced_prompting_system`)

## Current Qualified Slice: Issue #49

- Approved [atomic revisions plan](../docs/superpowers/plans/2026-11-14-issue49-atomic-revisions.md); branch `fix/issue-49-atomic-revisions`, base `fddf9f9`.
- Implementation commits: `10cab82` typed correction, `b01b5b5` deep isolation/controller allowances, and current `0980b4e0ff222188e880cc68266c3ff044e25ca2` HTTP/browser qualification and review repairs.
- No #49 PR yet. PR creation, current-head CI/runtime image and human review remain pending; implementation does not establish #35 acceptance.

| Verification | Verified current #49 result |
|---|---|
| Full guarded Python 3.12.14 | 439 passed in 54.81s |
| Full guarded Python 3.11.18 | 439 passed in 76.99s |
| Browser checks included in full count | Ten: eight Chromium scenarios, non-browser ownership/CORS and manifest verification |
| Two new Chromium revision cases | Passed separately in 13.17s; also included in full suite |
| Focused owning/real-HTTP qualification | 95 passed in 8.93s |
| Frontend Vitest | 147 passed across 17 files |
| Frontend guard, lint, TypeScript/production build | Passed |
| Independent review | Four findings witnessed RED→GREEN; final re-review found no remaining blockers in this slice |
| Current-head CI / runtime-image qualification | Pending at this documentation checkpoint; later PR checks are authoritative |

Configured environments, NLP assets and browser binaries were reused. Use the guarded wrappers; install assets during setup rather than test execution. Earlier local Docker setup failures are not image passes. Retained execution evidence: ignored `.qualification/issue49/progress.md` and associated logs.

## Implemented and Verified in This Slice

- [x] Iterative typed correction over actual accepted content, lossless legacy projection and absolute controller-owned version ceiling (default five); attempts remain monotonic and share the provider call ledger.
- [x] User-selected frozen-requirement no-regression rule: keep every previously passed check and improve a failed check; changed/missing requirements, ties, regressions or unknown evaluation cannot replace accepted state.
- [x] Deep candidate/task/plan/step/reflection/output/metadata isolation, including empty/restart/rejection paths; explicit zero allowances do not dispatch and authored counts cannot replenish budget.
- [x] Independently judged answer-plus-all-steps reward acceptance; measured 0.61 survives model-authored 1.0. Existing nullable-quality, artifact-filtering and typed-failure regressions remain green.
- [x] Provider-only HTTP and built-UI browser proof for accepted/rejected revisions; four final review repairs and zero/invalid-config controls reproduced before repair.
- [ ] Publish the focused PR/evidence and verify its exact-head CI/runtime image before claiming readiness.

## Historical Milestones, Not Current Acceptance Claims

#48 is closed and [PR #62](https://github.com/dhar174/advanced_prompting_system/pull/62) is merged. It established typed collaboration results, owned output conversion, selected metadata and safe partial outcomes. Prior #48 snapshots of 342 and 389 tests per Python version, and earlier 243/126 or 294/147 backend/frontend counts, are historical evidence rather than the current #49 baseline. Earlier green CI must not be presented as current-head qualification.

The #34/#38 stabilization and #39–#41 review-repair stack established useful guarded API/browser infrastructure; the original-scope audit still found acceptance defects despite green maintained suites. A narrow no-introduced-blocker review is not full-issue acceptance.

## Remaining Work and Acceptance

- [ ] #49 qualification/publication and human review; no current-slice merge is claimed.
- [ ] Issue #35 remains open with follow-ups #44–#61. Next priorities: #51, then #50; remaining API/UI/configuration/evidence work stays outside this slice.
- [ ] Original-scope human acceptance remains outstanding.
- [ ] #36 ToT stays gated; backend cancellation/durable resume remain unsupported. No paid/live calls, deployment or #49 merge occurred here.
