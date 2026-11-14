# Active Context (`advanced_prompting_system`)

## Current Focus

- Issue #35 remains open. Its original acceptance backlog is tracked by follow-ups #44–#61; green maintained suites or a narrow review do not establish whole-issue acceptance.
- #48 is closed and [PR #62](https://github.com/dhar174/advanced_prompting_system/pull/62) is merged. Its typed collaboration/owned-output foundation remains in place; do not keep reporting that PR as draft/open.
- Current slice: **#49 atomic revisions and candidate isolation**, on `fix/issue-49-atomic-revisions`, based on `fddf9f9`, under the approved [plan](../docs/superpowers/plans/2026-11-14-issue49-atomic-revisions.md).
- Current qualified implementation source: `0980b4e0ff222188e880cc68266c3ff044e25ca2`, following `10cab82` (typed correction controller) and `b01b5b5` (deep isolation/controller allowances). #49 has no PR yet; PR creation, exact-head CI and runtime-image qualification remain pending with the coordinator.

## Verified #49 Contracts

- Final-output correction is an iterative typed controller over the actual accepted text. Preserve the legacy string/positional projection; keyword-only `max_version=5` is an absolute controller-owned ceiling. At the ceiling dispatch nothing; rejected attempts still consume a monotonically increasing version and the shared provider ledger.
- The user chose **no-regression acceptance**: freeze initial unique requirement identities plus format/alignment/completeness checks. A proposal must retain every passed check and improve at least one failed check. Changed/missing checks, ties, regressions and unknown evaluation cannot replace accepted state.
- Candidate/task/plan/step/reflection/output/metadata snapshots are deep-copied across empty, rejected, merged and restart paths. Rejection retains accepted state and owned artifacts; local zero allowances are not coerced to one, and model-authored counts/versions do not replenish controller budgets.
- Reward-based revisions require independent finite evaluator quality and strict improvement. Composite judgment sees the actual answer plus all proposed steps. A measured 0.61 is retained even if authored reward tags claim 1.0; nullable unknown quality and earlier #48 failure/metadata protections remain intact.
- Earlier #48 foundations remain: typed text/artifact/failure producers, owned conversion/scope lifetime, lossless legacy questions, selected plan-linked metadata, safe partial failure, rejected-artifact filtering and versioned registered output collisions.

## Verified Evidence and Limits

- Full guarded suites at current source: **Python 3.12.14: 439 passed in 54.81s; Python 3.11.18: 439 passed in 76.99s**.
- Full suite includes **10 browser-suite checks: eight real Chromium scenarios, one non-browser ownership/CORS check and one manifest verifier**. Two new real-browser revision cases cover accepted and rejected corrections.
- Frontend: **147 passed across 17 files**; network guard, lint and production build pass. Focused owning/HTTP checks passed 95 cases and the two new Chromium revision scenarios passed separately.
- Four independent review findings were reproduced RED→GREEN and repaired; final re-review found no remaining blockers in the inspected slice. This does not certify broader #35 acceptance.
- At this documentation checkpoint, current exact-head CI and runtime-image qualification are pending, not green; later PR checks are authoritative. Existing local environments/assets were reused; do not claim fresh dependency resolution or local container success. No paid calls, deployment, #49 merge or original-scope human acceptance occurred.

## Next Priorities

1. Finish #49's PR/evidence publication, exact-head CI and review without treating it as completion of #35.
2. Address #51 (standalone provider policy), then #50 (whole-task finalized deliverables). Downloads/UI, follow-up/feedback, progress/configuration and broader acceptance evidence remain separate follow-ups.
3. Keep #36 Tree of Thoughts gated. Backend cancellation and durable resume remain unsupported; ToT is not next while #35 blockers remain.
