# Project Progress & Baselines (`advanced_prompting_system`)

## Current Qualified Slice: Issue #51

- Approved [finite standalone policy plan](../docs/superpowers/plans/2026-11-14-issue51-standalone-policy.md); branch `fix/issue-51-standalone-provider-policy`, merged-main base `a85a913`.
- Implementation commits: `f821fd4` shared configuration, `c1fcadd` finite main boundary, `14a77dc` real-path qualification/review repairs, and current source `8f4300ac6c83d63f4ec50dbfeb90e79dc278cb83` numeric validation consistency.
- At this documentation checkpoint, no #51 PR exists yet. Publication, exact-head CI, clean runtime image and downloaded evidence verification remain pending; later PR checks are authoritative.

| Verification | Verified current #51 result |
|---|---|
| Full guarded Python 3.12.14 | 504 passed in 55.51s |
| Full guarded Python 3.11.18 | 504 passed in 74.95s |
| Browser checks in each full suite | Ten: eight actual Chromium scenarios, ownership/CORS and manifest verification |
| Focused configuration/direct/provider/integration checks | 76 passed in 7.69s |
| Frontend Vitest | 147 passed across 17 files in 13.78s |
| Frontend guard / lint / production build | Passed; build 7.64s; existing Browserslist warning |
| Independent whole-branch review | Two accounting findings repaired RED→GREEN; final recheck has no remaining actionable findings |
| CRLF-aware diff | Clean |
| Exact-head CI / clean runtime image / downloaded evidence | Pending at this checkpoint |

Local environments, NLP assets and browsers were reused. Frontend builds did not overlap browser runs. Use guarded wrappers; install assets during setup. Retained local evidence is under ignored `.qualification/issue51/`; local tests do not prove a clean image build.

## Implemented and Verified

- [x] Canonical provider configuration/model validation in `provider_client.py`, compatibility exports, strict environment values and optional tighten-only standalone limits.
- [x] Supported direct main boundary before early provider work; valid finite caller context wins unchanged; incomplete/unbounded/malformed parent policies fail before dispatch.
- [x] One ledger/deadline across real solo/team, multi-step, revision, selection/judging and embedding calls; zero SDK retries and bounded explicit retries/waits.
- [x] Context-bound reservation accounting, consistent consumed-count validation and no allowance replenishment across later work.
- [x] Safe early/mid-run stops, partial preservation, unknown stopped quality, late-result rejection, cancellation, exact-last-call success, cleanup and independent simultaneous instances.
- [x] Literal corrected README example runs with a deterministic SDK bootstrap; #48/#49 ownership/revision/metadata and protected HTTP regressions remain green.
- [ ] Publish focused PR and verify exact-head CI/runtime image and downloaded browser evidence before declaring delivery complete.

## Historical Milestones

#48 / [PR #62](https://github.com/dhar174/advanced_prompting_system/pull/62) and #49 / [PR #63](https://github.com/dhar174/advanced_prompting_system/pull/63) are merged; both issues are closed. They established typed owned collaboration results, selected plan-linked metadata, atomic no-regression corrections, deep candidate isolation and truthful evaluator quality. The previous 439-test #49 baseline and older counts are historical evidence, not current-head qualification.

#34/#38 and the #39–#41 review stack established guarded API/browser infrastructure. Subsequent audits still found #35 acceptance defects despite green maintained suites. Neither historical CI nor a narrow review establishes original-scope acceptance.

## Remaining Work and Acceptance

- [ ] Complete #51 publication/current-head gates and human review; no #51 merge is claimed.
- [ ] #50 whole-task finalized deliverables is next. API/UI, configuration/setup/operations and broader acceptance evidence remain separate follow-ups.
- [ ] Issue #35 remains open; #61 tracks final requirements disposition and human acceptance.
- [ ] #36 ToT remains gated. No paid/live calls, deployment, merge or #35 closure occurred in this slice.
