# Project Progress & Baselines (`advanced_prompting_system`)

## Current Qualified Slice: Issue #51

- Approved [finite standalone policy plan](../docs/superpowers/plans/2026-11-14-issue51-standalone-policy.md); branch `fix/issue-51-standalone-provider-policy`, merged-main base `a85a913`.
- Original implementation commits: `f821fd4` shared configuration, `c1fcadd` finite main boundary, `14a77dc` real-path qualification and `8f4300a` numeric validation. Current review-fix source: `684053885775b9b4e9e4b6888271b87883c0abcc`.
- [PR #64](https://github.com/dhar174/advanced_prompting_system/pull/64) is open and unmerged. Its original `247bc73` head passed CI/evidence; current review-fix publication, CI/runtime image, downloaded evidence and comment closure remain pending at this checkpoint. Later PR checks are authoritative.

| Verification | Verified current #51 result |
|---|---|
| Full guarded Python 3.12.14 | 525 passed in 57.59s |
| Full guarded Python 3.11.18 | 525 passed in 70.47s |
| Browser checks in each full suite | Ten: eight actual Chromium scenarios, ownership/CORS and manifest verification |
| Focused configuration/direct/provider/integration checks | 97 passed in 7.84s, including 21 new PR #64 cases |
| Frontend Vitest / guard / lint / build | Prior 147 tests/17 files passed; no UI/API/configuration changes, so not locally repeated in this repair; current-head CI pending |
| Independent review | Fresh review passed 21 review regressions in 8.62s; no actionable findings |
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
- [x] PR #64 callback/stats deadlock and termination locking, strict scope allowances and filename policy propagation repairs; nine policy and three filename failures witnessed before repair. Documented setup-only spaCy prerequisite while retaining production missing-model fallback.
- [ ] Publish current review fixes, verify exact-head CI/runtime image and downloaded browser evidence, then reply individually and resolve verified review threads.

## Historical Milestones

#48 / [PR #62](https://github.com/dhar174/advanced_prompting_system/pull/62) and #49 / [PR #63](https://github.com/dhar174/advanced_prompting_system/pull/63) are merged; both issues are closed. They established typed owned collaboration results, selected plan-linked metadata, atomic no-regression corrections, deep candidate isolation and truthful evaluator quality. The previous 439-test #49 baseline and older counts are historical evidence, not current-head qualification.

#34/#38 and the #39–#41 review stack established guarded API/browser infrastructure. Subsequent audits still found #35 acceptance defects despite green maintained suites. Neither historical CI nor a narrow review establishes original-scope acceptance.

## Remaining Work and Acceptance

- [ ] Complete PR #64 review replies/current-head gates and human review; no #51 merge is claimed.
- [ ] #50 whole-task finalized deliverables is next. API/UI, configuration/setup/operations and broader acceptance evidence remain separate follow-ups.
- [ ] Issue #35 remains open; #61 tracks final requirements disposition and human acceptance.
- [ ] #36 ToT remains gated. No paid/live calls, deployment, merge or #35 closure occurred in this slice.
