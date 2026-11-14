# Issue 49 implementation plan

Approved in the conversation after reviewing all 17 remaining #35 follow-ups.
Base: merged main `fddf9f975f27df4975b354959feddd86d9e7ab77`.

## Goal and decisions

Make all correction/revision paths atomic, deeply isolated and locally bounded while preserving #48's typed ownership/result contracts. Use deterministic providers and existing network guards; no paid calls, deployment, merges or #35 closure. #50 owns finalized-deliverable guarantees and #51 owns establishing standalone provider policy.

1. Replace recursive final-output analysis with an iterative typed controller. Keep the legacy string projection and positional arguments; add keyword-only `max_version=5`. Positive integer versions are controller-owned; at the ceiling dispatch nothing. Each correction attempt consumes a version and shared provider calls. Correct actual accepted text, never examples. Preserve accepted content on rejection/failure; report safe non-success through typed production callers.
2. The user chose no-regression acceptance: freeze initial unique requirement identities plus format/alignment/completeness flags. A proposal must retain every passed check and improve at least one failed check. Missing/changed checks, ties, regressions and unknown evaluation cannot replace accepted state.
3. Deep-copy nested candidates/task/plan/steps/reflections/outputs/metadata on empty, rejected, merged and restart paths. Reward-based revisions use independently validated finite evaluator scores and strict improvement; never accept a revision on its authored reward alone. Preserve prior accepted work, nullable quality and artifact filtering.
4. Zero explicit step/call allowance performs no generation; invalid/negative allowances fail before dispatch. Zero restart/revision limits disable those attempts. Keep intentional zero sentinels such as agents=0 and max_plan_steps=0. Separate reasoning IDs from allowance counters. Model count tags cannot set executable remaining budgets; restarts never replenish consumed allowance or caller policy.

## Deliverables and verification

- [x] Typed final-output correction controller and all four production finalization branches; focused red/green tests; commit.
- [x] Deep candidate snapshots, independent revision evaluation and controller limits; red/green tests; commit.
- [ ] Provider-only real conversation/main/HTTP accepted/rejected revision scenarios and built-UI browser proof; commit.
- [ ] Guarded full Python 3.11/3.12, frontend guard/tests/lint/build, Chromium, clean runtime-image qualification, independent whole-branch review and CRLF-aware diff checks.
- [ ] Exact-source qualification documentation, PR, and verified evidence with remaining limitations. No automatic issue acceptance or merge.

Critical regressions: version5/max5 zero work; version4/max5 one correction; version5/max8 at most three; actual failing text; no-regression/changed-checklist rejection; SDK/refusal/malformed/timeout safety; deep nested mutations; zero/negative limits; reasoning ID12 with two steps; count999999; consumed calls across rejection/restart; existing #48 regressions remain green.
