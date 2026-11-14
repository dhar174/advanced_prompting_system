# Issue 49: atomic revision qualification

Production source: `0980b4e0ff222188e880cc68266c3ff044e25ca2`, based on merged main `fddf9f975f27df4975b354959feddd86d9e7ab77`. Documentation commits do not change this production implementation. Final PR-head CI is a separate gate; historical #48 results do not qualify these changes.

## Behavior and compatibility

- `analyze_final_output_result` returns an internal `OutputRevisionResult`: accepted `FinalOutput`, its valid analysis, attempted version, stop reason and optional safe error. The existing analyzer remains an accepted-text projection with its positional arguments unchanged and a keyword-only `max_version=5`.
- Versions are positive controller-owned integers. At or above the absolute ceiling no analysis or correction is dispatched. Each correction attempt advances the attempt version, including rejections; only acceptance advances the accepted output's version. Model-returned versions never replace these counters.
- The user selected strict no-regression checks: freeze initial unique requirement identities and format/alignment/completeness flags, retain every passed check, and improve at least one failed check. Changed/missing checklists, ties, regressions and unknown evaluation cannot replace accepted content. Examples remain examples, never correction state.
- Correction helper success remains `FinalOutput`; ordinary failures now return a safe typed `ErrorResult` instead of an error string. All four production finalization branches consume the typed analyzer and skip writers on non-success. Accepted content remains visible as **Partial Output** without violating the failure-kind contract.
- Policy exceptions preserve their budget/timeout/cancellation identity and carry copied accepted revision diagnostics through real conversation/main/dispatcher callers. A stopped correction cannot become a completed run merely because useful content exists.
- Empty/merged/restarted candidates deeply isolate task/plan, steps/reflections, outputs and collaboration metadata. Explicit commit uses stable step identity so copied accepted prefix steps are not duplicated.
- Explicit zero local step/call allowances perform no generation; invalid negative/non-integral values fail before dispatch. Configured `max_sc_calls=0` is preserved. Zero revision/restart limits disable those attempts; `agents=0` and `max_plan_steps=0` retain their intentional meanings.
- Controller counters determine executable remaining budgets. Model count tags cannot grant allowance. Reasoning identity 12 with allowance 2 means two possible steps, not zero. Restarts retain consumed local allowances and the caller's shared provider ledger/deadline.
- Exploration judges a composite containing every proposed reasoning step and the actual answer through the existing independent evaluator. It rejects excess replacement steps, derives their budgets from controller allowance and deep-copies the returned task. This is one aggregate candidate judgment, not a claim that every step has an independently measured score.
- Accepted solo/legacy revisions preserve the measured evaluator reward (for example, 0.61) instead of authored reward/final-reward tags (for example, 1.0). Both supported answer tags remain coherent with accepted legacy revision state.

No GraphQL response shape, authentication model or legacy conversation tuple changes. Existing #48 ownership, rejected-artifact, nullable-quality, partial-diagnostic and takeover regressions remain enabled.

## Regression map

| Contract | Maintained evidence |
|---|---|
| Actual failing text; version 5/5 no work, 4/5 one correction, 5/8 three corrections; fixed checklist and safe failures | `test_revision_integrity.py` |
| Four finalization vote paths skip writers and preserve partial output; policy stop carries last accepted correction | `test_revision_integrity.py` |
| Deep empty/merged/parallel snapshots, zero/invalid limits, model count rejection, identity 12 and restart accounting | `test_revision_limits.py` |
| Independently judged exploration; replacement task/count boundary; measured 0.61 vs authored 1.0 and answer-only legacy revisions | `test_revision_limits.py` plus strengthened #48 review controls |
| Both enabled modes: real HTTP → resolver → dispatcher → engine/conversation → provider/writer, accepted/rejected/provider-failed corrections | `test_revision_integration.py` |
| Both modes retain the last accepted partial correction at an exact shared provider ceiling | `test_revision_integration.py` |
| Built UI accepts a real correction and renders preserved original content on rejected correction | Two additive Chromium scenarios in `test_browser_e2e.py` |

Fixtures are injected below orchestration for integration/browser proofs. Narrow controller units are labeled separately. No dispatcher, main loop, conversation runner, selection or registry reader is replaced in those integration proofs. Existing outbound guards are unchanged; only the shared conversation mapping fixture was migrated to the new typed analyzer boundary.

Fresh independent whole-branch review found four gaps: dynamic task aliasing, dynamic count metadata, measured score replacement by authored tags, and dropped answer-only legacy revisions. Each received a witnessed failing regression and a passing fix. Re-review found no remaining blockers in the inspected slice. Review also strengthened evaluator rejection controls to use actual Reflection objects and browser rejection to assert rendered partial content.

## Verification and limits

Full guarded local suites pass **439 tests on Python 3.11.18 and 3.12.14**. Each full count includes ten browser-suite checks: eight actual Chromium scenarios, ownership/CORS and evidence verification. Frontend guard, **147 tests across 17 files**, lint and TypeScript/production build pass. The existing Browserslist database warning remains.

```sh
PLAYWRIGHT_BROWSERS_PATH=/absolute/browser/cache bash tools/run-offline.sh -q
bash tools/run-frontend-offline.sh guard
bash tools/run-frontend-offline.sh test
bash tools/run-frontend-offline.sh lint
bash tools/run-frontend-offline.sh build
git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --check
```

Local environments, NLP assets and browser binaries were reused. Frontend builds did not overlap browser runs. Clean setup and tracked-source non-root runtime-image qualification must be read from the final PR-head CI, independently of local results. Evidence retains scenario ledgers, HTTP/server records, traces/screenshots, hashes and source identities; the PR description records exact workflow/artifact links after verification.

This slice establishes local revision/controller guarantees and composes with an existing caller-owned provider policy. It does **not** establish the missing standalone provider-policy boundary (#51). Whole-task finalized deliverable validation remains #50; comprehensive browser/acceptance correlation remains #59/#60, and #35 human acceptance remains #61. Supported concurrent HTTP runs use separate engineer instances; this adds no shared-engine concurrency API. Production log redaction/retention remains #58. ToT stays gated under #36. No paid calls, deployment, merge or #35 closure occurred.
