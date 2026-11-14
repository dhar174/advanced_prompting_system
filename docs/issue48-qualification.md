# Issue 48: typed collaboration result qualification

[PR #62](https://github.com/dhar174/advanced_prompting_system/pull/62) addresses the nested result/ownership and metadata blockers from [issue #48](https://github.com/dhar174/advanced_prompting_system/issues/48). Issue #35 acceptance remains outstanding. This does not qualify the separate #49–#61 backlog or genuine ToT (#36).

## Review-fix qualification (supersedes the historical counts below)

Antigravity's ten inline comments and review summary, plus Codex's overlapping inline finding, are addressed in `8d0a9065b83c68ec81ced8ec6218553d451552ab`. The [comment disposition matrix](issue48-review-disposition.md) records fixes, regressions, and the distinction between real HTTP/service evidence and defensive unit coverage. Fresh independent review found no remaining blockers.

The guarded full suites now pass **389 tests on each of Python 3.11.18 and 3.12.14**. Frontend network guard, **147 tests**, lint and production build pass. These are local results for the review-fix source; exact PR-head CI, including the tracked-source non-root production image, must be checked separately through [PR checks](https://github.com/dhar174/advanced_prompting_system/pull/62/checks). Historical workflow links and counts below describe the earlier implementation and are not evidence for a later head.

Solo takeover atomically replaces current-step state while retaining prior accepted records. Measured zero can explore; unknown or invalid quality remains unknown. Ordinary conversation failures retain actual partial diagnostics, terminal errors always expose safe errors/events, and per-step transcripts carry ordered identities. Final evaluation failures preserve useful work as incomplete without a main-level retry; explicit policy stops remain distinct. Reused engineer instances reset bookkeeping without replenishing their caller's provider allowance.

## Historical verified source and behavior

The local full-suite source is `4c328b215e4bb38f7412d27945f59d75995a2514`, based on main `619bd150ebbbed94e763a258579228a05b735273`. Subsequent output-mapping indentation/documentation changes are qualified separately by the PR's exact-head checks. The [source-head workflow](https://github.com/dhar174/advanced_prompting_system/actions/runs/43567823628) and [PR checks](https://github.com/dhar174/advanced_prompting_system/pull/62/checks) distinguish final-head CI from these local results.

- `ConversationOutput` explicitly represents text, an owned artifact, or failure. `ConversationResult` carries copied actual history/questions and nullable measured quality. Kind is established by the chosen output tool; ordinary path-like text is never opened.
- `run_conversation_result` and `finalize_output_result` are the internal typed entry points. Public `run_conversation` retains its three-element tuple and `finalize_output` retains its compatibility projection. Legacy question `reply` and `asking_assistant` fields remain available.
- Active-owner registry authority is checked through conversion, including direct adapter calls. Nested calls reuse the existing owner, provider ledger and deadline. Direct artifact scope lifetime is fixed; establishing a standalone provider-policy boundary remains #51.
- Empty/refused/malformed outputs, missing/unreadable/foreign files, and empty tool payloads are explicit non-success outcomes. A nonempty PDF container cannot make an empty tool payload successful. Ordinary content is preserved; structured errors/server-generated artifact annotations do not publish sensitive paths.
- Actual nested output is evaluated with `judge_step`; only a validated evaluator score becomes a reward. Unknown quality remains null and a legitimate measured 1.0 remains valid. Final task quality still comes from the final-answer evaluator.
- Candidates retain copied accepted base state so cumulative comparisons use the same base. Selected plan/step-linked records are committed with accepted state; rejected histories/questions/artifacts are excluded. Repeated registered filenames produce a distinct version, preserving earlier accepted bytes.
- GraphQL Conversation/Question outputs add nullable `planStepNumber`; existing queries and input shapes remain valid. Advanced results serialize actual selected histories/questions. A later typed failure preserves earlier accepted metadata and emits a safe error.

## Regression evidence

| Contract | Maintained proof |
|---|---|
| Owned success before foreign/unregistered denial; reused/colliding labels; parallel owners; symlink/deleted/empty/stat-error files | `test_collaboration_results.py` |
| Text/artifact/failure classification; text resembling a path; PDF empty-payload rejection; versioned bytes; direct scope cleanup | `test_collaboration_results.py` |
| Measured 0.61 and 1.0; unknown evaluator quality; copied base and metadata round trips; lossless legacy questions | `test_collaboration_results.py` |
| Two selected steps; rejected metadata and artifacts; partial second-step failure; exact shared ceiling reached inside the second nested conversation | `test_graphql_integration.py` |
| Actual model-directed writer cannot overwrite a foreign registered artifact; complete-response sentinel checks | `test_graphql_integration.py` |
| Legacy mutations, malformed results, public error safety and existing stabilization contracts | Updated mapping fixtures plus original maintained backend suites |

A fresh independent whole-branch reviewer found four blocking gaps: direct artifact lifetime, rejected-artifact publication, empty PDF tool payloads, and dropped legacy question fields. Each has a witnessed failing-before/passing-after regression. A further overwrite regression protects earlier registered output versions. Final focused unit/HTTP qualification passes 68 cases.

## Historical local verification

| Check | Actual result |
|---|---|
| Guarded full Python 3.12.14 suite | 342 passed, 45.05 seconds |
| Guarded full Python 3.11.18 suite | 342 passed, 61.75 seconds |
| Frontend network-denial probe | Passed |
| Frontend tests | 147 passed across 17 files |
| Frontend lint and TypeScript/production build | Passed; stale Browserslist database warning retained |
| Final additive schema mapping check | 8 passed |
| Repository-prescribed CRLF-aware whitespace check | Passed |
| Exact-source clean CI | [All five jobs passed](https://github.com/dhar174/advanced_prompting_system/actions/runs/43567823628), including Chromium and production runtime image |

The full Python counts include the existing eight browser-suite checks: six real Chromium scenarios, the non-browser ownership/CORS case, and the evidence verifier. Local environments, NLP assets and browser binaries were reused; this is not a new dependency-resolution claim. Frontend builds did not overlap the final browser runs. Earlier failed/setup runs are retained under ignored `.qualification/issue48/` alongside final logs.

Setup must install dependencies/NLP/Chromium before guarded execution and use one explicit browser cache path. From a configured checkout:

```sh
PLAYWRIGHT_BROWSERS_PATH=/absolute/browser/cache bash tools/run-offline.sh -q
bash tools/run-frontend-offline.sh guard
bash tools/run-frontend-offline.sh test
bash tools/run-frontend-offline.sh lint
bash tools/run-frontend-offline.sh build
```

Two local tracked-source Docker builds failed at `npm ci` with npm's “Exit handler never called” error, including a build-proxy retry. No local image/startup pass is claimed. Judge clean Linux setup and the non-root/runtime-only image from the exact-head CI production-image job, not from the development suite or an attempted Docker command.

## Remaining boundaries

General revision/state isolation is #49; whole-task finalized-deliverable invariants are #50; standalone provider policy is #51; authenticated downloads/UI are #52. Usable follow-up/feedback, truthful broader progress, configuration/operator readiness and complete acceptance evidence remain separate. ToT stays gated; backend cancellation/durable resume remain unsupported. No paid provider calls, deployment, merge or original-scope human acceptance occurred during this work.
