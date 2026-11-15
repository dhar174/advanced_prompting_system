# Issue 51: finite standalone provider policy

Original implementation source before PR #64 review repairs: `8f4300ac6c83d63f4ec50dbfeb90e79dc278cb83`, based on merged main `a85a913f673fe65afb22485b1e919f7fcaedebe1` (PR #63 / #49). The PR description records the current tested head and later review-repair evidence. Historical results do not qualify a newer head.

## Supported behavior

- `AdvancedPromptEngineer.main(task) -> Interaction` establishes a provider scope before retrieval, rephrasing, output discovery or complexity/planning. It composes with the existing artifact scope and cleans up owned scopes on every exit.
- `provider_client.py` owns canonical provider configuration and validation; `execution_modes.py` retains compatibility exports without making the engine import orchestration. Defaults remain 40 calls and 120 seconds, controlled by `APS_PROVIDER_MAX_CALLS` and `APS_PROVIDER_TIMEOUT_SECONDS`. Canonical malformed, nonfinite and excessive settings fail rather than silently clamping.
- Optional constructor `max_calls` and `timeout_seconds` may tighten environment ceilings, not expand them. Explicit zero calls dispatch nothing; timeout must be positive and finite. Omitted model selection inherits the effective default; an explicit model must pass the allowlist, even when it equals the historical default string. Configuration mutations are revalidated at the run boundary.
- The user selected **caller policy wins**: an existing valid finite context retains its identity, consumed ledger, deadline, model policy and cancellation signal. Constructor call/time limits apply only to standalone roots. Incomplete/unbounded parent policies, inconsistent consumed counts and conflicting explicit models fail before dispatch.
- Planning, embeddings, generation, revisions, selection, judging, complexity and nested collaboration share one ledger. Embedding identifiers remain separate from the chat-model allowlist. SDK retries remain disabled, explicit retry attempts are charged individually and retry waits are clipped to remaining time. Restarts and later plan steps cannot replenish the run allowance.
- Reservation markers are bound to a specific context, and the whole-run entry suspends/restores ambient single-call reservations. An outer reservation cannot make an entire engine run free. Normal explicit reservation/proxy pairs still charge exactly once.
- Early policy stops return a fresh safe non-success Interaction with unknown quality. Later stops retain useful accepted work and diagnostics, without promoting partial content to success. Every return checks deadline/cancellation; exactly consuming the final permitted call does not itself invalidate a completed result.

The timeout contract is cooperative: SDK timeouts receive remaining time, late results are rejected and no further work is dispatched after expiry. This does not hard-preempt arbitrary Python code that ignores its timeout. Unexpected non-policy exceptions preserve their previous propagation behavior while restoring owned contexts.

## Regression map

| Contract | Maintained evidence |
|---|---|
| Shared canonical configuration, strict environment/constructor validation, omitted/explicit model and tighten-only precedence | `test_standalone_policy_config.py` |
| Real cap 0/1 dispatch counts, inherited model, finite parent composition, exact-last-call success, embedding accounting and SDK options | `test_standalone_provider_boundary.py` |
| Early/mid-run stops, late result rejection, cancellation, context restoration, ambient reservation isolation and malformed/rolled-back parent ledger denial | `test_standalone_provider_boundary.py` |
| Two required nested steps under one finite ledger, earlier accepted work after exhaustion, accepted/rejected/provider-failed revisions | `test_standalone_policy_integration.py` |
| Actual explicit retry attempt/wait, last-result cancellation, independent simultaneous runs and executable literal README example | `test_standalone_policy_integration.py` |
| Existing protected unified/legacy HTTP, typed results, ownership, measured quality, revision isolation and partial diagnostics | Full guarded suite, including #48/#49 integration regressions |

Direct integration fixtures replace only the SDK/provider below orchestration. They do not replace the dispatcher, main loop, conversation runner, selection, judges or registry reader. Narrow cleanup units are labeled separately. Existing outbound guards are unchanged.

Before PR publication, independent whole-branch review found two additional accounting gaps: ambient reservations bypassing some charges and invalid consumed counts entering a parent policy. Both received witnessed failing regressions and repairs, together with malformed-number validation controls. Final independent re-review found no remaining actionable findings; it separately passed 66 configuration/boundary/provider checks and nine integration scenarios. A final numeric consistency regression also proves an extremely large constructor timeout raises a clear ValueError rather than overflowing during validation.

## PR 64 review repairs

Antigravity's review of published head `247bc73` identified three further repairs and the qualification prerequisite documented below. Fix source: `684053885775b9b4e9e4b6888271b87883c0abcc`.

- The context uses a reentrant lock so a cancellation callback can synchronously call `get_execution_stats()` without deadlocking. Cancellation, timeout and disallowed-model termination updates use that same lock; SDK calls remain outside it. Shared deadline checks replace duplicate unlocked post-call updates and preserve exact-last-call success.
- `provider_execution_scope(max_calls=...)` accepts only `None` or an exact nonnegative integer. Booleans, fractions, numeric strings, negatives and nonfinite values fail before installing a new context. Omission, zero and trusted caller allowances such as 300 remain supported; the standalone environment ceiling does not constrain an existing caller policy.
- Plan-step finalization now has one shared filename-generation block. Policy exceptions propagate immediately instead of producing a synthetic filename; ordinary helper errors retain the existing fallback.

`test_pr64_review_regressions.py` has 21 cases. Four bounded subprocesses reproduce the former callback/stats deadlock without leaving hung workers. SDK-backed tests retain one billed/dispatched call before cancellation in direct/reserved paths, and two actual calls before budget/timeout/cancellation at filename generation. Strict-scope tests preserve the parent on failure and cover `None`/0/300; an ordinary filename error still returns its fallback. Nine initial failures and three filename failures were witnessed before repair. The aggregate focused run passed 97 cases; a fresh independent reviewer separately passed all 21 review regressions and found no actionable findings at the fix source. The PR description records full-suite and exact-head CI results after qualification.

At the review-fix source, full guarded suites passed **525 tests on Python 3.12.14 (57.59s) and Python 3.11.18 (70.47s)**, each including the ten browser-suite checks. CRLF-aware diff checks are clean. No frontend/API/configuration files changed in this repair, so local frontend checks were not repeated; the new-head CI still qualifies the frontend, browser and clean production image. At this documentation checkpoint, that new-head CI and remote review replies remain pending; the PR description and replies record their final outcomes.

## Verification and remaining gates

### Multi-step qualification prerequisite

Install spaCy's **`en_core_web_sm`** model during environment setup, before guarded tests:

```sh
python -m spacy download en_core_web_sm
```

Use the same Python environment for setup and test execution. Exact-head backend/browser CI installs this model explicitly. The direct test `test_direct_nested_multistep_cap_is_shared_and_preserves_prior_output` and existing HTTP multi-step tests require it: without the model, `is_complex_llm()` returns its supported safe `0.0` fallback and the engine creates a one-step fallback plan, so those two-step assertions cannot qualify their intended behavior. A missing-model failure is a setup failure for these qualification scenarios, not evidence that the multi-step contract passed. Do not download assets during guarded execution or replace/skip planning to hide the prerequisite.

Production's missing-model fallback remains supported and is qualified separately by import/fallback and runtime-image checks. The test prerequisite does not make NLP asset loading eager or require a model download during application startup.

### Commands and historical baseline

The final source qualification commands are:

```sh
PLAYWRIGHT_BROWSERS_PATH=/absolute/browser/cache bash tools/run-offline.sh -q
bash tools/run-frontend-offline.sh guard
bash tools/run-frontend-offline.sh test
bash tools/run-frontend-offline.sh lint
bash tools/run-frontend-offline.sh build
git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --check
```

Historical full guarded local suites at the original implementation source passed **504 tests on Python 3.12.14 (55.51s) and Python 3.11.18 (74.95s)**. Each full count includes ten browser-suite checks: eight actual Chromium scenarios, ownership/CORS and evidence verification. The focused guarded run passed 76 tests. Frontend network denial, 147 tests across 17 files, lint and production build passed; the existing Browserslist warning remains.

The original published head `247bc734f93f160d8914cb06b13dc734932fdefe` subsequently passed both push/PR workflows and downloaded evidence verification. Review repairs require their own current-head qualification. The PR description records current outcomes, exact workflow links and artifact verification after those gates settle.

Local environments, NLP assets and browser binaries were reused. Frontend builds do not overlap browser runs. Clean dependency setup and tracked-source non-root runtime-image qualification are read from current-head CI, not inferred from local tests. Browser evidence retains per-scenario ledgers, HTTP/server records, traces, screenshots, manifests and hashes. Bundle verification reconciles totals, source identity, retained file hashes and nested-archive sentinel scans.

This resolves #51 only. Whole-task finalized deliverable validation remains #50; aliases, `.env` startup loading and broader configuration/setup remain #56/#57. Broader acceptance evidence remains #59/#60/#61, production log retention remains #58, and #36 ToT stays gated. No new streaming, shared-engine concurrency API, hard-kill mechanism or durable resume is included. Public mutations, application-token access, legacy tuples and lazy imports remain supported. No paid calls, deployment, merge or #35 closure occurred.
