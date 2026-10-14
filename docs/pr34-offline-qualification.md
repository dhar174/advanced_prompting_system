# PR #34 offline qualification candidate

This is an unpublished Phase 0–1 candidate for [completion issue #35](https://github.com/dhar174/advanced_prompting_system/issues/35).
The ten latest PR #34 findings have local repairs and production-path regression
coverage. This is **not full-system, release, or browser end-to-end certification**.
There are 24 explicit legacy test quarantines, frontend lint failures, and later
adapter/revision/API/UI work still open. No issue or review thread was closed.

## Identity and scope

- Repository: `dhar174/advanced_prompting_system`.
- Checkout: `D:\projects\advanced_prompting_system\advanced_prompting_system` on BORGBRAIN.
- Original branch: `fix/system-stabilization-and-memory-enhancements`.
- Original PR head: `f16ea9dc37dbae1f41624f33bbdd1200b5610c17`; main: `09c3f2d1956b2f2c6b8d4512df9d01e598b4363a`.
- Local candidate branch: `codex/pr34-offline-qualification`. `git rev-parse HEAD`
  identifies the candidate; `.qualification/evidence.json` records the exact tested
  commit, source/artifact hashes, commands, results and environment snapshot.
- Initial tree was clean, with one checkout. Existing branches and global settings
  were preserved. All test environments, dependencies, caches and generated artifacts
  are project-local. Existing system Python packages were used read-only.
- No push, merge, deployment, external comment, paid model call, model download,
  issue closure, UI expansion or ToT implementation was performed.

## Findings and regression evidence

All links refer to review 5510636681 at the pinned original PR head. Disposition is
“repaired locally,” not maintainer approval or remote thread resolution. The repair
commit is the candidate commit recorded in the evidence manifest.

| Review finding | Repair | Production regression |
|---|---|---|
| [PlanStep caller mapping](https://github.com/dhar174/advanced_prompting_system/pull/34#discussion_r4264589713) | `main()` passes explicit keyword arguments, the actual PlanStep, and an integer collaborative budget | `test_main_calls_real_adapter_with_planstep` executes real `main()` into the real adapter, stopping at the selection boundary |
| [Direct spaCy consumers](https://github.com/dhar174/advanced_prompting_system/pull/34#discussion_r4264589780) | Every remaining direct consumer resolves the lazy NLP object and handles absence; lexical filtering preserves input without optional NLTK data | `test_optional_spacy_unavailable_does_not_call_provider` (five consumers, None/False), cached success/failure and lexical-data tests in `test_offline_runtime.py` |
| [Partial-majority definition](https://github.com/dhar174/advanced_prompting_system/pull/34#discussion_r4264589821) | Return the selected definition | `test_partial_majority_returns_selected_definition` |
| [Continuation vote errors](https://github.com/dhar174/advanced_prompting_system/pull/34#discussion_r4264589863) | Count only CONTINUE/END as valid votes | `test_all_error_continue_vote_enters_bounded_fallback` |
| [Low-participation finalization](https://github.com/dhar174/advanced_prompting_system/pull/34#discussion_r4264589898) | Set an explicit finalization decision | `test_low_participation_finalizes` |
| [Legacy arguments](https://github.com/dhar174/advanced_prompting_system/pull/34#discussion_r4264589916) | Deserialize through normal ingestion; also preserve legacy decisions | `test_memory_legacy_forms_round_trip`, existing typed/dict/compaction memory tests |
| [List to-do entries](https://github.com/dhar174/advanced_prompting_system/pull/34#discussion_r4264589953) | Deserialize through `add_to_do_item()` | List to-do round-trip parameter and existing memory tests |
| [Normalization external calls](https://github.com/dhar174/advanced_prompting_system/pull/34#discussion_r4264589982) | Explicit shared fixture covers helper and direct structured SDK boundaries | Three real `run_conversation()` normalization tests under fail-on-unconfigured-SDK policy |
| [Smoke external calls](https://github.com/dhar174/advanced_prompting_system/pull/34#discussion_r4264590018) | Same explicit fixture; real orchestrator and tuple assertion retained | `TestSmokeSuite.test_4_run_conversation_contract_mock_shape` |
| [Copied vote test logic](https://github.com/dhar174/advanced_prompting_system/pull/34#discussion_r4264590055) | Remove copied append loop; assert actual returned production history | `test_final_vote_branches_execute_production`: unanimous YES, majority NO, tie, all-error, weak YES majority |

Independent review also found the majority-NO/weak-YES path could bypass all
finalization, and a dissent revision could use an undefined generation prompt.
Both now route through mediator handling and generate from the revised decision.
Revision and mediator-failure tests exercise NO/tie/all-error cases with one
mediator attempt and one finalization. Existing mediator-error policy still keeps
the current proposal; this is bounded legacy behavior, not a claim of correctness
or verification of that proposal.

Imports no longer construct OpenAI clients in these three modules, download NLTK
data, eagerly import the transformer pipeline, or execute the structured-classifier
experiment. SDK resource proxies resolve only on invocation. The embedding boundary
preserves explicit missing-configuration errors and initializes diagnostics before
lazy construction can fail. Its pre-existing OFFLINE_MODE seeded-vector fallback
is retained; it establishes structural test behavior, not semantic quality.

Read-only independent review identified the voting/import/NLP/error-diagnostic
problems above and rechecked their repairs. Final targeted recheck found no unresolved
introduced defect. The reviewer did not execute tests; execution evidence is below.

## Environment, setup and authoritative commands

Qualified on Windows 11, CPython 3.12.2, Node 24.13.0, pytest 8.4.2, OpenAI SDK
3.26.0, Pydantic 2.13.5, spaCy 3.8.16, Strawberry 0.327.7, Flask 3.1.2, Vitest 3.1.4
and Vite 6.3.5. Complete distribution versions are in `.qualification/environment.json`.
This is one measured environment, not a supported-version range or a clean lockfile
installation proof. The repository still needs backend dependency metadata.

Setup used `uv venv --python <existing-Python312> --system-site-packages .qualification/venv`.
Flask was absent; `uv pip install --python .qualification/venv/Scripts/python.exe Flask==3.1.2`
installed Flask and its dependencies locally. Frontend setup used the existing lockfile:
`npm ci --ignore-scripts --no-audit --no-fund` (404 packages). Dependency installation
was an explicit setup stage; no model assets were installed. Both setup commands
used project-local caches/TEMP. See `python-setup-flask.txt` and `npm-ci.txt` in the logs.

From the repository root in PowerShell:

```powershell
./tools/run-offline.ps1 -PytestArgs '--collect-only','-q'
./tools/run-offline.ps1 -PytestArgs '-q','--junitxml=.qualification/logs/backend-final.xml'
./tools/run-frontend-offline.ps1 -Action guard
./tools/run-frontend-offline.ps1 -Action test
./tools/run-frontend-offline.ps1 -Action lint
./tools/run-frontend-offline.ps1 -Action build
git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --check
```

The Python wrapper removes provider credentials, disables plugin autoload, scopes
caches/temp directories, and preloads a socket audit guard before collection and
in Python subprocesses. External DNS/TCP/UDP raises a BaseException so ordinary
application exception handling cannot convert it to success. Unconfigured lazy SDK
calls fail tests. Tests run in isolated project-local temporary working directories.
Fresh-process tests assert imports do not construct clients/load models/download data,
external attempts are denied, and loopback works. Node's preload guards socket/DNS
lookup/datagram calls and removes credentials; a separate Node denial probe passes.
These are process-level test controls, not an OS firewall or arbitrary native-process
sandbox. No OS policy was changed. Frontend tests use fixture network boundaries.

## Results and evidence limits

The machine-readable manifest is the authority for final exact-commit results.
Logs are retained in `.qualification/logs/`; they contain fixture data only.

| Check | Result | Evidence |
|---|---|---|
| Original backend collection, guarded/no key | Exit 2, five collection errors, 15 collected before errors | `baseline-collection.txt` |
| After first import prerequisites | Exit 2, three remaining collection errors from import-time classifier experiment | `baseline-runtime.txt` |
| Baseline after import repair | 58 passed / 4 failed (unconfigured normalization/smoke calls) | `baseline-after-import-fixes.txt` |
| Focused review regression before repairs | 19 failed / 3 passed | `regressions-before.txt` |
| First repaired focused run | 21 passed / 1 failed, exposing majority-NO defect | `regressions-after.txt` |
| Full previously excluded advanced audit after fixture target correction and preserved embedding policy | 24 passed / 24 failed; failures explicitly quarantined | `legacy-complete-audit.txt` |
| Maintained final backend suite | Expected 124 passed / 24 skipped; exact commit outcome in manifest and final XML | `backend-final.txt`, `backend-final.xml` |
| Flask/GraphQL read-only startup | Passed real application import and test-client GET `{ hello }` | `api-startup.txt` |
| Frontend tests | 111 passed in 10 files | `frontend-test.txt` |
| Frontend TypeScript + production build | Passed | `frontend-build.txt` |
| Frontend lint | Failed: 13 errors, 1 warning | `frontend-lint.txt` |
| Independent final source review | No unresolved introduced defect in inspected scope | Review summary above |

The frontend source was unchanged during this work. Its lint debt includes explicit
`any` in tests, empty React attribute interfaces, unused catch bindings/imports,
and a Fast Refresh export warning. The backend retains invalid-escape and dependency
deprecation warnings. Build reports outdated Browserslist data; no update was made.

No broad test filename ignores remain. [The per-method inventory](offline-test-inventory.md)
records all 24 quarantines, restored tests and historical parser helpers. `test_b.py`
and `test_c.py` contain alternate helper implementations and no collected tests.
The earlier duplicate finalization test class was shadowed at import; it is retained
under an explicitly historical non-test name. The existing `historical_verification`
scripts and root `validation_test.py` / `final_verification_test.py` are historical
diagnostics, not certifying tests. `.qualification`, frontend dependencies and frontend
tests are deliberately outside backend discovery.

## Execution and side-effect inventory

| Surface | Caller and contract | Classification / qualification |
|---|---|---|
| `AdvancedPromptEngineer.main(task)` | Standalone entry; returns Interaction; retrieval → planning → collaborative selection → self-consistency → artifacts | Supported target, not yet fully qualified; actual adapter binding proved, full run/revision/agents=0 deferred to #35 Phase 2 |
| `run_conversation(...)` | GraphQL mutation and advanced collaborative adapter; legacy `(history, questions, final_output)` tuple | Supported target; affected voting/normalization/memory paths exercised with explicit boundary fixtures |
| `collaborative_reasoning_main`, `invoke_collaborative_reasoning` | Internal advanced selection/tuple-to-output/Interaction adapters | Internal helpers; direct conversion tests plus real caller binding; complete chain remains Phase 2 |
| `self_consistency`, legacy `collaborative_reasoning` | Advanced nested reasoning/revision paths | Internal helpers with known candidate-state/revision/budget debt; not independent UI modes |
| GraphQL `runConversation`, `submitFeedback` | Existing React hooks → Flask `/graphql` → schema | Only collaborative mode currently exposed; schema contract tests and startup pass; no full-stack browser proof |
| React conversation/configuration components | Current conversation mutation, personality/round controls, history and feedback rendering | 111 fixture-driven tests and build pass; no execution-mode registry or advanced-pipeline UI integration |
| Genuine Tree of Thoughts | Planned in #36 | Future capability; not implemented or enabled |
| Root finance/meeting/example scripts, output-generator backups/fixed copies, historical verification | Standalone historical examples and duplicate implementations | Not supported application modes; not executed or certified |

Provider inventory: `conversation_manager` and `advanced_prompting` use lazy `chat.completions.create`
and `beta.chat.completions.parse` for definitions, voting, extraction, planning, judging,
revision and output selection. `complexity_measures` also uses structured/text completions
and embeddings. Legacy functions still read/set module-level OpenAI key configuration
when explicitly invoked; shared run-level configuration remains Phase 3. Retrieval's
external-search helper currently returns mock snippets, not a verified live RAG service.
Optional transformer SRL and spaCy paths load only on demand; tokenizer assets can
still request a download when explicitly used, which the offline guard rejects.

Remaining import/file effects: complexity configures timestamped logging and reads or
trains/saves a small local sklearn pickle; feedback creates a relative `logs` directory.
Conversation execution writes `conversation_history.json`; advanced execution and
output generators write output files/project directories. Tests isolate runtime working
directories under `.qualification`, while import-time local artifacts stay in the
authorized repository. None of this establishes durable resumable runs or safe remote
artifact serving. Memory tests cover ingestion, round trips and compaction only.

## Next work and release blockers

1. Review this local candidate; keep PR #34 open. Publishing is a separate decision.
2. Resolve the 24 documented advanced-test quarantines under #35 Phase 2 with scripted
   providers and bounded-call assertions, including changed revision text, candidate
   isolation, parser merge/reflection semantics, score/Plan shape and agents=0.
3. Repair the 13 frontend lint errors, establish backend dependency metadata and a
   reproducible clean-environment gate. No required test workflow currently exists;
   `.github/workflows/remove-logs.yml` is not a qualification workflow.
4. Continue #35 shared contracts, real API dispatch and UI mode/lifecycle integration,
   followed by the browser path matrix. No UI path is promoted based on these unit tests.
5. Implement genuine ToT separately in #36 after its shared provider/search contracts
   are ready. No reasoning-quality claim follows from this qualification.

There is no remaining filesystem permission blocker. Release acceptance remains open
because of the documented test/lint/dependency/end-to-end gaps; no remote action is pending.
