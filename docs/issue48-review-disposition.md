# PR 62 review disposition

The review fixes are in `8d0a9065b83c68ec81ced8ec6218553d451552ab`. This is a focused continuation of [PR #62](https://github.com/dhar174/advanced_prompting_system/pull/62), not acceptance of issue #35 or the separate #49–#61 workstreams. The PR remains open for human integration.

## Comment-to-evidence matrix

All links below identify the original actionable inline comment, including Codex's overlapping finding. The maintained tests are in `test_pr62_review_regressions.py` and `test_pr62_review_integration.py` unless another file is named.

| Comment | Repair and resulting contract | Focused regression |
|---|---|---|
| [Codex: discarded collaboration](https://github.com/dhar174/advanced_prompting_system/pull/62#discussion_r4572094675) | Solo exploration returns a fresh candidate from copied accepted base plus the replacement. Discarded current-step records, outputs and artifact identities are excluded; earlier accepted work remains. | `test_exploration_atomically_replaces_equal_length_candidate`; HTTP `test_http_solo_takeover_excludes_discarded_debate_and_artifacts` |
| [Antigravity: atomic takeover and zero](https://github.com/dhar174/advanced_prompting_system/pull/62#discussion_r4572600874) | Replace equal-length steps/reflections together; accept only valid strict improvement; measured zero explores, unknown quality does not. Rebase every replacement step/reflection consistently. | Zero/low HTTP parameters; rejection controls; `test_multi_step_exploration_rebases_each_shared_reflection_once`; unknown-score no-dispatch control |
| [Empty eligible error](https://github.com/dhar174/advanced_prompting_system/pull/62#discussion_r4572600881) | Common failure recorder supplies typed error and copied fatal partial diagnostics before either failure break. A solo candidate cannot borrow a previous plan step's answer to mask current failure. | `test_defensive_empty_eligible_branch_keeps_structured_error`; real HTTP `test_http_fatal_partial_work_has_errors_and_both_steps_diagnostics` |
| [Serialized error dictionaries](https://github.com/dhar174/advanced_prompting_system/pull/62#discussion_r4572600886) | Validate/copy non-null error models or dictionaries at construction; retain error and run state through dictionary projections. | `test_serialized_error_dict_is_validated_and_copied` |
| [Engineer state and final judge](https://github.com/dhar174/advanced_prompting_system/pull/62#discussion_r4572600888) | Reset attempted IDs and convenience metadata per main run, publish candidate metadata only after acceptance. Validate finite final scores in `[0,1]`; invalid/refused/failed evaluations remain unknown with a safe incomplete result. Preserve explicit policy stops and avoid main-level evaluator retry. | `test_reused_engine_resets_bookkeeping_and_keeps_shared_provider_ledger`; endpoint/invalid-score controls; policy-stop parameters; HTTP final evaluation and selected-unevaluated cases |
| [Mid-conversation failure](https://github.com/dhar174/advanced_prompting_system/pull/62#discussion_r4572600892) | Per-invocation live progress retains copied actual history and normalized legacy questions on ordinary `LLMCallError`; policy exceptions retain separate propagation. | `test_mid_conversation_provider_failure_retains_accumulated_turns`; provider-failure HTTP parameter; existing shared-ceiling tests |
| [One-artifact invariant](https://github.com/dhar174/advanced_prompting_system/pull/62#discussion_r4572600897) | Producer, resolver and legacy projection agree on exactly one registered artifact. | `test_artifact_output_requires_exactly_one_reference` |
| [Empty payloads and annotations](https://github.com/dhar174/advanced_prompting_system/pull/62#discussion_r4572600900) | Reject empty JSON/CSV collections before writing. All four finalization branches use explicit failure text or bounded basename/MIME/size artifact summaries. | Empty dictionary/list × JSON/CSV parameters; `test_failure_and_artifact_annotations_are_bounded_and_honest` |
| [Public error guarantee](https://github.com/dhar174/advanced_prompting_system/pull/62#discussion_r4572600903) | Terminal error always has a safe structured error and `run_error`, including useful prior output and missing upstream diagnostics. | `test_error_with_prior_output_always_has_public_error_and_event`; fatal second-step HTTP cases |
| [Transcript order and identity](https://github.com/dhar174/advanced_prompting_system/pull/62#discussion_r4572600908) | Interleave each step's actual history and completed summary, attach plan identity, and label non-success output as partial. Only successful completion gets `Finalizer`. | `test_partial_failure_metadata_and_step_order_survive_public_mapping`; fatal partial and evaluation HTTP cases |
| [Acceptance coverage](https://github.com/dhar174/advanced_prompting_system/pull/62#discussion_r4572600912) | Provider-only HTTP proves takeover, partial failure and path-like text safety. Actual service orchestration proves owned success before foreign writer denial under reused/colliding labels. Actual nested adapter/converter/registry proves forged foreign and unregistered reference read/list denial after owner success. | New integration suite; `test_actual_collaborative_adapter_denies_foreign_typed_artifact_after_owner_success`; existing parallel/traversal/symlink/overwrite tests |

## Evidence boundaries and reproduction corrections

At reviewed head `b934607`, ordinary self-consistency candidates have no collaboration records. Consequently failed collaboration plus a partial solo candidate does **not** itself make `eligible` empty: the solo candidate remains eligible. The literal empty-list branch is covered as a defensive unit test. The provider-only HTTP scenario separately demonstrates the real failed-conversation/partial-solo behavior and public failure contract. No impossible mocked candidate is presented as end-to-end evidence.

Similarly, the old direct in-memory `from_dict(to_dict())` retained an error model; the broken case was a serialized dictionary error. The regression exercises that actual boundary.

GraphQL does not accept caller-supplied correlation labels. Colliding/reused labels therefore enter through supported `RunRequest` service input, with the real dispatcher, engine, nested provider and writer. Providers cannot manufacture typed `ArtifactRef` objects; the forged-reference test replaces only the typed producer, then exercises real nested selection, conversion and registry authorization. The provider-only HTTP tests replace no orchestration, selection, judge or registry reader.

Five self-consistency paths can independently evaluate five distinct candidates. The evaluator-unavailable HTTP regression distinguishes those attempts from a prohibited main-level retry/finalization of the selected unevaluated candidate. Direct final-evaluation policy tests assert exactly one attempt on budget, timeout or cancellation.

## Qualification

- Full guarded Python 3.11.18 and 3.12.14 suites: **389 passed each**, including six Chromium scenarios, ownership/CORS and evidence verification.
- Frontend: **147 tests across 17 files**, network-denial probe, lint and production build pass.
- Fresh independent contract review: no remaining blockers after repairing shared-reflection rebasing and nullable self-consistency merge/error propagation.
- CRLF-aware whitespace check passes. Original outbound guards are unchanged.
- Exact documentation/PR head and clean production image qualification are established by the latest [PR checks](https://github.com/dhar174/advanced_prompting_system/pull/62/checks), separately from historical `b934607` results. Individual replies record the exact verified head and workflow links.

Commands and environment limitations remain in [qualification notes](issue48-qualification.md). Install `en_core_web_sm`, frontend dependencies (including Apollo), and Chromium before guarded execution; missing local prerequisites are not evidence of a code regression or a passing qualification. No paid calls, deployment, merge or issue #35 closure occurred.
