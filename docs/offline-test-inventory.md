# Offline test inventory

Audit of the three files excluded at PR #34 head f16ea9d. Initial audit:
34 failures / 14 passes (`.qualification/logs/legacy-audit.txt`). After completing
Issue #35 Phase 2 quarantine resolution and deterministic offline provider fixtures,
the full test suite stands at **48 passed / 0 failed** (`test_advanced_prompting.py`).
All 24 previously quarantined tests are unquarantined and passing offline.
The offline embedding fallback uses seeded synthetic vectors; passing parser tests
using that fallback prove structural behavior only, not semantic matching or reasoning quality.
No filename ignores remain. A renamed shadowed duplicate class is historical
source, not certifying coverage. Default discovery also excludes `historical_verification`
and the separate frontend/node_modules trees; these are not backend release tests.

Owner/tracking: [completion issue #35](https://github.com/dhar174/advanced_prompting_system/issues/35).
All Phase 2 quarantines were resolved without live provider calls or weakening assertions,
and Issue #35 PR 1–PR 3 expanded offline coverage across execution-mode contracts, GraphQL/HTTP security,
unmocked Flask `/graphql` orchestration (`test_graphql_integration.py`), and offline Playwright Chromium E2E (`test_browser_e2e.py`).

## Suite Expansion Summary (Issue #35 PR 1–PR 3)

| Suite / File | Scope | Disposition |
|---|---|---|
| `test_advanced_prompting.py` | 48 unit/parser/adapter tests | Passing offline |
| `test_issue35_pr1_regressions.py` | Engine retrieval, provider config/model allowlist, artifact scoping | Passing offline |
| `test_execution_modes.py` | `collaborative`, `advanced_pipeline`, gated `tree_of_thought` (`#36`) | Passing offline |
| `test_graphql_contract.py` | Strawberry schema queries, `runExecution`, CORS/Bearer/size/depth/rate-limit checks | Passing offline |
| `test_graphql_integration.py` | 9 unmocked HTTP `/graphql` -> `execute_run` -> real orchestration & artifact tests | Passing offline |
| `test_browser_e2e.py` | 6 offline Chromium Playwright tests (`OfflineE2EServer` + loopback route guard) | Passing offline |
| `collaborative-assistant-frontend/` | 15 Vitest test suites (126 tests), ESLint, TypeScript + Vite build | Passing offline |

| Test or helper | Disposition |
|---|---|
| `test_advanced_prompting.py::TestParseResponse::test_none_response_handling` | Restored to normal collection; passing offline |
| `test_advanced_prompting.py::TestParseResponse::test_non_string_response_handling` | Restored to normal collection; passing offline |
| `test_advanced_prompting.py::TestParseResponse::test_reflection_missing_reward` | Resolved with mock judge/chat fixtures; unquarantined and passing offline |
| `test_advanced_prompting.py::TestParseResponse::test_reflection_with_special_characters` | Restored to normal collection; passing offline |
| `test_advanced_prompting.py::TestParseResponse::test_non_numeric_count_and_reward` | Resolved with mock judge/chat fixtures; unquarantined and passing offline |
| `test_advanced_prompting.py::TestParseResponse::test_steps_and_reflections_objs_provided` | Resolved with deduplicated reflection combiners; unquarantined and passing offline |
| `test_advanced_prompting.py::TestParseResponse::test_overlapping_steps_and_reflections` | Resolved with revision-state reflection repair; unquarantined and passing offline |
| `test_advanced_prompting.py::TestParseResponse::test_unexpected_tags_in_response` | Restored to normal collection; passing offline |
| `test_advanced_prompting.py::TestParseResponse::test_multiple_answers_and_rewards` | Resolved with mock judge/chat fixtures; unquarantined and passing offline |
| `test_advanced_prompting.py::TestParseResponse::test_missing_count_tag` | Restored to normal collection; passing offline |
| `test_advanced_prompting.py::TestParseResponse::test_missing_step_tag` | Restored to normal collection; passing offline |
| `test_advanced_prompting.py::TestParseResponse::test_empty_step_and_reflection_tags` | Resolved with empty description check before embedding; unquarantined and passing offline |
| `test_advanced_prompting.py::TestParseResponse::test_basic_response_parsing` | Restored to normal collection; passing offline |
| `test_advanced_prompting.py::TestParseResponse::test_multiple_steps_parsing` | Restored to normal collection; passing offline |
| `test_advanced_prompting.py::TestParseResponse::test_missing_reflection_handling` | Resolved with mock judge/chat fixtures; unquarantined and passing offline |
| `test_advanced_prompting.py::TestParseResponse::test_budget_counting` | Resolved with mock judge/chat fixtures; unquarantined and passing offline |
| `test_advanced_prompting.py::TestParseResponse::test_interaction_merging` | Restored to normal collection; passing offline |
| `test_advanced_prompting.py::TestParseResponse::test_reward_score_validation` | Resolved with mock judge/chat fixtures; unquarantined and passing offline |
| `test_advanced_prompting.py::TestParseResponse::test_empty_response_handling` | Restored to normal collection; passing offline |
| `test_advanced_prompting.py::TestParseResponse::test_malformed_tags_handling` | Resolved regex requirement for unclosed reflection tags; unquarantined and passing offline |
| `test_advanced_prompting.py::TestParseResponse::test_initial_budget_handling` | Resolved with mock judge/chat fixtures; unquarantined and passing offline |
| `test_advanced_prompting.py::TestParseResponse::test_existing_steps_integration` | Resolved with missing object insertion order in consolidate_steps; unquarantined and passing offline |
| `test_advanced_prompting.py::TestParseResponse::test_reflection_objects_integration` | Resolved with deduplicated reflection combiners; unquarantined and passing offline |
| `test_advanced_prompting.py::TestParseResponse::test_duplicate_step_handling` | Restored to normal collection; passing offline |
| `test_advanced_prompting.py::TestParseResponse::test_reward_range_validation` | Resolved with mock judge/chat fixtures; unquarantined and passing offline |
| `test_advanced_prompting.py::TestAdvancedPromptEngineer::test_count_tokens` | Resolved with offline fallback on tiktoken NetworkForbidden; unquarantined and passing offline |
| `test_advanced_prompting.py::TestAdvancedPromptEngineer::test_generate_initial_prompt` | Resolved unpack contract for (system_prompt, initial_prompt); unquarantined and passing offline |
| `test_advanced_prompting.py::TestAdvancedPromptEngineer::test_judge_step` | Resolved with mock judge/chat fixtures; unquarantined and passing offline |
| `test_advanced_prompting.py::TestAdvancedPromptEngineer::test_component_decision` | Resolved plan iteration and sort key dict_values type; unquarantined and passing offline |
| `test_advanced_prompting.py::TestAdvancedPromptEngineer::test_judge_step_completion` | Resolved CompletionStatus boolean unpacking; unquarantined and passing offline |
| `test_advanced_prompting.py::TestAdvancedPromptEngineer::test_finalize_step_output` | Resolved plan iteration; unquarantined and passing offline |
| `test_advanced_prompting.py::TestAdvancedPromptEngineer::test_automatic_chain_of_thought` | Restored to normal collection; passing offline |
| `test_advanced_prompting.py::TestAdvancedPromptEngineer::test_least_to_most` | Restored to normal collection; passing offline |
| `test_advanced_prompting.py::TestAdvancedPromptEngineer::test_progressive_hint` | Restored to normal collection; passing offline |
| `test_advanced_prompting.py::TestAdvancedPromptEngineer::test_assess_complexity` | Resolved with ml_complexity_model fallback; unquarantined and passing offline |
| `test_advanced_prompting.py::TestAdvancedPromptEngineer::test_adjust_step_budget` | Restored to normal collection; passing offline |
| `test_advanced_prompting.py::TestAdvancedPromptEngineer::test_dynamic_confidence_exploration` | Resolved with mock chat completion fixtures; unquarantined and passing offline |
| `test_advanced_prompting.py::TestAdvancedPromptEngineer::test_merge_interactions` | Resolved deduplication preserving steps order in merge_interactions; unquarantined and passing offline |
| `test_advanced_prompting.py::TestAdvancedPromptEngineer::test_choose_best_response` | Resolved with beta parse mock; unquarantined and passing offline |
| `test_advanced_prompting.py::HistoricalFinalizePlanStepOutputExample::test_finalize_planstep_output_success` | Previously shadowed duplicate; retained as non-certifying example |
| `test_advanced_prompting.py::HistoricalFinalizePlanStepOutputExample::test_finalize_planstep_output_empty_response` | Previously shadowed duplicate; retained as non-certifying example |
| `test_advanced_prompting.py::HistoricalFinalizePlanStepOutputExample::test_finalize_planstep_output_exception_handling` | Previously shadowed duplicate; retained as non-certifying example |
| `test_advanced_prompting.py::TestAdvancedPromptingAdapterMethods::test_determine_output_type_from_content` | Restored to normal collection; passing offline |
| `test_advanced_prompting.py::TestAdvancedPromptingAdapterMethods::test_select_personalities_for_planstep` | Restored to normal collection; passing offline |
| `test_advanced_prompting.py::TestAdvancedPromptingAdapterMethods::test_should_use_multi_agent_reasoning` | Restored to normal collection; passing offline |
| `test_advanced_prompting.py::TestAdvancedPromptingAdapterMethods::test_convert_conversation_output_to_final_step_output_file_exists` | Restored to normal collection; passing offline |
| `test_advanced_prompting.py::TestAdvancedPromptingAdapterMethods::test_convert_conversation_output_to_final_step_output_file_not_exists` | Restored to normal collection; passing offline |
| `test_advanced_prompting.py::TestAdvancedPromptingAdapterMethods::test_invoke_collaborative_reasoning` | Restored to normal collection; passing offline |
| `test_advanced_prompting.py::TestFinalizePlanStepOutput::test_finalize_planstep_output_success` | Restored to normal collection; passing offline |
| `test_advanced_prompting.py::TestFinalizePlanStepOutput::test_finalize_planstep_output_empty_response` | Restored to normal collection; passing offline |
| `test_advanced_prompting.py::TestFinalizePlanStepOutput::test_finalize_planstep_output_exception_handling` | Restored to normal collection; passing offline |
| `test_b.py::consolidate_steps` | Alternate parser helper, no collected tests; retained for historical comparison only; #35 Phase 2 |
| `test_b.py::process_steps` | Alternate parser helper, no collected tests; retained for historical comparison only; #35 Phase 2 |
| `test_b.py::judge_step` | Alternate parser helper, no collected tests; retained for historical comparison only; #35 Phase 2 |
| `test_b.py::cosine_similarity_custom` | Alternate parser helper, no collected tests; retained for historical comparison only; #35 Phase 2 |
| `test_b.py::get_embedding` | Alternate parser helper, no collected tests; retained for historical comparison only; #35 Phase 2 |
| `test_b.py::parse_response` | Alternate parser helper, no collected tests; retained for historical comparison only; #35 Phase 2 |
| `test_b.py::main` | Alternate parser helper, no collected tests; retained for historical comparison only; #35 Phase 2 |
| `test_c.py::judge_step` | Alternate parser helper, no collected tests; retained for historical comparison only; #35 Phase 2 |
| `test_c.py::cosine_similarity_custom` | Alternate parser helper, no collected tests; retained for historical comparison only; #35 Phase 2 |
| `test_c.py::get_embedding` | Alternate parser helper, no collected tests; retained for historical comparison only; #35 Phase 2 |
| `test_c.py::handle_length_mismatch` | Alternate parser helper, no collected tests; retained for historical comparison only; #35 Phase 2 |
| `test_c.py::insert_missing_step` | Alternate parser helper, no collected tests; retained for historical comparison only; #35 Phase 2 |
| `test_c.py::validate_steps` | Alternate parser helper, no collected tests; retained for historical comparison only; #35 Phase 2 |
| `test_c.py::ensure_content_similarity` | Alternate parser helper, no collected tests; retained for historical comparison only; #35 Phase 2 |
| `test_c.py::remove_nonnumeric_counts` | Alternate parser helper, no collected tests; retained for historical comparison only; #35 Phase 2 |
| `test_c.py::consolidate_steps` | Alternate parser helper, no collected tests; retained for historical comparison only; #35 Phase 2 |
| `test_c.py::process_steps` | Alternate parser helper, no collected tests; retained for historical comparison only; #35 Phase 2 |
| `test_c.py::parse_response` | Alternate parser helper, no collected tests; retained for historical comparison only; #35 Phase 2 |
| `test_c.py::main` | Alternate parser helper, no collected tests; retained for historical comparison only; #35 Phase 2 |
