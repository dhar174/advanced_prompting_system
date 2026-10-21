# Offline test inventory

Audit of the three files excluded at PR #34 head f16ea9d. Initial audit:
34 failures / 14 passes (`.qualification/logs/legacy-audit.txt`). After correcting
three SDK patch targets and preserving the existing explicit offline embedding
policy, the full unquarantined audit was **24 failed / 24 passed**
(`.qualification/logs/legacy-complete-audit.txt`). All 24 passing tests are restored.
The 24 failing methods have individual skips below with reasons and removal
conditions. They are unresolved debt, not passes. The pre-existing offline embedding
fallback uses seeded synthetic vectors; passing parser tests using that fallback
prove structural behavior only, not semantic matching or reasoning quality.
No filename ignores remain. A renamed shadowed duplicate class is historical
source, not certifying coverage. Default discovery also excludes `historical_verification`
and the separate frontend/node_modules trees; these are not backend release tests.

Owner/tracking: [completion issue #35 Phase 2](https://github.com/dhar174/advanced_prompting_system/issues/35).
Remove each quarantine only after its provider/model dependencies are scripted and
the original semantic assertion passes against production within a bounded run.
Do not use a live provider or weaken assertions to clear this inventory.

| Test or helper | Disposition |
|---|---|
| `test_advanced_prompting.py::TestParseResponse::test_none_response_handling` | Restored to normal collection; passes the unquarantined audit |
| `test_advanced_prompting.py::TestParseResponse::test_non_string_response_handling` | Restored to normal collection; passes the unquarantined audit |
| `test_advanced_prompting.py::TestParseResponse::test_reflection_missing_reward` | parser reaches embedding/judging providers without scripted fixtures |
| `test_advanced_prompting.py::TestParseResponse::test_reflection_with_special_characters` | Restored to normal collection; passes the unquarantined audit |
| `test_advanced_prompting.py::TestParseResponse::test_non_numeric_count_and_reward` | parser reaches embedding/judging providers without scripted fixtures |
| `test_advanced_prompting.py::TestParseResponse::test_steps_and_reflections_objs_provided` | parser reaches embedding/judging providers without scripted fixtures |
| `test_advanced_prompting.py::TestParseResponse::test_overlapping_steps_and_reflections` | Existing reflection wins over expected updated reflection; #35 Phase 2 revision-state repair required |
| `test_advanced_prompting.py::TestParseResponse::test_unexpected_tags_in_response` | Restored to normal collection; passes the unquarantined audit |
| `test_advanced_prompting.py::TestParseResponse::test_multiple_answers_and_rewards` | parser reaches embedding/judging providers without scripted fixtures |
| `test_advanced_prompting.py::TestParseResponse::test_missing_count_tag` | Restored to normal collection; passes the unquarantined audit |
| `test_advanced_prompting.py::TestParseResponse::test_missing_step_tag` | Restored to normal collection; passes the unquarantined audit |
| `test_advanced_prompting.py::TestParseResponse::test_empty_step_and_reflection_tags` | empty parsed step reaches embedding input validation |
| `test_advanced_prompting.py::TestParseResponse::test_basic_response_parsing` | Restored to normal collection; passes the unquarantined audit |
| `test_advanced_prompting.py::TestParseResponse::test_multiple_steps_parsing` | Restored to normal collection; passes the unquarantined audit |
| `test_advanced_prompting.py::TestParseResponse::test_missing_reflection_handling` | parser reaches embedding/judging providers without scripted fixtures |
| `test_advanced_prompting.py::TestParseResponse::test_budget_counting` | parser reaches embedding/judging providers without scripted fixtures |
| `test_advanced_prompting.py::TestParseResponse::test_interaction_merging` | Restored to normal collection; passes the unquarantined audit |
| `test_advanced_prompting.py::TestParseResponse::test_reward_score_validation` | parser reaches embedding/judging providers without scripted fixtures |
| `test_advanced_prompting.py::TestParseResponse::test_empty_response_handling` | Restored to normal collection; passes the unquarantined audit |
| `test_advanced_prompting.py::TestParseResponse::test_malformed_tags_handling` | parser reaches embedding/judging providers without scripted fixtures |
| `test_advanced_prompting.py::TestParseResponse::test_initial_budget_handling` | parser reaches embedding/judging providers without scripted fixtures |
| `test_advanced_prompting.py::TestParseResponse::test_existing_steps_integration` | parser reaches embedding/judging providers without scripted fixtures |
| `test_advanced_prompting.py::TestParseResponse::test_reflection_objects_integration` | parser reaches embedding/judging providers without scripted fixtures |
| `test_advanced_prompting.py::TestParseResponse::test_duplicate_step_handling` | Restored to normal collection; passes the unquarantined audit |
| `test_advanced_prompting.py::TestParseResponse::test_reward_range_validation` | parser reaches embedding/judging providers without scripted fixtures |
| `test_advanced_prompting.py::TestAdvancedPromptEngineer::test_count_tokens` | tokenizer asset is absent; token counting tries a download |
| `test_advanced_prompting.py::TestAdvancedPromptEngineer::test_generate_initial_prompt` | legacy test expects the opposite prompt tuple order; settle caller contract |
| `test_advanced_prompting.py::TestAdvancedPromptEngineer::test_judge_step` | advanced workflow has no scripted provider fixture or bounded-call assertion |
| `test_advanced_prompting.py::TestAdvancedPromptEngineer::test_component_decision` | advanced workflow has no scripted provider fixture or bounded-call assertion |
| `test_advanced_prompting.py::TestAdvancedPromptEngineer::test_judge_step_completion` | advanced workflow has no scripted provider fixture or bounded-call assertion |
| `test_advanced_prompting.py::TestAdvancedPromptEngineer::test_finalize_step_output` | advanced workflow has no scripted provider fixture or bounded-call assertion |
| `test_advanced_prompting.py::TestAdvancedPromptEngineer::test_automatic_chain_of_thought` | Restored to normal collection; passes the unquarantined audit |
| `test_advanced_prompting.py::TestAdvancedPromptEngineer::test_least_to_most` | Restored to normal collection; passes the unquarantined audit |
| `test_advanced_prompting.py::TestAdvancedPromptEngineer::test_progressive_hint` | Restored to normal collection; passes the unquarantined audit |
| `test_advanced_prompting.py::TestAdvancedPromptEngineer::test_assess_complexity` | offline complexity fallback returns float instead of score/Plan tuple |
| `test_advanced_prompting.py::TestAdvancedPromptEngineer::test_adjust_step_budget` | Restored to normal collection; passes the unquarantined audit |
| `test_advanced_prompting.py::TestAdvancedPromptEngineer::test_dynamic_confidence_exploration` | advanced workflow has no scripted provider fixture or bounded-call assertion |
| `test_advanced_prompting.py::TestAdvancedPromptEngineer::test_merge_interactions` | production merge loses one expected step |
| `test_advanced_prompting.py::TestAdvancedPromptEngineer::test_choose_best_response` | advanced workflow has no scripted provider fixture or bounded-call assertion |
| `test_advanced_prompting.py::HistoricalFinalizePlanStepOutputExample::test_finalize_planstep_output_success` | Previously shadowed duplicate; retained as non-certifying example, remove once duplicate expectations reconciled under #35 Phase 2 |
| `test_advanced_prompting.py::HistoricalFinalizePlanStepOutputExample::test_finalize_planstep_output_empty_response` | Previously shadowed duplicate; retained as non-certifying example, remove once duplicate expectations reconciled under #35 Phase 2 |
| `test_advanced_prompting.py::HistoricalFinalizePlanStepOutputExample::test_finalize_planstep_output_exception_handling` | Previously shadowed duplicate; retained as non-certifying example, remove once duplicate expectations reconciled under #35 Phase 2 |
| `test_advanced_prompting.py::TestAdvancedPromptingAdapterMethods::test_determine_output_type_from_content` | Restored to normal collection; passes the unquarantined audit |
| `test_advanced_prompting.py::TestAdvancedPromptingAdapterMethods::test_select_personalities_for_planstep` | Restored to normal collection; passes the unquarantined audit |
| `test_advanced_prompting.py::TestAdvancedPromptingAdapterMethods::test_should_use_multi_agent_reasoning` | Restored to normal collection; passes the unquarantined audit |
| `test_advanced_prompting.py::TestAdvancedPromptingAdapterMethods::test_convert_conversation_output_to_final_step_output_file_exists` | Restored to normal collection; passes the unquarantined audit |
| `test_advanced_prompting.py::TestAdvancedPromptingAdapterMethods::test_convert_conversation_output_to_final_step_output_file_not_exists` | Restored to normal collection; passes the unquarantined audit |
| `test_advanced_prompting.py::TestAdvancedPromptingAdapterMethods::test_invoke_collaborative_reasoning` | Restored to normal collection; passes the unquarantined audit |
| `test_advanced_prompting.py::TestFinalizePlanStepOutput::test_finalize_planstep_output_success` | Restored to normal collection; passes the unquarantined audit |
| `test_advanced_prompting.py::TestFinalizePlanStepOutput::test_finalize_planstep_output_empty_response` | Restored to normal collection; passes the unquarantined audit |
| `test_advanced_prompting.py::TestFinalizePlanStepOutput::test_finalize_planstep_output_exception_handling` | Restored to normal collection; passes the unquarantined audit |
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
