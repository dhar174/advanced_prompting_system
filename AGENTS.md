# AGENTS.md

## Purpose

This file is the authoritative, tool-neutral repository operating contract and agent-focused companion to [`README.md`](README.md).
Read and follow this document before performing any code, test, or documentation work in `advanced_prompting_system`.

In this repository, "agent" refers to two distinct families that must be kept clearly separated:

1. **Runtime Collaborative & Prompting Personas**: The AI reasoning steps in [`advanced_prompting.py`](advanced_prompting.py) and the 26 debate assistant personalities defined in [`assistant_personalities.py`](assistant_personalities.py) and orchestrated by [`conversation_manager.py`](conversation_manager.py).
2. **Contributor-Facing Engineering Agents**: The Google Antigravity 2.0+ custom coordinator and specialist subagents under [`.agents/agents/`](.agents/agents/), behavioral rules under [`.agents/rules/`](.agents/rules/) (and [`.agent/rules/`](.agent/rules/)), and workspace skills under [`.agents/skills/`](.agents/skills/) that help maintainers plan, implement, review, and validate changes.

---

## Read First

1. [`README.md`](README.md) — Project overview and quick start.
2. [`docs/pr34-offline-qualification.md`](docs/pr34-offline-qualification.md) — Offline qualification boundaries, repaired PR #34 findings, execution inventory, and Issue #35/#36 roadmap.
3. [`docs/offline-test-inventory.md`](docs/offline-test-inventory.md) — Complete per-method backend test inventory (48/48 `test_advanced_prompting.py` tests unquarantined and passing offline).
4. [`GEMINI.md`](GEMINI.md) — Antigravity workspace bootstrap and 5-stage orchestration lifecycle.
5. [`.agents/doc/antigravity-agent-team.md`](.agents/doc/antigravity-agent-team.md) — Expected subagent team delegation workflow and one-writer operating model.
6. [`memory-bank/activeContext.md`](memory-bank/activeContext.md) and [`memory-bank/progress.md`](memory-bank/progress.md) — Active session context and milestone tracking.

---

## Repository Map & Core Surfaces

### Production Python Backend (Repository Root)
- [`advanced_prompting.py`](advanced_prompting.py) — Main reasoning engine (`AdvancedPromptEngineer`, `PromptEngineeringConfig`, `Task`, `Step`, `Reflection`, `Interaction`, `FinalStepOutput`, `FinalPlanStepOutput`, `CompletionStatus`). Orchestrates retrieval, complexity assessment, step budgeting, multi-strategy reasoning (`self_consistency`, `automatic_chain_of_thought`, `least_to_most`, `progressive_hint`, `dynamic_confidence_exploration`), XML tag parsing (`parse_response`, `consolidate_steps`, `merge_interactions`), reward judging (`judge_step`, `judge_step_completion`, `choose_best_response`), and collaborative adapters (`invoke_collaborative_reasoning`, `collaborative_reasoning_main`, `finalize_planstep_output`).
- [`conversation_manager.py`](conversation_manager.py) — Multi-agent round deliberation, consensus voting (`define_problem`, `cast_binary_vote`, `cast_confidence_vote`, `BinaryVote`, `ConfidenceVote`, `ErrorResult`, `MediatorReviseOrDecide`), structured `ConversationMemory` (`ArgumentItem`, `DecisionItem`, `ToDoItem`), message normalization (`normalize_message`, `validate_message_structure`), and output finalization (`finalize_output`, `analyze_final_output`).
- [`complexity_measures.py`](complexity_measures.py) — Task complexity analysis and step planning (`Plan`, `PlanStep`, `Subtask`), linguistic/structural complexity scorers (`is_real_words`, `is_complex_graph`, `is_complex_recursive`, `is_complex_ast`, `is_complex_llm`, Flesch reading ease), lazy NLP loader (`get_nlp_spacy()`), sklearn complexity classifier (`ml_complexity_model.pkl`), `cosine_similarity_custom` (dense & `scipy.sparse.csr_matrix`), and `get_embedding` (offline seeded-vector fallback vs. explicit live configuration check).
- [`provider_client.py`](provider_client.py) — `LazyClient` and `get_client()` ensuring importing application modules never instantiates `openai.OpenAI()` or requires credentials at import time.
- [`assistant_personalities.py`](assistant_personalities.py) — 26 distinct debate personalities (`Analytical Thinker`, `Creative Innovator`, `Skeptical Critic`, `Mediator`, `Strategist`, `Coding Guru`, etc.) and `assistant_instructions`.
- [`output_generator.py`](output_generator.py) & [`function_definitions.py`](function_definitions.py) — Pydantic tool parameter schemas (`GenerateJSONOutput`, `GeneratePDFOutput`, etc.) and 7 protected file output generators (`generate_json_output`, `generate_pdf_output`, `generate_text_file_output`, `generate_html_output`, `generate_python_script`, `generate_code_snippet`, `generate_csv_output`) plus `generate_simple_concise_answer_message`.
- [`app.py`](app.py) & [`schema.py`](schema.py) — Flask web server and Strawberry GraphQL schema (`/graphql`) exposing `Query.hello`, `Mutation.run_conversation`, and `Mutation.submit_feedback`, plus static SPA serving from `collaborative-assistant-frontend/dist`.
- [`feedback_manager.py`](feedback_manager.py) — Persists conversation feedback entries to `logs/conversation_logs.json`.

### React / TypeScript Frontend (`collaborative-assistant-frontend/`)
- [`collaborative-assistant-frontend/src/`](collaborative-assistant-frontend/src) — React 19 + TypeScript + Vite 6 + Tailwind CSS + Apollo Client application (`ConversationPage`, `AssistantConfigPanel`, `ChatInput`, `ConversationDisplay`, `FeedbackDisplay`, `MessageBubble`, `useRunConversation`, `useSubmitFeedback`, and 10 Vitest test suites with 111 passing tests).

### Offline Qualification Harness & Test Suites
- [`conftest.py`](conftest.py) — Enforces `sys._aps_network_guard`, `forbid_unconfigured_provider` (`provider_client.get_client` fails unscripted calls), `isolated_runtime_directory` (`tmp_path` working directory isolation), and the shared `conversation_case` fixture.
- [`tools/run-offline.ps1`](tools/run-offline.ps1) & [`tools/run-offline.sh`](tools/run-offline.sh) — Strips provider API keys/tokens, isolates cache/temp dirs under `.qualification/`, and preloads [`tools/offline/sitecustomize.py`](tools/offline/sitecustomize.py) to block external DNS/TCP/UDP sockets (`NetworkForbidden(BaseException)`).
- [`tools/run-frontend-offline.ps1`](tools/run-frontend-offline.ps1) & [`tools/run-frontend-offline.sh`](tools/run-frontend-offline.sh) — Preloads [`tools/offline/node-network-guard.cjs`](tools/offline/node-network-guard.cjs) and executes `guard`, `test`, `lint`, or `build`.
- Authoritative Backend Test Files:
  - [`test_advanced_prompting.py`](test_advanced_prompting.py) (48 tests: `TestParseResponse`, `TestAdvancedPromptEngineer`, `TestAdvancedPromptingAdapterMethods`, `TestFinalizePlanStepOutput`)
  - [`test_memory_management.py`](test_memory_management.py)
  - [`test_pr34_regressions.py`](test_pr34_regressions.py)
  - [`test_voting_semantics.py`](test_voting_semantics.py)
  - [`test_message_handling.py`](test_message_handling.py)
  - [`test_import_integrity.py`](test_import_integrity.py)
  - [`test_offline_runtime.py`](test_offline_runtime.py)
  - [`test_retrieval.py`](test_retrieval.py)
  - [`test_graphql_contract.py`](test_graphql_contract.py)
  - [`test_smoke.py`](test_smoke.py)
  - [`test_finalize_step_output.py`](test_finalize_step_output.py)

### Historical / Non-Certifying Files (Read-Only Reference)
- `test_b.py`, `test_c.py` — Alternate historical parser helpers with no collected tests.
- `output_generator_backup.py`, `output_generator_fixed.py`, `historical_verification/`, `validation_test.py`, `final_verification_test.py`, `finance_manager_final_a.py`, `personal_finance_*.py`, `meeting_plan.py`, `project_visualizations_plan.py`, `mediator_summary_next_steps.py` — Standalone historical examples and diagnostics; not supported application modes. Do not wire production imports to these files.

---

## Non-Negotiable Architectural & Safety Invariants

1. **One-Production-Writer Discipline**:
   - Work through `aps-repo-coordinator` as the single production writer. Specialist subagents investigate, design tests, audit contracts, and review diffs without making uncoordinated edits to shared files.
2. **Lazy SDK & Zero Import-Time Side Effects**:
   - Importing `advanced_prompting`, `conversation_manager`, `complexity_measures`, `app`, or `schema` must NEVER construct `openai.OpenAI()`, require `OPENAI_API_KEY`, download NLTK/HuggingFace/tiktoken assets, or eagerly load `spacy.load("en_core_web_sm")`.
   - All LLM calls must route through `provider_client.LazyClient` proxies or explicitly injected test fixtures.
   - All 5 direct spaCy consumers in `complexity_measures.py` (`is_real_words`, `is_complex_graph`, `is_complex_recursive`, `is_complex_ast`, `is_complex_llm`) must resolve `get_nlp_spacy()` lazily and return safe defaults (`0.0` or original text) without calling LLM providers when the model is absent.
3. **Offline Qualification Guard**:
   - Never weaken `conftest.py`, `tools/offline/sitecustomize.py`, or `tools/offline/node-network-guard.cjs`.
   - Bare `pytest` intentionally fails with `UsageError` when `sys._aps_network_guard` is not active. Always run backend tests via `./tools/run-offline.ps1` (Windows) or `./tools/run-offline.sh` (Linux/macOS).
4. **Resilient Voting & Consensus Semantics**:
   - `cast_binary_vote()` returns `Union[BinaryVote, ErrorResult]`; `cast_confidence_vote()` returns `Union[ConfidenceVote, ErrorResult]`.
   - Because any Pydantic `BaseModel` instance (including `BinaryVote(vote=False)` and `ErrorResult`) is truthy in Python, NEVER check `if vote:`. Always check `isinstance(vote, BinaryVote) and vote.vote is True`.
   - Continuation votes must count only explicit `CONTINUE` / `END` votes; all-error continuation votes must enter bounded fallback (`round == 0` final decision).
   - Majority-NO, weak-YES, tie votes, and low participation must route through `MediatorReviseOrDecide` and generate from the revised decision.
5. **Cross-Module Adapter & Schema Contracts**:
   - `AdvancedPromptEngineer.main(task)` must pass explicit keyword arguments, the actual `PlanStep` object, and an integer collaborative budget into `invoke_collaborative_reasoning` / `collaborative_reasoning_main`.
   - `conversation_manager.run_conversation(conversation, assistant_personalities, lead_personality, num_rounds)` returns the 3-tuple `(conversation_history, questions_asked, final_output)`.
   - `normalize_message()` in `conversation_manager.py` must reject `ErrorResult` objects with `ValueError` so failed LLM calls never enter conversation history as assistant messages.
   - `ConversationMemory` must preserve both typed model access (`ArgumentItem`, `DecisionItem`, `ToDoItem`) and legacy dict/string/list round-trip serialization (`to_dict()` / `from_dict()`).
6. **Artifact & Repository Hygiene**:
   - Never commit `.qualification/`, generated `conversation_history.json` from test runs, secrets, API keys, or node_modules changes unless explicitly requested.
   - All 7 file output generators in `output_generator.py` must retain `try/except (IOError, OSError, PermissionError)` blocks that log clear error messages and re-raise.
   - Never merge, force-push, or perform destructive git cleanup without explicit user approval.

---

## Authoritative Commands

### Backend Offline Test Suite (Windows PowerShell)
```powershell
# Collect-only verification
./tools/run-offline.ps1 -PytestArgs '--collect-only','-q'

# Full offline backend test suite (148 passed / 0 failed)
./tools/run-offline.ps1 -PytestArgs '-q'

# Focused test execution (always via run-offline.ps1)
./tools/run-offline.ps1 -PytestArgs 'test_advanced_prompting.py','-v'
./tools/run-offline.ps1 -PytestArgs 'test_voting_semantics.py','test_pr34_regressions.py','-v'
./tools/run-offline.ps1 -PytestArgs 'test_memory_management.py','test_message_handling.py','-v'
./tools/run-offline.ps1 -PytestArgs 'test_import_integrity.py','test_offline_runtime.py','test_graphql_contract.py','test_smoke.py','-v'
```

### Frontend Offline Verification (Windows PowerShell)
```powershell
# Verify Node outbound socket/DNS denial guard
./tools/run-frontend-offline.ps1 -Action guard

# Run Vitest unit & integration test suite (111 passed across 10 files)
./tools/run-frontend-offline.ps1 -Action test

# Run TypeScript typecheck & Vite production bundle build
./tools/run-frontend-offline.ps1 -Action build

# Run ESLint check
./tools/run-frontend-offline.ps1 -Action lint
```

### Whitespace & Diff Hygiene Check
```powershell
git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --check
```

---

## Source-of-Truth and Drift Rules

- This file [`AGENTS.md`](AGENTS.md) is the authoritative, tool-neutral repository operating contract.
- [`GEMINI.md`](GEMINI.md) is the Antigravity/Gemini workspace bootstrap. It points agents back to this contract and documents Antigravity-specific discovery surfaces and the 5-stage subagent workflow.
- [`.agents/rules/*.md`](.agents/rules/) (and mirrored [`.agent/rules/*.md`](.agent/rules/)) operationalize this contract for Antigravity.
- [`.agents/agents/<name>/agent.md`](.agents/agents/) defines workspace custom agents and subagents.
- [`.agents/skills/<skill>/SKILL.md`](.agents/skills/) defines reusable workspace skills.
- [`.agents/doc/antigravity-agent-team.md`](.agents/doc/antigravity-agent-team.md) describes the expected subagent delegation workflow.
- [`memory-bank/`](memory-bank/) and `mem0ry4ai` (`project:advanced_prompting_system`) provide cross-session context, which remains subordinate to current repository code, tests, and this contract.
