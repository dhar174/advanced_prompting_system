---
trigger: always_on
description: "Non-negotiable architectural invariants, lazy SDK loading, offline runtime guards, voting semantics, ConversationMemory contracts, and GraphQL/React parity."
---

# Code Architecture & Invariants Rules

This document defines the non-negotiable architectural invariants and code conventions for `advanced_prompting_system`. All agents and contributors must strictly adhere to these rules.

---

## 1. Lazy SDK & Zero Import-Time Side Effects

1. **Lazy OpenAI Client (`provider_client.py`)**:
   - Importing `advanced_prompting`, `conversation_manager`, `complexity_measures`, `app`, or `schema` must NEVER instantiate `openai.OpenAI()` or require `OPENAI_API_KEY` at import time.
   - Use `provider_client.LazyClient` so SDK resources resolve only when an API method (`chat.completions.create`, `beta.chat.completions.parse`, `embeddings.create`) is actually invoked.
2. **Lazy NLP & Optional Model Loading (`complexity_measures.py`)**:
   - Never call `spacy.load("en_core_web_sm")`, `nltk.download(...)`, or initialize HuggingFace transformer pipelines at module import time.
   - All 5 direct spaCy consumers (`is_real_words`, `is_complex_graph`, `is_complex_recursive`, `is_complex_ast`, `is_complex_llm`) must resolve `get_nlp_spacy()` lazily and return safe defaults (`"Keep original text"` / original text for `is_real_words`, `0.0` for complexity scorers) without calling any LLM provider when the model is absent.
   - `is_real_words` must catch `LookupError` when optional NLTK `punkt`, `wordnet`, or `cmudict` corpora are unavailable.
3. **Offline Token Counting & Embeddings**:
   - `AdvancedPromptEngineer.count_tokens` must catch `NetworkForbidden` / `BaseException` / `OSError` when `tiktoken` cannot download encoding files offline and fall back to deterministic heuristic counting.
   - `complexity_measures.get_embedding` must raise an explicit `RuntimeError` when `OFFLINE_MODE=0` and `client is None`, while returning deterministic seeded synthetic vectors when `OFFLINE_MODE=1`.

---

## 2. Offline Socket/DNS Guard & Test Isolation (`conftest.py` & `tools/`)

1. **Mandatory Guarded Test Wrappers**:
   - `conftest.py` enforces `sys._aps_network_guard = True` (set by `tools/offline/sitecustomize.py` via `./tools/run-offline.ps1`). Bare `pytest` intentionally fails with `UsageError`.
   - Never weaken `forbid_unconfigured_provider` or `isolated_runtime_directory` (`monkeypatch.chdir(tmp_path)`) in `conftest.py`.
2. **`NetworkForbidden(BaseException)` Awareness**:
   - External socket, DNS, and UDP calls raise `NetworkForbidden`, which inherits from `BaseException` (not `Exception`) so broad `except Exception:` handlers cannot silently swallow accidental network attempts during tests.

---

## 3. Multi-Agent Voting & Consensus Invariants (`conversation_manager.py`)

1. **`BinaryVote` / `ConfidenceVote` / `ErrorResult` Truthiness Rule**:
   - `cast_binary_vote()` returns `Union[BinaryVote, ErrorResult]`; `cast_confidence_vote()` returns `Union[ConfidenceVote, ErrorResult]`.
   - Every Pydantic `BaseModel` instance—including `BinaryVote(vote=False)` and `ErrorResult`—is truthy (`bool(obj) is True`)!
   - NEVER write `if vote:`. Always explicitly check `isinstance(vote, BinaryVote) and vote.vote is True` (or `vote.vote is False`) and separate `isinstance(vote, ErrorResult)`.
2. **Consensus Edge-Case Routing**:
   - `define_problem`: Partial-majority approval returns the selected definition; all-error votes raise an explicit exception.
   - Continuation voting: Count only valid `CONTINUE` / `END` votes; all-error continuation votes enter bounded fallback (`round == 0` final decision).
   - Low participation: Below-threshold valid votes trigger explicit finalization.
   - Final decision voting: Majority-NO, weak-YES, tie votes, and all-error final votes must route through `MediatorReviseOrDecide` and generate from the revised decision.

---

## 4. Cross-Module Adapter & Memory Contracts

1. **`PlanStep` Collaborative Adapter Seam**:
   - `AdvancedPromptEngineer.main()` must pass explicit keyword arguments, the actual `PlanStep` instance, and an integer collaborative budget into `invoke_collaborative_reasoning` / `collaborative_reasoning_main`.
2. **XML Response Parsing & Consolidation (`advanced_prompting.py`)**:
   - `parse_response`, `consolidate_steps`, and `merge_interactions` must handle `None`/empty/non-string responses, malformed/unclosed tags, non-numeric `<count>`/`<reward>` values, empty descriptions before embedding, deduplicated reflection combiners, and insertion-order preservation.
3. **`ConversationMemory` & Message Normalization (`conversation_manager.py`)**:
   - `ArgumentItem`, `DecisionItem`, and `ToDoItem` must support both attribute access and dict-style access (`__getitem__`, `.get()`).
   - `ConversationMemory.from_dict()` must deserialize legacy string/list entries via `add_argument`, `add_to_do_item`, and `add_decision` so `from_dict(to_dict())` is lossless.
   - `normalize_message()` must raise `ValueError` when given an `ErrorResult` object.
   - Local custom exception `ConversationValidationError` must never shadow `pydantic.ValidationError`.
4. **Protected File Output Generation (`output_generator.py`)**:
   - All 7 file-writing functions (`generate_json_output`, `generate_pdf_output`, `generate_text_file_output`, `generate_html_output`, `generate_python_script`, `generate_code_snippet`, `generate_csv_output`) must wrap file writes in `try ... except (IOError, OSError, PermissionError)` and re-raise.

---

## 5. GraphQL & React Frontend Contract Parity (`app.py`, `schema.py`, `collaborative-assistant-frontend/`)

1. **3-Tuple & Schema Alignment**:
   - `conversation_manager.run_conversation` returns `(conversation_history, questions_asked, final_output)`.
   - `schema.Mutation.run_conversation` maps `questions_asked` via `q.get("question") or q.get("reply") or ""` and returns `RunConversationResponse` matching the React client's `runConversation` mutation shape (`conversation`, `questions`, `finalOutput`).
2. **Production vs. Historical Separation**:
   - Do not import from or modify standalone historical scripts (`test_b.py`, `test_c.py`, `output_generator_backup.py`, `output_generator_fixed.py`, `historical_verification/`, `finance_manager_final_a.py`, `personal_finance_*.py`, `meeting_plan.py`) when implementing production features.
