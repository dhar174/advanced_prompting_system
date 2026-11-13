# System Patterns & Architectural Invariants (`advanced_prompting_system`)

## 1. High-Level System Architecture

```text
User / GraphQL Client (React 19 + Apollo)
   │
   ├──> app.py / schema.py (Flask + Strawberry GraphQL /graphql)
   │       ├── Mutation.runConversation ──> conversation_manager.run_conversation(...)
   │       └── Mutation.submitFeedback  ──> feedback_manager.log_conversation(...)
   │
   └──> advanced_prompting.py (AdvancedPromptEngineer.main)
           ├── complexity_measures.py (assess_complexity -> Plan / PlanStep / Subtask)
           ├── Multi-Strategy Reasoning (CoT, Least-to-Most, Progressive-Hint, Self-Consistency)
           ├── Collaborative Adapter (invoke_collaborative_reasoning -> run_conversation)
           └── Output Generators (output_generator.py / function_definitions.py)
```

## 2. Non-Negotiable Codebase Patterns

### Lazy SDK Resolution (`provider_client.py`)
- `LazyClient` defers `openai.OpenAI()` construction until a method call (`chat.completions.create`, `beta.chat.completions.parse`, `embeddings.create`) occurs.
- Module imports must never require `OPENAI_API_KEY` or open network sockets.

### Process-Level Socket/DNS Denial (`tools/offline/`)
- `tools/offline/sitecustomize.py` sets `sys._aps_network_guard = True` and hooks Python `socket` / DNS functions to raise `NetworkForbidden(BaseException)` on non-loopback connections.
- `conftest.py` fails collection if `sys._aps_network_guard` is missing, patches `provider_client.get_client` to fail on unscripted calls (`forbid_unconfigured_provider`), and isolates working directories via `monkeypatch.chdir(tmp_path)`.

### Voting & Consensus Truthiness Pattern (`conversation_manager.py`)
- `cast_binary_vote` -> `Union[BinaryVote, ErrorResult]`; `cast_confidence_vote` -> `Union[ConfidenceVote, ErrorResult]`.
- Because Pydantic `BaseModel` instances are always truthy (`bool(BinaryVote(vote=False)) is True`), code must explicitly test `isinstance(v, BinaryVote) and v.vote is True`.
- Non-unanimous final votes (majority NO, weak YES, tie, low participation, all-error) route through `MediatorReviseOrDecide`.

### `ConversationMemory` Dual-Access Pattern (`conversation_manager.py`)
- `ArgumentItem`, `DecisionItem`, and `ToDoItem` are Pydantic models that also implement `__getitem__` and `.get()` for backward compatibility with dict-expecting callers.
- `ConversationMemory.from_dict()` routes entries through `add_argument`, `add_to_do_item`, and `add_decision` to normalize legacy string/list formats.

## 3. Antigravity Multi-Agent Team Pattern
- **Single Production Writer**: `aps-repo-coordinator` owns all production file writes.
- **Bookend Memory Agents**: `memory-scout` runs in Stage 1 (pre-planning); `memory-steward` runs in Stage 5 (post-verification).
- **Read-Only Specialists**: `architecture-contract-scout`, `prompt-reasoning-architect`, `consensus-conversation-specialist`, `complexity-nlp-engineer`, `graphql-frontend-specialist`, and `offline-qa-gatekeeper` inspect code, run offline tests, and return surgical diff proposals.
