# feedback_manager.py

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
import json
import os
import tempfile
import threading
from typing import Any, Dict, List, Optional

LOG_DIR = "logs"
_DEFAULT_LOG_FILE = os.path.join(LOG_DIR, "conversation_logs.json")
LOG_FILE = _DEFAULT_LOG_FILE

MAX_CONVERSATION_ITEMS = 500
MAX_FEEDBACK_ITEMS = 100
MAX_QUESTION_ITEMS = 100
MAX_TEXT_CHARS = 50_000
MAX_COMMENT_CHARS = 5_000
MAX_LABEL_CHARS = 200

_LOG_LOCK = threading.Lock()


def _resolve_log_file() -> str:
    """Resolve the feedback log file path lazily at write time."""
    env_override = os.environ.get("APS_LOG_FILE", "").strip()
    if env_override:
        return env_override
    if LOG_FILE != _DEFAULT_LOG_FILE:
        return LOG_FILE
    return os.path.join(LOG_DIR, "conversation_logs.json")


def _to_plain(item: Any) -> Any:
    """Convert Strawberry types, Pydantic models, or nested containers to JSON-safe dicts."""
    if item is None or isinstance(item, (str, int, float, bool)):
        return item
    if isinstance(item, dict):
        return {str(k): _to_plain(v) for k, v in item.items()}
    if isinstance(item, (list, tuple)):
        return [_to_plain(v) for v in item]
    if hasattr(item, "model_dump") and callable(item.model_dump):
        return _to_plain(item.model_dump())
    if dataclasses.is_dataclass(item):
        return _to_plain(dataclasses.asdict(item))
    if hasattr(item, "__dict__"):
        return _to_plain(vars(item))
    return str(item)


def _validate_and_normalize(
    conversation_history: Any,
    feedback: Any,
    questions: Any = None,
) -> Dict[str, Any]:
    """Validate feedback payloads and normalize entries to plain dictionaries."""
    if not isinstance(conversation_history, (list, tuple)) or len(conversation_history) == 0:
        raise ValueError("Conversation history must be a non-empty list.")
    if len(conversation_history) > MAX_CONVERSATION_ITEMS:
        raise ValueError(f"Conversation history exceeds maximum of {MAX_CONVERSATION_ITEMS} entries.")

    if not isinstance(feedback, (list, tuple)) or len(feedback) == 0:
        raise ValueError("Feedback must contain at least one rating entry.")
    if len(feedback) > MAX_FEEDBACK_ITEMS:
        raise ValueError(f"Feedback exceeds maximum of {MAX_FEEDBACK_ITEMS} entries.")

    norm_conversation: List[Dict[str, Any]] = []
    for entry in conversation_history:
        plain = _to_plain(entry)
        if not isinstance(plain, dict):
            raise ValueError("Each conversation entry must be an object.")
        content = str(plain.get("content", "") or "")
        if len(content) > MAX_TEXT_CHARS:
            raise ValueError(f"Conversation entry content exceeds {MAX_TEXT_CHARS} characters.")
        role_str = str(plain.get("role", "user") or "user")[:MAX_LABEL_CHARS]
        name_val = None if plain.get("name") is None else str(plain.get("name"))[:MAX_LABEL_CHARS]
        norm_conversation.append(
            {
                "role": role_str,
                "name": name_val,
                "content": content,
            }
        )

    norm_feedback: List[Dict[str, Any]] = []
    for item in feedback:
        plain = _to_plain(item)
        if not isinstance(plain, dict):
            raise ValueError("Each feedback item must be an object.")
        assistant = str(plain.get("assistant", "") or "").strip()
        if not assistant:
            raise ValueError("Feedback entry assistant name must not be empty.")
        if len(assistant) > MAX_LABEL_CHARS:
            raise ValueError(f"Feedback assistant name exceeds {MAX_LABEL_CHARS} characters.")
        rating = plain.get("rating")
        if isinstance(rating, bool) or not isinstance(rating, int) or rating < 1 or rating > 5:
            raise ValueError("Feedback rating must be an integer between 1 and 5.")
        comments = plain.get("comments")
        if comments is not None:
            comments = str(comments)
            if len(comments) > MAX_COMMENT_CHARS:
                raise ValueError(f"Feedback comments exceed maximum length of {MAX_COMMENT_CHARS} characters.")
        norm_feedback.append(
            {
                "assistant": assistant,
                "rating": rating,
                "comments": comments,
            }
        )

    norm_questions: Optional[List[Dict[str, Any]]] = None
    if questions is not None:
        if not isinstance(questions, (list, tuple)):
            raise ValueError("Questions must be a list when provided.")
        if len(questions) > MAX_QUESTION_ITEMS:
            raise ValueError(f"Questions exceed maximum of {MAX_QUESTION_ITEMS} entries.")
        norm_questions = []
        for q in questions:
            plain = _to_plain(q)
            if not isinstance(plain, dict):
                raise ValueError("Each question entry must be an object.")
            q_assistant = str(plain.get("assistant", "") or "").strip()[:MAX_LABEL_CHARS]
            q_text = str(plain.get("question", "") or "")
            if len(q_text) > MAX_COMMENT_CHARS:
                raise ValueError(f"Question text exceeds maximum length of {MAX_COMMENT_CHARS} characters.")
            norm_questions.append(
                {
                    "assistant": q_assistant,
                    "question": q_text,
                }
            )

    return {
        "conversation": norm_conversation,
        "feedback": norm_feedback,
        "questions": norm_questions,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


def log_conversation(conversation_history, feedback, questions=None):
    """Validate and atomically append feedback records to disk."""
    log_entry = _validate_and_normalize(conversation_history, feedback, questions)
    log_file = _resolve_log_file()
    log_dir = os.path.dirname(os.path.abspath(log_file))

    with _LOG_LOCK:
        os.makedirs(log_dir, exist_ok=True)
        data: List[Dict[str, Any]] = []
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            if raw:
                try:
                    loaded = json.loads(raw)
                    if isinstance(loaded, list):
                        data = loaded
                except json.JSONDecodeError:
                    data = []
        data.append(log_entry)

        fd, tmp_path = tempfile.mkstemp(prefix=".feedback_", suffix=".tmp", dir=log_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                json.dump(data, tmp_file, indent=4, ensure_ascii=False)
            os.replace(tmp_path, log_file)
        except BaseException:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise
