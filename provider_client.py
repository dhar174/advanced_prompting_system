"""Lazy SDK access and request-scoped provider policy enforcement.

Importing the application never instantiates ``openai.OpenAI()`` or requires
credentials. When an execution request enters ``provider_execution_scope()``,
all outbound calls share one validated model allowlist, monotonic deadline,
call ledger, and cancellation signal.
"""
from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, field
from functools import lru_cache
import inspect
import os
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple


class ProviderPolicyError(RuntimeError):
    """Base exception for request-scoped provider policy violations."""


class RunBudgetExceededError(ProviderPolicyError):
    """Raised when call count or deadline budget is exhausted."""


class RunTimeoutError(RunBudgetExceededError):
    """Raised when the request monotonic deadline is exceeded."""


class RunCancelledError(ProviderPolicyError):
    """Raised when an active execution cancellation callback returns True."""


class DisallowedModelError(ValueError):
    """Raised when a client or caller requests a model outside the allowlist."""


@dataclass
class ProviderExecutionContext:
    """Request-scoped provider policy, shared call ledger, and deadline state."""

    default_model: str = "gpt-4o-mini"
    allowed_models: Optional[Tuple[str, ...]] = None
    temperature: Optional[float] = None
    timeout_seconds: Optional[float] = None
    deadline_monotonic: Optional[float] = None
    max_calls: Optional[int] = None
    calls_made: int = 0
    cancel_check: Optional[Callable[[], bool]] = None
    termination_reason: Optional[str] = None
    recorded_calls: List[Dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


_ACTIVE_EXECUTION_CONTEXT: contextvars.ContextVar[Optional[ProviderExecutionContext]] = (
    contextvars.ContextVar("aps_provider_execution_context", default=None)
)
_IN_RESERVED_CALL: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "aps_in_reserved_provider_call", default=False
)


def get_active_execution_context() -> Optional[ProviderExecutionContext]:
    """Return the active request-scoped provider execution context, if any."""
    return _ACTIVE_EXECUTION_CONTEXT.get()


@contextlib.contextmanager
def provider_execution_scope(
    *,
    default_model: Optional[str] = None,
    allowed_models: Optional[Sequence[str]] = None,
    temperature: Optional[float] = None,
    timeout_seconds: Optional[float] = None,
    max_calls: Optional[int] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    deadline_monotonic: Optional[float] = None,
) -> Iterator[ProviderExecutionContext]:
    """Establish a request-scoped provider policy and shared call ledger with cleanup."""
    normalized_allowed: Optional[Tuple[str, ...]] = None
    if allowed_models is not None:
        cleaned = tuple(str(m).strip() for m in allowed_models if str(m or "").strip())
        normalized_allowed = cleaned

    resolved_default = str(default_model or "").strip() or (
        normalized_allowed[0] if normalized_allowed else "gpt-4o-mini"
    )
    if normalized_allowed and resolved_default not in normalized_allowed:
        raise DisallowedModelError(
            f"Requested model '{resolved_default}' is not permitted. "
            f"Allowed models: {', '.join(normalized_allowed)}."
        )

    computed_deadline: Optional[float] = deadline_monotonic
    if computed_deadline is None and timeout_seconds is not None:
        timeout_val = float(timeout_seconds)
        if timeout_val > 0:
            computed_deadline = time.monotonic() + timeout_val
        else:
            computed_deadline = time.monotonic()

    resolved_max_calls: Optional[int] = None
    if max_calls is not None:
        resolved_max_calls = max(int(max_calls), 0)

    ctx = ProviderExecutionContext(
        default_model=resolved_default,
        allowed_models=normalized_allowed,
        temperature=float(temperature) if temperature is not None else None,
        timeout_seconds=float(timeout_seconds) if timeout_seconds is not None else None,
        deadline_monotonic=computed_deadline,
        max_calls=resolved_max_calls,
        cancel_check=cancel_check,
    )
    token = _ACTIVE_EXECUTION_CONTEXT.set(ctx)
    try:
        yield ctx
    finally:
        _ACTIVE_EXECUTION_CONTEXT.reset(token)


def validate_model_for_context(
    model: Optional[str], ctx: Optional[ProviderExecutionContext] = None
) -> Optional[str]:
    """Validate an explicit model identifier against the active context allowlist."""
    active = ctx or _ACTIVE_EXECUTION_CONTEXT.get()
    candidate = str(model or "").strip()
    if not candidate:
        return active.default_model if active else None
    if active and active.allowed_models:
        if candidate not in active.allowed_models:
            active.termination_reason = "disallowed_model"
            raise DisallowedModelError(
                f"Requested model '{candidate}' is not permitted. "
                f"Allowed models: {', '.join(active.allowed_models)}."
            )
    return candidate


def get_effective_model(explicit: Optional[str] = None, fallback: str = "gpt-4o-mini") -> str:
    """Resolve and validate the effective model for the active request context."""
    ctx = _ACTIVE_EXECUTION_CONTEXT.get()
    explicit_clean = str(explicit or "").strip()
    if ctx is None:
        return explicit_clean or fallback
    if explicit_clean:
        validated = validate_model_for_context(explicit_clean, ctx)
        return validated or ctx.default_model or fallback
    validated_default = validate_model_for_context(ctx.default_model, ctx)
    return validated_default or fallback


def get_effective_temperature(
    explicit: Optional[float] = None, fallback: Optional[float] = None
) -> Optional[float]:
    """Resolve temperature from explicit value, active context, or fallback."""
    if explicit is not None:
        return float(explicit)
    ctx = _ACTIVE_EXECUTION_CONTEXT.get()
    if ctx is not None and ctx.temperature is not None:
        return float(ctx.temperature)
    return fallback


def get_remaining_timeout() -> Optional[float]:
    """Return remaining seconds until monotonic deadline, raising RunTimeoutError if expired."""
    ctx = _ACTIVE_EXECUTION_CONTEXT.get()
    if ctx is None or ctx.deadline_monotonic is None:
        return None
    remaining = float(ctx.deadline_monotonic) - time.monotonic()
    if remaining <= 0:
        ctx.termination_reason = "timeout"
        raise RunTimeoutError("Execution timeout exceeded during provider run.")
    return remaining


def check_provider_budget() -> None:
    """Check cancellation, deadline, and call ceiling without consuming a call."""
    ctx = _ACTIVE_EXECUTION_CONTEXT.get()
    if ctx is None:
        return
    with ctx._lock:
        if ctx.cancel_check and ctx.cancel_check():
            ctx.termination_reason = "cancelled"
            raise RunCancelledError("Execution cancelled by caller.")
        if ctx.deadline_monotonic is not None and time.monotonic() >= float(ctx.deadline_monotonic):
            ctx.termination_reason = "timeout"
            raise RunTimeoutError("Execution timeout exceeded during provider run.")
        if ctx.max_calls is not None and int(ctx.calls_made) >= int(ctx.max_calls):
            ctx.termination_reason = "budget_exhausted"
            raise RunBudgetExceededError(
                f"Provider call budget ({ctx.max_calls}) exhausted during run."
            )


def consume_provider_call(
    model: Optional[str] = None,
    *,
    call_type: str = "llm",
    validate_model: bool = True,
) -> str:
    """Validate policy, check deadline/cancel/cap, and charge one outbound call."""
    ctx = _ACTIVE_EXECUTION_CONTEXT.get()
    if ctx is None:
        return str(model or "").strip() or "gpt-4o-mini"

    with ctx._lock:
        if ctx.cancel_check and ctx.cancel_check():
            ctx.termination_reason = "cancelled"
            raise RunCancelledError("Execution cancelled by caller.")
        if ctx.deadline_monotonic is not None and time.monotonic() >= float(ctx.deadline_monotonic):
            ctx.termination_reason = "timeout"
            raise RunTimeoutError("Execution timeout exceeded during provider run.")

        effective_model = (
            validate_model_for_context(model, ctx)
            if validate_model
            else (str(model or "").strip() or ctx.default_model)
        ) or ctx.default_model

        if ctx.max_calls is not None and int(ctx.calls_made) >= int(ctx.max_calls):
            ctx.termination_reason = "budget_exhausted"
            raise RunBudgetExceededError(
                f"Provider call budget ({ctx.max_calls}) exhausted during run."
            )

        ctx.calls_made = int(ctx.calls_made) + 1
        ctx.recorded_calls.append(
            {
                "index": ctx.calls_made,
                "model": effective_model,
                "call_type": call_type,
            }
        )
        return effective_model


@contextlib.contextmanager
def reserve_call_scope(
    model: Optional[str] = None,
    *,
    call_type: str = "llm",
    validate_model: bool = True,
) -> Iterator[str]:
    """Reserve one call in the active ledger around a single outbound provider request."""
    effective_model = consume_provider_call(
        model=model, call_type=call_type, validate_model=validate_model
    )
    token = _IN_RESERVED_CALL.set(True)
    try:
        yield effective_model
        ctx = _ACTIVE_EXECUTION_CONTEXT.get()
        if ctx is not None and ctx.deadline_monotonic is not None:
            if time.monotonic() >= float(ctx.deadline_monotonic):
                ctx.termination_reason = "timeout"
                raise RunTimeoutError("Execution timeout exceeded during provider call.")
    finally:
        _IN_RESERVED_CALL.reset(token)


def get_execution_stats() -> Dict[str, Any]:
    """Return a snapshot of the active execution context ledger and policy."""
    ctx = _ACTIVE_EXECUTION_CONTEXT.get()
    if ctx is None:
        return {}
    with ctx._lock:
        remaining: Optional[float] = None
        if ctx.deadline_monotonic is not None:
            remaining = max(0.0, float(ctx.deadline_monotonic) - time.monotonic())
        return {
            "default_model": ctx.default_model,
            "allowed_models": list(ctx.allowed_models) if ctx.allowed_models is not None else None,
            "calls_made": int(ctx.calls_made),
            "max_calls": ctx.max_calls,
            "timeout_seconds": ctx.timeout_seconds,
            "remaining_seconds": remaining,
            "termination_reason": ctx.termination_reason,
            "recorded_calls": list(ctx.recorded_calls),
        }


@lru_cache(maxsize=1)
def get_client():
    import openai
    return openai.OpenAI()


def _apply_client_options(client_obj: Any, remaining_timeout: Optional[float]) -> Any:
    """Configure per-request timeout and disable implicit SDK retries when supported."""
    if hasattr(type(client_obj), "with_options"):
        opts: Dict[str, Any] = {"max_retries": 0}
        if remaining_timeout is not None:
            opts["timeout"] = remaining_timeout
        return client_obj.with_options(**opts)
    return client_obj


def get_configured_client() -> Any:
    """Resolve the SDK client configured with the active policy timeout and retry limit."""
    client_obj = get_client()
    ctx = _ACTIVE_EXECUTION_CONTEXT.get()
    if ctx is None:
        return client_obj
    remaining = get_remaining_timeout()
    return _apply_client_options(client_obj, remaining)


def has_configured_provider() -> bool:
    """Return True if credentials exist or a custom test provider fixture is active."""
    if os.getenv("OPENAI_API_KEY"):
        return True
    # Detect patched get_client (e.g., FakeProviderClient in tests or offline E2E),
    # while explicitly excluding the autouse conftest fail-closed guard.
    wrapped = getattr(get_client, "__wrapped__", None)
    if wrapped is None and not hasattr(get_client, "cache_info"):
        return getattr(get_client, "__name__", "") != "unexpected_call"
    return False


def _callable_accepts_timeout(fn: Any) -> bool:
    """Return True if a provider method explicitly accepts ``timeout`` or ``**kwargs``."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD or param.name == "timeout":
            return True
    return False


class LazyClient:
    """Resolve the SDK resource only when a method is called.

    The stable resource objects also allow tests to patch individual SDK calls
    without constructing a live client or supplying dummy credentials.
    """

    def __init__(self, path=()):
        self._path = path

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        resource = LazyClient(self._path + (name,))
        setattr(self, name, resource)
        return resource

    def __call__(self, *args, **kwargs):
        ctx = _ACTIVE_EXECUTION_CONTEXT.get()
        if ctx is None:
            resource = get_client()
            for name in self._path:
                resource = getattr(resource, name)
            return resource(*args, **kwargs)

        is_chat_call = len(self._path) > 0 and self._path[0] in ("chat", "beta")
        req_model = kwargs.get("model")

        if _IN_RESERVED_CALL.get():
            if ctx.cancel_check and ctx.cancel_check():
                ctx.termination_reason = "cancelled"
                raise RunCancelledError("Execution cancelled by caller.")
            if is_chat_call and req_model is not None:
                validate_model_for_context(req_model, ctx)
        else:
            consume_provider_call(
                model=req_model,
                call_type=".".join(self._path) or "llm",
                validate_model=is_chat_call,
            )

        remaining = get_remaining_timeout()
        client_obj = _apply_client_options(get_client(), remaining)
        resource = client_obj
        for name in self._path:
            resource = getattr(resource, name)

        if remaining is not None and _callable_accepts_timeout(resource):
            if kwargs.get("timeout") is not None:
                kwargs["timeout"] = min(float(kwargs["timeout"]), remaining)
            else:
                kwargs["timeout"] = remaining

        result = resource(*args, **kwargs)
        if ctx.deadline_monotonic is not None and time.monotonic() >= float(ctx.deadline_monotonic):
            ctx.termination_reason = "timeout"
            raise RunTimeoutError("Execution timeout exceeded during provider call.")
        return result
