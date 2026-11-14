# output_generator.py

from __future__ import annotations

import contextlib
import contextvars
import csv
import hashlib
import html
import io
import json
import mimetypes
import os
import re
import threading
import uuid
import ntpath
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

_DEFAULT_RUN_SCOPE = "default"
_SCOPE_ID_PATTERN = re.compile(r"^(?:run|direct)_[0-9a-f]{32}$")


@dataclass(frozen=True)
class _ArtifactScope:
    """Server-owned opaque artifact scope bound to an optional caller correlation label."""

    owner_id: str
    correlation_id: str


_ACTIVE_SCOPE: contextvars.ContextVar[Optional[_ArtifactScope]] = contextvars.ContextVar(
    "aps_active_artifact_scope", default=None
)


class ArtifactRef(BaseModel):
    """Metadata descriptor for a tracked generated artifact scoped to a single run."""

    artifact_id: str = Field(..., description="Unique identifier for the generated artifact.")
    run_id: str = Field(..., description="Opaque server-generated run identifier owning this artifact.")
    correlation_id: Optional[str] = Field(
        default=None,
        description="Caller-supplied correlation label for tracing (separate from run ownership).",
    )
    filename: str = Field(..., description="Basename of the generated artifact.")
    path: str = Field(..., description="Absolute path to the artifact on disk.")
    mime_type: str = Field(default="application/octet-stream", description="Detected MIME type.")
    size_bytes: int = Field(default=0, ge=0, description="File size in bytes.")
    sha256: str = Field(default="", description="SHA-256 hex digest of the artifact bytes.")


def _direct_owner_id(raw_id: str) -> str:
    """Derive a deterministic, collision-resistant owner key for unscoped run_id strings."""
    digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:32]
    return f"direct_{digest}"


def _storage_dir_name(owner_id: str) -> str:
    """Map an owner_id to a collision-resistant, traversal-free single directory segment."""
    if owner_id == _DEFAULT_RUN_SCOPE:
        return _DEFAULT_RUN_SCOPE
    if _SCOPE_ID_PATTERN.fullmatch(owner_id):
        return owner_id
    digest = hashlib.sha256(owner_id.encode("utf-8")).hexdigest()[:32]
    return f"run_{digest}"


def _normalize_path_for_compare(path: str) -> str:
    """Normalize path separators and platform casing for safe path comparisons."""
    return os.path.normcase(os.path.normpath(path))


def _paths_equal(path_a: str, path_b: str) -> bool:
    """Return True iff two paths refer to the same normalized location."""
    return _normalize_path_for_compare(path_a) == _normalize_path_for_compare(path_b)


def _is_within_directory(target_path: str, base_dir: str) -> bool:
    """Return True iff target_path is inside or equal to base_dir on the same drive."""
    if not target_path or not base_dir:
        return False
    try:
        norm_target = _normalize_path_for_compare(target_path)
        norm_base = _normalize_path_for_compare(base_dir)
        return os.path.commonpath([norm_target, norm_base]) == norm_base
    except ValueError:
        return False


class RunArtifactRegistry:
    """Thread-safe registry of generated artifacts isolated strictly per server-owned run scope."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._by_run: Dict[str, Dict[str, ArtifactRef]] = {}
        self._scopes_by_owner: Dict[str, _ArtifactScope] = {}
        self._latest_owner_by_label: Dict[str, str] = {}

    def register_scope(self, scope: _ArtifactScope, bind_label: bool = True) -> None:
        """Record a server-generated scope and optionally bind its exact correlation label."""
        with self._lock:
            self._scopes_by_owner[scope.owner_id] = scope
            if bind_label:
                self._latest_owner_by_label[scope.correlation_id] = scope.owner_id

    def resolve_owner_id(self, run_id: Optional[str] = None) -> str:
        """
        Resolve a caller-supplied run_id or correlation label to its exact server owner_id.

        Active ContextVar scope takes precedence when run_id is None or matches the active
        scope's opaque owner_id or exact correlation_id. Distinct correlation labels
        (e.g. 'tenant/a' vs 'tenant:a') never alias, and reused correlation labels resolve
        to distinct per-run owner_ids.
        """
        active = _ACTIVE_SCOPE.get()
        if run_id is None or (isinstance(run_id, str) and run_id == ""):
            return active.owner_id if active is not None else _DEFAULT_RUN_SCOPE

        raw = str(run_id)
        if "\x00" in raw:
            raise ValueError("Artifact run_id must not contain NUL bytes.")

        if active is not None and (raw == active.owner_id or raw == active.correlation_id):
            return active.owner_id

        if active is not None:
            raise PermissionError("Cannot select a foreign artifact owner inside an active run.")

        with self._lock:
            if raw in self._scopes_by_owner or raw in self._by_run:
                return raw
            if raw in self._latest_owner_by_label:
                return self._latest_owner_by_label[raw]

        if raw == _DEFAULT_RUN_SCOPE:
            return _DEFAULT_RUN_SCOPE
        return _direct_owner_id(raw)

    def _resolve_correlation_id(self, owner_id: str, raw_run_id: Optional[str]) -> str:
        """Return the human correlation label associated with owner_id."""
        active = _ACTIVE_SCOPE.get()
        if active is not None and active.owner_id == owner_id:
            return active.correlation_id
        with self._lock:
            scope = self._scopes_by_owner.get(owner_id)
            if scope is not None:
                return scope.correlation_id
        if raw_run_id is not None and str(raw_run_id) != "":
            return str(raw_run_id)
        return _DEFAULT_RUN_SCOPE

    def register(self, filepath: str, run_id: Optional[str] = None) -> ArtifactRef:
        """Register an explicitly created artifact path under its resolved owner scope."""
        abs_input = os.path.abspath(filepath)
        if os.path.islink(abs_input):
            raise PermissionError(f"Cannot register symlink '{filepath}' as a tracked artifact.")

        owner_id = self.resolve_owner_id(run_id)
        correlation_id = self._resolve_correlation_id(owner_id, run_id)
        real_path = os.path.realpath(abs_input)
        size_bytes = 0
        sha256_hex = ""
        if os.path.isfile(real_path):
            with open(real_path, "rb") as f:
                raw = f.read()
            size_bytes = len(raw)
            sha256_hex = hashlib.sha256(raw).hexdigest()

        mime_type, _ = mimetypes.guess_type(real_path)
        if real_path.lower().endswith(".pdf"):
            mime_type = "application/pdf"
        elif real_path.lower().endswith(".json"):
            mime_type = "application/json"
        elif real_path.lower().endswith(".html"):
            mime_type = "text/html"
        elif real_path.lower().endswith(".csv"):
            mime_type = "text/csv"
        elif real_path.lower().endswith(".py"):
            mime_type = "text/x-python"
        elif real_path.lower().endswith(".txt"):
            mime_type = "text/plain"

        with self._lock:
            active = _ACTIVE_SCOPE.get()
            if active is not None and active.owner_id == owner_id:
                self._scopes_by_owner.setdefault(owner_id, active)
            run_map = self._by_run.setdefault(owner_id, {})
            existing = run_map.get(real_path)
            artifact_id = existing.artifact_id if existing else f"art_{uuid.uuid4().hex[:12]}"
            ref = ArtifactRef(
                artifact_id=artifact_id,
                run_id=owner_id,
                correlation_id=correlation_id,
                filename=os.path.basename(real_path),
                path=real_path,
                mime_type=mime_type or "application/octet-stream",
                size_bytes=size_bytes,
                sha256=sha256_hex,
            )
            run_map[real_path] = ref
            return ref

    def is_tracked(self, filepath: str, run_id: Optional[str] = None) -> bool:
        """Check whether filepath is explicitly registered under the target run scope."""
        if not filepath or not isinstance(filepath, str):
            return False
        try:
            abs_input = os.path.abspath(filepath)
            if os.path.islink(filepath) or os.path.islink(abs_input):
                return False
            owner_id = self.resolve_owner_id(run_id)
            real_path = os.path.realpath(abs_input)
        except Exception:
            return False
        with self._lock:
            run_map = self._by_run.get(owner_id, {})
            return real_path in run_map

    def get_ref(self, filepath: str, run_id: Optional[str] = None) -> Optional[ArtifactRef]:
        """Retrieve ArtifactRef for a tracked path within the target run scope."""
        if not filepath or not isinstance(filepath, str):
            return None
        try:
            abs_input = os.path.abspath(filepath)
            if os.path.islink(filepath) or os.path.islink(abs_input):
                return None
            owner_id = self.resolve_owner_id(run_id)
            real_path = os.path.realpath(abs_input)
        except Exception:
            return None
        with self._lock:
            return self._by_run.get(owner_id, {}).get(real_path)

    def list_artifacts(self, run_id: Optional[str] = None) -> List[ArtifactRef]:
        """List all tracked artifacts for a specific run scope."""
        try:
            owner_id = self.resolve_owner_id(run_id)
        except PermissionError:
            return []
        with self._lock:
            return list(self._by_run.get(owner_id, {}).values())

    def clear(self, run_id: Optional[str] = None) -> None:
        """Clear tracked artifacts for a single run_id, or all runs if run_id is None."""
        with self._lock:
            if run_id is None:
                self._by_run.clear()
                self._scopes_by_owner.clear()
                self._latest_owner_by_label.clear()
            else:
                owner_id = self.resolve_owner_id(run_id)
                self._by_run.pop(owner_id, None)
                scope = self._scopes_by_owner.pop(owner_id, None)
                if scope is not None and self._latest_owner_by_label.get(scope.correlation_id) == owner_id:
                    self._latest_owner_by_label.pop(scope.correlation_id, None)
                if str(run_id) in self._latest_owner_by_label and self._latest_owner_by_label[str(run_id)] == owner_id:
                    self._latest_owner_by_label.pop(str(run_id), None)


_REGISTRY = RunArtifactRegistry()


def get_artifact_registry() -> RunArtifactRegistry:
    """Return the singleton run-scoped artifact registry."""
    return _REGISTRY


def get_active_run_id() -> Optional[str]:
    """Return the active server-owned artifact run_id from contextvars, if any."""
    active = _ACTIVE_SCOPE.get()
    return active.owner_id if active is not None else None


@contextlib.contextmanager
def artifact_run_scope(run_id: Optional[str]) -> Iterator[str]:
    """
    Bind a fresh opaque server-owned run scope for artifact generation and lookup.

    The caller-supplied run_id is preserved as an exact correlation label separate from
    the unique server-generated storage/ownership identity so colliding or reused
    correlation labels never share directories or registry buckets.
    """
    if run_id is not None and "\x00" in str(run_id):
        raise ValueError("Artifact run_id must not contain NUL bytes.")
    raw_label = str(run_id) if run_id is not None and str(run_id) != "" else None
    owner_id = f"run_{uuid.uuid4().hex}"
    correlation_label = raw_label if raw_label is not None else owner_id
    scope = _ArtifactScope(owner_id=owner_id, correlation_id=correlation_label)
    token = _ACTIVE_SCOPE.set(scope)
    _REGISTRY.register_scope(scope, bind_label=(raw_label is not None))
    try:
        yield owner_id
    finally:
        _ACTIVE_SCOPE.reset(token)


def get_artifacts_base_dir() -> str:
    """Return the application-owned base directory for generated run artifacts."""
    configured = os.environ.get("APS_ARTIFACTS_DIR")
    base = configured if configured else os.path.join(os.getcwd(), ".aps_artifacts")
    raw_base = os.path.abspath(base)
    os.makedirs(raw_base, exist_ok=True)
    return os.path.realpath(raw_base)


def get_run_artifact_dir(run_id: Optional[str] = None) -> str:
    """Return (and lazily create) the dedicated collision-resistant artifact directory for a run."""
    owner_id = _REGISTRY.resolve_owner_id(run_id)
    dir_name = _storage_dir_name(owner_id)
    raw_base = get_artifacts_base_dir()
    os.makedirs(raw_base, exist_ok=True)
    base_dir = os.path.realpath(raw_base)
    run_dir = os.path.realpath(os.path.join(base_dir, dir_name))
    if not _is_within_directory(run_dir, base_dir) or _paths_equal(run_dir, base_dir):
        raise PermissionError("Invalid artifact run scope path")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def resolve_output_filepath(filename: str, run_id: Optional[str] = None) -> str:
    """
    Resolve an output filename safely.

    - Inside an active run scope (or explicit run_id), relative filenames are confined inside
      that run's dedicated artifact directory (`APS_ARTIFACTS_DIR/<opaque_run_dir>/...`),
      rejecting `..` traversal, drive-relative escapes, or symlink escapes.
    - Relative filenames without an active run scope resolve under CWD without escaping CWD.
    - Explicit absolute paths are permitted only if lexical parent traversal segments (`..`) are
      absent (and when inside an active run scope, must remain confined to the run directory).
    """
    if not filename or not isinstance(filename, str):
        raise ValueError("Artifact filename must be a non-empty string.")

    raw_name = filename.strip()
    if not raw_name or "\x00" in raw_name:
        raise ValueError("Artifact filename must be a valid non-empty string.")

    portable_parts = raw_name.replace("\\", "/").split("/")
    if ".." in portable_parts or (os.name != "nt" and ntpath.splitdrive(raw_name)[0]):
        raise PermissionError("Artifact path contains traversal or a foreign drive.")

    active_run = run_id if (run_id is not None and str(run_id) != "") else get_active_run_id()

    if not os.path.isabs(raw_name):
        drive, _ = os.path.splitdrive(raw_name)
        if drive or raw_name.startswith(("/", "\\")):
            raise PermissionError(
                f"Artifact path '{filename}' escapes allowed artifact directory."
            )
        anchor_dir = get_run_artifact_dir(active_run) if active_run else os.path.realpath(os.getcwd())
        candidate = os.path.abspath(os.path.join(anchor_dir, raw_name))
        if os.path.islink(candidate):
            raise PermissionError(
                f"Artifact path '{filename}' escapes allowed artifact directory via symlink."
            )
        real_parent = os.path.realpath(os.path.dirname(candidate))
        real_anchor = os.path.realpath(anchor_dir)
        if not _is_within_directory(real_parent, real_anchor):
            raise PermissionError(
                f"Artifact path '{filename}' escapes allowed artifact directory '{anchor_dir}'."
            )
        real_target = os.path.realpath(candidate)
        if not _is_within_directory(real_target, real_anchor) or _paths_equal(real_target, real_anchor):
            raise PermissionError(
                f"Artifact path '{filename}' escapes allowed artifact directory via symlink."
            )
        parent_dir = os.path.dirname(real_target)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        return real_target

    # Explicit absolute path
    normalized_parts = os.path.normpath(raw_name).split(os.sep)
    if ".." in normalized_parts or ".." in raw_name.replace("\\", "/").split("/"):
        raise PermissionError(f"Path traversal ('..') is not allowed in artifact path '{filename}'.")
    abs_path = os.path.abspath(raw_name)
    if os.path.islink(abs_path):
        raise PermissionError(
            f"Absolute artifact path '{filename}' cannot target a symlink."
        )
    real_target = os.path.realpath(abs_path)
    if active_run:
        real_anchor = os.path.realpath(get_run_artifact_dir(active_run))
        if not _is_within_directory(real_target, real_anchor) or _paths_equal(real_target, real_anchor):
            raise PermissionError(
                f"Absolute artifact path '{filename}' escapes active run sandbox '{real_anchor}'."
            )
    parent_dir = os.path.dirname(real_target)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    return real_target if active_run else abs_path


def track_generated_artifact(filepath: str, run_id: Optional[str] = None) -> str:
    """Register an explicitly created artifact file path for runtime verification."""
    ref = _REGISTRY.register(filepath, run_id=run_id)
    return ref.path


def is_tracked_artifact(filepath: str, run_id: Optional[str] = None) -> bool:
    """Check whether a filepath was explicitly generated and tracked for the target run."""
    return _REGISTRY.is_tracked(filepath, run_id=run_id)


def get_tracked_artifact_ref(
    filepath: str, run_id: Optional[str] = None
) -> Optional[ArtifactRef]:
    """Return ArtifactRef metadata if filepath is tracked in the target run scope."""
    return _REGISTRY.get_ref(filepath, run_id=run_id)


def list_run_artifacts(run_id: Optional[str] = None) -> List[ArtifactRef]:
    """Return all ArtifactRef entries recorded for the target run scope."""
    return _REGISTRY.list_artifacts(run_id=run_id)


def clear_tracked_artifacts(run_id: Optional[str] = None) -> None:
    """Clear registered artifact paths (for a specific run or all runs)."""
    _REGISTRY.clear(run_id=run_id)


def read_tracked_artifact(
    filepath: str,
    run_id: Optional[str] = None,
    max_bytes: int = 1_000_000,
) -> Optional[str]:
    """
    Safely read a UTF-8 text preview/content from an artifact owned by `run_id`.
    Returns None for untracked paths, symlink escapes, missing files, or binary PDFs.
    """
    if not filepath or not isinstance(filepath, str):
        return None
    try:
        abs_input = os.path.abspath(filepath)
        if os.path.islink(filepath) or os.path.islink(abs_input):
            return None
        ref = _REGISTRY.get_ref(filepath, run_id=run_id)
        if ref is None or not os.path.isfile(ref.path) or os.path.islink(ref.path):
            return None
        if os.path.realpath(abs_input) != ref.path:
            return None
    except Exception:
        return None
    if ref.mime_type == "application/pdf":
        return None
    try:
        with open(ref.path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(max_bytes)
    except Exception:
        return None


def generate_simple_concise_answer_message(
    final_decision: str, filename: str = "final_output.txt"
) -> str:
    """
    Generates a simple concise text answer message.

    Args:
        final_decision (str): The final_decision to be included in the message.

    Returns:
        str: The simple concise text answer message.
    """
    return final_decision


def generate_json_output(
    final_decision: Any, filename: str = "final_output.json"
) -> str:
    """
    Generates a JSON file from the provided final_decision.

    Args:
        final_decision (Any): The final_decision to be serialized into JSON.
        filename (str): The name of the output JSON file.

    Returns:
        str: The path to the generated JSON file.
    """
    try:
        target_path = resolve_output_filepath(filename)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(final_decision, f, indent=4, ensure_ascii=False)
        return track_generated_artifact(target_path)
    except (IOError, OSError, PermissionError) as e:
        print(f"[Error] Error writing JSON file '{filename}': {e}")
        raise


def _escape_pdf_text(text: str) -> str:
    """Escape characters for a PDF 1.4 literal text string encoded as Latin-1."""
    normalized = text.encode("latin-1", errors="replace").decode("latin-1")
    return normalized.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap_pdf_lines(text: str, max_chars: int = 86) -> List[str]:
    """Wrap raw text lines cleanly to fit standard Letter page width."""
    wrapped: List[str] = []
    for raw_line in str(text or "").splitlines() or [""]:
        line = raw_line.expandtabs(4)
        if not line:
            wrapped.append("")
            continue
        while len(line) > max_chars:
            split_at = line.rfind(" ", 0, max_chars + 1)
            if split_at <= 0:
                split_at = max_chars
            wrapped.append(line[:split_at].rstrip())
            line = line[split_at:].lstrip()
        wrapped.append(line)
    return wrapped or [""]


def _build_minimal_pdf_bytes(text: str) -> bytes:
    """Build a valid, zero-dependency %PDF-1.4 document containing text lines."""
    lines = _wrap_pdf_lines(text)
    lines_per_page = 46
    pages_lines: List[List[str]] = [
        lines[i : i + lines_per_page] for i in range(0, len(lines), lines_per_page)
    ] or [[""]]

    objects: List[bytes] = []

    # 1: Catalog, 2: Pages, 3: Helvetica Font
    page_obj_nums: List[int] = []
    next_obj_num = 4
    page_streams: List[Tuple[int, int, bytes]] = []
    for page_chunk in pages_lines:
        page_num = next_obj_num
        content_num = next_obj_num + 1
        next_obj_num += 2
        page_obj_nums.append(page_num)

        stream_cmds = ["BT", "/F1 11 Tf", "50 742 Td", "14 TL"]
        for idx, line in enumerate(page_chunk):
            escaped = _escape_pdf_text(line)
            if idx == 0:
                stream_cmds.append(f"({escaped}) Tj")
            else:
                stream_cmds.append(f"T* ({escaped}) Tj")
        stream_cmds.append("ET")
        stream_bytes = "\n".join(stream_cmds).encode("latin-1", errors="replace")
        page_streams.append((page_num, content_num, stream_bytes))  # type: ignore[arg-type]

    kids_refs = " ".join(f"{num} 0 R" for num in page_obj_nums)
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objects.append(
        f"2 0 obj\n<< /Type /Pages /Kids [{kids_refs}] /Count {len(page_obj_nums)} >>\nendobj\n".encode(
            "ascii"
        )
    )
    objects.append(
        b"3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    )

    for page_num, content_num, stream_bytes in page_streams:  # type: ignore[misc]
        objects.append(
            (
                f"{page_num} 0 obj\n"
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_num} 0 R >>\n"
                f"endobj\n"
            ).encode("ascii")
        )
        objects.append(
            (
                f"{content_num} 0 obj\n"
                f"<< /Length {len(stream_bytes)} >>\n"
                f"stream\n"
            ).encode("ascii")
            + stream_bytes
            + b"\nendstream\nendobj\n"
        )

    buf = io.BytesIO()
    buf.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: List[int] = [0]
    for obj in objects:
        offsets.append(buf.tell())
        buf.write(obj)

    xref_offset = buf.tell()
    total_objs = len(objects) + 1
    buf.write(f"xref\n0 {total_objs}\n".encode("ascii"))
    buf.write(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        buf.write(f"{offset:010d} 00000 n \n".encode("ascii"))

    buf.write(
        (
            f"trailer\n<< /Size {total_objs} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return buf.getvalue()


def generate_pdf_output(final_decision: str, filename: str = "final_output.pdf") -> str:
    """
    Generates a structurally valid %PDF-1.4 file from the provided final_decision.

    Args:
        final_decision (str): The final_decision to be included in the PDF.
        filename (str): The name of the output PDF file.

    Returns:
        str: The path to the generated PDF file.
    """
    try:
        target_path = resolve_output_filepath(filename)
        if isinstance(final_decision, (dict, list)):
            text = json.dumps(final_decision, indent=2, ensure_ascii=False)
        else:
            text = str(final_decision or "")
        pdf_bytes = _build_minimal_pdf_bytes(text)
        with open(target_path, "wb") as f:
            f.write(pdf_bytes)
        return track_generated_artifact(target_path)
    except (IOError, OSError, PermissionError) as e:
        print(f"[Error] Error writing PDF file '{filename}': {e}")
        raise


def generate_text_file_output(
    final_decision: str, filename: str = "final_output.txt"
) -> str:
    """
    Generates a plain text file from the provided final_decision.

    Args:
        final_decision (str): The final_decision to be written into the text file.
        filename (str): The name of the output text file.

    Returns:
        str: The path to the generated text file.
    """
    try:
        target_path = resolve_output_filepath(filename)
        content = (
            json.dumps(final_decision, indent=2, ensure_ascii=False)
            if isinstance(final_decision, (dict, list))
            else str(final_decision or "")
        )
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)
        return track_generated_artifact(target_path)
    except (IOError, OSError, PermissionError) as e:
        print(f"[Error] Error writing text file '{filename}': {e}")
        raise


def generate_html_output(
    final_decision: Union[str, Dict[str, Any], List[Any]],
    filename: str = "final_output.html",
) -> str:
    """
    Generates an HTML5 document from the provided final_decision with XSS escaping.

    Args:
        final_decision (Union[str, Dict[str, Any], List[Any]]): Content to render in HTML.
        filename (str): The name of the output HTML file.

    Returns:
        str: The path to the generated HTML file.
    """
    sections: List[str] = []
    if isinstance(final_decision, dict):
        for key, value in final_decision.items():
            safe_k = html.escape(str(key))
            safe_v = html.escape(str(value))
            sections.append(f"<section><h2>{safe_k}</h2><p>{safe_v}</p></section>")
    elif isinstance(final_decision, list):
        items_html = "".join(f"<li>{html.escape(str(item))}</li>" for item in final_decision)
        sections.append(f"<ul>{items_html}</ul>")
    else:
        raw_text = str(final_decision or "")
        paragraphs = [p for p in raw_text.splitlines() if p.strip()]
        if paragraphs:
            sections.extend(f"<p>{html.escape(p)}</p>" for p in paragraphs)
        else:
            sections.append("<p></p>")

    html_content = (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        '<head><meta charset="utf-8"><title>Final Output</title></head>\n'
        f"<body>{''.join(sections)}</body>\n"
        "</html>\n"
    )

    try:
        target_path = resolve_output_filepath(filename)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return track_generated_artifact(target_path)
    except (IOError, OSError, PermissionError) as e:
        print(f"[Error] Error writing HTML file '{filename}': {e}")
        raise


def generate_python_script(
    final_decision: str, filename: str = "final_output.py"
) -> str:
    """
    Generates a Python script file from the provided final_decision.

    Args:
        final_decision (str): The final_decision to be included in the Python script.
        filename (str): The name of the output Python script file.

    Returns:
        str: The path to the generated Python script file.
    """
    try:
        target_path = resolve_output_filepath(filename)
        cleaned = str(final_decision or "")
        # Remove everything before the first set of ``` and after the last set of ``` using regex
        cleaned = re.sub(r"^[\s\S]*?```([\s\S]*?)```[\s\S]*$", r"\1", cleaned)
        # If the word "python" is at the beginning of the code snippet, remove it
        cleaned = re.sub(r"^python\r?\n?", "", cleaned).strip()
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(cleaned)
        return track_generated_artifact(target_path)
    except (IOError, OSError, PermissionError) as e:
        print(f"[Error] Error writing Python script file '{filename}': {e}")
        raise


def generate_code_snippet(
    final_decision: str, filename: str = "final_output.py"
) -> str:
    """
    Generates a code snippet file from the provided final_decision.

    Args:
        final_decision (str): The final_decision to be included in the code snippet.
        filename (str): The name of the output code snippet file.

    Returns:
        str: The path to the generated code snippet file.
    """
    try:
        target_path = resolve_output_filepath(filename)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(str(final_decision or ""))
        return track_generated_artifact(target_path)
    except (IOError, OSError, PermissionError) as e:
        print(f"[Error] Error writing code snippet file '{filename}': {e}")
        raise


def generate_csv_output(
    final_decision: Union[str, Dict[str, Any], List[Any]],
    filename: str = "final_output.csv",
) -> str:
    """
    Generates a RFC-4180 compliant CSV file from the provided final_decision.

    Args:
        final_decision (Union[str, Dict[str, Any], List[Any]]): Structured or text data for CSV.
        filename (str): The name of the output CSV file.

    Returns:
        str: The path to the generated CSV file.
    """
    try:
        target_path = resolve_output_filepath(filename)
        with open(target_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            if isinstance(final_decision, dict):
                writer.writerow(["key", "value"])
                for key, value in final_decision.items():
                    writer.writerow([str(key), str(value)])
            elif isinstance(final_decision, list):
                if final_decision and all(isinstance(item, dict) for item in final_decision):
                    headers: List[str] = []
                    for row in final_decision:
                        for k in row.keys():
                            if str(k) not in headers:
                                headers.append(str(k))
                    writer.writerow(headers)
                    for row in final_decision:
                        writer.writerow([str(row.get(h, "")) for h in headers])
                else:
                    writer.writerow(["value"])
                    for item in final_decision:
                        writer.writerow([str(item)])
            else:
                raw_text = str(final_decision or "")
                lines = [ln for ln in raw_text.splitlines() if ln.strip()]
                if lines and all("," in ln for ln in lines):
                    parsed_reader = csv.reader(io.StringIO("\n".join(lines)))
                    for row in parsed_reader:
                        writer.writerow(row)
                else:
                    writer.writerow(["content"])
                    for ln in lines or [""]:
                        writer.writerow([ln])
        return track_generated_artifact(target_path)
    except (IOError, OSError, PermissionError) as e:
        print(f"[Error] Error writing CSV file '{filename}': {e}")
        raise
