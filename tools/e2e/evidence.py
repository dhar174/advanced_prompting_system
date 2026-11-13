"""Scenario-specific, redacted evidence for offline browser qualification only."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import threading
import uuid
import zipfile


FIXTURE_VERSION = "issue35-browser-v2"


def _outcome(phases: dict[str, str]) -> str:
    if "failed" in phases.values():
        return "failed"
    if "skipped" in phases.values():
        return "skipped"
    return phases.get("call", "setup_failed")


class EvidenceSession:
    """Pytest report listener and append-only scenario manifest for one invocation."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.reports: dict[str, dict[str, str]] = {}
        self._lock = threading.Lock()
        self.commit = os.environ.get("APS_QUALIFICATION_COMMIT") or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
        self.root = repo_root / ".qualification/logs/browser-e2e" / f"session_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)
        self.manifest = {"commit": self.commit, "fixture_version": FIXTURE_VERSION, "scenarios": []}

    def pytest_runtest_logreport(self, report):
        phases = self.reports.setdefault(report.nodeid, {})
        phases[report.when] = report.outcome
        # Teardown reports arrive after fixture finalization. Persist them too so
        # a failed cleanup can never remain recorded as a successful scenario.
        with self._lock:
            for record in self.manifest["scenarios"]:
                if record["test"] == report.nodeid:
                    record["phases"] = dict(phases)
                    record["outcome"] = _outcome(phases)
                    self._write_manifest()

    def scenario(self, nodeid: str) -> "ScenarioEvidence":
        return ScenarioEvidence(self, nodeid)

    def append(self, record: dict) -> None:
        with self._lock:
            self.manifest["scenarios"].append(record)
            self._write_manifest()

    def _write_manifest(self) -> None:
        target = self.root / "manifest.json"
        staging = target.with_suffix(".tmp")
        staging.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        staging.replace(target)


class ScenarioEvidence(logging.Handler):
    def __init__(self, session: EvidenceSession, nodeid: str):
        super().__init__()
        self.session, self.nodeid = session, nodeid
        slug = re.sub(r"[^A-Za-z0-9_-]", "_", nodeid.split("::")[-1])[:90]
        self.directory = session.root / f"{slug}_{uuid.uuid4().hex[:12]}"
        self.directory.mkdir()
        self.secrets: set[str] = set()
        self.http: list[dict] = []
        self.server_logs: list[dict] = []
        self.trace_active = False

    def redact(self, value):
        if isinstance(value, dict):
            return {
                key: "[REDACTED]" if key.lower() in {"authorization", "api_token"} else self.redact(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self.redact(item) for item in value]
        if not isinstance(value, str):
            return value
        for secret in sorted(self.secrets, key=len, reverse=True):
            if secret:
                value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", value)
        return value.replace("/private/aps-sensitive-path", "[REDACTED PATH]")

    def emit(self, record):
        self.server_logs.append({"level": record.levelname, "message": self.redact(record.getMessage())})

    def json(self, name: str, payload) -> None:
        (self.directory / name).write_text(json.dumps(self.redact(payload), indent=2), encoding="utf-8")

    def record_response(self, response) -> None:
        if not response.url.endswith("/graphql"):
            return
        request = response.request
        try:
            request_body = request.post_data_json if request.method == "POST" else None
            payload = response.json()
        except Exception:
            request_body, payload = None, None
        self.http.append(self.redact({
            "method": request.method,
            "status": response.status,
            "request": request_body,
            "response": payload,
        }))

    def sanitize_trace(self, trace: Path) -> None:
        """Redact trace JSON/network/source text before retaining the ZIP archive."""
        staging = trace.with_suffix(".redacted.zip")
        with zipfile.ZipFile(trace) as source, zipfile.ZipFile(staging, "w", zipfile.ZIP_DEFLATED) as target:
            for info in source.infolist():
                data = source.read(info.filename)
                try:
                    decoded = data.decode("utf-8")
                except UnicodeDecodeError:
                    pass  # Screenshots use masked password fields and never show tokens.
                else:
                    data = self.redact(decoded).encode("utf-8")
                target.writestr(info, data)
        staging.replace(trace)

    def finalize(self, server) -> None:
        ledger = server.ledger.to_dict()
        self.json("provider-ledger.json", ledger)
        self.json("http.json", self.http)
        self.json("server.json", self.server_logs)
        artifacts = []
        for path in sorted((server.runtime_dir / ".aps_artifacts").rglob("*")):
            if path.is_file() and not path.is_symlink():
                artifacts.append({
                    "path": str(path.relative_to(server.runtime_dir)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size": path.stat().st_size,
                })
        self.json("artifacts.json", artifacts)
        outcomes = self.session.reports.get(self.nodeid, {})
        outcome = _outcome(outcomes)
        modes = set()
        for item in self.http:
            for result in ((item.get("response") or {}).get("data") or {}).values():
                if isinstance(result, dict) and result.get("effectiveMode"):
                    modes.add(result["effectiveMode"])
            request = item.get("request") or {}
            requested_mode = ((request.get("variables") or {}).get("input") or {}).get("mode")
            if requested_mode:
                modes.add(requested_mode)
        files = [{"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "size": p.stat().st_size}
                 for p in sorted(self.directory.iterdir()) if p.is_file()]
        self.session.append({
            "test": self.nodeid, "directory": self.directory.name,
            "commit": self.session.commit, "fixture_version": FIXTURE_VERSION,
            "outcome": outcome, "phases": outcomes, "modes": sorted(modes),
            "provider_calls": ledger["total_calls"],
            "correlation_ids": sorted({e["correlation_id"] for e in ledger["entries"] if e["correlation_id"]}),
            "files": files,
        })


def verify_manifest(directory: Path) -> dict:
    """Check every retained scenario and reconcile its hashes and provider-call total."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    scenarios = manifest["scenarios"]
    assert scenarios, "Qualification evidence contains no scenarios"
    assert len({s["directory"] for s in scenarios}) == len(scenarios)
    for scenario in scenarios:
        root = directory / scenario["directory"]
        for item in scenario["files"]:
            path = root / item["path"]
            assert path.stat().st_size == item["size"], path
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], path
        ledger = json.loads((root / "provider-ledger.json").read_text(encoding="utf-8"))
        assert ledger["total_calls"] == len(ledger["entries"]) == scenario["provider_calls"]
        assert scenario["commit"] == manifest["commit"]
    return manifest
