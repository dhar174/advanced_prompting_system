"""
Offline live Flask + SPA test server runner for Playwright E2E qualification.

Launches `app.create_app()` over loopback (`127.0.0.1:<port>`) backed by
`FakeProviderClient` + `ProviderCallLedger` and optional per-request E2E scenario
header dispatch (`X-APS-E2E-Scenario`) strictly within the test harness.
"""

from __future__ import annotations

import os
import socket
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from werkzeug.serving import make_server

from tools.e2e.provider_fixture import FakeProviderClient, ProviderCallLedger, scenario_scope


def _pick_free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class OfflineE2EServer:
    """Thread-hosted loopback Flask + built SPA server for offline Playwright verification."""

    def __init__(
        self,
        *,
        runtime_dir: Path,
        static_folder: Optional[Path] = None,
        rate_limit_rpm: int = 120,
        api_token: str = "",
    ) -> None:
        self.runtime_dir = Path(runtime_dir)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.static_folder = Path(
            static_folder
            or Path(__file__).resolve().parents[2] / "collaborative-assistant-frontend" / "dist"
        )
        self.ledger = ProviderCallLedger()
        self.provider = FakeProviderClient(ledger=self.ledger)
        self.port = _pick_free_loopback_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.rate_limit_rpm = rate_limit_rpm
        self.api_token = api_token
        self._server = None
        self._thread: Optional[threading.Thread] = None
        self._orig_get_client = None
        self._orig_cwd: Optional[Path] = None

    def __enter__(self) -> "OfflineE2EServer":
        import provider_client
        from app import create_app

        self._orig_cwd = Path.cwd()
        os.chdir(self.runtime_dir)

        self._orig_get_client = provider_client.get_client
        client_ref = self.provider

        def e2e_get_client():
            return client_ref

        e2e_get_client.__name__ = "e2e_fake_get_client"
        provider_client.get_client = e2e_get_client

        self._orig_log_file = os.environ.get("APS_LOG_FILE")
        self._orig_allowed_origins = os.environ.get("APS_ALLOWED_ORIGINS")
        # Keep server-side allowed models clean and ensure log writes land in runtime_dir
        os.environ["APS_LOG_FILE"] = str(self.runtime_dir / "logs" / "conversation_logs.json")
        os.environ["APS_ALLOWED_ORIGINS"] = f"{self.base_url},http://localhost:{self.port}"

        flask_app = create_app(
            {
                "TESTING": True,
                "STATIC_FOLDER": str(self.static_folder),
                "RATE_LIMIT_RPM": self.rate_limit_rpm,
                "API_TOKEN": self.api_token,
                "ALLOWED_ORIGINS": (self.base_url, f"http://localhost:{self.port}"),
            }
        )
        inner_wsgi = flask_app.wsgi_app

        def scenario_middleware(environ: Dict[str, Any], start_response: Any):
            scenario = str(environ.get("HTTP_X_APS_E2E_SCENARIO") or "").strip() or "default"
            with scenario_scope(scenario):
                return inner_wsgi(environ, start_response)

        flask_app.wsgi_app = scenario_middleware  # type: ignore[method-assign]
        self._server = make_server("127.0.0.1", self.port, flask_app, threaded=True)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        import provider_client

        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._orig_get_client is not None:
            provider_client.get_client = self._orig_get_client
        if self._orig_log_file is None:
            os.environ.pop("APS_LOG_FILE", None)
        else:
            os.environ["APS_LOG_FILE"] = self._orig_log_file
        if self._orig_allowed_origins is None:
            os.environ.pop("APS_ALLOWED_ORIGINS", None)
        else:
            os.environ["APS_ALLOWED_ORIGINS"] = self._orig_allowed_origins
        if self._orig_cwd is not None:
            os.chdir(self._orig_cwd)
