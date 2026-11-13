from __future__ import annotations

from collections import deque
import hmac
import logging
import os
import threading
import time
from typing import Any, Deque, Dict, Optional, Set
from urllib.parse import urlparse

from flask import Flask, Response, jsonify, request, send_from_directory
import strawberry
from strawberry.flask.views import GraphQLView
from werkzeug.utils import safe_join

from provider_client import has_configured_provider
from schema import Mutation, Query

logger = logging.getLogger(__name__)

_SCHEMA_EXTENSIONS = []
try:
    from strawberry.extensions import QueryDepthLimiter, MaxTokensLimiter

    _SCHEMA_EXTENSIONS.extend([
        lambda: QueryDepthLimiter(max_depth=12),
        lambda: MaxTokensLimiter(max_token_count=1500),
    ])
except ImportError:
    pass

# Create the Strawberry schema at module scope cleanly (zero provider/model I/O at import)
schema = strawberry.Schema(query=Query, mutation=Mutation, extensions=_SCHEMA_EXTENSIONS)

_DEFAULT_LOCAL_ORIGINS: Set[str] = {
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://testserver",
}


def _get_allowed_origins(app: Flask) -> Set[str]:
    configured = app.config.get("ALLOWED_ORIGINS")
    if configured is not None:
        if isinstance(configured, str):
            return {o.strip().rstrip("/").lower() for o in configured.split(",") if o.strip()}
        return {str(o).strip().rstrip("/").lower() for o in configured if str(o).strip()}
    env_origins = os.environ.get("APS_ALLOWED_ORIGINS", "").strip()
    if env_origins:
        return {o.strip().rstrip("/").lower() for o in env_origins.split(",") if o.strip()}
    return set(_DEFAULT_LOCAL_ORIGINS)


def _is_origin_allowed(origin: str, app: Flask) -> bool:
    norm = origin.strip().rstrip("/").lower()
    allowed = _get_allowed_origins(app)
    if norm in allowed:
        return True
    # Only auto-allow exact loopback host_url when user has NOT explicitly overridden APS_ALLOWED_ORIGINS
    explicit_env = os.environ.get("APS_ALLOWED_ORIGINS", "").strip()
    explicit_cfg = app.config.get("ALLOWED_ORIGINS")
    if not explicit_env and explicit_cfg is None:
        host_origin = request.host_url.rstrip("/").lower()
        parsed_host = urlparse(host_origin).hostname or ""
        if parsed_host in {"localhost", "127.0.0.1", "::1"} and norm == host_origin:
            return True
    return False


def _get_rate_limit_rpm(app: Flask) -> int:
    if "RATE_LIMIT_RPM" in app.config and app.config["RATE_LIMIT_RPM"] is not None:
        try:
            return int(app.config["RATE_LIMIT_RPM"])
        except (TypeError, ValueError):
            return 60
    raw = os.environ.get("APS_RATE_LIMIT_RPM", "60").strip()
    try:
        return int(raw)
    except ValueError:
        return 60


def _get_api_token(app: Flask) -> str:
    if "API_TOKEN" in app.config and app.config["API_TOKEN"] is not None:
        return str(app.config["API_TOKEN"]).strip()
    return os.environ.get("APS_API_TOKEN", "").strip()


def _is_graphiql_enabled(app: Flask) -> bool:
    if "ENABLE_GRAPHIQL" in app.config and app.config["ENABLE_GRAPHIQL"] is not None:
        return bool(app.config["ENABLE_GRAPHIQL"])
    return os.environ.get("APS_ENABLE_GRAPHIQL", "0").strip() == "1"


def create_app(test_config: Optional[Dict[str, Any]] = None) -> Flask:
    """Create and configure the hardened Flask + Strawberry GraphQL application."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_static = os.path.join(base_dir, "collaborative-assistant-frontend", "dist")
    static_folder = (
        (test_config or {}).get("STATIC_FOLDER")
        or os.environ.get("APS_STATIC_FOLDER", "").strip()
        or default_static
    )

    flask_app = Flask(__name__, static_folder=static_folder, static_url_path="/_static")

    try:
        default_max_bytes = int(os.environ.get("APS_MAX_CONTENT_LENGTH", str(1024 * 1024)))
    except ValueError:
        default_max_bytes = 1024 * 1024

    flask_app.config.from_mapping(
        MAX_CONTENT_LENGTH=default_max_bytes,
        JSON_SORT_KEYS=False,
    )
    if test_config:
        flask_app.config.update(test_config)

    rate_lock = threading.Lock()
    rate_buckets: Dict[str, Deque[float]] = {}

    @flask_app.before_request
    def _enforce_graphql_guards() -> Optional[Response]:
        max_len = flask_app.config.get("MAX_CONTENT_LENGTH")
        if max_len is not None and request.content_length is not None and request.content_length > int(max_len):
            resp = jsonify({"error": "Request payload exceeds maximum allowed size."})
            resp.status_code = 413
            return resp

        if request.path.rstrip("/") != "/graphql":
            return None

        origin = request.headers.get("Origin")
        if origin and not _is_origin_allowed(origin, flask_app):
            resp = jsonify({"error": "Origin not allowed."})
            resp.status_code = 403
            return resp

        if request.method == "OPTIONS":
            resp = Response("", status=204)
            return resp

        api_token = _get_api_token(flask_app)
        if api_token:
            auth_header = request.headers.get("Authorization", "")
            prefix = "Bearer "
            if not auth_header.startswith(prefix):
                resp = jsonify({"error": "Authentication required."})
                resp.status_code = 401
                return resp
            supplied = auth_header[len(prefix):].strip()
            if not hmac.compare_digest(supplied.encode("utf-8"), api_token.encode("utf-8")):
                resp = jsonify({"error": "Invalid bearer token."})
                resp.status_code = 401
                return resp

        if request.method == "GET" and not _is_graphiql_enabled(flask_app) and "query" not in request.args:
            resp = jsonify({"error": "GraphiQL interface is disabled."})
            resp.status_code = 405
            return resp

        if request.method == "POST":
            mimetype = (request.mimetype or "").lower()
            if mimetype != "application/json":
                resp = jsonify({"error": "Content-Type must be application/json."})
                resp.status_code = 415
                return resp

        if request.method in ("GET", "POST"):
            rpm_limit = _get_rate_limit_rpm(flask_app)
            if rpm_limit > 0:
                client_ip = request.remote_addr or "unknown"
                now = time.monotonic()
                cutoff = now - 60.0
                with rate_lock:
                    if len(rate_buckets) > 1000:
                        stale_ips = [ip for ip, dq in rate_buckets.items() if not dq or dq[-1] <= cutoff]
                        for ip in stale_ips:
                            rate_buckets.pop(ip, None)
                    bucket = rate_buckets.setdefault(client_ip, deque())
                    while bucket and bucket[0] <= cutoff:
                        bucket.popleft()
                    if len(bucket) >= rpm_limit:
                        retry_after = max(1, int(60 - (now - bucket[0])) + 1)
                        resp = jsonify({"error": "Rate limit exceeded. Please retry later."})
                        resp.status_code = 429
                        resp.headers["Retry-After"] = str(retry_after)
                        return resp
                    bucket.append(now)

        return None

    @flask_app.after_request
    def _apply_security_headers(response: Response) -> Response:
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")

        origin = request.headers.get("Origin")
        if origin and _is_origin_allowed(origin, flask_app):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            vary = response.headers.get("Vary", "")
            if "Origin" not in vary:
                response.headers["Vary"] = f"{vary}, Origin".strip(", ")

        if request.path.rstrip("/") == "/graphql":
            response.headers["Cache-Control"] = "no-store"

        return response

    flask_app.add_url_rule(
        "/graphql",
        view_func=GraphQLView.as_view("graphql_view", schema=schema),
        methods=["GET", "POST", "OPTIONS"],
    )

    @flask_app.route("/healthz", methods=["GET"])
    def healthz() -> Response:
        return jsonify({"status": "ok"})

    @flask_app.route("/readyz", methods=["GET"])
    def readyz() -> Response:
        folder = flask_app.static_folder or ""
        bundle_ready = bool(folder and os.path.isfile(os.path.join(folder, "index.html")))
        return jsonify(
            {
                "status": "ready",
                "provider_configured": bool(has_configured_provider()),
                "static_bundle_available": bundle_ready,
            }
        )

    @flask_app.route("/", defaults={"path": ""})
    @flask_app.route("/<path:path>")
    def serve_react_app(path: str):
        folder = flask_app.static_folder or ""
        if not folder or not os.path.isdir(folder):
            return jsonify({"error": "Frontend build is not available."}), 404

        if path:
            safe_target = safe_join(folder, path)
            if safe_target is None:
                return jsonify({"error": "Not found."}), 404
            if os.path.isfile(safe_target):
                rel_path = os.path.relpath(safe_target, folder).replace(os.sep, "/")
                return send_from_directory(folder, rel_path)
            # If request targets a missing static asset file, return 404 rather than index.html
            ext = os.path.splitext(path)[1].lower()
            if path.startswith("assets/") or ext in {
                ".js", ".mjs", ".css", ".map", ".json", ".png", ".jpg", ".jpeg",
                ".gif", ".svg", ".ico", ".webp", ".woff", ".woff2", ".ttf", ".txt",
            }:
                return jsonify({"error": "Static asset not found."}), 404

        index_file = os.path.join(folder, "index.html")
        if os.path.isfile(index_file):
            return send_from_directory(folder, "index.html")
        return jsonify({"error": "Frontend bundle not found."}), 404

    @flask_app.errorhandler(404)
    def _handle_404(_err: Any) -> Response:
        resp = jsonify({"error": "Not found."})
        resp.status_code = 404
        return resp

    @flask_app.errorhandler(413)
    def _handle_413(_err: Any) -> Response:
        resp = jsonify({"error": "Request payload exceeds maximum allowed size."})
        resp.status_code = 413
        return resp

    @flask_app.errorhandler(500)
    def _handle_500(_err: Any) -> Response:
        logger.exception("Unhandled server exception")
        resp = jsonify({"error": "Internal server error."})
        resp.status_code = 500
        return resp

    return flask_app


app = create_app()

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    debug_enabled = os.environ.get("FLASK_DEBUG", "0").strip() == "1"
    app.run(host=host, port=port, debug=debug_enabled)
