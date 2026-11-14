"""
Production WSGI entrypoint for the Advanced Prompting System GraphQL & SPA server.

Binds non-debug WSGI serving to 0.0.0.0:$PORT (default 8000; Cloud Run injects PORT=8080).
Use with gunicorn (Linux/containers):
    gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 8 --timeout 120 wsgi:app
"""

from __future__ import annotations

import os

from app import create_app

app = create_app()
application = app


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    try:
        from waitress import serve  # type: ignore[import-untyped]

        serve(app, host=host, port=port)
    except ImportError:
        app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
