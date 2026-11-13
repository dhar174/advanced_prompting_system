# Multi-stage production container for Advanced Prompting System (Cloud Run compliant)

FROM node:22-alpine AS frontend-builder
WORKDIR /frontend
COPY collaborative-assistant-frontend/package.json collaborative-assistant-frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY collaborative-assistant-frontend/ ./
RUN npm run build

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    HOST=0.0.0.0 \
    APS_ENABLE_GRAPHIQL=0

WORKDIR /app

# Install only runtime package + production WSGI server (no test/dev extras)
COPY pyproject.toml README.md ./
COPY advanced_prompting.py app.py assistant_personalities.py complexity_measures.py \
     conversation_manager.py execution_modes.py feedback_manager.py function_definitions.py \
     output_generator.py provider_client.py schema.py wsgi.py ./
COPY ml_complexity_model.pkl ./
COPY --from=frontend-builder /frontend/dist ./collaborative-assistant-frontend/dist

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[server]" \
    && useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/logs /tmp \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 8 --timeout 120 wsgi:app"]
