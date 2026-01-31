# Multi-stage Dockerfile for Genesis Synthetic Data Platform
# Optimized for production with minimal image size

# Stage 1: Builder
FROM python:3.11-slim as builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install dependencies
COPY pyproject.toml setup.py ./
COPY genesis/version.py genesis/
RUN pip install --upgrade pip && \
    pip install .[api]

# Stage 2: Production image
FROM python:3.11-slim as production

LABEL maintainer="Genesis Team"
LABEL version="1.2.0"
LABEL description="Genesis Synthetic Data Generation Platform"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    GENESIS_ENV=production

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY genesis/ genesis/

# Create non-root user
RUN useradd --create-home --shell /bin/bash genesis && \
    chown -R genesis:genesis /app
USER genesis

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: start API server
CMD ["uvicorn", "genesis.api.server:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 3: Development image (optional build target)
FROM production as development

USER root

# Install development dependencies
RUN pip install pytest pytest-cov black ruff mypy

# Switch back to genesis user
USER genesis

# Development command with hot reload
CMD ["uvicorn", "genesis.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage 4: GPU-enabled image (optional build target)
FROM nvidia/cuda:11.8-runtime-ubuntu22.04 as gpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    GENESIS_ENV=production

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/bin/python

# Copy virtual environment and install GPU deps
COPY --from=builder /opt/venv /opt/venv
RUN pip install torch --index-url https://download.pytorch.org/whl/cu118

COPY genesis/ genesis/

RUN useradd --create-home --shell /bin/bash genesis && \
    chown -R genesis:genesis /app
USER genesis

EXPOSE 8000

CMD ["uvicorn", "genesis.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
