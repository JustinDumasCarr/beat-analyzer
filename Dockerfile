# Beat Analyzer - Audio Analysis API
# Multi-stage build for smaller final image

# =============================================================================
# Stage 1: Builder - Install dependencies and compile native extensions
# =============================================================================
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python build dependencies first
RUN pip install --no-cache-dir --upgrade pip wheel cython

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 2: Runtime - Minimal image with only runtime dependencies
# =============================================================================
FROM python:3.11-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser analyze.py server.py db.py ./

# Create directories for data persistence
RUN mkdir -p /data/db /data/uploads /home/appuser/.cache && \
    chown -R appuser:appuser /data /home/appuser/.cache

# Switch to non-root user
USER appuser

# Environment variables
ENV DATABASE_PATH=/data/db/beat_analyzer.db \
    UPLOAD_DIR=/data/uploads \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default command - run API server
CMD ["python", "analyze.py", "--serve", "--host", "0.0.0.0", "--port", "8000"]
