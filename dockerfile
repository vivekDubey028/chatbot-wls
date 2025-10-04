# Multi-stage build
FROM python:3.12-slim AS deps

# Install build deps (for compiling packages like faiss-cpu, torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Upgrade pip, setuptools, and wheel first to avoid backend issues with Python 3.12
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and data
COPY . .

# Run preprocessing to generate index (idempotent: skips if files exist)
RUN python preprocess-data.py

# Runtime stage (minimal image)
FROM python:3.12-slim AS runtime

# Install minimal runtime deps (e.g., for gunicorn, no build tools needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy installed deps from deps stage
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy app files (code, data, templates) from deps stage
COPY --from=deps /app /app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5000

# Health check (assumes /health endpoint in petshop-chatbot.py; add if missing)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run with Gunicorn for prod (workers=4; adjust for CPU; timeout=120s for AI inference)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "--log-level", "info", "petshop-chatbot:app"]