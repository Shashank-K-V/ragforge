# ======================================================================= #
#  RAGForge — Multi-stage Dockerfile                                        #
#                                                                           #
#  Stage 1 (builder): install Python deps into a virtual environment.       #
#  Stage 2 (runtime): copy only the venv + app code — no build tools.      #
#                                                                           #
#  Compatible with:                                                         #
#   • docker-compose (local dev)                                            #
#   • Hugging Face Spaces (Docker SDK — exposes port 7860)                  #
#   • Railway / Render (detects Dockerfile automatically)                   #
# ======================================================================= #

# --------------------------------------------------------------------------
# Stage 1 — builder
# --------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Prevent .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install system build deps (needed by some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libffi-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment in a predictable location
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies first (layer-cached until requirements change)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# --------------------------------------------------------------------------
# Stage 2 — runtime
# --------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    # HuggingFace model cache inside the container image layer
    HF_HOME="/app/.cache/huggingface" \
    TRANSFORMERS_CACHE="/app/.cache/huggingface"

# Only the runtime C libs needed (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash ragforge
WORKDIR /app
RUN chown ragforge:ragforge /app

# Copy application source
COPY --chown=ragforge:ragforge app/ ./app/
COPY --chown=ragforge:ragforge docs/ ./docs/

# Create writable directories for uploads and vector store
RUN mkdir -p /app/chroma_db /app/uploaded_docs /app/.cache/huggingface && \
    chown -R ragforge:ragforge /app

USER ragforge

# HuggingFace Spaces uses port 7860; Railway/Render use $PORT
EXPOSE 7860

# Health check — waits 30 s for the embedding model to load, then polls
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Uvicorn with 2 workers — single worker on HF Spaces free tier (1 vCPU)
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--log-level", "info"]
