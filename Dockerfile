# ── Stage 1: Build ───────────────────────────────────────────────────────
FROM python:3.10-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set up a new user 'user' with UID 1000
RUN useradd -m -u 1000 user
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files and ensure the new user owns them
COPY --chown=user . .

# Switch to the non-root user
USER user

# ── Healthcheck ──────────────────────────────────────────────────────────
# Updated to port 7860 to match HF requirements
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Expose & Run ─────────────────────────────────────────────────────────
# Hugging Face Spaces listens on port 7860
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]