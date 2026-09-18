FROM python:3.11-slim

WORKDIR /app

# Install Coin-OR CBC solver for linear programming
RUN apt-get update && apt-get install -y --no-install-recommends \
    coinor-cbc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with layer caching and resilient network timeout
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy application source code
COPY app/ ./app/

# Create a non-root user for container security hardening
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser \
    && chown -R appuser:appuser /app

# Set production runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0

EXPOSE 8000

USER appuser

# Automated container healthcheck verifying /health readiness
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
