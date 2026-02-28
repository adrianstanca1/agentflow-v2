FROM python:3.12-slim

WORKDIR /app

# System deps: curl for healthcheck, docker CLI for Docker integration
RUN apt-get update && apt-get install -y \
    curl gcc git ca-certificates gnupg \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && chmod a+r /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian bookworm stable" \
       > /etc/apt/sources.list.d/docker.list \
    && apt-get update && apt-get install -y docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Frontend (pre-built)
COPY frontend/dist ./frontend/dist

# App code
COPY server.py auth_providers.py github_integration.py docker_integration.py ./
COPY openclaw ./openclaw

# Copy .env.example as template (actual .env mounted at runtime)
COPY .env.example .env.example

RUN mkdir -p /workspace

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python3", "server.py"]
