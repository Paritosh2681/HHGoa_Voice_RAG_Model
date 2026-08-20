FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and assets
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY assets/ ./assets/
COPY README.md .

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    HHGOA_HOST=0.0.0.0 \
    HHGOA_PORT=8000

EXPOSE 8000

CMD ["python", "-m", "backend.main"]
