FROM python:3.11-slim

WORKDIR /app

# System deps for faiss-cpu, pymupdf, lingua
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.prod.txt .
RUN pip install --no-cache-dir -r requirements.prod.txt

# Copy application source
COPY src/ ./src/
COPY scripts/start.sh ./start.sh
RUN chmod +x start.sh

EXPOSE 8000

ENTRYPOINT ["./start.sh"]
