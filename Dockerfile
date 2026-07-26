# ─────────────────────────────────────────────────────────────
# Deepfake Detection System — Hugging Face Spaces Dockerfile
# ─────────────────────────────────────────────────────────────
# HF Spaces exposes port 7860. We bind Gunicorn there.
# Free CPU Basic Space = 2 vCPU, 16 GB RAM → TF loads fine.
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System dependencies for OpenCV + TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for Docker layer caching
COPY backend/requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Create required directories
RUN mkdir -p backend/uploads/images backend/uploads/videos database

# HF Spaces runs as non-root user (id 1000); ensure dirs are writable
RUN chmod -R 777 backend/uploads database

# Expose HF Spaces port
EXPOSE 7860

# Environment defaults (override via HF Space secrets)
ENV FLASK_CONFIG=production
ENV PORT=7860

# Start Gunicorn — 2 workers keeps RAM under 16 GB with TensorFlow
# Timeout 120s for video analysis (can take 60-90s per video)
CMD ["gunicorn", \
     "--chdir", "backend", \
     "--bind", "0.0.0.0:7860", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "120", \
     "--keep-alive", "5", \
     "wsgi:app"]
