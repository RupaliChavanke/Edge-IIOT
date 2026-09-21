FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    librdkafka-dev \
    && rm -rf /var/lib/apt/lists/*

# Configure robust pip environment settings to prevent network timeouts
ENV PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=10 \
    PYTHONUNBUFFERED=1

# Upgrade pip and install lightweight CPU PyTorch directly
# This avoids downloading 1.6GB+ of CUDA libraries (nvidia-cudnn, nccl, etc.) that cause timeouts
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch

# Copy requirements and install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy application source
COPY . .

# Expose Streamlit port
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

