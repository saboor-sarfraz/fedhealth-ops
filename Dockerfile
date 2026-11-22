FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install a CPU-only PyTorch wheel instead of the much larger CUDA build.
RUN pip install --no-cache-dir --timeout 120 --retries 5 \
    --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.9.1+cpu"
RUN pip install --no-cache-dir --timeout 120 --retries 5 -r requirements.txt

COPY models/ ./models/
COPY hospital_clients/ ./hospital_clients/
COPY api/ ./api/

RUN mkdir -p /app/data /app/outputs

EXPOSE 8080 8000
