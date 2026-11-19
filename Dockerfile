FROM python:3.11-slim

LABEL maintainer="AI Security & ML Robustness Lab" \
      description="Adversarial ML Evasion & Layered Robustness Laboratory for Tabular IDS Classifiers"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Install minimal system runtime libraries
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (prefer lightweight CPU wheel for PyTorch in container builds)
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.1.0' \
    && pip install -r requirements.txt

# Copy source code and install package
COPY . .
RUN pip install --no-deps -e . \
    && useradd --create-home --shell /bin/bash labuser \
    && mkdir -p /app/results /app/checkpoints \
    && chown -R labuser:labuser /app

USER labuser

ENTRYPOINT ["python", "main.py"]
CMD ["benchmark"]
