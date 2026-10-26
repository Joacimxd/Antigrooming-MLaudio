# Imagen de inferencia del estimador de riesgo (CPU).
# Requiere haber ejecutado el notebook: models/best_model.joblib y models/model_card.json
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GR_PRETRAINED_DIR=/app/pretrained \
    HF_HOME=/app/pretrained/hf \
    GR_DEVICE=cpu \
    PYTHONPATH=/app

# ffmpeg: decodificar mp3/m4a/ogg/webm. libsndfile: wav/flac.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch CPU-only: imagen ~3x más pequeña que con CUDA.
RUN pip install "torch>=2.3,<2.9" "torchaudio>=2.3,<2.9" \
    --index-url https://download.pytorch.org/whl/cpu
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY grooming_risk/ grooming_risk/
COPY scripts/ scripts/
COPY models/ models/
# Descarga en build los pesos de ECAPA/WavLM que use el modelo ganador.
RUN python scripts/download_pretrained.py

COPY api/ api/

RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
