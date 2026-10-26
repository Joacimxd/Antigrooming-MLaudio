"""API FastAPI del estimador de riesgo.

Sirve el modelo que el notebook guardó como `models/best_model.joblib` (el de
mejor desempeño en test). La API es agnóstica al extractor: lee del bundle si
el ganador fue handcrafted, ECAPA o WavLM y carga lo necesario.

Toda respuesta incluye un `disclaimer`: el resultado es una ALERTA, no un veredicto.
"""
from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from grooming_risk.audio import load_audio
from grooming_risk.config import DISCLAIMER, MODELS_DIR, SAMPLE_RATE
from grooming_risk.pipeline import GroomingRiskAnalyzer

MAX_UPLOAD_MB = float(os.getenv("GR_MAX_UPLOAD_MB", "50"))
MAX_AUDIO_SEC = float(os.getenv("GR_MAX_AUDIO_SEC", "600"))
ALLOWED_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".webm"}

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["analyzer"] = GroomingRiskAnalyzer(MODELS_DIR, device=os.getenv("GR_DEVICE", "cpu"))
    yield
    state.clear()


app = FastAPI(
    title="Estimador de riesgo de grooming por edad de voz",
    description="Estima la probabilidad de interacción menor–adulto en un audio. "
                "**Es solo una alerta de apoyo, no un decisor definitivo.** " + DISCLAIMER,
    version="1.0.0",
    lifespan=lifespan,
)


class SpeakerOut(BaseModel):
    speaker: int
    duration_sec: float
    segments_sec: list[list[float]]
    probabilities: dict[str, float]
    predicted_age_group: str


class AnalysisOut(BaseModel):
    risk_score: float = Field(..., ge=0, le=1)
    risk_percent: float
    risk_level: str
    explanation: str
    minor_speaker: int | None
    adult_speaker: int | None
    num_speakers: int
    speakers: list[SpeakerOut]
    audio_duration_sec: float
    model: dict
    disclaimer: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "analyzer" in state}


@app.get("/model-info")
def model_info():
    analyzer: GroomingRiskAnalyzer = state["analyzer"]
    return {**analyzer.age.card, "disclaimer": DISCLAIMER}


@app.post("/analyze", response_model=AnalysisOut)
async def analyze(
    file: UploadFile = File(..., description="Audio de la conversación"),
    num_speakers: int | None = Query(None, ge=1, le=10,
                                     description="Número de hablantes si se conoce"),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(415, f"Formato no soportado: {suffix or '?'}. Usa {sorted(ALLOWED_EXT)}")
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Archivo mayor a {MAX_UPLOAD_MB} MB")

    # El audio se escribe en un temporal y se borra al terminar: no se guarda nada.
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        try:
            wav = await run_in_threadpool(load_audio, tmp.name, SAMPLE_RATE, MAX_AUDIO_SEC)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"No se pudo decodificar el audio: {exc}") from exc

    if len(wav) < SAMPLE_RATE:
        raise HTTPException(422, "El audio dura menos de 1 segundo")
    analyzer: GroomingRiskAnalyzer = state["analyzer"]
    return await run_in_threadpool(analyzer.analyze, wav, num_speakers)
