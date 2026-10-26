"""Configuración central compartida por el notebook, la API y los tests."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("GR_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = Path(os.getenv("GR_MODELS_DIR", PROJECT_ROOT / "models"))
PRETRAINED_DIR = Path(os.getenv("GR_PRETRAINED_DIR", PROJECT_ROOT / "pretrained"))

SAMPLE_RATE = 16_000

# Segmentos usados para entrenar/inferir la edad.
MIN_SEGMENT_SEC = 1.0
MAX_SEGMENT_SEC = 6.0

# Grupos de edad (clases del modelo).
CLASSES = ["nino", "adolescente", "adulto"]
CHILD_MAX_AGE = 12        # nino: <= 12 años
TEEN_MAX_AGE = 17         # adolescente: 13–17 años, adulto: >= 18


def age_to_class(age: float | int | None) -> str | None:
    if age is None:
        return None
    try:
        age = float(age)
    except (TypeError, ValueError):
        return None
    if age != age or age <= 0:  # NaN o inválido
        return None
    if age <= CHILD_MAX_AGE:
        return "nino"
    if age <= TEEN_MAX_AGE:
        return "adolescente"
    return "adulto"


# --- Reglas de riesgo -------------------------------------------------------
# Riesgo base cuando, con certeza total, un hablante de la clase menor convive
# en el mismo audio con un hablante adulto distinto.
RISK_WEIGHTS = {
    ("nino", "adulto"): 0.90,
    ("adolescente", "adulto"): 0.60,
}
RISK_FLOOR = 0.05  # riesgo mínimo reportado (nunca se afirma "riesgo cero")
RISK_LEVELS = [(0.30, "bajo"), (0.60, "medio"), (1.01, "alto")]

# --- Modelos preentrenados --------------------------------------------------
ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
WAVLM_SOURCE = "microsoft/wavlm-base-plus"
# Capas intermedias de WavLM: concentran información de hablante/paralingüística.
WAVLM_LAYERS = (3, 4, 5, 6, 7, 8)
AUDEERING_SOURCE = "audeering/wav2vec2-large-robust-24-ft-age-gender"

# --- Diarización ligera -----------------------------------------------------
DIAR_WINDOW_SEC = 1.5
DIAR_HOP_SEC = 0.75
DIAR_COSINE_THRESHOLD = 0.55  # distancia coseno para separar hablantes
VAD_TOP_DB = 30

DISCLAIMER = (
    "Este resultado es solo una ALERTA / estimación probabilística de riesgo basada "
    "en la edad aparente de las voces. NO determina que exista grooming ni sustituye "
    "la revisión humana especializada. Muchas conversaciones legítimas (familia, "
    "docentes, médicos) entre menores y adultos generarán alertas."
)
