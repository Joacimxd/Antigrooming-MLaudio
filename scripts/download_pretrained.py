"""Descarga los modelos preentrenados que necesita el modelo guardado en models/.

Se ejecuta en el `docker build` para que el contenedor arranque sin internet.
    python scripts/download_pretrained.py
"""
import json

from grooming_risk.config import MODELS_DIR
from grooming_risk.features import EcapaExtractor, build_extractor

card = json.loads((MODELS_DIR / "model_card.json").read_text())
print(f"Extractor del mejor modelo: {card['extractor']}")
build_extractor(card["extractor"], device="cpu")
EcapaExtractor(device="cpu")  # siempre necesario para la diarización
print("Modelos preentrenados descargados.")
