"""Pipeline de inferencia de punta a punta: audio → hablantes → edad → riesgo."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from .audio import chunk, load_audio
from .config import CLASSES, DISCLAIMER, MODELS_DIR, SAMPLE_RATE
from .diarization import diarize
from .features import EcapaExtractor, build_extractor
from .risk import compute_risk

BUNDLE_NAME = "best_model.joblib"
CARD_NAME = "model_card.json"


def save_bundle(pipeline, extractor_name: str, metrics: dict, model_dir: Path = MODELS_DIR,
                extra: dict | None = None) -> Path:
    """Guarda el clasificador + qué extractor necesita, para que la API sea agnóstica."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    card = {"extractor": extractor_name, "classes": CLASSES, "metrics": metrics, **(extra or {})}
    joblib.dump({"pipeline": pipeline, **card}, model_dir / BUNDLE_NAME)
    (model_dir / CARD_NAME).write_text(json.dumps(card, indent=2, ensure_ascii=False, default=str))
    return model_dir / BUNDLE_NAME


def reorder_proba(proba: np.ndarray, model_classes) -> np.ndarray:
    """Lleva las columnas de predict_proba al orden canónico CLASSES (0 si falta una clase)."""
    model_classes = list(model_classes)
    out = np.zeros((proba.shape[0], len(CLASSES)), dtype=float)
    for k, c in enumerate(CLASSES):
        if c in model_classes:
            out[:, k] = proba[:, model_classes.index(c)]
    return out


class AgeClassifier:
    """Clasificador de grupo de edad para un fragmento de voz de un solo hablante."""

    def __init__(self, model_dir: Path = MODELS_DIR, device: str | None = None):
        bundle = joblib.load(Path(model_dir) / BUNDLE_NAME)
        self.pipeline = bundle["pipeline"]
        self.extractor_name = bundle["extractor"]
        self.card = {k: v for k, v in bundle.items() if k != "pipeline"}
        self.extractor = build_extractor(self.extractor_name, device)

    def predict_proba(self, wavs: list[np.ndarray]) -> np.ndarray:
        X = np.stack([self.extractor.extract(w) for w in wavs])
        return reorder_proba(self.pipeline.predict_proba(X), self.pipeline.classes_)


class GroomingRiskAnalyzer:
    def __init__(self, model_dir: Path = MODELS_DIR, device: str | None = None):
        self.age = AgeClassifier(model_dir, device)
        ext = self.age.extractor
        self.embedder = ext if isinstance(ext, EcapaExtractor) else EcapaExtractor(device="cpu")

    def analyze(self, wav: np.ndarray, num_speakers: int | None = None) -> dict:
        speakers = diarize(wav, self.embedder, num_speakers=num_speakers)
        results, probs_list = [], []
        for sp in speakers:
            pieces = chunk(sp.audio(wav))
            if not pieces:
                continue
            probs = self.age.predict_proba(pieces)
            weights = np.array([len(p) for p in pieces], dtype=float)
            p = (probs * weights[:, None]).sum(0) / weights.sum()
            p_dict = {c: round(float(v), 4) for c, v in zip(CLASSES, p)}
            probs_list.append(p_dict)
            results.append({
                "speaker": sp.id,
                "duration_sec": round(sp.duration, 2),
                "segments_sec": [[round(s / SAMPLE_RATE, 2), round(e / SAMPLE_RATE, 2)]
                                 for s, e in sp.segments],
                "probabilities": p_dict,
                "predicted_age_group": CLASSES[int(np.argmax(p))],
            })
        risk = compute_risk(probs_list)
        # compute_risk indexa sobre los hablantes con audio suficiente → traducir a su id.
        for key in ("minor_speaker", "adult_speaker"):
            if risk[key] is not None:
                risk[key] = results[risk[key]]["speaker"]
        return {
            "risk_score": risk["score"],
            "risk_percent": round(100 * risk["score"], 1),
            "risk_level": risk["level"],
            "explanation": risk["reason"],
            "minor_speaker": risk["minor_speaker"],
            "adult_speaker": risk["adult_speaker"],
            "num_speakers": len(results),
            "speakers": results,
            "audio_duration_sec": round(len(wav) / SAMPLE_RATE, 2),
            "model": {"extractor": self.age.extractor_name,
                      "metrics": self.age.card.get("metrics", {})},
            "disclaimer": DISCLAIMER,
        }

    def analyze_file(self, path, num_speakers: int | None = None) -> dict:
        return self.analyze(load_audio(path), num_speakers=num_speakers)
