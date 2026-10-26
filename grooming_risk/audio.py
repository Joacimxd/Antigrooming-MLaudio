"""Carga, normalización, segmentación y aumentación de audio."""
from __future__ import annotations

import numpy as np

from .config import MAX_SEGMENT_SEC, MIN_SEGMENT_SEC, SAMPLE_RATE, VAD_TOP_DB


def load_audio(path, sr: int = SAMPLE_RATE, max_sec: float | None = None) -> np.ndarray:
    """Carga cualquier formato (wav/flac/mp3/ogg/m4a vía ffmpeg) a mono float32 16 kHz."""
    import librosa

    wav, _ = librosa.load(str(path), sr=sr, mono=True, duration=max_sec)
    return normalize(wav)


def normalize(wav: np.ndarray, target_rms: float = 0.05) -> np.ndarray:
    """Normaliza volumen (RMS) para que el modelo no aprenda la ganancia de cada corpus."""
    wav = np.asarray(wav, dtype=np.float32)
    wav = wav - wav.mean() if wav.size else wav
    rms = float(np.sqrt(np.mean(wav ** 2))) if wav.size else 0.0
    if rms > 1e-6:
        wav = wav * (target_rms / rms)
    return np.clip(wav, -1.0, 1.0).astype(np.float32)


def trim_silence(wav: np.ndarray, top_db: int = VAD_TOP_DB) -> np.ndarray:
    import librosa

    intervals = librosa.effects.split(wav, top_db=top_db)
    if len(intervals) == 0:
        return wav
    return np.concatenate([wav[s:e] for s, e in intervals])


def speech_regions(wav: np.ndarray, sr: int = SAMPLE_RATE, top_db: int = VAD_TOP_DB,
                   min_gap_sec: float = 0.3) -> list[tuple[int, int]]:
    """VAD por energía: devuelve regiones (inicio, fin) en muestras, fusionando huecos cortos."""
    import librosa

    intervals = librosa.effects.split(wav, top_db=top_db)
    merged: list[list[int]] = []
    for s, e in intervals:
        if merged and s - merged[-1][1] < min_gap_sec * sr:
            merged[-1][1] = int(e)
        else:
            merged.append([int(s), int(e)])
    return [(s, e) for s, e in merged if e - s >= 0.3 * sr]


def chunk(wav: np.ndarray, sr: int = SAMPLE_RATE, max_sec: float = MAX_SEGMENT_SEC,
          min_sec: float = MIN_SEGMENT_SEC) -> list[np.ndarray]:
    """Divide un audio largo en trozos de <= max_sec, descartando restos < min_sec."""
    step = int(max_sec * sr)
    pieces = [wav[i:i + step] for i in range(0, len(wav), step)]
    return [p for p in pieces if len(p) >= min_sec * sr]


def augment(wav: np.ndarray, rng: np.random.Generator, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Aumentación orientada a romper el sesgo de canal entre corpus.

    Como cada dataset tiene micrófonos/códecs distintos, el modelo podría aprender
    "de qué corpus viene el audio" en lugar de "qué edad tiene el hablante".
    Simulamos canal telefónico, ruido y cambios de ganancia. NO se usa pitch-shift
    porque alteraría justo la señal de edad (F0/formantes).
    """
    import librosa
    from scipy.signal import butter, sosfilt

    out = wav.copy()
    if rng.random() < 0.5:  # canal telefónico: banda 300–3400 Hz pasando por 8 kHz
        sos = butter(4, [300, 3400], btype="band", fs=sr, output="sos")
        out = sosfilt(sos, out).astype(np.float32)
        out = librosa.resample(librosa.resample(out, orig_sr=sr, target_sr=8000),
                               orig_sr=8000, target_sr=sr)
    if rng.random() < 0.7:  # ruido blanco a SNR 10–30 dB
        snr_db = rng.uniform(10, 30)
        p_sig = np.mean(out ** 2) + 1e-10
        noise = rng.normal(0, np.sqrt(p_sig / 10 ** (snr_db / 10)), size=out.shape)
        out = out + noise.astype(np.float32)
    if rng.random() < 0.3:  # recorte suave (micrófonos baratos)
        out = np.tanh(out * rng.uniform(2, 6)).astype(np.float32)
    return normalize(out)
