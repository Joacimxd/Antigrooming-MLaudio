"""Carga, normalización, segmentación y aumentación de audio."""
from __future__ import annotations

import numpy as np

from .config import DATA_DIR, MAX_SEGMENT_SEC, MIN_SEGMENT_SEC, SAMPLE_RATE, VAD_TOP_DB

NOISE_DIR = DATA_DIR / "noise"  # wav/mp3/flac opcionales: risas, calle, cortacésped...


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


def _mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    n = min(len(signal), len(noise))
    signal, noise = signal[:n], noise[:n]
    p_sig = float(np.mean(signal ** 2)) + 1e-10
    p_noi = float(np.mean(noise ** 2)) + 1e-10
    noise = noise * np.sqrt(p_sig / (p_noi * 10 ** (snr_db / 10)))
    return (signal + noise).astype(np.float32)


def _fit_length(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(x) >= n:
        start = int(rng.integers(0, len(x) - n + 1))
        return x[start:start + n]
    reps = int(np.ceil(n / max(len(x), 1)))
    return np.tile(x, reps)[:n]


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.normal(0, 1, n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    spec *= 1.0 / np.maximum(freqs, 1.0 / n) ** 0.5
    out = np.fft.irfft(spec, n=n)
    return (out / (np.max(np.abs(out)) + 1e-10)).astype(np.float32)


def _brown_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    out = np.cumsum(rng.normal(0, 1, n))
    out -= out.mean()
    return (out / (np.max(np.abs(out)) + 1e-10)).astype(np.float32)


def _load_background(n: int, rng: np.random.Generator, sr: int) -> np.ndarray | None:
    """Mezcla un archivo real de datasets/noise/ (OST, SFX, ventilador...)."""
    if not NOISE_DIR.is_dir():
        return None
    files = [p for p in NOISE_DIR.iterdir()
             if p.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg", ".m4a"}]
    if not files:
        return None
    import librosa

    path = files[int(rng.integers(len(files)))]
    clip, _ = librosa.load(str(path), sr=sr, mono=True)
    if clip.size < sr // 4:
        return None
    return _fit_length(clip.astype(np.float32), n, rng)


def _fondo_sintetico(n: int, rng: np.random.Generator, sr: int) -> np.ndarray:
    """Ambiente de partida: rumble del mundo o ventilador de PC. Sin voces."""
    from scipy.signal import butter, sosfilt

    t = np.arange(n) / sr
    if rng.random() < 0.5:
        base = _brown_noise(n, rng)
        sos = butter(2, 400, btype="low", fs=sr, output="sos")
        env = 0.55 + 0.45 * np.sin(2 * np.pi * rng.uniform(0.4, 1.8) * t)
        return sosfilt(sos, base * env).astype(np.float32)
    fan = _pink_noise(n, rng)
    sos = butter(2, [180, 900], btype="band", fs=sr, output="sos")
    return sosfilt(sos, fan).astype(np.float32)


def _cama_musical(n: int, rng: np.random.Generator, sr: int) -> np.ndarray:
    """BGM sintético: pad con tono. La música de fondo es lo que más confunde el F0."""
    t = np.arange(n) / sr
    root = float(rng.choice([110.0, 130.81, 146.83, 164.81, 174.61, 196.0]))
    pad = np.zeros(n, dtype=np.float32)
    for ratio, w in ((1.0, 0.45), (1.25, 0.28), (1.5, 0.22), (2.0, 0.15)):
        f = root * ratio * float(rng.uniform(0.99, 1.01))
        pad += (w * np.sin(2 * np.pi * f * t)).astype(np.float32)
    lfo = 0.65 + 0.35 * np.sin(2 * np.pi * rng.uniform(0.15, 0.8) * t)
    return (pad * lfo).astype(np.float32)


def _sfx_juego(n: int, rng: np.random.Generator, sr: int) -> np.ndarray:
    """SFX de partida: pasos/disparos rítmicos y beeps de UI. Tampoco es habla."""
    from scipy.signal import butter, sosfilt

    out = np.zeros(n, dtype=np.float32)
    step = max(1, int(sr * 60.0 / rng.uniform(80, 140)))
    hit_n = max(8, int(0.035 * sr))
    hit = rng.normal(0, 1, hit_n).astype(np.float32) * np.linspace(1, 0, hit_n, dtype=np.float32)
    hit = sosfilt(butter(1, 900, btype="low", fs=sr, output="sos"), hit).astype(np.float32)
    for i in range(0, n - hit_n, step):
        if rng.random() < 0.7:
            out[i:i + hit_n] += hit * rng.uniform(0.25, 0.9)
    for _ in range(int(rng.integers(1, 4))):
        f = float(rng.choice([523.25, 659.25, 783.99, 1046.5]))
        dur = int(rng.uniform(0.04, 0.11) * sr)
        start = int(rng.integers(0, max(1, n - dur)))
        env = np.hanning(dur).astype(np.float32)
        out[start:start + dur] += 0.3 * np.sin(2 * np.pi * f * np.arange(dur) / sr) * env
    return out


def _f_telefono(out, rng, sr):
    import librosa
    from scipy.signal import butter, sosfilt

    sos = butter(4, [300, 3400], btype="band", fs=sr, output="sos")
    out = sosfilt(sos, out).astype(np.float32)
    return librosa.resample(librosa.resample(out, orig_sr=sr, target_sr=8000),
                            orig_sr=8000, target_sr=sr)


def _f_ruido_blanco(out, rng, sr):
    noise = rng.normal(0, 1, size=out.shape).astype(np.float32)
    return _mix_at_snr(out, noise, rng.uniform(10, 30))


def _f_saturacion(out, rng, sr):
    return np.tanh(out * rng.uniform(2, 6)).astype(np.float32)


def _f_ruido_rosa(out, rng, sr):
    return _mix_at_snr(out, _pink_noise(len(out), rng), rng.uniform(12, 28))


def _f_huecos(out, rng, sr):
    frac = rng.uniform(0.02, 0.08)
    width = max(1, int(frac * len(out)))
    start = int(rng.integers(0, max(1, len(out) - width)))
    fade = max(1, int(0.01 * sr))
    hole = out[start:start + width].copy()
    ramp = np.linspace(0, 1, min(fade, max(1, len(hole) // 2)), dtype=np.float32)
    hole[:len(ramp)] *= ramp[::-1]
    hole[-len(ramp):] *= ramp
    if len(hole) > 2 * len(ramp):
        hole[len(ramp):len(hole) - len(ramp)] = 0
    out = out.copy()
    out[start:start + width] = hole
    return out


def _f_hum(out, rng, sr):
    t = np.arange(len(out)) / sr
    f0 = 50.0 if rng.random() < 0.5 else 60.0
    hum = np.sin(2 * np.pi * f0 * t) + 0.35 * np.sin(2 * np.pi * 2 * f0 * t)
    return _mix_at_snr(out, hum.astype(np.float32), rng.uniform(15, 30))


def _f_clicks(out, rng, sr):
    clicks = np.zeros(len(out), dtype=np.float32)
    n_clicks = int(rng.integers(3, 18))
    idx = rng.integers(0, len(out), size=n_clicks)
    clicks[idx] = rng.uniform(-0.9, 0.9, size=n_clicks)
    return (out + 0.25 * clicks).astype(np.float32)


def _f_eco(out, rng, sr):
    delay = int(rng.uniform(0.06, 0.18) * sr)
    decay = rng.uniform(0.22, 0.45)
    echo = np.zeros_like(out)
    if delay < len(out):
        echo[delay:] = out[:-delay] * decay
        out = (out + echo).astype(np.float32)
    return out


def _f_bitcrush(out, rng, sr):
    bits = int(rng.integers(5, 9))
    levels = float(2 ** (bits - 1))
    return (np.round(out * levels) / levels).astype(np.float32)


def _f_ambiente(out, rng, sr):
    fondo = _load_background(len(out), rng, sr)
    if fondo is None:
        fondo = _fondo_sintetico(len(out), rng, sr)
    return _mix_at_snr(out, fondo, rng.uniform(8, 22))


def _f_cama_musical(out, rng, sr):
    return _mix_at_snr(out, _cama_musical(len(out), rng, sr), rng.uniform(8, 20))


def _f_sfx_juego(out, rng, sr):
    return _mix_at_snr(out, _sfx_juego(len(out), rng, sr), rng.uniform(10, 24))


FILTROS = {
    "telefono": _f_telefono,
    "ruido_blanco": _f_ruido_blanco,
    "saturacion": _f_saturacion,
    "ruido_rosa": _f_ruido_rosa,
    "huecos": _f_huecos,
    "hum": _f_hum,
    "clicks": _f_clicks,
    "eco": _f_eco,
    "bitcrush": _f_bitcrush,
    "ambiente": _f_ambiente,
    "cama_musical": _f_cama_musical,
    "sfx_juego": _f_sfx_juego,
}

FILTRO_P = {
    "telefono": 0.5, "ruido_blanco": 0.7, "saturacion": 0.3, "ruido_rosa": 0.4,
    "huecos": 0.3, "hum": 0.25, "clicks": 0.25, "eco": 0.3, "bitcrush": 0.25,
    "ambiente": 0.5, "cama_musical": 0.5, "sfx_juego": 0.4,
}


def aplicar_filtro(wav: np.ndarray, nombre: str, rng: np.random.Generator,
                   sr: int = SAMPLE_RATE) -> np.ndarray:
    """Aplica un solo filtro (para experimentos). No renormaliza, para ver el RMS real."""
    if nombre not in FILTROS:
        raise KeyError(f"Filtro desconocido: {nombre}. Opciones: {list(FILTROS)}")
    out = np.asarray(wav, dtype=np.float32).copy()
    if out.size == 0:
        return out
    return FILTROS[nombre](out, rng, sr)


def augment(wav: np.ndarray, rng: np.random.Generator, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Aumentación para voz-en-juego: canal sucio + cama de partida. Sin pitch-shift."""
    out = np.asarray(wav, dtype=np.float32).copy()
    if out.size == 0:
        return out
    for nombre, p in FILTRO_P.items():
        if rng.random() < p:
            out = FILTROS[nombre](out, rng, sr)
    return normalize(out)
