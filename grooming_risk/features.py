"""Extractores de características. Todos comparten la interfaz:

    extractor.name          -> str
    extractor.extract(wav)  -> np.ndarray 1D (wav mono float32 a 16 kHz)

1. HandcraftedExtractor  – F0, formantes, MFCC y espectrales (inspirado en eGeMAPS).
2. EcapaExtractor        – embeddings de hablante ECAPA-TDNN (SpeechBrain, VoxCeleb).
3. WavLMExtractor        – representaciones autosupervisadas WavLM-base-plus (capas medias).
Además AudeeringAgeModel: modelo de edad ya entrenado, usado solo como referencia.
"""
from __future__ import annotations

import numpy as np

from .config import (AUDEERING_SOURCE, CHILD_MAX_AGE, ECAPA_SOURCE, MAX_SEGMENT_SEC,
                     PRETRAINED_DIR, SAMPLE_RATE, TEEN_MAX_AGE, WAVLM_LAYERS, WAVLM_SOURCE)


def best_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _crop(wav: np.ndarray, max_sec: float = MAX_SEGMENT_SEC) -> np.ndarray:
    n = int(max_sec * SAMPLE_RATE)
    if len(wav) <= n:
        return wav
    start = (len(wav) - n) // 2
    return wav[start:start + n]


# ---------------------------------------------------------------------------
# 1. Acústicas clásicas
# ---------------------------------------------------------------------------
class HandcraftedExtractor:
    """Correlatos acústicos directos de la edad:
    - F0 (tono): niños ~250–400 Hz, mujeres adultas ~200 Hz, hombres ~120 Hz.
    - Formantes F1–F3: el tracto vocal corto de un niño los desplaza hacia arriba.
    - MFCC + deltas: envolvente espectral (timbre) y su dinámica.
    - Espectrales y ZCR: brillo, planitud, ritmo de articulación.
    """

    name = "handcrafted"

    def __init__(self, n_mfcc: int = 20):
        self.n_mfcc = n_mfcc

    @staticmethod
    def _stats(x: np.ndarray) -> list[float]:
        x = x[np.isfinite(x)]
        if x.size == 0:
            return [0.0] * 5
        return [float(np.mean(x)), float(np.std(x)), float(np.median(x)),
                float(np.percentile(x, 10)), float(np.percentile(x, 90))]

    @staticmethod
    def _formants(frames: np.ndarray, sr: int, order: int = 12) -> np.ndarray:
        import librosa

        out = []
        window = np.hamming(frames.shape[0])
        for frame in frames.T:
            frame = np.append(frame[0], frame[1:] - 0.97 * frame[:-1]) * window
            if np.sum(frame ** 2) < 1e-8:
                continue
            try:
                a = librosa.lpc(frame, order=order)
            except Exception:  # noqa: BLE001 – frames degenerados
                continue
            roots = [r for r in np.roots(a) if np.imag(r) > 0]
            freqs = sorted(np.angle(roots) * sr / (2 * np.pi))
            freqs = [f for f in freqs if 90 < f < 5500]
            if len(freqs) >= 3:
                out.append(freqs[:3])
        return np.array(out) if out else np.full((1, 3), np.nan)

    def extract(self, wav: np.ndarray) -> np.ndarray:
        import librosa

        sr = SAMPLE_RATE
        wav = _crop(wav)
        hop, n_fft = 160, 512
        rms = librosa.feature.rms(y=wav, frame_length=n_fft, hop_length=hop)[0]
        voiced = rms > np.percentile(rms, 40)

        f0 = librosa.yin(wav, fmin=60, fmax=600, sr=sr, frame_length=1024, hop_length=hop)
        n = min(len(f0), len(voiced))
        f0_v = f0[:n][voiced[:n]]
        log_f0 = np.log(f0_v) if f0_v.size else np.array([np.nan])

        frames = librosa.util.frame(np.pad(wav, (0, n_fft)), frame_length=400, hop_length=hop)
        n = min(frames.shape[1], len(voiced))
        voiced_idx = np.where(voiced[:n])[0][::3]  # submuestreo por velocidad
        formants = self._formants(frames[:, voiced_idx], sr)

        mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=self.n_mfcc, n_fft=n_fft, hop_length=hop)
        d_mfcc = librosa.feature.delta(mfcc)
        spec = [
            librosa.feature.spectral_centroid(y=wav, sr=sr, n_fft=n_fft, hop_length=hop)[0],
            librosa.feature.spectral_bandwidth(y=wav, sr=sr, n_fft=n_fft, hop_length=hop)[0],
            librosa.feature.spectral_rolloff(y=wav, sr=sr, n_fft=n_fft, hop_length=hop)[0],
            librosa.feature.spectral_flatness(y=wav, n_fft=n_fft, hop_length=hop)[0],
            librosa.feature.zero_crossing_rate(wav, frame_length=n_fft, hop_length=hop)[0],
        ]
        contrast = librosa.feature.spectral_contrast(y=wav, sr=sr, n_fft=n_fft, hop_length=hop)

        feats: list[float] = []
        feats += self._stats(log_f0)
        for k in range(3):
            feats += self._stats(formants[:, k])
        feats += list(mfcc.mean(1)) + list(mfcc.std(1))
        feats += list(d_mfcc.mean(1)) + list(d_mfcc.std(1))
        for s in spec:
            feats += [float(np.mean(s)), float(np.std(s))]
        feats += list(contrast.mean(1))
        feats += [float(np.mean(voiced))]
        return np.nan_to_num(np.asarray(feats, dtype=np.float32))


# ---------------------------------------------------------------------------
# 2. ECAPA-TDNN (x-vectors modernos)
# ---------------------------------------------------------------------------
class EcapaExtractor:
    """Embedding de 192 dims entrenado para verificación de hablante en VoxCeleb.
    Codifica la identidad vocal (tamaño de tracto vocal, tono, timbre), que es
    justamente lo que cambia con la edad. También se usa para la diarización."""

    name = "ecapa"

    def __init__(self, device: str = "cpu"):
        from speechbrain.inference.speaker import EncoderClassifier

        self.device = device
        self.model = EncoderClassifier.from_hparams(
            source=ECAPA_SOURCE, savedir=str(PRETRAINED_DIR / "ecapa"),
            run_opts={"device": device})
        self.model.eval()

    def extract_batch(self, wavs: list[np.ndarray]) -> np.ndarray:
        import torch

        lens = [len(w) for w in wavs]
        batch = np.zeros((len(wavs), max(lens)), dtype=np.float32)
        for i, w in enumerate(wavs):
            batch[i, :len(w)] = w
        rel = torch.tensor([l / max(lens) for l in lens], dtype=torch.float32)
        with torch.no_grad():
            emb = self.model.encode_batch(torch.from_numpy(batch), wav_lens=rel)
        return emb.squeeze(1).cpu().numpy()

    def extract(self, wav: np.ndarray) -> np.ndarray:
        return self.extract_batch([_crop(wav)])[0]


# ---------------------------------------------------------------------------
# 3. WavLM (autosupervisado)
# ---------------------------------------------------------------------------
class WavLMExtractor:
    """WavLM-base-plus (94M params) preentrenado con 94k h de habla.
    Lidera SUPERB en tareas de hablante/paralingüística. Las capas intermedias
    (3–8) concentran información del hablante; las finales, contenido léxico.
    Usamos el promedio de esas capas con pooling media+desviación (1536 dims)."""

    name = "wavlm"

    def __init__(self, device: str | None = None, layers=WAVLM_LAYERS):
        import torch
        from transformers import AutoFeatureExtractor, WavLMModel

        self.device = device or best_device()
        self.layers = list(layers)
        cache = str(PRETRAINED_DIR / "hf")
        self.processor = AutoFeatureExtractor.from_pretrained(WAVLM_SOURCE, cache_dir=cache)
        self.model = WavLMModel.from_pretrained(WAVLM_SOURCE, cache_dir=cache).to(self.device)
        self.model.eval()
        self._torch = torch

    def extract(self, wav: np.ndarray) -> np.ndarray:
        torch = self._torch
        inputs = self.processor(_crop(wav), sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            out = self.model(inputs.input_values.to(self.device), output_hidden_states=True)
        hs = torch.stack([out.hidden_states[i] for i in self.layers]).mean(0)[0]  # (T, 768)
        return torch.cat([hs.mean(0), hs.std(0)]).cpu().numpy().astype(np.float32)


def build_extractor(name: str, device: str | None = None):
    if name == "handcrafted":
        return HandcraftedExtractor()
    if name == "ecapa":
        return EcapaExtractor(device="cpu" if device in (None, "mps") else device)
    if name == "wavlm":
        return WavLMExtractor(device=device)
    raise ValueError(f"Extractor desconocido: {name}")


# ---------------------------------------------------------------------------
# Referencia: modelo de edad preentrenado de audEERING (solo evaluación)
# ---------------------------------------------------------------------------
class AudeeringAgeModel:
    """wav2vec2-large-robust afinado para edad/género (aGender, Common Voice, TIMIT,
    VoxCeleb2). Sirve como cota de referencia "zero-shot": ¿cuánto ganamos entrenando
    con nuestros datos? Licencia CC BY-NC-SA 4.0 → no se despliega en la API."""

    name = "audeering_zero_shot"

    def __init__(self, device: str | None = None, sigma_years: float = 6.0):
        import torch
        from torch import nn
        from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel, Wav2Vec2Processor

        class ModelHead(nn.Module):
            def __init__(self, config, num_labels):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, num_labels)

            def forward(self, x):
                x = torch.tanh(self.dense(self.dropout(x)))
                return self.out_proj(self.dropout(x))

        class AgeGenderModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.wav2vec2 = Wav2Vec2Model(config)
                self.age = ModelHead(config, 1)
                self.gender = ModelHead(config, 3)
                self.post_init()

            def forward(self, input_values):
                hidden = self.wav2vec2(input_values)[0].mean(dim=1)
                return self.age(hidden), torch.softmax(self.gender(hidden), dim=1)

        self.device = device or best_device()
        cache = str(PRETRAINED_DIR / "hf")
        self.processor = Wav2Vec2Processor.from_pretrained(AUDEERING_SOURCE, cache_dir=cache)
        self.model = AgeGenderModel.from_pretrained(AUDEERING_SOURCE, cache_dir=cache).to(self.device)
        self.model.eval()
        self.sigma = sigma_years
        self._torch = torch

    def predict_age(self, wav: np.ndarray) -> float:
        torch = self._torch
        x = self.processor(_crop(wav), sampling_rate=SAMPLE_RATE).input_values[0]
        x = torch.tensor(x, dtype=torch.float32)[None].to(self.device)
        with torch.no_grad():
            age, _ = self.model(x)
        return float(age.item() * 100)

    def predict_proba(self, wav: np.ndarray) -> np.ndarray:
        """Convierte la edad puntual en probabilidades de clase con incertidumbre gaussiana."""
        from scipy.stats import norm

        age = self.predict_age(wav)
        c1 = norm.cdf((CHILD_MAX_AGE + 0.5 - age) / self.sigma)
        c2 = norm.cdf((TEEN_MAX_AGE + 0.5 - age) / self.sigma)
        return np.array([c1, c2 - c1, 1 - c2])
