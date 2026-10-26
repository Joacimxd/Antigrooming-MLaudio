"""Diarización ligera: ¿quién habla cuándo?

VAD por energía → ventanas de 1.5 s → embeddings ECAPA → clustering aglomerativo
por distancia coseno. No requiere token de Hugging Face (a diferencia de
pyannote) y corre en CPU. Es menos precisa con solapamientos y voces muy
parecidas; para producción conviene evaluar pyannote/NeMo.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .audio import speech_regions
from .config import DIAR_COSINE_THRESHOLD, DIAR_HOP_SEC, DIAR_WINDOW_SEC, SAMPLE_RATE


@dataclass
class Speaker:
    id: int
    segments: list[tuple[int, int]] = field(default_factory=list)  # en muestras

    @property
    def duration(self) -> float:
        return sum(e - s for s, e in self.segments) / SAMPLE_RATE

    def audio(self, wav: np.ndarray) -> np.ndarray:
        return np.concatenate([wav[s:e] for s, e in self.segments])


def _windows(regions, sr):
    win, hop = int(DIAR_WINDOW_SEC * sr), int(DIAR_HOP_SEC * sr)
    out = []
    for s, e in regions:
        if e - s <= win:
            out.append((s, e))
            continue
        starts = list(range(s, e - win + 1, hop))
        if e - (starts[-1] + win) > hop // 2:
            starts.append(e - win)
        out += [(st, st + win) for st in starts]
    return out


def diarize(wav: np.ndarray, embedder, num_speakers: int | None = None,
            threshold: float = DIAR_COSINE_THRESHOLD, min_speaker_sec: float = 1.0,
            batch_size: int = 32) -> list[Speaker]:
    """`embedder` debe tener `extract_batch(list[np.ndarray]) -> (N, D)` (EcapaExtractor)."""
    from sklearn.cluster import AgglomerativeClustering

    sr = SAMPLE_RATE
    windows = _windows(speech_regions(wav, sr), sr)
    if not windows:
        return []

    embs = np.concatenate([
        embedder.extract_batch([wav[s:e] for s, e in windows[i:i + batch_size]])
        for i in range(0, len(windows), batch_size)])
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9

    if len(windows) == 1 or num_speakers == 1:
        labels = np.zeros(len(windows), dtype=int)
    elif num_speakers:
        labels = AgglomerativeClustering(n_clusters=min(num_speakers, len(windows)),
                                         metric="cosine", linkage="average").fit_predict(embs)
    else:
        labels = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold,
                                         metric="cosine", linkage="average").fit_predict(embs)

    # Fusionar ventanas consecutivas del mismo hablante en turnos sin solapamiento.
    turns: dict[int, list[list[int]]] = {}
    prev_end = 0
    for (s, e), lab in zip(windows, labels):
        s = max(s, prev_end)
        segs = turns.setdefault(int(lab), [])
        if segs and s <= segs[-1][1]:
            segs[-1][1] = e
        else:
            segs.append([s, e])
        prev_end = e

    speakers = [Speaker(id=0, segments=[tuple(x) for x in segs]) for segs in turns.values()]
    speakers = [sp for sp in speakers if sp.duration >= min_speaker_sec]
    speakers.sort(key=lambda sp: sp.segments[0][0])
    for i, sp in enumerate(speakers):
        sp.id = i
    return speakers
