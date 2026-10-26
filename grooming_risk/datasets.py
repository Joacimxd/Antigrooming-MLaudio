"""Cargadores de los distintos corpus y unión en un único DataFrame.

Cada cargador devuelve un DataFrame con columnas:
    path, speaker_id, age, age_class, source

`speaker_id` siempre se prefija con el corpus para que la partición por hablante
sea correcta al mezclar datasets. Los cargadores que no encuentren su carpeta
devuelven un DataFrame vacío, así puedes usar solo los corpus que tengas.

Estructura esperada en data/raw/ (ver data/README.md):
    common_voice/<idioma>/validated.tsv + clips/
    speechocean762/{train,test}/{wav.scp,utt2spk,spk2age} + WAVE/
    myst/**/<student_id>/<session_id>/*.flac
    extra/<nino|adolescente|adulto>/<speaker_id>/*.(wav|flac|mp3|...)
    custom/metadata.csv  (path,speaker_id,age[,age_class])
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CLASSES, RAW_DIR, age_to_class

COLUMNS = ["path", "speaker_id", "age", "age_class", "source"]
AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus"}

# Common Voice solo da décadas. "teens" = 13–19, así que es una etiqueta ruidosa
# (incluye 18–19); se trata como adolescente y se documenta como limitación.
CV_AGE = {"teens": 16, "twenties": 25, "thirties": 35, "fourties": 45, "forties": 45,
          "fifties": 55, "sixties": 65, "seventies": 75, "eighties": 85, "nineties": 95}


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def _finish(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        return _empty()
    df = df.copy()
    df["source"] = source
    df["speaker_id"] = source + ":" + df["speaker_id"].astype(str)
    if "age_class" not in df or df["age_class"].isna().all():
        df["age_class"] = df["age"].map(age_to_class)
    df = df[df["age_class"].isin(CLASSES)]
    df = df[df["path"].map(lambda p: Path(p).exists())]
    return df[COLUMNS].reset_index(drop=True)


def load_common_voice(root: Path = RAW_DIR / "common_voice", max_per_speaker: int = 10,
                      seed: int = 0) -> pd.DataFrame:
    """Mozilla Common Voice (cualquier idioma). Aporta adolescentes y adultos."""
    frames = []
    for tsv in sorted(Path(root).glob("*/validated.tsv")):
        df = pd.read_csv(tsv, sep="\t", quoting=3, low_memory=False,
                         usecols=lambda c: c in {"client_id", "path", "age"})
        df = df.dropna(subset=["age"])
        df = df[df["age"].isin(CV_AGE)]
        # Limitar clips por hablante: CV tiene hablantes con miles de grabaciones.
        df = (df.groupby("client_id", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), max_per_speaker), random_state=seed)))
        frames.append(pd.DataFrame({
            "path": df["path"].map(lambda p: str(tsv.parent / "clips" / p)),
            "speaker_id": df["client_id"].str[:16],
            "age": df["age"].map(CV_AGE),
        }))
    return _finish(pd.concat(frames) if frames else _empty(), "common_voice")


def _read_kaldi_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return out


def load_speechocean762(root: Path = RAW_DIR / "speechocean762") -> pd.DataFrame:
    """speechocean762 (OpenSLR 101, CC BY 4.0): 250 hablantes, ~mitad niños, con edad exacta."""
    root = Path(root)
    rows = []
    for split in ("train", "test"):
        d = root / split
        wavs, utt2spk, spk2age = (_read_kaldi_map(d / f) for f in ("wav.scp", "utt2spk", "spk2age"))
        for utt, rel in wavs.items():
            spk = utt2spk.get(utt) or Path(rel).parent.name.replace("SPEAKER", "")
            age = spk2age.get(spk)
            rows.append({"path": str(root / rel), "speaker_id": spk,
                         "age": float(age) if age else np.nan})
    return _finish(pd.DataFrame(rows), "speechocean762")


def load_myst(root: Path = RAW_DIR / "myst", max_per_speaker: int = 20,
              seed: int = 0) -> pd.DataFrame:
    """MyST (LDC2021S05): niños de 3º–5º de primaria (~8–11 años), habla conversacional."""
    root = Path(root)
    files = sorted(root.rglob("*.flac")) if root.exists() else []
    if not files:
        return _empty()
    df = pd.DataFrame({"path": [str(f) for f in files],
                       "speaker_id": [f.parent.parent.name for f in files]})
    df = (df.groupby("speaker_id", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), max_per_speaker), random_state=seed)))
    df["age"] = 9.5  # sin edad individual; el grado escolar garantiza la clase "nino"
    return _finish(df, "myst")


def load_labeled_folders(root: Path = RAW_DIR / "extra") -> pd.DataFrame:
    """Corpus adicionales ya ordenados por clase: extra/<clase>/<hablante>/*.wav.

    Útil para CSLU Kids, PF-STAR, CMU Kids, TIMIT, grabaciones propias con
    consentimiento, etc.
    """
    root = Path(root)
    rows = []
    for cls in CLASSES:
        for f in (root / cls).rglob("*") if (root / cls).exists() else []:
            if f.suffix.lower() in AUDIO_EXT:
                rows.append({"path": str(f), "speaker_id": f"{cls}/{f.parent.name}",
                             "age": np.nan, "age_class": cls})
    return _finish(pd.DataFrame(rows), "extra")


def load_custom_csv(csv: Path = RAW_DIR / "custom" / "metadata.csv") -> pd.DataFrame:
    """CSV propio con columnas path,speaker_id,age (rutas relativas al CSV o absolutas)."""
    csv = Path(csv)
    if not csv.exists():
        return _empty()
    df = pd.read_csv(csv)
    df["path"] = df["path"].map(lambda p: str(p if Path(p).is_absolute() else csv.parent / p))
    if "age_class" not in df:
        df["age_class"] = df["age"].map(age_to_class)
    return _finish(df, "custom")


LOADERS = {
    "common_voice": load_common_voice,
    "speechocean762": load_speechocean762,
    "myst": load_myst,
    "extra": load_labeled_folders,
    "custom": load_custom_csv,
}


def build_dataset(sources: list[str] | None = None, max_per_class: int | None = None,
                  seed: int = 0) -> pd.DataFrame:
    """Une todos los corpus disponibles. `max_per_class` recorta la clase mayoritaria
    (normalmente adultos de Common Voice) muestreando por hablante para conservar diversidad."""
    frames = [LOADERS[name]() for name in (sources or LOADERS)]
    df = pd.concat([f for f in frames if not f.empty] or [_empty()], ignore_index=True)
    df = df.drop_duplicates("path")
    if max_per_class and not df.empty:
        rng = np.random.default_rng(seed)
        parts = []
        for _, g in df.groupby("age_class"):
            if len(g) <= max_per_class:
                parts.append(g)
                continue
            counts = g["speaker_id"].value_counts()
            keep, n = [], 0
            for spk in rng.permutation(counts.index.to_numpy()):
                keep.append(spk)
                n += counts[spk]
                if n >= max_per_class:
                    break
            parts.append(g[g["speaker_id"].isin(keep)])
        df = pd.concat(parts, ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)
