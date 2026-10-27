"""Genera muestras_audio/: un ejemplo escuchable de cada etapa del sistema.

Usa los mismos fragmentos y semillas que el notebook, así que cada audio corresponde
a lo que miden sus tablas (sección 4: filtros; sección 9: conversaciones; sección 10:
moduladores). Requiere los corpus (datasets/merged) y los modelos entrenados (models/);
escucharlas después no requiere nada de eso (grooming_risk.muestras.escuchar).

    python scripts/generar_muestras_audio.py
"""
from __future__ import annotations

import json
import shutil
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import soundfile as sf

from grooming_risk.audio import FILTRO_P, FILTROS, aplicar_filtro, load_audio, normalize, trim_silence
from grooming_risk.config import FEATURES_DIR, MODELS_DIR, SAMPLE_RATE
from grooming_risk.muestras import INDICE, MUESTRAS_DIR

SEED = 42
DATA = ROOT / "datasets"
MERGED = DATA / "merged"
NOMBRE = {"nino": "niño", "adolescente": "adolescente", "adulto": "adulto"}
TITULO_FILTRO = {"telefono": "Teléfono", "ruido_blanco": "Ruido blanco", "saturacion": "Saturación",
                 "ruido_rosa": "Ruido rosa", "huecos": "Huecos", "hum": "Zumbido eléctrico", "clicks": "Clics",
                 "eco": "Eco", "bitcrush": "Reducción de bits", "ambiente": "Ambiente de partida",
                 "cama_musical": "Música de fondo", "sfx_juego": "Efectos de juego"}
filas: list[dict] = []


def licencia(source: str) -> str:
    return "CC0 (Common Voice)" if source == "common_voice" else "MIT (ENNI, voz sintetizada)"


def guardar(rel: str, wav: np.ndarray, etapa: str, titulo: str, descripcion: str, fuente: str):
    """WAV PCM de 16 bits a 16 kHz. Solo se reescala si el pico pasa de 1 (si no, se recortaría)."""
    wav = np.asarray(wav, dtype=np.float32)
    pico = float(np.max(np.abs(wav))) if wav.size else 0.0
    if pico > 0.99:
        wav = wav * (0.99 / pico)
    destino = MUESTRAS_DIR / rel
    destino.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destino, wav, SAMPLE_RATE, subtype="PCM_16")
    filas.append({"etapa": etapa, "archivo": rel, "titulo": titulo, "descripcion": descripcion,
                  "duracion_s": round(len(wav) / SAMPLE_RATE, 2), "licencia": fuente})


def augment_con_registro(wav, rng):
    """Igual que grooming_risk.audio.augment, pero devuelve también qué filtros aplicó."""
    out, usados = np.asarray(wav, dtype=np.float32).copy(), []
    for nombre, p in FILTRO_P.items():
        if rng.random() < p:
            out = FILTROS[nombre](out, rng, SAMPLE_RATE)
            usados.append(TITULO_FILTRO[nombre].lower())
    return normalize(out), usados


def cargar_df():
    df = pd.read_csv(MERGED / "metadata.csv")
    df["path"] = df["path"].map(lambda p: str(MERGED / p))
    valid = ~np.isnan(np.load(FEATURES_DIR / "handcrafted.npz")["X"]).any(1)
    return df, valid


# ---------------------------------------------------------------------------
def etapa_preprocesamiento(ejemplos):
    etapa = "01_preprocesamiento"
    for clase, r in ejemplos.items():
        lic = licencia(r.source)
        nombre = NOMBRE[clase]
        orig = DATA / r.original_path
        destino = MUESTRAS_DIR / etapa / f"{clase}_0_original{orig.suffix}"
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(orig, destino)
        info = sf.info(orig)
        filas.append({"etapa": etapa, "archivo": destino.relative_to(MUESTRAS_DIR).as_posix(),
                      "titulo": f"{nombre.capitalize()}: original del corpus",
                      "descripcion": f"Archivo tal cual en el corpus ({info.format}, {info.samplerate} Hz).",
                      "duracion_s": round(info.duration, 2), "licencia": lic})
        wav16 = sf.read(r.path, dtype="float32")[0]
        guardar(f"{etapa}/{clase}_1_mono16k.wav", wav16, etapa, f"{nombre.capitalize()}: mono a 16 kHz",
                "Conversión del conjunto unido (construccion_dataset.ipynb): un canal y 16 kHz.", lic)
        norm = load_audio(r.path, max_sec=8.0)
        guardar(f"{etapa}/{clase}_2_normalizado.wav", norm, etapa, f"{nombre.capitalize()}: volumen normalizado",
                "Primeros 8 s con el volumen igualado (RMS = 0,05) para que no importe qué tan fuerte grabó cada corpus.", lic)
        guardar(f"{etapa}/{clase}_3_sin_silencios.wav", trim_silence(norm), etapa,
                f"{nombre.capitalize()}: sin silencios",
                "Detección de actividad de voz por energía: se quita todo lo que está a más de 30 dB bajo el pico. "
                "Es la entrada de los extractores de características.", lic)


def etapa_filtros(ejemplos):
    etapa = "02_filtros"
    for clase, r in ejemplos.items():
        lic = licencia(r.source)
        nombre = NOMBRE[clase]
        wav = trim_silence(load_audio(r.path, max_sec=8.0))
        guardar(f"{etapa}/{clase}_00_original.wav", wav, etapa, f"{nombre.capitalize()}: sin filtro",
                "Fragmento de ejemplo de la sección 4 del notebook, ya sin silencios.", lic)
        rng = np.random.default_rng(SEED)        # mismo rng y mismo orden que experimento_clase()
        for k, f in enumerate(FILTROS, start=1):
            guardar(f"{etapa}/{clase}_{k:02d}_{f}.wav", aplicar_filtro(wav, f, rng), etapa,
                    f"{nombre.capitalize()}: {TITULO_FILTRO[f].lower()}",
                    f"Solo el filtro «{TITULO_FILTRO[f].lower()}» (es la fila «{f}» de la tabla de la sección 4).", lic)
        rng = np.random.default_rng(SEED)
        for k in range(1, 3):
            sucio, usados = augment_con_registro(wav, rng)
            guardar(f"{etapa}/{clase}_aug_{k}.wav", sucio, etapa, f"{nombre.capitalize()}: copia aumentada {k}",
                    "Lo que ve el entrenamiento: augment() aplica cada filtro con su probabilidad. "
                    f"Esta copia recibió: {', '.join(usados) or 'ninguno'}.", lic)


def etapa_moduladores(df, valid, mod):
    etapa = "03_moduladores"
    r = df[valid & (df.age_class == "adulto").to_numpy() & (df.split == "train").to_numpy()].iloc[0]
    lic = licencia(r.source)
    wav = load_audio(r.path, max_sec=8.0)
    guardar(f"{etapa}/adulto_original.wav", wav, etapa, "Adulto original",
            "Mismo adulto de entrenamiento que la celda de prueba de moduladores de la sección 10.", lic)
    titulos = {"pitch_simple": "Cambio de tono simple", "praat": "Praat (cambio de género)", "world": "Vocoder WORLD"}
    rng = np.random.default_rng(SEED + 2)        # misma semilla que el acople con el canal del notebook
    for tipo, fn in mod["TIPOS"].items():
        for factor, sentido in ((1.4, "arriba"), (1 / 1.4, "abajo")):
            out = fn(wav, factor)
            f0 = mod["f0_mediano"](out)
            guardar(f"{etapa}/{tipo}_{sentido}.wav", out, etapa, f"{titulos[tipo]}, tono ×{factor:.2f}",
                    f"Tono {'más agudo (suena más joven)' if factor > 1 else 'más grave'}; formantes × {factor ** 0.5:.2f}. "
                    f"F0 mediano: {f0:.0f} Hz (original: {mod['f0_mediano'](wav):.0f} Hz).", lic)
            if factor > 1:
                sucio, usados = augment_con_registro(trim_silence(out), rng)
                guardar(f"{etapa}/{tipo}_{sentido}_con_canal.wav", sucio, etapa,
                        f"{titulos[tipo]} ×{factor:.2f} y después el canal",
                        "Acople del detector: primero el modulador y después augment(). "
                        f"Filtros aplicados: {', '.join(usados) or 'ninguno'}.", lic)


def etapa_conversaciones(df, valid):
    from grooming_risk.diarization import diarize
    from grooming_risk.pipeline import GroomingRiskAnalyzer

    etapa = "04_conversaciones"
    analyzer = GroomingRiskAnalyzer(MODELS_DIR, device="cpu")
    test = df[valid & (df.split == "test").to_numpy()]
    pausa = np.zeros(int(0.4 * SAMPLE_RATE), np.float32)

    def clip(p):
        return trim_silence(load_audio(p, max_sec=6.0))

    # Hablantes de PRUEBA (el modelo nunca los vio); elección fija y reproducible
    enni = test[test.source == "enni"]
    nino_spk = sorted(s for s, g in enni.groupby("speaker_id") if len(g) >= 3)[0]
    ninos = enni[enni.speaker_id == nino_spk].sort_values("path").path.tolist()[:2]
    adultos = test[(test.age_class == "adulto") & test.duration.between(4, 6)].sort_values("path")
    adulto_a = adultos.iloc[0]
    adulto_b = adultos[adultos.speaker_id != adulto_a.speaker_id].iloc[0]
    adol = test[(test.age_class == "adolescente") & test.duration.between(4, 6)].sort_values("path").iloc[0]
    mod = pd.read_csv(DATA / "modulated" / "metadata_mod.csv")
    mod = mod[(mod.split == "test") & (mod.tipo_modulador == "pitch_simple")].iloc[0]

    escenarios = {
        "nino_adulto": ("Niño + adulto", [clip(ninos[0]), clip(adulto_a.path), clip(ninos[1])],
                        "MIT (ENNI, voz sintetizada) + CC0 (Common Voice)"),
        "adolescente_adulto": ("Adolescente + adulto", [clip(adol.path), clip(adulto_a.path)], "CC0 (Common Voice)"),
        "adulto_adulto": ("Adulto + adulto", [clip(adulto_a.path), clip(adulto_b.path)], "CC0 (Common Voice)"),
        "adulto_modulado_adulto": ("Adulto con modulador + adulto",
                                   [clip(DATA / "modulated" / mod.path), clip(adulto_b.path)], "CC0 (Common Voice)"),
    }
    for clave, (titulo, turnos, lic) in escenarios.items():
        wav = np.concatenate([x for t in turnos for x in (t, pausa)])
        res = analyzer.analyze(wav, num_speakers=2)
        guardar(f"{etapa}/{clave}.wav", wav, etapa, titulo,
                f"Conversación simulada con hablantes de prueba (pausas de 0,4 s). Riesgo estimado: "
                f"{res['risk_score']:.2f} ({res['risk_level']}). Es una alerta, no un veredicto.", lic)
        for sp in diarize(wav, analyzer.embedder, num_speakers=2):
            info = next(s for s in res["speakers"] if s["speaker"] == sp.id)
            p, m = info["probabilities"], info["voice_modulation"]
            guardar(f"{etapa}/{clave}_hablante_{sp.id}.wav", sp.audio(wav), etapa,
                    f"{titulo}: hablante {sp.id} tras la diarización",
                    f"Lo que la diarización atribuyó a este hablante. Edad estimada: {NOMBRE[info['predicted_age_group']]} "
                    f"(niño {p['nino']:.2f}, adolescente {p['adolescente']:.2f}, adulto {p['adulto']:.2f}); "
                    f"posible voz modificada: {'sí' if m['possible_modulation'] else 'no'} ({m['probability']:.2f}).", lic)
        res.pop("model", None)
        (MUESTRAS_DIR / etapa / f"{clave}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


def cargar_moduladores():
    """Ejecuta la celda de moduladores de la sección 10 del notebook (una sola implementación)."""
    nb = json.loads((ROOT / "notebooks" / "riesgo_grooming_edad_voz.ipynb").read_text())
    celdas = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    codigo = next(c for c in celdas if "def mod_pitch_simple" in c)
    codigo_f0 = next(c for c in celdas if "def f0_mediano" in c).split("\nes_ejemplo")[0]
    g = {"np": np, "Path": Path, "ROOT": ROOT, "SUBSAMPLE": None, "EXTRACTORS": ["ecapa"],
         "SAMPLE_RATE": SAMPLE_RATE}
    exec(codigo, g)
    exec(codigo_f0, g)
    return g


if __name__ == "__main__":
    if MUESTRAS_DIR.exists():
        for p in MUESTRAS_DIR.iterdir():
            if p.is_dir():
                shutil.rmtree(p)
    df, valid = cargar_df()
    ejemplos = {c: df[df.age_class == c].sample(1, random_state=SEED).iloc[0]   # = audio_de_clase()
                for c in ["nino", "adolescente", "adulto"]}
    etapa_preprocesamiento(ejemplos)
    etapa_filtros(ejemplos)
    etapa_moduladores(df, valid, cargar_moduladores())
    etapa_conversaciones(df, valid)
    pd.DataFrame(filas).to_csv(INDICE, index=False)
    total = sum(p.stat().st_size for p in MUESTRAS_DIR.rglob("*") if p.is_file())
    print(f"{len(filas)} muestras · {total / 1e6:.1f} MB → {MUESTRAS_DIR}")
