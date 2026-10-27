"""Regenera todas las figuras de reporte/figuras/ a su ancho de impresión.

Cada figura se dibuja con el ancho exacto con el que la incluye reporte.tex
(\\linewidth = 16,2 cm), así LaTeX no la reescala y el texto de la figura
(7,5–8,5 pt) queda más pequeño que el pie de figura (\\small = 10 pt).

Requiere haber corrido el notebook completo (cachés en datasets/features/,
models/ y datasets/modulated/). Lo costoso (predicciones de los seis modelos y
conversaciones simuladas) se recalcula con el código y las semillas del notebook
y se guarda en datasets/features/reporte_*.

    python scripts/figuras_reporte.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import joblib
import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf

from grooming_risk.config import CLASSES, FEATURES_DIR, MODELS_DIR, SAMPLE_RATE

SEED = 42
FIG = ROOT / "reporte" / "figuras"
DATA = ROOT / "datasets"
MERGED = DATA / "merged"
ANCHO = 16.2 / 2.54                      # \linewidth en pulgadas (A4, márgenes de 2,4 cm)
NOMBRE = {"nino": "niño", "adolescente": "adolescente", "adulto": "adulto"}
ETIQ = [NOMBRE[c] for c in CLASSES]
ETIQ_CORTA = ["niño", "adol.", "adulto"]          # paneles angostos
AZUL, NARANJA, GRIS, TINTA, REJILLA = "#2A78D6", "#EB6834", "#9C9A94", "#52514E", "#E3E2DF"
PALETA = {"niño": "#2A78D6", "adolescente": "#EB6834", "adulto": "#5E9E57"}
# Etiquetas de Common Voice traducidas (el reporte va en español)
DECADA = {"teens": "10–19", "twenties": "20–29", "thirties": "30–39", "fourties": "40–49",
          "fifties": "50–59", "sixties": "60–69", "seventies": "70–79", "eighties": "80–89"}
ACENTO = {"us": "EE. UU.", "england": "Inglaterra", "indian": "India", "australia": "Australia",
          "canada": "Canadá", "scotland": "Escocia", "newzealand": "Nueva Zelanda", "ireland": "Irlanda",
          "african": "África", "wales": "Gales", "philippines": "Filipinas", "hongkong": "Hong Kong",
          "malaysia": "Malasia", "bermuda": "Bermudas", "southatlandtic": "Atlántico Sur",
          "singapore": "Singapur", "(sin dato)": "(sin dato)"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "legend.title_fontsize": 7.5,
    "text.color": TINTA, "axes.labelcolor": TINTA,
    "xtick.color": TINTA, "ytick.color": TINTA, "axes.edgecolor": REJILLA, "axes.grid": True,
    "grid.color": REJILLA, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.dpi": 300, "figure.dpi": 100,
})


def coma(v, nd=2):
    return f"{v:.{nd}f}".replace(".", ",")


def _num(v, _pos=None):
    """Mismo formato que el texto del reporte: coma decimal y espacio fino como separador de miles."""
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v)):,}".replace(",", "\u2009")
    return f"{v:.10g}".replace(".", ",")


def formatear(fig):
    from matplotlib.ticker import FuncFormatter, ScalarFormatter

    for ax in fig.axes:
        for eje in (ax.xaxis, ax.yaxis):
            if isinstance(eje.get_major_formatter(), ScalarFormatter):
                eje.set_major_formatter(FuncFormatter(_num))


def guardar(fig, nombre):
    formatear(fig)
    fig.tight_layout()
    fig.savefig(FIG / nombre)
    plt.close(fig)
    print("ok", nombre)


def cargar_df():
    df = pd.read_csv(MERGED / "metadata.csv")
    df["path"] = df["path"].map(lambda p: str(MERGED / p))
    df["clase"] = df.age_class.map(NOMBRE)
    return df


# ---------------------------------------------------------------------------
# Corpus de origen y conjunto unido
# ---------------------------------------------------------------------------
def fig_enni():
    enni = pd.read_csv(DATA / "enni-child-speech-synthesis" / "metadata.csv", dtype={"speaker_id": str})
    enni["path"] = enni["file"].map(lambda f: str(DATA / "enni-child-speech-synthesis" / f))
    g = enni.dropna(subset=["gender"]).groupby("speaker_id")["gender"].first()
    enni["gender"] = enni["gender"].fillna(enni["speaker_id"].map(g)).map({"M": "masculino", "F": "femenino"})
    enni["duration"] = [sf.info(p).duration for p in enni["path"]]
    spk = enni.groupby("speaker_id").agg(edad=("age", "first"), genero=("gender", "first"),
                                         clips=("file", "size"))
    fig, ax = plt.subplots(1, 3, figsize=(ANCHO, 2.1))
    sns.countplot(data=spk, x="edad", hue="genero", ax=ax[0], palette=[AZUL, NARANJA])
    ax[0].set(title="Hablantes por edad y género", xlabel="edad (años)", ylabel="hablantes", ylim=(0, 50))
    ax[0].legend(title=None, frameon=False, ncol=2, loc="upper left", fontsize=7, handlelength=1)
    sns.histplot(spk["clips"], bins=25, ax=ax[1], color=AZUL)
    ax[1].set(title="Fragmentos por hablante", xlabel="fragmentos", ylabel="hablantes")
    sns.histplot(enni["duration"], bins=35, ax=ax[2], color=AZUL)
    ax[2].set(title="Duración de los fragmentos", xlabel="duración (s)", ylabel="fragmentos")
    guardar(fig, "enni_hablantes_clips_duracion.png")


def fig_cv():
    from grooming_risk.datasets import CV_AGE

    frames = []
    for csv in sorted((DATA / "archive").glob("cv-*.csv")):
        d = pd.read_csv(csv)
        d["subset"] = csv.stem
        frames.append(d)
    cv = pd.concat(frames, ignore_index=True)
    cv = cv[(cv["subset"] != "cv-invalid") & cv["age"].isin(CV_AGE)].copy()
    orden = [a for a in CV_AGE if a in set(cv["age"])]
    cv["genero"] = cv["gender"].map({"male": "masculino", "female": "femenino", "other": "otro"})
    fig, ax = plt.subplots(1, 2, figsize=(ANCHO, 2.6), gridspec_kw={"width_ratios": [1.3, 1]})
    cv["decada"] = cv["age"].map(DECADA)
    sns.countplot(data=cv, x="decada", hue="genero", order=[DECADA[a] for a in orden], ax=ax[0],
                  palette=[AZUL, NARANJA, GRIS], hue_order=["masculino", "femenino", "otro"])
    ax[0].set(title="Fragmentos por década de edad y género", xlabel="edad declarada (años)", ylabel="fragmentos")
    ax[0].tick_params(axis="x", rotation=40, labelsize=7)
    ax[0].legend(title=None, frameon=False)
    (cv["accent"].fillna("(sin dato)").map(ACENTO).value_counts() / 1000).plot.barh(ax=ax[1], color=AZUL)
    ax[1].set(title="Fragmentos por acento", xlabel="fragmentos (miles)", ylabel="")
    ax[1].tick_params(axis="y", labelsize=7)
    ax[1].invert_yaxis()
    guardar(fig, "cv_clips_decada_acento.png")


def fig_corpus_clase(df):
    fig, ax = plt.subplots(1, 2, figsize=(0.85 * ANCHO, 2.3))
    t = pd.crosstab(df.source.map({"common_voice": "Common Voice", "enni": "ENNI"}), df.clase)[ETIQ]
    t.plot.bar(stacked=True, ax=ax[0], color=[PALETA[c] for c in ETIQ], rot=0)
    ax[0].set(title="Fragmentos por corpus y clase", xlabel="corpus", ylabel="fragmentos", ylim=(0, 18000))
    ax[0].legend(title=None, frameon=False, ncol=3, loc="upper center", fontsize=6.5, handlelength=1,
                 columnspacing=0.8)
    h = df.groupby("clase").speaker_id.nunique().reindex(ETIQ)
    ax[1].bar(ETIQ, h.values, color=[PALETA[c] for c in ETIQ])
    for i, v in enumerate(h.values):
        ax[1].text(i, v + 4, str(v), ha="center", fontsize=7.5)
    ax[1].set(title="Hablantes por clase", ylabel="hablantes (o pseudo-hablantes)", ylim=(0, 390))
    guardar(fig, "clips_corpus_hablantes_clase.png")


def fig_dataset(df):
    fig, ax = plt.subplots(1, 3, figsize=(ANCHO, 2.4), gridspec_kw={"width_ratios": [1, 0.8, 1.2]})
    d = df.assign(genero=df.gender.map({"male": "masculino", "female": "femenino"}).fillna("sin dato"))
    sns.countplot(data=d, x="clase", hue="genero", order=ETIQ, ax=ax[0],
                  palette=[AZUL, NARANJA, GRIS], hue_order=["masculino", "femenino", "sin dato"])
    ax[0].set(title="Fragmentos por clase y género", xlabel="", ylabel="fragmentos", ylim=(0, 7200))
    ax[0].set_xticks(range(3), ETIQ_CORTA)
    ax[0].legend(title=None, frameon=False, ncol=3, loc="upper center", fontsize=6.5, handlelength=1,
                 columnspacing=0.8)
    sns.boxplot(data=df, x="clase", y="duration", order=ETIQ, ax=ax[1], showfliers=False,
                palette=PALETA, hue="clase", legend=False)
    ax[1].set(title="Duración por clase", xlabel="", ylabel="duración (s)")
    ax[1].set_xticks(range(3), ETIQ_CORTA)
    acc = (df[df.source == "common_voice"].assign(accent=lambda d: d.accent.map(ACENTO))
             .groupby("clase")["accent"].value_counts(normalize=True)
             .unstack(0)[["adolescente", "adulto"]])
    acc = acc.loc[acc.sum(1).sort_values(ascending=False).index[:10]]
    acc.plot.barh(ax=ax[2], color=[PALETA["adolescente"], PALETA["adulto"]])
    ax[2].invert_yaxis()
    ax[2].set(title="Acento (10 principales)", xlabel="proporción", ylabel="")
    ax[2].legend(title=None, frameon=False, loc="lower right")
    guardar(fig, "dataset_clase_duracion_acento.png")


def fig_canal(df):
    def stats(path):
        y, sr = sf.read(path, dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE, res_type="soxr_hq")
        db = 20 * np.log10(librosa.feature.rms(y=y)[0] + 1e-8)
        S = np.abs(librosa.stft(y, n_fft=512)) ** 2
        f = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=512)
        return {"Nivel\n(dBFS)": np.percentile(db, 95), "Piso de ruido\n(dB)": np.percentile(db, 10),
                "SNR aprox.\n(dB)": np.percentile(db, 95) - np.percentile(db, 10),
                "Frecuencia del 99 %\nde la energía (Hz)": np.median(
                    librosa.feature.spectral_rolloff(S=S, sr=SAMPLE_RATE, roll_percent=0.99)),
                "Energía sobre\n4 kHz (%)": 100 * S[f > 4000].sum() / max(S.sum(), 1e-12)}

    # Igual que construccion_dataset.ipynb: 150 fragmentos por clase, audio original (antes de normalizar)
    m = df.sample(frac=1, random_state=0).groupby("age_class").head(150)
    st = pd.DataFrame([stats(DATA / p) for p in m.original_path])
    st["clase"] = m.clase.to_numpy()
    cols = [c for c in st.columns if c != "clase"]
    fig, ax = plt.subplots(1, len(cols), figsize=(ANCHO, 2.4))
    for a, c in zip(ax, cols):
        sns.boxplot(data=st, x="clase", y=c, order=ETIQ, ax=a, showfliers=False, palette=PALETA,
                    hue="clase", legend=False)
        a.set(title=c, xlabel="", ylabel="")
        a.set_xticks([])
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=PALETA[c], label=c) for c in ETIQ], loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    formatear(fig)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "dataset_canal_por_clase.png")
    plt.close(fig)
    print("ok dataset_canal_por_clase.png")


def fig_particiones(df):
    t = pd.crosstab(df.split, df.clase).loc[["train", "val", "test"], ETIQ]
    t.index = ["entrenamiento", "validación", "prueba"]
    fig, ax = plt.subplots(figsize=(0.7 * ANCHO, 2.2))
    t.plot.bar(ax=ax, color=[PALETA[c] for c in ETIQ], rot=0, width=0.75)
    for cont in ax.containers:
        ax.bar_label(cont, fontsize=6.5, padding=1, labels=[_num(v) for v in cont.datavalues])
    ax.set(ylabel="fragmentos", xlabel="")
    ax.legend(title=None, frameon=False)
    guardar(fig, "particiones.png")


def fig_f0(df):
    X = np.load(FEATURES_DIR / "handcrafted.npz")["X"]
    valid = ~np.isnan(X).any(1)
    d = pd.DataFrame({"F0": np.exp(X[:, 0]), "clase": df.clase})[valid]
    fig, ax = plt.subplots(figsize=(0.62 * ANCHO, 2.3))
    sns.boxplot(data=d, x="clase", y="F0", order=ETIQ, ax=ax, showfliers=False, palette=PALETA,
                hue="clase", legend=False)
    for i, c in enumerate(ETIQ):
        med = d[d.clase == c].F0.median()
        ax.text(i, med + 6, f"{med:.0f} Hz", ha="center", fontsize=7, color="white", weight="bold")
    ax.set(xlabel="", ylabel="F0 media del fragmento (Hz)")
    guardar(fig, "f0_por_clase.png")


# ---------------------------------------------------------------------------
# Modelos de edad
# ---------------------------------------------------------------------------
MODELOS = {"B_ecapa_svm": "B · ECAPA + SVM", "C_wavlm_logreg": "C · WavLM + reg. logística",
           "B2_ecapa_rf": "B2 · ECAPA + bosque aleatorio", "A2_handcrafted_rf": "A2 · Acústico + bosque aleatorio",
           "A_handcrafted_gbm": "A · Acústico + pot. del gradiente",
           "A3_handcrafted_logreg": "A3 · Acústico + reg. logística"}


def fig_comparativa():
    cmp_ = json.loads((MODELS_DIR / "model_card.json").read_text())["comparison_test_clip"]
    orden = sorted(cmp_, key=lambda k: cmp_[k]["f1_macro"])
    f1 = [cmp_[k]["f1_macro"] for k in orden]
    ad = [cmp_[k]["recall_adolescente"] for k in orden]
    fig, ax = plt.subplots(figsize=(0.85 * ANCHO, 2.9))
    y, h = np.arange(len(orden)), 0.38
    b1 = ax.barh(y + h / 2, f1, h, color=AZUL, label="F1-macro")
    b2 = ax.barh(y - h / 2, ad, h, color=NARANJA, label="Sensibilidad para adolescentes")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2, coma(b.get_width()),
                    va="center", fontsize=7)
    ax.set_yticks(y, [MODELOS[k] for k in orden])
    ax.set(xlim=(0, 1), xlabel="valor en el conjunto de prueba")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, frameon=False)
    guardar(fig, "comparativa_modelos.png")


def predicciones_test(df):
    """Reentrena los seis modelos exactamente como el notebook (sección 5) y guarda sus
    probabilidades en prueba. Verifica que el F1-macro coincida con model_card.json."""
    cache = FEATURES_DIR / "reporte_probas_test.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["y"], {k: z[k] for k in z.files if k != "y"}
    from sklearn.base import clone
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.frozen import FrozenEstimator
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import ParameterGrid
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    from grooming_risk.pipeline import reorder_proba

    feats = {n: (np.load(FEATURES_DIR / f"{n}.npz")["X"], np.load(FEATURES_DIR / f"{n}.npz")["X_aug"])
             for n in ["handcrafted", "ecapa", "wavlm"]}
    valid = ~np.isnan(feats["handcrafted"][0]).any(1)
    y = df.age_class.to_numpy()
    idx = {s: np.where((df.split == s).to_numpy() & valid)[0] for s in ["train", "val", "test"]}

    def get_xy(name, split, with_aug=False):
        X, X_aug = feats[name]
        i = idx[split]
        Xs, ys = [X[i]], [y[i]]
        if with_aug:
            for k in range(X_aug.shape[1]):
                ok = ~np.isnan(X_aug[i, k]).any(1)
                Xs.append(X_aug[i, k][ok]); ys.append(y[i][ok])
        return np.vstack(Xs), np.concatenate(ys)

    modelos = {
        "A_handcrafted_gbm": ("handcrafted", HistGradientBoostingClassifier(max_iter=400, class_weight="balanced",
                              early_stopping=True, random_state=SEED),
                              {"learning_rate": [0.05, 0.1], "max_depth": [None, 6], "l2_regularization": [0.0, 1.0]}),
        "B_ecapa_svm": ("ecapa", make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced",
                        random_state=SEED)), {"svc__C": [0.3, 1, 3, 10], "svc__gamma": ["scale"]}),
        "C_wavlm_logreg": ("wavlm", make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced",
                           max_iter=5000)), {"logisticregression__C": [0.001, 0.01, 0.1, 1.0]}),
        "A2_handcrafted_rf": ("handcrafted", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                              n_jobs=-1, random_state=SEED), {"max_depth": [12, 24], "min_samples_leaf": [2, 8]}),
        "A3_handcrafted_logreg": ("handcrafted", make_pipeline(StandardScaler(), LogisticRegression(
                                  class_weight="balanced", max_iter=5000)),
                                  {"logisticregression__C": [0.01, 0.1, 1.0, 10.0]}),
        "B2_ecapa_rf": ("ecapa", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                        n_jobs=-1, random_state=SEED), {"max_depth": [12, 24], "min_samples_leaf": [2, 8]}),
    }
    card = json.loads((MODELS_DIR / "model_card.json").read_text())["comparison_test_clip"]
    probas = {}
    for name, (feat, base, grid) in modelos.items():
        Xtr, ytr = get_xy(feat, "train", with_aug=True)
        Xva, yva = get_xy(feat, "val")
        best = (-1, None)
        for params in ParameterGrid(grid):
            m = clone(base).set_params(**params).fit(Xtr, ytr)
            s = f1_score(yva, m.predict(Xva), average="macro")
            if s > best[0]:
                best = (s, m)
        cal = CalibratedClassifierCV(FrozenEstimator(best[1]), method="sigmoid").fit(Xva, yva)
        Xte, yte = get_xy(feat, "test")
        probas[name] = reorder_proba(cal.predict_proba(Xte), cal.classes_)
        f1 = f1_score(yte, np.array(CLASSES)[probas[name].argmax(1)], average="macro")
        print(f"{name}: F1 test = {f1:.3f} (notebook {card[name]['f1_macro']:.3f})")
        assert abs(f1 - card[name]["f1_macro"]) < 0.002, "No coincide con el notebook"
    np.savez_compressed(cache, y=yte, **probas)
    return yte, probas


def fig_confusion(df):
    from sklearn.metrics import confusion_matrix

    y, probas = predicciones_test(df)
    orden = ["B_ecapa_svm", "C_wavlm_logreg", "B2_ecapa_rf", "A2_handcrafted_rf", "A_handcrafted_gbm",
             "A3_handcrafted_logreg"]
    fig, axes = plt.subplots(2, 3, figsize=(ANCHO, 4.6))
    for ax, k in zip(axes.ravel(), orden):
        cm = confusion_matrix(y, np.array(CLASSES)[probas[k].argmax(1)], labels=CLASSES, normalize="true")
        sns.heatmap(cm, annot=np.vectorize(coma)(cm), fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False,
                    ax=ax, xticklabels=ETIQ, yticklabels=ETIQ, annot_kws={"fontsize": 7.5},
                    linewidths=0.5, linecolor="white")
        ax.set(title=MODELOS[k], xlabel="clase predicha", ylabel="clase real")
        ax.grid(False)
    guardar(fig, "matrices_confusion.png")


# ---------------------------------------------------------------------------
# Filtros, conversaciones simuladas y moduladores
# ---------------------------------------------------------------------------
def fig_mfcc_filtros(df):
    """MFCC de un mismo fragmento adulto limpio y con tres filtros (como la sección 4 del notebook)."""
    from grooming_risk.audio import aplicar_filtro, load_audio, trim_silence

    fila = df[df.age_class == "adulto"].sample(1, random_state=SEED).iloc[0]
    wav = trim_silence(load_audio(fila.path, max_sec=8.0))
    rng = np.random.default_rng(SEED)
    variantes = {"original": wav}
    for f in ["telefono", "ruido_blanco", "cama_musical"]:
        variantes[f] = aplicar_filtro(wav, f, rng)
    titulos = {"original": "Original", "telefono": "Teléfono", "ruido_blanco": "Ruido blanco",
               "cama_musical": "Música de fondo"}
    fig, axes = plt.subplots(1, 4, figsize=(ANCHO, 2.0), sharey=True)
    for ax, (k, w) in zip(axes, variantes.items()):
        m = librosa.feature.mfcc(y=w, sr=SAMPLE_RATE, n_mfcc=20, n_fft=512, hop_length=160)
        t = np.arange(m.shape[1]) * 160 / SAMPLE_RATE
        im = ax.pcolormesh(t, np.arange(1, 21), m, cmap="magma", vmin=-250, vmax=150, shading="auto")
        ax.set(title=titulos[k], xlabel="tiempo (s)")
        ax.grid(False)
    axes[0].set_ylabel("coeficiente MFCC")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01)
    cb.ax.tick_params(labelsize=7)
    formatear(fig)
    fig.savefig(FIG / "mfcc_filtros.png", bbox_inches="tight")
    plt.close(fig)
    print("ok mfcc_filtros.png", Path(fila.path).name)


def simulacion(df):
    """Repite la sección 9 del notebook (mismo rng y mismos hablantes de prueba)."""
    cache = FEATURES_DIR / "reporte_simulacion.csv"
    if cache.exists():
        return pd.read_csv(cache)
    from grooming_risk.audio import load_audio, trim_silence
    from grooming_risk.pipeline import GroomingRiskAnalyzer

    valid = ~np.isnan(np.load(FEATURES_DIR / "handcrafted.npz")["X"]).any(1)
    test_df_valid = df.iloc[np.where((df.split == "test").to_numpy() & valid)[0]]
    analyzer = GroomingRiskAnalyzer(MODELS_DIR, device="cpu")
    test_pool = test_df_valid.groupby(["age_class", "speaker_id"]).path.apply(list)
    rng = np.random.default_rng(SEED)

    def make_conversation(classes, rng, turns=3, pause=0.4):
        speakers = []
        for c in classes:
            options = [s for s in test_pool.loc[c].index if s not in speakers]
            speakers.append(options[rng.integers(len(options))])
        clips = {s: [trim_silence(load_audio(p, max_sec=6.0))
                     for p in test_pool.loc[(c, s)][:1 if s.startswith("cv:") else turns]]
                 for c, s in zip(classes, speakers)}
        silence = np.zeros(int(pause * SAMPLE_RATE), np.float32)
        parts = []
        for t in range(turns):
            for s in speakers:
                if t < len(clips[s]):
                    parts += [clips[s][t], silence]
        return np.concatenate(parts)

    escenarios = {"nino + adulto": ["nino", "adulto"], "adolescente + adulto": ["adolescente", "adulto"],
                  "adulto + adulto": ["adulto", "adulto"], "nino + nino": ["nino", "nino"],
                  "adulto solo": ["adulto"]}
    sim = []
    for esc, classes in escenarios.items():
        for _ in range(20):
            out = analyzer.analyze(make_conversation(classes, rng))
            sim.append({"escenario": esc, "riesgo": out["risk_score"], "nivel": out["risk_level"],
                        "hablantes_detectados": out["num_speakers"]})
    sim = pd.DataFrame(sim)
    print(sim.groupby("escenario").agg(riesgo_medio=("riesgo", "mean"),
                                       alto=("nivel", lambda s: (s == "alto").mean()),
                                       hablantes=("hablantes_detectados", "mean")).round(3))
    sim.to_csv(cache, index=False)
    return sim


def fig_simulacion(df):
    sim = simulacion(df)
    nombres = {"nino + adulto": "niño + adulto", "adolescente + adulto": "adolescente + adulto",
               "adulto + adulto": "adulto + adulto", "nino + nino": "niño + niño", "adulto solo": "adulto solo"}
    sim["escenario"] = sim.escenario.map(nombres)
    fig, ax = plt.subplots(figsize=(0.75 * ANCHO, 2.3))
    sns.boxplot(data=sim, x="riesgo", y="escenario", order=list(nombres.values()), ax=ax,
                color="#BFD6F2", showfliers=False, width=0.55)
    sns.stripplot(data=sim, x="riesgo", y="escenario", order=list(nombres.values()), ax=ax,
                  color=AZUL, size=2.5, jitter=0.18)
    ax.axvline(0.3, ls="--", c="#E8A33D", lw=1, label="umbral medio (0,30)")
    ax.axvline(0.6, ls="--", c=NARANJA, lw=1, label="umbral alto (0,60)")
    ax.set(xlim=(0, 1), xlabel="riesgo estimado", ylabel="")
    ax.legend(frameon=False, loc="lower right")
    guardar(fig, "riesgo_escenarios_simulados.png")


def fig_detector():
    """Detector desplegado (acoplado con el canal), en prueba con canal."""
    df = pd.read_csv(MERGED / "metadata.csv")
    valid = ~np.isnan(np.load(FEATURES_DIR / "handcrafted.npz")["X"]).any(1)
    e = np.load(FEATURES_DIR / "ecapa.npz")
    Xo = e["X_aug"][:, 0].copy()
    fuera = np.where(valid & (df.split != "train").to_numpy())[0]
    Xo[fuera] = np.load(FEATURES_DIR / "ecapa_originales_canal.npz", allow_pickle=True)["X"]
    o, Xo = df[valid], Xo[valid]
    m = pd.read_csv(DATA / "modulated" / "metadata_mod.csv")
    ok = ~np.isnan(np.load(FEATURES_DIR / "ecapa_modulados.npz", allow_pickle=True)["X"]).any(1)
    Xm = np.load(FEATURES_DIR / "ecapa_modulados_canal.npz", allow_pickle=True)["X"][ok]
    m = m[ok]
    d = joblib.load(MODELS_DIR / "detector_modulador.joblib")
    tm = {"pitch_simple": "cambio de tono simple", "praat": "Praat", "world": "WORLD (no visto)"}
    grupos = [(f"original · {NOMBRE[c]}", o[(o.split == "test") & (o.age_class == c)].index, GRIS) for c in CLASSES]
    datos = [(g, d["model"].predict_proba(Xo[np.isin(o.index, i)])[:, 1], col) for g, i, col in grupos]
    datos += [(f"modulado · {tm[t]}", d["model"].predict_proba(Xm[((m.split == "test") & (m.tipo_modulador == t)).values])[:, 1], AZUL)
              for t in tm]
    datos = datos[::-1]
    fig, ax = plt.subplots(figsize=(0.85 * ANCHO, 2.4))
    bp = ax.boxplot([s for _, s, _ in datos], vert=False, widths=0.55, showfliers=False, patch_artist=True,
                    medianprops={"color": "white", "lw": 1.4}, whiskerprops={"color": TINTA},
                    capprops={"color": TINTA}, boxprops={"lw": 0})
    for b, (_, _, c) in zip(bp["boxes"], datos):
        b.set_facecolor(c)
    ax.set_yticks(range(1, len(datos) + 1), [g for g, _, _ in datos])
    u = d["umbral"]
    ax.axvline(u, ls="--", c=NARANJA, lw=1.1)
    ax.text(u + 0.01, len(datos) + 0.4, f"umbral = {coma(u)}", color=NARANJA, fontsize=7.5, va="center")
    ax.set(xlim=(-0.02, 1.02), ylim=(0.4, len(datos) + 0.8), xlabel="probabilidad calibrada de voz modificada")
    ax.grid(axis="y", visible=False)
    guardar(fig, "detector_moduladores.png")


def fig_genero(df):
    """Tono por clase y género, y sensibilidad del modelo desplegado por clase y género (prueba)."""
    from grooming_risk.pipeline import reorder_proba

    H = np.load(FEATURES_DIR / "handcrafted.npz")["X"]
    valid = ~np.isnan(H).any(1)
    d = df.assign(genero=df.gender.map({"male": "hombres", "female": "mujeres"}))
    ac = d[valid].assign(F0=np.exp(H[valid, 2])).dropna(subset=["genero"])
    E = np.load(FEATURES_DIR / "ecapa.npz")["X"]
    b = joblib.load(MODELS_DIR / "best_model.joblib")
    t = (d.split == "test").to_numpy() & valid
    te = d[t].assign(pred=np.array(CLASSES)[reorder_proba(b["pipeline"].predict_proba(E[t]),
                                                          b["pipeline"].classes_).argmax(1)])
    te = te.dropna(subset=["genero"])
    colores = {"hombres": AZUL, "mujeres": NARANJA}

    fig, ax = plt.subplots(1, 2, figsize=(ANCHO, 2.5))
    sns.boxplot(data=ac, x="clase", y="F0", hue="genero", order=ETIQ, hue_order=list(colores),
                palette=colores, showfliers=False, ax=ax[0], width=0.7)
    ax[0].set(title="Tono por clase y género", xlabel="", ylabel="F0 mediana del fragmento (Hz)")
    ax[0].legend(title=None, frameon=False, loc="lower left")
    sens = (te.assign(ok=te.pred == te.age_class).groupby(["clase", "genero"])
              .agg(s=("ok", "mean"), h=("speaker_id", "nunique")).reset_index())
    sns.barplot(data=sens, x="clase", y="s", hue="genero", order=ETIQ, hue_order=list(colores),
                palette=colores, ax=ax[1], width=0.7)
    for cont in ax[1].containers:
        ax[1].bar_label(cont, labels=[coma(v) for v in cont.datavalues], fontsize=7, padding=1)
    h = sens.pivot(index="clase", columns="genero", values="h").reindex(ETIQ)
    ax[1].set_xticks(range(3), [f"{c}\n({h.loc[c, 'hombres']} H · {h.loc[c, 'mujeres']} M)" for c in ETIQ])
    ax[1].set(title="Sensibilidad del modelo B en prueba", xlabel="", ylabel="sensibilidad", ylim=(0, 1.2))
    ax[1].set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax[1].legend(title=None, frameon=False, loc="upper right", ncol=2)
    guardar(fig, "genero_tono_sensibilidad.png")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    df = cargar_df()
    fig_enni(); fig_cv(); fig_corpus_clase(df); fig_dataset(df); fig_canal(df); fig_particiones(df)
    fig_f0(df); fig_comparativa(); fig_confusion(df); fig_mfcc_filtros(df); fig_simulacion(df)
    fig_detector(); fig_genero(df)
