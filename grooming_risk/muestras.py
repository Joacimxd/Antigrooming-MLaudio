"""Muestras de audio de cada etapa del sistema, versionadas en muestras_audio/.

Las genera scripts/generar_muestras_audio.py. Reproducirlas no requiere los corpus
ni los modelos entrenados: basta con clonar el repositorio.

Provienen de Common Voice (CC0; adolescentes y adultos) y de la voz infantil
sintetizada de ENNI (MIT). Es la única excepción a no versionar audios (decisión del equipo).
"""
from __future__ import annotations

from .config import PROJECT_ROOT

MUESTRAS_DIR = PROJECT_ROOT / "muestras_audio"
INDICE = MUESTRAS_DIR / "indice.csv"


def escuchar(etapa: str) -> None:
    """Muestra en el notebook, con un reproductor, las muestras de una etapa del índice."""
    import json

    import pandas as pd
    from IPython.display import Audio, Markdown, display

    indice = pd.read_csv(INDICE)
    filas = indice[indice.etapa == etapa]
    if filas.empty:
        raise ValueError(f"Etapa desconocida: {etapa}. Opciones: {sorted(indice.etapa.unique())}")
    for r in filas.itertuples():
        display(Markdown(f"**{r.titulo}** · `{r.archivo}`  \n{r.descripcion}"))
        display(Audio(str(MUESTRAS_DIR / r.archivo)))
    resultados = sorted((MUESTRAS_DIR / etapa).glob("*.json"))
    for p in resultados:
        display(Markdown(f"Resultado del sistema (`{p.relative_to(MUESTRAS_DIR)}`):"))
        print(json.dumps(json.loads(p.read_text()), indent=1, ensure_ascii=False))
