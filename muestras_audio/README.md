# Muestras de audio por etapa

Ejemplos escuchables de cada proceso del sistema, para entenderlo **sin descargar los corpus
ni entrenar los modelos**. Se escuchan desde los notebooks (celdas «Escucha») con
`grooming_risk.muestras.escuchar("<etapa>")`, o abriendo los archivos directamente.
`indice.csv` describe cada archivo (etapa, título, descripción, duración y licencia).

| Carpeta | Qué contiene | Notebook |
|---|---|---|
| `01_preprocesamiento/` | Un niño, un adolescente y un adulto: original del corpus → mono 16 kHz → volumen normalizado → sin silencios | `construccion_dataset.ipynb` (sección 8) |
| `02_filtros/` | Los mismos tres fragmentos con cada uno de los 12 filtros por separado y dos copias de `augment()` (lo que ve el entrenamiento) | `riesgo_grooming_edad_voz.ipynb` (sección 4) |
| `03_moduladores/` | Un adulto con los tres moduladores (tono ×1,4 y ×0,71) y la versión modulador → canal del detector acoplado | sección 10 |
| `04_conversaciones/` | Cuatro conversaciones simuladas, el audio de cada hablante tras la diarización y la respuesta del sistema (`.json`) | sección 9 |

Los fragmentos de las carpetas 01 y 02 son los mismos que miden las tablas de la sección 4, así que
cada audio corresponde a una fila de esas tablas.

**Origen y licencias.** Common Voice (CC0) para adolescentes y adultos, y la voz infantil
**sintetizada** del corpus ENNI (MIT). Las conversaciones usan hablantes de prueba, que el modelo
nunca vio al entrenar.

**Regenerar** (requiere `datasets/merged` y `models/`): `python scripts/generar_muestras_audio.py`.

> El riesgo que acompaña a las conversaciones es una **alerta para priorizar revisión humana**,
> nunca un veredicto: que un menor hable con un adulto no es *grooming*.
