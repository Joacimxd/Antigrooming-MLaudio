# Guía del proyecto — Estimador de riesgo de grooming por edad de voz

## ⚠️ PRINCIPIO RECTOR: ES UNA ALERTA, NO UN DECISOR

**Este sistema es exclusivamente un estimador de riesgo / sistema de alerta temprana.
NO es, ni debe usarse como, un decisor definitivo.**

- El modelo **solo estima la edad aparente** de las voces de un audio y la probabilidad de que
  haya una persona menor de edad hablando con una persona adulta. **No analiza el contenido**
  de la conversación, ni intenciones, ni contexto.
- Que un menor hable con un adulto **no es grooming**. Padres, familiares, docentes, médicos,
  entrenadores o tutores producen exactamente la misma señal. **La gran mayoría de las alertas
  serán falsos positivos desde el punto de vista del grooming.**
- La estimación de edad por voz tiene errores: adolescentes con voz grave, adultos con voz aguda,
  ruido, mala calidad de audio o voces de idiomas/acentos poco representados en el entrenamiento.
- Un riesgo de "90 %" significa: *"el modelo está bastante seguro de que hay un niño y un adulto
  distintos en este audio"*. **No** significa "90 % de probabilidad de que esté ocurriendo grooming".
- **Ninguna acción con consecuencias** (denuncias, bloqueos de cuentas, sanciones, acusaciones,
  decisiones legales o disciplinarias) puede tomarse basándose únicamente en este puntaje.
  Toda alerta debe pasar por **revisión humana especializada** que considere el contexto completo.
- El puntaje sirve para **priorizar** qué revisar, no para concluir.

Al modificar este proyecto, cualquier colaborador debe:
1. Mantener el campo `disclaimer` en **todas** las respuestas de la API (`grooming_risk/config.py::DISCLAIMER`).
2. No renombrar salidas a términos que sugieran certeza ("grooming detectado", "culpable", "confirmado").
   Usar "riesgo", "alerta", "estimación".
3. No añadir decisiones automáticas (bloqueos, reportes) acopladas al puntaje.
4. Documentar cualquier cambio en los pesos de riesgo (`RISK_WEIGHTS`) y en los umbrales.

## Privacidad y ética
- Los audios contienen voces de menores: datos personales sensibles. No versionarlos (`datasets/raw/`
  está en `.gitignore`), respetar las licencias de cada corpus y usarlos solo para investigación.
- Excepción decidida por el equipo: `muestras_audio/` (ejemplos escuchables de cada etapa, Common Voice CC0
  y voz infantil sintetizada de ENNI) sí se versiona. Se regenera con `scripts/generar_muestras_audio.py`.
- La API no guarda audios: se escriben en un temporal que se borra al terminar la petición.
- Cualquier despliegue real requiere base legal, consentimiento/aviso adecuado y evaluación de
  impacto (p.ej. leyes de protección de datos aplicables).

## Qué hace el sistema
```
audio → VAD + diarización (ECAPA + clustering) → por hablante: P(nino), P(adolescente), P(adulto)
      → riesgo = max_{i≠j} [ P_i(nino)·P_j(adulto)·0.90 + P_i(adolescente)·P_j(adulto)·0.60 ]
      → nivel: bajo (<0.30) · medio (0.30–0.60) · alto (≥0.60); piso 0.05
```
Clases: `nino` ≤ 12 años, `adolescente` 13–17, `adulto` ≥ 18.

## Estructura
```
grooming_risk/         paquete compartido por notebook y API
  config.py            clases, pesos de riesgo, umbrales, modelos preentrenados, DISCLAIMER
  datasets.py          cargadores: Common Voice, speechocean762, MyST, carpetas extra, CSV propio
  audio.py             carga 16 kHz mono, normalización, VAD, troceado, aumentación de canal
  features.py          extractores: handcrafted (F0/formantes/MFCC), ECAPA, WavLM; referencia audEERING
  diarization.py       diarización ligera sin token de HF
  risk.py              fórmula de riesgo
  pipeline.py          bundle del modelo + GroomingRiskAnalyzer (end-to-end)
notebooks/riesgo_grooming_edad_voz.ipynb   entrenamiento, comparación y selección
api/main.py            FastAPI: /health, /model-info, /analyze
scripts/download_pretrained.py             descarga pesos de ECAPA/WavLM (usado en Docker)
scripts/generar_muestras_audio.py          genera muestras_audio/ (un ejemplo escuchable por etapa)
scripts/figuras_reporte.py                 regenera reporte/figuras/ a su ancho de impresión
muestras_audio/        audios de ejemplo versionados + indice.csv (se escuchan con grooming_risk.muestras)
reporte/               reporte.tex (LaTeX, compila con tectonic) y figuras/
models/                best_model.joblib + model_card.json (los genera el notebook)
datasets/README.md         dónde conseguir y cómo ordenar cada corpus
tests/                 pytest de la lógica de riesgo
```

## Modelos comparados (ver justificación en el notebook, sección 5)
- **A** acústico clásico (F0, formantes, MFCC) + HistGradientBoosting
- **B** embeddings ECAPA-TDNN + SVM RBF
- **C** WavLM-base-plus (capas 3–8) + regresión logística
- **D** audEERING wav2vec2 edad/género, solo como referencia zero-shot (licencia NC, no se despliega)

Todos con `class_weight="balanced"`, hiperparámetros en validación y calibración sigmoide.
El ganador (F1-macro por hablante en test, desempate por recall de `nino`) se guarda en
`models/best_model.joblib`; la API carga el extractor que indique el bundle.

## Comandos
```bash
# entorno (probado con Python 3.12; evitar 3.14 por compatibilidad de torch/speechbrain)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
jupyter lab notebooks/riesgo_grooming_edad_voz.ipynb

# API local (requiere models/best_model.joblib)
uvicorn api.main:app --reload
curl -F "file=@conversacion.wav" http://localhost:8000/analyze

# Docker
docker compose up --build
```

## Convenciones
- Código y comentarios en español; nombres de clases sin tildes (`nino`).
- Partición siempre **por hablante** (`StratifiedGroupKFold`); nunca mezclar un hablante entre splits.
- Toda aumentación se aplica por igual a todas las clases y nunca altera el tono (no pitch-shift).
- Al añadir un corpus: nuevo cargador en `datasets.py` que devuelva `path, speaker_id, age, age_class, source`
  y prefije `speaker_id` con el corpus (lo hace `_finish`).

## Limitaciones conocidas
- `teens` de Common Voice abarca 13–19 años (etiqueta ruidosa).
- Sesgo de canal entre corpus: revisar la auditoría de la sección 7 del notebook.
- La diarización ligera falla con habla solapada y con voces muy parecidas.
- La evaluación con conversaciones simuladas es optimista; falta validación con conversaciones reales anotadas.
