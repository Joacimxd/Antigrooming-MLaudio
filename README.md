# Estimador de riesgo de grooming por edad de voz

> **Solo es una alerta / estimador de riesgo, no un decisor definitivo.** Detecta la probable
> presencia de un menor hablando con un adulto, algo que ocurre en innumerables conversaciones
> legítimas. Toda alerta requiere revisión humana. Ver [CLAUDE.md](CLAUDE.md).

## Resumen
1. **Diarización**: separa a los hablantes del audio.
2. **Edad por hablante**: probabilidades `nino` (≤12), `adolescente` (13–17), `adulto` (≥18).
3. **Riesgo**: niño + adulto con certeza → 90 %; adolescente + adulto → 60 %; escalado por la incertidumbre.

Se comparan tres modelos (acústico + GBM, ECAPA + SVM, WavLM + regresión logística) y uno de
referencia; el mejor se exporta automáticamente para la API.

## Escuchar el sistema sin entrenar
`muestras_audio/` trae ejemplos de cada etapa (preprocesamiento, los 12 filtros, moduladores de voz y
conversaciones simuladas con su diarización y su riesgo). Se escuchan en las celdas «Escucha» de los
notebooks o abriendo los archivos; ver [muestras_audio/README.md](muestras_audio/README.md).

## Pasos
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Descarga y ordena los corpus → datasets/README.md
# 2. Entrena: ejecuta el notebook completo (primero con SUBSAMPLE=600 para probar)
jupyter lab notebooks/riesgo_grooming_edad_voz.ipynb

# 3. API local
uvicorn api.main:app --reload       # docs en http://localhost:8000/docs

# 4. Docker (después de entrenar; copia models/ a la imagen)
docker compose up --build
curl -F "file=@conversacion.wav" "http://localhost:8000/analyze"
curl -F "file=@llamada.mp3" "http://localhost:8000/analyze?num_speakers=2"
```

## Respuesta de ejemplo
`voice_modulation` es una bandera **informativa** por hablante (detector de moduladores de voz, sección 10
del notebook): no cambia el riesgo ni las probabilidades de edad; si se activa, la edad estimada de ese
hablante es menos fiable y conviene priorizar su revisión humana. Es `null` si no existe
`models/detector_modulador.joblib`.

```json
{
  "risk_score": 0.83,
  "risk_percent": 83.0,
  "risk_level": "alto",
  "minor_speaker": 1,
  "adult_speaker": 0,
  "num_speakers": 2,
  "speakers": [
    {"speaker": 0, "duration_sec": 14.2, "probabilities": {"nino": 0.02, "adolescente": 0.05, "adulto": 0.93}, "predicted_age_group": "adulto", "segments_sec": [[0.0, 4.1], ...],
     "voice_modulation": {"probability": 0.004, "possible_modulation": false, "threshold": 0.369}},
    {"speaker": 1, "duration_sec": 9.8,  "probabilities": {"nino": 0.88, "adolescente": 0.09, "adulto": 0.03}, "predicted_age_group": "nino", "segments_sec": [[4.5, 7.9], ...],
     "voice_modulation": {"probability": 0.001, "possible_modulation": false, "threshold": 0.369}}
  ],
  "disclaimer": "Este resultado es solo una ALERTA / estimación probabilística de riesgo ..."
}
```
