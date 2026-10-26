# Datos

> Los audios de menores son datos sensibles. No los subas a repositorios, no los compartas
> fuera de las licencias de cada corpus y bórralos cuando ya no los necesites.
> Todo `datasets/` (salvo este README) está en `.gitignore`.

Coloca cada corpus nuevo en su carpeta y añádelo en `notebooks/construccion_dataset.ipynb`, que genera el dataset unido `datasets/merged/` (ENNI + Common Voice) usado para entrenar. Las características extraídas se guardan en `datasets/features/`.

```
datasets/raw/
├── common_voice/
│   ├── es/                 # uno o varios idiomas
│   │   ├── validated.tsv
│   │   └── clips/*.mp3
│   └── en/ ...
├── speechocean762/
│   ├── WAVE/SPEAKERxxxx/*.WAV
│   ├── train/{wav.scp, utt2spk, spk2age}
│   └── test/{wav.scp, utt2spk, spk2age}
├── myst/                   # cualquier profundidad: **/<id_estudiante>/<id_sesion>/*.flac
├── extra/
│   ├── nino/<id_hablante>/*.wav
│   ├── adolescente/<id_hablante>/*.wav
│   └── adulto/<id_hablante>/*.wav
└── custom/
    └── metadata.csv        # columnas: path,speaker_id,age   (path relativo al CSV)
```

## Corpus recomendados

| Corpus | Qué aporta | Cómo obtenerlo |
|---|---|---|
| **Mozilla Common Voice** | Adultos y adolescentes (`teens`) en muchos idiomas, incluido español | Mozilla Data Collective (antes commonvoice.mozilla.org/datasets). Descarga el idioma y extrae en `common_voice/<idioma>/` |
| **speechocean762** | ~250 hablantes, mitad niños, con edad exacta y mismo canal que los adultos | OpenSLR recurso 101 (CC BY 4.0). Extrae tal cual en `speechocean762/` |
| **MyST Children's Conversational Speech** | Niños de 8–11 años, habla conversacional | LDC, catálogo LDC2021S05 (licencia de investigación) |
| **CSLU Kids' Speech** | Niños de kínder a 10º grado (≈5–16 años) | LDC2007S18. Organízalo en `extra/<clase>/<hablante>/` según el grado |
| **PF-STAR / CMU Kids** | Más niños (inglés británico / americano) | ELRA / LDC97S63 → `extra/nino/` |
| **TIMIT / VoxCeleb (con edades)** | Adultos con canal limpio / variado | LDC93S1 / anotaciones de edad publicadas para VoxCeleb → `extra/adulto/` o `custom/` |

### Consejos para reducir el sesgo de canal
- Busca que **cada corpus aporte más de una clase** (speechocean762 es ideal).
- Si solo tienes niños de un corpus, añade adultos grabados en condiciones parecidas.
- Revisa la sección 7 del notebook ("Auditoría de sesgo por corpus") tras cada entrenamiento.
