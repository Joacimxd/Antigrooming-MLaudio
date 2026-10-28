"""Estimador de riesgo de grooming basado en la edad aparente de los hablantes.

IMPORTANTE: es una herramienta de ALERTA, no un decisor. Ver GUIA_PROYECTO.md.
"""
from .config import CLASSES, DISCLAIMER, age_to_class
from .risk import compute_risk

__all__ = ["CLASSES", "DISCLAIMER", "age_to_class", "compute_risk"]
