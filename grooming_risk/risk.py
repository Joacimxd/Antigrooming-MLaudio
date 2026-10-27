"""Cálculo del puntaje de riesgo a partir de las probabilidades de edad por hablante.

Regla: para cada par ordenado de hablantes distintos (i, j)

    riesgo(i, j) = Σ_c  P_i(c) · P_j(adulto) · W[c, adulto]

con W[nino, adulto] = 0.90 y W[adolescente, adulto] = 0.60. El riesgo del audio
es el máximo sobre todos los pares (basta UNA pareja menor–adulto para alertar),
con un piso de RISK_FLOOR. Así, si el modelo está seguro de que hay un niño y un
adulto distintos, el riesgo es 90 %; si duda, el riesgo baja proporcionalmente.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from .config import RISK_FLOOR, RISK_LEVELS, RISK_WEIGHTS


def risk_level(score: float) -> str:
    for upper, name in RISK_LEVELS:
        if score < upper:
            return name
    return RISK_LEVELS[-1][1]


def compute_risk(speakers: Sequence[Mapping[str, float]]) -> dict:
    """`speakers`: lista de dicts {clase: probabilidad} (una entrada por hablante)."""
    best = {"score": RISK_FLOOR, "minor_speaker": None, "adult_speaker": None}
    for i, p_i in enumerate(speakers):
        for j, p_j in enumerate(speakers):
            if i == j:
                continue
            score = sum(p_i.get(minor, 0.0) * p_j.get(adult, 0.0) * w
                        for (minor, adult), w in RISK_WEIGHTS.items())
            if score > best["score"]:
                best = {"score": score, "minor_speaker": i, "adult_speaker": j}

    score = round(float(best["score"]), 4)
    return {**best, "score": score, "level": risk_level(score)}
