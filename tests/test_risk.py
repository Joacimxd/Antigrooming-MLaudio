import pytest

from grooming_risk.config import RISK_FLOOR, age_to_class
from grooming_risk.risk import compute_risk

CHILD = {"nino": 1.0, "adolescente": 0.0, "adulto": 0.0}
TEEN = {"nino": 0.0, "adolescente": 1.0, "adulto": 0.0}
ADULT = {"nino": 0.0, "adolescente": 0.0, "adulto": 1.0}


@pytest.mark.parametrize("age,cls", [(5, "nino"), (12, "nino"), (13, "adolescente"),
                                     (17, "adolescente"), (18, "adulto"), (None, None), ("x", None)])
def test_age_to_class(age, cls):
    assert age_to_class(age) == cls


def test_child_with_adult_is_90_percent():
    r = compute_risk([ADULT, CHILD])
    assert r["score"] == pytest.approx(0.90)
    assert r["level"] == "alto"
    assert (r["minor_speaker"], r["adult_speaker"]) == (1, 0)


def test_teen_with_adult_is_60_percent():
    assert compute_risk([TEEN, ADULT])["score"] == pytest.approx(0.60)


@pytest.mark.parametrize("speakers", [[ADULT, ADULT], [CHILD, CHILD], [CHILD], [], [ADULT]])
def test_no_minor_adult_pair_gives_floor(speakers):
    r = compute_risk(speakers)
    assert r["score"] == RISK_FLOOR and r["level"] == "bajo"


def test_uncertainty_scales_risk():
    unsure_child = {"nino": 0.5, "adolescente": 0.2, "adulto": 0.3}
    r = compute_risk([unsure_child, ADULT])
    assert r["score"] == pytest.approx(0.5 * 0.9 + 0.2 * 0.6)
