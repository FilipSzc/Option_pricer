"""Struktury danych oraz funkcje pomocnicze"""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

KATALOG_PROJEKTU: Path = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Rynek:
    S: float
    T: float
    r: float
    sigma: float


@dataclass
class WynikMC:
    cena: float
    se_antytetyczne: float
    se_naiwne: float
    estymatory: np.ndarray
    sciezki_plot: np.ndarray
    siatka_t: np.ndarray
    srednie_arytm: np.ndarray
    wyplaty_zdysk: np.ndarray


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def payoff_vec(x: np.ndarray, K: float, typ: str) -> np.ndarray:
    return np.maximum(x - K, 0.0) if typ == "call" else np.maximum(K - x, 0.0)


def se_z_par(estymatory: np.ndarray) -> float:
    """Błąd standardowy policzony z uśrednionych par antytetycznych (z, -z)."""
    pary = 0.5 * (estymatory[0::2] + estymatory[1::2])
    return float(np.std(pary, ddof=1) / math.sqrt(len(pary)))
