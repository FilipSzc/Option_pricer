"""Silnik 2: Opcja Amerykańska – drzewo dwumianowe Cox-Ross-Rubinstein (CRR)."""

from dataclasses import replace
import math

import numpy as np

from .models import Rynek, payoff_vec


def wycen_amerykanska_crr(
    rynek: Rynek, K: float, q: float, typ: str, kroki: int
) -> tuple[float, float, float, float]:
    """Zwraca (cena, delta, gamma, theta_1d) wyznaczone bezpośrednio z węzłów drzewa."""
    S, T, r, sigma = rynek.S, rynek.T, rynek.r, rynek.sigma
    dt = T / kroki
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    dysk = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not (0.0 < p < 1.0):
        raise ValueError(
            "Niespełniony warunek 0 < p < 1 w drzewie CRR. Zwiększ liczbę kroków."
        )

    j = np.arange(kroki + 1)
    wart = payoff_vec(S * (u ** j) * (d ** (kroki - j)), K, typ)
    v20 = v21 = v22 = v10 = v11 = 0.0

    for i in range(kroki - 1, -1, -1):
        wart = dysk * (p * wart[1:] + (1.0 - p) * wart[:-1])
        idx = np.arange(i + 1)
        wart = np.maximum(wart, payoff_vec(S * (u ** idx) * (d ** (i - idx)), K, typ))
        if i == 2:
            v20, v21, v22 = float(wart[0]), float(wart[1]), float(wart[2])
        elif i == 1:
            v10, v11 = float(wart[0]), float(wart[1])

    v00 = float(wart[0])
    s10, s11 = S * d, S * u
    s20, s21, s22 = S * (d ** 2), S, S * (u ** 2)
    delta = (v11 - v10) / (s11 - s10)
    gamma = (
        (v22 - v21) / (s22 - s21) - (v21 - v20) / (s21 - s20)
    ) / (0.5 * (s22 - s20))
    theta_1d = ((v21 - v00) / (2.0 * dt)) / 365.0
    return v00, delta, gamma, theta_1d


def greki_crr(
    rynek: Rynek, K: float, q: float, typ: str, kroki: int
) -> tuple[float, dict[str, float]]:
    """Delta, Gamma i Theta z węzłów drzewa; Vega i Rho metodą różnic centralnych."""
    cena, delta, gamma, theta = wycen_amerykanska_crr(rynek, K, q, typ, kroki)
    h_sig, h_r = min(0.005, rynek.sigma * 0.1), 0.0025
    vega = (
        wycen_amerykanska_crr(
            replace(rynek, sigma=rynek.sigma + h_sig), K, q, typ, kroki
        )[0]
        - wycen_amerykanska_crr(
            replace(rynek, sigma=rynek.sigma - h_sig), K, q, typ, kroki
        )[0]
    ) / (2.0 * h_sig) * 0.01
    rho = (
        wycen_amerykanska_crr(replace(rynek, r=rynek.r + h_r), K, q, typ, kroki)[0]
        - wycen_amerykanska_crr(replace(rynek, r=rynek.r - h_r), K, q, typ, kroki)[0]
    ) / (2.0 * h_r) * 0.01
    return cena, {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }
