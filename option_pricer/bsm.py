"""Silnik 1: Opcja Europejska – analityczny model Black-Scholes-Merton (BSM)."""

import math

from .models import Rynek, norm_cdf, norm_pdf


def wycen_bsm(rynek: Rynek, K: float, q: float, typ: str) -> float:
    S, T, r, sigma = rynek.S, rynek.T, rynek.r, rynek.sigma
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    if typ == "call":
        return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)


def greki_bsm(rynek: Rynek, K: float, q: float, typ: str) -> dict[str, float]:
    S, T, r, sigma = rynek.S, rynek.T, rynek.r, rynek.sigma
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    dq, dr, pdf = math.exp(-q * T), math.exp(-r * T), norm_pdf(d1)

    gamma = dq * pdf / (S * sigma * sqrt_T)
    vega = S * dq * pdf * sqrt_T * 0.01
    if typ == "call":
        delta = dq * norm_cdf(d1)
        theta = (
            -(S * dq * pdf * sigma) / (2.0 * sqrt_T)
            - r * K * dr * norm_cdf(d2)
            + q * S * dq * norm_cdf(d1)
        )
        rho = K * T * dr * norm_cdf(d2) * 0.01
    else:
        delta = dq * (norm_cdf(d1) - 1.0)
        theta = (
            -(S * dq * pdf * sigma) / (2.0 * sqrt_T)
            + r * K * dr * norm_cdf(-d2)
            - q * S * dq * norm_cdf(-d1)
        )
        rho = -K * T * dr * norm_cdf(-d2) * 0.01

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta / 365.0,
        "rho": rho,
    }
