"""Silnik 3: Opcja Azjatycka (arytm.) – Monte Carlo z antytetyką i zmienną kontrolną."""

from dataclasses import replace
from functools import partial
import math
from typing import Callable

import numpy as np

from .models import Rynek, WynikMC, norm_cdf, payoff_vec, se_z_par


def wycen_azjatycka_geom_dyskretna(
    rynek: Rynek, K: float, q: float, typ: str, n_obs: int
) -> float:
    """Dokładna cena dyskretnej opcji azjatyckiej o średniej geometrycznej (zmienna kontrolna)."""
    S, T, r, sigma = rynek.S, rynek.T, rynek.r, rynek.sigma
    M = n_obs
    mu_g = (r - q - 0.5 * sigma ** 2) * T * (M + 1) / (2.0 * M)
    var_g = (sigma ** 2) * T * (M + 1) * (2 * M + 1) / (6.0 * M ** 2)
    std_g = math.sqrt(var_g)
    F_g = S * math.exp(mu_g + 0.5 * var_g)
    d1 = (math.log(S / K) + mu_g + var_g) / std_g
    d2 = d1 - std_g
    dysk = math.exp(-r * T)
    if typ == "call":
        return dysk * (F_g * norm_cdf(d1) - K * norm_cdf(d2))
    return dysk * (K * norm_cdf(-d2) - F_g * norm_cdf(-d1))


def wycen_azjatycka_mc(
    rynek: Rynek,
    K: float,
    q: float,
    typ: str,
    n_obs: int,
    n_sciezek: int,
    seed: int,
) -> WynikMC:
    if n_sciezek < 500 or n_sciezek % 2 != 0:
        raise ValueError("Liczba ścieżek MC musi być parzysta i >= 500.")

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_sciezek // 2, n_obs))
    z_pelne = np.empty((n_sciezek, n_obs), dtype=np.float64)
    z_pelne[0::2] = z
    z_pelne[1::2] = -z

    dt = rynek.T / n_obs
    przyrosty = (
        (rynek.r - q - 0.5 * rynek.sigma ** 2) * dt
        + rynek.sigma * math.sqrt(dt) * z_pelne
    )
    log_s = np.empty((n_sciezek, n_obs + 1), dtype=np.float64)
    log_s[:, 0] = math.log(rynek.S)
    np.cumsum(przyrosty, axis=1, out=log_s[:, 1:])
    log_s[:, 1:] += math.log(rynek.S)
    sciezki = np.exp(log_s)

    fixingi = sciezki[:, 1:]
    dysk = math.exp(-rynek.r * rynek.T)
    sr_arytm = np.mean(fixingi, axis=1)
    sr_geom = np.exp(np.mean(np.log(fixingi), axis=1))

    wyp_a = dysk * payoff_vec(sr_arytm, K, typ)
    wyp_g = dysk * payoff_vec(sr_geom, K, typ)
    cena_g = wycen_azjatycka_geom_dyskretna(rynek, K, q, typ, n_obs)

    pary_a = 0.5 * (wyp_a[0::2] + wyp_a[1::2])
    pary_g = 0.5 * (wyp_g[0::2] + wyp_g[1::2])
    var_g = float(np.var(pary_g, ddof=1))
    beta = (
        float(np.cov(pary_a, pary_g, ddof=1)[0, 1] / var_g)
        if var_g > 1e-18
        else 0.0
    )
    estymatory = wyp_a + beta * (cena_g - wyp_g)

    return WynikMC(
        cena=max(float(np.mean(estymatory)), 0.0),
        se_antytetyczne=se_z_par(estymatory),
        se_naiwne=float(np.std(estymatory, ddof=1) / math.sqrt(n_sciezek)),
        estymatory=estymatory,
        # 50 niezależnych ścieżek (po jednej z każdej pary antytetycznej), bez sztucznego wygładzania średniej
        sciezki_plot=sciezki[0:100:2],
        siatka_t=np.linspace(0.0, rynek.T, n_obs + 1),
        srednie_arytm=sr_arytm,
        wyplaty_zdysk=wyp_a,
    )


def _cena_mc(f_mc: Callable[[Rynek], WynikMC], rynek: Rynek) -> float:
    return f_mc(rynek).cena


def greki_numeryczne_mc(
    f_mc: Callable[[Rynek], WynikMC], rynek: Rynek
) -> dict[str, float]:
    """Greki przez różnice centralne ze wspólnym ziarnem losowości (Common Random Numbers)."""
    f = partial(_cena_mc, f_mc)
    v0 = f(rynek)
    h_S = max(0.01 * rynek.S, 1e-4)
    h_sig = min(0.005, rynek.sigma * 0.1)
    h_r = 0.0025
    h_T = min(0.01, rynek.T * 0.1)

    v_p = f(replace(rynek, S=rynek.S + h_S))
    v_m = f(replace(rynek, S=rynek.S - h_S))
    return {
        "delta": (v_p - v_m) / (2.0 * h_S),
        "gamma": (v_p - 2.0 * v0 + v_m) / (h_S ** 2),
        "vega": (
            f(replace(rynek, sigma=rynek.sigma + h_sig))
            - f(replace(rynek, sigma=rynek.sigma - h_sig))
        )
        / (2.0 * h_sig)
        * 0.01,
        "theta": -(
            f(replace(rynek, T=rynek.T + h_T)) - f(replace(rynek, T=rynek.T - h_T))
        )
        / (2.0 * h_T)
        / 365.0,
        "rho": (
            f(replace(rynek, r=rynek.r + h_r)) - f(replace(rynek, r=rynek.r - h_r))
        )
        / (2.0 * h_r)
        * 0.01,
    }
