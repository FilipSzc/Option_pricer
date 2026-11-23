"""Rysowanie wykresów"""

from matplotlib.figure import Figure
import numpy as np

from .models import WynikMC, payoff_vec


def rysuj_deterministyczny(
    fig: Figure,
    tytul: str,
    typ: str,
    S0: float,
    K: float,
    cena0: float,
    siatka_S: np.ndarray,
    ceny: np.ndarray,
    delty: np.ndarray,
    gammy: np.ndarray,
    ceny_eu: np.ndarray | None = None,
) -> None:
    fig.clear()
    ax1, ax2 = fig.subplots(2, 1, sharex=True)

    ax1.plot(siatka_S, ceny, color="navy", linewidth=2.2, label="Cena opcji V(S)")
    ax1.plot(
        siatka_S,
        payoff_vec(siatka_S, K, typ),
        color="gray",
        linestyle="--",
        label="Payoff w wygaśnięciu",
    )
    if ceny_eu is not None:
        ax1.plot(
            siatka_S,
            ceny_eu,
            color="darkorange",
            linestyle=":",
            linewidth=1.8,
            label="Odpowiednik europejski (BSM)",
        )
    ax1.scatter(
        [S0],
        [cena0],
        color="crimson",
        s=50,
        zorder=5,
        label=f"Punkt bieżący ({S0:.1f}; {cena0:.2f})",
    )
    ax1.axvline(K, color="dimgray", linestyle=":", alpha=0.75, label=f"Strike K = {K:.1f}")
    ax1.set_title(tytul, fontsize=11, fontweight="bold")
    ax1.set_ylabel("Wartość opcji")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(siatka_S, delty, color="teal", linewidth=2, label="Delta")
    ax2.axvline(S0, color="crimson", linestyle=":", alpha=0.75, label=f"Spot S0 = {S0:.1f}")
    ax2.set_xlabel("Cena instrumentu bazowego S")
    ax2.set_ylabel("Delta", color="teal")
    ax2.tick_params(axis="y", labelcolor="teal")
    ax2.grid(True, alpha=0.3)

    ax2_r = ax2.twinx()
    ax2_r.plot(siatka_S, gammy, color="darkorange", linestyle="--", linewidth=1.6, label="Gamma")
    ax2_r.set_ylabel("Gamma", color="darkorange")
    ax2_r.tick_params(axis="y", labelcolor="darkorange")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=8)
    ax2.set_title("Profil Delta i Gamma w funkcji ceny spot S", fontsize=10)
    fig.tight_layout()


def rysuj_monte_carlo(
    fig: Figure, wynik: WynikMC, typ: str, K: float, cena_geom: float
) -> None:
    fig.clear()
    axes = fig.subplots(2, 2)
    fig.suptitle(
        f"Opcja Azjatycka ({typ.upper()}) – Symulacje Monte Carlo + Control Variate",
        fontsize=11,
        fontweight="bold",
    )

    # 1) Przykładowe niezależne ścieżki
    ax_p = axes[0, 0]
    for s in wynik.sciezki_plot:
        ax_p.plot(wynik.siatka_t, s, color="steelblue", alpha=0.25, linewidth=0.8)
    marker = "o" if len(wynik.siatka_t) <= 25 else None
    ax_p.plot(
        wynik.siatka_t,
        np.mean(wynik.sciezki_plot, axis=0),
        color="navy",
        marker=marker,
        markersize=3,
        linewidth=2,
        label="Średnia z 50 niezależnych ścieżek",
    )
    ax_p.axhline(K, color="black", linestyle="--", linewidth=1.2, label=f"Strike K = {K:.1f}")
    ax_p.set_title(
        f"Przykładowe ścieżki ({len(wynik.sciezki_plot)} niezależnych z {len(wynik.estymatory):,})"
    )
    ax_p.set_xlabel("Czas (lata)")
    ax_p.set_ylabel("Cena S(t)")
    ax_p.legend(fontsize=8)
    ax_p.grid(True, alpha=0.3)

    # 2) Zbieżność MC z poprawnym SE par antytetycznych
    ax_c = axes[0, 1]
    pary = 0.5 * (wynik.estymatory[0::2] + wynik.estymatory[1::2])
    n_par = len(pary)
    idx = np.unique(np.linspace(min(50, n_par), n_par, min(200, n_par), dtype=int))
    cs1, cs2 = np.cumsum(pary), np.cumsum(pary ** 2)
    sr = cs1[idx - 1] / idx
    war = np.maximum(
        (cs2[idx - 1] - (cs1[idx - 1] ** 2) / idx) / np.maximum(idx - 1, 1), 0.0
    )
    se = np.sqrt(war / idx)
    os_n = idx * 2

    ogr_lbl = "Dolne ogr." if typ == "call" else "Górne ogr."
    ax_c.plot(os_n, sr, color="navy", linewidth=1.8, label=f"MC + CV ({wynik.cena:.4f})")
    ax_c.fill_between(
        os_n, sr - 1.96 * se, sr + 1.96 * se, color="steelblue", alpha=0.25, label="95% CI (z par)"
    )
    ax_c.axhline(
        cena_geom,
        color="crimson",
        linestyle="--",
        linewidth=1.3,
        label=f"{ogr_lbl} geom. ({cena_geom:.4f})",
    )
    ax_c.set_title("Zbieżność estymatora Monte Carlo")
    ax_c.set_xlabel("Liczba ścieżek N")
    ax_c.set_ylabel("Szacowana cena")
    ax_c.legend(fontsize=8)
    ax_c.grid(True, alpha=0.3)

    # 3) Rozkład średniej arytmetycznej (ITM vs OTM)
    ax_h = axes[1, 0]
    vals = wynik.srednie_arytm
    itm = (vals > K) if typ == "call" else (vals < K)
    p_lo, p_hi = np.percentile(vals, [0.5, 99.5])
    bins = np.linspace(p_lo, max(p_hi, p_lo + 1e-6), 40)
    ax_h.hist(vals[~itm], bins=bins, color="lightgray", edgecolor="gray", label="OTM")
    ax_h.hist(
        vals[itm],
        bins=bins,
        color="seagreen",
        edgecolor="darkgreen",
        alpha=0.75,
        label=f"ITM ({np.mean(itm):.1%})",
    )
    ax_h.axvline(K, color="black", linestyle="--", label=f"K = {K:.1f}")
    ax_h.set_title("Rozkład średniej ceny z fixingów")
    ax_h.set_xlabel("Średnia arytmetyczna ceny")
    ax_h.set_ylabel("Liczba ścieżek")
    ax_h.legend(fontsize=8)
    ax_h.grid(True, alpha=0.3, axis="y")

    # 4) Rozkład dodatnich wypłat zdyskontowanych
    ax_w = axes[1, 1]
    dod = wynik.wyplaty_zdysk[wynik.wyplaty_zdysk > 0.0]
    udzial = len(dod) / len(wynik.wyplaty_zdysk)
    if len(dod) > 5:
        q99 = float(np.percentile(dod, 99.0))
        ax_w.hist(
            dod[dod <= q99],
            bins=35,
            color="darkorange",
            edgecolor="saddlebrown",
            alpha=0.8,
        )
        ax_w.axvline(
            wynik.cena, color="navy", linewidth=2, label=f"Cena opcji: {wynik.cena:.3f}"
        )
        ax_w.legend(fontsize=8)
    else:
        ax_w.text(
            0.5,
            0.5,
            f"Brak wypłat dodatnich (udział: {udzial:.2%})",
            ha="center",
            va="center",
            transform=ax_w.transAxes,
        )
    ax_w.set_title(f"Zdyskontowane wypłaty > 0 (udział: {udzial:.1%})")
    ax_w.set_xlabel("Wypłata zdyskontowana")
    ax_w.set_ylabel("Liczba ścieżek")
    ax_w.grid(True, alpha=0.3, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
