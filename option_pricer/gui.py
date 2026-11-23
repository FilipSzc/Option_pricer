"""Interfejs"""

from dataclasses import replace
from functools import partial
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np

from .asian_mc import (
    greki_numeryczne_mc,
    wycen_azjatycka_geom_dyskretna,
    wycen_azjatycka_mc,
)
from .bsm import greki_bsm, wycen_bsm
from .crr import greki_crr, wycen_amerykanska_crr
from .models import KATALOG_PROJEKTU, Rynek
from .plotting import rysuj_deterministyczny, rysuj_monte_carlo


class KalkulatorOpcjiGUI(tk.Tk):
    OPCJE = [
        "Europejska (Black-Scholes-Merton)",
        "Amerykańska (Drzewo dwumianowe CRR)",
        "Azjatycka (Monte Carlo + Control Variate)",
    ]

    def __init__(self) -> None:
        super().__init__()
        self.title("Kalkulator Wyceny Opcji (BSM / CRR / Monte Carlo)")
        self.geometry("1380x820")
        self._zbuduj_ui()
        self._odswiez_specyficzne()
        self.wycen()

    def _zbuduj_ui(self) -> None:
        glowny = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        glowny.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        lewy = ttk.Frame(glowny, width=460)
        glowny.add(lewy, weight=0)

        ramka_typ = ttk.LabelFrame(lewy, text="1. Wybór Opcji", padding=8)
        ramka_typ.pack(fill=tk.X, pady=(0, 6))
        self.var_opcja = tk.StringVar(value=self.OPCJE[0])
        self.var_kierunek = tk.StringVar(value="call")

        combo = ttk.Combobox(
            ramka_typ,
            textvariable=self.var_opcja,
            values=self.OPCJE,
            state="readonly",
            width=40,
        )
        combo.grid(row=0, column=0, columnspan=2, sticky=tk.EW, pady=2)
        combo.bind("<<ComboboxSelected>>", self._odswiez_specyficzne)
        ttk.Radiobutton(
            ramka_typ, text="Call (Kupno)", variable=self.var_kierunek, value="call"
        ).grid(row=1, column=0, sticky=tk.W, pady=4)
        ttk.Radiobutton(
            ramka_typ, text="Put (Sprzedaż)", variable=self.var_kierunek, value="put"
        ).grid(row=1, column=1, sticky=tk.W, pady=4)
        ramka_typ.columnconfigure(0, weight=1)

        ramka_wsp = ttk.LabelFrame(lewy, text="2. Parametry Rynkowe", padding=8)
        ramka_wsp.pack(fill=tk.X, pady=(0, 6))
        self.var_S = tk.StringVar(value="100.0")
        self.var_K = tk.StringVar(value="100.0")
        self.var_T = tk.StringVar(value="1.0")
        self.var_r = tk.StringVar(value="5.0")
        self.var_q = tk.StringVar(value="0.0")
        self.var_sigma = tk.StringVar(value="20.0")
        for idx, (etyk, var) in enumerate(
            [
                ("Cena spot S0:", self.var_S),
                ("Cena wykonania K:", self.var_K),
                ("Czas T (lata):", self.var_T),
                ("Stopa wolna od ryzyka r (%):", self.var_r),
                ("Stopa dywidendy q (%):", self.var_q),
                ("Zmienność σ (%):", self.var_sigma),
            ]
        ):
            ttk.Label(ramka_wsp, text=etyk).grid(row=idx, column=0, sticky=tk.W, pady=2)
            ttk.Entry(ramka_wsp, textvariable=var, width=16).grid(
                row=idx, column=1, sticky=tk.E, pady=2
            )
        ramka_wsp.columnconfigure(0, weight=1)

        self.ramka_spec = ttk.LabelFrame(lewy, text="3. Parametry Metody", padding=8)
        self.ramka_spec.pack(fill=tk.X, pady=(0, 6))
        self.var_kroki = tk.StringVar(value="300")
        self.var_obs = tk.StringVar(value="12")
        self.var_sciezki = tk.StringVar(value="20000")
        self.var_seed = tk.StringVar(value="42")

        btn = ttk.Frame(lewy)
        btn.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(btn, text="Wyceń i generuj wykres", command=self.wycen).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4)
        )
        ttk.Button(btn, text="Zapisz PNG", command=self.zapisz_png).pack(side=tk.LEFT)

        ramka_w = ttk.LabelFrame(lewy, text="Wyniki Wyceny i Greki", padding=6)
        ramka_w.pack(fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(ramka_w)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.pole = tk.Text(
            ramka_w,
            font=("Consolas", 9),
            wrap=tk.NONE,
            yscrollcommand=scroll.set,
            width=58,
        )
        self.pole.pack(fill=tk.BOTH, expand=True)
        scroll.config(command=self.pole.yview)

        prawy = ttk.Frame(glowny)
        glowny.add(prawy, weight=1)
        self.figura = Figure(figsize=(9.2, 7.4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figura, master=prawy)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, prawy).update()

    def _odswiez_specyficzne(self, _event=None) -> None:
        for w in self.ramka_spec.winfo_children():
            w.destroy()
        self.ramka_spec.columnconfigure(0, weight=1)
        opcja = self.var_opcja.get()
        if opcja.startswith("Europejska"):
            ttk.Label(
                self.ramka_spec,
                text="Brak dodatkowych parametrów (wzór zamknięty BSM).",
            ).grid(row=0, column=0, sticky=tk.W)
        elif opcja.startswith("Amerykańska"):
            ttk.Label(self.ramka_spec, text="Liczba kroków drzewa CRR (N):").grid(
                row=0, column=0, sticky=tk.W, pady=2
            )
            ttk.Entry(self.ramka_spec, textvariable=self.var_kroki, width=16).grid(
                row=0, column=1, sticky=tk.E, pady=2
            )
        else:
            for r, (etyk, var) in enumerate(
                [
                    ("Liczba obserwacji (M):", self.var_obs),
                    ("Liczba ścieżek MC (parzysta):", self.var_sciezki),
                    ("Ziarno losowości (seed):", self.var_seed),
                ]
            ):
                ttk.Label(self.ramka_spec, text=etyk).grid(
                    row=r, column=0, sticky=tk.W, pady=2
                )
                ttk.Entry(self.ramka_spec, textvariable=var, width=16).grid(
                    row=r, column=1, sticky=tk.E, pady=2
                )

    @staticmethod
    def _f(var: tk.StringVar) -> float:
        return float(var.get().strip().replace(",", "."))

    def wycen(self) -> None:
        try:
            rynek = Rynek(
                S=self._f(self.var_S),
                T=self._f(self.var_T),
                r=self._f(self.var_r) / 100.0,
                sigma=self._f(self.var_sigma) / 100.0,
            )
            K = self._f(self.var_K)
            q = self._f(self.var_q) / 100.0
            typ = self.var_kierunek.get()
            opcja = self.var_opcja.get()
            if rynek.S <= 0 or K <= 0 or rynek.T <= 0 or rynek.sigma <= 0:
                raise ValueError("Parametry S0, K, T oraz σ muszą być większe od zera.")

            info: list[str] = []
            siatka_S = np.linspace(
                max(0.05, 0.55 * min(rynek.S, K)), 1.45 * max(rynek.S, K), 60
            )

            if opcja.startswith("Europejska"):
                cena = wycen_bsm(rynek, K, q, typ)
                g = greki_bsm(rynek, K, q, typ)
                info.append("Greki i wykres Delta/Gamma: dokładne wzory analityczne BSM.")

                ceny_s = np.array(
                    [wycen_bsm(replace(rynek, S=float(s)), K, q, typ) for s in siatka_S]
                )
                greki_siatka = [
                    greki_bsm(replace(rynek, S=float(s)), K, q, typ) for s in siatka_S
                ]
                delty_s = np.array([gr["delta"] for gr in greki_siatka])
                gammy_s = np.array([gr["gamma"] for gr in greki_siatka])

                rysuj_deterministyczny(
                    self.figura,
                    f"{opcja} – {typ.upper()}",
                    typ,
                    rynek.S,
                    K,
                    cena,
                    siatka_S,
                    ceny_s,
                    delty_s,
                    gammy_s,
                )

            elif opcja.startswith("Amerykańska"):
                kroki = int(self.var_kroki.get())
                if kroki < 10:
                    raise ValueError("Liczba kroków CRR musi wynosić >= 10.")
                cena, g = greki_crr(rynek, K, q, typ, kroki)
                cena_eu = wycen_bsm(rynek, K, q, typ)
                info.append(f"Odpowiednik europejski (BSM):   {cena_eu:.4f}")
                info.append(f"Premia wcześniejszego wykon.:   {cena - cena_eu:+.4f}")
                info.append(
                    f"Delta/Gamma/Theta z węzłów CRR (N={kroki}); Vega/Rho numerycznie."
                )

                wyniki_s = [
                    wycen_amerykanska_crr(
                        replace(rynek, S=float(s)), K, q, typ, kroki
                    )
                    for s in siatka_S
                ]
                ceny_am = np.array([w[0] for w in wyniki_s])
                delty_am = np.array([w[1] for w in wyniki_s])
                gammy_am = np.array([w[2] for w in wyniki_s])
                ceny_eu = np.array(
                    [wycen_bsm(replace(rynek, S=float(s)), K, q, typ) for s in siatka_S]
                )

                rysuj_deterministyczny(
                    self.figura,
                    f"{opcja} – {typ.upper()}",
                    typ,
                    rynek.S,
                    K,
                    cena,
                    siatka_S,
                    ceny_am,
                    delty_am,
                    gammy_am,
                    ceny_eu=ceny_eu,
                )

            else:
                n_obs = int(self.var_obs.get())
                n_sciezek = int(self.var_sciezki.get())
                seed = int(self.var_seed.get())
                if n_obs < 1:
                    raise ValueError("Liczba fixingów M musi być >= 1.")
                f_mc = partial(
                    wycen_azjatycka_mc,
                    K=K,
                    q=q,
                    typ=typ,
                    n_obs=n_obs,
                    n_sciezek=n_sciezek,
                    seed=seed,
                )
                mc = f_mc(rynek)
                cena = mc.cena
                g = greki_numeryczne_mc(f_mc, rynek)
                cena_geom = wycen_azjatycka_geom_dyskretna(rynek, K, q, typ, n_obs)

                ogr = "Dolne ogr." if typ == "call" else "Górne ogr."
                relacja = (
                    "cena arytm. >= geom., z AM >= GM"
                    if typ == "call"
                    else "cena arytm. <= geom., z AM >= GM"
                )
                zysk = (
                    (mc.se_naiwne / mc.se_antytetyczne)
                    if mc.se_antytetyczne > 1e-15
                    else 1.0
                )

                info.append(
                    f"SE (pary antytetyczne):         {mc.se_antytetyczne:.5f} (naiwne: {mc.se_naiwne:.5f}, {zysk:.2f}x)"
                )
                info.append(
                    f"95% przedział ufności:          [{cena - 1.96 * mc.se_antytetyczne:.4f}; {cena + 1.96 * mc.se_antytetyczne:.4f}]"
                )
                info.append(f"{ogr} (dyskretna geom.):   {cena_geom:.4f} ({relacja})")
                info.append(
                    "Uwaga: Theta mierzy zmianę tenoru T (skaluje siatkę fixingów T/M)."
                )
                rysuj_monte_carlo(self.figura, mc, typ, K, cena_geom)

        except ValueError as exc:
            messagebox.showerror("Błędne parametry", str(exc))
            return

        linie = [
            f"Opcja: {opcja} | Kierunek: {typ.upper()}",
            "-" * 56,
            f"CENA OPCJI:                     {cena:.4f}",
            *info,
            "",
            "=== WRAŻLIWOŚCI (GREKI) ===",
            f"Delta (dV/dS):                  {g['delta']:+.5f}",
            f"Gamma (d2V/dS2):                {g['gamma']:+.5f}",
            f"Vega  (za +1 pp σ):             {g['vega']:+.5f}",
            f"Theta (dzienna, /365):          {g['theta']:+.5f}",
            f"Rho   (za +1 pp r):             {g['rho']:+.5f}",
        ]
        self.pole.delete("1.0", tk.END)
        self.pole.insert("1.0", "\n".join(linie))
        self.canvas.draw()

    def zapisz_png(self) -> None:
        sciezka = filedialog.asksaveasfilename(
            initialdir=str(KATALOG_PROJEKTU),
            initialfile="wycena_opcji.png",
            defaultextension=".png",
            filetypes=[("Plik PNG", "*.png")],
        )
        if sciezka:
            self.figura.savefig(sciezka, dpi=150, bbox_inches="tight")
            messagebox.showinfo("Zapisano", f"Zapisano wykres:\n{sciezka}")
