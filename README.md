# Kalkulator Wyceny Opcji (BSM / CRR / Monte Carlo)

Aplikacja do wyceny i analizy wrażliwości (Greki) trzech rodzaji opcji — każdą inną, dedykowaną metodą wyceny:

1. **Opcja Europejska** – analityczny model Black-Scholes-Merton (BSM).
2. **Opcja Amerykańska** – drzewo dwumianowe Cox-Ross-Rubinstein (CRR)
3. **Opcja Azjatycka** – symulacje Monte Carlo i zmienną kontrolną

## Struktura projektu

```text
option-pricer/
├── main.py                  # uruchomienie aplikacji
├── requirements.txt         # pakiety
├── option_pricer/
│   ├── __init__.py          
│   ├── models.py            # Struktury (Rynek, WynikMC) i funkcje pomocnicze
│   ├── bsm.py               # Wycena i Greki BSM (opcja europejska)
│   ├── crr.py               # Drzewo dwumianowe CRR (opcja amerykańska)
│   ├── asian_mc.py          # Monte Carlo + Control Variate (opcja azjatycka)
│   ├── plotting.py          # Wykresy 
│   └── gui.py               # Interfejs 
└── tests/
    └── test_pricers.py      # Testy 
```

## Uruchomienie

```bash
python -m venv .venv
.venv\Scripts\activate


pip install -r requirements.txt
python main.py
```

### Uruchomienie testów

```bash
python -m unittest discover -s tests -v
```

## Założenia i ograniczenia

- **Stała zmienność ($\sigma$):** model zakłada klasyczny geometryczny ruch Browna bez *volatility smile* (jednakowa zmienność dla każdego strike'u $K$ i terminu $T$).
- **Ciągła dywidenda ($q$):** dywidenda jest modelowana jako ciągła stopa procentowa, a nie konkretne wypłaty gotówkowe w wybranych dniach.
- **Interpretacja Thety dla opcji azjatyckiej:** Theta jest liczona przez zmianę całkowitego czasu trwania opcji ($T \pm h_T$) przy stałej liczbie obserwacji $M$, co proporcjonalnie wydłuża lub skraca odstępy między fixingami ($T/M$). Oznacza to wrażliwość ceny na długość nowo zawieranego kontraktu, a nie dokładny dzienny upływ czasu dla opcji o już ustalonych datach obserwacji.
