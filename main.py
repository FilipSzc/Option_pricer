"""Start aplikacji do wyceny opcji (BSM / CRR / Monte Carlo)."""

from option_pricer.gui import KalkulatorOpcjiGUI


def main() -> None:
    app = KalkulatorOpcjiGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
