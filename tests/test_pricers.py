"""Testy"""

import math
import unittest

from option_pricer import (
    Rynek,
    greki_bsm,
    greki_crr,
    wycen_amerykanska_crr,
    wycen_azjatycka_geom_dyskretna,
    wycen_azjatycka_mc,
    wycen_bsm,
)


class TestOptionPricers(unittest.TestCase):
    def setUp(self) -> None:
        self.rynek = Rynek(S=100.0, T=1.0, r=0.05, sigma=0.20)
        self.K = 100.0

    def test_bsm_put_call_parity(self) -> None:
        q = 0.02
        call = wycen_bsm(self.rynek, self.K, q, "call")
        put = wycen_bsm(self.rynek, self.K, q, "put")
        lhs = call - put
        rhs = self.rynek.S * math.exp(-q * self.rynek.T) - self.K * math.exp(
            -self.rynek.r * self.rynek.T
        )
        self.assertAlmostEqual(lhs, rhs, places=10)

    def test_bsm_delta_put_call_relation(self) -> None:
        q = 0.0
        g_call = greki_bsm(self.rynek, self.K, q, "call")
        g_put = greki_bsm(self.rynek, self.K, q, "put")
        self.assertAlmostEqual(g_call["delta"] - g_put["delta"], 1.0, places=10)
        self.assertAlmostEqual(g_call["gamma"], g_put["gamma"], places=10)

    def test_american_call_without_dividend_matches_bsm(self) -> None:
        bsm_call = wycen_bsm(self.rynek, self.K, 0.0, "call")
        crr_call, _ = greki_crr(self.rynek, self.K, 0.0, "call", kroki=400)
        self.assertAlmostEqual(crr_call, bsm_call, places=2)

    def test_american_put_has_early_exercise_premium(self) -> None:
        bsm_put = wycen_bsm(self.rynek, self.K, 0.0, "put")
        crr_put = wycen_amerykanska_crr(self.rynek, self.K, 0.0, "put", kroki=300)[0]
        self.assertGreater(crr_put, bsm_put)

    def test_asian_call_exceeds_geometric_lower_bound(self) -> None:
        geom_call = wycen_azjatycka_geom_dyskretna(self.rynek, self.K, 0.0, "call", n_obs=12)
        mc_call = wycen_azjatycka_mc(
            self.rynek, self.K, 0.0, "call", n_obs=12, n_sciezek=10000, seed=42
        )
        self.assertGreaterEqual(mc_call.cena + 3.0 * mc_call.se_antytetyczne, geom_call)

    def test_asian_put_below_geometric_upper_bound(self) -> None:
        geom_put = wycen_azjatycka_geom_dyskretna(self.rynek, self.K, 0.0, "put", n_obs=12)
        mc_put = wycen_azjatycka_mc(
            self.rynek, self.K, 0.0, "put", n_obs=12, n_sciezek=10000, seed=42
        )
        self.assertLessEqual(mc_put.cena - 3.0 * mc_put.se_antytetyczne, geom_put)


if __name__ == "__main__":
    unittest.main()
