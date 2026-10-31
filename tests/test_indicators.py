import unittest
import numpy as np
import pandas as pd
import sys, types
sys.modules.setdefault("akshare", types.SimpleNamespace())
import etf_monitor as em

class IndicatorTests(unittest.TestCase):
    def test_normalize_daily_sorts_deduplicates(self):
        raw = pd.DataFrame({"日期":["2024-01-02","2024-01-01","2024-01-02"],
                            "收盘":[2,1,2.1]})
        got = em.normalize_daily(raw, "test")
        self.assertEqual(len(got), 2)
        self.assertEqual(got.iloc[-1]["close"], 2.1)

    def test_rsi_bounds_and_warmup(self):
        prices = pd.DataFrame({"close": np.arange(1, 80, dtype=float)})
        got = em.calc_indicators(prices)
        self.assertTrue(pd.isna(got.loc[13, "rsi"]))
        self.assertAlmostEqual(got.loc[14, "rsi"], 100.0)

    def test_pe_window_requires_coverage(self):
        s = pd.Series(np.linspace(10, 20, 300),
                      index=pd.date_range("2024-01-01", periods=300))
        got = em.calc_pe_windows(s)
        self.assertIsNotNone(got["近1年"])
        self.assertIsNone(got["近3年"])
        self.assertIsNone(got["近5年"])

if __name__ == "__main__":
    unittest.main()
