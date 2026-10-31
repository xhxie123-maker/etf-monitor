import unittest
import numpy as np
import pandas as pd
import sys, types
sys.modules.setdefault("akshare", types.SimpleNamespace())
sys.path.insert(0, ".")
import backtest as bt


def _synthetic(n=400, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    # 构造带趋势的序列：前段上行、中段震荡、后段下行，便于出现换仓
    trend = np.concatenate([np.linspace(1.0, 1.8, n // 3),
                            np.linspace(1.8, 1.7, n // 3),
                            np.linspace(1.7, 1.2, n - 2 * (n // 3))])
    close = trend * (1 + rng.normal(0, 0.008, n))
    return pd.DataFrame({"date": idx, "close": np.maximum(close, 0.1)})


class BacktestTests(unittest.TestCase):
    def test_load_prices_normalizes(self):
        df = pd.DataFrame({"date": ["2024-01-02", "2024-01-01", "2024-01-02"],
                           "close": [2, 1, 2.1]})
        # 写临时 CSV 验证读取规范化
        import tempfile, os
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            df.to_csv(f.name, index=False)
            path = f.name
        try:
            g = bt.load_prices(path)
            self.assertEqual(len(g), 2)
            self.assertEqual(g.iloc[-1]["close"], 2.1)
        finally:
            os.unlink(path)

    def test_signals_next_day(self):
        df = _synthetic(n=300)
        out = bt.compute_signals(df)
        # 信号是条件的前一日：当日无法利用当日收盘后的信息
        cond = (out["ma20"] > out["ma60"]) & (out["ma60"] > out["ma120"])
        for i in range(130, len(out)):
            self.assertEqual(out["signal"].iloc[i], int(cond.shift(1).fillna(0).iloc[i]))

    def test_run_backtest_outputs_metrics(self):
        df = _synthetic(n=500)
        r = bt.run_backtest(df, fee=0.001, slippage=0.001)
        self.assertTrue(r["ok"])
        for k in ("strategy_ann", "buyhold_ann", "strategy_mdd", "buyhold_mdd",
                  "sharpe", "exposure_pct", "trades", "turnover_per_year"):
            self.assertIn(k, r)
        self.assertIsInstance(r["strategy_mdd"], float)
        self.assertLessEqual(r["strategy_mdd"], 0)
        self.assertGreaterEqual(r["exposure_pct"], 0)
        self.assertLessEqual(r["exposure_pct"], 100)

    def test_run_backtest_too_short(self):
        df = _synthetic(n=100)
        r = bt.run_backtest(df)
        self.assertFalse(r["ok"])


if __name__ == "__main__":
    unittest.main()
