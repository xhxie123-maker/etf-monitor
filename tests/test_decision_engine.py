import unittest
import numpy as np
import pandas as pd
import sys, types
sys.modules.setdefault("akshare", types.SimpleNamespace())
sys.path.insert(0, ".")
import decision_engine as de


class TrendStateTests(unittest.TestCase):
    def test_uptrend(self):
        t = de.trend_state(1.0, 0.9, 0.8, slope_ma20=0.01)
        self.assertEqual(t["label"], "上升")
        self.assertGreaterEqual(t["score"], 0.6)

    def test_downtrend(self):
        t = de.trend_state(0.8, 0.9, 1.0, slope_ma20=-0.01)
        self.assertEqual(t["label"], "下降")
        self.assertLessEqual(t["score"], -0.6)

    def test_missing_ma(self):
        t = de.trend_state(None, 0.9, 1.0)
        self.assertEqual(t["label"], "数据不足")
        self.assertIsNone(t["score"])

    def test_nan_ma(self):
        t = de.trend_state(float("nan"), 0.9, 1.0)
        self.assertEqual(t["label"], "数据不足")


class ValuationStateTests(unittest.TestCase):
    def test_below_min_samples_not_available(self):
        v = de.valuation_state(25.0, 150)
        self.assertEqual(v["label"], "数据不足")
        self.assertFalse(v["available"])
        self.assertIsNone(v["score"])

    def test_cheap_with_enough_samples(self):
        v = de.valuation_state(15.0, 250)
        self.assertEqual(v["label"], "低估")
        self.assertEqual(v["score"], 1.0)

    def test_expensive(self):
        v = de.valuation_state(85.0, 250)
        self.assertEqual(v["label"], "高估")
        self.assertEqual(v["score"], -1.0)


class RiskTests(unittest.TestCase):
    def _frame(self, n=250):
        idx = pd.date_range("2023-01-01", periods=n, freq="B")
        close = pd.Series(np.linspace(1.0, 1.5, n), index=idx)
        high = close * 1.01
        low = close * 0.99
        return close, high, low

    def test_risk_metrics_fields(self):
        c, h, l = self._frame()
        r = de.risk_metrics(c, h, l)
        for k in ("atr", "adx", "vol20_ann", "dd60", "dd120"):
            self.assertIsNotNone(r[k])
        self.assertGreater(r["atr"], 0)

    def test_risk_state_low_when_calm(self):
        r = {"dd120": -5.0, "vol20_ann": 12.0, "adx": 18.0}
        s = de.risk_state(r)
        self.assertEqual(s["label"], "低风险")
        self.assertEqual(s["score"], 1.0)

    def test_risk_state_high_on_deep_drawdown(self):
        r = {"dd120": -30.0, "vol20_ann": 15.0, "adx": 20.0}
        s = de.risk_state(r)
        self.assertEqual(s["label"], "高风险")
        self.assertEqual(s["score"], 0.0)

    def test_risk_state_high_vol(self):
        r = {"dd120": -8.0, "vol20_ann": 40.0, "adx": 20.0}
        s = de.risk_state(r)
        self.assertEqual(s["label"], "高风险")


class DecideTests(unittest.TestCase):
    def test_downtrend_high_risk_no_buy(self):
        t = de.trend_state(0.8, 0.9, 1.0, slope_ma20=-0.01)
        v = de.valuation_state(10.0, 300)   # 估值很便宜
        r = {"label": "高风险", "score": 0.0, "desc": "回撤深", "available": True}
        d = de.decide(t, v, r, 300)
        # 关键：趋势下降/风险高时即使估值低估也不给"分批建仓"
        self.assertNotEqual(d["suggestion"], "允许分批建仓/定投加档")
        self.assertIn("暂停新增", d["suggestion"])
        self.assertIsNotNone(d["score"])

    def test_uptrend_cheap_lowrisk_allows_buy(self):
        t = de.trend_state(1.0, 0.9, 0.8, slope_ma20=0.01)
        v = de.valuation_state(15.0, 300)
        r = {"label": "低风险", "score": 1.0, "desc": "温和", "available": True}
        d = de.decide(t, v, r, 300)
        self.assertEqual(d["suggestion"], "允许分批建仓/定投加档")
        self.assertGreaterEqual(d["score"], 70)

    def test_valuation_missing_reweights(self):
        t = de.trend_state(1.0, 0.9, 0.8, slope_ma20=0.01)
        v = de.valuation_state(None, 100)   # 官方积累不足，估值不参与
        r = {"label": "低风险", "score": 1.0, "desc": "温和", "available": True}
        d = de.decide(t, v, r, 100)
        self.assertEqual(d["val"], "数据不足")
        self.assertIsNotNone(d["score"])   # 仍可用趋势×风险给出判断
        self.assertLess(d["quality"], 100)

    def test_all_missing_no_decision(self):
        t = de.trend_state(None, None, None)
        v = de.valuation_state(None, 0)
        r = {"label": "数据不足", "score": None, "desc": "无", "available": False}
        d = de.decide(t, v, r, 0)
        self.assertIsNone(d["score"])
        self.assertEqual(d["suggestion"], "暂不判断（数据不足）")


class BatchPlanTests(unittest.TestCase):
    def test_allow_when_ok(self):
        t = {"label": "上升", "score": 1.0}
        r = {"label": "低风险", "score": 1.0}
        v = {"label": "合理", "score": 0.0}
        p = de.batch_plan(t, r, v)
        self.assertEqual(p["status"], "允许")
        self.assertEqual(len(p["steps"]), 3)
        self.assertEqual(p["steps"][0]["ratio"], "40%")
        self.assertEqual(p["steps"][1]["ratio"], "30%")
        self.assertEqual(p["steps"][2]["ratio"], "30%")

    def test_pause_on_downtrend(self):
        t = {"label": "下降", "score": -1.0}
        r = {"label": "低风险", "score": 1.0}
        v = {"label": "低估", "score": 1.0}
        p = de.batch_plan(t, r, v)
        self.assertEqual(p["status"], "暂停")

    def test_pause_on_high_risk(self):
        t = {"label": "上升", "score": 1.0}
        r = {"label": "高风险", "score": 0.0}
        v = {"label": "低估", "score": 1.0}
        p = de.batch_plan(t, r, v)
        self.assertEqual(p["status"], "暂停")


class OfficialPeWindowsTests(unittest.TestCase):
    def test_insufficient_history_returns_none(self):
        h = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=100),
                          "pe": np.linspace(8, 12, 100)})
        self.assertIsNone(de.official_pe_windows(h))

    def test_enough_history_computes_windows(self):
        n = 300
        h = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=n, freq="B"),
                          "pe": np.linspace(8, 12, n)})
        w = de.official_pe_windows(h)
        self.assertIsNotNone(w)
        self.assertIsNotNone(w["近1年"])
        self.assertIsNone(w["近3年"])   # 300 条 < 750*0.8=600，近3年窗口不足
        self.assertGreater(w["近1年"]["n"], 0)


class AtrRefTests(unittest.TestCase):
    def test_atr_ref(self):
        a = de.atr_ref(10.0, 0.5)
        self.assertAlmostEqual(a["ref_low"], 9.5)
        self.assertAlmostEqual(a["ref_high"], 10.5)
        self.assertAlmostEqual(a["watch_low"], 9.25)
        self.assertIn("参考波动区间", a["text"])

    def test_atr_ref_missing(self):
        a = de.atr_ref(None, None)
        self.assertIn("暂无", a["text"])


if __name__ == "__main__":
    unittest.main()
