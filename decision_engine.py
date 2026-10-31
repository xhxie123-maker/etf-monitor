# -*- coding: utf-8 -*-
"""
V3 交易决策引擎（etf-monitor-v3）
=================================
将判断拆为 趋势 / 估值 / 风险 三个维度，输出综合评分(0-100)、
数据完整度、建议档位、主要依据、分批建仓纪律与 ATR 参考价位。

设计原则（V3 核心）：
1. 长期趋势向下或风险过高时，不再因为 RSI 超卖 / PE 低估就直接给出强烈买入建议。
2. 估值只使用"积累的中证官方 PE 观测"：至少 200 条有效记录才计算历史分位并参与决策；
   样本不足时该维度明确标注"数据不足"，不参与综合评分，宁可暂不判断也不硬算买卖结论。
3. 本模块为纯函数（仅依赖 pandas/numpy），便于单元测试与复用。

所有输出均为量化参考，不构成投资建议。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# 估值维度参与决策所需的最少官方 PE 有效记录数
PE_MIN_OFFICIAL = 200
# 官方 PE 分位计算窗口（交易日）：与主信号窗口一致
PE_WINDOWS = {"近1年": 250, "近3年": 750, "近5年": 1250}
PE_MAIN_WINDOW = "近3年"

# 估值分位阈值（%）
PE_LOW = 30.0
PE_HIGH = 70.0

# 风险阈值
RISK_DD120_HIGH = -25.0     # 距120日高点回撤 ≤ -25% → 高风险
RISK_DD120_MID = -15.0      # ≤ -15% → 中风险
RISK_VOL_HIGH = 35.0        # 20日年化波动率 ≥ 35% → 高风险
RISK_VOL_MID = 25.0         # ≥ 25% → 中风险
RISK_ADX_STRONG = 40.0      # ADX ≥ 40 且趋势下降 → 高风险


def trend_state(ma20: float | None, ma60: float | None, ma120: float | None,
                slope_ma20: float | None = None) -> dict:
    """
    趋势维度：由 MA20/MA60/MA120 结构与 MA20 斜率判断。
    返回 {label, score, desc}
    label: 上升 / 下降 / 震荡 / 数据不足
    score: 上升=1.0, 下降=-1.0, 震荡=0.0, 数据不足=None（不参与）
    """
    if ma20 is None or ma60 is None or ma120 is None or any(
            (x is not None and (isinstance(x, float) and math.isnan(x))) for x in (ma20, ma60, ma120)):
        # 数据不足（任一关键均线缺失）
        if (ma20 is None or (isinstance(ma20, float) and math.isnan(ma20))) or \
           (ma60 is None or (isinstance(ma60, float) and math.isnan(ma60))) or \
           (ma120 is None or (isinstance(ma120, float) and math.isnan(ma120))):
            return {"label": "数据不足", "score": None, "desc": "均线数据不足，趋势暂不判断"}
    # 显式结构判断（忽略 NaN 的参与）
    valid = [x for x in (ma20, ma60, ma120) if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(valid) < 2:
        return {"label": "数据不足", "score": None, "desc": "均线数据不足，趋势暂不判断"}
    ma20, ma60, ma120 = (float(x) if x is not None and not (isinstance(x, float) and math.isnan(x)) else None
                         for x in (ma20, ma60, ma120))
    score = 0.0
    desc_parts = []
    if ma20 is not None and ma60 is not None:
        if ma20 > ma60:
            score += 0.5
            desc_parts.append("MA20>MA60")
        else:
            score -= 0.5
            desc_parts.append("MA20<MA60")
    if ma60 is not None and ma120 is not None:
        if ma60 > ma120:
            score += 0.5
            desc_parts.append("MA60>MA120")
        else:
            score -= 0.5
            desc_parts.append("MA60<MA120")
    # MA20 斜率辅助（5 日变化）
    if slope_ma20 is not None and not (isinstance(slope_ma20, float) and math.isnan(slope_ma20)):
        if slope_ma20 > 0:
            score += 0.2 * min(abs(float(slope_ma20)) / 0.01, 1.0)
            desc_parts.append("MA20斜率向上")
        elif slope_ma20 < 0:
            score -= 0.2 * min(abs(float(slope_ma20)) / 0.01, 1.0)
            desc_parts.append("MA20斜率向下")
    score = max(-1.0, min(1.0, score))
    if score >= 0.6:
        label = "上升"
    elif score <= -0.6:
        label = "下降"
    else:
        label = "震荡"
    return {"label": label, "score": score, "desc": "、".join(desc_parts) if desc_parts else label}


def valuation_state(pe_pct: float | None, official_count: int) -> dict:
    """
    估值维度：仅官方 PE 积累 ≥ PE_MIN_OFFICIAL 才参与决策。
    返回 {label, score, desc, available}
    label: 低估 / 合理 / 高估 / 数据不足
    score: 低估=1.0, 高估=-1.0, 合理=0.0, 数据不足=None
    """
    if official_count is None or official_count < PE_MIN_OFFICIAL or pe_pct is None or \
       (isinstance(pe_pct, float) and math.isnan(pe_pct)):
        return {"label": "数据不足", "score": None, "desc": f"官方PE积累{0 if official_count is None else int(official_count)}条(<{PE_MIN_OFFICIAL})，估值不参与决策", "available": False}
    if pe_pct < PE_LOW:
        return {"label": "低估", "score": 1.0, "desc": f"官方PE分位{pe_pct:.0f}% 低估", "available": True}
    if pe_pct > PE_HIGH:
        return {"label": "高估", "score": -1.0, "desc": f"官方PE分位{pe_pct:.0f}% 高估", "available": True}
    return {"label": "合理", "score": 0.0, "desc": f"官方PE分位{pe_pct:.0f}% 中性", "available": True}


def _wilder_smooth(values: pd.Series, period: int) -> pd.Series:
    """Wilder 平滑：从首个有效观测开始，前 period 期均值初始化，之后逐点递推 (prev*(p-1)+cur)/p。
    输入允许含 NaN 前缀（如 DX 序列），自动跳过无效区间。"""
    out = pd.Series(np.nan, index=values.index, dtype=float)
    vals = values.to_numpy(dtype=float)
    n = len(vals)
    valid_idx = np.where(~np.isnan(vals))[0]
    if len(valid_idx) < period + 1:
        return out
    # 初始窗口：从第 period 个有效值位置开始，用其之前最近的 period 个有效值求均值
    for start in range(period, len(valid_idx)):
        anchor = valid_idx[start]
        prev = valid_idx[:start]
        if len(prev) >= period:
            out.iloc[anchor] = vals[prev[-period:]].mean()
            break
    # 递推：对每个有效观测逐点平滑（初始锚点之后；缺失值保持 NaN 不参与）
    for i in range(1, n):
        if np.isnan(vals[i]):
            continue
        prev_val = out.iloc[i - 1]
        if np.isnan(prev_val):
            seg = vals[valid_idx[valid_idx <= i]][-period:]
            if len(seg) >= period:
                out.iloc[i] = seg.mean()
        else:
            out.iloc[i] = (prev_val * (period - 1) + vals[i]) / period
    return out


def risk_metrics(closes: pd.Series, highs: pd.Series | None = None,
                 lows: pd.Series | None = None) -> dict:
    """
    风险维度原始指标（末值 + 序列内部使用）：
      atr(14)、adx(14)、vol20_ann（20日年化波动率%）、dd60/dd120（距60/120日高点回撤%）
    返回含最新值的 dict；数据不足的字段为 None。
    """
    closes = closes.dropna().astype(float)
    if len(closes) < 30:
        return {"atr": None, "adx": None, "vol20_ann": None, "dd60": None, "dd120": None,
                "atr_pct": None}
    highs = highs.dropna().astype(float) if highs is not None else closes
    lows = lows.dropna().astype(float) if lows is not None else closes
    # 对齐索引
    idx = closes.index
    h = highs.reindex(idx).fillna(closes)
    l = lows.reindex(idx).fillna(closes)
    prev_close = closes.shift(1)
    tr = pd.concat([h - l, (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    atr = _wilder_smooth(tr, 14)
    # ADX(14)：+DM/-DM → Wilder 平滑 → DX → ADX
    up = h.diff()
    down = -l.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=idx)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=idx)
    atr_s = _wilder_smooth(tr, 14)
    plus_s = _wilder_smooth(plus_dm, 14)
    minus_s = _wilder_smooth(minus_dm, 14)
    di_plus = 100 * plus_s / atr_s.replace(0, np.nan)
    di_minus = 100 * minus_s / atr_s.replace(0, np.nan)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx = _wilder_smooth(dx, 14)
    ret = closes.pct_change()
    vol20 = ret.rolling(20).std() * math.sqrt(252) * 100
    dd60 = (closes / closes.rolling(60).max() - 1) * 100
    dd120 = (closes / closes.rolling(120).max() - 1) * 100

    def _last(s):
        v = s.dropna()
        return float(v.iloc[-1]) if len(v) else None

    atr_v = _last(atr)
    return {
        "atr": atr_v,
        "atr_pct": round(atr_v / float(closes.iloc[-1]) * 100, 2) if atr_v else None,
        "adx": _last(adx),
        "vol20_ann": _last(vol20),
        "dd60": _last(dd60),
        "dd120": _last(dd120),
    }


def risk_state(r: dict, trend: dict | None = None) -> dict:
    """
    风险等级：低/中/高。
    score: 低=1.0, 中=0.5, 高=0.0；数据不足=None。
    """
    if r is None:
        return {"label": "数据不足", "score": None, "desc": "风险数据不足", "available": False}
    dd120 = r.get("dd120")
    vol = r.get("vol20_ann")
    adx = r.get("adx")
    t_label = (trend or {}).get("label")
    parts = []
    if dd120 is not None and dd120 <= RISK_DD120_HIGH:
        parts.append(f"距120日高点回撤{dd120:.0f}%")
    if vol is not None and vol >= RISK_VOL_HIGH:
        parts.append(f"年化波动{vol:.0f}%")
    if adx is not None and adx >= RISK_ADX_STRONG and t_label == "下降":
        parts.append(f"ADX{adx:.0f}趋势下行")
    if parts:
        return {"label": "高风险", "score": 0.0, "desc": "；".join(parts), "available": True}
    mid = []
    if dd120 is not None and dd120 <= RISK_DD120_MID:
        mid.append(f"距120日高点回撤{dd120:.0f}%")
    if vol is not None and vol >= RISK_VOL_MID:
        mid.append(f"年化波动{vol:.0f}%")
    if mid:
        return {"label": "中风险", "score": 0.5, "desc": "；".join(mid), "available": True}
    return {"label": "低风险", "score": 1.0, "desc": "ATR/波动率/回撤均处温和区间", "available": True}


def data_quality(trend: dict, val: dict, risk: dict, official_count: int) -> int:
    """数据完整度 0-100：估值按官方积累条数计，趋势/风险按是否可用计"""
    q = []
    q.append(100 if official_count is not None and official_count >= PE_MIN_OFFICIAL else
             (int(official_count / PE_MIN_OFFICIAL * 100) if official_count and official_count > 0 else 0))
    q.append(100 if (trend.get("score") is not None) else 0)
    q.append(100 if (risk.get("score") is not None) else 0)
    return int(round(sum(q) / len(q)))


def decide(trend: dict, val: dict, risk: dict, official_count: int) -> dict:
    """
    综合决策：输出 综合评分(0-100)、数据完整度、建议档位、主要依据。
    估值缺失时权重在趋势/风险间重分配；趋势下降或风险过高优先压制买入档位。
    """
    t = trend.get("score")
    v = val.get("score")
    r = risk.get("score")
    t_label = trend.get("label")
    r_label = risk.get("label")
    if t is None and r is None:
        return {"score": None, "quality": data_quality(trend, val, risk, official_count),
                "suggestion": "暂不判断（数据不足）", "sug_color": "#94a3b8",
                "reasons": ["趋势/风险数据均不足，暂不生成交易建议"],
                "trend": t_label, "risk": r_label, "val": val.get("label")}
    # 权重：估值可用时 0.3/0.4/0.3；缺失时趋势/风险对半
    if v is not None:
        w_t, w_v, w_r = 0.30, 0.40, 0.30
        base = 100 * (w_t * (t if t is not None else 0.5) + w_v * v + w_r * (r if r is not None else 0.5))
    else:
        w_t, w_v, w_r = 0.5, 0.0, 0.5
        base = 100 * (w_t * (t if t is not None else 0.5) + w_r * (r if r is not None else 0.5))
    score = int(round(base))
    reasons = []
    reasons.append(trend.get("desc") or f"趋势{t_label}")
    if v is not None:
        reasons.append(val.get("desc"))
    else:
        reasons.append(val.get("desc"))  # 数据不足说明
    reasons.append(risk.get("desc") or f"风险{r_label}")

    # 档位（趋势/风险优先压制买入）
    if r_label == "高风险" or t_label == "下降":
        if r_label == "高风险":
            suggestion, sug_color = "暂停新增仓位，优先控制风险", "#3b82f6"
        else:
            suggestion, sug_color = "观望或按既定计划小额定投", "#94a3b8"
    elif score >= 70 and t_label in ("上升", "震荡") and val.get("label") != "高估":
        suggestion, sug_color = "允许分批建仓/定投加档", "#ef4444"
    elif score >= 55:
        suggestion, sug_color = "持有为主，等待回踩企稳", "#f97316"
    elif score >= 40:
        suggestion, sug_color = "观望或按既定计划小额定投", "#eab308"
    else:
        suggestion, sug_color = "暂停新增仓位，优先控制风险", "#3b82f6"
    return {"score": score, "quality": data_quality(trend, val, risk, official_count),
            "suggestion": suggestion, "sug_color": sug_color,
            "reasons": reasons, "trend": t_label, "risk": r_label, "val": val.get("label")}


def batch_plan(trend: dict, risk: dict, val: dict) -> dict:
    """
    分批建仓纪律（参考执行计划）：
      第一笔 40%：趋势条件成立（当前）；
      第二笔 30%：回踩 MA20 附近并重新站稳；
      第三笔 30%：突破近 20 日高点并有成交确认。
    趋势下降或风险过高 → 暂停新增主动仓位；估值高估 → 暂缓新增。
    """
    t_label = trend.get("label")
    r_label = risk.get("label")
    v_label = val.get("label")
    if t_label == "下降" or r_label == "高风险":
        return {"status": "暂停", "color": "#3b82f6",
                "text": "趋势下降或风险过高：暂停新增主动仓位，不机械补仓"}
    if v_label == "高估":
        return {"status": "暂缓", "color": "#eab308",
                "text": "估值高估：暂缓新增，等待估值回落或趋势确认"}
    steps = [
        {"phase": "第一笔", "ratio": "40%", "trigger": "趋势条件仍然成立（当前）"},
        {"phase": "第二笔", "ratio": "30%", "trigger": "回踩 MA20 附近并重新站稳"},
        {"phase": "第三笔", "ratio": "30%", "trigger": "突破近 20 日高点并有成交确认"},
    ]
    return {"status": "允许", "color": "#ef4444", "text": "允许分批建仓（参考执行计划）", "steps": steps}


def atr_ref(close: float | None, atr: float | None) -> dict:
    """ATR 波动率参考价位（非止损指令、非保证成交价）"""
    if close is None or atr is None or atr <= 0:
        return {"text": "ATR 数据不足，暂无参考价位"}
    return {
        "atr": round(atr, 4),
        "ref_low": round(close - atr, 4),
        "ref_high": round(close + atr, 4),
        "watch_low": round(close - 1.5 * atr, 4),
        "watch_high": round(close + 1.5 * atr, 4),
        "text": (f"ATR={atr:.4f}（{atr / close * 100:.1f}%），参考波动区间 {close - atr:.4f}~{close + atr:.4f}；"
                 f"跌破 {close - 1.5 * atr:.4f} 关注企稳，突破 {close + 1.5 * atr:.4f} 关注趋势确认"),
    }


def official_pe_windows(pe_hist: pd.DataFrame | None) -> dict:
    """
    基于积累库官方 PE 序列计算三窗口分位（决策引擎估值维度使用）。
    pe_hist: 该指数官方 PE 历史（含 date, pe），升序。
    返回 {窗口名: {pct,n,start,end}} 或 None；样本不足的窗口为 None；总记录 < PE_MIN_OFFICIAL 返回 None。
    """
    if pe_hist is None or pe_hist.empty:
        return None
    s = pe_hist.dropna(subset=["pe"]).sort_values("date")
    if len(s) < PE_MIN_OFFICIAL:
        return None
    s = s.set_index(pd.to_datetime(s["date"]))["pe"].astype(float)
    out = {}
    for label, n in PE_WINDOWS.items():
        w = s.tail(n)
        if len(w) < max(int(n * 0.80), PE_MIN_OFFICIAL):
            out[label] = None
            continue
        reference = w.iloc[:-1] if len(w) > 1 else w
        pct = float((reference < float(w.iloc[-1])).sum() / len(reference) * 100)
        out[label] = {"pct": round(pct, 1), "n": int(len(w)),
                      "start": str(w.index[0].date()), "end": str(w.index[-1].date())}
    return out
