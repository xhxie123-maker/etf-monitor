# -*- coding: utf-8 -*-
"""
ETF 智能监测工具
================
监测标的：515180 红利ETF / 516310 银行ETF / 515220 煤炭ETF
核心指标：价格趋势(MA20/MA60/BOLL/RSI)、指数估值分位、创业板红利比
输出     ：单窗口可视化看板 output/etf_dashboard.html（浏览器打开即看全部）
用法     ：python etf_monitor.py [--open]
          --open 生成后自动打开浏览器看板
数据源   ：东方财富(ETF/指数行情)、中证指数官网(官方估值)、本地积累库(估值分位历史)
说明     ：本工具仅为数据监测与量化参考，不构成投资建议。
"""
import os
import sys
import json
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import akshare as ak

# ============================================================
# 0. 配置区
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "output"
LIBS_DIR = BASE_DIR / "libs"
OUT_DIR.mkdir(exist_ok=True)

DASHBOARD_PATH = OUT_DIR / "etf_dashboard.html"
HISTORY_PATH = OUT_DIR / "valuation_history.csv"
LOG_PATH = OUT_DIR / "run_log.txt"

# 监测标的：名称 -> 代码（ETF）
ETF_POOL = {
    "红利ETF": {"code": "515180", "index_code": "000922", "index_name": "中证红利",
               "color": "#e6b31e", "role": "高股息压舱石", "base_w": 40, "max_w": 60, "min_w": 20},
    "银行ETF": {"code": "516310", "index_code": "399986", "index_name": "中证银行",
               "color": "#3b82f6", "role": "低估值稳息品种", "base_w": 30, "max_w": 40, "min_w": 15},
    "煤炭ETF": {"code": "515220", "index_code": "000820", "index_name": "中证煤炭",
               "color": "#f97316", "role": "强周期高波动品种", "base_w": 30, "max_w": 40, "min_w": 15},
}

# 创业板红利比 = 创业板指(399006) / 中证红利(000922)
GEM_INDEX = "399006"
DIV_INDEX = "000922"

# 历史回看年数（价格类指标分位/均线用）
HISTORY_YEARS = 3
# 指数估值来源代码（中证指数官网）
CSINDEX_MAP = {"000922": "000922", "399986": "399986", "000820": "000820"}

# 信号阈值
TH = {
    "pe_low": 30, "pe_high": 70,          # PE分位 低估/高估
    "rsi_low": 35, "rsi_high": 70,        # RSI 超卖/超买
    "dev": 5.0,                           # 偏离20日线阈值%
    "ratio_low": 0.35, "ratio_high": 0.60 # 创业板红利比 阈值
}

REQUEST_TIMEOUT = 20

# ============================================================
# 1. 工具函数
# ============================================================
def log(msg: str):
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def safe_call(name, fn, *args, **kwargs):
    """带超时与异常捕获的统一调用，失败返回 None；attempts 控制重试次数"""
    attempts = int(kwargs.pop("attempts", 1))
    last = None
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last = e
            log(f"[数据源] {name} 第{i + 1}/{attempts}次获取失败: {type(e).__name__}: {str(e)[:120]}")
            if i < attempts - 1:
                time.sleep(2)
    return None


# ============================================================
# 2. 数据获取（东财优先，新浪/腾讯兜底——海外服务器(如GitHub Actions)访问东财会被拒绝）
# ============================================================
def get_etf_daily_em(code: str):
    """ETF 前复权日线（东方财富）"""
    end = dt.date.today().strftime("%Y%m%d")
    start = (dt.date.today() - dt.timedelta(days=int(HISTORY_YEARS * 365))).strftime("%Y%m%d")
    df = ak.fund_etf_hist_em(symbol=code, period="daily",
                             start_date=start, end_date=end, adjust="qfq")
    df = df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close",
                            "最高": "high", "最低": "low", "涨跌幅": "pct_chg"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def get_etf_daily_sina(code: str):
    """ETF 日线（新浪，海外可访问；新浪接口为不复权数据，仅作东财失败时的兜底）"""
    prefix = "sh" if str(code).startswith("5") else "sz"
    df = ak.fund_etf_hist_sina(symbol=f"{prefix}{code}")
    df = df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close",
                            "最高": "high", "最低": "low"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["pct_chg"] = df["close"].pct_change() * 100
    return df


def get_etf_daily(code: str):
    """ETF 日线：东财优先（重试2次），失败自动切新浪"""
    df = safe_call("东财ETF行情", get_etf_daily_em, code, attempts=2)
    if df is None:
        log("东财ETF行情不可用，尝试新浪源")
        df = safe_call("新浪ETF行情", get_etf_daily_sina, code)
    return df


def get_index_daily_em(code: str):
    """指数日线（东方财富）"""
    end = dt.date.today().strftime("%Y%m%d")
    start = (dt.date.today() - dt.timedelta(days=int(HISTORY_YEARS * 365))).strftime("%Y%m%d")
    df = ak.index_zh_a_hist(symbol=code, period="daily",
                            start_date=start, end_date=end)
    df = df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close",
                            "最高": "high", "最低": "low", "涨跌幅": "pct_chg"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def get_index_daily_sina(code: str):
    """指数日线（新浪，海外可访问）"""
    prefix = "sh" if str(code).startswith("000") else "sz"
    df = ak.stock_zh_index_daily(symbol=f"{prefix}{code}")
    df = df.rename(columns={"date": "date", "open": "open", "close": "close",
                            "high": "high", "low": "low"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["pct_chg"] = df["close"].pct_change() * 100
    return df


def get_index_daily_tx(code: str):
    """指数日线（腾讯，海外可访问，数据新鲜）"""
    prefix = "sh" if str(code).startswith("000") else "sz"
    df = ak.stock_zh_index_daily_tx(symbol=f"{prefix}{code}")
    df = df.rename(columns={"date": "date", "open": "open", "close": "close",
                            "high": "high", "low": "low"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["pct_chg"] = df["close"].pct_change() * 100
    return df


def get_index_daily(code: str):
    """指数日线：东财优先，失败依次切新浪、腾讯；
    新浪源部分指数停更（如 000922 只到 2019），过期数据视为失败继续下探"""
    df = safe_call("东财指数行情", get_index_daily_em, code, attempts=2)
    if df is None:
        log("东财指数行情不可用，尝试新浪源")
        df = safe_call("新浪指数行情", get_index_daily_sina, code)
    if df is not None and df["date"].max() < pd.Timestamp(dt.date.today()) - pd.Timedelta(days=10):
        log(f"新浪指数行情数据过期(末日期 {df['date'].max().date()})，跳过")
        df = None
    if df is None:
        log("新浪指数行情不可用，尝试腾讯源")
        df = safe_call("腾讯指数行情", get_index_daily_tx, code)
    return df


def get_index_valuation_csindex(index_code: str):
    """中证指数官网估值（官方），返回 DataFrame: date, pe, pb, dy 已按日期升序"""
    df = ak.stock_zh_index_value_csindex(symbol=CSINDEX_MAP.get(index_code, index_code))
    df = df.rename(columns={"日期": "date", "市盈率1": "pe", "市净率": "pb",
                            "股息率1": "dy"})
    # 该文件为降序（新→旧），转升序；市净率/股息率字段可能存在或不存在
    if "pb" not in df.columns:
        df["pb"] = np.nan
    if "dy" not in df.columns:
        df["dy"] = np.nan
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "pe", "pb", "dy"]]


# ============================================================
# 3. 指标计算
# ============================================================
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """均线 / RSI / BOLL / MACD / 偏离度"""
    df = df.copy()
    for w in (5, 20, 60):
        df[f"ma{w}"] = df["close"].rolling(w).mean()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["dif"] = ema12 - ema26
    df["dea"] = df["dif"].ewm(span=9, adjust=False).mean()
    df["macd"] = 2 * (df["dif"] - df["dea"])
    mid = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    df["boll_mid"], df["boll_std"] = mid, std
    df["boll_up"] = mid + 2 * std
    df["boll_low"] = mid - 2 * std
    df["dev20"] = (df["close"] - mid) / mid * 100
    return df


def percentile_of(series: pd.Series, value: float) -> float:
    s = series.dropna()
    if len(s) == 0:
        return np.nan
    return float((s < value).sum() / len(s) * 100)


# ============================================================
# 4. 估值历史积累库
# ============================================================
def load_history() -> pd.DataFrame:
    if HISTORY_PATH.exists():
        try:
            df = pd.read_csv(HISTORY_PATH, parse_dates=["date"])
            if "index_code" in df.columns:
                # 统一为6位字符串代码，避免 "922" 与 "000922" 并存导致去重失效
                df["index_code"] = df["index_code"].astype(str).str.zfill(6)
            return df
        except Exception as e:
            log(f"读取估值历史失败: {e}")
    return pd.DataFrame(columns=["date", "index_code", "index_name", "pe", "pb", "dy", "source"])


def update_history(hist: pd.DataFrame, code: str, name: str, val_df: pd.DataFrame) -> pd.DataFrame:
    """把本次拉到的估值并入历史库（按 date+index_code 去重）"""
    if val_df is None or val_df.empty:
        return hist
    new = val_df.copy()
    new["index_code"] = str(code).zfill(6)
    new["index_name"] = name
    new["source"] = "csindex"
    merged = pd.concat([hist, new], ignore_index=True)
    merged["index_code"] = merged["index_code"].astype(str).str.zfill(6)
    merged = merged.drop_duplicates(subset=["date", "index_code"], keep="last")
    merged = merged.sort_values("date").reset_index(drop=True)
    try:
        merged.to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")
    except Exception as e:
        log(f"估值历史写盘失败: {e}")
    return merged


def get_valuation(code: str, name: str, hist: pd.DataFrame):
    """
    返回 (latest_dict, history_df, source_desc)
    latest_dict: {date, pe, pb, dy}
    history_df: 该指数 PE 历史(升序)
    """
    val = safe_call("中证指数官网估值", get_index_valuation_csindex, code)
    if val is not None and not val.empty:
        hist = update_history(hist, code, name, val)
        src = "中证指数官网(官方)"
    elif hist is not None and not hist[hist["index_code"] == code].empty:
        src = "本地积累库(上次数据)"
    else:
        return None, hist, "暂无估值数据"
    sub = hist[hist["index_code"] == code].sort_values("date")
    if sub.empty:
        return None, hist, src
    latest = sub.iloc[-1].to_dict()
    latest["date"] = str(latest["date"])[:10]
    return latest, hist, src


# ============================================================
# 5. 信号引擎
# ============================================================
def build_signal(name: str, cfg: dict, price: pd.DataFrame, val_latest: dict, pe_hist: pd.DataFrame):
    """返回该 ETF 的全部指标、信号、建议"""
    last = price.iloc[-1]
    pe = val_latest.get("pe") if val_latest else np.nan
    pb = val_latest.get("pb") if val_latest else np.nan
    dy = val_latest.get("dy") if val_latest else np.nan
    pe_pct = percentile_of(pe_hist["pe"], pe) if val_latest and not pe_hist.empty else np.nan

    score = 0.0
    signals = []

    # ---- 估值 ----
    if not np.isnan(pe_pct):
        if pe_pct < TH["pe_low"]:
            score += 2; signals.append(f"估值低估·PE分位{pe_pct:.0f}%")
        elif pe_pct > TH["pe_high"]:
            score -= 2; signals.append(f"估值高估·PE分位{pe_pct:.0f}%")
        else:
            signals.append(f"估值中性·PE分位{pe_pct:.0f}%")

    # ---- 技术面 ----
    rsi = float(last.get("rsi")) if not np.isnan(last.get("rsi")) else np.nan
    if not np.isnan(rsi):
        if rsi < TH["rsi_low"]:
            score += 1; signals.append(f"RSI超卖({rsi:.0f})")
        elif rsi > TH["rsi_high"]:
            score -= 1; signals.append(f"RSI超买({rsi:.0f})")
        else:
            signals.append(f"RSI中性({rsi:.0f})")

    close = float(last["close"])
    if not np.isnan(last.get("boll_low")) and close < last["boll_low"]:
        score += 1; signals.append("跌破布林下轨")
    elif not np.isnan(last.get("boll_up")) and close > last["boll_up"]:
        score -= 1; signals.append("突破布林上轨")

    if not np.isnan(last.get("ma20")) and not np.isnan(last.get("ma60")):
        if last["ma20"] > last["ma60"]:
            score += 0.5; signals.append("中期均线多头")
        else:
            score -= 0.5; signals.append("中期均线空头")

    dev20 = float(last.get("dev20")) if not np.isnan(last.get("dev20")) else 0.0
    if dev20 < -TH["dev"]:
        score += 0.5; signals.append(f"偏离20日线{dev20:.1f}%(超跌)")
    elif dev20 > TH["dev"]:
        score -= 0.5; signals.append(f"偏离20日线{dev20:.1f}%(超涨)")

    # ---- 结论 ----
    if score >= 2.5:
        conclusion, concl_color, concl_txt = "强烈分批买入", "#ef4444", "估值与技术双低区域，分批建仓/加仓，严控单笔仓位"
    elif score >= 0.5:
        conclusion, concl_color, concl_txt = "逢低分批买入", "#f97316", "偏低估区域，可分批买入或维持定投"
    elif score > -1.0:
        conclusion, concl_color, concl_txt = "观望持有", "#94a3b8", "估值与趋势均衡，持有观察，不追高不杀跌"
    else:
        conclusion, concl_color, concl_txt = "逢高减仓", "#3b82f6", "偏高位区域，分批止盈/减仓，落袋为安"

    # ---- 定投档位（按 PE 分位）----
    if np.isnan(pe_pct):
        dca = "数据积累中，先按正常定投"
        dca_color = "#94a3b8"
    elif pe_pct < 20:
        dca, dca_color = "双倍定投(2×)", "#ef4444"
    elif pe_pct < 40:
        dca, dca_color = "正常定投(1×)", "#f97316"
    elif pe_pct < 70:
        dca, dca_color = "减半定投(0.5×)", "#eab308"
    else:
        dca, dca_color = "暂停定投·仅持有", "#3b82f6"

    # ---- 做T参考（短周期支撑/压力）----
    support = []
    pressure = []
    if not np.isnan(last.get("ma20")):
        support.append(round(float(last["ma20"]), 4))
    if not np.isnan(last.get("boll_low")):
        support.append(round(float(last["boll_low"]), 4))
    if not np.isnan(last.get("boll_up")):
        pressure.append(round(float(last["boll_up"]), 4))
    recent_high = float(price["high"].tail(5).max())
    pressure.append(round(recent_high, 4))
    t_text = ""
    if not np.isnan(rsi):
        if rsi > 70:
            t_text = "高位慎追，做T以减仓为主"
        elif rsi < 30:
            t_text = "超跌区，做T可留意低吸反抽"
        else:
            t_text = "中性区，箱体内高抛低吸"

    # ---- 长期提示 ----
    long_txt = (f"{cfg['role']}。基准仓位{cfg['base_w']}%，"
                f"估值低估可加至{cfg['max_w']}%，高估减至{cfg['min_w']}%。"
                f"当前股息率 {dy:.2f}%" if not np.isnan(dy) else
                f"{cfg['role']}。基准仓位{cfg['base_w']}%，低估加至{cfg['max_w']}%，高估减至{cfg['min_w']}%。")

    return {
        "name": name, "code": cfg["code"], "color": cfg["color"],
        "close": close,
        "pct_chg": float(last.get("pct_chg", 0.0)) if not np.isnan(last.get("pct_chg", 0.0)) else 0.0,
        "ma20": float(last["ma20"]) if not np.isnan(last.get("ma20")) else None,
        "ma60": float(last["ma60"]) if not np.isnan(last.get("ma60")) else None,
        "rsi": None if np.isnan(rsi) else round(rsi, 1),
        "dev20": round(dev20, 2),
        "pe": None if np.isnan(pe) else round(float(pe), 2),
        "pb": None if np.isnan(pb) else round(float(pb), 2),
        "dy": None if np.isnan(dy) else round(float(dy), 2),
        "pe_pct": None if np.isnan(pe_pct) else round(pe_pct, 1),
        "pe_samples": int(len(pe_hist)) if not pe_hist.empty else 0,
        "signals": signals,
        "score": round(score, 1),
        "conclusion": conclusion, "concl_color": concl_color, "concl_txt": concl_txt,
        "dca": dca, "dca_color": dca_color,
        "support": support, "pressure": pressure, "t_text": t_text,
        "long_txt": long_txt,
        "latest_date": str(price["date"].iloc[-1].date()),
    }


def build_ratio():
    """创业板红利比 = 创业板指/中证红利，返回最新值+分位+建议+3年序列"""
    gem = safe_call("创业板指行情", get_index_daily, GEM_INDEX)
    div = safe_call("中证红利行情", get_index_daily, DIV_INDEX)
    if gem is None or div is None:
        return None
    m = pd.merge(gem[["date", "close"]], div[["date", "close"]],
                 on="date", suffixes=("_gem", "_div"))
    m["ratio"] = m["close_gem"] / m["close_div"]
    m = m.dropna()
    # 仅取近3年样本（与看板“近3年分位”口径一致；腾讯/新浪源返回全量历史）
    m = m.tail(int(HISTORY_YEARS * 250)).reset_index(drop=True)
    latest = float(m["ratio"].iloc[-1])
    pct = percentile_of(m["ratio"], latest)
    if latest < TH["ratio_low"]:
        sug, sug_color = "成长风格占优期：创业板相对低估，可增配成长、减配红利", "#f97316"
    elif latest > TH["ratio_high"]:
        sug, sug_color = "红利风格占优期：红利相对低估，可增配红利价值、减配成长", "#ef4444"
    else:
        sug, sug_color = "风格均衡期：维持现有配置比例，不追风格", "#94a3b8"
    return {
        "latest": latest, "pct": round(pct, 1), "sug": sug, "sug_color": sug_color,
        "dates": [str(d.date()) for d in m["date"]],
        "ratios": [round(x, 4) for x in m["ratio"]],
        "gem_close": [round(x, 2) for x in m["close_gem"]],
        "div_close": [round(x, 2) for x in m["close_div"]],
        "n": len(m),
    }


# ============================================================
# 6. 看板生成（单窗口 HTML）
# ============================================================
def render_dashboard(meta: dict, ratio: dict, etfs: list):
    echarts_src = "../libs/echarts.min.js"
    payload = {
        "meta": meta,
        "ratio": ratio,
        "etfs": etfs,
        "th": TH,
    }
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ETF 智能监测看板</title>
<script src="{echarts_src}"></script>
<style>
  :root {{ --bg:#0b1020; --card:#141a2e; --line:#232c46; --text:#e6ebf5; --sub:#8a93ab;
           --green:#22c55e; --red:#ef4444; }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text);
         font-family:"Microsoft YaHei","PingFang SC",system-ui,sans-serif; padding:18px; }}
  .wrap {{ max-width:1280px; margin:0 auto; }}
  h1 {{ font-size:22px; font-weight:700; }}
  .header {{ display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;
             gap:10px; margin-bottom:14px; }}
  .header .sub {{ color:var(--sub); font-size:13px; }}
  .badges {{ display:flex; gap:8px; flex-wrap:wrap; }}
  .badge {{ font-size:12px; padding:3px 10px; border-radius:12px; border:1px solid var(--line); color:var(--sub); }}
  .badge.ok {{ color:#4ade80; border-color:#166534; }}
  .badge.warn {{ color:#fbbf24; border-color:#854d0e; }}
  .grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; }}
  .card h3 {{ font-size:15px; margin-bottom:10px; display:flex; justify-content:space-between; align-items:center; }}
  .card h3 .code {{ color:var(--sub); font-weight:400; font-size:12px; }}
  .num-row {{ display:flex; align-items:baseline; gap:10px; margin-bottom:6px; }}
  .big {{ font-size:28px; font-weight:700; }}
  .up {{ color:var(--red); }} .down {{ color:var(--green); }}
  .kv {{ display:grid; grid-template-columns:1fr 1fr; gap:6px 14px; margin:10px 0; font-size:13px; }}
  .kv .k {{ color:var(--sub); }} .kv .v {{ font-weight:600; }}
  .chips {{ display:flex; flex-wrap:wrap; gap:6px; margin:8px 0; }}
  .chip {{ font-size:12px; padding:2px 8px; border-radius:10px; border:1px solid var(--line); color:var(--sub); }}
  .banner {{ border-radius:8px; padding:8px 10px; font-size:13px; font-weight:700; margin:6px 0; }}
  .mini {{ font-size:12px; color:var(--sub); line-height:1.7; }}
  .chart {{ width:100%; }}
  .ratio-card {{ grid-column:span 3; }}
  .ratio-main {{ display:flex; gap:20px; align-items:stretch; flex-wrap:wrap; }}
  .ratio-nums {{ min-width:260px; }}
  .ratio-chart {{ flex:1; min-width:420px; height:240px; }}
  .ratio-bar {{ height:8px; border-radius:4px; margin:8px 0;
                background:linear-gradient(90deg,#22c55e 0%,#22c55e 30%,#eab308 30%,#eab308 60%,#ef4444 60%,#ef4444 100%); }}
  .ratio-mark {{ display:flex; justify-content:space-between; font-size:11px; color:var(--sub); }}
  .etf-grid {{ grid-template-columns:repeat(3,1fr); }}
  .price-charts {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:14px; }}
  .tall {{ height:300px; }}
  .pe-charts {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:14px; }}
  .foot {{ margin-top:16px; color:var(--sub); font-size:12px; text-align:center; line-height:1.8; }}
  .rules {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:14px; }}
  .rules li {{ margin-left:16px; }}
  @media (max-width:900px) {{ .grid,.etf-grid,.price-charts,.pe-charts,.rules {{ grid-template-columns:1fr; }}
      .ratio-card {{ grid-column:span 1; }} }}
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <div>
      <h1>ETF 智能监测看板</h1>
      <div class="sub">红利 515180 · 银行 516310 · 煤炭 515220 ｜ 数据日期 {meta['data_date']} ｜ 生成 {meta['gen_time']}</div>
    </div>
    <div class="badges" id="badges"></div>
  </div>

  <!-- 创业板红利比 -->
  <div class="grid">
  <div class="card ratio-card">
    <h3>风格轮动 · 创业板红利比 <span class="code">创业板指 {GEM_INDEX} ÷ 中证红利 {DIV_INDEX}</span></h3>
    <div class="ratio-main">
      <div class="ratio-nums">
        <div class="num-row"><span class="big" id="ratio-val">--</span>
          <span style="font-size:13px;color:var(--sub)" id="ratio-pct">--</span></div>
        <div class="ratio-bar"></div>
        <div class="ratio-mark"><span>&lt;0.35 成长占优</span><span>均衡</span><span>&gt;0.60 红利占优</span></div>
        <div class="banner" id="ratio-sug" style="margin-top:10px">--</div>
        <div class="mini">比值=创业板指收盘÷中证红利收盘；分位为近{HISTORY_YEARS}年历史百分位。比值走高=红利相对便宜。</div>
      </div>
      <div class="ratio-chart" id="ratio-chart"></div>
    </div>
  </div>
  </div>

  <!-- 三只 ETF 卡片 -->
  <div class="grid etf-grid" id="etf-cards"></div>

  <!-- 价格走势 -->
  <div class="price-charts" id="price-charts"></div>

  <!-- 估值分位走势（本地积累） -->
  <div class="pe-charts" id="pe-charts"></div>

  <!-- 操作纪律 -->
  <div class="rules">
    <div class="card"><h3>定投纪律</h3><ul class="mini" style="list-style:none">
      <li>· PE分位&lt;20%：双倍定投</li>
      <li>· PE分位20%~40%：正常定投</li>
      <li>· PE分位40%~70%：减半定投</li>
      <li>· PE分位&gt;70%：暂停定投，仅持底仓</li>
      <li>· 单日大跌&gt;3%可额外加投一份</li>
    </ul></div>
    <div class="card"><h3>再平衡</h3><ul class="mini" style="list-style:none">
      <li>· 每半年（6月/12月）强制再平衡一次</li>
      <li>· 单只仓位偏离基准 ±10% 触发调整</li>
      <li>· 创业板红利比穿越 0.35/0.60 时调整风格配比</li>
    </ul></div>
    <div class="card"><h3>长期纪律</h3><ul class="mini" style="list-style:none">
      <li>· 煤炭为强周期品种，仓位上限 40%，切忌单吊</li>
      <li>· 银行低估值弹性小，作为组合压舱石</li>
      <li>· 红利在成长牛市中会跑输大盘，保持组合思维</li>
      <li>· 所有信号为量化参考，须结合宏观与自身资金状况</li>
    </ul></div>
  </div>

  <div class="foot">
    数据来源：东方财富（行情）· 中证指数官网（官方估值）· 本地积累库（估值分位）<br>
    本工具仅作数据监测与量化参考，不构成投资建议。市场有风险，投资需谨慎。<br>
    估值分位样本随每日运行持续积累，样本越多分位越可靠。
  </div>
</div>

<script>
const DATA = {payload_json};

/* 徽章：数据源状态 */
const badges = DATA.meta.badges.map(b => `<span class="badge ${{b.ok?'ok':'warn'}}">${{b.label}}</span>`).join('');
document.getElementById('badges').innerHTML = badges;

/* 创业板红利比 */
const R = DATA.ratio;
if (R) {{
  document.getElementById('ratio-val').textContent = R.latest.toFixed(4);
  document.getElementById('ratio-pct').textContent = '历史分位 ' + R.pct + '% (' + R.n + ' 个交易日)';
  const sug = document.getElementById('ratio-sug');
  sug.textContent = R.sug;
  sug.style.color = R.sug_color;
  const rc = echarts.init(document.getElementById('ratio-chart'));
  rc.setOption({{
    backgroundColor:'transparent',
    grid:{{left:50,right:16,top:20,bottom:30}},
    tooltip:{{trigger:'axis'}},
    xAxis:{{type:'category',data:R.dates,axisLabel:{{color:'#8a93ab',fontSize:11}}}},
    yAxis:{{type:'value',scale:true,axisLabel:{{color:'#8a93ab'}},splitLine:{{lineStyle:{{color:'#232c46'}}}}}},
    series:[{{
      type:'line',data:R.ratios,smooth:true,symbol:'none',
      lineStyle:{{color:'#a78bfa',width:2}},
      areaStyle:{{color:'rgba(167,139,250,0.12)'}},
      markLine:{{silent:true,symbol:'none',
        data:[{{yAxis:0.35,label:{{formatter:'0.35 成长占优',color:'#22c55e'}},lineStyle:{{color:'#22c55e',type:'dashed'}}}},
              {{yAxis:0.60,label:{{formatter:'0.60 红利占优',color:'#ef4444'}},lineStyle:{{color:'#ef4444',type:'dashed'}}}}]}}
    }}]
  }});
  window.addEventListener('resize', () => rc.resize());
}}

/* ETF 卡片 */
function fmtNum(v, d=2) {{ return v===null||v===undefined ? '--' : Number(v).toFixed(d); }}
const cardsEl = document.getElementById('etf-cards');
DATA.etfs.forEach(e => {{
  const up = e.pct_chg >= 0;
  const peInfo = e.pe_pct!==null
    ? `PE ${{fmtNum(e.pe)}} · 分位 <b>${{e.pe_pct}}%</b>（样本${{e.pe_samples}}期）`
    : `PE ${{fmtNum(e.pe)}} · 分位 积累中`;
  const supp = e.support.length ? e.support.map(x=>fmtNum(x,3)).join(' / ') : '--';
  const pres = e.pressure.length ? e.pressure.map(x=>fmtNum(x,3)).join(' / ') : '--';
  cardsEl.insertAdjacentHTML('beforeend', `
    <div class="card" style="border-top:3px solid ${{e.color}}">
      <h3>${{e.name}} <span class="code">${{e.code}}</span></h3>
      <div class="num-row"><span class="big">${{fmtNum(e.close,3)}}</span>
        <span class="${{up?'up':'down'}}">${{e.pct_chg>=0?'+':''}}${{e.pct_chg.toFixed(2)}}%</span>
        <span style="font-size:12px;color:var(--sub)">${{e.latest_date}}</span></div>
      <div class="kv">
        <span class="k">PE / 分位</span><span class="v">${{peInfo}}</span>
        <span class="k">股息率</span><span class="v">${{e.dy!==null?e.dy+'%':'--'}}</span>
        <span class="k">RSI(14)</span><span class="v">${{fmtNum(e.rsi,0)}}</span>
        <span class="k">MA20 / MA60</span><span class="v">${{fmtNum(e.ma20,3)}} / ${{fmtNum(e.ma60,3)}}</span>
        <span class="k">偏离20日线</span><span class="v">${{e.dev20>0?'+':''}}${{e.dev20.toFixed(1)}}%</span>
        <span class="k">做T支撑/压力</span><span class="v">${{supp}} / ${{pres}}</span>
      </div>
      <div class="chips">${{e.signals.map(s=>`<span class="chip">${{s}}</span>`).join('')}}</div>
      <div class="banner" style="background:${{e.concl_color}}22;color:${{e.concl_color}}">${{e.conclusion}}（评分 ${{e.score}}）—— ${{e.concl_txt}}</div>
      <div class="mini">定投：<b style="color:${{e.dca_color}}">${{e.dca}}</b> ｜ ${{e.t_text}}</div>
      <div class="mini" style="margin-top:6px">长期：${{e.long_txt}}</div>
    </div>`);
}});

/* 价格走势图 */
const pcEl = document.getElementById('price-charts');
DATA.etfs.forEach(e => {{
  const div = document.createElement('div');
  div.className = 'card';
  div.style.borderTop = '3px solid ' + e.color;
  div.innerHTML = `<h3>${{e.name}} 价格走势 <span class="code">近250交易日·收盘/MA20/MA60/布林带</span></h3>
    <div class="chart tall" id="pc-${{e.code}}"></div>`;
  pcEl.appendChild(div);
  const chart = echarts.init(document.getElementById('pc-' + e.code));
  const dates = e.dates, close = e.closes, ma20 = e.ma20s, ma60 = e.ma60s,
        bup = e.boll_ups, blow = e.boll_lows;
  chart.setOption({{
    backgroundColor:'transparent',
    grid:{{left:50,right:16,top:20,bottom:30}},
    tooltip:{{trigger:'axis'}},
    legend:{{data:['收盘','MA20','MA60','布林上','布林下'],textStyle:{{color:'#8a93ab',fontSize:10}},top:0}},
    xAxis:{{type:'category',data:dates,axisLabel:{{color:'#8a93ab',fontSize:10,hideOverlap:true}}}},
    yAxis:{{type:'value',scale:true,axisLabel:{{color:'#8a93ab',fontSize:10}},splitLine:{{lineStyle:{{color:'#232c46'}}}}}},
    series:[
      {{name:'布林上',type:'line',data:bup,symbol:'none',lineStyle:{{color:'#475569',width:1,type:'dashed'}},stack:'b'}},
      {{name:'布林下',type:'line',data:blow,symbol:'none',lineStyle:{{color:'#475569',width:1,type:'dashed'}},stack:'b'}},
      {{name:'MA20',type:'line',data:ma20,symbol:'none',lineStyle:{{color:'#f59e0b',width:1}}}},
      {{name:'MA60',type:'line',data:ma60,symbol:'none',lineStyle:{{color:'#10b981',width:1}}}},
      {{name:'收盘',type:'line',data:close,symbol:'none',lineStyle:{{color:e.color,width:2}},areaStyle:{{color:'rgba(255,255,255,0.03)'}}}}
    ]
  }});
  window.addEventListener('resize', () => chart.resize());
}});

/* 估值分位图 */
const peEl = document.getElementById('pe-charts');
DATA.etfs.forEach(e => {{
  const div = document.createElement('div');
  div.className = 'card';
  div.style.borderTop = '3px solid ' + e.color;
  div.innerHTML = `<h3>${{e.name}} 估值分位走势 <span class="code">PE历史百分位·本地积累</span></h3>
    <div class="chart" style="height:170px" id="pe-${{e.code}}"></div>`;
  peEl.appendChild(div);
  const chart = echarts.init(document.getElementById('pe-' + e.code));
  chart.setOption({{
    backgroundColor:'transparent',
    grid:{{left:40,right:14,top:16,bottom:24}},
    tooltip:{{trigger:'axis'}},
    xAxis:{{type:'category',data:e.pe_dates,axisLabel:{{show:false}}}},
    yAxis:{{type:'value',min:0,max:100,axisLabel:{{color:'#8a93ab',fontSize:10,formatter:'{{value}}%'}},
            splitLine:{{lineStyle:{{color:'#232c46'}}}}}},
    series:[{{type:'bar',data:e.pe_pcts,itemStyle:{{color:'rgba(148,163,184,0.55)',borderRadius:[2,2,0,0]}}}}]
  }});
  window.addEventListener('resize', () => chart.resize());
}});
</script>
</body>
</html>"""
    try:
        DASHBOARD_PATH.write_text(html, encoding="utf-8")
        return str(DASHBOARD_PATH)
    except Exception as e:
        log(f"看板写盘失败: {e}")
        return None


def render_fallback(meta: dict):
    """云端容错：全部数据源失败时生成占位看板（保留上一次成功快照不被覆盖）"""
    if DASHBOARD_PATH.exists():
        return str(DASHBOARD_PATH)  # 已有历史看板则保留原样
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ETF 智能监测看板（数据暂不可用）</title>
<style>
  body {{ background:#0b1020; color:#e6ebf5; font-family:"Microsoft YaHei",system-ui,sans-serif;
         display:flex; align-items:center; justify-content:center; min-height:100vh; margin:0; }}
  .box {{ text-align:center; max-width:520px; padding:24px; }}
  h1 {{ font-size:20px; margin-bottom:12px; }}
  p {{ color:#8a93ab; font-size:14px; line-height:1.8; }}
</style>
</head>
<body>
<div class="box">
  <h1>ETF 智能监测看板 · 数据暂不可用</h1>
  <p>本次运行（{meta['gen_time']}）所有行情数据源均获取失败。<br>
     请稍后在仓库 Actions 页手动「Run workflow」重试。<br>
     若此前已有看板，将保留上一次成功快照。</p>
</div>
</body>
</html>"""
    try:
        DASHBOARD_PATH.write_text(html, encoding="utf-8")
        return str(DASHBOARD_PATH)
    except Exception as e:
        log(f"占位看板写盘失败: {e}")
        return None


# ============================================================
# 7. 主流程
# ============================================================
def main():
    run_at = dt.datetime.now()
    cloud_mode = "--cloud" in sys.argv  # 云端(GitHub Actions)容错模式
    meta = {"data_date": "--", "gen_time": run_at.strftime("%Y-%m-%d %H:%M:%S"), "badges": []}
    badges = [{"label": "东方财富行情", "ok": True}, {"label": "中证指数官方估值", "ok": True},
              {"label": "估值分位·本地积累", "ok": True}]

    # ---- 风格轮动：创业板红利比 ----
    ratio = build_ratio()
    if ratio is None:
        badges[0]["ok"] = False
        log("创业板红利比计算失败（创业板指/中证红利行情缺失）")

    # ---- 估值历史库 ----
    hist = load_history()

    # ---- 三只 ETF ----
    etfs = []
    data_dates = set()
    for name, cfg in ETF_POOL.items():
        price_df = safe_call(f"{name}行情", get_etf_daily, cfg["code"])
        if price_df is None:
            badges[0]["ok"] = False
            log(f"{name} 行情缺失，跳过")
            continue
        price_df = calc_indicators(price_df)
        data_dates.add(str(price_df["date"].iloc[-1].date()))

        val_latest, pe_hist_df, v_src = get_valuation(cfg["index_code"], cfg["index_name"], hist)
        hist = pe_hist_df  # 保持全局历史
        if "本地积累" in v_src:
            badges[2]["ok"] = False
            badges[1]["ok"] = False
        if "暂无" in v_src:
            badges[1]["ok"] = False
            badges[2]["ok"] = False

        sig = build_signal(name, cfg, price_df, val_latest, hist[hist["index_code"] == cfg["index_code"]])
        # 图表数据
        tail = price_df.tail(250)
        sig["dates"] = [str(d.date()) for d in tail["date"]]
        sig["closes"] = [round(float(x), 4) for x in tail["close"]]
        sig["ma20s"] = [None if pd.isna(x) else round(float(x), 4) for x in tail["ma20"]]
        sig["ma60s"] = [None if pd.isna(x) else round(float(x), 4) for x in tail["ma60"]]
        sig["boll_ups"] = [None if pd.isna(x) else round(float(x), 4) for x in tail["boll_up"]]
        sig["boll_lows"] = [None if pd.isna(x) else round(float(x), 4) for x in tail["boll_low"]]
        # 估值分位序列
        sub = hist[hist["index_code"] == cfg["index_code"]].sort_values("date").reset_index(drop=True)
        if len(sub) > 1:
            pcts, dates = [], []
            for i in range(len(sub)):
                if pd.notna(sub.loc[i, "pe"]):
                    pcts.append(round(percentile_of(sub["pe"].iloc[:i + 1], sub.loc[i, "pe"]), 1))
                    dates.append(str(sub.loc[i, "date"])[:10])
            sig["pe_pcts"] = pcts
            sig["pe_dates"] = dates
        else:
            sig["pe_pcts"], sig["pe_dates"] = [], []
        etfs.append(sig)
        log(f"{name}: 收盘 {sig['close']:.3f} | PE分位 {sig['pe_pct']}%({sig['pe_samples']}期) | 结论 {sig['conclusion']}")

    meta["data_date"] = "/".join(sorted(data_dates)) if data_dates else "--"
    meta["badges"] = badges

    if not etfs:
        log("所有标的数据缺失，本次不生成新看板")
        if cloud_mode:
            path = render_fallback(meta)
            log(f"云端容错: 保留/生成占位看板 {path}")
            return 0
        return 1

    path = render_dashboard(meta, ratio, etfs)
    if path is None:
        return 1
    log(f"看板已生成: {path}")
    if "--open" in sys.argv:
        import webbrowser
        webbrowser.open(Path(path).as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
