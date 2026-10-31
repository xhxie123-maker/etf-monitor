# -*- coding: utf-8 -*-
"""
ETF 智能监测工具
================
监测标的：515180 红利ETF / 516310 银行ETF / 515220 煤炭ETF
核心指标：价格趋势(MA20/MA60/BOLL/RSI)、指数估值分位、创业板红利比
输出     ：单窗口可视化看板 output/etf_dashboard.html（浏览器打开即看全部）
用法     ：python etf_monitor.py [--open]
          --open 生成后自动打开浏览器看板
数据源   ：东方财富/腾讯/新浪(行情回退链)、中证指数官网(官方估值)、日频PE序列(官方最新PE×价格比估算)
分位口径 ：近1年/近3年/近5年三窗口滚动分位，主信号取近3年；分位旁标注样本数(交易日)与时间跨度
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
# 价格获取窗口（年）：需覆盖最长 PE 分位窗口（近5年≈1250 交易日）
PRICE_YEARS = 5
# PE 分位滚动窗口（交易日）：近1年/近3年/近5年
PE_WINDOWS = {"近1年": 250, "近3年": 750, "近5年": 1250}
PE_MAIN_WINDOW = "近3年"   # 主信号（估值打分/定投档位/长期仓位）使用的窗口
# 指数估值来源代码（中证指数官网）
CSINDEX_MAP = {"000922": "000922", "399986": "399986", "000820": "000820"}

# 信号阈值
TH = {
    "pe_low": 30, "pe_high": 70,          # PE分位 低估/高估
    "rsi_low": 35, "rsi_high": 70,        # RSI 超卖/超买
    "dev": 5.0,                           # 偏离20日线阈值%
    "ratio_low": 0.35, "ratio_high": 0.60 # （V2 起弃用：创业板红利比改用归一化相对强弱 + 分位30/70解释，此阈值仅保留）
}

# 宏观环境阈值（可自行调整）
MACRO_TH = {
    "y10_low": 2.0, "y10_high": 2.8,   # 10Y国债收益率%：<2.0 宽松支撑红利 / >2.8 偏紧
    "mkt_hi": 1.2, "mkt_lo": 0.8,      # 两市成交额较20日均量：>1.2 放量 / <0.8 缩量
}
# 政策风向：无法自动抓取定性政策，每周手动更新此条目（看板原样展示）
MACRO_POLICY = {
    "title": "政策风向（手动维护）",
    "text": "当前无新增重大政策信号；关注点：LPR 利率走向、财政发力节奏、行业监管动态。",
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


def normalize_daily(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """统一行情字段并检查日期、价格、重复记录。"""
    if df is None or df.empty:
        raise ValueError(f"{source}: 空行情")
    aliases = {"日期":"date", "开盘":"open", "收盘":"close", "最高":"high", "最低":"low",
               "成交量":"volume", "成交额":"amount", "涨跌幅":"pct_chg"}
    df = df.rename(columns=aliases).copy()
    if not {"date", "close"}.issubset(df.columns):
        raise ValueError(f"{source}: 缺少 date/close 字段")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ("open", "high", "low", "close", "volume", "amount", "pct_chg"):
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "close"])
    df = df[df["close"] > 0].sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if df.empty: raise ValueError(f"{source}: 清洗后无有效行情")
    if "pct_chg" not in df.columns: df["pct_chg"] = df["close"].pct_change() * 100
    return df


# ============================================================
# 2. 数据获取（东财优先，新浪兜底——海外服务器(如GitHub Actions)访问东财会被拒绝）
# ============================================================
def get_etf_daily_em(code: str):
    """ETF 前复权日线（东方财富）"""
    end = dt.date.today().strftime("%Y%m%d")
    start = (dt.date.today() - dt.timedelta(days=int(PRICE_YEARS * 365))).strftime("%Y%m%d")
    df = ak.fund_etf_hist_em(symbol=code, period="daily",
                             start_date=start, end_date=end, adjust="qfq")
    return normalize_daily(df, "东方财富ETF")


def get_etf_daily_sina(code: str):
    """ETF 日线（新浪，海外可访问；新浪接口为不复权数据，仅作东财失败时的兜底）"""
    prefix = "sh" if str(code).startswith("5") else "sz"
    df = ak.fund_etf_hist_sina(symbol=f"{prefix}{code}")
    df = df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close",
                            "最高": "high", "最低": "low"})
    return normalize_daily(df, "备用行情")


def get_etf_daily_tx(code: str):
    """ETF 日线（腾讯，海外可访问、数据新鲜；接口为不复权数据，仅作东财失败时的兜底）"""
    prefix = "sh" if str(code).startswith("5") else "sz"
    df = ak.stock_zh_a_hist_tx(symbol=f"{prefix}{code}")
    df = df.rename(columns={"date": "date", "open": "open", "close": "close",
                            "high": "high", "low": "low"})
    return normalize_daily(df, "备用行情")


def get_etf_daily(code: str):
    """ETF 日线：东财优先（重试2次），失败依次切腾讯、新浪（腾讯带新鲜度门槛）"""
    df = safe_call("东财ETF行情", get_etf_daily_em, code, attempts=2)
    if df is None:
        log("东财ETF行情不可用，尝试腾讯源")
        df = safe_call("腾讯ETF行情", get_etf_daily_tx, code)
    if df is not None and df["date"].max() < pd.Timestamp(dt.date.today()) - pd.Timedelta(days=10):
        log(f"腾讯ETF行情数据过期(末日期 {df['date'].max().date()})，跳过")
        df = None
    if df is None:
        log("腾讯ETF行情不可用，尝试新浪源")
        df = safe_call("新浪ETF行情", get_etf_daily_sina, code)
    return df


def get_index_daily_em(code: str):
    """指数日线（东方财富）"""
    end = dt.date.today().strftime("%Y%m%d")
    start = (dt.date.today() - dt.timedelta(days=int(PRICE_YEARS * 365))).strftime("%Y%m%d")
    df = ak.index_zh_a_hist(symbol=code, period="daily",
                            start_date=start, end_date=end)
    df = df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close",
                            "最高": "high", "最低": "low", "涨跌幅": "pct_chg"})
    return normalize_daily(df, "指数行情")


def get_index_daily_sina(code: str):
    """指数日线（新浪，海外可访问）"""
    prefix = "sh" if str(code).startswith("000") else "sz"
    df = ak.stock_zh_index_daily(symbol=f"{prefix}{code}")
    df = df.rename(columns={"date": "date", "open": "open", "close": "close",
                            "high": "high", "low": "low"})
    return normalize_daily(df, "备用行情")


def get_index_daily_tx(code: str):
    """指数日线（腾讯，海外可访问，数据新鲜）"""
    prefix = "sh" if str(code).startswith("000") else "sz"
    df = ak.stock_zh_index_daily_tx(symbol=f"{prefix}{code}")
    df = df.rename(columns={"date": "date", "open": "open", "close": "close",
                            "high": "high", "low": "low"})
    return normalize_daily(df, "备用行情")


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
    return normalize_daily(df, "指数行情")[["date", "pe", "pb", "dy"]]


# ============================================================
# 3. 指标计算
# ============================================================
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """均线 / RSI / BOLL / MACD / 偏离度"""
    df = df.copy()
    for w in (5, 20, 60, 120):
        df[f"ma{w}"] = df["close"].rolling(w).mean()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    # RSI(14) Wilder：前14期均值初始化，之后按 Wilder 递推。
    period = 14
    avg_gain = pd.Series(np.nan, index=df.index, dtype=float)
    avg_loss = pd.Series(np.nan, index=df.index, dtype=float)
    if len(df) > period:
        avg_gain.iloc[period] = gain.iloc[1:period+1].mean()
        avg_loss.iloc[period] = loss.iloc[1:period+1].mean()
        for i in range(period+1, len(df)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1]*(period-1)+gain.iloc[i])/period
            avg_loss.iloc[i] = (avg_loss.iloc[i-1]*(period-1)+loss.iloc[i])/period
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - 100/(1+rs)
    df.loc[(avg_loss == 0) & (avg_gain > 0), "rsi"] = 100
    df.loc[(avg_loss == 0) & (avg_gain == 0), "rsi"] = 50
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


def build_pe_series(price_df, val_latest, acc_hist) -> "pd.Series | None":
    """
    构建日频 PE 序列（估算口径）：
      官方最新 PE × 指数点位/收盘价比例回推全历史（假设 EPS 近似恒定），
      官方积累库中的真实 PE 覆盖对应日期（近约20个交易日为官方值），
      末端强制对齐官方最新 PE。
    注：price_df 应传指数日线（真实点位，无复权问题）；仅指数日线缺失时
        才允许用 ETF 行情兜底（分位口径会略差）。
    返回 pd.Series(index=日期, name='pe') 升序；数据不足返回 None。
    """
    if price_df is None or price_df.empty or not val_latest or pd.isna(val_latest.get("pe")):
        return None
    pe0 = float(val_latest["pe"])
    p = price_df[["date", "close"]].dropna().sort_values("date")
    if len(p) < 10:
        return None
    # 锚点：估值日期（官方/积累库最近一期）对应的收盘价；缺失时用最新收盘价
    try:
        anchor = pd.to_datetime(val_latest.get("date") or p["date"].iloc[-1])
    except Exception:
        anchor = p["date"].iloc[-1]
    sub = p[p["date"] <= anchor]
    anchor_close = float(sub["close"].iloc[-1]) if not sub.empty else float(p["close"].iloc[-1])
    if not anchor_close or anchor_close <= 0:
        return None
    pe_s = pd.Series((pe0 * p["close"] / anchor_close).values,
                     index=pd.DatetimeIndex(p["date"]), name="pe")
    # 官方真实 PE 覆盖（积累库含中证官网近20期）
    if acc_hist is not None and not acc_hist.empty:
        acc = acc_hist.dropna(subset=["pe"])
        for _, row in acc.iterrows():
            d = pd.to_datetime(row["date"])
            if d in pe_s.index:
                pe_s.loc[d] = float(row["pe"])
    if pd.to_datetime(anchor).date() == pd.to_datetime(pe_s.index[-1]).date():
        pe_s.iloc[-1] = pe0
    return pe_s


def calc_pe_windows(pe_s: "pd.Series | None"):
    """
    三窗口（近1年/近3年/近5年）PE 分位。
    返回 {窗口名: {pct, n, start, end}}；窗口样本不足返回 None；无数据返回 None。
    """
    if pe_s is None or len(pe_s) == 0:
        return None
    out = {}
    for label, n in PE_WINDOWS.items():
        w = pe_s.tail(n)
        if len(w) < int(n * 0.80):
            out[label] = None
            continue
        reference = w.iloc[:-1] if len(w) > 1 else w
        out[label] = {
            "pct": round(percentile_of(reference, float(w.iloc[-1])), 1),
            "n": int(len(w)),
            "start": str(w.index[0].date()),
            "end": str(w.index[-1].date()),
        }
    return out


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
    elif hist is not None and not hist[hist["index_code"] == str(code).zfill(6)].empty:
        src = "本地积累库(上次数据)"
    else:
        return None, hist, "暂无估值数据"
    sub = hist[hist["index_code"] == str(code).zfill(6)].sort_values("date")
    if sub.empty:
        return None, hist, src
    latest = sub.iloc[-1].to_dict()
    latest["date"] = str(latest["date"])[:10]
    return latest, hist, src


# ============================================================
# 5. 信号引擎
# ============================================================
def build_signal(name: str, cfg: dict, price: pd.DataFrame, val_latest: dict, pe_series: "pd.Series | None"):
    """返回该 ETF 的全部指标、信号、建议。pe_series 为日频 PE 序列（估算口径）"""
    last = price.iloc[-1]
    pe = val_latest.get("pe") if val_latest else np.nan
    pb = val_latest.get("pb") if val_latest else np.nan
    dy = val_latest.get("dy") if val_latest else np.nan
    pe_windows = calc_pe_windows(pe_series)
    w_main = pe_windows.get(PE_MAIN_WINDOW) if pe_windows else None
    pe_pct = w_main["pct"] if w_main else np.nan  # 主信号分位（近3年）
    pe_pcts, pe_dates = [], []
    if pe_series is not None and len(pe_series) > 1:
        # 近3年滚动窗口分位走势（750 交易日窗口逐点）
        v = pe_series.tail(1250)
        idx = v.index
        for i in range(len(v)):
            w = v.iloc[max(0, i - (PE_WINDOWS[PE_MAIN_WINDOW] - 1)): i + 1]
            pe_pcts.append(round(percentile_of(w, float(v.iloc[i])), 1))
            pe_dates.append(str(idx[i].date()))

    score = 0.0
    signals = []

    # ---- 估值 ----
    if not np.isnan(pe_pct):
        if pe_pct < TH["pe_low"]:
            score += 2; signals.append(f"估值低估·3年分位{pe_pct:.0f}%")
        elif pe_pct > TH["pe_high"]:
            score -= 2; signals.append(f"估值高估·3年分位{pe_pct:.0f}%")
        else:
            signals.append(f"估值中性·3年分位{pe_pct:.0f}%")

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

    # ---- 长期趋势过滤 ----
    ma60, ma120 = last.get("ma60", np.nan), last.get("ma120", np.nan)
    downtrend = pd.notna(ma60) and pd.notna(ma120) and ma60 < ma120
    if downtrend and score >= 0.5:
        signals.append("长期趋势过滤：MA60低于MA120，限制买入信号")

    # ---- 结论 ----
    if score >= 2.5 and not downtrend:
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
        # 阈值与信号段统一（TH["rsi_low"]/TH["rsi_high"]），避免两处口径不一致
        if rsi > TH["rsi_high"]:
            t_text = "高位慎追，做T以减仓为主"
        elif rsi < TH["rsi_low"]:
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
        "ma120": float(last["ma120"]) if not np.isnan(last.get("ma120")) else None,
        "rsi": None if np.isnan(rsi) else round(rsi, 1),
        "dev20": round(dev20, 2),
        "pe": None if np.isnan(pe) else round(float(pe), 2),
        "pb": None if np.isnan(pb) else round(float(pb), 2),
        "dy": None if np.isnan(dy) else round(float(dy), 2),
        "pe_pct": None if np.isnan(pe_pct) else round(pe_pct, 1),
        "pe_windows": pe_windows,   # {近1年/近3年/近5年: {pct,n,start,end}}
        "pe_pcts": pe_pcts, "pe_dates": pe_dates,
        "signals": signals,
        "score": round(score, 1),
        "conclusion": conclusion, "concl_color": concl_color, "concl_txt": concl_txt,
        "dca": dca, "dca_color": dca_color,
        "support": support, "pressure": pressure, "t_text": t_text,
        "long_txt": long_txt,
        "latest_date": str(price["date"].iloc[-1].date()),
    }


# ============================================================
# 5. 宏观环境（利率 / 政策 / 情绪）
# ============================================================
def get_mkt_amount():
    """两市成交额序列（上证000001 + 深证成指399001，腾讯指数日线 amount，单位：千元）。
    海外环境（GitHub Actions）腾讯通道可用；失败返回 None 不阻塞主流程。"""
    try:
        a = ak.stock_zh_index_daily_tx(symbol="sh000001")
        b = ak.stock_zh_index_daily_tx(symbol="sz399001")
        da, db = a.sort_values("date"), b.sort_values("date")
        m = pd.merge(da[["date", "amount"]], db[["date", "amount"]],
                     on="date", suffixes=("_a", "_b"))
        m["total"] = m["amount_a"] + m["amount_b"]
        return m[["date", "total"]]
    except Exception as e:
        log(f"[数据源] 两市成交额获取失败: {type(e).__name__}: {str(e)[:120]}")
        return None


def get_macro() -> dict:
    """宏观环境采集：
    利率 = 中国10Y国债收益率（英为财情）+ 较约20日前变化(bp)
    政策 = LPR(1Y/5Y，央行) + 手动维护条目 MACRO_POLICY
    情绪 = 两市成交额（上证+深证，腾讯）及其较20日均量比
    任一失败仅置 None，看板显示 '--'，不影响 ETF 主流程。"""
    m = {"y10": None, "y10_chg": None, "y10_label": None, "y10_date": None,
         "lpr1": None, "lpr5": None,
         "mkt": None, "mkt_ratio": None, "mkt_date": None, "ok": True}

    # ---- 利率环境：10Y 国债收益率 ----
    try:
        d = ak.bond_zh_us_rate(start_date=(dt.date.today() - dt.timedelta(days=120)).strftime("%Y%m%d"))
        d = d.dropna(subset=["中国国债收益率10年"]).sort_values("日期").reset_index(drop=True)
        if len(d) >= 2:
            y = float(d["中国国债收益率10年"].iloc[-1])
            base = float(d["中国国债收益率10年"].iloc[-21]) if len(d) > 21 else float(d["中国国债收益率10年"].iloc[0])
            m["y10"], m["y10_chg"] = y, (y - base) * 100  # bp
            m["y10_date"] = str(d["日期"].iloc[-1])
            if y < MACRO_TH["y10_low"]:
                m["y10_label"] = "低利率·宽松（利好红利/高股息）"
            elif y > MACRO_TH["y10_high"]:
                m["y10_label"] = "高利率·偏紧（压制估值）"
            else:
                m["y10_label"] = "利率中性"
    except Exception as e:
        log(f"[数据源] 10Y国债收益率获取失败: {type(e).__name__}: {str(e)[:120]}")

    # ---- 政策风向：LPR（自动硬指标）+ 手动条目 ----
    try:
        d = ak.macro_china_lpr()
        d = d.sort_values("TRADE_DATE")
        m["lpr1"] = float(d["LPR1Y"].iloc[-1])
        m["lpr5"] = float(d["LPR5Y"].iloc[-1])
    except Exception as e:
        log(f"[数据源] LPR获取失败: {type(e).__name__}: {str(e)[:120]}")

    # ---- 市场情绪：两市成交额 ----
    try:
        mm = get_mkt_amount()
        if mm is not None and len(mm) >= 21:
            last = float(mm["total"].iloc[-1])
            avg20 = float(mm["total"].iloc[-21:-1].mean())
            m["mkt"] = last
            m["mkt_ratio"] = last / avg20 if avg20 > 0 else None
            m["mkt_date"] = str(mm["date"].iloc[-1])
    except Exception as e:
        log(f"[数据源] 市场情绪采集失败: {type(e).__name__}: {str(e)[:120]}")

    return m


def build_ratio():
    """创业板红利比 = 创业板指/中证红利，返回最新值+分位+建议+3年序列"""
    gem = safe_call("创业板指行情", get_index_daily, GEM_INDEX)
    div = safe_call("中证红利行情", get_index_daily, DIV_INDEX)
    if gem is None or div is None:
        return None
    m = pd.merge(gem[["date", "close"]], div[["date", "close"]],
                 on="date", suffixes=("_gem", "_div"))
    m = m.dropna()
    m = m[(m["close_gem"] > 0) & (m["close_div"] > 0)].copy()
    if m.empty:
        return None
    m["ratio"] = (m["close_gem"] / m["close_gem"].iloc[0]) / (m["close_div"] / m["close_div"].iloc[0])
    # 仅取近3年样本（与看板“近3年分位”口径一致；腾讯/新浪源返回全量历史）
    m = m.tail(int(HISTORY_YEARS * 250)).reset_index(drop=True)
    if len(m) < 20:
        return None
    latest = float(m["ratio"].iloc[-1])
    pct = percentile_of(m["ratio"].iloc[:-1], latest)
    if pct < 30:
        sug, sug_color = "成长相对表现处于历史低位：观察相对表现，不据此单独交易", "#f97316"
    elif pct > 70:
        sug, sug_color = "成长相对表现处于历史高位：观察红利相对表现，不据此单独交易", "#ef4444"
    else:
        sug, sug_color = "相对表现处于历史中段：维持配置纪律", "#94a3b8"
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
def render_dashboard(meta: dict, ratio: dict, etfs: list, macro: dict = None):
    echarts_src = "../libs/echarts.min.js"
    payload = {
        "meta": meta,
        "ratio": ratio,
        "etfs": etfs,
        "macro": macro if macro else {},
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
  .wins {{ font-size:11px; color:var(--sub); font-weight:400; line-height:1.55; margin-top:2px; }}
  .wins b {{ color:var(--text); font-weight:600; }}
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
  .macro-card {{ grid-column:span 3; }}
  .macro-row {{ display:flex; gap:14px; flex-wrap:wrap; }}
  .macro-item {{ flex:1; min-width:200px; background:rgba(255,255,255,0.03); border:1px solid var(--line);
                 border-radius:8px; padding:10px 12px; }}
  .macro-item .k {{ display:block; color:var(--sub); font-size:11px; margin-bottom:5px; }}
  .macro-item .v {{ font-weight:600; font-size:13px; line-height:1.6; }}
  .macro-item .v.ok {{ color:#4ade80; }} .macro-item .v.warn {{ color:#fbbf24; }}
  .macro-note {{ margin-top:8px; }}
  .etf-grid {{ grid-template-columns:repeat(3,1fr); }}
  .price-charts {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:14px; }}
  .tall {{ height:300px; }}
  .pe-charts {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:14px; }}
  .foot {{ margin-top:16px; color:var(--sub); font-size:12px; text-align:center; line-height:1.8; }}
  .rules {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:14px; }}
  .rules li {{ margin-left:16px; }}
  @media (max-width:900px) {{ .grid,.etf-grid,.price-charts,.pe-charts,.rules {{ grid-template-columns:1fr; }}
      .ratio-card,.macro-card {{ grid-column:span 1; }} }}
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

  <!-- 宏观环境：利率 / 政策 / 情绪 -->
  <div class="grid">
  <div class="card macro-card">
    <h3>宏观环境 <span class="code">利率环境 · 政策风向 · 市场情绪 ｜ 每日自动更新</span></h3>
    <div class="macro-row">
      <div class="macro-item"><span class="k">利率环境</span><span class="v" id="macro-rate">--</span></div>
      <div class="macro-item"><span class="k">政策风向</span><span class="v" id="macro-policy">--</span></div>
      <div class="macro-item"><span class="k">市场情绪</span><span class="v" id="macro-mkt">--</span></div>
    </div>
    <div class="mini macro-note" id="macro-note"></div>
  </div>
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
        <div class="ratio-mark"><span>&lt;30% 成长相对低位</span><span>中段</span><span>&gt;70% 相对高位</span></div>
        <div class="banner" id="ratio-sug" style="margin-top:10px">--</div>
        <div class="mini">比值=创业板指收盘÷中证红利收盘；分位为近{HISTORY_YEARS}年历史百分位。相对强弱走高=创业板相对中证红利表现更强；分位不代表估值便宜。</div>
      </div>
      <div class="ratio-chart" id="ratio-chart"></div>
    </div>
  </div>
  </div>

  <!-- 三只 ETF 卡片 -->
  <div class="grid etf-grid" id="etf-cards"></div>

  <!-- 价格走势 -->
  <div class="price-charts" id="price-charts"></div>

  <!-- 估值分位走势（近3年滚动窗口） -->
  <div class="pe-charts" id="pe-charts"></div>

  <!-- 操作纪律 -->
  <div class="rules">
    <div class="card"><h3>定投纪律 <span style="font-size:11px;color:var(--sub);font-weight:400">（按近3年PE分位）</span></h3><ul class="mini" style="list-style:none">
      <li>· PE分位&lt;20%：双倍定投</li>
      <li>· PE分位20%~40%：正常定投</li>
      <li>· PE分位40%~70%：减半定投</li>
      <li>· PE分位&gt;70%：暂停定投，仅持底仓</li>
      <li>· 单日大跌&gt;3%可额外加投一份</li>
    </ul></div>
    <div class="card"><h3>再平衡</h3><ul class="mini" style="list-style:none">
      <li>· 每半年（6月/12月）强制再平衡一次</li>
      <li>· 单只仓位偏离基准 ±10% 触发调整</li>
      <li>· 创业板/红利相对强弱分位&lt;30%或&gt;70%时关注风格切换，不据此单独交易</li>
    </ul></div>
    <div class="card"><h3>长期纪律</h3><ul class="mini" style="list-style:none">
      <li>· 煤炭为强周期品种，仓位上限 40%，切忌单吊</li>
      <li>· 银行低估值弹性小，作为组合压舱石</li>
      <li>· 红利在成长牛市中会跑输大盘，保持组合思维</li>
      <li>· 所有信号为量化参考，须结合宏观与自身资金状况</li>
    </ul></div>
  </div>

  <div class="foot">
    数据来源：东方财富/腾讯/新浪（行情）· 中证指数官网（官方估值）· 英为财情（国债利率）· 央行（LPR）· 腾讯指数（两市成交）<br>
    PE分位口径：近1年250交易日 / 近3年750 / 近5年1250，样本数为窗口内实际交易日；主信号取近3年。<br>
    PE 历史序列为估算口径（官方最新PE × 指数点位比例回推，近约20个交易日为官方真实值），分位供相对估值参考。<br>
    宏观指标为每日自动采集（利率/情绪）与手动维护条目（政策风向）的组合，仅供环境参考。<br>
    本工具仅作数据监测与量化参考，不构成投资建议。市场有风险，投资需谨慎。
  </div>
</div>

<script>
const DATA = {payload_json};

/* 徽章：数据源状态 */
const badges = DATA.meta.badges.map(b => `<span class="badge ${{b.ok?'ok':'warn'}}">${{b.label}}</span>`).join('');
document.getElementById('badges').innerHTML = badges;

/* 宏观环境：利率 / 政策 / 情绪 */
(function () {{
  const M = DATA.macro || {{}};
  const el = (id) => document.getElementById(id);
  const y10 = M.y10 !== null && M.y10 !== undefined;
  if (y10) {{
    el('macro-rate').innerHTML = '10Y国债 <b>' + M.y10.toFixed(2) + '%</b>（' + (M.y10_chg>=0?'+':'') + M.y10_chg.toFixed(0) + 'bp 较20日前 · ' + M.y10_date + '）';
    const r = el('macro-rate');
    r.classList.add(M.y10_label && M.y10_label.indexOf('宽松')>=0 ? 'ok' : (M.y10_label && M.y10_label.indexOf('偏紧')>=0 ? 'warn' : ''));
  }} else {{
    el('macro-rate').textContent = '10Y国债 --（数据暂缺）';
  }}
  const pol = [];
  if (M.lpr1 !== null && M.lpr1 !== undefined) pol.push('LPR 1Y <b>' + M.lpr1.toFixed(1) + '%</b> / 5Y <b>' + M.lpr5.toFixed(1) + '%</b>');
  if (pol.length) el('macro-policy').innerHTML = pol.join(' ｜ ');
  else el('macro-policy').textContent = 'LPR --（数据暂缺）';
  const mk = M.mkt !== null && M.mkt !== undefined;
  if (mk) {{
    const yiyuan = (M.mkt/1e5).toFixed(0);          // 千元→亿元
    const wanYi = (M.mkt/1e9).toFixed(2);           // 千元→万亿元
    const rt = M.mkt_ratio;
    let lab = '平稳', cls = '';
    if (rt !== null && rt !== undefined) {{
      if (rt >= 1.2) {{ lab = '放量(' + rt.toFixed(2) + 'x20日均)'; cls = 'warn'; }}
      else if (rt <= 0.8) {{ lab = '缩量(' + rt.toFixed(2) + 'x20日均)'; cls = 'ok'; }}
      else lab = '平稳(' + rt.toFixed(2) + 'x20日均)';
    }}
    el('macro-mkt').innerHTML = '两市成交约 <b>' + wanYi + '万亿</b>（' + yiyuan + '亿 · ' + M.mkt_date + '）· ' + lab;
    if (cls) el('macro-mkt').classList.add(cls);
  }} else {{
    el('macro-mkt').textContent = '两市成交 --（数据暂缺）';
  }}
  el('macro-note').textContent = '利率/情绪为自动采集口径；政策风向为手动维护条目（每周更新）——详见操作纪律与README。';
}})();

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
        data:[{{yAxis:1.0,label:{{formatter:'相对基准 1.0',color:'#8a93ab'}},lineStyle:{{color:'#8a93ab',type:'dashed'}}}}]}}
    }}]
  }});
  window.addEventListener('resize', () => rc.resize());
}}

/* ETF 卡片 */
function fmtNum(v, d=2) {{ return v===null||v===undefined ? '--' : Number(v).toFixed(d); }}
const cardsEl = document.getElementById('etf-cards');
DATA.etfs.forEach(e => {{
  const up = e.pct_chg >= 0;
  const w = e.pe_windows || null;
  const w1 = w && w['近1年'] ? w['近1年'] : null;
  const w3 = w && w['近3年'] ? w['近3年'] : null;
  const w5 = w && w['近5年'] ? w['近5年'] : null;
  const peInfo = w1
    ? `PE ${{fmtNum(e.pe)}} · 1年 <b>${{w1.pct}}%</b> / 3年 <b>${{w3.pct}}%</b> / 5年 <b>${{w5.pct}}%</b>`
    : `PE ${{fmtNum(e.pe)}} · 分位 积累中`;
  const peSamples = w1
    ? `<div class="wins"><span>近1年 样本 <b>${{w1.n}}</b> 期 · ${{w1.start}}~${{w1.end}}</span><span>近3年 样本 <b>${{w3.n}}</b> 期 · ${{w3.start}}~${{w3.end}}</span><span>近5年 样本 <b>${{w5.n}}</b> 期 · ${{w5.start}}~${{w5.end}}</span></div>`
    : '--';
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
        <span class="k">分位样本</span><span class="v">${{peSamples}}</span>
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
  div.innerHTML = `<h3>${{e.name}} 估值分位走势 <span class="code">PE分位·近3年滚动窗口(750交易日)</span></h3>
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
    series:[{{type:'line',data:e.pe_pcts,symbol:'none',lineStyle:{{color:'rgba(148,163,184,0.85)',width:1}},
             areaStyle:{{color:'rgba(148,163,184,0.12)'}}}}]
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
              {"label": "估值分位·日频估算(3Y)", "ok": True}]

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
        if "官网" not in v_src:
            badges[1]["ok"] = False  # 本次未取到官方估值（沿用积累库/上次数据）
        if "暂无" in v_src:
            badges[1]["ok"] = False
            badges[2]["ok"] = False

        # 日频 PE 序列（估算口径：官方最新PE × 指数点位比例回推，官方积累库覆盖近期）
        # 说明：PE 属于指数（000922/399986/000820），指数日线为真实点位、无复权问题；
        #       ETF 价格受分红/复权因子影响会扭曲 PE 回推，故 PE 序列一律基于指数日线。
        idx_df = safe_call(f"{cfg['index_name']}指数行情", get_index_daily, cfg["index_code"])
        if idx_df is None:
            log(f"{name} 指数日线缺失，停止生成日频PE估算，避免复权ETF价格冒充指数点位")
        sub_acc = hist[hist["index_code"] == cfg["index_code"]]
        pe_series = build_pe_series(idx_df, val_latest, sub_acc) if idx_df is not None else None
        if pe_series is None or len(pe_series) < 250:
            badges[2]["ok"] = False
            log(f"{name} 日频PE序列不足(样本{0 if pe_series is None else len(pe_series)})，分位不可用")

        sig = build_signal(name, cfg, price_df, val_latest, pe_series)
        # 图表数据
        tail = price_df.tail(250)
        sig["dates"] = [str(d.date()) for d in tail["date"]]
        sig["closes"] = [round(float(x), 4) for x in tail["close"]]
        sig["ma20s"] = [None if pd.isna(x) else round(float(x), 4) for x in tail["ma20"]]
        sig["ma60s"] = [None if pd.isna(x) else round(float(x), 4) for x in tail["ma60"]]
        sig["boll_ups"] = [None if pd.isna(x) else round(float(x), 4) for x in tail["boll_up"]]
        sig["boll_lows"] = [None if pd.isna(x) else round(float(x), 4) for x in tail["boll_low"]]
        etfs.append(sig)
        w = sig.get("pe_windows") or {}
        p1 = w.get("近1年") or {}
        p3 = w.get("近3年") or {}
        p5 = w.get("近5年") or {}
        log(f"{name}: 收盘 {sig['close']:.3f} | PE {sig['pe']} 分位 1Y {p1.get('pct')}% / "
            f"3Y {p3.get('pct')}% / 5Y {p5.get('pct')}% (n={p1.get('n')}/{p3.get('n')}/{p5.get('n')}) | 结论 {sig['conclusion']}")

    meta["data_date"] = "/".join(sorted(data_dates)) if data_dates else "--"
    meta["badges"] = badges

    if not etfs:
        log("所有标的数据缺失，本次不生成新看板")
        if cloud_mode:
            path = render_fallback(meta)
            log(f"云端容错: 保留/生成占位看板 {path}")
            return 0
        return 1

    path = render_dashboard(meta, ratio, etfs, macro=get_macro())
    if path is None:
        return 1
    log(f"看板已生成: {path}")
    if "--open" in sys.argv:
        import webbrowser
        webbrowser.open(Path(path).as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
