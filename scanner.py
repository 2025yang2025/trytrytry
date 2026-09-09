# scanner.py
# ==============================================================================
# 台股 6 大策略選股 Pro v2.1 MTF
#
# 策略：
# S1：30K MACD 綠柱縮小 + KD > 30
# S2：60K MACD 綠柱縮小 + KD > 30
# S3：日K MACD > 0 + KD > 20
# S4：週K MACD 最近突破0軸 + MACD > 0 + KD > 50
# S5：月K MACD 最近突破0軸 + MACD > 0 + KD > 50
# S6：S3 + S4 + S5 多週期共振
#
# 額外觸發：
# 60K MACD 綠柱縮小 → 紅柱
# ==============================================================================

import os
import time
import html
import warnings
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# ==============================================================================
# 基本設定
# ==============================================================================
VERSION = "Pro v2.1 MTF"

TWSE_API_URLS = [
    "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
    "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_d"
]

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
    )
}

# ==============================================================================
# 選股參數
# ==============================================================================
MIN_AVG_VOLUME_20 = 1_000_000  # 20日平均成交量門檻 (股)

DAILY_CHUNK_SIZE = 150
INTRADAY_CHUNK_SIZE = 50

# 30K / 60K KD 門檻 (已調整為 30)
INTRADAY_KD_THRESHOLD = 30

# 日/週/月 KD 門檻
DAILY_KD_THRESHOLD = 20
WEEKLY_KD_THRESHOLD = 50
MONTHLY_KD_THRESHOLD = 50

# MACD 零軸突破回溯 K 棒數
WEEKLY_ZERO_CROSS_LOOKBACK = 6
MONTHLY_ZERO_CROSS_LOOKBACK = 3

# 60K 觸發回溯
M60_TRIGGER_LOOKBACK = 2

# 分時圖掃描上限 (若日K候選股過多時做防護)
INTRADAY_SCAN_LIMIT = 100

# ==============================================================================
# 動態股票名稱
# ==============================================================================
DYNAMIC_STOCK_NAMES = {}

# ==============================================================================
# 安全轉數字
# ==============================================================================
def safe_float(value, default=np.nan):
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.replace(",", "").strip()
        result = float(value)
        if np.isnan(result):
            return default
        return result
    except Exception:
        return default

# ==============================================================================
# Yahoo 股票代號轉換
# ==============================================================================
def get_ticker_code(code: str) -> str:
    code = str(code).strip()
    if code.endswith(".TW") or code.endswith(".TWO"):
        return code
    return f"{code}.TW"

# ==============================================================================
# HTML 轉義
# ==============================================================================
def escape_html(text: str) -> str:
    return html.escape(str(text), quote=False)

# ==============================================================================
# 股票名稱標籤
# ==============================================================================
def get_stock_label(code: str, name: Optional[str] = None) -> str:
    if name:
        return f"{code} {name}"
    if code in DYNAMIC_STOCK_NAMES:
        return f"{code} {DYNAMIC_STOCK_NAMES[code]}"
    return code

# ==============================================================================
# TWSE 股票清單取得 (含備用機制)
# ==============================================================================
def fetch_all_taiwan_market_tickers() -> pd.DataFrame:
    for url in TWSE_API_URLS:
        try:
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=15, verify=False)
            if response.status_code == 200:
                data = response.json()
                if data and isinstance(data, list):
                    df = pd.DataFrame(data)
                    
                    code_col = next((c for c in ['證券代號', 'Code', 'code'] if c in df.columns), None)
                    name_col = next((c for c in ['證券名稱', 'Name', 'name'] if c in df.columns), None)
                    
                    if code_col:
                        df = df.rename(columns={code_col: "code"})
                        if name_col:
                            df = df.rename(columns={name_col: "name"})
                        else:
                            df["name"] = ""
                            
                        df["code"] = df["code"].astype(str).str.strip()
                        df = df[df["code"].str.match(r"^\d{4}$", na=False)].copy()
                        
                        if not df.empty:
                            print(f"✅ 成功自 TWSE 取得 {len(df)} 檔股票資訊")
                            return df
        except Exception as e:
            print(f"⚠️ 嘗試讀取 {url} 失敗: {e}")
            
    print("❌ 無法取得 TWSE 股票清單")
    return pd.DataFrame()

# ==============================================================================
# Safe Yahoo Finance Download
# ==============================================================================
def safe_download_yf(
    tickers: List[str], period: str, interval: str, retries: int = 3
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    for attempt in range(retries):
        try:
            data = yf.download(
                tickers=tickers,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=True,
            )
            if data is None or data.empty:
                raise ValueError("Yahoo Finance 無資料")
            return data
        except Exception as e:
            print(f"⚠️ Yahoo 下載失敗 ({attempt + 1}/{retries})：{e}")
            time.sleep(2)

    return pd.DataFrame()

# ==============================================================================
# DataFrame 正規化
# ==============================================================================
def normalize_dataframe(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()

    try:
        if isinstance(data.columns, pd.MultiIndex):
            if ticker in data.columns.get_level_values(0):
                df = data[ticker].copy()
            elif ticker in data.columns.get_level_values(1):
                df = data.xs(ticker, axis=1, level=1).copy()
            else:
                df = data.copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(-1)
        else:
            df = data.copy()

        df.columns = [str(c).lower() for c in df.columns]
        required = ["open", "high", "low", "close", "volume"]

        for col in required:
            if col not in df.columns:
                return pd.DataFrame()

        df = df[required].copy()
        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["close"])
        return df
    except Exception as e:
        print(f"⚠️ DataFrame 正規化失敗 {ticker}: {e}")
        return pd.DataFrame()

# ==============================================================================
# 重採樣 週K / 月K
# ==============================================================================
def resample_klines(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    resampled = df.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return resampled.dropna(subset=["close"])

# ==============================================================================
# MACD 計算
# ==============================================================================
def calculate_macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

# ==============================================================================
# KD 計算
# ==============================================================================
def calculate_kd(
    df: pd.DataFrame, period: int = 9, k_period: int = 3, d_period: int = 3
):
    low_min = df["low"].rolling(period).min()
    high_max = df["high"].rolling(period).max()
    denominator = high_max - low_min

    rsv = (df["close"] - low_min) / denominator.replace(0, np.nan) * 100
    k = rsv.ewm(alpha=1 / k_period, adjust=False).mean()
    d = k.ewm(alpha=1 / d_period, adjust=False).mean()
    return k, d

# ==============================================================================
# 判斷指標條件
# ==============================================================================
def check_macd_green_shrink(hist: pd.Series) -> bool:
    """判斷 MACD 綠柱 (hist < 0) 是否正在縮小 (當前柱形大於上一根)"""
    if len(hist) < 2:
        return False
    curr, prev = hist.iloc[-1], hist.iloc[-2]
    return (curr < 0) and (curr > prev)

def check_zero_cross(macd: pd.Series, lookback: int) -> bool:
    """判斷最近 lookback 根 K 棒內是否有從 <= 0 向上穿越至 > 0"""
    if len(macd) < lookback + 1:
        return False
    subset = macd.iloc[-(lookback + 1):]
    for i in range(1, len(subset)):
        if subset.iloc[i - 1] <= 0 and subset.iloc[i] > 0:
            return True
    return False

def check_60k_trigger(hist: pd.Series) -> bool:
    """判斷 60K MACD 綠柱轉紅柱 (前值 < 0，當前 > 0)"""
    if len(hist) < 2:
        return False
    return hist.iloc[-2] < 0 and hist.iloc[-1] > 0

# ==============================================================================
# 單股分析
# ==============================================================================
def analyze_stock(code: str, df_daily: pd.DataFrame, df_30k: Optional[pd.DataFrame], df_60k: Optional[pd.DataFrame]) -> Dict[str, Any]:
    result = {
        "code": code,
        "label": get_stock_label(code),
        "s1": False, "s2": False, "s3": False,
        "s4": False, "s5": False, "s6": False,
        "trigger_60k": False
    }

    if df_daily.empty or len(df_daily) < 35:
        return result

    # 檢查 20 日均量
    avg_vol_20 = df_daily["volume"].iloc[-20:].mean()
    if avg_vol_20 < MIN_AVG_VOLUME_20:
        return result

    # --- 日 K ---
    macd_d, _, hist_d = calculate_macd(df_daily["close"])
    k_d, d_d = calculate_kd(df_daily)
    
    s3_cond = (macd_d.iloc[-1] > 0) and (k_d.iloc[-1] > DAILY_KD_THRESHOLD)
    result["s3"] = s3_cond

    # --- 週 K ---
    df_weekly = resample_klines(df_daily, "W-FRI")
    if len(df_weekly) >= 35:
        macd_w, _, _ = calculate_macd(df_weekly["close"])
        k_w, _ = calculate_kd(df_weekly)
        s4_cond = (
            macd_w.iloc[-1] > 0 and
            k_w.iloc[-1] > WEEKLY_KD_THRESHOLD and
            check_zero_cross(macd_w, WEEKLY_ZERO_CROSS_LOOKBACK)
        )
        result["s4"] = s4_cond

    # --- 月 K ---
    df_monthly = resample_klines(df_daily, "ME")
    if len(df_monthly) >= 35:
        macd_m, _, _ = calculate_macd(df_monthly["close"])
        k_m, _ = calculate_kd(df_monthly)
        s5_cond = (
            macd_m.iloc[-1] > 0 and
            k_m.iloc[-1] > MONTHLY_KD_THRESHOLD and
            check_zero_cross(macd_m, MONTHLY_ZERO_CROSS_LOOKBACK)
        )
        result["s5"] = s5_cond

    # S6 多週期共振
    result["s6"] = result["s3"] and result["s4"] and result["s5"]

    # --- 30K 分時 ---
    if df_30k is not None and not df_30k.empty and len(df_30k) >= 35:
        _, _, hist_30k = calculate_macd(df_30k["close"])
        k_30k, _ = calculate_kd(df_30k)
        if check_macd_green_shrink(hist_30k) and (k_30k.iloc[-1] > INTRADAY_KD_THRESHOLD):
            result["s1"] = True

    # --- 60K 分時 ---
    if df_60k is not None and not df_60k.empty and len(df_60k) >= 35:
        _, _, hist_60k = calculate_macd(df_60k["close"])
        k_60k, _ = calculate_kd(df_60k)
        if check_macd_green_shrink(hist_60k) and (k_60k.iloc[-1] > INTRADAY_KD_THRESHOLD):
            result["s2"] = True
        
        # 60K 轉紅柱觸發
        if check_60k_trigger(hist_60k):
            result["trigger_60k"] = True

    return result

# ==============================================================================
# 主程式執行入口
# ==============================================================================
def main():
    print(f"🚀 開始執行 台股 6 大策略選股 {VERSION}")
    
    # 1. 取得股票清單
    df_tickers = fetch_all_taiwan_market_tickers()
    if df_tickers.empty:
        print("❌ 無法取得標的，程式結束。")
        return

    codes = df_tickers["code"].tolist()
    for _, row in df_tickers.iterrows():
        DYNAMIC_STOCK_NAMES[row["code"]] = row.get("name", "")

    yf_tickers = [get_ticker_code(c) for c in codes]
    
    # 2. 批次下載日 K 資料
    print(f"📥 開始下載 {len(yf_tickers)} 檔股票之日 K 資料...")
    results = []
    
    for i in range(0, len(yf_tickers), DAILY_CHUNK_SIZE):
        chunk = yf_tickers[i:i + DAILY_CHUNK_SIZE]
        raw_d = safe_download_yf(chunk, period="1y", interval="1d")
        
        for t_code in chunk:
            code = t_code.replace(".TW", "").replace(".TWO", "")
            df_d = normalize_dataframe(raw_d, t_code)
            if df_d.empty:
                continue
            
            # 先做初步日/週/月篩選
            res = analyze_stock(code, df_d, None, None)
            if any([res["s3"], res["s4"], res["s5"]]):
                results.append((code, df_d))

    print(f"🔍 日K級別符合條件之候選股共 {len(results)} 檔，準備下載分時資料...")

    # 3. 針對初步符合者下載 30K / 60K 分時 K 棒 (限制上限)
    candidates = [code for code, _ in results[:INTRADAY_SCAN_LIMIT]]
    cand_yf = [get_ticker_code(c) for c in candidates]

    dict_30k = {}
    dict_60k = {}

    if cand_yf:
        raw_30k = safe_download_yf(cand_yf, period="1mo", interval="30m")
        raw_60k = safe_download_yf(cand_yf, period="1mo", interval="60m")

        for t_code in cand_yf:
            code = t_code.replace(".TW", "").replace(".TWO", "")
            dict_30k[code] = normalize_dataframe(raw_30k, t_code)
            dict_60k[code] = normalize_dataframe(raw_60k, t_code)

    # 4. 彙整最終選股結果
    final_results = []
    for code, df_d in results:
        df_30 = dict_30k.get(code)
        df_60 = dict_60k.get(code)
        res = analyze_stock(code, df_d, df_30, df_60)
        final_results.append(res)

    # 5. 輸出報表
    df_out = pd.DataFrame(final_results)
    print("\n==================================================================")
    print("📊 策略選股結果摘要")
    print("==================================================================")
    for s in ["s1", "s2", "s3", "s4", "s5", "s6", "trigger_60k"]:
        matched = df_out[df_out[s] == True]["label"].tolist() if not df_out.empty else []
        print(f"策略 [{s.upper()}]: 共有 {len(matched)} 檔 -> {', '.join(matched)}")

if __name__ == "__main__":
    main()
