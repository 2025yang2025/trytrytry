# scanner.py
# ==============================================================================
# 台股 策略選股 Pro v2.2 MTF
#
# 策略架構：
# 第一策略：日K MACD > 0 + KD > 20 (原 S3)
# 第二策略：週K MACD 最近突破0軸 + MACD > 0 + KD > 50 (原 S4)
#
# 顯示格式：代號 中文名稱 最新價格
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
VERSION = "Pro v2.2 MTF"

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
# 選股參數與條件定義
# ==============================================================================
MIN_AVG_VOLUME_20 = 1_000_000  # 20日平均成交量門檻 (股)
DAILY_CHUNK_SIZE = 150

# 策略 1 條件設定 (日K)
STRATEGY_1_NAME = "第一策略：日K MACD > 0 + KD > 20"
DAILY_KD_THRESHOLD = 20

# 策略 2 條件設定 (週K)
STRATEGY_2_NAME = "第二策略：週K MACD 最近突破0軸 + MACD > 0 + KD > 50"
WEEKLY_KD_THRESHOLD = 50
WEEKLY_ZERO_CROSS_LOOKBACK = 6

# Dynamic stock names storage
DYNAMIC_STOCK_NAMES = {}

# ==============================================================================
# 工具函式
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

def get_ticker_code(code: str) -> str:
    code = str(code).strip()
    if code.endswith(".TW") or code.endswith(".TWO"):
        return code
    return f"{code}.TW"

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
                        df["name"] = df[name_col] if name_col else ""
                        df["code"] = df["code"].astype(str).str.strip()
                        df = df[df["code"].str.match(r"^\d{4}$", na=False)].copy()
                        
                        if not df.empty:
                            print(f"✅ 成功自 TWSE 取得 {len(df)} 檔股票資訊")
                            return df
        except Exception as e:
            print(f"⚠️ 嘗試讀取 {url} 失敗: {e}")
            
    print("❌ 無法取得 TWSE 股票清單")
    return pd.DataFrame()

def safe_download_yf(tickers: List[str], period: str, interval: str, retries: int = 3) -> pd.DataFrame:
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
# 技術指標計算
# ==============================================================================
def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def calculate_kd(df: pd.DataFrame, period: int = 9, k_period: int = 3, d_period: int = 3):
    low_min = df["low"].rolling(period).min()
    high_max = df["high"].rolling(period).max()
    denominator = high_max - low_min

    rsv = (df["close"] - low_min) / denominator.replace(0, np.nan) * 100
    k = rsv.ewm(alpha=1 / k_period, adjust=False).mean()
    d = k.ewm(alpha=1 / d_period, adjust=False).mean()
    return k, d

def check_zero_cross(macd: pd.Series, lookback: int) -> bool:
    if len(macd) < lookback + 1:
        return False
    subset = macd.iloc[-(lookback + 1):]
    for i in range(1, len(subset)):
        if subset.iloc[i - 1] <= 0 and subset.iloc[i] > 0:
            return True
    return False

# ==============================================================================
# 股票分析邏輯
# ==============================================================================
def analyze_stock(code: str, df_daily: pd.DataFrame) -> Dict[str, Any]:
    name = DYNAMIC_STOCK_NAMES.get(code, "未知")
    latest_price = round(df_daily["close"].iloc[-1], 2) if not df_daily.empty else 0.0

    result = {
        "code": code,
        "name": name,
        "price": latest_price,
        "strategy_1": False,
        "strategy_2": False
    }

    if df_daily.empty or len(df_daily) < 35:
        return result

    # 檢查 20 日均量門檻
    avg_vol_20 = df_daily["volume"].iloc[-20:].mean()
    if avg_vol_20 < MIN_AVG_VOLUME_20:
        return result

    # --- 第一策略：日 K 邏輯 ---
    macd_d, _, _ = calculate_macd(df_daily["close"])
    k_d, _ = calculate_kd(df_daily)
    
    if (macd_d.iloc[-1] > 0) and (k_d.iloc[-1] > DAILY_KD_THRESHOLD):
        result["strategy_1"] = True

    # --- 第二策略：週 K 邏輯 ---
    df_weekly = resample_klines(df_daily, "W-FRI")
    if len(df_weekly) >= 35:
        macd_w, _, _ = calculate_macd(df_weekly["close"])
        k_w, _ = calculate_kd(df_weekly)
        
        cond_weekly = (
            macd_w.iloc[-1] > 0 and
            k_w.iloc[-1] > WEEKLY_KD_THRESHOLD and
            check_zero_cross(macd_w, WEEKLY_ZERO_CROSS_LOOKBACK)
        )
        if cond_weekly:
            result["strategy_2"] = True

    return result

# ==============================================================================
# 主程式
# ==============================================================================
def main():
    print(f"🚀 開始執行 台股策略選股 {VERSION}")
    
    df_tickers = fetch_all_taiwan_market_tickers()
    if df_tickers.empty:
        print("❌ 無法取得標的，程式結束。")
        return

    codes = df_tickers["code"].tolist()
    for _, row in df_tickers.iterrows():
        DYNAMIC_STOCK_NAMES[row["code"]] = row.get("name", "")

    yf_tickers = [get_ticker_code(c) for c in codes]
    
    print(f"📥 開始下載 {len(yf_tickers)} 檔股票之日 K 資料...")
    all_results = []
    
    for i in range(0, len(yf_tickers), DAILY_CHUNK_SIZE):
        chunk = yf_tickers[i:i + DAILY_CHUNK_SIZE]
        raw_d = safe_download_yf(chunk, period="1y", interval="1d")
        
        for t_code in chunk:
            code = t_code.replace(".TW", "").replace(".TWO", "")
            df_d = normalize_dataframe(raw_d, t_code)
            if df_d.empty:
                continue
            
            res = analyze_stock(code, df_d)
            if res["strategy_1"] or res["strategy_2"]:
                all_results.append(res)

    # ==========================================================================
    # 輸出最終格式
    # ==========================================================================
    df_res = pd.DataFrame(all_results)
    
    print("\n==================================================================")
    print("📊 策略選股結果")
    print("==================================================================")

    # 1. 輸出第一策略
    print(f"\n📌 {STRATEGY_1_NAME}")
    print("-" * 50)
    if not df_res.empty and "strategy_1" in df_res.columns:
        s1_stocks = df_res[df_res["strategy_1"] == True]
        if not s1_stocks.empty:
            for _, row in s1_stocks.iterrows():
                print(f"{row['code']} {row['name']} ${row['price']}")
        else:
            print("無符合標的")
    else:
        print("無符合標的")

    # 2. 輸出第二策略
    print(f"\n📌 {STRATEGY_2_NAME}")
    print("-" * 50)
    if not df_res.empty and "strategy_2" in df_res.columns:
        s2_stocks = df_res[df_res["strategy_2"] == True]
        if not s2_stocks.empty:
            for _, row in s2_stocks.iterrows():
                print(f"{row['code']} {row['name']} ${row['price']}")
        else:
            print("無符合標的")
    else:
        print("無符合標的")

if __name__ == "__main__":
    main()
