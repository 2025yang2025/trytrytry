# scanner.py
# ==============================================================================
# 台股 策略選股 Pro v3.0 (分K/日K/週K/月K 多週期)
#
# 策略一：30分K MACD 負值減少 + KD > 50
# 策略二：60分K MACD 負值減少 + KD > 50
# 策略三：日K MACD > 0 + KD > 30
# 策略四：週K MACD > 0 + KD > 50
# 策略五：月K MACD > 0 + KD > 50
#
# 顯示格式：代號 中文名稱 最新價格
# ==============================================================================

import time
import warnings
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# ==============================================================================
# 基本設定與 API
# ==============================================================================
VERSION = "Pro v3.0 MTF"

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

MIN_AVG_VOLUME_20 = 1_000_000  # 20日平均成交量門檻 (股)
CHUNK_SIZE = 100

DYNAMIC_STOCK_NAMES = {}

# ==============================================================================
# 工具函式
# ==============================================================================
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

def safe_download_yf(tickers: List[str], period: str, interval: str, retries: int = 2) -> pd.DataFrame:
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
            if data is not None and not data.empty:
                return data
        except Exception:
            time.sleep(1)

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
    except Exception:
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
# 技術指標計算與選股邏輯
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

def is_macd_hist_negative_reducing(hist: pd.Series) -> bool:
    """判斷 MACD 柱狀體是否為負值且正在減少 (負值收斂)"""
    if len(hist) < 2:
        return False
    # 最新一根柱狀體為負，且數值大於前一根 (例：-0.5 > -1.2)
    return (hist.iloc[-1] < 0) and (hist.iloc[-1] > hist.iloc[-2])

# ==============================================================================
# 策略邏輯檢測
# ==============================================================================
def check_strategy_1(df_30m: pd.DataFrame) -> bool:
    """【策略一】30分K MACD負值減少 + KD > 50"""
    if df_30m.empty or len(df_30m) < 35:
        return False
    _, _, hist = calculate_macd(df_30m["close"])
    k, _ = calculate_kd(df_30m)
    return is_macd_hist_negative_reducing(hist) and (k.iloc[-1] > 50)

def check_strategy_2(df_60m: pd.DataFrame) -> bool:
    """【策略二】60分K MACD負值減少 + KD > 50"""
    if df_60m.empty or len(df_60m) < 35:
        return False
    _, _, hist = calculate_macd(df_60m["close"])
    k, _ = calculate_kd(df_60m)
    return is_macd_hist_negative_reducing(hist) and (k.iloc[-1] > 50)

def check_strategy_3(df_daily: pd.DataFrame) -> bool:
    """【策略三】日K MACD > 0 + KD > 30"""
    if df_daily.empty or len(df_daily) < 35:
        return False
    macd, _, _ = calculate_macd(df_daily["close"])
    k, _ = calculate_kd(df_daily)
    return (macd.iloc[-1] > 0) and (k.iloc[-1] > 30)

def check_strategy_4(df_weekly: pd.DataFrame) -> bool:
    """【策略四】週K MACD > 0 + KD > 50"""
    if df_weekly.empty or len(df_weekly) < 35:
        return False
    macd, _, _ = calculate_macd(df_weekly["close"])
    k, _ = calculate_kd(df_weekly)
    return (macd.iloc[-1] > 0) and (k.iloc[-1] > 50)

def check_strategy_5(df_monthly: pd.DataFrame) -> bool:
    """【策略五】月K MACD > 0 + KD > 50"""
    if df_monthly.empty or len(df_monthly) < 35:
        return False
    macd, _, _ = calculate_macd(df_monthly["close"])
    k, _ = calculate_kd(df_monthly)
    return (macd.iloc[-1] > 0) and (k.iloc[-1] > 50)

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
    all_results = []

    print(f"📥 開始掃描 {len(yf_tickers)} 檔股票資料...")

    for i in range(0, len(yf_tickers), CHUNK_SIZE):
        chunk = yf_tickers[i:i + CHUNK_SIZE]
        
        # 1. 下載日 K 資料 (計算策略 3, 4, 5)
        raw_daily = safe_download_yf(chunk, period="2y", interval="1d")
        # 2. 下載分 K 資料 (計算策略 1, 2)
        raw_30m = safe_download_yf(chunk, period="1mo", interval="30m")
        raw_60m = safe_download_yf(chunk, period="1mo", interval="60m")

        for t_code in chunk:
            code = t_code.replace(".TW", "").replace(".TWO", "")
            df_d = normalize_dataframe(raw_daily, t_code)
            
            if df_d.empty or len(df_d) < 35:
                continue

            # 檢查 20 日均量門檻
            avg_vol_20 = df_d["volume"].iloc[-20:].mean()
            if avg_vol_20 < MIN_AVG_VOLUME_20:
                continue

            name = DYNAMIC_STOCK_NAMES.get(code, "未知")
            latest_price = round(df_d["close"].iloc[-1], 2)

            # 準備各週期 K 線
            df_w = resample_klines(df_d, "W-FRI")
            df_m = resample_klines(df_d, "ME")
            df_30m = normalize_dataframe(raw_30m, t_code)
            df_60m = normalize_dataframe(raw_60m, t_code)

            # 驗證策略
            s1 = check_strategy_1(df_30m)
            s2 = check_strategy_2(df_60m)
            s3 = check_strategy_3(df_d)
            s4 = check_strategy_4(df_w)
            s5 = check_strategy_5(df_m)

            if any([s1, s2, s3, s4, s5]):
                all_results.append({
                    "code": code,
                    "name": name,
                    "price": latest_price,
                    "s1": s1,
                    "s2": s2,
                    "s3": s3,
                    "s4": s4,
                    "s5": s5,
                })

    # ==========================================================================
    # 輸出最終格式
    # ==========================================================================
    df_res = pd.DataFrame(all_results)
    
    print("\n==================================================================")
    print("📊 策略選股結果")
    print("==================================================================")

    strategies = [
        ("s1", "【策略一】30分K MACD負值減少 + KD大於50"),
        ("s2", "【策略二】60分K MACD負值減少 + KD大於50"),
        ("s3", "【策略三】日K MACD大於0 + KD大於30"),
        ("s4", "【策略四】週K MACD大於0 + KD大於50"),
        ("s5", "【策略五】月K MACD大於0 + KD大於50"),
    ]

    for key, title in strategies:
        print(f"\n📌 {title}")
        print("-" * 50)
        if not df_res.empty and key in df_res.columns:
            matched = df_res[df_res[key] == True]
            if not matched.empty:
                for _, row in matched.iterrows():
                    print(f"{row['code']} {row['name']} ${row['price']}")
            else:
                print("無符合標的")
        else:
            print("無符合標的")

if __name__ == "__main__":
    main()
