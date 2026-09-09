# scanner.py
# ==============================================================================
# 台股 6 大策略選股 Pro v2.1 MTF
#
# 策略：
# S1：30K MACD 綠柱縮小 + KD > 50
# S2：60K MACD 綠柱縮小 + KD > 50
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

TWSE_API_URL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"

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

# 30K / 60K KD 門檻
INTRADAY_KD_THRESHOLD = 50

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
# TWSE 股票清單取得
# ==============================================================================
def fetch_all_taiwan_market_tickers() -> pd.DataFrame:
    # 嘗試 API 1: TWSE OpenAPI STOCK_DAY_ALL
    url_1 = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
    # 嘗試 API 2: TWSE 備用 OpenAPI (BWIBBU_d 包含所有上市個股代號)
    url_2 = "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_d"
    
    for url in [url_1, url_2]:
        try:
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data and isinstance(data, list):
                    df = pd.DataFrame(data)
                    
                    # 相容不同的 API 欄位名稱
                    code_col = next((c for c in ['證券代號', 'Code', 'code'] if c in df.columns), None)
                    name_col = next((c for c in ['證券名稱', 'Name', 'name'] if c in df.columns), None)
                    
                    if code_col:
                        df = df.rename(columns={code_col: "code"})
                        if name_col:
                            df = df.rename(columns={name_col: "name"})
                        else:
                            df["name"] = ""
                            
                        df["code"] = df["code"].astype(str).str.strip()
                        # 只篩選 4 碼純數字的普通股
                        df = df[df["code"].str.match(r"^\d{4}$", na=False)].copy()
                        
                        if not df.empty:
                            print(f"✅ 成功從 {url.split('/')[-1]} 取得 {len(df)} 檔股票資訊")
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
    """將日 K 線資料重採樣為 週K ('W') 或 月K ('ME' 或 'M')"""
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
# KD 狀態判斷
# ==============================================================================
def get_kd_state(df: pd.DataFrame) -> Dict[str, Any]:
    result = {"k": np.nan, "d": np.nan, "golden_cross": False}
    if df.empty or len(df) < 10:
        return result

    k, d = calculate_kd(df)
    if len(k) < 2:
        return result

    current_k = safe_float(k.iloc[-1])
    current_d = safe_float(d.iloc[-1])
    previous_k = safe_float(k.iloc[-2])
    previous_d = safe_float(d.iloc[-2])

    golden_cross = (
        not np.isnan(previous_k)
        and not np.isnan(previous_d)
        and not np.isnan(current_k)
        and not np.isnan(current_d)
        and previous_k <= previous_d
        and current_k > current_d
    )

    return {"k": current_k, "d": current_d, "golden_cross": golden_cross}

# ==============================================================================
# MACD 狀態判斷
# ==============================================================================
def get_macd_state(df: pd.DataFrame) -> Dict[str, Any]:
    result = {
        "macd": np.nan,
        "signal": np.nan,
        "hist": np.nan,
        "hist_prev": np.nan,
        "hist_prev2": np.nan,
    }
    if df.empty or len(df) < 30:
        return result

    macd, signal, hist = calculate_macd(df["close"])

    return {
        "macd": safe_float(macd.iloc[-1]),
        "signal": safe_float(signal.iloc[-1]),
        "hist": safe_float(hist.iloc[-1]),
        "hist_prev": safe_float(hist.iloc[-2]),
        "hist_prev2": safe_float(hist.iloc[-3]),
    }

# ==============================================================================
# MACD 最近突破零軸判斷
# ==============================================================================
def get_recent_zero_cross_info(
    df: pd.DataFrame, lookback: int
) -> Dict[str, Any]:
    result = {"crossed": False, "bars_ago": None, "current_above_zero": False}
    if df.empty or len(df) < 30:
        return result

    macd, _, _ = calculate_macd(df["close"])
    current_macd = safe_float(macd.iloc[-1])
    result["current_above_zero"] = (
        not np.isnan(current_macd) and current_macd > 0
    )

    start = max(1, len(macd) - lookback)
    for i in range(len(macd) - 1, start - 1, -1):
        current = safe_float(macd.iloc[i])
        previous = safe_float(macd.iloc[i - 1])

        if (
            not np.isnan(current)
            and not np.isnan(previous)
            and previous <= 0
            and current > 0
        ):
            result["crossed"] = True
            result["bars_ago"] = len(macd) - 1 - i
            break

    return result

# ==============================================================================
# MACD 綠柱縮小 + KD 檢查
# ==============================================================================
def check_macd_negative_reducing_kd(
    df: pd.DataFrame, kd_threshold: float
) -> Dict[str, Any]:
    result = {
        "pass": False,
        "hist": np.nan,
        "hist_prev": np.nan,
        "hist_prev2": np.nan,
        "k": np.nan,
        "d": np.nan,
        "golden_cross": False,
        "green_shrinking": False,
    }

    if df.empty or len(df) < 30:
        return result

    macd_state = get_macd_state(df)
    kd_state = get_kd_state(df)

    hist = macd_state["hist"]
    hist_prev = macd_state["hist_prev"]
    hist_prev2 = macd_state["hist_prev2"]

    result["hist"] = hist
    result["hist_prev"] = hist_prev
    result["hist_prev2"] = hist_prev2
    result["k"] = kd_state["k"]
    result["d"] = kd_state["d"]
    result["golden_cross"] = kd_state["golden_cross"]

    green_shrinking = (
        not np.isnan(hist)
        and not np.isnan(hist_prev)
        and hist < 0
        and hist > hist_prev
    )
    result["green_shrinking"] = green_shrinking

    kd_ok = (
        not np.isnan(kd_state["k"])
        and not np.isnan(kd_state["d"])
        and kd_state["k"] > kd_threshold
        and kd_state["d"] > kd_threshold
    )

    result["pass"] = green_shrinking and kd_ok
    return result

# ==============================================================================
# MACD > 0 + KD 檢查
# ==============================================================================
def check_macd_above_zero_kd(
    df: pd.DataFrame, kd_threshold: float
) -> Dict[str, Any]:
    result = {
        "pass": False,
        "macd": np.nan,
        "signal": np.nan,
        "k": np.nan,
        "d": np.nan,
        "golden_cross": False,
    }

    if df.empty or len(df) < 30:
        return result

    macd_state = get_macd_state(df)
    kd_state = get_kd_state(df)

    result["macd"] = macd_state["macd"]
    result["signal"] = macd_state["signal"]
    result["k"] = kd_state["k"]
    result["d"] = kd_state["d"]
    result["golden_cross"] = kd_state["golden_cross"]

    macd_ok = not np.isnan(macd_state["macd"]) and macd_state["macd"] > 0
    kd_ok = (
        not np.isnan(kd_state["k"])
        and not np.isnan(kd_state["d"])
        and kd_state["k"] > kd_threshold
        and kd_state["d"] > kd_threshold
    )

    result["pass"] = macd_ok and kd_ok
    return result

# ==============================================================================
# 60K MACD 觸發檢查
# ==============================================================================
def check_60m_trigger(df: pd.DataFrame) -> Dict[str, Any]:
    result = {
        "green_shrinking": False,
        "green_to_red": False,
        "pass": False,
        "hist": np.nan,
        "hist_prev": np.nan,
        "hist_prev2": np.nan,
    }

    if df.empty or len(df) < 30:
        return result

    _, _, hist = calculate_macd(df["close"])

    current = safe_float(hist.iloc[-1])
    previous = safe_float(hist.iloc[-2])
    previous2 = safe_float(hist.iloc[-3])

    result["hist"] = current
    result["hist_prev"] = previous
    result["hist_prev2"] = previous2

    green_shrinking = (
        not np.isnan(current)
        and not np.isnan(previous)
        and not np.isnan(previous2)
        and current < 0
        and current > previous
        and previous > previous2
    )

    green_to_red = (
        not np.isnan(current)
        and not np.isnan(previous)
        and previous < 0
        and current >= 0
    )

    result["green_shrinking"] = green_shrinking
    result["green_to_red"] = green_to_red
    result["pass"] = green_shrinking or green_to_red
    return result

# ==============================================================================
# 日K強度計算
# ==============================================================================
def calculate_daily_strength(df: pd.DataFrame) -> float:
    if df.empty or len(df) < 20:
        return 0.0

    close = df["close"]
    ma20 = close.rolling(20).mean().iloc[-1]
    current = close.iloc[-1]

    if np.isnan(ma20) or ma20 == 0:
        return 0.0

    return float(((current / ma20) - 1) * 100)

# ==============================================================================
# 單股票完整掃描
# ==============================================================================
def scan_stock(
    code: str,
    name: str,
    daily_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    m30_df: pd.DataFrame,
    m60_df: pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    try:
        # S1: 30K
        s1_data = check_macd_negative_reducing_kd(
            m30_df, INTRADAY_KD_THRESHOLD
        )
        s1 = s1_data["pass"]

        # S2: 60K
        s2_data = check_macd_negative_reducing_kd(
            m60_df, INTRADAY_KD_THRESHOLD
        )
        s2 = s2_data["pass"]

        # 60K 觸發
        m60_trigger = check_60m_trigger(m60_df)

        # S3: 日K
        s3_data = check_macd_above_zero_kd(daily_df, DAILY_KD_THRESHOLD)
        s3 = s3_data["pass"]

        # S4: 週K
        weekly_zero = get_recent_zero_cross_info(
            weekly_df, WEEKLY_ZERO_CROSS_LOOKBACK
        )
        weekly_kd = get_kd_state(weekly_df)
        s4 = (
            weekly_zero["crossed"]
            and weekly_zero["current_above_zero"]
            and not np.isnan(weekly_kd["k"])
            and not np.isnan(weekly_kd["d"])
            and weekly_kd["k"] > WEEKLY_KD_THRESHOLD
            and weekly_kd["d"] > WEEKLY_KD_THRESHOLD
        )

        # S5: 月K
        monthly_zero = get_recent_zero_cross_info(
            monthly_df, MONTHLY_ZERO_CROSS_LOOKBACK
        )
        monthly_kd = get_kd_state(monthly_df)
        s5 = (
            monthly_zero["crossed"]
            and monthly_zero["current_above_zero"]
            and not np.isnan(monthly_kd["k"])
            and not np.isnan(monthly_kd["d"])
            and monthly_kd["k"] > MONTHLY_KD_THRESHOLD
            and monthly_kd["d"] > MONTHLY_KD_THRESHOLD
        )

        # S6: 多週期共振
        s6 = s3 and s4 and s5

        latest_price = np.nan
        if not daily_df.empty:
            latest_price = safe_float(daily_df["close"].iloc[-1])

        has_signal = (
            s1 or s2 or s3 or s4 or s5 or s6 or m60_trigger["pass"]
        )
        if not has_signal:
            return None

        return {
            "code": code,
            "name": name,
            "label": get_stock_label(code, name),
            "price": latest_price,
            "s1": s1,
            "s2": s2,
            "s3": s3,
            "s4": s4,
            "s5": s5,
            "s6": s6,
            "m30": s1_data,
            "m60": s2_data,
            "m60_trigger": m60_trigger,
            "daily": s3_data,
            "daily_strength": calculate_daily_strength(daily_df),
            "weekly": {"zero_cross": weekly_zero, "kd": weekly_kd},
            "monthly": {"zero_cross": monthly_zero, "kd": monthly_kd},
        }

    except Exception as e:
        print(f"⚠️ {code} 掃描錯誤：{e}")
        return None

# ==============================================================================
# 策略評分機制
# ==============================================================================
def calculate_strategy_score(item: Dict[str, Any]) -> int:
    score = 0
    if item["s1"]:
        score += 10
    if item["s2"]:
        score += 10
    if item["s3"]:
        score += 15
    if item["s4"]:
        score += 15
    if item["s5"]:
        score += 15
    if item["s6"]:
        score += 20

    if item["weekly"]["zero_cross"]["crossed"]:
        score += 5
    if item["monthly"]["zero_cross"]["crossed"]:
        score += 5

    if item["daily"]["golden_cross"]:
        score += 3
    if item["weekly"]["kd"]["golden_cross"]:
        score += 3
    if item["monthly"]["kd"]["golden_cross"]:
        score += 3

    trigger = item["m60_trigger"]
    if trigger["green_shrinking"]:
        score += 5
    if trigger["green_to_red"]:
        score += 10

    if item["s6"] and trigger["green_shrinking"]:
        score += 10
    if item["s6"] and trigger["green_to_red"]:
        score += 15

    return min(score, 100)

# ==============================================================================
# 評級分類
# ==============================================================================
def get_grade(score: int) -> str:
    if score >= 80:
        return "S"
    if score >= 65:
        return "A"
    if score >= 50:
        return "B"
    if score >= 35:
        return "C"
    return "D"

# ==============================================================================
# Telegram 單列格式化
# ==============================================================================
def format_stock_line(item: Dict[str, Any]) -> str:
    label = escape_html(item["label"])
    price = item.get("price", np.nan)
    price_text = "-" if np.isnan(price) else f"{price:.2f}"

    score = item.get("score", 0)
    grade = item.get("grade", "-")

    flags = []
    if item["s1"]:
        flags.append("30K")
    if item["s2"]:
        flags.append("60K")
    if item["s3"]:
        flags.append("日")
    if item["s4"]:
        flags.append("週")
    if item["s5"]:
        flags.append("月")

    trigger = item["m60_trigger"]
    if trigger["green_to_red"]:
        flags.append("60K轉紅")
    elif trigger["green_shrinking"]:
        flags.append("60K縮柱")

    strategy_text = ",".join(flags) if flags else "-"

    return f"• <b>{label}</b> ｜{price_text} ｜{grade} {score}分 ｜{strategy_text}"

# ==============================================================================
# Telegram 推播訊息發送
# ==============================================================================
def send_telegram_message(message: str) -> bool:
    token = os.getenv("TG_BOT_TOKEN")
    chat_id = os.getenv("TG_CHAT_ID")

    if not token or not chat_id:
        print("⚠️ 未設定 TG_BOT_TOKEN / TG_CHAT_ID")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        response = requests.post(url, json=payload, timeout=20)
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"❌ Telegram發送失敗：{e}")
        return False

# ==============================================================================
# 彙整 Telegram 報告內容
# ==============================================================================
def build_telegram_report(results: List[Dict[str, Any]]) -> str:
    if not results:
        return (
            f"🇹🇼 <b>台股選股 {VERSION}</b>\n\n"
            "本次沒有符合條件的股票。"
        )

    for item in results:
        item["score"] = calculate_strategy_score(item)
        item["grade"] = get_grade(item["score"])

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    s6_red = [
        x for x in results if x["s6"] and x["m60_trigger"]["green_to_red"]
    ]
    s6_shrink = [
        x for x in results if x["s6"] and x["m60_trigger"]["green_shrinking"]
    ]
    s6_all = [x for x in results if x["s6"]]
    s3_list = [x for x in results if x["s3"]]
    s4_list = [x for x in results if x["s4"]]
    s5_list = [x for x in results if x["s5"]]
    m60_shrink = [x for x in results if x["m60_trigger"]["green_shrinking"]]
    m60_red = [x for x in results if x["m60_trigger"]["green_to_red"]]

    lines = []
    lines.append(f"🇹🇼 <b>台股選股 {VERSION}</b>")
    lines.append("━━━━━━━━━━━━━━━━")
    lines.append(f"符合股票：<b>{len(results)}</b> 檔\n")
    lines.append("🎯 <b>核心邏輯</b>")
    lines.append("月K突破0 → 週K突破0 → 日K多方 → 60K綠柱縮小→紅柱\n")

    if s6_red:
        lines.append("🔥 <b>S6 + 60K 綠柱→紅柱</b>")
        for item in s6_red[:10]:
            lines.append(format_stock_line(item))
        lines.append("")

    if s6_shrink:
        lines.append("🚀 <b>S6 + 60K 綠柱縮小</b>")
        for item in s6_shrink[:10]:
            lines.append(format_stock_line(item))
        lines.append("")

    if s6_all:
        lines.append("💎 <b>S6 多週期共振</b>")
        for item in s6_all[:10]:
            lines.append(format_stock_line(item))
        lines.append("")

    if s3_list:
        lines.append("📈 <b>S3 日K多方</b>")
        for item in s3_list[:10]:
            lines.append(format_stock_line(item))
        lines.append("")

    if s4_list:
        lines.append("📊 <b>S4 週K突破</b>")
        for item in s4_list[:10]:
            lines.append(format_stock_line(item))
        lines.append("")

    if s5_list:
        lines.append("🗓 <b>S5 月K突破</b>")
        for item in s5_list[:10]:
            lines.append(format_stock_line(item))
        lines.append("")

    if m60_shrink:
        lines.append("🟢 <b>60K MACD 綠柱縮小</b>")
        for item in m60_shrink[:10]:
            lines.append(format_stock_line(item))
        lines.append("")

    if m60_red:
        lines.append("🔴 <b>60K MACD 綠柱→紅柱</b>")
        for item in m60_red[:10]:
            lines.append(format_stock_line(item))
        lines.append("")

    lines.append("🏆 <b>Top 20</b>")
    for i, item in enumerate(results[:20], 1):
        lines.append(
            f"{i}. {escape_html(item['label'])} ｜{item['grade']} {item['score']}分"
        )

    lines.append("\n━━━━━━━━━━━━━━━━")
    lines.append("⚠️ 僅供技術分析參考，不構成投資建議")

    return "\n".join(lines)

# ==============================================================================
# 主程式 (已完整補齊與修復)
# ==============================================================================
def main():
    print("=" * 70)
    print(f"🇹🇼 台股 6 大策略選股 {VERSION}")
    print("=" * 70)
    print("📌 S1：30K MACD 綠柱縮小 + KD > 50")
    print("📌 S2：60K MACD 綠柱縮小 + KD > 50")
    print("📌 S3：日K MACD > 0 + KD > 20")
    print("📌 S4：週K突破0 + KD > 50")
    print("📌 S5：月K突破0 + KD > 50")
    print("📌 S6：日K + 週K + 月K")
    print("=" * 70)

    # 1. 取得全市場股票代碼與名稱
    market_df = fetch_all_taiwan_market_tickers()
    if market_df.empty:
        print("❌ 無法取得 TWSE 股票清單")
        return

    print(f"📊 市場股票總數：{len(market_df)}")

    for _, row in market_df.iterrows():
        code = str(row["code"])
        name = str(row.get("name", ""))
        DYNAMIC_STOCK_NAMES[code] = name

    codes = market_df["code"].astype(str).tolist()

    # 2. 第一階段：日 K 線次選過濾 (過濾流動性與基礎指標)
    daily_candidates = {}
    total = len(codes)
    print("\n🔎 第一階段：進行日 K 線批次下載與篩選...")

    for start in range(0, total, DAILY_CHUNK_SIZE):
        chunk_codes = codes[start : start + DAILY_CHUNK_SIZE]
        tickers = [get_ticker_code(c) for c in chunk_codes]

        print(
            f"📥 下載日 K 資料：{start + 1}-{min(start + DAILY_CHUNK_SIZE, total)}/{total}"
        )

        data = safe_download_yf(tickers, period="2y", interval="1d")
        if data.empty:
            continue

        for code, ticker in zip(chunk_codes, tickers):
            df = normalize_dataframe(data, ticker)
            if df.empty or len(df) < 60:
                continue

            # 成交量門檻過濾 (以 20 日平均成交股數計算)
            avg_volume = df["volume"].tail(20).mean()
            if np.isnan(avg_volume) or avg_volume < MIN_AVG_VOLUME_20:
                continue

            # 保存通過日 K 基礎門檻的股票
            daily_candidates[code] = df

    print(f"✅ 日 K 第一階段初選完成，符合流動性門檻股票數：{len(daily_candidates)} 檔")

    if not daily_candidates:
        print("❌ 沒有符合基礎門檻的股票，程式結束。")
        return

    # 限縮盤中分時掃描數量，避免被 Yahoo API 限流
    candidate_codes = list(daily_candidates.keys())[:INTRADAY_SCAN_LIMIT]

    # 3. 第二階段：下載 30K 及 60K 分時資料並完成全週期掃描
    print(f"\n🔎 第二階段：下載 {len(candidate_codes)} 檔股票之 30K/60K 資料並進行多週期對比...")

    final_results = []
    intraday_total = len(candidate_codes)

    for start in range(0, intraday_total, INTRADAY_CHUNK_SIZE):
        chunk_codes = candidate_codes[start : start + INTRADAY_CHUNK_SIZE]
        tickers = [get_ticker_code(c) for c in chunk_codes]

        print(
            f"📥 下載分時 K 線：{start + 1}-{min(start + INTRADAY_CHUNK_SIZE, intraday_total)}/{intraday_total}"
        )

        # 批次下載 30m 與 60m
        data_30m = safe_download_yf(tickers, period="1mo", interval="30m")
        data_60m = safe_download_yf(tickers, period="1mo", interval="60m")

        for code, ticker in zip(chunk_codes, tickers):
            daily_df = daily_candidates[code]
            m30_df = normalize_dataframe(data_30m, ticker)
            m60_df = normalize_dataframe(data_60m, ticker)

            # 將日 K 重採樣出 週 K 與 月 K (節省 API 請求次數)
            weekly_df = resample_klines(daily_df, "W")
            monthly_df = resample_klines(daily_df, "ME")

            name = DYNAMIC_STOCK_NAMES.get(code, "")
            res = scan_stock(
                code=code,
                name=name,
                daily_df=daily_df,
                weekly_df=weekly_df,
                monthly_df=monthly_df,
                m30_df=m30_df,
                m60_df=m60_df,
            )

            if res:
                final_results.append(res)

    print(f"\n🎯 掃描完畢，最終符合策略總檔數：{len(final_results)} 檔")

    # 4. 建立 Telegram 報告並發送
    report_text = build_telegram_report(final_results)
    print("\n" + "=" * 30 + " 預覽報告 " + "=" * 30)
    print(report_text)
    print("=" * 68)

    sent = send_telegram_message(report_text)
    if sent:
        print("🚀 Telegram 報告發送成功！")
    else:
        print("⚠️ Telegram 未能成功發送（請檢查環境變數 TG_BOT_TOKEN / TG_CHAT_ID）。")

if __name__ == "__main__":
    main()
