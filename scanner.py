```python
# scanner.py
# ==============================================================================
# 🇹🇼 台股 6 大策略選股 Pro v2.1 MTF
#
# 核心邏輯：
#
# 1. 30K：
#    MACD 綠柱縮小 + KD > 50
#
# 2. 60K：
#    MACD 綠柱縮小 + KD > 50
#
# 3. 日K：
#    MACD > 0 + KD > 20
#
# 4. 週K：
#    MACD 最近突破 0 軸 + MACD > 0 + KD > 50
#
# 5. 月K：
#    MACD 最近突破 0 軸 + MACD > 0 + KD > 50
#
# 6. 多週期共振：
#    日K + 週K + 月K 同時成立
#
# 額外核心觸發：
#    60K MACD 綠柱縮小 → 紅柱
#
# ==============================================================================
# 安裝：
#
# pip install pandas yfinance requests
#
# Telegram：
# TG_BOT_TOKEN
# TG_CHAT_ID
#
# ==============================================================================

import os
import time
import html
import requests
import warnings
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import yfinance as yf


warnings.filterwarnings("ignore")


# ==============================================================================
# 設定
# ==============================================================================

VERSION = "Pro v2.1 MTF"

TWSE_API_URL = (
    "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
)

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
    )
}


# ==============================================================================
# 選股參數
# ==============================================================================

# 最低 20 日平均成交量
MIN_AVG_VOLUME_20 = 1_000_000


# 每次下載股票數量
DAILY_CHUNK_SIZE = 150
INTRADAY_CHUNK_SIZE = 50


# ----------------------------------------------------------------------
# KD 門檻
# ----------------------------------------------------------------------

# 日K
DAILY_KD_THRESHOLD = 20

# 週K
WEEKLY_KD_THRESHOLD = 50

# 月K
MONTHLY_KD_THRESHOLD = 50

# ⭐ 30K / 60K
# 使用者要求修改為 KD > 50
INTRADAY_KD_THRESHOLD = 50


# ----------------------------------------------------------------------
# MACD 零軸突破回溯
# ----------------------------------------------------------------------

WEEKLY_ZERO_CROSS_LOOKBACK = 6
MONTHLY_ZERO_CROSS_LOOKBACK = 3


# ----------------------------------------------------------------------
# 60K MACD 觸發回溯
# ----------------------------------------------------------------------

M60_TRIGGER_LOOKBACK = 2


# ----------------------------------------------------------------------
# 動態股票名稱
# ----------------------------------------------------------------------

DYNAMIC_STOCK_NAMES = {}


# ==============================================================================
# 基本工具
# ==============================================================================

def safe_float(value, default=np.nan):
    """
    安全轉換數字
    """
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
# 股票代號
# ==============================================================================

def get_ticker_code(code: str) -> str:
    """
    將台股代號轉成 Yahoo Finance 格式
    """
    code = str(code).strip()

    if code.endswith(".TW") or code.endswith(".TWO"):
        return code

    return f"{code}.TW"


# ==============================================================================
# HTML
# ==============================================================================

def escape_html(text: str) -> str:
    return html.escape(str(text), quote=False)


# ==============================================================================
# 股票名稱
# ==============================================================================

def get_stock_label(code: str, name: Optional[str] = None) -> str:

    if name:
        return f"{code} {name}"

    if code in DYNAMIC_STOCK_NAMES:
        return f"{code} {DYNAMIC_STOCK_NAMES[code]}"

    return code


# ==============================================================================
# 取得台股市場清單
# ==============================================================================

def fetch_all_taiwan_market_tickers() -> pd.DataFrame:

    try:

        response = requests.get(
            TWSE_API_URL,
            headers=REQUEST_HEADERS,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # TWSE API 欄位
        if "證券代號" in df.columns:

            df = df.rename(
                columns={
                    "證券代號": "code",
                    "證券名稱": "name",
                    "成交股數": "volume",
                    "成交金額": "turnover",
                }
            )

        if "code" not in df.columns:
            return pd.DataFrame()

        df["code"] = (
            df["code"]
            .astype(str)
            .str.strip()
        )

        # 只保留 4 碼股票
        df = df[
            df["code"].str.match(r"^\d{4}$", na=False)
        ].copy()

        if "name" in df.columns:

            df["name"] = (
                df["name"]
                .astype(str)
                .str.strip()
            )

        else:
            df["name"] = ""

        return df

    except Exception as e:

        print(
            f"❌ 取得 TWSE 股票清單失敗：{e}"
        )

        return pd.DataFrame()


# ==============================================================================
# Yahoo Finance 下載
# ==============================================================================

def safe_download_yf(
    tickers: List[str],
    period: str,
    interval: str,
    retries: int = 3
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
                threads=True
            )

            if data is None or data.empty:
                raise ValueError("Yahoo Finance 無資料")

            return data

        except Exception as e:

            print(
                f"⚠️ Yahoo 下載失敗 "
                f"({attempt + 1}/{retries})：{e}"
            )

            time.sleep(2)

    return pd.DataFrame()


# ==============================================================================
# DataFrame 正規化
# ==============================================================================

def normalize_dataframe(
    data: pd.DataFrame,
    ticker: str
) -> pd.DataFrame:

    if data is None or data.empty:
        return pd.DataFrame()

    try:

        # MultiIndex
        if isinstance(data.columns, pd.MultiIndex):

            if ticker in data.columns.get_level_values(0):

                df = data[ticker].copy()

            elif ticker in data.columns.get_level_values(1):

                df = data.xs(
                    ticker,
                    axis=1,
                    level=1
                ).copy()

            else:

                # 單股票情況
                df = data.copy()

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(-1)

        else:

            df = data.copy()

        df.columns = [
            str(c).lower()
            for c in df.columns
        ]

        required = [
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]

        for col in required:

            if col not in df.columns:
                return pd.DataFrame()

        df = df[required].copy()

        for col in required:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df.dropna(
            subset=["close"]
        )

        return df

    except Exception as e:

        print(
            f"⚠️ DataFrame 正規化失敗 {ticker}: {e}"
        )

        return pd.DataFrame()


# ==============================================================================
# MACD
# ==============================================================================

def calculate_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
):

    ema_fast = close.ewm(
        span=fast,
        adjust=False
    ).mean()

    ema_slow = close.ewm(
        span=slow,
        adjust=False
    ).mean()

    macd = ema_fast - ema_slow

    signal_line = macd.ewm(
        span=signal,
        adjust=False
    ).mean()

    hist = macd - signal_line

    return macd, signal_line, hist


# ==============================================================================
# KD
# ==============================================================================

def calculate_kd(
    df: pd.DataFrame,
    period: int = 9,
    k_period: int = 3,
    d_period: int = 3
):

    low_min = (
        df["low"]
        .rolling(period)
        .min()
    )

    high_max = (
        df["high"]
        .rolling(period)
        .max()
    )

    denominator = high_max - low_min

    rsv = (
        (df["close"] - low_min)
        / denominator.replace(0, np.nan)
        * 100
    )

    k = (
        rsv
        .ewm(
            alpha=1 / k_period,
            adjust=False
        )
        .mean()
    )

    d = (
        k
        .ewm(
            alpha=1 / d_period,
            adjust=False
        )
        .mean()
    )

    return k, d


# ==============================================================================
# KD 狀態
# ==============================================================================

def get_kd_state(
    df: pd.DataFrame
) -> Dict[str, Any]:

    if df.empty:
        return {
            "k": np.nan,
            "d": np.nan,
            "golden_cross": False
        }

    k, d = calculate_kd(df)

    if len(k) < 2:
        return {
            "k": np.nan,
            "d": np.nan,
            "golden_cross": False
        }

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

    return {
        "k": current_k,
        "d": current_d,
        "golden_cross": golden_cross
    }


# ==============================================================================
# MACD 狀態
# ==============================================================================

def get_macd_state(
    df: pd.DataFrame
) -> Dict[str, Any]:

    if df.empty:
        return {
            "macd": np.nan,
            "signal": np.nan,
            "hist": np.nan,
            "hist_prev": np.nan,
            "hist_prev2": np.nan
        }

    macd, signal, hist = calculate_macd(
        df["close"]
    )

    return {
        "macd": safe_float(macd.iloc[-1]),
        "signal": safe_float(signal.iloc[-1]),
        "hist": safe_float(hist.iloc[-1]),
        "hist_prev": safe_float(
            hist.iloc[-2]
        ) if len(hist) >= 2 else np.nan,
        "hist_prev2": safe_float(
            hist.iloc[-3]
        ) if len(hist) >= 3 else np.nan
    }


# ==============================================================================
# MACD 最近突破零軸
# ==============================================================================

def get_recent_zero_cross_info(
    df: pd.DataFrame,
    lookback: int
) -> Dict[str, Any]:

    result = {
        "crossed": False,
        "bars_ago": None,
        "current_above_zero": False
    }

    if df.empty or len(df) < 3:
        return result

    macd, _, _ = calculate_macd(
        df["close"]
    )

    current_macd = safe_float(
        macd.iloc[-1]
    )

    result["current_above_zero"] = (
        not np.isnan(current_macd)
        and current_macd > 0
    )

    start = max(
        1,
        len(macd) - lookback
    )

    for i in range(
        len(macd) - 1,
        start - 1,
        -1
    ):

        current = safe_float(
            macd.iloc[i]
        )

        previous = safe_float(
            macd.iloc[i - 1]
        )

        if (
            not np.isnan(current)
            and not np.isnan(previous)
            and previous <= 0
            and current > 0
        ):

            result["crossed"] = True
            result["bars_ago"] = (
                len(macd) - 1 - i
            )

            break

    return result


# ==============================================================================
# MACD 綠柱縮小 + KD
# ==============================================================================

def check_macd_negative_reducing_kd(
    df: pd.DataFrame,
    kd_threshold: float
) -> Dict[str, Any]:

    result = {
        "pass": False,
        "hist": np.nan,
        "hist_prev": np.nan,
        "hist_prev2": np.nan,
        "k": np.nan,
        "d": np.nan,
        "golden_cross": False,
        "green_shrinking": False
    }

    if df.empty or len(df) < 30:
        return result

    macd_state = get_macd_state(df)
    kd_state = get_kd_state(df)

    hist = macd_state["hist"]
    hist_prev = macd_state["hist_prev"]
    hist_prev2 = macd_state["hist_prev2"]

    result.update({
        "hist": hist,
        "hist_prev": hist_prev,
        "hist_prev2": hist_prev2,
        "k": kd_state["k"],
        "d": kd_state["d"],
        "golden_cross": kd_state["golden_cross"]
    })

    # ------------------------------------------------------------------
    # MACD 綠柱縮小
    #
    # hist < 0
    # 且目前柱體比前一根接近 0
    # ------------------------------------------------------------------

    green_shrinking = (
        not np.isnan(hist)
        and not np.isnan(hist_prev)
        and hist < 0
        and hist > hist_prev
    )

    result["green_shrinking"] = green_shrinking

    # ------------------------------------------------------------------
    # KD > threshold
    #
    # ⭐ 30K / 60K 現在 threshold = 50
    # ------------------------------------------------------------------

    kd_ok = (
        not np.isnan(kd_state["k"])
        and not np.isnan(kd_state["d"])
        and kd_state["k"] > kd_threshold
        and kd_state["d"] > kd_threshold
    )

    result["pass"] = (
        green_shrinking
        and kd_ok
    )

    return result


# ==============================================================================
# MACD > 0 + KD
# ==============================================================================

def check_macd_above_zero_kd(
    df: pd.DataFrame,
    kd_threshold: float
) -> Dict[str, Any]:

    result = {
        "pass": False,
        "macd": np.nan,
        "signal": np.nan,
        "k": np.nan,
        "d": np.nan,
        "golden_cross": False
    }

    if df.empty or len(df) < 30:
        return result

    macd_state = get_macd_state(df)
    kd_state = get_kd_state(df)

    result.update({
        "macd": macd_state["macd"],
        "signal": macd_state["signal"],
        "k": kd_state["k"],
        "d": kd_state["d"],
        "golden_cross": kd_state["golden_cross"]
    })

    macd_ok = (
        not np.isnan(macd_state["macd"])
        and macd_state["macd"] > 0
    )

    kd_ok = (
        not np.isnan(kd_state["k"])
        and not np.isnan(kd_state["d"])
        and kd_state["k"] > kd_threshold
        and kd_state["d"] > kd_threshold
    )

    result["pass"] = (
        macd_ok
        and kd_ok
    )

    return result


# ==============================================================================
# 60K MACD 觸發
# ==============================================================================
#
# 注意：
#
# 這裡是獨立的「60K 最終觸發」
#
# 不強制 KD > 50。
#
# 原因是使用者原本的核心概念：
#
# 60K MACD 綠柱縮小 → 紅柱
#
# KD > 50 已經放在「策略 2」裡面。
#
# ==============================================================================

def check_60m_trigger(
    df: pd.DataFrame
) -> Dict[str, Any]:

    result = {
        "green_shrinking": False,
        "green_to_red": False,
        "pass": False,
        "hist": np.nan,
        "hist_prev": np.nan,
        "hist_prev2": np.nan
    }

    if df.empty or len(df) < 30:
        return result

    macd, signal, hist = calculate_macd(
        df["close"]
    )

    current = safe_float(
        hist.iloc[-1]
    )

    previous = safe_float(
        hist.iloc[-2]
    )

    previous2 = safe_float(
        hist.iloc[-3]
    )

    result["hist"] = current
    result["hist_prev"] = previous
    result["hist_prev2"] = previous2

    # --------------------------------------------------------------
    # 綠柱縮小
    # --------------------------------------------------------------

    green_shrinking = (
        not np.isnan(current)
        and not np.isnan(previous)
        and not np.isnan(previous2)
        and current < 0
        and current > previous
        and previous > previous2
    )

    # --------------------------------------------------------------
    # 綠柱 → 紅柱
    # --------------------------------------------------------------

    green_to_red = (
        not np.isnan(current)
        and not np.isnan(previous)
        and previous < 0
        and current >= 0
    )

    result["green_shrinking"] = green_shrinking
    result["green_to_red"] = green_to_red

    result["pass"] = (
        green_shrinking
        or green_to_red
    )

    return result


# ==============================================================================
# 日K強度
# ==============================================================================

def calculate_daily_strength(
    df: pd.DataFrame
) -> float:

    if df.empty or len(df) < 20:
        return 0

    close = df["close"]

    ma20 = (
        close
        .rolling(20)
        .mean()
        .iloc[-1]
    )

    current = close.iloc[-1]

    if np.isnan(ma20) or ma20 == 0:
        return 0

    strength = (
        (current / ma20) - 1
    ) * 100

    return float(strength)


# ==============================================================================
# 單股票掃描
# ==============================================================================

def scan_stock(
    code: str,
    name: str,
    daily_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    m30_df: pd.DataFrame,
    m60_df: pd.DataFrame
) -> Optional[Dict[str, Any]]:

    try:

        # ==============================================================
        # Strategy 1
        # 30K MACD 綠柱縮小 + KD > 50
        # ==============================================================

        s1_data = check_macd_negative_reducing_kd(
            m30_df,
            INTRADAY_KD_THRESHOLD
        )

        s1 = s1_data["pass"]


        # ==============================================================
        # Strategy 2
        # 60K MACD 綠柱縮小 + KD > 50
        # ==============================================================

        s2_data = check_macd_negative_reducing_kd(
            m60_df,
            INTRADAY_KD_THRESHOLD
        )

        s2 = s2_data["pass"]


        # ==============================================================
        # 60K 最終觸發
        # ==============================================================

        m60_trigger = check_60m_trigger(
            m60_df
        )


        # ==============================================================
        # Strategy 3
        # 日K MACD > 0 + KD > 20
        # ==============================================================

        s3_data = check_macd_above_zero_kd(
            daily_df,
            DAILY_KD_THRESHOLD
        )

        s3 = s3_data["pass"]


        # ==============================================================
        # Strategy 4
        # 週K：
        #
        # 最近突破 0 軸
        # +
        # 現在 MACD > 0
        # +
        # KD > 50
        # ==============================================================

        weekly_zero = get_recent_zero_cross_info(
            weekly_df,
            WEEKLY_ZERO_CROSS_LOOKBACK
        )

        weekly_kd = get_kd_state(
            weekly_df
        )

        s4 = (
            weekly_zero["crossed"]
            and weekly_zero["current_above_zero"]
            and not np.isnan(weekly_kd["k"])
            and not np.isnan(weekly_kd["d"])
            and weekly_kd["k"] > WEEKLY_KD_THRESHOLD
            and weekly_kd["d"] > WEEKLY_KD_THRESHOLD
        )


        # ==============================================================
        # Strategy 5
        # 月K：
        #
        # 最近突破 0 軸
        # +
        # 現在 MACD > 0
        # +
        # KD > 50
        # ==============================================================

        monthly_zero = get_recent_zero_cross_info(
            monthly_df,
            MONTHLY_ZERO_CROSS_LOOKBACK
        )

        monthly_kd = get_kd_state(
            monthly_df
        )

        s5 = (
            monthly_zero["crossed"]
            and monthly_zero["current_above_zero"]
            and not np.isnan(monthly_kd["k"])
            and not np.isnan(monthly_kd["d"])
            and monthly_kd["k"] > MONTHLY_KD_THRESHOLD
            and monthly_kd["d"] > MONTHLY_KD_THRESHOLD
        )


        # ==============================================================
        # Strategy 6
        #
        # 日K + 週K + 月K
        #
        # 多週期共振
        # ==============================================================

        s6 = (
            s3
            and s4
            and s5
        )


        # ==============================================================
        # 沒有任何條件直接跳過
        # ==============================================================

        if not (
            s1
            or s2
            or s3
            or s4
            or s5
            or s6
            or m60_trigger["pass"]
        ):
            return None


        # ==============================================================
        # 日K強度
        # ==============================================================

        daily_strength = calculate_daily_strength(
            daily_df
        )


        # ==============================================================
        # 最新價格
        # ==============================================================

        latest_price = np.nan

        if not daily_df.empty:
            latest_price = safe_float(
                daily_df["close"].iloc[-1]
            )


        return {

            "code": code,
            "name": name,

            "label": get_stock_label(
                code,
                name
            ),

            "price": latest_price,

            # ----------------------------------------------------------
            # Strategies
            # ----------------------------------------------------------

            "s1": s1,
            "s2": s2,
            "s3": s3,
            "s4": s4,
            "s5": s5,
            "s6": s6,

            # ----------------------------------------------------------
            # 30K
            # ----------------------------------------------------------

            "m30": s1_data,

            # ----------------------------------------------------------
            # 60K
            # ----------------------------------------------------------

            "m60": s2_data,

            "m60_trigger": m60_trigger,

            # ----------------------------------------------------------
            # Daily
            # ----------------------------------------------------------

            "daily": s3_data,

            "daily_strength": daily_strength,

            # ----------------------------------------------------------
            # Weekly
            # ----------------------------------------------------------

            "weekly": {
                "zero_cross": weekly_zero,
                "kd": weekly_kd
            },

            # ----------------------------------------------------------
            # Monthly
            # ----------------------------------------------------------

            "monthly": {
                "zero_cross": monthly_zero,
                "kd": monthly_kd
            }
        }

    except Exception as e:

        print(
            f"⚠️ {code} 掃描錯誤：{e}"
        )

        return None


# ==============================================================================
# 策略評分
# ==============================================================================

def calculate_strategy_score(
    item: Dict[str, Any]
) -> int:

    score = 0

    # --------------------------------------------------------------
    # 基本策略
    # --------------------------------------------------------------

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


    # --------------------------------------------------------------
    # 零軸突破
    # --------------------------------------------------------------

    weekly_zero = (
        item["weekly"]["zero_cross"]
    )

    monthly_zero = (
        item["monthly"]["zero_cross"]
    )

    if weekly_zero["crossed"]:
        score += 5

    if monthly_zero["crossed"]:
        score += 5


    # --------------------------------------------------------------
    # KD 黃金交叉
    # --------------------------------------------------------------

    if item["daily"]["golden_cross"]:
        score += 3

    if item["weekly"]["kd"]["golden_cross"]:
        score += 3

    if item["monthly"]["kd"]["golden_cross"]:
        score += 3


    # --------------------------------------------------------------
    # 60K MACD
    # --------------------------------------------------------------

    trigger = item["m60_trigger"]

    if trigger["green_shrinking"]:
        score += 5

    if trigger["green_to_red"]:
        score += 10


    # --------------------------------------------------------------
    # 多週期共振 + 60K
    # --------------------------------------------------------------

    if (
        item["s6"]
        and trigger["green_shrinking"]
    ):
        score += 10

    if (
        item["s6"]
        and trigger["green_to_red"]
    ):
        score += 15


    return min(
        score,
        100
    )


# ==============================================================================
# 評級
# ==============================================================================

def get_grade(
    score: int
) -> str:

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
# 建立 30K / 60K 掃描池
# ==============================================================================

def build_intraday_scan_pool(
    results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    def rank_key(item):

        strategy_priority = (
            6 if item["s6"]
            else 5 if item["s5"]
            else 4 if item["s4"]
            else 3 if item["s3"]
            else 0
        )

        return (
            strategy_priority,
            item.get(
                "daily_strength",
                0
            )
        )

    return sorted(
        results,
        key=rank_key,
        reverse=True
    )


# ==============================================================================
# Telegram
# ==============================================================================

def send_telegram_message(
    message: str
) -> bool:

    token = os.getenv(
        "TG_BOT_TOKEN"
    )

    chat_id = os.getenv(
        "TG_CHAT_ID"
    )

    if not token or not chat_id:

        print(
            "⚠️ 未設定 TG_BOT_TOKEN / TG_CHAT_ID"
        )

        return False

    url = (
        f"https://api.telegram.org/"
        f"bot{token}/sendMessage"
    )

    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }

    try:

        response = requests.post(
            url,
            json=payload,
            timeout=20
        )

        response.raise_for_status()

        return True

    except Exception as e:

        print(
            f"❌ Telegram 發送失敗：{e}"
        )

        return False


# ==============================================================================
# Telegram 報告
# ==============================================================================

def build_telegram_report(
    results: List[Dict[str, Any]]
) -> str:

    if not results:

        return (
            f"🇹🇼 台股選股 {VERSION}\n\n"
            "本次沒有符合條件的股票。"
        )


    # --------------------------------------------------------------
    # 計算分數
    # --------------------------------------------------------------

    for item in results:

        item["score"] = (
            calculate_strategy_score(
                item
            )
        )

        item["grade"] = (
            get_grade(
                item["score"]
            )
        )


    # --------------------------------------------------------------
    # 排序
    # --------------------------------------------------------------

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )


    # --------------------------------------------------------------
    # 分類
    # --------------------------------------------------------------

    def get_list(
        condition
    ):

        return [
            x for x in results
            if condition(x)
        ]


    s6_red = get_list(
        lambda x:
            x["s6"]
            and x["m60_trigger"]["green_to_red"]
    )

    s6_shrink = get_list(
        lambda x:
            x["s6"]
            and x["m60_trigger"]["green_shrinking"]
    )

    s6_all = get_list(
        lambda x: x["s6"]
    )

    s3_list = get_list(
        lambda x: x["s3"]
    )

    s4_list = get_list(
        lambda x: x["s4"]
    )

    s5_list = get_list(
        lambda x: x["s5"]
    )

    m60_shrink = get_list(
        lambda x:
            x["m60_trigger"]["green_shrinking"]
    )

    m60_red = get_list(
        lambda x:
            x["m60_trigger"]["green_to_red"]
    )


    # --------------------------------------------------------------
    # 報告
    # --------------------------------------------------------------

    lines = []

    lines.append(
        f"🇹🇼 <b>台股選股 {VERSION}</b>"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━"
    )

    lines.append(
        f"符合股票：<b>{len(results)}</b> 檔"
    )

    lines.append("")

    lines.append(
        "🎯 <b>核心選股邏輯</b>"
    )

    lines.append(
        "月K突破0 → 週K突破0 → 日K多方 → "
        "60K綠柱縮小→紅柱"
    )

    lines.append("")

    # --------------------------------------------------------------
    # S6 + 60K 紅
    # --------------------------------------------------------------

    if s6_red:

        lines.append(
            "🔥 <b>S6 + 60K 綠柱→紅柱</b>"
        )

        for item in s6_red[:10]:

            lines.append(
                format_stock_line(item)
            )

        lines.append("")


    # --------------------------------------------------------------
    # S6 + 60K 縮小
    # --------------------------------------------------------------

    if s6_shrink:

        lines.append(
            "🚀 <b>S6 + 60K 綠柱縮小</b>"
        )

        for item in s6_shrink[:10]:

            lines.append(
                format_stock_line(item)
            )

        lines.append("")


    # --------------------------------------------------------------
    # S6
    # --------------------------------------------------------------

    if s6_all:

        lines.append(
            "💎 <b>S6 多週期共振</b>"
        )

        for item in s6_all[:10]:

            lines.append(
                format_stock_line(item)
            )

        lines.append("")


    # --------------------------------------------------------------
    # S3
    # --------------------------------------------------------------

    if s3_list:

        lines.append(
            "📈 <b>S3 日K多方</b>"
        )

        for item in s3_list[:10]:

            lines.append(
                format_stock_line(item)
            )

        lines.append("")


    # --------------------------------------------------------------
    # S4
    # --------------------------------------------------------------

    if s4_list:

        lines.append(
            "📊 <b>S4 週K突破</b>"
        )

        for item in s4_list[:10]:

            lines.append(
                format_stock_line(item)
            )

        lines.append("")


    # --------------------------------------------------------------
    # S5
    # --------------------------------------------------------------

    if s5_list:

        lines.append(
            "🗓 <b>S5 月K突破</b>"
        )

        for item in s5_list[:10]:

            lines.append(
                format_stock_line(item)
            )

        lines.append("")


    # --------------------------------------------------------------
    # 60K 綠柱縮小
    # --------------------------------------------------------------

    if m60_shrink:

        lines.append(
            "🟢 <b>60K MACD 綠柱縮小</b>"
        )

        for item in m60_shrink[:10]:

            lines.append(
                format_stock_line(item)
            )

        lines.append("")


    # --------------------------------------------------------------
    # 60K 綠轉紅
    # --------------------------------------------------------------

    if m60_red:

        lines.append(
            "🔴 <b>60K MACD 綠柱→紅柱</b>"
        )

        for item in m60_red[:10]:

            lines.append(
                format_stock_line(item)
            )

        lines.append("")


    # --------------------------------------------------------------
    # Top 20
    # --------------------------------------------------------------

    lines.append(
        "🏆 <b>Top 20</b>"
    )

    for i, item in enumerate(
        results[:20],
        1
    ):

        lines.append(
            f"{i}. "
            f"{escape_html(item['label'])} "
            f"｜{item['grade']} "
            f"{item['score']}分"
        )

    lines.append("")

    lines.append(
        "━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "⚠️ 僅供技術分析參考，不構成投資建議"
    )

    return "\n".join(lines)


# ==============================================================================
# 股票格式
# ==============================================================================

def format_stock_line(
    item: Dict[str, Any]
) -> str:

    label = escape_html(
        item["label"]
    )

    price = item.get(
        "price",
        np.nan
    )

    if np.isnan(price):

        price_text = "-"

    else:

        price_text = (
            f"{price:.2f}"
        )

    score = item.get(
        "score",
        0
    )

    grade = item.get(
        "grade",
        "-"
    )

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

    strategy_text = (
        ",".join(flags)
        if flags
        else "-"
    )

    return (
        f"• <b>{label}</b> "
        f"｜{price_text} "
        f"｜{grade} {score}分 "
        f"｜{strategy_text}"
    )


# ==============================================================================
# 主程式
# ==============================================================================

def main():

    print(
        "=" * 70
    )

    print(
        f"🇹🇼 台股 6 大策略選股 {VERSION}"
    )

    print(
        "=" * 70
    )

    print(
        "📌 30K KD > 50"
    )

    print(
        "📌 60K KD > 50"
    )

    print(
        "📌 日K KD > 20"
    )

    print(
        "📌 週K KD > 50"
    )

    print(
        "📌 月K KD > 50"
    )

    print(
        "📌 S6 = 日K + 週K + 月K"
    )

    print(
        "=" * 70
    )


    # ==================================================================
    # 取得股票清單
    # ==================================================================

    market_df = (
        fetch_all_taiwan_market_tickers()
    )

    if market_df.empty:

        print(
            "❌ 無法取得股票清單"
        )

        return


    print(
        f"📊 市場股票數：{len(market_df)}"
    )


    # ==================================================================
    # 建立股票名稱
    # ==================================================================

    for _, row in market_df.iterrows():

        code = str(
            row["code"]
        )

        name = str(
            row.get(
                "name",
                ""
            )
        )

        DYNAMIC_STOCK_NAMES[
            code
        ] = name


    codes = (
        market_df["code"]
        .astype(str)
        .tolist()
    )


    # ==================================================================
    # 第一階段：
    # 日K掃描
    # ==================================================================

    daily_results = []

    total = len(codes)

    print(
        "🔎 開始日K掃描..."
    )


    for start in range(
        0,
        total,
        DAILY_CHUNK_SIZE
    ):

        chunk_codes = codes[
            start:
            start + DAILY_CHUNK_SIZE
        ]

        tickers = [
            get_ticker_code(c)
            for c in chunk_codes
        ]

        print(
            f"📥 日K："
            f"{start + 1}-"
            f"{min(start + DAILY_CHUNK_SIZE, total)}"
            f"/{total}"
        )


        data = safe_download_yf(
            tickers,
            period="2y",
            interval="1d"
        )


        for code, ticker in zip(
            chunk_codes,
            tickers
        ):

            df = normalize_dataframe(
                data,
                ticker
            )

            if df.empty:
                continue

            if len(df) < 60:
                continue


            # ----------------------------------------------------------
            # 成交量過濾
            # ----------------------------------------------------------

            avg_volume = (
                df["volume"]
                .tail(20)
                .mean()
            )

            if (
                np.isnan(avg_volume)
                or avg_volume
                < MIN_AVG_VOLUME_20
            ):
                continue


            # ----------------------------------------------------------
            # 日K
            # ----------------------------------------------------------

            daily_check = (
                check_macd_above_zero_kd(
                    df,
                    DAILY_KD_THRESHOLD
                )
            )


            # ----------------------------------------------------------
            # 即使日K沒有 S3，
            # 後續仍可能因為週/月策略進入。
            #
            # 先保留所有高流動性股票。
            # ----------------------------------------------------------

            name = DYNAMIC_STOCK_NAMES.get(
                code,
                ""
            )

            daily_results.append({
                "code": code,
                "name": name,
                "daily_df": df,
                "daily_pass": daily_check["pass"],
                "daily_strength":
                    calculate_daily_strength(df)
            })


    print(
        f"✅ 日K初篩完成："
        f"{len(daily_results)} 檔"
    )


    if not daily_results:

        print(
            "❌ 沒有符合流動性條件的股票"
        )

        return


    # ==================================================================
    # 第二階段：
    # 週K / 月K
    # ==================================================================

    results = []

    print(
        "🔎 開始週K / 月K掃描..."
    )


    for idx, item in enumerate(
        daily_results,
        1
    ):

        code = item["code"]
        name = item["name"]

        ticker = get_ticker_code(
            code
        )

        print(
            f"📊 [{idx}/{len(daily_results)}] "
            f"{code} {name}"
        )


        try:

            data = safe_download_yf(
                [ticker],
                period="10y",
                interval="1wk"
            )

            weekly_df = normalize_dataframe(
                data,
                ticker
            )


            data = safe_download_yf(
                [ticker],
                period="15y",
                interval="1mo"
            )

            monthly_df = normalize_dataframe(
                data,
                ticker
            )


            if weekly_df.empty:
                continue

            if monthly_df.empty:
                continue


            # ----------------------------------------------------------
            # 先使用空的 30K / 60K
            #
            # 第二階段只確認日週月。
            # ----------------------------------------------------------

            result = scan_stock(
                code=code,
                name=name,
                daily_df=item["daily_df"],
                weekly_df=weekly_df,
                monthly_df=monthly_df,
                m30_df=pd.DataFrame(),
                m60_df=pd.DataFrame()
            )


            # ----------------------------------------------------------
            # scan_stock 需要 30K / 60K
            #
            # 因此這裡另外建立基礎候選。
            # ----------------------------------------------------------

            s3 = check_macd_above_zero_kd(
                item["daily_df"],
                DAILY_KD_THRESHOLD
            )

            weekly_zero = (
                get_recent_zero_cross_info(
                    weekly_df,
                    WEEKLY_ZERO_CROSS_LOOKBACK
                )
            )

            weekly_kd = get_kd_state(
                weekly_df
            )

            s4 = (
                weekly_zero["crossed"]
                and weekly_zero["current_above_zero"]
                and weekly_kd["k"] > WEEKLY_KD_THRESHOLD
                and weekly_kd["d"] > WEEKLY_KD_THRESHOLD
            )

            monthly_zero = (
                get_recent_zero_cross_info(
                    monthly_df,
                    MONTHLY_ZERO_CROSS_LOOKBACK
                )
            )

            monthly_kd = get_kd_state(
                monthly_df
            )

            s5 = (
                monthly_zero["crossed"]
                and monthly_zero["current_above_zero"]
                and monthly_kd["k"] > MONTHLY_KD_THRESHOLD
                and monthly_kd["d"] > MONTHLY_KD_THRESHOLD
            )


            # ----------------------------------------------------------
            # 只要 S3/S4/S5 任一成立，
            # 或多週期共振，就進入 30K/60K 深度掃描池
            # ----------------------------------------------------------

            if (
                s3["pass"]
                or s4
                or s5
            ):

                results.append({

                    "code": code,

                    "name": name,

                    "daily_df":
                        item["daily_df"],

                    "weekly_df":
                        weekly_df,

                    "monthly_df":
                        monthly_df,

                    "daily_strength":
                        item["daily_strength"],

                    "s3":
                        s3["pass"],

                    "s4":
                        s4,

                    "s5":
                        s5,

                    "weekly_zero":
                        weekly_zero,

                    "monthly_zero":
                        monthly_zero,

                    "weekly_kd":
                        weekly_kd,

                    "monthly_kd":
                        monthly_kd
                })


        except Exception as e:

            print(
                f"⚠️ {code} 週/月K錯誤：{e}"
            )

            continue


    print(
        f"✅ 週/月K初篩："
        f"{len(results)} 檔"
    )


    # ==================================================================
    # 第三階段：
    # 排序後下載 30K / 60K
    # ==================================================================

    results = sorted(
        results,
        key=lambda x: (
            x["s3"] +
            x["s4"] +
            x["s5"],
            x["daily_strength"]
        ),
        reverse=True
    )


    results = results[
        :INTRADAY_SCAN_LIMIT
    ] if "INTRADAY_SCAN_LIMIT" in globals() else results[:100]


    print(
        f"🚀 進入 30K / 60K 深度掃描："
        f"{len(results)} 檔"
    )


    final_results = []


    for idx, item in enumerate(
        results,
        1
    ):

        code = item["code"]
        name = item["name"]

        ticker = get_ticker_code(
            code
        )

        print(
            f"⏱ [{idx}/{len(results)}] "
            f"{code} {name}"
        )


        try:

            # ----------------------------------------------------------
            # 30分鐘
            # ----------------------------------------------------------

            data30 = safe_download_yf(
                [ticker],
                period="60d",
                interval="30m"
            )

            m30_df = normalize_dataframe(
                data30,
                ticker
            )


            # ----------------------------------------------------------
            # 60分鐘
            # ----------------------------------------------------------

            data60 = safe_download_yf(
                [ticker],
                period="730d",
                interval="60m"
            )

            m60_df = normalize_dataframe(
                data60,
                ticker
            )


            # ----------------------------------------------------------
            # 最終掃描
            # ----------------------------------------------------------

            result = scan_stock(

                code=code,

                name=name,

                daily_df=item[
                    "daily_df"
                ],

                weekly_df=item[
                    "weekly_df"
                ],

                monthly_df=item[
                    "monthly_df"
                ],

                m30_df=m30_df,

                m60_df=m60_df
            )


            if result is not None:

                final_results.append(
                    result
                )


        except Exception as e:

            print(
                f"⚠️ {code} 30K/60K錯誤：{e}"
            )

            continue


    # ==================================================================
    # 排序
    # ==================================================================

    for item in final_results:

        item["score"] = (
            calculate_strategy_score(
                item
            )
        )

        item["grade"] = (
            get_grade(
                item["score"]
            )
        )


    final_results = sorted(
        final_results,
        key=lambda x: x["score"],
        reverse=True
    )


    # ==================================================================
    # 顯示
    # ==================================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"🎯 最終符合："
        f"{len(final_results)} 檔"
    )

    print(
        "=" * 70
    )


    for i, item in enumerate(
        final_results[:50],
        1
    ):

        print(
            f"{i:02d}. "
            f"{item['label']} "
            f"| {item['grade']} "
            f"{item['score']}分 "
            f"| "
            f"S1={item['s1']} "
            f"S2={item['s2']} "
            f"S3={item['s3']} "
            f"S4={item['s4']} "
            f"S5={item['s5']} "
            f"S6={item['s6']} "
            f"| "
            f"60K縮柱="
            f"{item['m60_trigger']['green_shrinking']} "
            f"轉紅="
            f"{item['m60_trigger']['green_to_red']}"
        )


    # ==================================================================
    # Telegram
    # ==================================================================

    if final_results:

        report = build_telegram_report(
            final_results
        )

        send_telegram_message(
            report
        )

    else:

        report = build_telegram_report(
            []
        )

        send_telegram_message(
            report
        )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "✅ 選股完成"
    )

    print(
        "=" * 70
    )


# ==============================================================================
# Entry
# ==============================================================================

if __name__ == "__main__":
    main()
```
