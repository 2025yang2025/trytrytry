# ============================================================
# Taiwan Multi-Timeframe MACD Scanner
# Version: v3.0
#
# 核心邏輯：
#
# 月K MACD 突破 0 軸
#       ↓
# 週K MACD 突破 0 軸
#       ↓
# 確認中長期多方
#       ↓
# 日K MACD + KD 找準備發動
#       ↓
# 60分鐘 MACD 綠柱縮小
#       ↓
# 綠柱 → 紅柱 = 最強訊號
#
# Telegram:
#   TG_BOT_TOKEN
#   TG_CHAT_ID
# ============================================================

import os
import time
import html
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
import requests


# ============================================================
# 基本設定
# ============================================================

DYNAMIC_STOCK_NAMES = {}

TAIWAN_TZ = ZoneInfo("Asia/Taipei")

# ------------------------------------------------------------
# 流動性條件
# ------------------------------------------------------------

# 20日平均成交量 >= 1000 張
# 1000 張 = 1,000,000 股
MIN_AVG_VOLUME_LOTS = 1000

# ------------------------------------------------------------
# 月 / 週 MACD 零軸突破設定
# ------------------------------------------------------------

# 月K：
# 最近幾根月K內曾經突破 0 軸，
# 且目前 MACD 仍然 > 0
MONTHLY_CROSS_LOOKBACK = 3

# 週K：
# 最近幾根週K內曾經突破 0 軸，
# 且目前 MACD 仍然 > 0
WEEKLY_CROSS_LOOKBACK = 6

# ------------------------------------------------------------
# 日K條件
# ------------------------------------------------------------

DAILY_KD_THRESHOLD = 20

# 日K MACD 必須 > 0
DAILY_REQUIRE_MACD_ABOVE_ZERO = True

# ------------------------------------------------------------
# 60分鐘條件
# ------------------------------------------------------------

# 綠柱縮小：
# hist < 0
# 且目前 hist > 前一根 hist
#
# 例如：
# -0.80 → -0.60 → -0.35 → -0.10
#
# 越來越接近 0
#
INTRADAY_REQUIRE_TWO_BAR_SHRINK = True

# ------------------------------------------------------------
# Yahoo Finance
# ------------------------------------------------------------

YF_CHUNK_SIZE = 150
YF_RETRY = 2

# 60分鐘資料只需要最近 1 個月
INTRADAY_PERIOD = "1mo"

# ------------------------------------------------------------
# Telegram
# ------------------------------------------------------------

TELEGRAM_MAX_LENGTH = 3500


# ============================================================
# 取得台股清單
# ============================================================

def fetch_all_taiwan_market_tickers():
    """
    從 TWSE OpenAPI 取得上市股票清單。

    回傳：
        ['2330.TW', '2317.TW', ...]
    """

    url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/120 Safari/537.36"
        )
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        tickers = []

        for item in data:

            code = str(item.get("Code", "")).strip()
            name = str(item.get("Name", "")).strip()

            if code.isdigit() and len(code) == 4:

                ticker = f"{code}.TW"

                tickers.append(ticker)

                DYNAMIC_STOCK_NAMES[ticker] = name

        tickers = sorted(list(set(tickers)))

        print(f"取得上市股票：{len(tickers)} 檔")

        return tickers

    except Exception as e:

        print(f"取得台股清單失敗：{e}")

        return []


# ============================================================
# Yahoo Finance 批次下載
# ============================================================

def safe_download_yf(
    tickers,
    period,
    interval,
    chunk_size=YF_CHUNK_SIZE
):
    """
    批次下載 Yahoo Finance 資料。

    回傳：
        dict[
            ticker -> DataFrame
        ]
    """

    results = {}

    if not tickers:
        return results

    for start in range(0, len(tickers), chunk_size):

        chunk = tickers[start:start + chunk_size]

        print(
            f"下載 {interval}："
            f"{start + 1}-{min(start + chunk_size, len(tickers))}"
            f"/{len(tickers)}"
        )

        success = False

        for attempt in range(YF_RETRY + 1):

            try:

                data = yf.download(
                    chunk,
                    period=period,
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                    threads=True,
                    group_by="column"
                )

                if data is None or data.empty:
                    raise ValueError("Yahoo Finance 回傳空資料")

                # ------------------------------------------------
                # 單一股票
                # ------------------------------------------------

                if len(chunk) == 1:

                    ticker = chunk[0]

                    df = extract_ticker_df(
                        data,
                        ticker
                    )

                    if df is not None and not df.empty:
                        results[ticker] = df

                # ------------------------------------------------
                # 多股票
                # ------------------------------------------------

                else:

                    for ticker in chunk:

                        try:

                            df = extract_ticker_df(
                                data,
                                ticker
                            )

                            if df is not None and not df.empty:
                                results[ticker] = df

                        except Exception as e:

                            print(
                                f"{ticker} 解析失敗：{e}"
                            )

                success = True
                break

            except Exception as e:

                print(
                    f"Yahoo 下載失敗 "
                    f"{interval} "
                    f"第 {attempt + 1} 次：{e}"
                )

                if attempt < YF_RETRY:
                    time.sleep(2)

        if not success:
            print(
                f"區段 {start + 1}-"
                f"{min(start + chunk_size, len(tickers))}"
                f"下載失敗"
            )

        time.sleep(0.5)

    return results


# ============================================================
# Yahoo DataFrame 解析
# ============================================================

def extract_ticker_df(data, ticker):
    """
    將 yfinance 不同版本可能產生的 MultiIndex
    統一整理成：

        Open
        High
        Low
        Close
        Volume
    """

    if data is None or data.empty:
        return None

    df = None

    # --------------------------------------------------------
    # MultiIndex
    # --------------------------------------------------------

    if isinstance(data.columns, pd.MultiIndex):

        level0 = data.columns.get_level_values(0)
        level1 = data.columns.get_level_values(1)

        # (Price, Ticker)
        if ticker in level1:

            try:
                df = data.xs(
                    ticker,
                    axis=1,
                    level=1
                )
            except Exception:
                pass

        # (Ticker, Price)
        elif ticker in level0:

            try:
                df = data.xs(
                    ticker,
                    axis=1,
                    level=0
                )
            except Exception:
                pass

    else:

        df = data.copy()

    if df is None or df.empty:
        return None

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    # --------------------------------------------------------
    # 欄位清理
    # --------------------------------------------------------

    new_columns = []

    for col in df.columns:

        if isinstance(col, tuple):
            new_columns.append(str(col[-1]))
        else:
            new_columns.append(str(col))

    df.columns = new_columns

    # --------------------------------------------------------
    # 只保留必要欄位
    # --------------------------------------------------------

    available = [
        col
        for col in required
        if col in df.columns
    ]

    if len(available) < 4:
        return None

    df = df[available].copy()

    # --------------------------------------------------------
    # 數值化
    # --------------------------------------------------------

    for col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            col
            for col in [
                "Open",
                "High",
                "Low",
                "Close"
            ]
            if col in df.columns
        ]
    )

    df = df.sort_index()

    return df


# ============================================================
# MACD
# ============================================================

def calculate_macd(
    close_series,
    fast=12,
    slow=26,
    signal=9
):
    """
    標準 MACD
    """

    close = pd.to_numeric(
        close_series,
        errors="coerce"
    ).dropna()

    if len(close) < slow + signal:
        return (
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            pd.Series(dtype=float)
        )

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

    histogram = macd - signal_line

    return (
        macd,
        signal_line,
        histogram
    )


# ============================================================
# KD
# ============================================================

def calculate_kd(
    df,
    n=9,
    m1=3,
    m2=3
):
    """
    標準隨機指標 KD
    """

    if df is None or len(df) < n:
        return (
            pd.Series(dtype=float),
            pd.Series(dtype=float)
        )

    high = pd.to_numeric(
        df["High"],
        errors="coerce"
    )

    low = pd.to_numeric(
        df["Low"],
        errors="coerce"
    )

    close = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    lowest_low = low.rolling(
        n
    ).min()

    highest_high = high.rolling(
        n
    ).max()

    denominator = (
        highest_high - lowest_low
    )

    denominator = denominator.replace(
        0,
        pd.NA
    )

    rsv = (
        (close - lowest_low)
        / denominator
        * 100
    )

    k_values = []
    d_values = []

    k = 50.0
    d = 50.0

    for value in rsv:

        if pd.isna(value):

            k_values.append(k)
            d_values.append(d)

            continue

        k = (
            (m1 - 1) * k
            + float(value)
        ) / m1

        d = (
            (m2 - 1) * d
            + k
        ) / m2

        k_values.append(k)
        d_values.append(d)

    k_series = pd.Series(
        k_values,
        index=df.index
    )

    d_series = pd.Series(
        d_values,
        index=df.index
    )

    return (
        k_series,
        d_series
    )


# ============================================================
# MACD 零軸突破
# ============================================================

def macd_recent_zero_cross(
    macd,
    lookback=3
):
    """
    判斷 MACD 最近 lookback 根內
    是否曾經由 <= 0 突破至 > 0。

    同時要求目前 MACD > 0。

    例如：

        -0.8
        -0.4
        +0.1
        +0.3

    → True
    """

    if macd is None or len(macd) < 2:
        return False

    macd = pd.Series(
        macd
    ).dropna()

    if len(macd) < 2:
        return False

    # 目前仍在 0 軸上
    if macd.iloc[-1] <= 0:
        return False

    lookback = max(
        1,
        int(lookback)
    )

    start = max(
        1,
        len(macd) - lookback
    )

    for i in range(start, len(macd)):

        previous = macd.iloc[i - 1]
        current = macd.iloc[i]

        if previous <= 0 and current > 0:
            return True

    return False


# ============================================================
# 月K：多方確認
# ============================================================

def check_monthly_bullish(df):
    """
    月K：

    1. MACD > 0
    2. 最近 N 根月K 曾突破 0 軸

    → 確認長期多方
    """

    if df is None or len(df) < 40:
        return None

    close = df["Close"]

    macd, signal, hist = calculate_macd(
        close
    )

    if len(macd) < 5:
        return None

    current_macd = float(
        macd.iloc[-1]
    )

    cross = macd_recent_zero_cross(
        macd,
        MONTHLY_CROSS_LOOKBACK
    )

    return {
        "bullish": bool(cross),
        "macd": current_macd,
        "signal": float(signal.iloc[-1]),
        "hist": float(hist.iloc[-1]),
        "cross": bool(cross),
    }


# ============================================================
# 週K：多方確認
# ============================================================

def check_weekly_bullish(df):
    """
    週K：

    1. MACD > 0
    2. 最近 N 根週K 曾突破 0 軸

    → 確認中期多方
    """

    if df is None or len(df) < 60:
        return None

    close = df["Close"]

    macd, signal, hist = calculate_macd(
        close
    )

    if len(macd) < 5:
        return None

    current_macd = float(
        macd.iloc[-1]
    )

    cross = macd_recent_zero_cross(
        macd,
        WEEKLY_CROSS_LOOKBACK
    )

    return {
        "bullish": bool(cross),
        "macd": current_macd,
        "signal": float(signal.iloc[-1]),
        "hist": float(hist.iloc[-1]),
        "cross": bool(cross),
    }


# ============================================================
# 日K：找準備發動
# ============================================================

def check_daily_setup(
    df,
    kd_threshold=DAILY_KD_THRESHOLD
):
    """
    日K條件：

    1. MACD > 0
    2. KD > threshold
    3. MACD Histogram 改善

    這裡不要求日K一定剛突破 0，
    因為真正進場觸發交給 60 分鐘。
    """

    if df is None or len(df) < 120:
        return None

    close = df["Close"]

    macd, signal, hist = calculate_macd(
        close
    )

    if len(macd) < 5:
        return None

    k, d = calculate_kd(
        df
    )

    if len(k) == 0 or len(d) == 0:
        return None

    current_macd = float(
        macd.iloc[-1]
    )

    current_hist = float(
        hist.iloc[-1]
    )

    previous_hist = float(
        hist.iloc[-2]
    )

    current_k = float(
        k.iloc[-1]
    )

    current_d = float(
        d.iloc[-1]
    )

    macd_ok = (
        current_macd > 0
        if DAILY_REQUIRE_MACD_ABOVE_ZERO
        else True
    )

    kd_ok = (
        current_k > kd_threshold
        and current_d > kd_threshold
    )

    hist_improving = (
        current_hist > previous_hist
    )

    # 日K不是最終觸發，
    # 但至少要在多方區並且 MACD 有改善
    setup = (
        macd_ok
        and kd_ok
        and hist_improving
    )

    return {
        "setup": bool(setup),
        "macd": current_macd,
        "signal": float(signal.iloc[-1]),
        "hist": current_hist,
        "prev_hist": previous_hist,
        "k": current_k,
        "d": current_d,
        "hist_improving": hist_improving,
    }


# ============================================================
# 60分鐘：綠柱縮小
# ============================================================

def check_60m_macd_trigger(df):
    """
    60分鐘 MACD 最重要的進場觸發。

    狀態：

    A. 綠柱縮小

        -0.80
        -0.60
        -0.35

        → 越來越接近 0

    B. 綠柱 → 紅柱

        -0.30
        -0.10
        +0.05

        → 最強訊號

    C. 紅柱持續

        +0.10
        +0.20
        +0.35

        → 已經發動，可能錯過最佳切入點
    """

    if df is None or len(df) < 50:
        return None

    close = df["Close"]

    macd, signal, hist = calculate_macd(
        close
    )

    if len(hist) < 5:
        return None

    h1 = float(hist.iloc[-1])
    h2 = float(hist.iloc[-2])
    h3 = float(hist.iloc[-3])

    m1 = float(macd.iloc[-1])
    s1 = float(signal.iloc[-1])

    # --------------------------------------------------------
    # 綠柱縮小
    # --------------------------------------------------------

    shrink_one = (
        h1 < 0
        and h1 > h2
    )

    shrink_two = (
        h1 < 0
        and h1 > h2
        and h2 > h3
    )

    if INTRADAY_REQUIRE_TWO_BAR_SHRINK:
        green_shrinking = shrink_two
    else:
        green_shrinking = shrink_one

    # --------------------------------------------------------
    # 綠柱 → 紅柱
    # --------------------------------------------------------

    green_to_red = (
        h2 < 0
        and h1 >= 0
        and h1 > h2
    )

    # --------------------------------------------------------
    # 紅柱持續
    # --------------------------------------------------------

    red_positive = (
        h1 > 0
    )

    # --------------------------------------------------------
    # 狀態
    # --------------------------------------------------------

    if green_to_red:

        status = "GREEN_TO_RED"
        trigger = True

    elif green_shrinking:

        status = "GREEN_SHRINKING"
        trigger = True

    elif red_positive:

        status = "RED"
        trigger = False

    else:

        status = "OTHER"
        trigger = False

    return {
        "trigger": trigger,
        "status": status,
        "macd": m1,
        "signal": s1,
        "hist": h1,
        "prev_hist": h2,
        "prev2_hist": h3,
        "green_shrinking": green_shrinking,
        "green_to_red": green_to_red,
    }


# ============================================================
# 日K低檔爆量
# ============================================================

def check_low_position_volume_surge(
    df
):
    """
    保留原本的低檔爆量邏輯，
    但不作為主策略的必要條件。

    條件：

    1. 至少 120 日
    2. 價格位於 120 日區間低檔 30% 以下
    3. 今日成交量 >= 前 5 日平均成交量 × 2.5
    4. 今日紅K
    """

    if df is None or len(df) < 120:
        return False

    close = df["Close"]
    open_price = df["Open"]
    volume = df["Volume"]

    highest = close.rolling(
        120
    ).max()

    lowest = close.rolling(
        120
    ).min()

    price_range = (
        highest.iloc[-1]
        - lowest.iloc[-1]
    )

    if price_range <= 0:
        return False

    position = (
        (
            close.iloc[-1]
            - lowest.iloc[-1]
        )
        / price_range
    )

    previous_5_volume = (
        volume.shift(1)
        .rolling(5)
        .mean()
        .iloc[-1]
    )

    if pd.isna(previous_5_volume):
        return False

    volume_surge = (
        volume.iloc[-1]
        >= previous_5_volume * 2.5
    )

    red_candle = (
        close.iloc[-1]
        > open_price.iloc[-1]
    )

    return (
        position <= 0.30
        and volume_surge
        and red_candle
    )


# ============================================================
# 成交量過濾
# ============================================================

def check_liquidity(df):
    """
    20日平均成交量 >= 1000 張
    """

    if df is None or len(df) < 20:
        return False

    volume = pd.to_numeric(
        df["Volume"],
        errors="coerce"
    )

    avg_volume = (
        volume
        .rolling(20)
        .mean()
        .iloc[-1]
    )

    if pd.isna(avg_volume):
        return False

    avg_volume_lots = (
        avg_volume / 1000
    )

    return (
        avg_volume_lots
        >= MIN_AVG_VOLUME_LOTS
    )


# ============================================================
# 取得最新價格
# ============================================================

def get_latest_price(df):

    if df is None or df.empty:
        return None

    try:
        return float(
            df["Close"].iloc[-1]
        )
    except Exception:
        return None


# ============================================================
# 取得股票名稱
# ============================================================

def get_stock_name(ticker):

    return DYNAMIC_STOCK_NAMES.get(
        ticker,
        ticker.replace(".TW", "")
    )


# ============================================================
# 取得股票代碼
# ============================================================

def get_stock_code(ticker):

    return ticker.replace(
        ".TW",
        ""
    )


# ============================================================
# 格式化數值
# ============================================================

def fmt(value, digits=3):

    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "-"


# ============================================================
# 建立 Telegram 訊息
# ============================================================

def build_telegram_message(
    final_green_to_red,
    final_green_shrinking,
    confirmed_daily_candidates,
    low_volume_candidates,
    total_stocks
):

    now = datetime.now(
        TAIWAN_TZ
    ).strftime(
        "%Y-%m-%d %H:%M"
    )

    lines = []

    lines.append(
        "🚀 <b>台股多週期 MACD Scanner v3.0</b>"
    )

    lines.append(
        f"🕐 {html.escape(now)}"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🧭 <b>主策略邏輯</b>"
    )

    lines.append(
        "月K突破0軸"
    )

    lines.append(
        "↓"
    )

    lines.append(
        "週K突破0軸"
    )

    lines.append(
        "↓"
    )

    lines.append(
        "日K多方 + KD"
    )

    lines.append(
        "↓"
    )

    lines.append(
        "60分鐘綠柱縮小 → 紅柱"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        f"📊 掃描上市股票：{total_stocks} 檔"
    )

    lines.append(
        f"🟡 月週日確認：{len(confirmed_daily_candidates)} 檔"
    )

    lines.append(
        f"🟢 60分綠柱縮小：{len(final_green_shrinking)} 檔"
    )

    lines.append(
        f"🔥 60分綠→紅：{len(final_green_to_red)} 檔"
    )

    # ========================================================
    # 最強訊號
    # ========================================================

    lines.append("")
    lines.append(
        "🔥 <b>最強訊號｜60分綠柱 → 紅柱</b>"
    )

    if not final_green_to_red:

        lines.append(
            "目前沒有符合條件的標的"
        )

    else:

        for item in final_green_to_red:

            lines.append(
                format_candidate(
                    item,
                    "🔥 綠→紅"
                )
            )

    # ========================================================
    # 綠柱縮小
    # ========================================================

    lines.append("")
    lines.append(
        "🟢 <b>進場觀察｜60分綠柱縮小</b>"
    )

    if not final_green_shrinking:

        lines.append(
            "目前沒有符合條件的標的"
        )

    else:

        for item in final_green_shrinking:

            lines.append(
                format_candidate(
                    item,
                    "🟢 綠柱縮小"
                )
            )

    # ========================================================
    # 已確認但尚未觸發
    # ========================================================

    lines.append("")
    lines.append(
        "📌 <b>月K＋週K＋日K確認，等待60分</b>"
    )

    if not confirmed_daily_candidates:

        lines.append(
            "目前沒有符合條件的標的"
        )

    else:

        # 最多顯示 20 檔
        for item in confirmed_daily_candidates[:20]:

            lines.append(
                format_candidate(
                    item,
                    "📌 等待觸發"
                )
            )

    # ========================================================
    # 原始低檔爆量
    # ========================================================

    lines.append("")
    lines.append(
        "💥 <b>輔助｜低檔爆量</b>"
    )

    if not low_volume_candidates:

        lines.append(
            "目前沒有符合條件的標的"
        )

    else:

        for item in low_volume_candidates[:15]:

            code = html.escape(
                item["code"]
            )

            name = html.escape(
                item["name"]
            )

            price = fmt(
                item["price"],
                2
            )

            lines.append(
                f"💥 <b>{code} {name}</b> "
                f"現價 {price}"
            )

    return "\n".join(lines)


# ============================================================
# 候選股票格式
# ============================================================

def format_candidate(
    item,
    status
):

    code = html.escape(
        item["code"]
    )

    name = html.escape(
        item["name"]
    )

    price = fmt(
        item["price"],
        2
    )

    monthly = item["monthly"]
    weekly = item["weekly"]
    daily = item["daily"]
    intraday = item.get("intraday")

    lines = []

    lines.append(
        f"{status} <b>{code} {name}</b> "
        f"現價 {price}"
    )

    lines.append(
        f"　月MACD {fmt(monthly['macd'])} "
        f"週MACD {fmt(weekly['macd'])}"
    )

    lines.append(
        f"　日MACD {fmt(daily['macd'])} "
        f"KD {fmt(daily['k'], 1)}/"
        f"{fmt(daily['d'], 1)}"
    )

    if intraday:

        lines.append(
            f"　60分Hist "
            f"{fmt(intraday['hist'])}"
        )

    return "\n".join(lines)


# ============================================================
# Telegram 發送
# ============================================================

def send_telegram_message(
    message
):

    token = os.getenv(
        "TG_BOT_TOKEN"
    )

    chat_id = os.getenv(
        "TG_CHAT_ID"
    )

    if not token or not chat_id:

        print(
            "未設定 TG_BOT_TOKEN 或 TG_CHAT_ID"
        )

        return False

    url = (
        f"https://api.telegram.org/"
        f"bot{token}/sendMessage"
    )

    # Telegram 單則訊息限制
    chunks = []

    current = ""

    for line in message.splitlines():

        if (
            len(current)
            + len(line)
            + 1
            > TELEGRAM_MAX_LENGTH
        ):

            if current:
                chunks.append(
                    current
                )

            current = line

        else:

            if current:
                current += "\n"

            current += line

    if current:
        chunks.append(current)

    success = True

    for chunk in chunks:

        try:

            response = requests.post(
                url,
                json={
                    "chat_id": chat_id,
                    "text": chunk,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=20
            )

            response.raise_for_status()

            result = response.json()

            if not result.get("ok"):

                print(
                    "Telegram 回傳錯誤：",
                    result
                )

                success = False

        except Exception as e:

            print(
                f"Telegram 發送失敗：{e}"
            )

            success = False

    return success


# ============================================================
# 主程式
# ============================================================

def main():

    start_time = time.time()

    print(
        "================================================"
    )

    print(
        "🚀 Taiwan Multi-Timeframe MACD Scanner v3.0"
    )

    print(
        "================================================"
    )

    # ========================================================
    # 1. 取得台股
    # ========================================================

    tickers = (
        fetch_all_taiwan_market_tickers()
    )

    if not tickers:

        print(
            "沒有取得任何股票"
        )

        return

    # ========================================================
    # 2. 下載日K
    # ========================================================

    print("")
    print(
        "📥 開始下載日K..."
    )

    daily_data = safe_download_yf(
        tickers,
        period="1y",
        interval="1d"
    )

    print(
        f"日K取得：{len(daily_data)} 檔"
    )

    # ========================================================
    # 3. 日K流動性過濾
    # ========================================================

    liquid_tickers = []

    for ticker, df in daily_data.items():

        try:

            if len(df) < 120:
                continue

            if not check_liquidity(df):
                continue

            liquid_tickers.append(
                ticker
            )

        except Exception as e:

            print(
                f"{ticker} 流動性判斷錯誤：{e}"
            )

    print(
        f"流動性過濾後："
        f"{len(liquid_tickers)} 檔"
    )

    if not liquid_tickers:

        print(
            "沒有符合流動性條件的股票"
        )

        return

    # ========================================================
    # 4. 下載週K
    # ========================================================

    print("")
    print(
        "📥 開始下載週K..."
    )

    weekly_data = safe_download_yf(
        liquid_tickers,
        period="2y",
        interval="1wk"
    )

    print(
        f"週K取得：{len(weekly_data)} 檔"
    )

    # ========================================================
    # 5. 下載月K
    # ========================================================

    print("")
    print(
        "📥 開始下載月K..."
    )

    monthly_data = safe_download_yf(
        liquid_tickers,
        period="5y",
        interval="1mo"
    )

    print(
        f"月K取得：{len(monthly_data)} 檔"
    )

    # ========================================================
    # 6. 月K＋週K多方確認
    # ========================================================

    confirmed_monthly_weekly = []

    for ticker in liquid_tickers:

        df_m = monthly_data.get(
            ticker
        )

        df_w = weekly_data.get(
            ticker
        )

        if df_m is None or df_w is None:
            continue

        try:

            monthly = (
                check_monthly_bullish(
                    df_m
                )
            )

            if not monthly:
                continue

            if not monthly["bullish"]:
                continue

            weekly = (
                check_weekly_bullish(
                    df_w
                )
            )

            if not weekly:
                continue

            if not weekly["bullish"]:
                continue

            confirmed_monthly_weekly.append(
                {
                    "ticker": ticker,
                    "monthly": monthly,
                    "weekly": weekly,
                }
            )

        except Exception as e:

            print(
                f"{ticker} 月週K判斷錯誤：{e}"
            )

    print(
        f"月K＋週K多方確認："
        f"{len(confirmed_monthly_weekly)} 檔"
    )

    if not confirmed_monthly_weekly:

        message = build_telegram_message(
            [],
            [],
            [],
            [],
            len(tickers)
        )

        send_telegram_message(
            message
        )

        return

    # ========================================================
    # 7. 日K找準備發動
    # ========================================================

    confirmed_daily_candidates = []

    for item in confirmed_monthly_weekly:

        ticker = item["ticker"]

        df_d = daily_data.get(
            ticker
        )

        if df_d is None:
            continue

        try:

            daily = check_daily_setup(
                df_d
            )

            if not daily:
                continue

            if not daily["setup"]:
                continue

            item["daily"] = daily

            item["price"] = (
                get_latest_price(df_d)
            )

            item["code"] = (
                get_stock_code(ticker)
            )

            item["name"] = (
                get_stock_name(ticker)
            )

            confirmed_daily_candidates.append(
                item
            )

        except Exception as e:

            print(
                f"{ticker} 日K判斷錯誤：{e}"
            )

    print(
        f"月週日全部確認："
        f"{len(confirmed_daily_candidates)} 檔"
    )

    # ========================================================
    # 8. 下載 60分鐘
    #
    # 重要：
    #
    # 不再使用：
    #
    #     heavy_scan_pool[:50]
    #
    # 而是：
    #
    #     月K
    #     ↓
    #     週K
    #     ↓
    #     日K
    #     ↓
    #     真正候選
    #
    # 再下載 60分鐘。
    # ========================================================

    intraday_tickers = [
        item["ticker"]
        for item in confirmed_daily_candidates
    ]

    final_green_to_red = []
    final_green_shrinking = []

    if intraday_tickers:

        print("")
        print(
            "📥 開始下載 60分鐘K..."
        )

        intraday_data = safe_download_yf(
            intraday_tickers,
            period=INTRADAY_PERIOD,
            interval="60m"
        )

        print(
            f"60分鐘資料取得："
            f"{len(intraday_data)} 檔"
        )

        # ====================================================
        # 9. 60分鐘觸發判斷
        # ====================================================

        for item in confirmed_daily_candidates:

            ticker = item["ticker"]

            df_60 = intraday_data.get(
                ticker
            )

            if df_60 is None:
                continue

            try:

                intraday = (
                    check_60m_macd_trigger(
                        df_60
                    )
                )

                if not intraday:
                    continue

                item["intraday"] = (
                    intraday
                )

                # --------------------------------------------
                # 最強：
                # 綠柱 → 紅柱
                # --------------------------------------------

                if intraday[
                    "green_to_red"
                ]:

                    final_green_to_red.append(
                        item
                    )

                # --------------------------------------------
                # 次強：
                # 綠柱持續縮小
                # --------------------------------------------

                elif intraday[
                    "green_shrinking"
                ]:

                    final_green_shrinking.append(
                        item
                    )

            except Exception as e:

                print(
                    f"{ticker} 60分判斷錯誤：{e}"
                )

    # ========================================================
    # 10. 低檔爆量輔助策略
    # ========================================================

    low_volume_candidates = []

    for ticker, df in daily_data.items():

        try:

            if not check_liquidity(df):
                continue

            if not check_low_position_volume_surge(
                df
            ):
                continue

            low_volume_candidates.append(
                {
                    "ticker": ticker,
                    "code": get_stock_code(
                        ticker
                    ),
                    "name": get_stock_name(
                        ticker
                    ),
                    "price": get_latest_price(
                        df
                    ),
                }
            )

        except Exception as e:

            print(
                f"{ticker} 低檔爆量判斷錯誤：{e}"
            )

    # ========================================================
    # 11. 排序
    # ========================================================

    # 綠→紅：
    # MACD Hist 越高越優先
    final_green_to_red.sort(
        key=lambda x:
            x["intraday"]["hist"],
        reverse=True
    )

    # 綠柱縮小：
    # 越接近 0 越優先
    final_green_shrinking.sort(
        key=lambda x:
            x["intraday"]["hist"],
        reverse=True
    )

    # 月週日候選：
    # 日K Hist 越強越優先
    confirmed_daily_candidates.sort(
        key=lambda x:
            x["daily"]["hist"],
        reverse=True
    )

    # ========================================================
    # 12. 輸出結果
    # ========================================================

    print("")
    print(
        "================================================"
    )

    print(
        "🔥 最強｜60分綠 → 紅"
    )

    print(
        "================================================"
    )

    for item in final_green_to_red:

        print(
            item["code"],
            item["name"],
            "Price=",
            fmt(item["price"], 2),
            "M=",
            fmt(item["monthly"]["macd"]),
            "W=",
            fmt(item["weekly"]["macd"]),
            "D=",
            fmt(item["daily"]["macd"]),
            "60H=",
            fmt(item["intraday"]["hist"])
        )

    print("")
    print(
        "================================================"
    )

    print(
        "🟢 60分綠柱縮小"
    )

    print(
        "================================================"
    )

    for item in final_green_shrinking:

        print(
            item["code"],
            item["name"],
            "Price=",
            fmt(item["price"], 2),
            "M=",
            fmt(item["monthly"]["macd"]),
            "W=",
            fmt(item["weekly"]["macd"]),
            "D=",
            fmt(item["daily"]["macd"]),
            "60H=",
            fmt(item["intraday"]["hist"])
        )

    print("")
    print(
        "================================================"
    )

    print(
        "📌 月週日確認"
    )

    print(
        "================================================"
    )

    for item in confirmed_daily_candidates:

        print(
            item["code"],
            item["name"],
            "Price=",
            fmt(item["price"], 2)
        )

    # ========================================================
    # 13. Telegram
    # ========================================================

    message = build_telegram_message(
        final_green_to_red=final_green_to_red,
        final_green_shrinking=final_green_shrinking,
        confirmed_daily_candidates=confirmed_daily_candidates,
        low_volume_candidates=low_volume_candidates,
        total_stocks=len(tickers)
    )

    print("")
    print(
        "📨 發送 Telegram..."
    )

    send_telegram_message(
        message
    )

    # ========================================================
    # 14. 完成
    # ========================================================

    elapsed = (
        time.time()
        - start_time
    )

    print("")
    print(
        "================================================"
    )

    print(
        f"✅ 掃描完成，耗時 {elapsed:.1f} 秒"
    )

    print(
        f"🔥 綠→紅："
        f"{len(final_green_to_red)}"
    )

    print(
        f"🟢 綠柱縮小："
        f"{len(final_green_shrinking)}"
    )

    print(
        f"📌 月週日確認："
        f"{len(confirmed_daily_candidates)}"
    )

    print(
        "================================================"
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    main()
