# scanner.py
# ==============================================================================
# 🇹🇼 台股 7 大策略選股 Pro v2.1
#
# 功能：
#   1. TWSE 全市場股票清單
#   2. 日K / 週K / 月K 多週期掃描
#   3. 30m / 60m 短週期精掃
#   4. MACD + KD
#   5. 低檔爆量策略
#   6. 技術強度排名
#   7. 多策略交叉評分
#   8. ⭐ 新增策略7：S3 + S4 + S5 三週期共振
#   9. 🔥 新增60分鐘綠柱縮小 → 紅柱判斷
#  10. Telegram HTML 報告
#
# 環境變數：
#   TG_BOT_TOKEN
#   TG_CHAT_ID
#
# 安裝：
#   pip install pandas yfinance requests
# ==============================================================================

import os
import time
import html
import requests
import pandas as pd
import yfinance as yf


# ==============================================================================
# ⚙️ 基本設定
# ==============================================================================

VERSION = "Pro v2.1"

TWSE_API_URL = (
    "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
    )
}

# ------------------------------------------------------------------------------
# 最低 20 日平均成交量
# 1,000,000 股 = 1,000 張
# ------------------------------------------------------------------------------

MIN_AVG_VOLUME_20 = 1_000_000

# ------------------------------------------------------------------------------
# 30m / 60m 精掃最大股票數
# ------------------------------------------------------------------------------

INTRADAY_SCAN_LIMIT = 100

# ------------------------------------------------------------------------------
# Yahoo 分批下載數量
# ------------------------------------------------------------------------------

DAILY_CHUNK_SIZE = 150
INTRADAY_CHUNK_SIZE = 50

# ------------------------------------------------------------------------------
# 是否只掃普通股票
# ------------------------------------------------------------------------------

ONLY_COMMON_STOCK = True

# ------------------------------------------------------------------------------
# KD 門檻
# ------------------------------------------------------------------------------

KD_DAILY_THRESHOLD = 20
KD_WEEKLY_THRESHOLD = 50
KD_MONTHLY_THRESHOLD = 50
KD_INTRADAY_THRESHOLD = 20

# ------------------------------------------------------------------------------
# 60分鐘 MACD 綠柱縮小條件
# ------------------------------------------------------------------------------

# True：
# 必須連續兩根改善
#
# 例如：
# -0.80 → -0.55 → -0.30
#
# False：
# 只要最新一根比前一根改善即可
# ------------------------------------------------------------------------------

INTRADAY_REQUIRE_TWO_BAR_SHRINK = True


# ==============================================================================
# 📌 全域股票名稱
# ==============================================================================

DYNAMIC_STOCK_NAMES = {}


# ==============================================================================
# 🧰 工具
# ==============================================================================

def safe_float(value, default=0.0):
    """安全轉 float"""

    try:

        if pd.isna(value):
            return default

        return float(value)

    except Exception:

        return default


def get_ticker_code(ticker):
    """2330.TW → 2330"""

    return (
        ticker
        .replace(".TW", "")
        .replace(".TWO", "")
    )


def escape_html(text):
    """Telegram HTML escape"""

    return html.escape(str(text))


def get_stock_label(ticker):
    """建立 Telegram 股票標籤"""

    code = get_ticker_code(ticker)

    name = DYNAMIC_STOCK_NAMES.get(
        ticker,
        "",
    )

    if name:

        return (
            f"<code>{code}</code> "
            f"<i>{escape_html(name)}</i>"
        )

    return f"<code>{code}</code>"


# ==============================================================================
# 🇹🇼 TWSE 全市場股票清單
# ==============================================================================

def fetch_all_taiwan_market_tickers():

    all_tickers = []

    try:

        response = requests.get(
            TWSE_API_URL,
            headers=HEADERS,
            timeout=15,
        )

        response.raise_for_status()

        data = response.json()

        for item in data:

            code = str(
                item.get(
                    "Code",
                    "",
                )
            ).strip()

            name = str(
                item.get(
                    "Name",
                    "",
                )
            ).strip()

            if not code.isdigit():
                continue

            if len(code) != 4:
                continue

            ticker = f"{code}.TW"

            all_tickers.append(ticker)

            DYNAMIC_STOCK_NAMES[
                ticker
            ] = name

        all_tickers = sorted(
            set(all_tickers)
        )

        print(
            f"✅ TWSE 股票清單取得完成："
            f"{len(all_tickers)} 檔"
        )

        return all_tickers

    except Exception as e:

        print(
            f"❌ 撈取 TWSE 股票清單失敗：{e}"
        )

        return []


# ==============================================================================
# 🛡️ Yahoo Finance 安全下載
# ==============================================================================

def safe_download_yf(
    tickers,
    period,
    interval,
    chunk_size=100,
    max_retry=3,
):

    if not tickers:
        return pd.DataFrame()

    all_dfs = []

    total_chunks = (
        len(tickers)
        + chunk_size
        - 1
    ) // chunk_size

    for start in range(
        0,
        len(tickers),
        chunk_size,
    ):

        chunk = tickers[
            start:start + chunk_size
        ]

        chunk_no = (
            start // chunk_size
        ) + 1

        success = False

        for attempt in range(
            1,
            max_retry + 1,
        ):

            try:

                print(
                    f"📥 Yahoo "
                    f"{chunk_no}/{total_chunks} "
                    f"({len(chunk)} 檔) "
                    f"{interval} "
                    f"第 {attempt} 次"
                )

                df = yf.download(
                    tickers=chunk,
                    period=period,
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                    threads=True,
                    group_by="column",
                )

                if (
                    df is not None
                    and not df.empty
                ):

                    all_dfs.append(df)

                    success = True

                    break

                print(
                    f"⚠️ 批次 {chunk_no} "
                    f"回傳空資料"
                )

            except Exception as e:

                print(
                    f"⚠️ 批次 {chunk_no} "
                    f"下載失敗：{e}"
                )

            if attempt < max_retry:

                time.sleep(
                    2 * attempt
                )

        if not success:

            print(
                f"❌ 批次 "
                f"{chunk_no}/{total_chunks} "
                f"最終下載失敗"
            )

        time.sleep(0.5)

    if not all_dfs:
        return pd.DataFrame()

    try:

        result = pd.concat(
            all_dfs,
            axis=1,
        )

        return result

    except Exception as e:

        print(
            f"❌ 合併 Yahoo 資料失敗：{e}"
        )

        return pd.DataFrame()


# ==============================================================================
# 📊 MultiIndex 資料處理
# ==============================================================================

def extract_ticker_df(
    full_df,
    ticker,
):

    if (
        full_df is None
        or full_df.empty
    ):
        return pd.DataFrame()

    try:

        if isinstance(
            full_df.columns,
            pd.MultiIndex,
        ):

            for level_no in range(
                full_df.columns.nlevels
            ):

                try:

                    values = (
                        full_df
                        .columns
                        .get_level_values(
                            level_no
                        )
                    )

                    if ticker in values:

                        result = (
                            full_df.xs(
                                ticker,
                                axis=1,
                                level=level_no,
                            )
                        )

                        result = result.copy()

                        return normalize_columns(
                            result
                        )

                except Exception:
                    continue

            return pd.DataFrame()

        result = full_df.copy()

        return normalize_columns(
            result
        )

    except Exception:

        return pd.DataFrame()


# ==============================================================================
# 📊 欄位標準化
# ==============================================================================

def normalize_columns(df):

    if df is None or df.empty:
        return pd.DataFrame()

    rename_map = {}

    for col in df.columns:

        text = str(col).lower()

        if text == "open":
            rename_map[col] = "Open"

        elif text == "high":
            rename_map[col] = "High"

        elif text == "low":
            rename_map[col] = "Low"

        elif text == "close":
            rename_map[col] = "Close"

        elif text == "volume":
            rename_map[col] = "Volume"

    df = df.rename(
        columns=rename_map
    )

    return df


# ==============================================================================
# 📈 MACD
# ==============================================================================

def calculate_macd(
    close_series,
    fast=12,
    slow=26,
    signal=9,
):

    close = (
        pd.to_numeric(
            close_series,
            errors="coerce",
        )
        .dropna()
        .astype(float)
    )

    if len(close) < 35:

        return (
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            pd.Series(dtype=float),
        )

    fast_ema = close.ewm(
        span=fast,
        adjust=False,
    ).mean()

    slow_ema = close.ewm(
        span=slow,
        adjust=False,
    ).mean()

    macd_line = (
        fast_ema
        - slow_ema
    )

    signal_line = (
        macd_line
        .ewm(
            span=signal,
            adjust=False,
        )
        .mean()
    )

    histogram = (
        macd_line
        - signal_line
    )

    return (
        macd_line,
        signal_line,
        histogram,
    )


# ==============================================================================
# 📊 KD
# ==============================================================================

def calculate_kd(
    df_single,
    n=9,
    m1=3,
    m2=3,
):

    required = [
        "High",
        "Low",
        "Close",
    ]

    for col in required:

        if col not in df_single.columns:

            return (
                pd.Series(dtype=float),
                pd.Series(dtype=float),
            )

    df = (
        df_single[required]
        .copy()
    )

    for col in required:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.dropna()

    if len(df) < n:

        return (
            pd.Series(dtype=float),
            pd.Series(dtype=float),
        )

    lowest = (
        df["Low"]
        .rolling(
            window=n,
            min_periods=n,
        )
        .min()
    )

    highest = (
        df["High"]
        .rolling(
            window=n,
            min_periods=n,
        )
        .max()
    )

    denominator = (
        highest
        - lowest
    )

    rsv = (
        (
            df["Close"]
            - lowest
        )
        /
        denominator.replace(
            0,
            pd.NA,
        )
        * 100
    )

    rsv = rsv.fillna(50.0)

    k_values = []
    d_values = []

    k_prev = 50.0
    d_prev = 50.0

    for value in rsv:

        value = safe_float(
            value,
            50.0,
        )

        k_curr = (
            k_prev * (m1 - 1)
            + value
        ) / m1

        d_curr = (
            d_prev * (m2 - 1)
            + k_curr
        ) / m2

        k_values.append(
            k_curr
        )

        d_values.append(
            d_curr
        )

        k_prev = k_curr
        d_prev = d_curr

    return (
        pd.Series(
            k_values,
            index=df.index,
        ),
        pd.Series(
            d_values,
            index=df.index,
        ),
    )


# ==============================================================================
# 📌 KD 狀態
# ==============================================================================

def get_kd_state(df):

    k, d = calculate_kd(df)

    if (
        len(k) < 2
        or len(d) < 2
    ):

        return {
            "k": 0.0,
            "d": 0.0,
            "above": False,
            "golden_cross": False,
        }

    k_now = safe_float(
        k.iloc[-1]
    )

    d_now = safe_float(
        d.iloc[-1]
    )

    k_prev = safe_float(
        k.iloc[-2]
    )

    d_prev = safe_float(
        d.iloc[-2]
    )

    golden_cross = (
        k_prev <= d_prev
        and k_now > d_now
    )

    return {
        "k": k_now,
        "d": d_now,
        "above": k_now > d_now,
        "golden_cross": golden_cross,
    }


# ==============================================================================
# 📌 MACD 狀態
# ==============================================================================

def get_macd_state(df):

    if (
        df is None
        or df.empty
        or "Close" not in df.columns
    ):

        return None

    close = (
        pd.to_numeric(
            df["Close"],
            errors="coerce",
        )
        .dropna()
    )

    if len(close) < 35:
        return None

    macd, signal, hist = (
        calculate_macd(
            close
        )
    )

    if len(macd) < 3:
        return None

    macd_now = safe_float(
        macd.iloc[-1]
    )

    macd_prev = safe_float(
        macd.iloc[-2]
    )

    hist_now = safe_float(
        hist.iloc[-1]
    )

    hist_prev = safe_float(
        hist.iloc[-2]
    )

    hist_prev2 = safe_float(
        hist.iloc[-3]
    )

    signal_now = safe_float(
        signal.iloc[-1]
    )

    signal_prev = safe_float(
        signal.iloc[-2]
    )

    return {
        "macd": macd_now,

        "macd_prev":
            macd_prev,

        "signal":
            signal_now,

        "signal_prev":
            signal_prev,

        "hist":
            hist_now,

        "hist_prev":
            hist_prev,

        "hist_prev2":
            hist_prev2,

        "macd_positive":
            macd_now > 0,

        "macd_cross_zero":
            (
                macd_prev <= 0
                and macd_now > 0
            ),

        "macd_rising":
            macd_now > macd_prev,

        "hist_rising":
            hist_now > hist_prev,

        "hist_rising_2":
            (
                hist_now > hist_prev
                and
                hist_prev > hist_prev2
            ),

        "golden_cross":
            (
                signal_prev >= macd_prev
                and
                macd_now > signal_now
            ),

        "hist_negative_reducing":
            (
                hist_now < 0
                and
                hist_now > hist_prev
            ),
    }


# ==============================================================================
# 📈 策略 1 / 2
# MACD 負值減少 + KD
# ==============================================================================

def check_macd_negative_reducing_kd(
    df_tf,
    kd_threshold=20,
):

    try:

        if (
            df_tf is None
            or df_tf.empty
        ):

            return False, 0.0, {}

        clean = df_tf.dropna(
            subset=[
                "Close",
                "High",
                "Low",
            ]
        )

        if len(clean) < 35:

            return False, 0.0, {}

        macd_state = (
            get_macd_state(
                clean
            )
        )

        if macd_state is None:

            return False, 0.0, {}

        kd_state = (
            get_kd_state(
                clean
            )
        )

        macd_condition = (
            macd_state[
                "hist_negative_reducing"
            ]
            or
            (
                macd_state["macd"] < 0
                and
                macd_state[
                    "macd_rising"
                ]
            )
        )

        kd_condition = (
            kd_state["k"]
            > kd_threshold
            and
            kd_state["d"]
            > kd_threshold
        )

        result = (
            macd_condition
            and kd_condition
        )

        price = safe_float(
            clean[
                "Close"
            ].iloc[-1]
        )

        return (
            result,
            price,
            {
                "macd": macd_state,
                "kd": kd_state,
            },
        )

    except Exception:

        return False, 0.0, {}


# ==============================================================================
# 📈 策略 3 / 4 / 5
# MACD > 0 + KD
# ==============================================================================

def check_macd_above_zero_kd(
    df_tf,
    kd_threshold=20,
):

    try:

        if (
            df_tf is None
            or df_tf.empty
        ):

            return False, 0.0, {}

        clean = df_tf.dropna(
            subset=[
                "Close",
                "High",
                "Low",
            ]
        )

        if len(clean) < 35:

            return False, 0.0, {}

        macd_state = (
            get_macd_state(
                clean
            )
        )

        if macd_state is None:

            return False, 0.0, {}

        kd_state = (
            get_kd_state(
                clean
            )
        )

        macd_condition = (
            macd_state["macd"]
            > 0
        )

        kd_condition = (
            kd_state["k"]
            > kd_threshold
            and
            kd_state["d"]
            > kd_threshold
        )

        result = (
            macd_condition
            and kd_condition
        )

        price = safe_float(
            clean[
                "Close"
            ].iloc[-1]
        )

        return (
            result,
            price,
            {
                "macd": macd_state,
                "kd": kd_state,
            },
        )

    except Exception:

        return False, 0.0, {}


# ==============================================================================
# ⭐ 策略 7：S3 + S4 + S5 三週期共振
# ==============================================================================

def check_strategy_7_resonance(
    s3,
    s4,
    s5,
):
    """
    策略7：

        S3 日K
        +
        S4 週K
        +
        S5 月K

    三個策略同時成立。

    注意：
        策略7不是取代S3/S4/S5，
        而是額外建立「三週期共振」。
    """

    return (
        bool(s3)
        and
        bool(s4)
        and
        bool(s5)
    )


# ==============================================================================
# 🔥 60分鐘 MACD 綠柱縮小 → 紅柱
# ==============================================================================

def check_60m_macd_transition(
    df_60m
):
    """
    60分鐘核心觸發：

    1. 綠柱縮小

       -0.80
       -0.60
       -0.35

       → 越來越接近0

    2. 綠柱 → 紅柱

       -0.30
       -0.10
       +0.05

       → 最強訊號

    """

    if (
        df_60m is None
        or df_60m.empty
    ):

        return {
            "valid": False,
            "status": "NO_DATA",
        }

    clean = df_60m.dropna(
        subset=[
            "Close",
            "High",
            "Low",
        ]
    )

    if len(clean) < 35:

        return {
            "valid": False,
            "status": "NO_DATA",
        }

    macd_state = (
        get_macd_state(
            clean
        )
    )

    if macd_state is None:

        return {
            "valid": False,
            "status": "NO_DATA",
        }

    h1 = macd_state[
        "hist"
    ]

    h2 = macd_state[
        "hist_prev"
    ]

    h3 = macd_state[
        "hist_prev2"
    ]

    # --------------------------------------------------------------------------
    # 綠柱縮小
    # --------------------------------------------------------------------------

    shrink_one = (
        h1 < 0
        and
        h1 > h2
    )

    shrink_two = (
        h1 < 0
        and
        h1 > h2
        and
        h2 > h3
    )

    if INTRADAY_REQUIRE_TWO_BAR_SHRINK:

        green_shrinking = (
            shrink_two
        )

    else:

        green_shrinking = (
            shrink_one
        )

    # --------------------------------------------------------------------------
    # 綠柱 → 紅柱
    # --------------------------------------------------------------------------

    green_to_red = (
        h2 < 0
        and
        h1 >= 0
        and
        h1 > h2
    )

    # --------------------------------------------------------------------------
    # 判斷狀態
    # --------------------------------------------------------------------------

    if green_to_red:

        status = (
            "🔥 GREEN_TO_RED"
        )

    elif green_shrinking:

        status = (
            "🟢 GREEN_SHRINKING"
        )

    elif h1 > 0:

        status = (
            "🔴 RED"
        )

    elif h1 < 0:

        status = (
            "🟢 GREEN"
        )

    else:

        status = (
            "⚪ ZERO"
        )

    return {
        "valid": True,

        "status": status,

        "green_shrinking":
            green_shrinking,

        "green_to_red":
            green_to_red,

        "macd":
            macd_state["macd"],

        "signal":
            macd_state["signal"],

        "hist":
            h1,

        "hist_prev":
            h2,

        "hist_prev2":
            h3,
    }


# ==============================================================================
# 💥 策略 6
# 低檔爆量
# ==============================================================================

def check_strat_orig_8(
    df_daily,
):

    try:

        if df_daily is None:

            return (
                False,
                0.0,
                {},
            )

        required = [
            "Close",
            "High",
            "Low",
            "Open",
            "Volume",
        ]

        clean = (
            df_daily
            .dropna(
                subset=required
            )
            .copy()
        )

        if len(clean) < 120:

            return (
                False,
                0.0,
                {},
            )

        close = (
            clean["Close"]
            .astype(float)
        )

        open_price = (
            clean["Open"]
            .astype(float)
        )

        volume = (
            clean["Volume"]
            .astype(float)
        )

        low_120 = (
            close
            .rolling(120)
            .min()
            .iloc[-1]
        )

        high_120 = (
            close
            .rolling(120)
            .max()
            .iloc[-1]
        )

        if (
            pd.isna(low_120)
            or
            pd.isna(high_120)
            or
            high_120 <= low_120
        ):

            return (
                False,
                0.0,
                {},
            )

        current_price = (
            close.iloc[-1]
        )

        position = (
            current_price
            - low_120
        ) / (
            high_120
            - low_120
        )

        previous_volume_ma5 = (
            volume
            .rolling(5)
            .mean()
            .shift(1)
            .iloc[-1]
        )

        volume_ratio = (
            current_volume_ratio(
                volume,
                previous_volume_ma5,
            )
        )

        is_low_position = (
            position <= 0.30
        )

        is_volume_surge = (
            volume_ratio >= 2.5
        )

        is_red_k = (
            close.iloc[-1]
            > open_price.iloc[-1]
        )

        result = (
            is_low_position
            and
            is_volume_surge
            and
            is_red_k
        )

        return (
            result,
            current_price,
            {
                "position":
                    position,

                "volume_ratio":
                    volume_ratio,

                "red_k":
                    is_red_k,
            },
        )

    except Exception:

        return (
            False,
            0.0,
            {},
        )


def current_volume_ratio(
    volume,
    previous_ma5,
):

    if (
        previous_ma5 is None
        or
        previous_ma5 <= 0
    ):

        return 0.0

    return (
        safe_float(
            volume.iloc[-1]
        )
        /
        previous_ma5
    )


# ==============================================================================
# 📊 日K強度評分
# ==============================================================================

def calculate_daily_strength(
    df_daily,
):

    try:

        if df_daily is None:

            return 0.0, {}

        clean = df_daily.dropna(
            subset=[
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]
        )

        if len(clean) < 120:

            return 0.0, {}

        close = (
            clean["Close"]
            .astype(float)
        )

        volume = (
            clean["Volume"]
            .astype(float)
        )

        price = close.iloc[-1]

        ma5 = (
            close
            .rolling(5)
            .mean()
            .iloc[-1]
        )

        ma20 = (
            close
            .rolling(20)
            .mean()
            .iloc[-1]
        )

        ma60 = (
            close
            .rolling(60)
            .mean()
            .iloc[-1]
        )

        ma120 = (
            close
            .rolling(120)
            .mean()
            .iloc[-1]
        )

        volume_ma20 = (
            volume
            .rolling(20)
            .mean()
            .iloc[-1]
        )

        score = 0.0

        # ----------------------------------------------------------------------
        # 價格 > 均線
        # ----------------------------------------------------------------------

        if price > ma5:
            score += 5

        if price > ma20:
            score += 15

        if price > ma60:
            score += 10

        if price > ma120:
            score += 10

        # ----------------------------------------------------------------------
        # 均線多頭排列
        # ----------------------------------------------------------------------

        if (
            ma5 > ma20
            and
            ma20 > ma60
        ):

            score += 15

        if (
            ma20 > ma60
            and
            ma60 > ma120
        ):

            score += 10

        # ----------------------------------------------------------------------
        # MACD
        # ----------------------------------------------------------------------

        macd_state = (
            get_macd_state(
                clean
            )
        )

        if macd_state:

            if macd_state[
                "macd_positive"
            ]:

                score += 10

            if macd_state[
                "macd_rising"
            ]:

                score += 5

            if macd_state[
                "hist_rising"
            ]:

                score += 5

            if macd_state[
                "macd_cross_zero"
            ]:

                score += 10

        # ----------------------------------------------------------------------
        # KD
        # ----------------------------------------------------------------------

        kd = get_kd_state(
            clean
        )

        if kd["k"] > 50:
            score += 5

        if kd["k"] > kd["d"]:
            score += 5

        # ----------------------------------------------------------------------
        # 成交量
        # ----------------------------------------------------------------------

        if (
            volume_ma20
            >= MIN_AVG_VOLUME_20
        ):

            score += 5

        return (
            score,
            {
                "price":
                    price,

                "ma20":
                    ma20,

                "ma60":
                    ma60,

                "ma120":
                    ma120,

                "macd":
                    macd_state,

                "kd":
                    kd,
            },
        )

    except Exception:

        return 0.0, {}


# ==============================================================================
# 📊 多策略綜合評分
# ==============================================================================

def calculate_strategy_score(
    strategy_flags,
    daily_info=None,
    weekly_info=None,
    monthly_info=None,
    m30_info=None,
    m60_info=None,
):

    """
    Pro v2.1 評分。

    S1 = 10
    S2 = 10
    S3 = 15
    S4 = 15
    S5 = 15
    S6 = 15

    ⭐ S7 三週期共振 = 額外 10 分

    技術加分：
        MACD突破0
        KD黃金交叉
        多週期共振
    """

    score = 0

    if strategy_flags.get("s1"):
        score += 10

    if strategy_flags.get("s2"):
        score += 10

    if strategy_flags.get("s3"):
        score += 15

    if strategy_flags.get("s4"):
        score += 15

    if strategy_flags.get("s5"):
        score += 15

    if strategy_flags.get("s6"):
        score += 15

    # ==========================================================================
    # ⭐ 策略7：S3 + S4 + S5
    # ==========================================================================

    if strategy_flags.get("s7"):

        # 三週期共振額外加分
        score += 10

    infos = [
        daily_info,
        weekly_info,
        monthly_info,
        m30_info,
        m60_info,
    ]

    # ==========================================================================
    # MACD突破0
    # ==========================================================================

    for info in infos:

        if not info:
            continue

        macd = info.get(
            "macd"
        )

        if (
            macd
            and
            macd.get(
                "macd_cross_zero"
            )
        ):

            score += 2

    # ==========================================================================
    # KD黃金交叉
    # ==========================================================================

    for info in infos:

        if not info:
            continue

        kd = info.get(
            "kd"
        )

        if (
            kd
            and
            kd.get(
                "golden_cross"
            )
        ):

            score += 2

    # ==========================================================================
    # 多週期 MACD > 0
    # ==========================================================================

    positive_count = 0

    for info in infos:

        if not info:
            continue

        macd = info.get(
            "macd"
        )

        if (
            macd
            and
            macd.get(
                "macd_positive"
            )
        ):

            positive_count += 1

    if positive_count >= 3:
        score += 5

    if positive_count >= 4:
        score += 5

    # ==========================================================================
    # 上限100
    # ==========================================================================

    return min(
        score,
        100,
    )


# ==============================================================================
# 🏆 評級
# ==============================================================================

def get_grade(score):

    if score >= 80:
        return "🔥 S"

    if score >= 65:
        return "⭐ A"

    if score >= 50:
        return "🟢 B"

    if score >= 35:
        return "🟡 C"

    return "⚪ D"


# ==============================================================================
# 📋 股票掃描結果
# ==============================================================================

def scan_stock(
    ticker,
    daily_df,
    weekly_df=None,
    monthly_df=None,
    m30_df=None,
    m60_df=None,
):

    result = {

        "ticker":
            ticker,

        "name":
            DYNAMIC_STOCK_NAMES.get(
                ticker,
                "",
            ),

        "price":
            0.0,

        "s1":
            False,

        "s2":
            False,

        "s3":
            False,

        "s4":
            False,

        "s5":
            False,

        "s6":
            False,

        # ⭐ 新增
        "s7":
            False,

        "daily":
            {},

        "weekly":
            {},

        "monthly":
            {},

        "m30":
            {},

        "m60":
            {},

        # 60m 狀態
        "m60_trigger":
            {},

        "score":
            0,

        "grade":
            "⚪ D",
    }

    try:

        if (
            daily_df is None
            or daily_df.empty
        ):

            return result

        daily_clean = (
            daily_df.dropna(
                subset=[
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                ]
            )
        )

        if len(daily_clean) < 120:

            return result

        result["price"] = safe_float(
            daily_clean[
                "Close"
            ].iloc[-1]
        )

        # ======================================================================
        # 策略3：日K
        # ======================================================================

        s3, _, info3 = (
            check_macd_above_zero_kd(
                daily_clean,
                KD_DAILY_THRESHOLD,
            )
        )

        result["s3"] = s3
        result["daily"] = info3

        # ======================================================================
        # 策略4：週K
        # ======================================================================

        if (
            weekly_df is not None
            and
            not weekly_df.empty
        ):

            s4, _, info4 = (
                check_macd_above_zero_kd(
                    weekly_df,
                    KD_WEEKLY_THRESHOLD,
                )
            )

            result["s4"] = s4
            result["weekly"] = info4

        # ======================================================================
        # 策略5：月K
        # ======================================================================

        if (
            monthly_df is not None
            and
            not monthly_df.empty
        ):

            s5, _, info5 = (
                check_macd_above_zero_kd(
                    monthly_df,
                    KD_MONTHLY_THRESHOLD,
                )
            )

            result["s5"] = s5
            result["monthly"] = info5

        # ======================================================================
        # ⭐ 策略7：S3 + S4 + S5
        # ======================================================================

        result["s7"] = (
            check_strategy_7_resonance(
                result["s3"],
                result["s4"],
                result["s5"],
            )
        )

        # ======================================================================
        # 策略6：低檔爆量
        # ======================================================================

        s6, _, info6 = (
            check_strat_orig_8(
                daily_clean
            )
        )

        result["s6"] = s6

        # ======================================================================
        # 策略1：30m
        # ======================================================================

        if (
            m30_df is not None
            and
            not m30_df.empty
        ):

            s1, _, info1 = (
                check_macd_negative_reducing_kd(
                    m30_df,
                    KD_INTRADAY_THRESHOLD,
                )
            )

            result["s1"] = s1
            result["m30"] = info1

        # ======================================================================
        # 策略2：60m
        # ======================================================================

        if (
            m60_df is not None
            and
            not m60_df.empty
        ):

            s2, _, info2 = (
                check_macd_negative_reducing_kd(
                    m60_df,
                    KD_INTRADAY_THRESHOLD,
                )
            )

            result["s2"] = s2
            result["m60"] = info2

            # --------------------------------------------------------------
            # 🔥 額外60分鐘綠柱 → 紅柱判斷
            # --------------------------------------------------------------

            result[
                "m60_trigger"
            ] = check_60m_macd_transition(
                m60_df
            )

        # ======================================================================
        # 綜合評分
        # ======================================================================

        result["score"] = (
            calculate_strategy_score(

                {
                    "s1":
                        result["s1"],

                    "s2":
                        result["s2"],

                    "s3":
                        result["s3"],

                    "s4":
                        result["s4"],

                    "s5":
                        result["s5"],

                    "s6":
                        result["s6"],

                    "s7":
                        result["s7"],
                },

                daily_info=
                    result["daily"],

                weekly_info=
                    result["weekly"],

                monthly_info=
                    result["monthly"],

                m30_info=
                    result["m30"],

                m60_info=
                    result["m60"],
            )
        )

        result["grade"] = (
            get_grade(
                result["score"]
            )
        )

        return result

    except Exception:

        return result


# ==============================================================================
# 🧹 日K初篩
# ==============================================================================

def build_intraday_scan_pool(
    tickers,
    full_daily,
):

    candidates = []

    print(
        "🔎 開始建立 30m / 60m 精掃名單..."
    )

    for ticker in tickers:

        try:

            df_d = (
                extract_ticker_df(
                    full_daily,
                    ticker,
                )
            )

            if df_d.empty:
                continue

            clean = df_d.dropna(
                subset=[
                    "Close",
                    "High",
                    "Low",
                    "Open",
                    "Volume",
                ]
            )

            if len(clean) < 120:
                continue

            volume_ma20 = (
                clean["Volume"]
                .astype(float)
                .rolling(20)
                .mean()
                .iloc[-1]
            )

            if (
                volume_ma20
                < MIN_AVG_VOLUME_20
            ):

                continue

            strength, info = (
                calculate_daily_strength(
                    clean
                )
            )

            price = info.get(
                "price",
                0,
            )

            ma20 = info.get(
                "ma20",
                0,
            )

            if price <= ma20:
                continue

            candidates.append(
                (
                    ticker,
                    strength,
                )
            )

        except Exception:

            continue

    candidates.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    selected = [
        ticker
        for ticker, _
        in candidates[
            :INTRADAY_SCAN_LIMIT
        ]
    ]

    print(
        f"✅ 日K初篩："
        f"{len(candidates)} 檔"
    )

    print(
        f"🎯 30m/60m 精掃："
        f"{len(selected)} 檔"
    )

    if candidates:

        print(
            "🏆 Top 10："
        )

        for rank, (
            ticker,
            strength,
        ) in enumerate(
            candidates[:10],
            1,
        ):

            print(
                f"   {rank:02d}. "
                f"{ticker} "
                f"Strength={strength:.1f}"
            )

    return selected


# ==============================================================================
# 💬 Telegram
# ==============================================================================

def send_telegram_message(
    message,
    max_length=3500,
):

    bot_token = os.environ.get(
        "TG_BOT_TOKEN"
    )

    chat_id = os.environ.get(
        "TG_CHAT_ID"
    )

    if (
        not bot_token
        or not chat_id
    ):

        print(
            "❌ 未設定 "
            "TG_BOT_TOKEN / TG_CHAT_ID"
        )

        return False

    url = (
        "https://api.telegram.org/"
        f"bot{str(bot_token).strip()}/"
        "sendMessage"
    )

    lines = message.split("\n")

    chunks = []

    current = ""

    for line in lines:

        candidate = (
            current
            + line
            + "\n"
        )

        if len(candidate) > max_length:

            if current:

                chunks.append(
                    current
                )

            current = (
                line
                + "\n"
            )

        else:

            current = candidate

    if current:

        chunks.append(
            current
        )

    success = True

    for idx, chunk in enumerate(
        chunks,
        1,
    ):

        payload = {

            "chat_id":
                str(
                    chat_id
                ).strip(),

            "text":
                chunk.strip(),

            "parse_mode":
                "HTML",

            "disable_web_page_preview":
                True,
        }

        try:

            response = requests.post(
                url,
                json=payload,
                timeout=15,
            )

            data = response.json()

            if (
                response.status_code == 200
                and
                data.get("ok")
            ):

                print(
                    f"✅ Telegram "
                    f"{idx}/{len(chunks)} "
                    f"發送成功"
                )

            else:

                success = False

                print(
                    f"❌ Telegram 發送失敗："
                    f"{data}"
                )

        except Exception as e:

            success = False

            print(
                f"❌ Telegram 連線失敗："
                f"{e}"
            )

        time.sleep(0.5)

    return success


# ==============================================================================
# 📝 Telegram 報告
# ==============================================================================

def build_telegram_report(
    results,
    scan_count,
    elapsed,
    now_tw,
):

    active_results = [
        r
        for r in results
        if any(
            r.get(
                f"s{i}",
                False,
            )
            for i in range(1, 8)
        )
    ]

    active_results.sort(
        key=lambda x: (
            x["score"],
            x.get("s7", False),
        ),
        reverse=True,
    )

    # ==========================================================================
    # ⭐ S7 三週期共振
    # ==========================================================================

    resonance_results = [
        r
        for r in results
        if r.get(
            "s7",
            False,
        )
    ]

    resonance_results.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    # ==========================================================================
    # 🔥 S7 + 60m 綠→紅
    # ==========================================================================

    strongest_results = []

    for r in resonance_results:

        trigger = r.get(
            "m60_trigger",
            {}
        )

        if trigger.get(
            "green_to_red",
            False,
        ):

            strongest_results.append(
                r
            )

    # ==========================================================================
    # 🟢 S7 + 60m 綠柱縮小
    # ==========================================================================

    resonance_shrinking = []

    for r in resonance_results:

        trigger = r.get(
            "m60_trigger",
            {}
        )

        if (
            trigger.get(
                "green_shrinking",
                False,
            )
            and
            not trigger.get(
                "green_to_red",
                False,
            )
        ):

            resonance_shrinking.append(
                r
            )

    lines = []

    lines.append(
        "🇹🇼 "
        "<b>【台股 7 大策略選股 Pro v2.1】</b>"
    )

    lines.append(
        f"⚙️ Version："
        f"<code>{VERSION}</code>"
    )

    lines.append(
        "⚠️ 已過濾 20日均量 < 1000張"
    )

    lines.append(
        f"⏰ {now_tw}"
    )

    lines.append(
        "──────────────────"
    )

    lines.append(
        f"🔎 全市場掃描："
        f"<b>{scan_count}</b> 檔"
    )

    lines.append(
        f"🎯 多策略命中："
        f"<b>{len(active_results)}</b> 檔"
    )

    lines.append(
        f"⭐ S3+S4+S5共振："
        f"<b>{len(resonance_results)}</b> 檔"
    )

    lines.append(
        f"🔥 共振＋60分綠→紅："
        f"<b>{len(strongest_results)}</b> 檔"
    )

    lines.append(
        f"⏱ 耗時："
        f"<b>{elapsed:.1f}</b> 秒"
    )

    # ==========================================================================
    # 🔥 最強
    # ==========================================================================

    lines.append("")

    lines.append(
        "🔥 <b>【最強訊號】"
        "S3＋S4＋S5＋60分綠→紅</b>"
    )

    if not strongest_results:

        lines.append(
            "↳ 今日無符合標的。"
        )

    else:

        for rank, result in enumerate(
            strongest_results[:30],
            1,
        ):

            label = get_stock_label(
                result["ticker"]
            )

            trigger = result[
                "m60_trigger"
            ]

            lines.append(
                f"{rank:02d}. 🔥 "
                f"{label} "
                f"[{result['price']:.2f}] "
                f"<b>{result['score']}分</b>"
            )

            lines.append(
                "    ↳ "
                "日＋週＋月共振 "
                "｜60分綠柱→紅柱 "
                f"Hist={trigger.get('hist', 0):.4f}"
            )

    # ==========================================================================
    # 🟢 共振＋綠柱縮小
    # ==========================================================================

    lines.append("")

    lines.append(
        "🟢 <b>【S3＋S4＋S5共振】"
        "＋60分綠柱縮小</b>"
    )

    if not resonance_shrinking:

        lines.append(
            "↳ 今日無符合標的。"
        )

    else:

        for rank, result in enumerate(
            resonance_shrinking[:30],
            1,
        ):

            label = get_stock_label(
                result["ticker"]
            )

            trigger = result[
                "m60_trigger"
            ]

            lines.append(
                f"{rank:02d}. 🟢 "
                f"{label} "
                f"[{result['price']:.2f}] "
                f"<b>{result['score']}分</b>"
            )

            lines.append(
                "    ↳ "
                "日＋週＋月共振 "
                "｜60分綠柱縮小 "
                f"Hist={trigger.get('hist', 0):.4f}"
            )

    # ==========================================================================
    # ⭐ 策略7
    # ==========================================================================

    lines.append("")

    lines.append(
        "⭐ <b>【策略7】"
        "日K＋週K＋月K三週期共振</b>"
    )

    if not resonance_results:

        lines.append(
            "↳ 今日無符合標的。"
        )

    else:

        values = []

        for result in resonance_results[:30]:

            label = get_stock_label(
                result["ticker"]
            )

            values.append(
                f"{label}"
                f"[{result['price']:.2f}]"
                f"({result['score']}分)"
            )

        lines.append(
            "↳ "
            + "、".join(values)
        )

    # ==========================================================================
    # 🏆 綜合排名
    # ==========================================================================

    lines.append("")

    lines.append(
        "🏆 <b>【多策略綜合排名】</b>"
    )

    if not active_results:

        lines.append(
            "↳ 今日無符合標的。 💤"
        )

    else:

        for rank, result in enumerate(
            active_results[:30],
            1,
        ):

            ticker = result[
                "ticker"
            ]

            label = get_stock_label(
                ticker
            )

            score = result[
                "score"
            ]

            grade = result[
                "grade"
            ]

            price = result[
                "price"
            ]

            strategies = []

            for i in range(1, 8):

                if result.get(
                    f"s{i}",
                    False,
                ):

                    strategies.append(
                        f"S{i}"
                    )

            strategy_text = (
                " ".join(
                    strategies
                )
            )

            lines.append(
                f"{rank:02d}. "
                f"{grade} "
                f"{label} "
                f"[{price:.2f}] "
                f"<b>{score}分</b>"
            )

            lines.append(
                f"    ↳ "
                f"{strategy_text}"
            )

    # ==========================================================================
    # 📈 七大策略
    # ==========================================================================

    lines.append("")

    strategy_titles = {

        1:
            "30分K MACD負值減少 + KD > 20",

        2:
            "60分K MACD負值減少 + KD > 20",

        3:
            "日K MACD > 0 + KD > 20",

        4:
            "週K MACD > 0 + KD > 50",

        5:
            "月K MACD > 0 + KD > 50",

        6:
            "低檔爆量股",

        7:
            "⭐ 日K＋週K＋月K三週期共振",
    }

    icons = {

        1: "📈",
        2: "📊",
        3: "📈",
        4: "📊",
        5: "🌕",
        6: "💥",
        7: "⭐",
    }

    for strategy_no in range(
        1,
        8,
    ):

        matched = [
            r
            for r in results
            if r.get(
                f"s{strategy_no}",
                False,
            )
        ]

        matched.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        lines.append(
            f"{icons[strategy_no]} "
            f"<b>【策略{strategy_no}】"
            f"{strategy_titles[strategy_no]}"
            f"</b>"
        )

        if not matched:

            lines.append(
                "↳ 今日無符合標的。 💤"
            )

        else:

            values = []

            for result in matched[:30]:

                label = get_stock_label(
                    result["ticker"]
                )

                values.append(
                    f"{label}"
                    f"[{result['price']:.2f}]"
                    f"({result['score']}分)"
                )

            lines.append(
                "↳ "
                + "、".join(values)
            )

        lines.append("")

    # ==========================================================================
    # 📌 判讀方式
    # ==========================================================================

    lines.append(
        "──────────────────"
    )

    lines.append(
        "💡 <b>Pro v2.1 判讀方式</b>"
    )

    lines.append(
        "• S7 = S3＋S4＋S5 同時成立"
    )

    lines.append(
        "• S7 代表日／週／月三週期多方共振"
    )

    lines.append(
        "• ⭐ S7後再觀察60分MACD"
    )

    lines.append(
        "• 🟢 綠柱縮小＝準備發動"
    )

    lines.append(
        "• 🔥 綠柱→紅柱＝最強觸發"
    )

    lines.append(
        "• MACD突破0軸會額外加分"
    )

    lines.append(
        "• KD黃金交叉會額外加分"
    )

    lines.append(
        "• S7三週期共振額外＋10分"
    )

    return "\n".join(lines)


# ==============================================================================
# 🚀 MAIN
# ==============================================================================

def main():

    start_time = time.time()

    now_tw = (
        pd.Timestamp.now(
            tz="UTC"
        )
        .tz_convert(
            "Asia/Taipei"
        )
    )

    now_tw_str = now_tw.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    print("")
    print("=" * 70)

    print(
        "🇹🇼 台股 7 大策略選股 Pro v2.1"
    )

    print("=" * 70)

    print(
        f"⏰ {now_tw_str}"
    )

    print("")

    # ==========================================================================
    # STEP 1
    # ==========================================================================

    print(
        "⏳ STEP 1："
        "取得台股全市場股票清單"
    )

    tickers = (
        fetch_all_taiwan_market_tickers()
    )

    if not tickers:

        print(
            "❌ 無法取得股票清單，程式結束"
        )

        return

    # ==========================================================================
    # STEP 2
    # ==========================================================================

    print("")
    print(
        "⏳ STEP 2："
        "下載日K / 週K / 月K"
    )

    full_daily = safe_download_yf(
        tickers,
        period="1y",
        interval="1d",
        chunk_size=DAILY_CHUNK_SIZE,
    )

    if full_daily.empty:

        print(
            "❌ 日K資料下載失敗"
        )

        return

    full_weekly = safe_download_yf(
        tickers,
        period="2y",
        interval="1wk",
        chunk_size=DAILY_CHUNK_SIZE,
    )

    full_monthly = safe_download_yf(
        tickers,
        period="5y",
        interval="1mo",
        chunk_size=DAILY_CHUNK_SIZE,
    )

    # ==========================================================================
    # STEP 3
    # 日K初篩
    # ==========================================================================

    print("")
    print(
        "⏳ STEP 3："
        "日K流動性 + 技術強度初篩"
    )

    intraday_pool = (
        build_intraday_scan_pool(
            tickers,
            full_daily,
        )
    )

    # ==========================================================================
    # STEP 4
    # 下載30m / 60m
    # ==========================================================================

    full_30m = pd.DataFrame()
    full_60m = pd.DataFrame()

    if intraday_pool:

        print("")

        print(
            f"⏳ STEP 4："
            f"下載 {len(intraday_pool)} 檔 "
            f"30m / 60m"
        )

        full_30m = safe_download_yf(
            intraday_pool,
            period="1mo",
            interval="30m",
            chunk_size=INTRADAY_CHUNK_SIZE,
        )

        full_60m = safe_download_yf(
            intraday_pool,
            period="1mo",
            interval="60m",
            chunk_size=INTRADAY_CHUNK_SIZE,
        )

    # ==========================================================================
    # STEP 5
    # 全策略掃描
    # ==========================================================================

    print("")
    print(
        "⏳ STEP 5："
        "執行 7 大策略"
    )

    results = []

    strategy_counter = {

        "s1": 0,
        "s2": 0,
        "s3": 0,
        "s4": 0,
        "s5": 0,
        "s6": 0,
        "s7": 0,
    }

    for index, ticker in enumerate(
        tickers,
        1,
    ):

        try:

            # ------------------------------------------------------------------
            # 日K
            # ------------------------------------------------------------------

            df_daily = (
                extract_ticker_df(
                    full_daily,
                    ticker,
                )
            )

            if df_daily.empty:
                continue

            clean_daily = (
                df_daily.dropna(
                    subset=[
                        "Close",
                        "High",
                        "Low",
                        "Open",
                        "Volume",
                    ]
                )
            )

            if len(clean_daily) < 120:
                continue

            # ------------------------------------------------------------------
            # 流動性
            # ------------------------------------------------------------------

            volume_ma20 = (
                clean_daily[
                    "Volume"
                ]
                .astype(float)
                .rolling(20)
                .mean()
                .iloc[-1]
            )

            if (
                volume_ma20
                < MIN_AVG_VOLUME_20
            ):

                continue

            # ------------------------------------------------------------------
            # 週K
            # ------------------------------------------------------------------

            df_weekly = (
                extract_ticker_df(
                    full_weekly,
                    ticker,
                )
            )

            # ------------------------------------------------------------------
            # 月K
            # ------------------------------------------------------------------

            df_monthly = (
                extract_ticker_df(
                    full_monthly,
                    ticker,
                )
            )

            # ------------------------------------------------------------------
            # 30m
            # ------------------------------------------------------------------

            df_30m = pd.DataFrame()

            if ticker in intraday_pool:

                df_30m = (
                    extract_ticker_df(
                        full_30m,
                        ticker,
                    )
                )

            # ------------------------------------------------------------------
            # 60m
            # ------------------------------------------------------------------

            df_60m = pd.DataFrame()

            if ticker in intraday_pool:

                df_60m = (
                    extract_ticker_df(
                        full_60m,
                        ticker,
                    )
                )

            # ------------------------------------------------------------------
            # 掃描
            # ------------------------------------------------------------------

            result = scan_stock(

                ticker=ticker,

                daily_df=clean_daily,

                weekly_df=df_weekly,

                monthly_df=df_monthly,

                m30_df=df_30m,

                m60_df=df_60m,
            )

            # ------------------------------------------------------------------
            # 統計
            # ------------------------------------------------------------------

            for key in strategy_counter:

                if result.get(
                    key,
                    False,
                ):

                    strategy_counter[
                        key
                    ] += 1

            # ------------------------------------------------------------------
            # 至少命中一個策略
            # ------------------------------------------------------------------

            if any(
                result.get(
                    f"s{i}",
                    False,
                )
                for i in range(
                    1,
                    8,
                )
            ):

                results.append(
                    result
                )

        except Exception as e:

            print(
                f"⚠️ {ticker} "
                f"掃描異常：{e}"
            )

            continue

        # ----------------------------------------------------------------------
        # 每100檔顯示一次進度
        # ----------------------------------------------------------------------

        if index % 100 == 0:

            print(
                f"   掃描進度 "
                f"{index}/{len(tickers)}"
            )

    # ==========================================================================
    # STEP 6
    # 排序
    # ==========================================================================

    results.sort(
        key=lambda x: (
            x["score"],
            x.get(
                "s7",
                False,
            ),
            x["price"],
        ),
        reverse=True,
    )

    # ==========================================================================
    # 統計
    # ==========================================================================

    elapsed = (
        time.time()
        - start_time
    )

    print("")
    print("=" * 70)

    print(
        "🏁 掃描完成"
    )

    print("=" * 70)

    print(
        f"📊 全市場："
        f"{len(tickers)}"
    )

    print(
        f"🎯 30m/60m精掃："
        f"{len(intraday_pool)}"
    )

    print(
        f"🔥 多策略命中："
        f"{len(results)}"
    )

    print("")

    for key, count in (
        strategy_counter.items()
    ):

        print(
            f"{key.upper()}："
            f"{count}"
        )

    # ==========================================================================
    # ⭐ S7 統計
    # ==========================================================================

    resonance_count = (
        strategy_counter["s7"]
    )

    strongest_count = 0
    shrinking_count = 0

    for result in results:

        if not result.get(
            "s7",
            False,
        ):
            continue

        trigger = result.get(
            "m60_trigger",
            {}
        )

        if trigger.get(
            "green_to_red",
            False,
        ):

            strongest_count += 1

        elif trigger.get(
            "green_shrinking",
            False,
        ):

            shrinking_count += 1

    print("")
    print(
        "⭐ S7三週期共振："
        f"{resonance_count}"
    )

    print(
        "🟢 S7＋60分綠柱縮小："
        f"{shrinking_count}"
    )

    print(
        "🔥 S7＋60分綠→紅："
        f"{strongest_count}"
    )

    print(
        f"⏱ 總耗時："
        f"{elapsed:.1f} 秒"
    )

    # ==========================================================================
    # TOP 20
    # ==========================================================================

    print("")
    print(
        "🏆 TOP 20"
    )

    for rank, result in enumerate(
        results[:20],
        1,
    ):

        strategies = []

        for i in range(
            1,
            8,
        ):

            if result.get(
                f"s{i}",
                False,
            ):

                strategies.append(
                    f"S{i}"
                )

        trigger = result.get(
            "m60_trigger",
            {}
        )

        trigger_status = (
            trigger.get(
                "status",
                "",
            )
        )

        print(
            f"{rank:02d}. "
            f"{get_ticker_code(result['ticker'])} "
            f"{result['name']} "
            f"{result['price']:.2f} "
            f"| {result['score']}分 "
            f"| {' '.join(strategies)} "
            f"| {trigger_status}"
        )

    # ==========================================================================
    # STEP 7
    # Telegram
    # ==========================================================================

    print("")
    print(
        "⏳ STEP 7："
        "建立 Telegram 報告"
    )

    report = build_telegram_report(
        results=results,
        scan_count=len(tickers),
        elapsed=elapsed,
        now_tw=now_tw_str,
    )

    send_telegram_message(
        report
    )

    print("")
    print(
        "✅ 台股 7 大策略選股 "
        "Pro v2.1 完成"
    )


# ==============================================================================
# 程式入口
# ==============================================================================

if __name__ == "__main__":

    main()
