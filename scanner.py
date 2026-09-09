# scanner.py
# ==============================================================================
# 🇹🇼 台股 6 大策略選股 Pro v2
#
# 功能：
#   1. TWSE 全市場股票清單
#   2. 日K / 週K / 月K 多週期掃描
#   3. 30m / 60m 短週期精掃
#   4. MACD + KD
#   5. 低檔爆量策略
#   6. 技術強度排名
#   7. 多策略交叉評分
#   8. Telegram HTML 報告
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

VERSION = "Pro v2.0"

TWSE_API_URL = (
    "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
    )
}

# 最低 20 日平均成交量
MIN_AVG_VOLUME_20 = 1_000_000

# 30m / 60m 精掃最大股票數
INTRADAY_SCAN_LIMIT = 100

# Yahoo 分批下載數量
DAILY_CHUNK_SIZE = 150
INTRADAY_CHUNK_SIZE = 50

# 是否只掃普通股票
ONLY_COMMON_STOCK = True

# 策略門檻
KD_DAILY_THRESHOLD = 20
KD_WEEKLY_THRESHOLD = 50
KD_MONTHLY_THRESHOLD = 50
KD_INTRADAY_THRESHOLD = 20


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
    return ticker.replace(".TW", "").replace(".TWO", "")


def escape_html(text):
    """Telegram HTML escape"""
    return html.escape(str(text))


def get_stock_label(ticker):
    """建立 Telegram 股票標籤"""
    code = get_ticker_code(ticker)
    name = DYNAMIC_STOCK_NAMES.get(ticker, "")

    if name:
        return f"<code>{code}</code> <i>{escape_html(name)}</i>"

    return f"<code>{code}</code>"


# ==============================================================================
# 🇹🇼 TWSE 全市場股票清單
# ==============================================================================

def fetch_all_taiwan_market_tickers():
    """
    下載 TWSE 股票清單。

    只保留：
        4 位數字股票代碼
        .TW
    """

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

            code = str(item.get("Code", "")).strip()
            name = str(item.get("Name", "")).strip()

            if not code.isdigit():
                continue

            if len(code) != 4:
                continue

            ticker = f"{code}.TW"

            all_tickers.append(ticker)

            DYNAMIC_STOCK_NAMES[ticker] = name

        all_tickers = sorted(set(all_tickers))

        print(
            f"✅ TWSE 股票清單取得完成：{len(all_tickers)} 檔"
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
    """
    分批下載 Yahoo Finance。

    重要：
        不讓單一批次失敗造成整個程式死亡。
    """

    if not tickers:
        return pd.DataFrame()

    all_dfs = []

    total_chunks = (
        len(tickers) + chunk_size - 1
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

                if df is not None and not df.empty:

                    all_dfs.append(df)

                    success = True

                    break

                print(
                    f"⚠️ 批次 {chunk_no} 回傳空資料"
                )

            except Exception as e:

                print(
                    f"⚠️ 批次 {chunk_no} "
                    f"下載失敗：{e}"
                )

            if attempt < max_retry:
                time.sleep(2 * attempt)

        if not success:
            print(
                f"❌ 批次 {chunk_no}/{total_chunks} "
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

def extract_ticker_df(full_df, ticker):
    """
    從 yfinance MultiIndex DataFrame
    安全取得單一股票資料。

    相容：
        MultiIndex columns
        單一股票 columns
    """

    if full_df is None or full_df.empty:
        return pd.DataFrame()

    try:

        if isinstance(
            full_df.columns,
            pd.MultiIndex,
        ):

            # 常見格式：
            # Price / Ticker

            levels = full_df.columns.names

            # 嘗試找 ticker 所在 level
            for level_no in range(
                full_df.columns.nlevels
            ):

                try:

                    values = (
                        full_df
                        .columns
                        .get_level_values(level_no)
                    )

                    if ticker in values:

                        result = full_df.xs(
                            ticker,
                            axis=1,
                            level=level_no,
                        )

                        return result.copy()

                except Exception:
                    continue

            return pd.DataFrame()

        # 非 MultiIndex
        return full_df.copy()

    except Exception:
        return pd.DataFrame()


# ==============================================================================
# 📈 MACD
# ==============================================================================

def calculate_macd(
    close_series,
    fast=12,
    slow=26,
    signal=9,
):
    """
    標準 EMA MACD
    """

    close = (
        pd.to_numeric(
            close_series,
            errors="coerce",
        )
        .dropna()
        .astype(float)
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
        fast_ema - slow_ema
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
        macd_line - signal_line
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
    """
    台股常用 RSV / K / D。

    K0 = 50
    D0 = 50
    """

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
        highest - lowest
    )

    rsv = (
        (df["Close"] - lowest)
        / denominator.replace(
            0,
            pd.NA,
        )
        * 100
    )

    # 第一個不足 n 的值不參與
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
    """
    回傳：
        K
        D
        K>D
        KD黃金交叉
    """

    k, d = calculate_kd(df)

    if len(k) < 2 or len(d) < 2:
        return {
            "k": 0.0,
            "d": 0.0,
            "above": False,
            "golden_cross": False,
        }

    k_now = safe_float(k.iloc[-1])
    d_now = safe_float(d.iloc[-1])

    k_prev = safe_float(k.iloc[-2])
    d_prev = safe_float(d.iloc[-2])

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
    """
    取得 MACD 完整狀態。
    """

    if (
        df is None
        or df.empty
        or "Close" not in df.columns
    ):
        return None

    close = pd.to_numeric(
        df["Close"],
        errors="coerce",
    ).dropna()

    if len(close) < 35:
        return None

    macd, signal, hist = calculate_macd(
        close
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
        "macd_prev": macd_prev,
        "signal": signal_now,
        "signal_prev": signal_prev,
        "hist": hist_now,
        "hist_prev": hist_prev,
        "hist_prev2": hist_prev2,

        "macd_positive":
            macd_now > 0,

        "macd_cross_zero":
            macd_prev <= 0
            and macd_now > 0,

        "macd_rising":
            macd_now > macd_prev,

        "hist_rising":
            hist_now > hist_prev,

        "hist_rising_2":
            hist_now > hist_prev
            and hist_prev > hist_prev2,

        "golden_cross":
            signal_prev >= macd_prev
            and macd_now > signal_now,

        "hist_negative_reducing":
            hist_now < 0
            and hist_now > hist_prev,
    }


# ==============================================================================
# 📈 策略 1 / 2
# MACD 負值減少 + KD
# ==============================================================================

def check_macd_negative_reducing_kd(
    df_tf,
    kd_threshold=20,
):
    """
    原始策略：

        MACD 負值減少
        +
        KD > threshold

    Pro：
        增加 MACD 線連續改善判斷，
        但保留原本「Histogram 或 MACD」邏輯。
    """

    try:

        if df_tf is None or df_tf.empty:
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

        macd_state = get_macd_state(
            clean
        )

        if macd_state is None:
            return False, 0.0, {}

        kd_state = get_kd_state(
            clean
        )

        macd_condition = (
            (
                macd_state[
                    "hist_negative_reducing"
                ]
            )
            or
            (
                macd_state["macd"] < 0
                and macd_state[
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
            clean["Close"].iloc[-1]
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
    """
    原始條件：

        MACD > 0
        +
        K/D > threshold

    Pro 同時記錄：

        MACD剛突破0
        MACD上升
        Histogram上升
        KD黃金交叉
    """

    try:

        if df_tf is None or df_tf.empty:
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

        macd_state = get_macd_state(
            clean
        )

        if macd_state is None:
            return False, 0.0, {}

        kd_state = get_kd_state(
            clean
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
            clean["Close"].iloc[-1]
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
# 💥 策略 6
# 低檔爆量
# ==============================================================================

def check_strat_orig_8(
    df_daily,
):
    """
    低檔爆量：

        120日價格位置 <= 30%
        成交量 >= 前5日均量 2.5倍
        紅K
    """

    try:

        if df_daily is None:
            return False, 0.0, {}

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
            return False, 0.0, {}

        close = clean["Close"].astype(float)
        high = clean["High"].astype(float)
        low = clean["Low"].astype(float)
        open_price = clean["Open"].astype(float)
        volume = clean["Volume"].astype(float)

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
            or pd.isna(high_120)
            or high_120 <= low_120
        ):
            return False, 0.0, {}

        current_price = (
            close.iloc[-1]
        )

        position = (
            current_price - low_120
        ) / (
            high_120 - low_120
        )

        # 前5日均量
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
            and is_volume_surge
            and is_red_k
        )

        return (
            result,
            current_price,
            {
                "position": position,
                "volume_ratio": volume_ratio,
                "red_k": is_red_k,
            },
        )

    except Exception:
        return False, 0.0, {}


def current_volume_ratio(
    volume,
    previous_ma5,
):
    if previous_ma5 <= 0:
        return 0.0

    return (
        safe_float(volume.iloc[-1])
        / previous_ma5
    )


# ==============================================================================
# 📊 日K強度評分
# ==============================================================================

def calculate_daily_strength(
    df_daily,
):
    """
    日K初篩分數。

    用於決定哪些股票值得進入
    30m / 60m 精掃。

    不直接取前50檔。
    """

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

        close = clean["Close"].astype(float)
        volume = clean["Volume"].astype(float)

        price = close.iloc[-1]

        ma5 = (
            close.rolling(5)
            .mean()
            .iloc[-1]
        )

        ma20 = (
            close.rolling(20)
            .mean()
            .iloc[-1]
        )

        ma60 = (
            close.rolling(60)
            .mean()
            .iloc[-1]
        )

        ma120 = (
            close.rolling(120)
            .mean()
            .iloc[-1]
        )

        volume_ma20 = (
            volume.rolling(20)
            .mean()
            .iloc[-1]
        )

        score = 0.0

        # ------------------------------------------------------------------
        # 價格 > 均線
        # ------------------------------------------------------------------

        if price > ma5:
            score += 5

        if price > ma20:
            score += 15

        if price > ma60:
            score += 10

        if price > ma120:
            score += 10

        # ------------------------------------------------------------------
        # 均線多頭排列
        # ------------------------------------------------------------------

        if (
            ma5 > ma20
            and ma20 > ma60
        ):
            score += 15

        if (
            ma20 > ma60
            and ma60 > ma120
        ):
            score += 10

        # ------------------------------------------------------------------
        # MACD
        # ------------------------------------------------------------------

        macd_state = get_macd_state(
            clean
        )

        if macd_state:

            if macd_state["macd_positive"]:
                score += 10

            if macd_state["macd_rising"]:
                score += 5

            if macd_state["hist_rising"]:
                score += 5

            if macd_state["macd_cross_zero"]:
                score += 10

        # ------------------------------------------------------------------
        # KD
        # ------------------------------------------------------------------

        kd = get_kd_state(clean)

        if kd["k"] > 50:
            score += 5

        if (
            kd["k"] > kd["d"]
        ):
            score += 5

        # ------------------------------------------------------------------
        # 成交量
        # ------------------------------------------------------------------

        if (
            volume_ma20
            >= MIN_AVG_VOLUME_20
        ):
            score += 5

        return (
            score,
            {
                "price": price,
                "ma20": ma20,
                "ma60": ma60,
                "ma120": ma120,
                "macd": macd_state,
                "kd": kd,
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
    最高 100 分。

    策略本身：
        S1 = 10
        S2 = 10
        S3 = 15
        S4 = 15
        S5 = 15
        S6 = 15

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

    infos = [
        daily_info,
        weekly_info,
        monthly_info,
        m30_info,
        m60_info,
    ]

    # MACD突破0
    for info in infos:

        if not info:
            continue

        macd = info.get("macd")

        if (
            macd
            and macd.get(
                "macd_cross_zero"
            )
        ):
            score += 2

    # KD黃金交叉
    for info in infos:

        if not info:
            continue

        kd = info.get("kd")

        if (
            kd
            and kd.get(
                "golden_cross"
            )
        ):
            score += 2

    # 多週期共振
    positive_count = 0

    for info in infos:

        if not info:
            continue

        macd = info.get("macd")

        if (
            macd
            and macd.get(
                "macd_positive"
            )
        ):
            positive_count += 1

    if positive_count >= 3:
        score += 5

    if positive_count >= 4:
        score += 5

    return min(score, 100)


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
    """
    掃描單一股票。

    回傳完整策略結果。
    """

    result = {
        "ticker": ticker,
        "name": DYNAMIC_STOCK_NAMES.get(
            ticker,
            "",
        ),
        "price": 0.0,

        "s1": False,
        "s2": False,
        "s3": False,
        "s4": False,
        "s5": False,
        "s6": False,

        "daily": {},
        "weekly": {},
        "monthly": {},
        "m30": {},
        "m60": {},

        "score": 0,
        "grade": "⚪ D",
    }

    try:

        if (
            daily_df is None
            or daily_df.empty
        ):
            return result

        daily_clean = daily_df.dropna(
            subset=[
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]
        )

        if len(daily_clean) < 120:
            return result

        result["price"] = safe_float(
            daily_clean[
                "Close"
            ].iloc[-1]
        )

        # ------------------------------------------------------------------
        # 策略 3
        # ------------------------------------------------------------------

        s3, _, info3 = (
            check_macd_above_zero_kd(
                daily_clean,
                KD_DAILY_THRESHOLD,
            )
        )

        result["s3"] = s3
        result["daily"] = info3

        # ------------------------------------------------------------------
        # 策略 4
        # ------------------------------------------------------------------

        if (
            weekly_df is not None
            and not weekly_df.empty
        ):

            s4, _, info4 = (
                check_macd_above_zero_kd(
                    weekly_df,
                    KD_WEEKLY_THRESHOLD,
                )
            )

            result["s4"] = s4
            result["weekly"] = info4

        # ------------------------------------------------------------------
        # 策略 5
        # ------------------------------------------------------------------

        if (
            monthly_df is not None
            and not monthly_df.empty
        ):

            s5, _, info5 = (
                check_macd_above_zero_kd(
                    monthly_df,
                    KD_MONTHLY_THRESHOLD,
                )
            )

            result["s5"] = s5
            result["monthly"] = info5

        # ------------------------------------------------------------------
        # 策略 6
        # ------------------------------------------------------------------

        s6, _, info6 = (
            check_strat_orig_8(
                daily_clean
            )
        )

        result["s6"] = s6

        # ------------------------------------------------------------------
        # 策略 1
        # ------------------------------------------------------------------

        if (
            m30_df is not None
            and not m30_df.empty
        ):

            s1, _, info1 = (
                check_macd_negative_reducing_kd(
                    m30_df,
                    KD_INTRADAY_THRESHOLD,
                )
            )

            result["s1"] = s1
            result["m30"] = info1

        # ------------------------------------------------------------------
        # 策略 2
        # ------------------------------------------------------------------

        if (
            m60_df is not None
            and not m60_df.empty
        ):

            s2, _, info2 = (
                check_macd_negative_reducing_kd(
                    m60_df,
                    KD_INTRADAY_THRESHOLD,
                )
            )

            result["s2"] = s2
            result["m60"] = info2

        # ------------------------------------------------------------------
        # 綜合評分
        # ------------------------------------------------------------------

        result["score"] = (
            calculate_strategy_score(
                {
                    "s1": result["s1"],
                    "s2": result["s2"],
                    "s3": result["s3"],
                    "s4": result["s4"],
                    "s5": result["s5"],
                    "s6": result["s6"],
                },
                daily_info=result["daily"],
                weekly_info=result["weekly"],
                monthly_info=result["monthly"],
                m30_info=result["m30"],
                m60_info=result["m60"],
            )
        )

        result["grade"] = get_grade(
            result["score"]
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
    """
    不再使用：

        heavy_scan_pool[:50]

    改成：

        全市場
        ↓
        流動性
        ↓
        日K技術強度
        ↓
        排名
        ↓
        Top 100
    """

    candidates = []

    print(
        "🔎 開始建立 30m / 60m 精掃名單..."
    )

    for ticker in tickers:

        try:

            df_d = extract_ticker_df(
                full_daily,
                ticker,
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

            # --------------------------------------------------------------
            # 流動性
            # --------------------------------------------------------------

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

            # --------------------------------------------------------------
            # 技術強度
            # --------------------------------------------------------------

            strength, info = (
                calculate_daily_strength(
                    clean
                )
            )

            # 股價站上20MA
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

    # --------------------------------------------------------------
    # 真正依強度排序
    # --------------------------------------------------------------

    candidates.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    selected = [
        ticker
        for ticker, _ in candidates[
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
    """
    Telegram 發送。

    使用：
        TG_BOT_TOKEN
        TG_CHAT_ID
    """

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

        if (
            len(candidate)
            > max_length
        ):

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
            "chat_id": str(
                chat_id
            ).strip(),

            "text": chunk.strip(),

            "parse_mode": "HTML",

            "disable_web_page_preview": True,
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
                and data.get("ok")
            ):

                print(
                    f"✅ Telegram "
                    f"{idx}/{len(chunks)} 發送成功"
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
    """
    建立 Telegram 報告。
    """

    active_results = [
        r
        for r in results
        if any(
            r.get(f"s{i}", False)
            for i in range(1, 7)
        )
    ]

    active_results.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    lines = []

    lines.append(
        "🇹🇼 "
        "<b>【台股 6 大多頭選股 Pro v2】</b>"
    )

    lines.append(
        f"⚙️ Version：<code>{VERSION}</code>"
    )

    lines.append(
        "⚠️ "
        "已過濾 20日均量 < 1000張"
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
        f"⏱ 耗時："
        f"<b>{elapsed:.1f}</b> 秒"
    )

    lines.append("")

    # ----------------------------------------------------------------------
    # 🏆 綜合排名
    # ----------------------------------------------------------------------

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

            ticker = result["ticker"]

            label = get_stock_label(
                ticker
            )

            score = result["score"]

            grade = result["grade"]

            price = result["price"]

            strategies = []

            for i in range(1, 7):

                if result.get(
                    f"s{i}",
                    False,
                ):

                    strategies.append(
                        f"S{i}"
                    )

            strategy_text = " ".join(
                strategies
            )

            lines.append(
                f"{rank:02d}. "
                f"{grade} "
                f"{label} "
                f"[{price:.2f}] "
                f"<b>{score}分</b>"
            )

            lines.append(
                f"    ↳ {strategy_text}"
            )

    lines.append("")

    # ----------------------------------------------------------------------
    # 📈 六大策略
    # ----------------------------------------------------------------------

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
    }

    icons = {
        1: "📈",
        2: "📊",
        3: "📈",
        4: "📊",
        5: "🌕",
        6: "💥",
    }

    for strategy_no in range(1, 7):

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

    # ----------------------------------------------------------------------
    # 📌 最強訊號
    # ----------------------------------------------------------------------

    lines.append(
        "──────────────────"
    )

    lines.append(
        "💡 <b>Pro v2 判讀方式</b>"
    )

    lines.append(
        "• S級 ≥ 80：多策略共振"
    )

    lines.append(
        "• A級 65–79：強勢多頭"
    )

    lines.append(
        "• B級 50–64：偏多"
    )

    lines.append(
        "• 30m/60m：由日K強度排名後精掃"
    )

    lines.append(
        "• MACD突破0軸會額外加分"
    )

    lines.append(
        "• KD黃金交叉會額外加分"
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
        "🇹🇼 台股 6 大策略選股 Pro v2"
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
        "⏳ STEP 1：取得台股全市場股票清單"
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
        "⏳ STEP 2：下載日K / 週K / 月K"
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
        "⏳ STEP 3：日K流動性 + 技術強度初篩"
    )

    intraday_pool = (
        build_intraday_scan_pool(
            tickers,
            full_daily,
        )
    )

    # ==========================================================================
    # STEP 4
    # 下載 30m / 60m
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
        "⏳ STEP 5：執行 6 大策略"
    )

    results = []

    strategy_counter = {
        "s1": 0,
        "s2": 0,
        "s3": 0,
        "s4": 0,
        "s5": 0,
        "s6": 0,
    }

    for index, ticker in enumerate(
        tickers,
        1,
    ):

        try:

            df_daily = (
                extract_ticker_df(
                    full_daily,
                    ticker,
                )
            )

            if df_daily.empty:
                continue

            # --------------------------------------------------------------
            # 流動性
            # --------------------------------------------------------------

            if (
                "Volume"
                not in df_daily.columns
            ):
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

            volume_ma20 = (
                clean_daily["Volume"]
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

            # --------------------------------------------------------------
            # 週K
            # --------------------------------------------------------------

            df_weekly = (
                extract_ticker_df(
                    full_weekly,
                    ticker,
                )
            )

            # --------------------------------------------------------------
            # 月K
            # --------------------------------------------------------------

            df_monthly = (
                extract_ticker_df(
                    full_monthly,
                    ticker,
                )
            )

            # --------------------------------------------------------------
            # 30m
            # --------------------------------------------------------------

            df_30m = pd.DataFrame()

            if ticker in intraday_pool:

                df_30m = (
                    extract_ticker_df(
                        full_30m,
                        ticker,
                    )
                )

            # --------------------------------------------------------------
            # 60m
            # --------------------------------------------------------------

            df_60m = pd.DataFrame()

            if ticker in intraday_pool:

                df_60m = (
                    extract_ticker_df(
                        full_60m,
                        ticker,
                    )
                )

            # --------------------------------------------------------------
            # 掃描
            # --------------------------------------------------------------

            result = scan_stock(
                ticker=ticker,
                daily_df=clean_daily,
                weekly_df=df_weekly,
                monthly_df=df_monthly,
                m30_df=df_30m,
                m60_df=df_60m,
            )

            # --------------------------------------------------------------
            # 統計
            # --------------------------------------------------------------

            for key in strategy_counter:

                if result.get(
                    key,
                    False,
                ):

                    strategy_counter[
                        key
                    ] += 1

            # 只保留至少命中一個策略
            if any(
                result.get(
                    f"s{i}",
                    False,
                )
                for i in range(1, 7)
            ):

                results.append(
                    result
                )

        except Exception as e:

            print(
                f"⚠️ {ticker} 掃描異常：{e}"
            )

            continue

        # 每 100 檔顯示一次進度
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
        f"📊 全市場：{len(tickers)}"
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
            f"{key.upper()}：{count}"
        )

    print(
        f"⏱ 總耗時："
        f"{elapsed:.1f} 秒"
    )

    # ==========================================================================
    # Top 20
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

        for i in range(1, 7):

            if result.get(
                f"s{i}",
                False,
            ):

                strategies.append(
                    f"S{i}"
                )

        print(
            f"{rank:02d}. "
            f"{get_ticker_code(result['ticker'])} "
            f"{result['name']} "
            f""
            f"{result['price']:.2f} "
            f"| {result['score']}分 "
            f"| {' '.join(strategies)}"
        )

    # ==========================================================================
    # STEP 7
    # Telegram
    # ==========================================================================

    print("")
    print(
        "⏳ STEP 7：建立 Telegram 報告"
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
        "✅ 台股 6 大策略選股 Pro v2 完成"
    )


# ==============================================================================
# 程式入口
# ==============================================================================

if __name__ == "__main__":
    main()
