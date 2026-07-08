import pandas as pd
import yfinance as yf
import requests
import os
import time
import random

# ==============================================================================
# 🇹🇼 台股全市場技術面模組 (保持全自動含中文功能)
# ==============================================================================
DYNAMIC_STOCK_NAMES = {}

def fetch_all_taiwan_market_tickers():
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    all_tickers = []
    try:
        url_twse = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
        res = requests.get(url_twse, headers=headers, timeout=10)
        if res.status_code == 200:
            for item in res.json():
                code = item.get("Code", "").strip()
                name = item.get("Name", "").strip()
                # 聚焦電子與核心製造業板塊，防範 yfinance 阻擋 IP
                if code.isdigit() and len(code) == 4:
                    if code.startswith(('23', '24', '30', '32', '34', '35', '36', '37', '61', '62', '64', '80')):
                        ticker_id = f"{code}.TW"
                        all_tickers.append(ticker_id)
                        DYNAMIC_STOCK_NAMES[ticker_id] = name
    except Exception:
        pass
    if not all_tickers:
        backup_dict = {"2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科"}
        for k, v in backup_dict.items():
            all_tickers.append(k)
            DYNAMIC_STOCK_NAMES[k] = v
    return sorted(list(set(all_tickers)))

def calculate_macd(close_series, fast=12, slow=26, signal=9):
    fast_ema = close_series.ewm(span=fast, adjust=False).mean()
    slow_ema = close_series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def extract_close_series(df):
    if df.empty: return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.get_level_values(0): return df.xs('Close', axis=1, level=0).squeeze().astype(float)
        if 'Close' in df.columns.get_level_values(1): return df.xs('Close', axis=1, level=1).squeeze().astype(float)
    for col in df.columns:
        if str(col).strip().lower() == 'close': return df[col].squeeze().astype(float)
    return pd.Series(dtype=float)

def check_technical_resonance(ticker):
    """
    策略一：原版多週期三頻共振
    """
    try:
        df_60m = yf.download(ticker, period="1mo", interval="60m", progress=False)
        df_daily = yf.download(ticker, period="1y", interval="1d", progress=False)
        df_weekly = yf.download(ticker, period="2y", interval="1wk", progress=False)
        
        c_60m = extract_close_series(df_60m)
        c_daily = extract_close_series(df_daily)
        c_weekly = extract_close_series(df_weekly)
        
        if c_60m.empty or c_daily.empty or c_weekly.empty: return False

        w_macd, w_signal, w_hist = calculate_macd(c_weekly)
        d_macd, d_signal, d_hist = calculate_macd(c_daily)
        d_ma = c_daily.rolling(window=20).mean()
        m60_macd, m60_signal, m60_hist = calculate_macd(c_60m)

        if len(w_hist) < 1 or len(d_hist) < 1 or len(m60_hist) < 2: return False

        w_m, w_s, w_h = float(w_macd.iloc[-1]), float(w_signal.iloc[-1]), float(w_hist.iloc[-1])
        d_m, d_s, d_c, d_ma_val = float(d_macd.iloc[-1]), float(d_signal.iloc[-1]), float(c_daily.iloc[-1]), float(d_ma.iloc[-1])
        m60_m, m60_h, m60_h_prev = float(m60_macd.iloc[-1]), float(m60_hist.iloc[-1]), float(m60_hist.iloc[-2])

        weekly_bullish = (w_m > w_s) and (w_h > 0)
        daily_bullish = (d_m > 0) and (d_m > d_s)
        daily_above_ma = (d_c > d_ma_val)
        m60_cross_up = (m60_m > 0) and (m60_h > 0) and (m60_h_prev <= 0)

        if weekly_bullish and daily_bullish and daily_above_ma and m60_cross_up:
            return True
    except Exception:
        pass
    return False

def check_ma_tangling(ticker):
    """
    策略二：日K層級 5MA, 10MA, 20MA 均線糾結
    """
    try:
        # 只需要獲取日K線數據
        df_daily = yf.download(ticker, period="3mo", interval="1d", progress=False)
        c_daily = extract_close_series(df_daily)
        
        if c_daily.empty or len(c_daily) < 20: return False
        
        # 計算 5, 10, 20 日均線
        ma5 = c_daily.rolling(window=5).mean().iloc[-1]
        ma10 = c_daily.rolling(window=10).mean().iloc[-1]
        ma20 = c_daily.rolling(window=20).mean().iloc[-1]
        close_today = c_daily.iloc[-1]
        
        # 找出三條線的最高與最低值
        max_ma = max(ma5, ma10, ma20)
        min_ma = min(ma5, ma10, ma20)
        
        # 計算糾結區間帶（最大與最小均線差值佔 20MA 的比例）
        # 0.03 代表均線在 3% 寬度內，算非常糾結
        tangle_ratio = (max_ma - min_ma) / ma20
        
        # 條件 1: 均線糾結度小於 3%
        # 條件 2: 今日價格在 20MA 之上（代表偏多整理或準備向上突破）
        if tangle_ratio < 0.03 and close_today > ma20:
            return True
    except Exception:
        pass
    return False

# ==============================================================================
# 💬 Telegram 發送 (全 HTML 解析模式)
# ==============================================================================
def send_telegram_message(message):
    bot_token = os.environ.get("TG_BOT_TOKEN")
    chat_id = os.environ.get("TG_CHAT_ID")
    if not bot_token or not chat_id: return
    
    bot_token = str(bot_token).strip()
    chat_id = str(chat_id).strip()
    if bot_token.lower().startswith("bot"): bot_token = bot_token[3:]

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    try:
        res = requests.post(url, json=payload, timeout=10)
        print(f"📢 TG 發送反饋: 狀態碼 {res.status_code} | 內容: {res.text}")
    except Exception as e: 
        print(f"❌ Telegram 發送異常: {e}")

# ==============================================================================
# 🚀 主程式（台股多策略選股專用版）
# ==============================================================================
if __name__ == "__main__":
    now_tw = pd.Timestamp.now(tz='UTC').tz_convert('Asia/Taipei')
    tw_time_str = now_tw.strftime('%Y-%m-%d %H:%M:%S')

    print("🚀 啟動【台股盤後多策略篩選報告】...")
    
    # 獲取篩選後的台股核心標的名單
    tech_scan_pool = fetch_all_taiwan_market_tickers()
    
    strat1_matches = []
    strat2_matches = []

    print(f"⏳ 正在進行台股技術面安全分批掃描 (共 {len(tech_scan_pool)} 檔)...")
    for idx, ticker in enumerate(tech_scan_pool, 1):
        # 每 15 檔稍微隨機暫停，避免頻率過高被 API 擋 IP
        if idx % 15 == 0: 
            time.sleep(random.uniform(2.0, 3.5))
            
        name_zh = DYNAMIC_STOCK_NAMES.get(ticker, "")
        stock_label = f"<code>{ticker}</code>(<i>{name_zh}</i>)" if name_zh else f"<code>{ticker}</code>"

        # 檢測策略一：多週期三頻共振
        if check_technical_resonance(ticker):
            strat1_matches.append(stock_label)
            
        # 檢測策略二：日 K 均線糾結 (5MA/10MA/20MA)
        if check_ma_tangling(ticker):
            strat2_matches.append(stock_label)

    # 📝 建立精簡版台股獨立美化訊息 (全 HTML 語法)
    tw_msg = f"🇹🇼 <b>【台股市場：多策略選股報告】</b>\n⏰ 報告時間: {tw_time_str}\n"
    tw_msg += "───────────────────\n\n"
    
    tw_msg += "📈 <b>【策略一】原版多週期三頻共振</b>\n"
    tw_msg += "↳ " + (", ".join(strat1_matches) if strat1_matches else "今日無符合標的。 💤") + "\n\n"

    tw_msg += "🌀 <b>【策略二】日K短期均線糾結 (5MA/10MA/20MA)</b>\n"
    tw_msg += "↳ " + (", ".join(strat2_matches) if strat2_matches else "今日無符合標的。 💤") + "\n"

    # 發送 Telegram
    send_telegram_message(tw_msg)
    print("✅ 台股多策略報告發送完畢！")
