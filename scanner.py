import pandas as pd
import yfinance as yf
import requests
import os
import time
import random

# ==============================================================================
# 🇹🇼 台股全市場快速資料下載模組
# ==============================================================================
DYNAMIC_STOCK_NAMES = {}
FUNDAMENTAL_DATA = {}  
SHARES_OUTSTANDING_DATA = {}  
CHIPS_SUMMARY_DATA = {} # 存放一次性下載的籌碼總表快取

FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "")

def fetch_all_taiwan_market_tickers():
    """ 下載全台股市場代碼與證交所估值資料 """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    all_tickers = []
    
    # 1. 撈取所有股票代碼
    try:
        url_twse = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
        res = requests.get(url_twse, headers=headers, timeout=10)
        if res.status_code == 200:
            for item in res.json():
                code = item.get("Code", "").strip()
                name = item.get("Name", "").strip()
                if code.isdigit() and len(code) == 4:
                    ticker_id = f"{code}.TW"
                    all_tickers.append(ticker_id)
                    DYNAMIC_STOCK_NAMES[ticker_id] = name
    except Exception as e:
        print(f"⚠️ 撈取全市場名單異常: {e}")

    # 2. 同步撈取證交所估值 (PE/PB)
    try:
        url_valuation = "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL"
        res_val = requests.get(url_valuation, headers=headers, timeout=10)
        if res_val.status_code == 200:
            for item in res_val.json():
                code = item.get("Code", "").strip()
                ticker_id = f"{code}.TW"
                try:
                    pe = float(item.get("PEratio", 0)) if item.get("PEratio") else 0.0
                except: pe = 0.0
                try:
                    pb = float(item.get("PBRatio", 0)) if item.get("PBRatio") else 0.0
                except: pb = 0.0
                FUNDAMENTAL_DATA[ticker_id] = {"PE": pe, "PB": pb}
    except Exception as e:
        print(f"⚠️ 撈取證交所估值資料異常: {e}")

    return sorted(list(set(all_tickers)))

def fetch_fast_chips_summary():
    """ 策略五優化：一次性下載大戶持股週總表，避免在迴圈內重複發送 API """
    if not FINMIND_TOKEN: return
    url = "https://api.finmindtrade.com/api/v4/data"
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=14)).strftime('%Y-%m-%d')
    params = {
        "dataset": "taiwan_stock_holding_shares_per",
        "start_date": start_date,
        "token": FINMIND_TOKEN
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200:
            data = res.json().get("data", [])
            df_h = pd.DataFrame(data)
            if not df_h.empty:
                # 篩選出千張大戶等級
                df_1000 = df_h[df_h['HoldersLevel'] == '1,000,001以上'].sort_values(by='date')
                # 依股票代碼群組，比對最後一週是否大於前一週
                for stock_id, group in df_1000.groupby('stock_id'):
                    if len(group) >= 2:
                        if group['percent'].iloc[-1] > group['percent'].iloc[-2]:
                            CHIPS_SUMMARY_DATA[f"{stock_id}.TW"] = True
    except Exception as e:
        print(f"⚠️ 籌碼快取下載異常: {e}")

# ==============================================================================
# 📊 技術指標工具箱
# ==============================================================================
def calculate_macd(close_series, fast=12, slow=26, signal=9):
    fast_ema = close_series.ewm(span=fast, adjust=False).mean()
    slow_ema = close_series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def calculate_kd(df_single, n=9, m1=3, m2=3):
    low_min = df_single['Low'].astype(float).rolling(window=n).min()
    high_max = df_single['High'].astype(float).rolling(window=n).max()
    close = df_single['Close'].astype(float)
    rsv = (((close - low_min) / (high_max - low_min)) * 100).fillna(50)
    k_list, d_list = [50.0], [50.0]
    for i in range(1, len(rsv)):
        k_list.append((k_list[-1] * (m1 - 1) + rsv.iloc[i]) / m1)
        d_list.append((d_list[-1] * (m2 - 1) + k_list[-1]) / m2)
    return pd.Series(k_list, index=df_single.index), pd.Series(d_list, index=df_single.index)

def calculate_rsi(close_series, period=6):
    delta = close_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    return (100 - (100 / (1 + (gain / loss)))).fillna(50)

# ==============================================================================
# 🎯 完整 1 ~ 7 策略檢測邏輯
# ==============================================================================

def check_strat1_resonance(df_60m, df_daily, df_weekly):
    """ 策略一：多週期三頻共振 (MACD) + KD低檔金叉 """
    try:
        c_60m = df_60m['Close'].squeeze().astype(float)
        c_daily = df_daily['Close'].squeeze().astype(float)
        c_weekly = df_weekly['Close'].squeeze().astype(float)
        
        w_macd, w_signal, w_hist = calculate_macd(c_weekly)
        d_macd, d_signal, d_hist = calculate_macd(c_daily)
        d_ma = c_daily.rolling(window=20).mean()
        m60_macd, m60_signal, m60_hist = calculate_macd(c_60m)

        w_m, w_s, w_h = float(w_macd.iloc[-1]), float(w_signal.iloc[-1]), float(w_hist.iloc[-1])
        d_m, d_s, d_c, d_ma_val = float(d_macd.iloc[-1]), float(d_signal.iloc[-1]), float(c_daily.iloc[-1]), float(d_ma.iloc[-1])
        m60_m, m60_h, m60_h_prev = float(m60_macd.iloc[-1]), float(m60_hist.iloc[-1]), float(m60_hist.iloc[-2])

        macd_cond = (w_m > w_s) and (w_h > 0) and (d_m > 0) and (d_m > d_s) and (d_c > d_ma_val) and (m60_m > 0) and (m60_h > 0) and (m60_h_prev <= 0)
        if not macd_cond: return False

        k_60m, d_60m = calculate_kd(df_60m)
        k_daily, d_daily = calculate_kd(df_daily)
        
        def is_low_kd_gold(k_ser, d_ser, threshold=35):
            return (k_ser.iloc[-1] > d_ser.iloc[-1]) and (k_ser.iloc[-2] <= d_ser.iloc[-2]) and (k_ser.iloc[-1] <= threshold)

        if is_low_kd_gold(k_60m, d_60m) and is_low_kd_gold(k_daily, d_daily): return True
    except: pass
    return False

def check_bollinger_squeeze_fast(df_daily):
    """ 策略二：布林軌道壓縮修正版 (絕對頻寬 <= 6% 且隨時準備突破) """
    try:
        c_daily = df_daily['Close'].squeeze().astype(float)
        ma20 = c_daily.rolling(window=20).mean()
        std20 = c_daily.rolling(window=20).std()
        
        upper_band = ma20 + (2 * std20)
        lower_band = ma20 - (2 * std20)
        bandwidth = (upper_band - lower_band) / ma20
        
        current_bw = bandwidth.iloc[-1]
        if current_bw > 0.06: return False # 頻寬沒小於 6% 直接淘汰
        
        close_today = c_daily.iloc[-1]
        if close_today >= ma20.iloc[-1] and ((upper_band.iloc[-1] - close_today) / close_today <= 0.02):
            return True, current_bw * 100
    except: pass
    return False

def check_multi_timeframe_tangling(df_60m, df_daily, df_weekly):
    """ 策略三：時/日/週 全週期同步糾結 """
    try:
        c_60m = df_60m['Close'].squeeze().astype(float)
        c_daily = df_daily['Close'].squeeze().astype(float)
        c_weekly = df_weekly['Close'].squeeze().astype(float)
        
        m60_tangle = (max(c_60m.rolling(5).mean().iloc[-1], c_60m.rolling(10).mean().iloc[-1], c_60m.rolling(20).mean().iloc[-1]) - min(c_60m.rolling(5).mean().iloc[-1], c_60m.rolling(10).mean().iloc[-1], c_60m.rolling(20).mean().iloc[-1])) / c_60m.rolling(20).mean().iloc[-1]
        d_tangle = (max(c_daily.rolling(5).mean().iloc[-1], c_daily.rolling(10).mean().iloc[-1], c_daily.rolling(20).mean().iloc[-1]) - min(c_daily.rolling(5).mean().iloc[-1], c_daily.rolling(10).mean().iloc[-1], c_daily.rolling(20).mean().iloc[-1])) / c_daily.rolling(20).mean().iloc[-1]
        w_tangle = (max(c_weekly.rolling(5).mean().iloc[-1], c_weekly.rolling(10).mean().iloc[-1], c_weekly.rolling(20).mean().iloc[-1]) - min(c_weekly.rolling(5).mean().iloc[-1], c_weekly.rolling(10).mean().iloc[-1], c_weekly.rolling(20).mean().iloc[-1])) / c_weekly.rolling(20).mean().iloc[-1]
        
        if m60_tangle < 0.025 and d_tangle < 0.03 and w_tangle < 0.035 and c_daily.iloc[-1] > c_daily.rolling(20).mean().iloc[-1]: return True
    except: pass
    return False

def check_extreme_drop_fast(df_daily):
    """ 策略四：短線極限超賣 × 爆量紅K """
    try:
        c_daily = df_daily['Close'].squeeze().astype(float)
        rsi6 = calculate_rsi(c_daily, period=6).iloc[-1]
        if rsi6 < 20 and c_daily.iloc[-1] > df_daily['Open'].squeeze().astype(float).iloc[-1] and df_daily['Volume'].squeeze().astype(float).iloc[-1] > df_daily['Volume'].squeeze().astype(float).rolling(5).mean().iloc[-1]: return True
    except: pass
    return False

def check_strat7_low_price_high_turnover(ticker, df_daily):
    """ 策略七：主力進場換手股 (股價低檔 × 換手率 10% - 17%) """
    try:
        shares = SHARES_OUTSTANDING_DATA.get(ticker)
        if not shares:
            t = yf.Ticker(ticker)
            shares = t.info.get("sharesOutstanding")
            if shares: SHARES_OUTSTANDING_DATA[ticker] = shares
            else: return False
        
        c_daily = df_daily['Close'].squeeze().astype(float)
        v_daily = df_daily['Volume'].squeeze().astype(float)
        low_120, high_120 = c_daily.rolling(window=120).min().iloc[-1], c_daily.rolling(window=120).max().iloc[-1]
        
        price_position = (c_daily.iloc[-1] - low_120) / (high_120 - low_120)
        turnover_rate = (v_daily.iloc[-1] / shares) * 100
        if price_position <= 0.30 and 10.0 <= turnover_rate <= 17.0: 
            return True, price_position * 100, turnover_rate
    except: pass
    return False

# ==============================================================================
# 💬 Telegram 發送
# ==============================================================================
def send_telegram_message(message):
    bot_token = os.environ.get("TG_BOT_TOKEN")
    chat_id = os.environ.get("TG_CHAT_ID")
    if not bot_token or not chat_id: return
    url = f"https://api.telegram.org/bot{str(bot_token).strip()}/sendMessage"
    payload = {"chat_id": str(chat_id).strip(), "text": message, "parse_mode": "HTML"}
    try: requests.post(url, json=payload, timeout=10)
    except Exception as e: print(f"❌ Telegram 發送異常: {e}")

# ==============================================================================
# 🚀 高效主程式流程
# ==============================================================================
if __name__ == "__main__":
    start_time = time.time()
    now_tw = pd.Timestamp.now(tz='UTC').tz_convert('Asia/Taipei')
    tw_time_str = now_tw.strftime('%Y-%m-%d %H:%M:%S')

    print("🚀 啟動【台股 1~7 全策略極速過濾系統】...")
    tech_scan_pool = fetch_all_taiwan_market_tickers()
    fetch_fast_chips_summary() # 下載大戶籌碼快取

    print(f"⏳ 步驟 1: 打包下載全市場日K資料 (共 {len(tech_scan_pool)} 檔)...")
    full_df_daily = yf.download(tech_scan_pool, period="1y", interval="1d", progress=False, auto_adjust=True)
    
    # 分類儲存桶 (1 ~ 7 全到齊)
    strat1, strat2, strat3, strat4, strat5, strat6, strat7 = [], [], [], [], [], [], []
    
    # 建立需要進行「多週期(60分K/週K)」掃描的精簡名單
    heavy_scan_pool = []

    print("⏳ 步驟 2: 進行第一輪輕量策略篩選（日K與基本面）...")
    for ticker in tech_scan_pool:
        try:
            if ticker not in full_df_daily.columns.levels[1]: continue
            df_d = full_df_daily.xs(ticker, axis=1, level=1)
            if df_d.empty or len(df_d) < 120: continue # 策略七需要 120 天數據
            
            # 核心防線：20日均量 >= 1000張
            if df_d['Volume'].rolling(window=20).mean().iloc[-1] / 1000 < 1000: continue

            name_zh = DYNAMIC_STOCK_NAMES.get(ticker, "")
            stock_label = f"<code>{ticker}</code>(<i>{name_zh}</i>)" if name_zh else f"<code>{ticker}</code>"

            # 🛠️ 【策略二：布林壓縮】
            bo_check = check_bollinger_squeeze_fast(df_d)
            if bo_check: strat2.append(f"{stock_label}[頻寬:{bo_check[1]:.1f}%]")

            # 🛠️ 【策略四：極限超賣】
            if check_extreme_drop_fast(df_d): strat4.append(stock_label)

            # 🛠️ 【策略五：籌碼大戶模式】(直接從快取比對)
            if CHIPS_SUMMARY_DATA.get(ticker): strat5.append(f"{stock_label}[大戶持股連續加碼]")

            # 🛠️ 【策略六：基本面價值低估】
            val = FUNDAMENTAL_DATA.get(ticker, {})
            if 0 < val.get("PE", 0) <= 12.0 and 0 < val.get("PB", 0) <= 1.0:
                strat6.append(f"{stock_label}[PE:{val['PE']:.1f}, PB:{val['PB']:.2f}]")

            # 🛠️ 【策略七：主力進場換手股】
            turnover_check = check_strat7_low_price_high_turnover(ticker, df_d)
            if turnover_check: 
                strat7.append(f"{stock_label}[位置:{turnover_check[1]:.1f}%, 換手:{turnover_check[2]:.1f}%]")

            # 只要有通過量能防線，且具備基本多頭型態的股票，才給予多週期深潛資格
            if df_d['Close'].iloc[-1] > df_d['Close'].rolling(20).mean().iloc[-1]:
                heavy_scan_pool.append(ticker)

        except: continue

    # ⏳ 步驟 3: 多週期漏斗深度掃描 (只查最有可能發動的精選池)
    # 限制前 40 檔，避免被 Yahoo 封鎖 IP，同時也是為了省時間
    final_heavy_pool = heavy_scan_pool[:40]
    if final_heavy_pool:
        print(f"⏳ 步驟 3: 針對精選的 {len(final_heavy_pool)} 檔標的下載 60分K 與 週K...")
        full_df_60m = yf.download(final_heavy_pool, period="1mo", interval="60m", progress=False, auto_adjust=True)
        full_df_weekly = yf.download(final_heavy_pool, period="2y", interval="1wk", progress=False, auto_adjust=True)

        for ticker in final_heavy_pool:
            try:
                if ticker not in full_df_60m.columns.levels[1] or ticker not in full_df_weekly.columns.levels[1]: continue
                df_m60 = full_df_60m.xs(ticker, axis=1, level=1)
                df_w = full_df_weekly.xs(ticker, axis=1, level=1)
                df_d = full_df_daily.xs(ticker, axis=1, level=1)

                name_zh = DYNAMIC_STOCK_NAMES.get(ticker, "")
                stock_label = f"<code>{ticker}</code>(<i>{name_zh}</i>)" if name_zh else f"<code>{ticker}</code>"

                # 🛠️ 【策略一：多週期三頻共振】
                if check_strat1_resonance(df_m60, df_d, df_w): strat1.append(stock_label)

                # 🛠️ 【策略三：全週期同步糾結】
                if check_multi_timeframe_tangling(df_m60, df_d, df_w): strat3.append(stock_label)
            except: continue

    # 📝 建立重編號後的綜合美化報告訊息
    tw_msg = f"🇹🇼 <b>【台股多策略選股報告】</b>\n⚠️ <i>已過濾 20日均量 &lt; 1000張之殭屍股</i>\n⏰ 時間: {tw_time_str}\n"
    tw_msg += "───────────────────\n\n"
    
    tw_msg += "📈 <b>【策略一】原版多週期三頻共振 (MACD + KD 低檔金叉)</b>\n"
    tw_msg += f"↳ {', '.join(strat1) if strat1 else '今日無符合標的。 💤'}\n\n"

    tw_msg += "💥 <b>【策略二】布林軌道壓縮 (修正版：頻寬極致壓縮 ≤ 6% 臨界面)</b>\n"
    tw_msg += f"↳ {', '.join(strat2) if strat2 else '今日無符合標的。 💤'}\n\n"

    tw_msg += "💎 <b>【策略三】時/日/週 全週期同步糾結 (不限排列)</b>\n"
    tw_msg += f"↳ {', '.join(strat3) if strat3 else '今日無符合標的。 💤'}\n\n"

    tw_msg += "🔥 <b>【策略四】短線極限超賣 × 爆量紅K (恐慌止跌)</b>\n"
    tw_msg += f"↳ {', '.join(strat4) if strat4 else '今日無符合標的。 💤'}\n\n"

    tw_msg += "🎯 <b>【策略五】大戶控盤波段股 (行為偵測：千張大戶連續加碼)</b>\n"
    tw_msg += f"↳ {', '.join(strat5) if strat5 else '今日無符合標的。 💤'}\n\n"

    tw_msg += "💰 <b>【策略六】價值型低估股 (本益比 ≤ 12 × 股價淨值比 ≤ 1.0)</b>\n"
    tw_msg += f"↳ {', '.join(strat6) if strat6 else '今日無符合標的。 💤'}\n\n"

    tw_msg += "🔄 <b>【策略七】主力進場換手股 (股價半年低檔區 × 換手率 10% - 17%)</b>\n"
    tw_msg += f"↳ {', '.join(strat7) if strat7 else '今日無符合標的。 💤'}\n"

    send_telegram_message(tw_msg)
    print(f"✅ 1~7 全策略整合報告發送完畢！總耗時: {time.time() - start_time:.1f} 秒")
