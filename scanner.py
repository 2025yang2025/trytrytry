import pandas as pd
import yfinance as yf
import requests
import os
import time

# ==============================================================================
# 🇹🇼 台股全市場快速資料下載模組
# ==============================================================================
DYNAMIC_STOCK_NAMES = {}
FUNDAMENTAL_DATA = {}  
SHARES_OUTSTANDING_DATA = {}  
CHIPS_SUMMARY_DATA = {} 

FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "")

def fetch_all_taiwan_market_tickers():
    """ 下載全台股市場代碼與證交所估值資料 """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    all_tickers = []
    
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

    try:
        url_valuation = "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL"
        res_val = requests.get(url_valuation, headers=headers, timeout=10)
        if res_val.status_code == 200:
            for item in res_val.json():
                code = item.get("Code", "").strip()
                ticker_id = f"{code}.TW"
                try: pe = float(item.get("PEratio", 0)) if item.get("PEratio") else 0.0
                except: pe = 0.0
                try: pb = float(item.get("PBRatio", 0)) if item.get("PBRatio") else 0.0
                except: pb = 0.0
                FUNDAMENTAL_DATA[ticker_id] = {"PE": pe, "PB": pb}
    except Exception as e:
        print(f"⚠️ 撈取證交所估值資料異常: {e}")

    return sorted(list(set(all_tickers)))

def fetch_fast_chips_summary():
    """ 策略八：一次性下載大戶持股總表快取 """
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
                df_1000 = df_h[df_h['HoldersLevel'] == '1,000,001以上'].sort_values(by='date')
                for stock_id, group in df_1000.groupby('stock_id'):
                    if len(group) >= 2:
                        if group['percent'].iloc[-1] > group['percent'].iloc[-2]:
                            CHIPS_SUMMARY_DATA[f"{stock_id}.TW"] = True
    except Exception as e:
        print(f"⚠️ 籌碼快取下載異常: {e}")

# ==============================================================================
# 📊 技術指標算術模組
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
# 🎯 策略判斷邏輯（整合策略 1 ~ 12）
# ==============================================================================
def check_strat1_resonance(df_30m, df_60m):
    """ 策略一：30分與60分K棒 MACD 往0軸向上 + KD黃金交叉 """
    try:
        c_30m = df_30m['Close'].squeeze().astype(float)
        c_60m = df_60m['Close'].squeeze().astype(float)

        def check_single_tf(df_tf, c_tf):
            macd_line, signal_line, hist = calculate_macd(c_tf)
            is_macd_up = (macd_line.iloc[-1] > macd_line.iloc[-2]) and (macd_line.iloc[-1] >= macd_line.iloc[-3])
            is_macd_towards_zero = (macd_line.iloc[-1] >= -0.5) or (hist.iloc[-1] > 0)

            k_ser, d_ser = calculate_kd(df_tf)
            is_kd_gold = (k_ser.iloc[-1] > d_ser.iloc[-1]) and (k_ser.iloc[-2] <= d_ser.iloc[-2] or k_ser.iloc[-3] <= d_ser.iloc[-3])
            return is_macd_up and is_macd_towards_zero and is_kd_gold

        if check_single_tf(df_30m, c_30m) and check_single_tf(df_60m, c_60m):
            return True
    except: pass
    return False

def check_strat2_daily_weekly_resonance(df_daily, df_weekly):
    """ 策略二：日K與週K MACD 往0軸向上 + KD黃金交叉 """
    try:
        c_daily = df_daily['Close'].squeeze().astype(float)
        c_weekly = df_weekly['Close'].squeeze().astype(float)

        def check_single_tf(df_tf, c_tf):
            macd_line, signal_line, hist = calculate_macd(c_tf)
            is_macd_up = (macd_line.iloc[-1] > macd_line.iloc[-2]) and (macd_line.iloc[-1] >= macd_line.iloc[-3])
            is_macd_towards_zero = (macd_line.iloc[-1] >= -0.5) or (hist.iloc[-1] > 0)

            k_ser, d_ser = calculate_kd(df_tf)
            is_kd_gold = (k_ser.iloc[-1] > d_ser.iloc[-1]) and (k_ser.iloc[-2] <= d_ser.iloc[-2] or k_ser.iloc[-3] <= d_ser.iloc[-3])
            return is_macd_up and is_macd_towards_zero and is_kd_gold

        if check_single_tf(df_daily, c_daily) and check_single_tf(df_weekly, c_weekly):
            return True
    except: pass
    return False

def check_strat3_ma20_breakout(df_daily):
    """ 策略三：主力突破月線 (強勢突破 20MA + 成交量 > 5日均量 1.5倍 + KD指南針向上) """
    try:
        c_daily = df_daily['Close'].squeeze().astype(float)
        v_daily = df_daily['Volume'].squeeze().astype(float)
        o_daily = df_daily['Open'].squeeze().astype(float)
        
        ma20 = c_daily.rolling(window=20).mean()
        v_ma5 = v_daily.rolling(window=5).mean()

        is_break_ma20 = (c_daily.iloc[-1] > ma20.iloc[-1]) and (c_daily.iloc[-2] <= ma20.iloc[-2] or c_daily.iloc[-1] > o_daily.iloc[-1])
        is_vol_spike = v_daily.iloc[-1] >= (v_ma5.iloc[-2] * 1.5)

        k_ser, d_ser = calculate_kd(df_daily)
        is_kd_up = (k_ser.iloc[-1] > d_ser.iloc[-1]) and (k_ser.iloc[-1] > k_ser.iloc[-2])

        if is_break_ma20 and is_vol_spike and is_kd_up:
            vol_ratio = v_daily.iloc[-1] / v_ma5.iloc[-2] if v_ma5.iloc[-2] > 0 else 1.5
            return True, vol_ratio
    except: pass
    return False

def check_strat4_volume_breakout(df_daily):
    """ 🆕 策略四：關鍵均線多頭突破 × 量能倍增 (帶量突破) """
    try:
        if df_daily.empty or len(df_daily) < 20: return False
        
        c_daily = df_daily['Close'].squeeze().astype(float)
        v_daily = df_daily['Volume'].squeeze().astype(float)
        
        ma20 = c_daily.rolling(window=20).mean()
        close_today = c_daily.iloc[-1]
        close_yesterday = c_daily.iloc[-2]
        ma20_today = ma20.iloc[-1]
        ma20_yesterday = ma20.iloc[-2]
        
        price_break_cond = (close_today > ma20_today) and (close_yesterday <= ma20_yesterday or (close_today - close_yesterday) / close_yesterday > 0.02)
        if not price_break_cond: return False
        
        v_ma5 = v_daily.rolling(window=5).mean().iloc[-1]
        volume_today = v_daily.iloc[-1]
        volume_cond = volume_today > (v_ma5 * 1.5)
        if not volume_cond: return False
        
        k_series, d_series = calculate_kd(df_daily)
        k_today = k_series.iloc[-1]
        d_today = d_series.iloc[-1]
        kd_cond = (k_today > d_today) and (k_today < 75)
        
        if kd_cond:
            volume_ratio = volume_today / v_ma5 if v_ma5 > 0 else 1.0
            return True, volume_ratio
    except Exception:
        pass
    return False

def check_strat5_bollinger_squeeze_fast(df_daily):
    """ 策略五：布林軌道極致壓縮 """
    try:
        c_daily = df_daily['Close'].squeeze().astype(float)
        ma20 = c_daily.rolling(window=20).mean()
        std20 = c_daily.rolling(window=20).std()
        bandwidth = ((ma20 + (2 * std20)) - (ma20 - (2 * std20))) / ma20
        current_bw = bandwidth.iloc[-1]
        
        if current_bw <= 0.06 and (c_daily.iloc[-1] >= ma20.iloc[-1]) and (((ma20.iloc[-1] + (2 * std20.iloc[-1])) - c_daily.iloc[-1]) / c_daily.iloc[-1] <= 0.02):
            return True, current_bw * 100
    except: pass
    return False

def check_strat6_multi_timeframe_tangling(df_60m, df_daily, df_weekly):
    """ 策略六：全週期同步糾結 """
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

def check_strat7_extreme_drop_fast(df_daily):
    """ 策略七：短線極限超賣 × 爆量紅K """
    try:
        c_daily = df_daily['Close'].squeeze().astype(float)
        rsi6 = calculate_rsi(c_daily, period=6).iloc[-1]
        if rsi6 < 20 and c_daily.iloc[-1] > df_daily['Open'].squeeze().astype(float).iloc[-1] and df_daily['Volume'].squeeze().astype(float).iloc[-1] > df_daily['Volume'].squeeze().astype(float).rolling(5).mean().iloc[-1]: return True
    except: pass
    return False

def check_strat10_low_price_high_turnover(ticker, df_daily):
    """ 策略十：主力進場換手股 """
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

def check_strat11_two_month_squeeze_breakout(df_daily):
    """ 策略十一：雙月極限壓縮突破股 """
    try:
        if len(df_daily) < 41: return False
        c_daily = df_daily['Close'].squeeze().astype(float)
        h_daily = df_daily['High'].squeeze().astype(float)
        l_daily = df_daily['Low'].squeeze().astype(float)
        o_daily = df_daily['Open'].squeeze().astype(float)

        past_40_high = h_daily.iloc[-41:-1].max()
        past_40_low = l_daily.iloc[-41:-1].min()
        if past_40_low <= 0: return False

        squeeze_range = (past_40_high - past_40_low) / past_40_low
        is_squeezed = 0.10 <= squeeze_range <= 0.15

        today_close = c_daily.iloc[-1]
        today_open = o_daily.iloc[-1]

        is_breakout = today_close > past_40_high
        is_first_bar = c_daily.iloc[-2] <= past_40_high
        is_red_k = today_close > today_open

        if is_squeezed and is_breakout and is_first_bar and is_red_k:
            return True, squeeze_range * 100
    except: pass
    return False

def check_strat12_low_price_volume_surge(df_daily):
    """ 策略十二：低檔爆量股 (半年近120日低位階 ≤ 30% × 當日成交量 ≥ 5日均量2.5倍 × 實體紅K) """
    try:
        c_daily = df_daily['Close'].squeeze().astype(float)
        v_daily = df_daily['Volume'].squeeze().astype(float)
        o_daily = df_daily['Open'].squeeze().astype(float)

        low_120 = c_daily.rolling(window=120).min().iloc[-1]
        high_120 = c_daily.rolling(window=120).max().iloc[-1]
        if high_120 == low_120: return False

        price_position = (c_daily.iloc[-1] - low_120) / (high_120 - low_120)
        is_low_position = price_position <= 0.30

        v_ma5_prev = v_daily.rolling(window=5).mean().iloc[-2]
        is_volume_surge = (v_daily.iloc[-1] >= v_ma5_prev * 2.5) if v_ma5_prev > 0 else False

        is_red_k = c_daily.iloc[-1] > o_daily.iloc[-1]

        if is_low_position and is_volume_surge and is_red_k:
            vol_multiple = v_daily.iloc[-1] / v_ma5_prev if v_ma5_prev > 0 else 2.5
            return True, price_position * 100, vol_multiple
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
# 🚀 主程式 (全 1 ~ 12 策略整合)
# ==============================================================================
if __name__ == "__main__":
    start_time = time.time()
    now_tw = pd.Timestamp.now(tz='UTC').tz_convert('Asia/Taipei')
    tw_time_str = now_tw.strftime('%Y-%m-%d %H:%M:%S')

    print("🚀 啟動【台股 1~12 全策略極速過濾系統】...")
    tech_scan_pool = fetch_all_taiwan_market_tickers()
    fetch_fast_chips_summary()

    print(f"⏳ 步驟 1: 打包下載全市場日K資料 (共 {len(tech_scan_pool)} 檔)...")
    full_df_daily = yf.download(tech_scan_pool, period="1y", interval="1d", progress=False, auto_adjust=True)
    
    # 初始化 1 ~ 12 策略結果清單
    strat1, strat2, strat3, strat4, strat5, strat6, strat7, strat8, strat9, strat10, strat11, strat12 = [], [], [], [], [], [], [], [], [], [], [], []
    heavy_scan_pool = []

    print("⏳ 步驟 2: 進行第一輪日K與基本面策略篩選...")
    for ticker in tech_scan_pool:
        try:
            if ticker not in full_df_daily.columns.levels[1]: continue
            df_d = full_df_daily.xs(ticker, axis=1, level=1)
            if df_d.empty or len(df_d) < 120: continue 
            
            # 核心防線：20日均量 >= 1000張
            if df_d['Volume'].rolling(window=20).mean().iloc[-1] / 1000 < 1000: continue

            name_zh = DYNAMIC_STOCK_NAMES.get(ticker, "")
            stock_label = f"<code>{ticker}</code>(<i>{name_zh}</i>)" if name_zh else f"<code>{ticker}</code>"

            # 🛠️ 【策略三：主力突破月線】
            ma20_check = check_strat3_ma20_breakout(df_d)
            if ma20_check: strat3.append(f"{stock_label}[爆量:{ma20_check[1]:.1f}倍]")

            # 🛠️ 【策略四：關鍵均線多頭突破 × 量能倍增】
            vol_break = check_strat4_volume_breakout(df_d)
            if vol_break: strat4.append(f"{stock_label}[爆量:{vol_break[1]:.1f}倍]")

            # 🛠️ 【策略五：布林壓縮】
            bo_check = check_strat5_bollinger_squeeze_fast(df_d)
            if bo_check: strat5.append(f"{stock_label}[頻寬:{bo_check[1]:.1f}%]")

            # 🛠️ 【策略七：極限超賣】
            if check_strat7_extreme_drop_fast(df_d): strat7.append(stock_label)

            # 🛠️ 【策略八：籌碼大戶模式】
            if CHIPS_SUMMARY_DATA.get(ticker): strat8.append(f"{stock_label}[大戶持股連續加碼]")

            # 🛠️ 【策略九：基本面價值低估】
            val = FUNDAMENTAL_DATA.get(ticker, {})
            if 0 < val.get("PE", 0) <= 12.0 and 0 < val.get("PB", 0) <= 1.0:
                strat9.append(f"{stock_label}[PE:{val['PE']:.1f}, PB:{val['PB']:.2f}]")

            # 🛠️ 【策略十：主力進場換手股】
            turnover_check = check_strat10_low_price_high_turnover(ticker, df_d)
            if turnover_check: 
                strat10.append(f"{stock_label}[位置:{turnover_check[1]:.1f}%, 換手:{turnover_check[2]:.1f}%]")

            # 🛠️ 【策略十一：雙月極限壓縮突破股】
            sq_breakout = check_strat11_two_month_squeeze_breakout(df_d)
            if sq_breakout:
                strat11.append(f"{stock_label}[雙月區間:{sq_breakout[1]:.1f}%]")

            # 🛠️ 【策略十二：低檔爆量股】
            low_vol_check = check_strat12_low_price_volume_surge(df_d)
            if low_vol_check:
                strat12.append(f"{stock_label}[位階:{low_vol_check[1]:.1f}%, 爆量:{low_vol_check[2]:.1f}倍]")

            # 精選多週期掃描池
            if df_d['Close'].iloc[-1] > df_d['Close'].rolling(20).mean().iloc[-1]:
                heavy_scan_pool.append(ticker)

        except: continue

    # ⏳ 步驟 3: 下載多週期資料 (30分K、60分K 與 週K)
    final_heavy_pool = heavy_scan_pool[:40]
    if final_heavy_pool:
        print(f"⏳ 步驟 3: 下載精選 {len(final_heavy_pool)} 檔標的的 30分K、60分K 與 週K...")
        full_df_30m = yf.download(final_heavy_pool, period="1mo", interval="30m", progress=False, auto_adjust=True)
        full_df_60m = yf.download(final_heavy_pool, period="1mo", interval="60m", progress=False, auto_adjust=True)
        full_df_weekly = yf.download(final_heavy_pool, period="2y", interval="1wk", progress=False, auto_adjust=True)

        for ticker in final_heavy_pool:
            try:
                if (ticker not in full_df_30m.columns.levels[1] or 
                    ticker not in full_df_60m.columns.levels[1] or 
                    ticker not in full_df_weekly.columns.levels[1]): 
                    continue
                
                df_m30 = full_df_30m.xs(ticker, axis=1, level=1)
                df_m60 = full_df_60m.xs(ticker, axis=1, level=1)
                df_w = full_df_weekly.xs(ticker, axis=1, level=1)
                df_d = full_df_daily.xs(ticker, axis=1, level=1)

                name_zh = DYNAMIC_STOCK_NAMES.get(ticker, "")
                stock_label = f"<code>{ticker}</code>(<i>{name_zh}</i>)" if name_zh else f"<code>{ticker}</code>"

                # 🛠️ 【策略一：30m/60m MACD往0軸向上 + KD黃金交叉】
                if check_strat1_resonance(df_m30, df_m60): 
                    strat1.append(stock_label)

                # 🛠️ 【策略二：日K/週K MACD往0軸向上 + KD黃金交叉】
                if check_strat2_daily_weekly_resonance(df_d, df_w):
                    strat2.append(stock_label)

                # 🛠️ 【策略六：時/日/週 全週期同步糾結】
                if check_strat6_multi_timeframe_tangling(df_m60, df_d, df_w): 
                    strat6.append(stock_label)
            except: continue

    # 📝 Telegram 報告輸出
    tw_msg = f"🇹🇼 <b>【台股多策略選股報告】</b>\n⚠️ <i>已過濾 20日均量 &lt; 1000張之殭屍股</i>\n⏰ 時間: {tw_time_str}\n"
    tw_msg += "───────────────────\n\n"
    
    tw_msg += "📈 <b>【策略一】30分/60分K MACD趨向0軸向上 + KD黃金交叉</b>\n"
    tw_msg += f"↳ {', '.join(strat1) if strat1 else '今日無符合標的。 💤'}\n\n"

    tw_msg += "📊 <b>【策略二】日K/週K MACD趨向0軸向上 + KD黃金交叉</b>\n"
    tw_msg += f"↳ {', '.join(strat2) if strat2 else '今日無符合標的。 💤'}\n\n"

    tw_msg += "🚀 <b>【策略三】主力突破月線 (突破20MA × 成交量>1.5倍5日均量 × KD指南針向上)</b>\n"
    tw_msg += f"↳ {', '.join(strat3) if strat3 else '今日無符合標的。 💤'}\n\n"

    tw_msg += "⚡ <b>【策略四】關鍵均線多頭突破 × 量能倍增 (帶量突破)</b>\n"
    tw_msg += f"↳ {', '.join(strat4) if strat4 else '今日無符合標的。 💤'}\n\n"

    tw_msg += "💥 <b>【策略五】布林軌道壓縮 (頻寬極致壓縮 ≤ 6% 臨界面)</b>\n"
    tw_msg += f"↳ {', '.join(strat5) if strat5 else '今日無符合標的。 💤'}\n\n"

    tw_msg += "💎 <b>【策略六】時/日/週 全週期同步糾結 (不限排列)</b>\n"
    tw_msg += f"↳ {', '.join(strat6) if strat6 else '今日無符合標的。 💤'}\n\n"

    tw_msg += "🔥 <b>【策略七】短線極限超賣 × 爆量紅K (恐慌止跌)</b>\n"
    tw_msg += f"↳ {', '.join(strat7) if strat7 else '今日無符合標的。 💤'}\n\n"

    tw_msg += "🎯 <b>【策略八】大戶控盤波段股 (千張大戶連續加碼)</b>\n"
    tw_msg += f"↳ {', '.join(strat8) if strat8 else '今日無符合標的。 💤'}\n\n"

    tw_msg += "💰 <b>【策略九】價值型低估股 (本益比 ≤ 12 × 股價淨值比 ≤ 1.0)</b>\n"
    tw_msg += f"↳ {', '.join(strat9) if strat9 else '今日無符合標的。 💤'}\n\n"

    tw_msg += "🔄 <b>【策略十】主力進場換手股 (股價半年低檔區 × 換手率 10% - 17%)</b>\n"
    tw_msg += f"↳ {', '.join(strat10) if strat10 else '今日無符合標的。 💤'}\n\n"

    tw_msg += "✨ <b>【策略十一】雙月極限壓縮突破股 (雙月幅度 10%-15% × 首根創高紅K)</b>\n"
    tw_msg += f"↳ {', '.join(strat11) if strat11 else '今日無符合標的。 💤'}\n\n"

    tw_msg += "💥 <b>【策略十二】低檔爆量股 (半年位階 ≤ 30% × 成交量 ≥ 2.5倍5日均量 × 紅K)</b>\n"
    tw_msg += f"↳ {', '.join(strat12) if strat12 else '今日無符合標的。 💤'}\n"

    send_telegram_message(tw_msg)
    print(f"✅ 1~12 全策略報告發送完成！總耗時: {time.time() - start_time:.1f} 秒")
