import pandas as pd
import yfinance as yf
import requests
import os
import time

# ==============================================================================
# 🇹🇼 台股全市場快速資料下載模組
# ==============================================================================
DYNAMIC_STOCK_NAMES = {}

def fetch_all_taiwan_market_tickers():
    """ 下載全台股市場代碼與名稱 """
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

    return sorted(list(set(all_tickers)))

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
# 🎯 策略判斷邏輯
# ==============================================================================
def check_macd_kd_single_tf(df_tf):
    """ 通用模組：判斷單一週期的 MACD 往0軸向上 + KD黃金交叉 """
    try:
        if df_tf.empty or len(df_tf) < 30: return False, 0.0
        c_tf = df_tf['Close'].squeeze().astype(float)

        macd_line, signal_line, hist = calculate_macd(c_tf)
        is_macd_up = (macd_line.iloc[-1] > macd_line.iloc[-2]) and (macd_line.iloc[-1] >= macd_line.iloc[-3])
        is_macd_towards_zero = (macd_line.iloc[-1] >= -0.5) or (hist.iloc[-1] > 0)

        k_ser, d_ser = calculate_kd(df_tf)
        is_kd_gold = (k_ser.iloc[-1] > d_ser.iloc[-1]) and (k_ser.iloc[-2] <= d_ser.iloc[-2] or k_ser.iloc[-3] <= d_ser.iloc[-3])

        if is_macd_up and is_macd_towards_zero and is_kd_gold:
            return True, c_tf.iloc[-1]
    except: pass
    return False, 0.0

def check_strat4_ma20_breakout(df_daily):
    """ 策略四：主力突破月線 """
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
            return True, c_daily.iloc[-1]
    except: pass
    return False, 0.0

def check_strat5_volume_breakout(df_daily):
    """ 策略五：關鍵均線多頭突破 × 量能倍增 """
    try:
        if df_daily.empty or len(df_daily) < 20: return False, 0.0
        
        c_daily = df_daily['Close'].squeeze().astype(float)
        v_daily = df_daily['Volume'].squeeze().astype(float)
        
        ma20 = c_daily.rolling(window=20).mean()
        close_today = c_daily.iloc[-1]
        close_yesterday = c_daily.iloc[-2]
        ma20_today = ma20.iloc[-1]
        ma20_yesterday = ma20.iloc[-2]
        
        price_break_cond = (close_today > ma20_today) and (close_yesterday <= ma20_yesterday or (close_today - close_yesterday) / close_yesterday > 0.02)
        if not price_break_cond: return False, 0.0
        
        v_ma5 = v_daily.rolling(window=5).mean().iloc[-1]
        volume_today = v_daily.iloc[-1]
        volume_cond = volume_today > (v_ma5 * 1.5)
        if not volume_cond: return False, 0.0
        
        k_series, d_series = calculate_kd(df_daily)
        k_today = k_series.iloc[-1]
        d_today = d_series.iloc[-1]
        kd_cond = (k_today > d_today) and (k_today < 75)
        
        if kd_cond:
            return True, close_today
    except Exception:
        pass
    return False, 0.0

def check_strat6_extreme_drop_fast(df_daily):
    """ 原策略七：短線極限超賣 × 爆量紅K """
    try:
        c_daily = df_daily['Close'].squeeze().astype(float)
        rsi6 = calculate_rsi(c_daily, period=6).iloc[-1]
        if rsi6 < 20 and c_daily.iloc[-1] > df_daily['Open'].squeeze().astype(float).iloc[-1] and df_daily['Volume'].squeeze().astype(float).iloc[-1] > df_daily['Volume'].squeeze().astype(float).rolling(5).mean().iloc[-1]:
            return True, c_daily.iloc[-1]
    except: pass
    return False, 0.0

def check_strat7_low_price_volume_surge(df_daily):
    """ 原策略八：低檔爆量股 """
    try:
        c_daily = df_daily['Close'].squeeze().astype(float)
        v_daily = df_daily['Volume'].squeeze().astype(float)
        o_daily = df_daily['Open'].squeeze().astype(float)

        low_120 = c_daily.rolling(window=120).min().iloc[-1]
        high_120 = c_daily.rolling(window=120).max().iloc[-1]
        if high_120 == low_120: return False, 0.0, 0.0

        price_position = (c_daily.iloc[-1] - low_120) / (high_120 - low_120)
        is_low_position = price_position <= 0.30

        v_ma5_prev = v_daily.rolling(window=5).mean().iloc[-2]
        is_volume_surge = (v_daily.iloc[-1] >= v_ma5_prev * 2.5) if v_ma5_prev > 0 else False

        is_red_k = c_daily.iloc[-1] > o_daily.iloc[-1]

        if is_low_position and is_volume_surge and is_red_k:
            return True, c_daily.iloc[-1], price_position * 100
    except: pass
    return False, 0.0, 0.0

# ==============================================================================
# 💬 Telegram 發送模組
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
# 🚀 主程式
# ==============================================================================
if __name__ == "__main__":
    start_time = time.time()
    now_tw = pd.Timestamp.now(tz='UTC').tz_convert('Asia/Taipei')
    tw_time_str = now_tw.strftime('%Y-%m-%d %H:%M:%S')

    print("🚀 啟動【台股 7 大精簡策略選股系統】...")
    tech_scan_pool = fetch_all_taiwan_market_tickers()

    print(f"⏳ 步驟 1: 下載全市場日K與週K資料 (共 {len(tech_scan_pool)} 檔)...")
    full_df_daily = yf.download(tech_scan_pool, period="1y", interval="1d", progress=False, auto_adjust=True)
    full_df_weekly = yf.download(tech_scan_pool, period="2y", interval="1wk", progress=False, auto_adjust=True)

    # 初始化 1 ~ 7 策略結果清單
    strat1, strat2, strat3, strat4, strat5, strat6, strat7 = [], [], [], [], [], [], []
    heavy_scan_pool = []

    print("⏳ 步驟 2: 進行日K、週K相關策略篩選...")
    for ticker in tech_scan_pool:
        try:
            if ticker not in full_df_daily.columns.levels[1]: continue
            df_d = full_df_daily.xs(ticker, axis=1, level=1)
            if df_d.empty or len(df_d) < 120: continue 
            
            # 核心防線：20日均量 >= 1000張
            if df_d['Volume'].rolling(window=20).mean().iloc[-1] / 1000 < 1000: continue

            name_zh = DYNAMIC_STOCK_NAMES.get(ticker, "")
            stock_label = f"<code>{ticker}</code>(<i>{name_zh}</i>)" if name_zh else f"<code>{ticker}</code>"

            # 🛠️ 【策略二：日K MACD往0軸向上 + KD金叉】
            res2, price2 = check_macd_kd_single_tf(df_d)
            if res2:
                strat2.append(f"{stock_label}[{price2:.2f}元]")

            # 🛠️ 【策略三：週K MACD往0軸向上 + KD金叉】
            if ticker in full_df_weekly.columns.levels[1]:
                df_w = full_df_weekly.xs(ticker, axis=1, level=1)
                res3, price3 = check_macd_kd_single_tf(df_w)
                if res3:
                    strat3.append(f"{stock_label}[{price2 if res2 else df_d['Close'].iloc[-1]:.2f}元]")

            # 🛠️ 【策略四：主力突破月線】
            res4, price4 = check_strat4_ma20_breakout(df_d)
            if res4: 
                strat4.append(f"{stock_label}[{price4:.2f}元]")

            # 🛠️ 【策略五：關鍵均線多頭突破 × 量能倍增】
            res5, price5 = check_strat5_volume_breakout(df_d)
            if res5: 
                strat5.append(f"{stock_label}[{price5:.2f}元]")

            # 🛠️ 【策略六：短線極限超賣 × 爆量紅K】
            res6, price6 = check_strat6_extreme_drop_fast(df_d)
            if res6: 
                strat6.append(f"{stock_label}[{price6:.2f}元]")

            # 🛠️ 【策略七：低檔爆量股】
            res7, price7, pos7 = check_strat7_low_price_volume_surge(df_d)
            if res7:
                strat7.append(f"{stock_label}[{price7:.2f}元]")

            # 收集適合下載 60分K 的精選名單
            if df_d['Close'].iloc[-1] > df_d['Close'].rolling(20).mean().iloc[-1]:
                heavy_scan_pool.append(ticker)

        except: continue

    # ⏳ 步驟 3: 下載 60分K 資料，進行策略一判斷
    final_heavy_pool = heavy_scan_pool[:50]
    if final_heavy_pool:
        print(f"⏳ 步驟 3: 下載精選 {len(final_heavy_pool)} 檔標的的 60分K 資料...")
        full_df_60m = yf.download(final_heavy_pool, period="1mo", interval="60m", progress=False, auto_adjust=True)

        for ticker in final_heavy_pool:
            try:
                if ticker not in full_df_60m.columns.levels[1]: continue
                df_m60 = full_df_60m.xs(ticker, axis=1, level=1)
                
                name_zh = DYNAMIC_STOCK_NAMES.get(ticker, "")
                stock_label = f"<code>{ticker}</code>(<i>{name_zh}</i>)" if name_zh else f"<code>{ticker}</code>"

                # 🛠️ 【策略一：60分K MACD往0軸向上 + KD金叉】
                res1, price1 = check_macd_kd_single_tf(df_m60)
                if res1: 
                    strat1.append(f"{stock_label}[{price1:.2f}元]")
            except: continue

    # 📝 Telegram 報告輸出
    tw_msg = f"🇹🇼 <b>【台股多策略選股報告】</b>\n⚠️ <i>已過濾 20日均量 &lt; 1000張之殭屍股</i>\n⏰ 時間: {tw_time_str}\n"
    tw_msg += "───────────────────\n\n"
    
    tw_msg += "📈 <b>【策略一】60分K MACD趨向0軸向上 + KD黃金交叉</b>\n"
    tw_msg += f"↳ {', '.join(strat1) if strat1 else '今日無符合標的。 💤'}\n\n"

    tw_msg += "📊 <b>【策略二】日K MACD趨向0軸向上 + KD黃金交叉</b>\n"
    tw_msg += f"↳ {', '.join(strat2) if strat2 else '今日無符合標的。 💤'}\n\n"

    tw_msg += "📈 <b>【策略三】週K MACD趨向0軸向上 + KD黃金交叉</b>\n"
    tw_msg += f"↳ {', '.join(strat3) if strat3 else '今日無符合標的。 💤'}\n\n"

    tw_msg += "🚀 <b>【策略四】主力突破月線 (突破20MA × 成交量>1.5倍5日均量 × KD指南針向上)</b>\n"
    tw_msg += f"↳ {', '.join(strat4) if strat4 else '今日無符合標的。 💤'}\n\n"

    tw_msg += "⚡ <b>【策略五】關鍵均線多頭突破 × 量能倍增 (帶量突破)</b>\n"
    tw_msg += f"↳ {', '.join(strat5) if strat5 else '今日無符合標的。 💤'}\n\n"

    tw_msg += "🔥 <b>【策略六】短線極限超賣 × 爆量紅K (恐慌止跌)</b>\n"
    tw_msg += f"↳ {', '.join(strat6) if strat6 else '今日無符合標的。 💤'}\n\n"

    tw_msg += "💥 <b>【策略七】低檔爆量股 (半年位階 ≤ 30% × 成交量 ≥ 2.5倍5日均量 × 紅K)</b>\n"
    tw_msg += f"↳ {', '.join(strat7) if strat7 else '今日無符合標的。 💤'}\n"

    send_telegram_message(tw_msg)
    print(f"✅ 策略報告發送完成！總耗時: {time.time() - start_time:.1f} 秒")
