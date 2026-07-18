import pandas as pd
import yfinance as yf
import requests
import os
import time
import random

# ==============================================================================
# 🇹🇼 台股全市場技術面與基本面模組
# ==============================================================================
DYNAMIC_STOCK_NAMES = {}
FUNDAMENTAL_DATA = {}  
SHARES_OUTSTANDING_DATA = {}  

FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "")

def fetch_all_taiwan_market_tickers():
    """ 下載全台股市場代碼（不限產業），並同步撈取證交所盤後估值資料 """
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
                try:
                    pe = float(item.get("PEratio", 0)) if item.get("PEratio") else 0.0
                except:
                    pe = 0.0
                try:
                    pb = float(item.get("PBRatio", 0)) if item.get("PBRatio") else 0.0
                except:
                    pb = 0.0
                FUNDAMENTAL_DATA[ticker_id] = {"PE": pe, "PB": pb}
    except Exception as e:
        print(f"⚠️ 撈取證交所估值資料異常: {e}")

    if not all_tickers:
        backup_dict = {"2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科"}
        for k, v in backup_dict.items():
            all_tickers.append(k)
            DYNAMIC_STOCK_NAMES[k] = v
            
    return sorted(list(set(all_tickers)))

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
# 🎯 策略檢測邏輯（重新排序與編號）
# ==============================================================================

def check_strat1_resonance(df_60m, df_daily, df_weekly):
    """ 策略一：多週期三頻共振 (MACD) + KD低檔金叉 """
    try:
        c_60m = df_60m['Close'].squeeze().astype(float)
        c_daily = df_daily['Close'].squeeze().astype(float)
        c_weekly = df_weekly['Close'].squeeze().astype(float)
        if c_60m.empty or c_daily.empty or c_weekly.empty: return False

        w_macd, w_signal, w_hist = calculate_macd(c_weekly)
        d_macd, d_signal, d_hist = calculate_macd(c_daily)
        d_ma = c_daily.rolling(window=20).mean()
        m60_macd, m60_signal, m60_hist = calculate_macd(c_60m)

        if len(w_hist) < 1 or len(d_hist) < 1 or len(m60_hist) < 2: return False

        w_m, w_s, w_h = float(w_macd.iloc[-1]), float(w_signal.iloc[-1]), float(w_hist.iloc[-1])
        d_m, d_s, d_c, d_ma_val = float(d_macd.iloc[-1]), float(d_signal.iloc[-1]), float(c_daily.iloc[-1]), float(d_ma.iloc[-1])
        m60_m, m60_h, m60_h_prev = float(m60_macd.iloc[-1]), float(m60_hist.iloc[-1]), float(m60_hist.iloc[-2])

        macd_cond = (w_m > w_s) and (w_h > 0) and (d_m > 0) and (d_m > d_s) and (d_c > d_ma_val) and (m60_m > 0) and (m60_h > 0) and (m60_h_prev <= 0)
        if not macd_cond: return False

        k_60m, d_60m = calculate_kd(df_60m)
        k_daily, d_daily = calculate_kd(df_daily)
        
        def is_low_kd_gold(k_ser, d_ser, threshold=35):
            if len(k_ser) < 2: return False
            return (k_ser.iloc[-1] > d_ser.iloc[-1]) and (k_ser.iloc[-2] <= d_ser.iloc[-2]) and (k_ser.iloc[-1] <= threshold)

        if is_low_kd_gold(k_60m, d_60m) and is_low_kd_gold(k_daily, d_daily): return True
    except: pass
    return False

def check_bollinger_squeeze(df_daily):
    """ 策略二：布林軌道壓縮 (Bandwidth 創近 60 日新低) """
    try:
        if df_daily.empty or len(df_daily) < 20: return False
        c_daily = df_daily['Close'].squeeze().astype(float)
        
        ma20 = c_daily.rolling(window=20).mean()
        std20 = c_daily.rolling(window=20).std()
        
        upper_band = ma20 + (2 * std20)
        lower_band = ma20 - (2 * std20)
        bandwidth = (upper_band - lower_band) / ma20
        
        if len(bandwidth) < 60: return False
        
        current_bw = bandwidth.iloc[-1]
        min_bw_60 = bandwidth.iloc[-60:].min()
        
        if current_bw == min_bw_60 and current_bw <= 0.08:
            return True, current_bw * 100
    except: pass
    return False

def check_multi_timeframe_tangling(df_60m, df_daily, df_weekly):
    """ 策略三：時/日/週 全週期同步糾結 """
    try:
        c_60m = df_60m['Close'].squeeze().astype(float)
        c_daily = df_daily['Close'].squeeze().astype(float)
        c_weekly = df_weekly['Close'].squeeze().astype(float)
        if len(c_60m) < 20 or len(c_daily) < 20 or len(c_weekly) < 20: return False
        
        m60_tangle = (max(c_60m.rolling(5).mean().iloc[-1], c_60m.rolling(10).mean().iloc[-1], c_60m.rolling(20).mean().iloc[-1]) - min(c_60m.rolling(5).mean().iloc[-1], c_60m.rolling(10).mean().iloc[-1], c_60m.rolling(20).mean().iloc[-1])) / c_60m.rolling(20).mean().iloc[-1]
        d_tangle = (max(c_daily.rolling(5).mean().iloc[-1], c_daily.rolling(10).mean().iloc[-1], c_daily.rolling(20).mean().iloc[-1]) - min(c_daily.rolling(5).mean().iloc[-1], c_daily.rolling(10).mean().iloc[-1], c_daily.rolling(20).mean().iloc[-1])) / c_daily.rolling(20).mean().iloc[-1]
        w_tangle = (max(c_weekly.rolling(5).mean().iloc[-1], c_weekly.rolling(10).mean().iloc[-1], c_weekly.rolling(20).mean().iloc[-1]) - min(c_weekly.rolling(5).mean().iloc[-1], c_weekly.rolling(10).mean().iloc[-1], c_weekly.rolling(20).mean().iloc[-1])) / c_weekly.rolling(20).mean().iloc[-1]
        
        if m60_tangle < 0.025 and d_tangle < 0.03 and w_tangle < 0.035 and c_daily.iloc[-1] > c_daily.rolling(20).mean().iloc[-1]: return True
    except: pass
    return False

def check_extreme_drop_volume_up(df_daily):
    """ 策略四：短線極限超賣 × 爆量紅K """
    try:
        if df_daily.empty or len(df_daily) < 20: return False
        c_daily = df_daily['Close'].squeeze().astype(float)
        o_daily = df_daily['Open'].squeeze().astype(float)
        v_daily = df_daily['Volume'].squeeze().astype(float)
        rsi6 = calculate_rsi(c_daily, period=6).iloc[-1]
        if rsi6 < 20 and c_daily.iloc[-1] > o_daily.iloc[-1] and v_daily.iloc[-1] > v_daily.rolling(5).mean().iloc[-1]: return True
    except: pass
    return False

def check_smart_money_behavior(ticker):
    """ 策略五：關鍵分點大戶控盤行為模式 (大戶加碼 + 排除隔日沖) """
    stock_id = ticker.split(".")[0]
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=15)).strftime('%Y-%m-%d')
    url = "https://api.finmindtrade.com/api/v4/data"
    
    try:
        params = {"dataset": "taiwan_stock_broker_bs", "data_id": stock_id, "start_date": start_date, "end_date": end_date, "token": FINMIND_TOKEN}
        res = requests.get(url, params=params, timeout=10)
        if res.status_code != 200: return False
        data = res.json().get("data", [])
        if not data: return False
        
        df = pd.DataFrame(data)
        df['net_buy'] = (df['buyShare'] - df['sellShare']) / 1000
        
        t_days = sorted(df['date'].unique())[-3:]
        if len(t_days) < 3: return False
        
        df_recent = df[df['date'].isin(t_days)]
        broker_summary = df_recent.groupby('brokerName')['net_buy'].agg(['count', 'sum'])
        top_buyer = broker_summary.sort_values(by='sum', ascending=False).index[0]
        top_buyer_sum = broker_summary.sort_values(by='sum', ascending=False)['sum'].iloc[0]
        
        df_top_broker = df[df['brokerName'] == top_buyer].sort_values(by='date')
        if len(df_top_broker) >= 3:
            correlation = df_top_broker['net_buy'].diff().shift(-1).corr(df_top_broker['net_buy'])
            if correlation < -0.6: return False
            
        params_holder = {"dataset": "taiwan_stock_holding_shares_per", "data_id": stock_id, "start_date": (pd.Timestamp.now() - pd.Timedelta(days=30)).strftime('%Y-%m-%d'), "end_date": end_date, "token": FINMIND_TOKEN}
        res_h = requests.get(url, params=params_holder, timeout=10)
        if res_h.status_code == 200:
            data_h = res_h.json().get("data", [])
            if data_h:
                df_h = pd.DataFrame(data_h)
                df_1000 = df_h[df_h['HoldersLevel'] == '1,000,001以上'].sort_values(by='date')
                if len(df_1000) >= 2:
                    if df_1000['percent'].iloc[-1] > df_1000['percent'].iloc[-2]:
                        return True, top_buyer, top_buyer_sum
    except: pass
    return False

def check_strat6_undervalued(ticker):
    """ 策略六：基本面價值型低估股 """
    data = FUNDAMENTAL_DATA.get(ticker)
    if not data: return False
    pe, pb = data.get("PE", 0), data.get("PB", 0)
    if 0 < pe <= 12.0 and 0 < pb <= 1.0: return True, pe, pb
    return False

def check_strat7_low_price_high_turnover(ticker, df_daily):
    """ 策略七：主力進場換手股 (股價低檔 × 換手率 10% - 17%) """
    try:
        if df_daily.empty or len(df_daily) < 120: return False
        shares = SHARES_OUTSTANDING_DATA.get(ticker)
        if not shares:
            t = yf.Ticker(ticker)
            shares = t.info.get("sharesOutstanding")
            if shares: SHARES_OUTSTANDING_DATA[ticker] = shares
            else: return False
        
        c_daily = df_daily['Close'].squeeze().astype(float)
        v_daily = df_daily['Volume'].squeeze().astype(float)
        low_120, high_120 = c_daily.rolling(window=120).min().iloc[-1], c_daily.rolling(window=120).max().iloc[-1]
        close_today = c_daily.iloc[-1]
        if high_120 == low_120: return False
        price_position = (close_today - low_120) / (high_120 - low_120)
        turnover_rate = (v_daily.iloc[-1] / shares) * 100
        if price_position <= 0.30 and 10.0 <= turnover_rate <= 17.0: return True, price_position * 100, turnover_rate
    except: pass
    return False

# ==============================================================================
# 💬 Telegram 發送
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
    try: requests.post(url, json=payload, timeout=10)
    except Exception as e: print(f"❌ Telegram 發送異常: {e}")

# ==============================================================================
# 🚀 主程式
# ==============================================================================
if __name__ == "__main__":
    now_tw = pd.Timestamp.now(tz='UTC').tz_convert('Asia/Taipei')
    tw_time_str = now_tw.strftime('%Y-%m-%d %H:%M:%S')

    print("🚀 啟動【台股盤後多策略全市場篩選報告】...")
    tech_scan_pool = fetch_all_taiwan_market_tickers()
    
    if not tech_scan_pool:
        print("❌ 未能取得任何股票代碼，程式結束。")
        exit()

    print(f"⏳ 步驟 1: 批次下載全市場日K資料進行量能過濾 (共 {len(tech_scan_pool)} 檔)...")
    full_df_daily = yf.download(tech_scan_pool, period="1y", interval="1d", progress=False, auto_adjust=True)
    
    qualified_tickers = []
    for ticker in tech_scan_pool:
        try:
            if len(tech_scan_pool) == 1:
                v_daily = full_df_daily['Volume'].squeeze()
            else:
                if ticker not in full_df_daily.columns.levels[1]: continue
                v_daily = full_df_daily.xs(ticker, axis=1, level=1)['Volume'].squeeze()
                
            if len(v_daily) >= 20:
                v_ma20_sheets = v_daily.rolling(window=20).mean().iloc[-1] / 1000
                if v_ma20_sheets >= 1000: qualified_tickers.append(ticker)
        except: continue

    print(f"🎯 通過量能防線（20日均量 >= 1000張）股票共 {len(qualified_tickers)} 檔。")
    
    # 初始化 1 ~ 7 新策略清單
    strat1, strat2, strat3, strat4, strat5, strat6, strat7 = [], [], [], [], [], [], []

    if qualified_tickers:
        print("⏳ 步驟 2: 批次下載精選股票的 60分K 與 週K 資料...")
        full_df_60m = yf.download(qualified_tickers, period="1mo", interval="60m", progress=False, auto_adjust=True)
        full_df_weekly = yf.download(qualified_tickers, period="2y", interval="1wk", progress=False, auto_adjust=True)

        print("⏳ 步驟 3: 記憶體內高速多維策略流檢測中...")
        for ticker in qualified_tickers:
            try:
                if len(qualified_tickers) == 1:
                    df_d, df_m60, df_w = full_df_daily.copy(), full_df_60m.copy(), full_df_weekly.copy()
                else:
                    if ticker not in full_df_daily.columns.levels[1] or ticker not in full_df_60m.columns.levels[1] or ticker not in full_df_weekly.columns.levels[1]: continue
                    df_d = full_df_daily.xs(ticker, axis=1, level=1)
                    df_m60 = full_df_60m.xs(ticker, axis=1, level=1)
                    df_w = full_df_weekly.xs(ticker, axis=1, level=1)

                if df_d.empty or df_m60.empty or df_w.empty: continue

                name_zh = DYNAMIC_STOCK_NAMES.get(ticker, "")
                stock_label = f"<code>{ticker}</code>(<i>{name_zh}</i>)" if name_zh else f"<code>{ticker}</code>"

                # 策略一
                if check_strat1_resonance(df_m60, df_d, df_w): strat1.append(stock_label)
                
                # 策略二 (布林壓縮)
                bo_check = check_bollinger_squeeze(df_d)
                if bo_check: strat2.append(f"{stock_label}[頻寬:{bo_check[1]:.1f}%]")
                
                # 策略三
                if check_multi_timeframe_tangling(df_m60, df_d, df_w): strat3.append(stock_label)
                
                # 策略四
                if check_extreme_drop_volume_up(df_d): strat4.append(stock_label)
                
                # 策略五 (籌碼行為模式)
                money_check = check_smart_money_behavior(ticker)
                if money_check: strat5.append(f"{stock_label}[{money_check[1]}波段進駐:+{int(money_check[2])}張]")
                
                # 策略六
                val_check = check_strat6_undervalued(ticker)
                if val_check: strat6.append(f"{stock_label}[PE:{val_check[1]:.1f}, PB:{val_check[2]:.2f}]")

                # 策略七
                turnover_check = check_strat7_low_price_high_turnover(ticker, df_d)
                if turnover_check: strat7.append(f"{stock_label}[位置:{turnover_check[1]:.1f}%, 換手:{turnover_check[2]:.1f}%]")
                    
                time.sleep(random.uniform(0.3, 0.6))

            except KeyError: continue
            except: continue

    # 📝 建立重編號後的綜合美化報告訊息
    tw_msg = f"🇹🇼 <b>【台股多策略選股報告】</b>\n⚠️ <i>已過濾 20日均量 &lt; 1000張之殭屍股</i>\n⏰ 時間: {tw_time_str}\n"
    tw_msg += "───────────────────\n\n"
    
    tw_msg += "📈 <b>【策略一】原版多週期三頻共振 (MACD + KD 低檔金叉)</b>\n"
    tw_msg += f"↳ {', '.join(strat1) if strat1 else '今日無符合標的。 💤'}\n\n"

    tw_msg += "💥 <b>【策略二】布林軌道壓縮 (蓄勢變盤股：頻寬創60日新低 ≤ 8%)</b>\n"
    tw_msg += f"↳ {', '.join(strat2) if strat2 else '今日無符合標的。 💤'}\n\n"

    tw_msg += "💎 <b>【策略三】時/日/週 全週期同步糾結 (不限排列)</b>\n"
    tw_msg += f"↳ {', '.join(strat3) if strat3 else '今日無符合標的。 💤'}\n\n"

    tw_msg += "🔥 <b>【策略四】短線極限超賣 × 爆量紅K (恐慌止跌)</b>\n"
    tw_msg += f"↳ {', '.join(strat4) if strat4 else '今日無符合標的。 💤'}\n\n"

    tw_msg += "🎯 <b>【策略五】大戶控盤波段股 (行為偵測：大戶加碼 + 排除隔日沖)</b>\n"
    tw_msg += f"↳ {', '.join(strat5) if strat5 else '今日無符合標的。 💤'}\n\n"

    tw_msg += "💰 <b>【策略六】價值型低估股 (本益比 ≤ 12 × 股價淨值比 ≤ 1.0)</b>\n"
    tw_msg += f"↳ {', '.join(strat6) if strat6 else '今日無符合標的。 💤'}\n\n"

    tw_msg += "🔄 <b>【策略七】主力進場換手股 (股價半年低檔區 × 換手率 10% - 17%)</b>\n"
    tw_msg += f"↳ {', '.join(strat7) if strat7 else '今日無符合標的。 💤'}\n"

    send_telegram_message(tw_msg)
    print("✅ 台股新策略多維綜合報告（序號重排版）發送完畢！")
