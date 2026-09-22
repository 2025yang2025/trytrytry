import pandas as pd
import yfinance as yf
import requests
import os
import time

# ==============================================================================
# 🇹🇼 台股全市場快速資料下載模組 (含 ETF 過濾)
# ==============================================================================
DYNAMIC_STOCK_NAMES = {}

def fetch_all_taiwan_market_tickers():
    """ 下載全台股市場代碼與名稱，並排除 ETF """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    all_tickers = []
    
    try:
        url_twse = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
        res = requests.get(url_twse, headers=headers, timeout=10)
        if res.status_code == 200:
            for item in res.json():
                code = item.get("Code", "").strip()
                name = item.get("Name", "").strip()
                
                # 🛡️ 過濾邏輯：只留標準 4 位數純個股，排除 00 開頭之 ETF 與特殊證券
                if code.isdigit() and len(code) == 4 and not code.startswith("00") and not code.startswith("0"):
                    ticker_id = f"{code}.TW"
                    all_tickers.append(ticker_id)
                    DYNAMIC_STOCK_NAMES[ticker_id] = name
    except Exception as e:
        print(f"⚠️ 撈取全市場名單異常: {e}")

    return sorted(list(set(all_tickers)))

# ==============================================================================
# 🛡️ 安全分批下載模組
# ==============================================================================
def safe_download_yf(tickers, period, interval, chunk_size=250):
    """ 分批安全下載 yfinance 資料 """
    all_dfs = []
    total_chunks = (len(tickers) + chunk_size - 1) // chunk_size

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        current_chunk = (i // chunk_size) + 1
        
        for attempt in range(2):
            try:
                df = yf.download(
                    chunk, 
                    period=period, 
                    interval=interval, 
                    progress=False, 
                    auto_adjust=True,
                    threads=True
                )
                if not df.empty:
                    all_dfs.append(df)
                break
            except Exception as e:
                if attempt == 1:
                    print(f"⚠️ 批次 {current_chunk}/{total_chunks} 下載失敗: {e}")
                time.sleep(1)

        time.sleep(0.3)

    if all_dfs:
        return pd.concat(all_dfs, axis=1)
    return pd.DataFrame()

# ==============================================================================
# 📊 技術指標算術模組
# ==============================================================================
def calculate_macd(close_series, fast=12, slow=26, signal=9):
    fast_ema = close_series.ewm(span=fast, adjust=False).mean()
    slow_ema = close_series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema  # 即 DIF
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()  # 即 DEM / MACD
    histogram = macd_line - signal_line  # 柱狀體
    return macd_line, signal_line, histogram

def calculate_kd(df_single, n=9, m1=3, m2=3):
    """ 嚴謹計算 KD 值 """
    df_clean = df_single[['High', 'Low', 'Close']].dropna().astype(float)
    if len(df_clean) < n:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    low_min = df_clean['Low'].rolling(window=n).min()
    high_max = df_clean['High'].rolling(window=n).max()
    close = df_clean['Close']
    
    denom = high_max - low_min
    rsv = (((close - low_min) / denom) * 100).fillna(50)
    
    k_vals = []
    d_vals = []
    k_prev, d_prev = 50.0, 50.0
    
    for val in rsv:
        k_curr = (k_prev * (m1 - 1) + val) / m1
        d_curr = (d_prev * (m2 - 1) + k_curr) / m2
        k_vals.append(k_curr)
        d_vals.append(d_curr)
        k_prev, d_prev = k_curr, d_curr
        
    return pd.Series(k_vals, index=df_clean.index), pd.Series(d_vals, index=df_clean.index)

# ==============================================================================
# 🎯 策略判斷邏輯
# ==============================================================================
def check_macd_above_zero_kd(df_tf, kd_threshold=20):
    """ 判斷 MACD > 0 且 KD 雙線 > kd_threshold """
    try:
        df_clean = df_tf.dropna(subset=['Close', 'High', 'Low'])
        if len(df_clean) < 30: return False, 0.0
        c_tf = df_clean['Close'].astype(float)

        macd_line, _, _ = calculate_macd(c_tf)
        macd_val = macd_line.iloc[-1]
        if pd.isna(macd_val) or macd_val <= 0:
            return False, 0.0

        k_ser, d_ser = calculate_kd(df_clean)
        if k_ser.empty or d_ser.empty: return False, 0.0
        
        k_val = k_ser.iloc[-1]
        d_val = d_ser.iloc[-1]
        if pd.isna(k_val) or pd.isna(d_val): return False, 0.0

        if (k_val > kd_threshold) and (d_val > kd_threshold):
            return True, c_tf.iloc[-1]
    except Exception:
        pass
    return False, 0.0

def check_macd_heading_to_zero_kd(df_tf, kd_threshold=20):
    """ 判斷 MACD 往 0 軸向上 (MACD < 0 且向上增加) + KD > kd_threshold """
    try:
        df_clean = df_tf.dropna(subset=['Close', 'High', 'Low'])
        if len(df_clean) < 30: return False, 0.0
        c_tf = df_clean['Close'].astype(float)

        macd_line, _, _ = calculate_macd(c_tf)
        is_macd_heading_up = (macd_line.iloc[-1] < 0) and (macd_line.iloc[-1] > macd_line.iloc[-2])

        k_ser, d_ser = calculate_kd(df_clean)
        if k_ser.empty or d_ser.empty: return False, 0.0
        
        k_val = k_ser.iloc[-1]
        d_val = d_ser.iloc[-1]
        is_kd_cond = (k_val > kd_threshold) and (d_val > kd_threshold)

        if is_macd_heading_up and is_kd_cond:
            return True, c_tf.iloc[-1]
    except Exception:
        pass
    return False, 0.0

def check_strategy_ma20_rebound(df_d, df_w, df_m):
    """ 🛠️ 新增策略十：月/週趨勢偏多 + 日線 20MA (月線) 有撐且 MACD 柱狀體翻紅/擴展 """
    try:
        # 1. 月 K 檢測: DIF (MACD Line) > 0
        df_m_clean = df_m.dropna(subset=['Close'])
        if len(df_m_clean) < 26: return False, 0.0
        dif_m, _, _ = calculate_macd(df_m_clean['Close'].astype(float))
        if pd.isna(dif_m.iloc[-1]) or dif_m.iloc[-1] <= 0:
            return False, 0.0

        # 2. 週 K 檢測: DIF > 0 且 (Hist > 0 或 Hist_curr > Hist_prev)
        df_w_clean = df_w.dropna(subset=['Close'])
        if len(df_w_clean) < 26: return False, 0.0
        dif_w, _, hist_w = calculate_macd(df_w_clean['Close'].astype(float))
        if pd.isna(dif_w.iloc[-1]) or dif_w.iloc[-1] <= 0:
            return False, 0.0
        
        is_w_hist_ok = (hist_w.iloc[-1] > 0) or (hist_w.iloc[-1] > hist_w.iloc[-2])
        if not is_w_hist_ok:
            return False, 0.0

        # 3. 日 K 檢測: 回測 20MA 有撐 且 MACD 柱狀體翻紅
        df_d_clean = df_d.dropna(subset=['Close', 'High', 'Low'])
        if len(df_d_clean) < 35: return False, 0.0
        
        close_d = df_d_clean['Close'].astype(float)
        low_d = df_d_clean['Low'].astype(float)
        ma20 = close_d.rolling(20).mean()
        
        c_val = close_d.iloc[-1]
        l_val = low_d.iloc[-1]
        ma20_val = ma20.iloc[-1]

        # 日線支撐判斷: 當日低點有碰到月線且收盤站上，或者收盤價離月線很近 (0.985 ~ 1.025 倍)
        is_touch_ma20 = (l_val <= ma20_val and c_val >= ma20_val) or (0.985 <= (c_val / ma20_val) <= 1.025)
        
        if not is_touch_ma20:
            return False, 0.0

        # MACD 柱狀體翻紅 (上一期 <= 0 且最新一期 > 0)
        _, _, hist_d = calculate_macd(close_d)
        is_hist_turn_red = (hist_d.iloc[-2] <= 0 and hist_d.iloc[-1] > 0) or (hist_d.iloc[-1] > 0 and hist_d.iloc[-1] > hist_d.iloc[-2])

        if is_hist_turn_red:
            return True, c_val

    except Exception:
        pass
    return False, 0.0

def get_kd_latest(df_tf):
    """ 取得最新一筆 K 與 D 值 """
    try:
        df_clean = df_tf.dropna(subset=['Close', 'High', 'Low'])
        if len(df_clean) < 15: return None, None
        k_ser, d_ser = calculate_kd(df_clean)
        if not k_ser.empty and not d_ser.empty:
            return k_ser.iloc[-1], d_ser.iloc[-1]
    except Exception:
        pass
    return None, None

# ==============================================================================
# 💬 Telegram 發送模組
# ==============================================================================
def send_telegram_message(message, max_length=3500):
    bot_token = os.environ.get("TG_BOT_TOKEN")
    chat_id = os.environ.get("TG_CHAT_ID")
    
    if not bot_token or not chat_id:
        print("❌ 錯誤：未設定 TG_BOT_TOKEN 或 TG_CHAT_ID 環境變數！")
        return

    url = f"https://api.telegram.org/bot{str(bot_token).strip()}/sendMessage"

    lines = message.split('\n')
    chunks = []
    current_chunk = ""

    for line in lines:
        if len(current_chunk) + len(line) + 1 > max_length:
            chunks.append(current_chunk)
            current_chunk = line + "\n"
        else:
            current_chunk += line + "\n"
    if current_chunk:
        chunks.append(current_chunk)

    for idx, chunk in enumerate(chunks, 1):
        payload = {
            "chat_id": str(chat_id).strip(),
            "text": chunk.strip(),
            "parse_mode": "HTML"
        }
        try:
            res = requests.post(url, json=payload, timeout=10)
            res_json = res.json()
            if res.status_code == 200 and res_json.get("ok"):
                print(f"✅ Telegram 訊息段落 ({idx}/{len(chunks)}) 發送成功！")
            else:
                print(f"❌ Telegram 發送失敗 (HTTP {res.status_code}): {res_json}")
        except Exception as e:
            print(f"❌ Telegram 發送連線異常: {e}")
        time.sleep(0.5)

# ==============================================================================
# 🚀 主程式
# ==============================================================================
if __name__ == "__main__":
    start_time = time.time()
    now_tw = pd.Timestamp.now(tz='UTC').tz_convert('Asia/Taipei')
    tw_time_str = now_tw.strftime('%Y-%m-%d %H:%M:%S')

    print("🚀 啟動【台股精準個股選股系統】...")
    tech_scan_pool = fetch_all_taiwan_market_tickers()

    print(f"⏳ 步驟 1: 下載全市場個股日K、週K與月K資料 (已排除 ETF，共 {len(tech_scan_pool)} 檔)...")
    full_df_daily = safe_download_yf(tech_scan_pool, period="1y", interval="1d", chunk_size=250)
    full_df_weekly = safe_download_yf(tech_scan_pool, period="2y", interval="1wk", chunk_size=250)
    full_df_monthly = safe_download_yf(tech_scan_pool, period="5y", interval="1mo", chunk_size=250)

    strat1_map, strat2_map, strat3_map = {}, {}, {}
    strat4_map, strat5_map = {}, {}
    strat9_map, strat10_map = {}, {}
    heavy_scan_pool = []

    print("⏳ 步驟 2: 進行多週期與 MA20 支撐策略篩選...")
    for ticker in tech_scan_pool:
        try:
            if ticker not in full_df_daily.columns.levels[1]: continue
            df_d = full_df_daily.xs(ticker, axis=1, level=1)
            if df_d.empty or len(df_d.dropna(subset=['Close'])) < 120: continue 
            
            # 🛠️ 嚴格門檻：20日均量 >= 2500張 (Volume / 1000.0 >= 2500)
            avg_vol_20_shares = df_d['Volume'].rolling(window=20).mean().iloc[-1]
            if (avg_vol_20_shares / 1000.0) < 2500: continue

            name_zh = DYNAMIC_STOCK_NAMES.get(ticker, "")
            stock_label = f"<code>{ticker}</code>(<i>{name_zh}</i>)" if name_zh else f"<code>{ticker}</code>"

            # 🛠️ 【策略一：月K MACD > 0 + KD > 20】
            if ticker in full_df_monthly.columns.levels[1]:
                df_m = full_df_monthly.xs(ticker, axis=1, level=1)
                res1, _ = check_macd_above_zero_kd(df_m, kd_threshold=20)
                if res1:
                    strat1_map[ticker] = f"{stock_label}[{df_d['Close'].dropna().iloc[-1]:.2f}元]"

            # 🛠️ 【策略二：週K MACD > 0 + KD > 50】
            if ticker in full_df_weekly.columns.levels[1]:
                df_w = full_df_weekly.xs(ticker, axis=1, level=1)
                res2, _ = check_macd_above_zero_kd(df_w, kd_threshold=50)
                if res2:
                    strat2_map[ticker] = f"{stock_label}[{df_d['Close'].dropna().iloc[-1]:.2f}元]"

            # 🛠️ 【策略三：日K MACD > 0 + KD > 20】
            res3, price3 = check_macd_above_zero_kd(df_d, kd_threshold=20)
            if res3:
                strat3_map[ticker] = f"{stock_label}[{price3:.2f}元]"

            # 🛠️ 【新策略：月/週長線多頭 + 日K 20MA (月線) 有撐且 MACD 柱狀體翻紅】
            if (ticker in full_df_monthly.columns.levels[1]) and (ticker in full_df_weekly.columns.levels[1]):
                df_m = full_df_monthly.xs(ticker, axis=1, level=1)
                df_w = full_df_weekly.xs(ticker, axis=1, level=1)
                res10, price10 = check_strategy_ma20_rebound(df_d, df_w, df_m)
                if res10:
                    strat10_map[ticker] = f"{stock_label}[{price10:.2f}元]"

            # 收集適合掃描分 K 的精選個股名單
            heavy_scan_pool.append(ticker)

        except Exception:
            continue

    # ⏳ 步驟 3: 下載 30分K 與 60分K 資料
    final_heavy_pool = heavy_scan_pool[:100]
    if final_heavy_pool:
        print(f"⏳ 步驟 3: 下載精選 {len(final_heavy_pool)} 檔標的之 30分K 與 60分K 資料...")
        full_df_30m = safe_download_yf(final_heavy_pool, period="1mo", interval="30m", chunk_size=50)
        full_df_60m = safe_download_yf(final_heavy_pool, period="1mo", interval="60m", chunk_size=50)

        for ticker in final_heavy_pool:
            try:
                name_zh = DYNAMIC_STOCK_NAMES.get(ticker, "")
                stock_label = f"<code>{ticker}</code>(<i>{name_zh}</i>)" if name_zh else f"<code>{ticker}</code>"
                df_d = full_df_daily.xs(ticker, axis=1, level=1)

                # 🛠️ 【策略四：60分K MACD往0軸向上 + KD > 20】
                if ticker in full_df_60m.columns.levels[1]:
                    df_m60 = full_df_60m.xs(ticker, axis=1, level=1)
                    res4, price4 = check_macd_heading_to_zero_kd(df_m60, kd_threshold=20)
                    if res4: 
                        strat4_map[ticker] = f"{stock_label}[{price4:.2f}元]"

                # 🛠️ 【策略五：30分K MACD往0軸向上 + KD > 20】
                if ticker in full_df_30m.columns.levels[1]:
                    df_m30 = full_df_30m.xs(ticker, axis=1, level=1)
                    res5, price5 = check_macd_heading_to_zero_kd(df_m30, kd_threshold=20)
                    if res5:
                        strat5_map[ticker] = f"{stock_label}[{price5:.2f}元]"

                # 🛠️ 【策略九：反向 - 30m, 60m, 日, 週, 月 KD 皆 < 70 且跌破日10日線】
                close_d = df_d['Close'].dropna()
                ma10_d = close_d.rolling(10).mean()
                if close_d.iloc[-1] < ma10_d.iloc[-1]:
                    kd_30m_k, kd_30m_d = get_kd_latest(full_df_30m.xs(ticker, axis=1, level=1)) if ticker in full_df_30m.columns.levels[1] else (None, None)
                    kd_60m_k, kd_60m_d = get_kd_latest(full_df_60m.xs(ticker, axis=1, level=1)) if ticker in full_df_60m.columns.levels[1] else (None, None)
                    kd_d_k, kd_d_d = get_kd_latest(df_d)
                    kd_w_k, kd_w_d = get_kd_latest(full_df_weekly.xs(ticker, axis=1, level=1)) if ticker in full_df_weekly.columns.levels[1] else (None, None)
                    kd_m_k, kd_m_d = get_kd_latest(full_df_monthly.xs(ticker, axis=1, level=1)) if ticker in full_df_monthly.columns.levels[1] else (None, None)

                    all_kds = [kd_30m_k, kd_30m_d, kd_60m_k, kd_60m_d, kd_d_k, kd_d_d, kd_w_k, kd_w_d, kd_m_k, kd_m_d]
                    if all(val is not None and val < 70 for val in all_kds):
                        strat9_map[ticker] = f"{stock_label}[{close_d.iloc[-1]:.2f}元]"

            except Exception:
                continue

    # 🛠️ 【策略六：策略三 ∩ 策略四 (日K ∩ 60分K)】
    strat6_tickers = set(strat3_map.keys()) & set(strat4_map.keys())
    strat6 = [strat3_map[t] for t in strat6_tickers]

    # 🛠️ 【策略七：策略一 ∩ 策略二 (月K ∩ 週K)】
    strat7_tickers = set(strat1_map.keys()) & set(strat2_map.keys())
    strat7 = [strat1_map[t] for t in strat7_tickers]

    # 🛠️ 【策略八：策略六 ∩ 策略七】
    strat8_tickers = strat6_tickers & strat7_tickers
    strat8 = [strat3_map[t] for t in strat8_tickers]

    strat1 = list(strat1_map.values())
    strat2 = list(strat2_map.values())
    strat3 = list(strat3_map.values())
    strat4 = list(strat4_map.values())
    strat5 = list(strat5_map.values())
    strat9 = list(strat9_map.values())
    strat10 = list(strat10_map.values())

    # 📝 Telegram 報告組裝
    tw_msg = f"🇹🇼 <b>【台股精準個股選股報告】</b>\n⚠️ <i>已排除 ETF & 過濾 20日均量 &lt; 2500張股票</i>\n⏰ 時間: {tw_time_str}\n"
    tw_msg += "───────────────────\n\n"
    
    tw_msg += "🚀 <b>【精準拉回策略】月/週趨勢多頭 + 日K月線打腳且柱狀體翻紅</b>\n"
    tw_msg += f"↳ {', '.join(strat10) if strat10 else '無符合標的。 💤'}\n\n"

    tw_msg += "🌕 <b>【策略一】月K MACD &gt; 0 + KD &gt; 20</b>\n"
    tw_msg += f"↳ {', '.join(strat1) if strat1 else '無符合標的。 💤'}\n\n"

    tw_msg += "📊 <b>【策略二】週K MACD &gt; 0 + KD &gt; 50</b>\n"
    tw_msg += f"↳ {', '.join(strat2) if strat2 else '無符合標的。 💤'}\n\n"

    tw_msg += "📈 <b>【策略三】日K MACD &gt; 0 + KD &gt; 20</b>\n"
    tw_msg += f"↳ {', '.join(strat3) if strat3 else '無符合標的。 💤'}\n\n"

    tw_msg += "⏱️ <b>【策略四】60分K MACD往0軸向上 + KD &gt; 20</b>\n"
    tw_msg += f"↳ {', '.join(strat4) if strat4 else '無符合標的。 💤'}\n\n"

    tw_msg += "⚡ <b>【策略五】30分K MACD往0軸向上 + KD &gt; 20</b>\n"
    tw_msg += f"↳ {', '.join(strat5) if strat5 else '無符合標的。 💤'}\n\n"

    tw_msg += "🎯 <b>【策略六】日/60分級別共振（策略三 ∩ 策略四）</b>\n"
    tw_msg += f"↳ {', '.join(strat6) if strat6 else '無重疊標的。 💤'}\n\n"

    tw_msg += "🔥 <b>【策略七】長線月/週級別共振（策略一 ∩ 策略二）</b>\n"
    tw_msg += f"↳ {', '.join(strat7) if strat7 else '無重疊標的。 💤'}\n\n"

    tw_msg += "💎 <b>【策略八】多週期全強勢共振（策略六 ∩ 策略七）</b>\n"
    tw_msg += f"↳ {', '.join(strat8) if strat8 else '無重疊標的。 💤'}\n\n"

    tw_msg += "❄️ <b>【策略九】反向弱勢防守（全週期 KD &lt; 70 且跌破10日線）</b>\n"
    tw_msg += f"↳ {', '.join(strat9) if strat9 else '無符合標的。 💤'}\n"

    send_telegram_message(tw_msg)
    print(f"✅ 策略報告發送完成！總耗時: {time.time() - start_time:.1f} 秒")
