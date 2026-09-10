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
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

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

        macd_line, signal_line, hist = calculate_macd(c_tf)
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

def check_macd_negative_reducing_kd(df_tf, kd_threshold=50):
    """ 策略四：60分K MACD 綠柱縮小（柱狀體負值且向上增加） + KD > kd_threshold """
    try:
        df_clean = df_tf.dropna(subset=['Close', 'High', 'Low'])
        if len(df_clean) < 30: return False, 0.0
        c_tf = df_clean['Close'].astype(float)

        _, _, hist = calculate_macd(c_tf)
        
        # MACD 綠柱縮小 (柱狀體負值且最新一根數值大於前一根)
        is_hist_negative_reducing = (hist.iloc[-1] < 0) and (hist.iloc[-1] > hist.iloc[-2])

        k_ser, d_ser = calculate_kd(df_clean)
        if k_ser.empty or d_ser.empty: return False, 0.0
        
        k_val = k_ser.iloc[-1]
        d_val = d_ser.iloc[-1]
        is_kd_cond = (k_val > kd_threshold) and (d_val > kd_threshold)

        if is_hist_negative_reducing and is_kd_cond:
            return True, c_tf.iloc[-1]
    except Exception:
        pass
    return False, 0.0

def check_strat_5_30m(df_30m, df_daily, kd_threshold=50):
    """ 策略五：30分K MACD > 0 + KD > kd_threshold + 價格站上日K 5日均線 """
    try:
        df_clean_30m = df_30m.dropna(subset=['Close', 'High', 'Low'])
        df_clean_d = df_daily.dropna(subset=['Close'])
        
        if len(df_clean_30m) < 30 or len(df_clean_d) < 5: 
            return False, 0.0

        c_30m = df_clean_30m['Close'].astype(float)
        
        # 1. 30分K MACD > 0
        macd_line, _, _ = calculate_macd(c_30m)
        if pd.isna(macd_line.iloc[-1]) or macd_line.iloc[-1] <= 0:
            return False, 0.0

        # 2. 30分K KD > kd_threshold (改為 50)
        k_ser, d_ser = calculate_kd(df_clean_30m)
        if k_ser.empty or d_ser.empty: return False, 0.0
        if k_ser.iloc[-1] <= kd_threshold or d_ser.iloc[-1] <= kd_threshold:
            return False, 0.0

        # 3. 價格站上日 K 5 日均線
        ma5_daily = df_clean_d['Close'].rolling(window=5).mean().iloc[-1]
        latest_price = c_30m.iloc[-1]

        if latest_price > ma5_daily:
            return True, latest_price
    except Exception:
        pass
    return False, 0.0

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

    print("🚀 啟動【台股 5 大多頭選股系統】...")
    tech_scan_pool = fetch_all_taiwan_market_tickers()

    print(f"⏳ 步驟 1: 下載全市場日K、週K與月K資料 (共 {len(tech_scan_pool)} 檔)...")
    full_df_daily = safe_download_yf(tech_scan_pool, period="1y", interval="1d", chunk_size=250)
    full_df_weekly = safe_download_yf(tech_scan_pool, period="2y", interval="1wk", chunk_size=250)
    full_df_monthly = safe_download_yf(tech_scan_pool, period="5y", interval="1mo", chunk_size=250)

    strat1, strat2, strat3, strat4, strat5 = [], [], [], [], []
    heavy_scan_pool = []

    print("⏳ 步驟 2: 進行月K、週K、日K策略篩選...")
    for ticker in tech_scan_pool:
        try:
            if ticker not in full_df_daily.columns.levels[1]: continue
            df_d = full_df_daily.xs(ticker, axis=1, level=1)
            if df_d.empty or len(df_d.dropna(subset=['Close'])) < 120: continue 
            
            # 過濾：20日均量 >= 1000張
            if df_d['Volume'].rolling(window=20).mean().iloc[-1] / 1000 < 1000: continue

            name_zh = DYNAMIC_STOCK_NAMES.get(ticker, "")
            stock_label = f"<code>{ticker}</code>(<i>{name_zh}</i>)" if name_zh else f"<code>{ticker}</code>"

            # 🛠️ 【策略一：月K MACD > 0 + KD > 50】
            if ticker in full_df_monthly.columns.levels[1]:
                df_m = full_df_monthly.xs(ticker, axis=1, level=1)
                res1, _ = check_macd_above_zero_kd(df_m, kd_threshold=50)
                if res1:
                    strat1.append(f"{stock_label}[{df_d['Close'].dropna().iloc[-1]:.2f}元]")

            # 🛠️ 【策略二：週K MACD > 0 + KD > 20】
            if ticker in full_df_weekly.columns.levels[1]:
                df_w = full_df_weekly.xs(ticker, axis=1, level=1)
                res2, _ = check_macd_above_zero_kd(df_w, kd_threshold=20)
                if res2:
                    strat2.append(f"{stock_label}[{df_d['Close'].dropna().iloc[-1]:.2f}元]")

            # 🛠️ 【策略三：日K MACD > 0 + KD > 20】
            res3, price3 = check_macd_above_zero_kd(df_d, kd_threshold=20)
            if res3:
                strat3.append(f"{stock_label}[{price3:.2f}元]")

            # 收集適合掃描分 K 的精選名單 (股價站上 20日線)
            if df_d['Close'].dropna().iloc[-1] > df_d['Close'].rolling(20).mean().iloc[-1]:
                heavy_scan_pool.append(ticker)

        except Exception:
            continue

    # ⏳ 步驟 3: 下載 30分K 與 60分K 資料
    final_heavy_pool = heavy_scan_pool[:50]
    if final_heavy_pool:
        print(f"⏳ 步驟 3: 下載精選 {len(final_heavy_pool)} 檔標的之 30分K 與 60分K 資料...")
        full_df_30m = safe_download_yf(final_heavy_pool, period="1mo", interval="30m", chunk_size=50)
        full_df_60m = safe_download_yf(final_heavy_pool, period="1mo", interval="60m", chunk_size=50)

        for ticker in final_heavy_pool:
            try:
                name_zh = DYNAMIC_STOCK_NAMES.get(ticker, "")
                stock_label = f"<code>{ticker}</code>(<i>{name_zh}</i>)" if name_zh else f"<code>{ticker}</code>"
                df_d = full_df_daily.xs(ticker, axis=1, level=1)

                # 🛠️ 【策略四：60分K MACD綠柱縮小 + KD > 50】
                if ticker in full_df_60m.columns.levels[1]:
                    df_m60 = full_df_60m.xs(ticker, axis=1, level=1)
                    res4, price4 = check_macd_negative_reducing_kd(df_m60, kd_threshold=50)
                    if res4: 
                        strat4.append(f"{stock_label}[{price4:.2f}元]")

                # 🛠️ 【策略五：30分K MACD > 0 + KD > 50 + 站上5日線】
                if ticker in full_df_30m.columns.levels[1]:
                    df_m30 = full_df_30m.xs(ticker, axis=1, level=1)
                    res5, price5 = check_strat_5_30m(df_m30, df_d, kd_threshold=50)
                    if res5:
                        strat5.append(f"{stock_label}[{price5:.2f}元]")

            except Exception:
                continue

    # 📝 Telegram 報告組裝
    tw_msg = f"🇹🇼 <b>【台股 5 大精準選股報告】</b>\n⚠️ <i>已過濾 20日均量 &lt; 1000張之股票</i>\n⏰ 時間: {tw_time_str}\n"
    tw_msg += "───────────────────\n\n"
    
    tw_msg += "🌕 <b>【策略一】月K MACD &gt; 0 + KD &gt; 50</b>\n"
    tw_msg += f"↳ {', '.join(strat1) if strat1 else '無符合標的。 💤'}\n\n"

    tw_msg += "📊 <b>【策略二】週K MACD &gt; 0 + KD &gt; 20</b>\n"
    tw_msg += f"↳ {', '.join(strat2) if strat2 else '無符合標的。 💤'}\n\n"

    tw_msg += "📈 <b>【策略三】日K MACD &gt; 0 + KD &gt; 20</b>\n"
    tw_msg += f"↳ {', '.join(strat3) if strat3 else '無符合標的。 💤'}\n\n"

    tw_msg += "⏱️ <b>【策略四】60分K MACD綠柱縮小 + KD &gt; 50</b>\n"
    tw_msg += f"↳ {', '.join(strat4) if strat4 else '無符合標的。 💤'}\n\n"

    tw_msg += "⚡ <b>【策略五】30分K MACD &gt; 0 + KD &gt; 50 + 站上5日線</b>\n"
    tw_msg += f"↳ {', '.join(strat5) if strat5 else '無符合標的。 💤'}\n"

    send_telegram_message(tw_msg)
    print(f"✅ 策略報告發送完成！總耗時: {time.time() - start_time:.1f} 秒")
