import os
import requests
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

TELEGRAM_TOKEN = os.environ.get("TG_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TG_CHAT_ID")

def get_all_taiwan_stock_tickers():
    """抓取全台股普通股清單"""
    tickers = []
    try:
        url_twse = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
        res_twse = requests.get(url_twse, timeout=10)
        if res_twse.status_code == 200:
            for item in res_twse.json():
                code = item.get('Code', '')
                if len(code) == 4 and code.isdigit():
                    tickers.append(f"{code}.TW")

        url_tpex = "https://www.tpex.org.tw/openapi/v1/mopsfront_t187ap03_O"
        res_tpex = requests.get(url_tpex, timeout=10)
        if res_tpex.status_code == 200:
            for item in res_tpex.json():
                code = item.get('SecuritiesCompanyCode', '')
                if len(code) == 4 and code.isdigit():
                    tickers.append(f"{code}.TWO")
    except Exception as e:
        print(f"取得全市場清單失敗: {e}")
        tickers = ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW"]
        
    return list(set(tickers))

def calculate_kd(df, n=9):
    """通用 KD 指標計算 (9, 3, 3)"""
    low_min = df['Low'].rolling(n).min()
    high_max = df['High'].rolling(n).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    rsv = rsv.fillna(50)  # 防呆填補
    k = rsv.ewm(com=2).mean()
    d = k.ewm(com=2).mean()
    return k, d

def check_stock_strategy(ticker_symbol):
    """檢查單一股票多週期條件"""
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # 抓取各週期 K 線資料（適度拉長 period 確保資料足夠計算 MA20 與 KD）
        df_monthly = ticker.history(period="3y", interval="1mo")
        df_weekly  = ticker.history(period="1y", interval="1wk")
        df_daily   = ticker.history(period="3mo", interval="1d")
        df_60m     = ticker.history(period="1mo", interval="60m")
        df_30m     = ticker.history(period="1mo", interval="30m")

        if df_monthly.empty or df_weekly.empty or df_daily.empty or df_60m.empty or df_30m.empty:
            return None

        # 1. 濾除成交量過低標的（近一日成交張數 < 500 張）
        if df_daily['Volume'].iloc[-1] < 500000:
            return None

        # 2. 月 K 多頭 (收盤 > MA20)
        df_monthly['MA20'] = df_monthly['Close'].rolling(20).mean()
        if len(df_monthly) < 20 or df_monthly['Close'].iloc[-1] <= df_monthly['MA20'].iloc[-1]:
            return None

        # 3. 週 K 條件：週 K 的 KD > 60
        k_weekly, d_weekly = calculate_kd(df_weekly)
        if len(df_weekly) < 20 or k_weekly.iloc[-1] <= 60 or d_weekly.iloc[-1] <= 60:
            return None

        # 4. 日 K 低檔 KD 金叉 (前日 K < 35，當日 K > D)
        k_daily, d_daily = calculate_kd(df_daily)
        daily_k_now = k_daily.iloc[-1]
        daily_d_now = d_daily.iloc[-1]
        daily_k_prev = k_daily.iloc[-2]

        if not (daily_k_now > daily_d_now and daily_k_prev < 35):
            return None

        # 5. 60K 條件：60K 的 KD > 50
        k_60m, d_60m = calculate_kd(df_60m)
        if len(df_60m) < 20 or k_60m.iloc[-1] <= 50 or d_60m.iloc[-1] <= 50:
            return None

        # 6. 30K 條件：30K 的 KD > 50
        k_30m, d_30m = calculate_kd(df_30m)
        if len(df_30m) < 20 or k_30m.iloc[-1] <= 50 or d_30m.iloc[-1] <= 50:
            return None

        clean_symbol = ticker_symbol.replace(".TW", "").replace(".TWO", "")
        close_price = round(df_daily['Close'].iloc[-1], 2)
        return f"• <b>{clean_symbol}</b> (收盤價: {close_price})"

    except Exception:
        return None

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        res = requests.post(url, json=payload, timeout=10)
        print(f"Telegram 發送結果狀態碼: {res.status_code}")
    except Exception as e:
        print(f"Telegram 發送失敗: {e}")

def main():
    all_tickers = get_all_taiwan_stock_tickers()
    total_count = len(all_tickers)
    print(f"取得全市場共 {total_count} 檔股票，開始進行多線程篩選...")

    selected_stocks = []
    
    # 採用多線程 ThreadPoolExecutor，設定 10 個平行工作任務
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(check_stock_strategy, symbol): symbol for symbol in all_tickers}
        completed = 0
        for future in as_completed(future_to_ticker):
            completed += 1
            if completed % 200 == 0:
                print(f"進度: {completed}/{total_count}")
            
            result = future.result()
            if result:
                selected_stocks.append(result)

    print(f"篩選完成，共找到 {len(selected_stocks)} 檔符合標的。")

    if selected_stocks:
        stock_text = "\n".join(selected_stocks)
        msg = f"<b>📈 全台股多週期共振轉折選股結果</b>\n\n{stock_text}\n\n<i>條件：月MA20多頭 + 週KD>60 + 日K低檔金叉 + 60k/30k KD>50</i>"
    else:
        msg = "<b>📈 全台股多週期選股結果</b>\n\n今日全市場無符合條件之標的。"

    send_telegram_message(msg)

if __name__ == "__main__":
    main()
