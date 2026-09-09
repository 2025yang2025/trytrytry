import os
import requests
import pandas as pd
import yfinance as yf

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def get_all_taiwan_stock_tickers():
    """從證交所 OpenAPI 抓取全台股上市與上櫃股票代碼"""
    tickers = []
    try:
        # 1. 抓取上市股票清單 (TWSE)
        url_twse = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
        res_twse = requests.get(url_twse, timeout=10)
        if res_twse.status_code == 200:
            for item in res_twse.json():
                code = item.get('Code', '')
                # 排除 ETF、權證、受益憑證（僅保留普通股 4 位數代號）
                if len(code) == 4 and code.isdigit():
                    tickers.append(f"{code}.TW")

        # 2. 抓取上櫃股票清單 (TPEx)
        url_tpex = "https://www.tpex.org.tw/openapi/v1/mopsfront_t187ap03_O"
        res_tpex = requests.get(url_tpex, timeout=10)
        if res_tpex.status_code == 200:
            for item in res_tpex.json():
                code = item.get('SecuritiesCompanyCode', '')
                if len(code) == 4 and code.isdigit():
                    tickers.append(f"{code}.TWO")
    except Exception as e:
        print(f"取得全市場清單失敗: {e}")
        # 備用核心股票清單
        tickers = ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW", "3037.TW"]
        
    return list(set(tickers))

def check_stock_strategy(ticker_symbol):
    """檢查單一股票多週期條件"""
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # 抓取各週期 K 線
        df_monthly = ticker.history(period="2y", interval="1mo")
        df_weekly  = ticker.history(period="1y", interval="1wk")
        df_daily   = ticker.history(period="6mo", interval="1d")
        df_60m     = ticker.history(period="1mo", interval="60m")
        df_30m     = ticker.history(period="1mo", interval="30m")

        if df_monthly.empty or df_weekly.empty or df_daily.empty or df_60m.empty or df_30m.empty:
            return False, ""

        # 濾除流動性太差（成交量過低）的股票，避免誤判
        if df_daily['Volume'].iloc[-1] < 500000: # 500張
            return False, ""

        # 1. 月 K 多頭
        df_monthly['MA20'] = df_monthly['Close'].rolling(20).mean()
        monthly_bull = df_monthly['Close'].iloc[-1] > df_monthly['MA20'].iloc[-1]

        # 2. 週 K 多頭
        df_weekly['MA20'] = df_weekly['Close'].rolling(20).mean()
        weekly_bull = df_weekly['Close'].iloc[-1] > df_weekly['MA20'].iloc[-1]

        # 3. 日 K 低檔 KD 金叉
        low_min = df_daily['Low'].rolling(9).min()
        high_max = df_daily['High'].rolling(9).max()
        rsv = (df_daily['Close'] - low_min) / (high_max - low_min) * 100
        df_daily['K'] = rsv.ewm(com=2).mean()
        df_daily['D'] = df_daily['K'].ewm(com=2).mean()
        daily_reversal = (df_daily['K'].iloc[-1] > df_daily['D'].iloc[-1]) and (df_daily['K'].iloc[-2] < 35)

        # 4. 60K 站上 MA20 且 MA20 走平向上
        df_60m['MA20'] = df_60m['Close'].rolling(20).mean()
        k60_reversal = (df_60m['Close'].iloc[-1] > df_60m['MA20'].iloc[-1]) and (df_60m['MA20'].iloc[-1] >= df_60m['MA20'].iloc[-2])

        # 5. 30K 站上 MA20
        df_30m['MA20'] = df_30m['Close'].rolling(20).mean()
        k30_reversal = (df_30m['Close'].iloc[-1] > df_30m['MA20'].iloc[-1])

        if monthly_bull and weekly_bull and daily_reversal and k60_reversal and k30_reversal:
            close_price = round(df_daily['Close'].iloc[-1], 2)
            return True, f"收盤價: {close_price}"
            
    except Exception:
        return False, ""
        
    return False, ""

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    requests.post(url, json=payload)

def main():
    all_tickers = get_all_taiwan_stock_tickers()
    print(f"取得全市場共 {len(all_tickers)} 檔股票，開始進行多週期篩選...")

    selected_stocks = []
    for symbol in all_tickers:
        is_match, info = check_stock_strategy(symbol)
        if is_match:
            clean_symbol = symbol.replace(".TW", "").replace(".TWO", "")
            selected_stocks.append(f"• <b>{clean_symbol}</b> ({info})")

    if selected_stocks:
        stock_text = "\n".join(selected_stocks)
        msg = f"<b>📈 全台股多週期共振轉折選股結果</b>\n\n{stock_text}\n\n<i>條件：月/週多頭 + 日K低檔金叉 + 60k/30k轉轉</i>"
    else:
        msg = "<b>📈 全台股多週期選股結果</b>\n\n今日全市場無符合條件之標的。"

    send_telegram_message(msg)

if __name__ == "__main__":
    main()
