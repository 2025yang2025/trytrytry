def fetch_all_taiwan_market_tickers():
    """ 下載全台股市場代碼（不限產業），並同步撈取證交所盤後估值資料 """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    all_tickers = []
    
    # 1. 撈取全市場基本交易資料（解除產業限制，真正全市場納入）
    try:
        url_twse = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
        res = requests.get(url_twse, headers=headers, timeout=10)
        if res.status_code == 200:
            for item in res.json():
                code = item.get("Code", "").strip()
                name = item.get("Name", "").strip()
                # 只要是 4 碼純數字的個股（上市/櫃主要股票）就全部納入
                if code.isdigit() and len(code) == 4:
                    ticker_id = f"{code}.TW"
                    all_tickers.append(ticker_id)
                    DYNAMIC_STOCK_NAMES[ticker_id] = name
    except Exception as e:
        print(f"⚠️ 撈取全市場名單異常: {e}")

    # 2. 撈取證交所官方個股本益比、股價淨值比 (每日盤後更新)
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

    if not all_tickers:
        backup_dict = {"2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科"}
        for k, v in backup_dict.items():
            all_tickers.append(k)
            DYNAMIC_STOCK_NAMES[k] = v
            
    return sorted(list(set(all_tickers)))
