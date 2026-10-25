"""一時的: J-Quantsの日足を約2年分まとめて取得して保存し、分割調整を確認する（GitHub Actionsで実行）"""
import os
import pandas as pd
from jquantsapi import ClientV2
import jq_prices
today = pd.Timestamp.now(tz='Asia/Tokyo').tz_localize(None).normalize()
start = today - pd.DateOffset(years=2)
jq_prices.update(ClientV2(api_key=os.environ['JQUANTS_API_KEY']), start, today)
prices, last = jq_prices.load(start)
print('銘柄数', len(prices), '最終日', last)
for t in ['8227.T', '8303.T', '6740.T', '7203.T']:
    d = prices.get(t)
    if d is not None:
        r = d['Close'].pct_change().abs() * 100
        print(t, len(d), '日  最大の1日の動き', round(r.max(), 1), '%', r.idxmax().date(), ' 直近終値', d['Close'].iloc[-1])
