"""一時的: Yahooで1日±50%超の動きがあった日の前後について、J-Quantsの株価を取得して比べる（GitHub Actionsで実行）"""
import os, time
import pandas as pd
from jquantsapi import ClientV2
cli = ClientV2(api_key=os.environ['JQUANTS_API_KEY'])
ev = pd.read_csv('probe_jq_events.csv', parse_dates=['week'])
out = []
for t, dt in zip(ev['ticker'], ev['week']):
    code = t.replace('.T', '')
    try:
        x = cli.get_eq_bars_daily(code=code, from_yyyymmdd=(dt - pd.Timedelta(days=10)).strftime('%Y%m%d'),
                                  to_yyyymmdd=(dt + pd.Timedelta(days=10)).strftime('%Y%m%d'))
        x['ticker'], x['event'] = t, dt
        out.append(x)
    except Exception as e:
        print(t, dt.date(), 'error', str(e)[:150])
    time.sleep(1.1)
os.makedirs('probe_jq', exist_ok=True)
pd.concat(out).to_csv('probe_jq/jq_bars.csv', index=False)
print('saved', sum(len(x) for x in out))
