import os, jpx_alerts
os.makedirs('probe_jpx', exist_ok=True)
df = jpx_alerts.fetch()
open('probe_jpx/report_delist.txt', 'w', encoding='utf-8').write(df.to_string() + "\n\nTOB判定: " + str(df['理由'].map(jpx_alerts.is_tob).value_counts().to_dict()))
# retry
