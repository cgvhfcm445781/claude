"""
週次シグナル通知（Discord）。iPhoneのショートカットなどから GitHub Actions の weekly_alert.yml を手動実行して使う。

通知する内容（いずれも直近の確定した週の金曜終値時点。週の途中に実行した場合は前週末時点）
1. 急落リバウンド: 日経平均の4週騰落が -7%以下ならシグナル。買う候補（4週騰落の下位20%を下げた順に最大10銘柄）と
   +20%の利確の目安、最長保有の期限（8週後）
2. 割安株: 低PER & 12か月線より上 & 信用残が重くない 上位10銘柄（13週ごとに入れ替え）
3. 順張り: 過去12か月（直近1か月を除く）の上昇率が高い & 信用残が重くない 上位10銘柄（13週ごとに入れ替え）
   ※2と3は半分ずつ持つ想定（バックテストでは各10銘柄の半々で年+25.8%、最大DD-19%）

データ
- 株価: yfinance（実行のたびに直近2年分を取得）
- 決算・信用残: J-Quants（data_jq/ に保存済みのデータに、足りない日の分だけ追加取得してコミット）
環境変数: DISCORD_WEBHOOK_URL, JQUANTS_API_KEY
"""
import json
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from Weekly_backtest import get_jpx_stock_list, clean_daily

warnings.simplefilter('ignore')
DISCORD_WEBHOOK_URL = os.environ.get('DISCORD_WEBHOOK_URL', '')
JQ_DIR = 'data_jq'
REBALANCE_ANCHOR = pd.Timestamp('2026-08-14')   # 割安株・順張りの入れ替え基準日（ここから13週ごと）
CRASH_TH, LOSER_PCT, CRASH_N, TARGET, HOLD_WEEKS = -7, 0.20, 10, 0.20, 8
PICK_N = 10
MIN_DAILY_VALUE = 50_000_000
PRICE_MIN, PRICE_MAX = 1000, 4000


# ---------------- 通知 ----------------
def send_discord(content):
    print(content)
    if not DISCORD_WEBHOOK_URL:
        print("⚠ DISCORD_WEBHOOK_URL が未設定のため送信をスキップ")
        return
    lines, chunk = content.split('\n'), ''
    for ln in lines:   # 2000文字制限があるので行単位で分割
        if len(chunk) + len(ln) + 1 > 1900:
            requests.post(DISCORD_WEBHOOK_URL, json={'content': chunk}, timeout=15)
            time.sleep(0.5)
            chunk = ''
        chunk += ln + '\n'
    if chunk.strip():
        requests.post(DISCORD_WEBHOOK_URL, json={'content': chunk}, timeout=15)


# ---------------- J-Quants（足りない分だけ追加取得） ----------------
def code_to_ticker(code):
    c = str(code)
    return (c[:4] if len(c) == 5 else c) + '.T'


def update_jquants():
    key = os.environ.get('JQUANTS_API_KEY', '')
    fin = pd.read_parquet(os.path.join(JQ_DIR, 'fin_summary.parquet'))
    mar = pd.read_parquet(os.path.join(JQ_DIR, 'margin.parquet'))
    if not key:
        print("⚠ JQUANTS_API_KEY が未設定のため、保存済みのデータだけで計算します")
        return fin, mar
    from jquantsapi import ClientV2
    cli = ClientV2(api_key=key)
    today = pd.Timestamp.now(tz='Asia/Tokyo').tz_localize(None).normalize()
    new_fin, new_mar = [], []
    for d in pd.bdate_range(pd.to_datetime(fin['DiscDate']).max() + pd.Timedelta(days=1), today):
        try:
            x = cli.get_fin_summary(date_yyyymmdd=d.strftime('%Y%m%d'))
            if len(x):
                new_fin.append(x)
        except Exception as e:
            print(f"⚠ 決算 {d:%Y-%m-%d}: {str(e)[:120]}")
        time.sleep(0.7)
    for d in pd.bdate_range(pd.to_datetime(mar['Date']).max() + pd.Timedelta(days=1), today):
        try:
            x = cli.get_mkt_margin_interest(date_yyyymmdd=d.strftime('%Y%m%d'))
            if len(x):
                new_mar.append(x)
        except Exception as e:
            print(f"⚠ 信用残 {d:%Y-%m-%d}: {str(e)[:120]}")
        time.sleep(0.7)

    def merge(old, new, path):
        if not new:
            return old
        df = pd.concat([old] + new, ignore_index=True)
        df = df.astype({c: 'string' for c in df.columns if df[c].dtype == object}).drop_duplicates()
        df.to_parquet(path, index=False)
        return df
    fin = merge(fin, new_fin, os.path.join(JQ_DIR, 'fin_summary.parquet'))
    mar = merge(mar, new_mar, os.path.join(JQ_DIR, 'margin.parquet'))
    print(f"J-Quants 追加取得: 決算 {sum(len(x) for x in new_fin)} 件, 信用残 {sum(len(x) for x in new_mar)} 件")
    return fin, mar


# ---------------- 株価 ----------------
def download_prices(tickers, batch=100):
    out = {}
    for i in range(0, len(tickers), batch):
        b = tickers[i:i + batch]
        for attempt in range(2):
            try:
                raw = yf.download(b, period='2y', interval='1d', group_by='ticker', auto_adjust=True,
                                  threads=True, progress=False)
                break
            except Exception as e:
                print(f"⚠ batch {i}: {e}")
                time.sleep(10)
        else:
            continue
        for t in b:
            try:
                d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                d = d.dropna(how='all')
                if getattr(d.index, 'tz', None) is not None:
                    d.index = d.index.tz_localize(None)
                if len(d) >= 260:
                    out[t] = clean_daily(d[['Open', 'High', 'Low', 'Close', 'Volume']].copy())
            except KeyError:
                pass
        print(f"株価 {min(i + batch, len(tickers))}/{len(tickers)}")
        time.sleep(1)
    return out


def last_completed_week_end(dates):
    """直近の確定した週の最終営業日。今日の足（引け前の可能性がある）は使わない"""
    now = pd.Timestamp.now(tz='Asia/Tokyo').tz_localize(None)
    dates = pd.DatetimeIndex(dates)
    if now.hour < 16:
        dates = dates[dates < now.normalize()]
    last = dates.max()
    fri = last + pd.Timedelta(days=(4 - last.weekday()) % 7)      # その週の金曜
    if last.weekday() == 4 or now.normalize() > fri:              # 金曜の引け後、または週末・翌週
        return last
    return dates[dates <= fri - pd.Timedelta(weeks=1)].max()      # 週の途中なら前週末


# ---------------- 計算 ----------------
def stock_features(prices, asof):
    rows = []
    for t, d in prices.items():
        d = d[d.index <= asof]
        if len(d) < 260 or d.index[-1] < asof - pd.Timedelta(days=7):
            continue
        C = d['Close']
        wk = C.resample('W-FRI').last().dropna()
        me = C.resample('ME').last()
        if (asof + pd.offsets.BDay(1)).month == asof.month:     # 月の途中なら当月の足はまだ確定していない
            me = me[me.index < asof.replace(day=1)]
        val20 = (C * d['Volume']).tail(20).mean()
        rows.append({'ticker': t, 'price': C.iloc[-1], 'value20': val20,
                     'ret4': (wk.iloc[-1] / wk.iloc[-5] - 1) * 100 if len(wk) > 5 else np.nan,
                     'mom12_1': C.iloc[-21] / C.iloc[-251] - 1 if len(C) > 251 else np.nan,
                     'm_above': me.iloc[-1] > me.tail(12).mean() if len(me) >= 12 else False})
    f = pd.DataFrame(rows)
    f['universe'] = (f['value20'] >= MIN_DAILY_VALUE) & f['price'].between(PRICE_MIN, PRICE_MAX)
    return f


def add_fundamentals(f, fin, mar, asof):
    fin = fin.copy()
    for c in ['FEPS', 'NxFEPS']:
        fin[c] = pd.to_numeric(fin[c], errors='coerce')
    fin['ticker'] = fin['Code'].map(code_to_ticker)
    fin = fin[fin['DocType'].str.contains('FinancialStatements', na=False)]
    fin = fin[pd.to_datetime(fin['DiscDate']) < asof]
    fin['fwd_eps'] = np.where(fin['CurPerType'] == 'FY', fin['NxFEPS'], fin['FEPS'])
    last = fin.sort_values(['DiscDate', 'DiscTime']).groupby('ticker').last()
    f['fwd_eps'] = f['ticker'].map(last['fwd_eps'])
    f['E/P'] = f['fwd_eps'] / f['price']
    mar = mar.copy()
    mar['ticker'] = mar['Code'].map(code_to_ticker)
    mar['Date'] = pd.to_datetime(mar['Date'])
    mar = mar[mar['Date'] <= asof - pd.Timedelta(days=3)]   # 公表済み（申込日の翌週）のものだけ
    latest = mar[mar['Date'] == mar['Date'].max()]
    lv = latest.groupby('ticker')['LongVol'].apply(lambda s: pd.to_numeric(s, errors='coerce').sum())
    sv = latest.groupby('ticker')['ShrtVol'].apply(lambda s: pd.to_numeric(s, errors='coerce').sum())
    vol_w = f['value20'] / f['price'] * 5
    f['long_w'] = f['ticker'].map(lv) / vol_w
    f['short_w'] = f['ticker'].map(sv) / vol_w
    u = f['universe']
    f['heavy_margin'] = (f['long_w'] > f.loc[u, 'long_w'].median() * 2) | (f['short_w'] > f.loc[u, 'short_w'].median() * 2)
    return f, mar['Date'].max()


def main():
    t0 = time.time()
    fin, mar = update_jquants()
    meta = get_jpx_stock_list()
    names = dict(zip(meta['Ticker'], meta['銘柄名']))
    prices = download_prices(meta['Ticker'].tolist())
    mk = {s: yf.download(s, period='1y', auto_adjust=True, progress=False)['Close'].squeeze().dropna() for s in ('^N225', '^VIX')}
    asof = last_completed_week_end(mk['^N225'].index)
    nk_w = mk['^N225'][mk['^N225'].index <= asof].resample('W-FRI').last().dropna()
    nk4 = (nk_w.iloc[-1] / nk_w.iloc[-5] - 1) * 100
    vix_w = mk['^VIX'][mk['^VIX'].index <= asof + pd.Timedelta(days=1)].resample('W-FRI').last().dropna()
    vix_off = vix_w.iloc[-1] < vix_w.tail(4).max() * 0.85

    f = stock_features(prices, asof)
    f, mar_date = add_fundamentals(f, fin, mar, asof)
    f['name'] = f['ticker'].map(names)
    u = f[f['universe']].copy()
    nm = lambda r: f"{r.ticker.replace('.T', '')} {str(r['name'])[:10]}"

    lines = [f"📊 **週次シグナル**（{asof:%Y-%m-%d} 終値時点）", '']
    # 1. 急落リバウンド
    crash = nk4 <= CRASH_TH
    lines.append(f"■ 急落リバウンド: 日経平均の4週騰落 {nk4:+.1f}%（発動 {CRASH_TH}%以下） / VIX {vix_w.iloc[-1]:.1f}"
                 f"{'（ピークアウト済み）' if vix_off else ''}")
    if crash:
        u['r4rank'] = u['ret4'].rank(pct=True)
        c = u[u['r4rank'] <= LOSER_PCT].sort_values('ret4').head(CRASH_N)
        lines.append(f"🚨 **シグナルあり** → 翌営業日の寄り付きで買い、+{TARGET:.0%}で利確の指値、"
                     f"届かなければ {(asof + pd.Timedelta(weeks=HOLD_WEEKS)):%Y-%m-%d} の終値で売り")
        for _, r in c.iterrows():
            lines.append(f"・{nm(r)}  {r.price:,.0f}円（4週{r.ret4:+.1f}%） 利確目安 {r.price * (1 + TARGET):,.0f}円")
        lines.append("※急落中は、割安株・順張りの資金をこちらに回す（終わったら戻す）")
    else:
        lines.append("シグナルなし")
    lines.append('')
    # 2・3. 割安株・順張り
    k = int(np.ceil((asof - REBALANCE_ANCHOR).days / 91))
    next_reb = REBALANCE_ANCHOR + pd.Timedelta(weeks=13 * max(k, 0))
    is_reb = abs((asof - next_reb).days) <= 3
    if not is_reb and next_reb < asof:
        next_reb += pd.Timedelta(weeks=13)
    lines.append(f"■ 割安株・順張り（各{PICK_N}銘柄を半分ずつ、13週ごとに入れ替え）"
                 f"{' 🔁 **今週が入れ替え週**（翌営業日の寄り付きで入れ替え）' if is_reb else f' 次の入れ替え: {next_reb:%Y-%m-%d}の引け後'}")
    light = ~u['heavy_margin'].fillna(False)
    v = u[light & u['m_above'] & (u['E/P'] > 0)].nlargest(PICK_N, 'E/P')
    lines.append("【割安株】低PER & 12か月線より上 & 信用残が重くない")
    for _, r in v.iterrows():
        lines.append(f"・{nm(r)}  {r.price:,.0f}円  PER {1 / r['E/P']:.1f}倍")
    m = u[light & u['mom12_1'].notna()].nlargest(PICK_N, 'mom12_1')
    lines.append("【順張り】過去12か月（直近1か月を除く）の上昇率上位 & 信用残が重くない")
    for _, r in m.iterrows():
        lines.append(f"・{nm(r)}  {r.price:,.0f}円  12か月 {r.mom12_1 * 100:+.0f}%")
    lines.append('')
    lines.append(f"（株価 {len(f)}銘柄・ユニバース {len(u)}銘柄 / 信用残 {mar_date:%Y-%m-%d}時点 / 処理 {time.time() - t0:.0f}秒）")
    send_discord('\n'.join(lines))


if __name__ == '__main__':
    main()
