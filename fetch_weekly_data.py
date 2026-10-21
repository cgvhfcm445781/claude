"""
エントリー/決済ルールを自由に検証するための週足データを取得して保存するスクリプト。
（GitHub Actions の fetch_weekly_data.yml から実行される想定）

出力（data/ 以下、parquet）
- weekly_prices_{0..3}.parquet : 全銘柄の週足(W-FRI)。確定済みの週のみ
    ticker, week, Open, High, Low, Close, Volume, Value, first_day, split_factor
    split_factor: その週より後に起きた株式分割の累積倍率。
                  「分割前の実際の株価 ≒ Close × split_factor」（配当調整分の誤差は残る）
- market_weekly.parquet : 日経平均(^N225) と TOPIX連動ETF(1306.T) の週足
- meta.parquet : ticker, name, sector, market
"""
import os
import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from Weekly_backtest import get_jpx_stock_list, clean_daily, build_weekly

warnings.simplefilter('ignore', FutureWarning)

DATA_DIR = "data"
PERIOD = "20y"   # 2008年(リーマンショック)のような長い下落局面も検証に含めるため
BATCH = 50
N_SHARDS = 4   # 1ファイルあたりのサイズをGitHubの上限(100MB)より十分小さくするため分割


def to_weekly(df_d):
    splits = df_d['Stock Splits'].replace(0, np.nan) if 'Stock Splits' in df_d else pd.Series(dtype=float)
    df_d = clean_daily(df_d[['Open', 'High', 'Low', 'Close', 'Volume']].copy())
    w = build_weekly(df_d)
    if len(w) == 0:
        return w
    # 各週より「後」に起きた分割の累積倍率
    sp = splits.dropna()
    w['split_factor'] = [float(sp[sp.index > d].prod()) if len(sp) else 1.0 for d in w['last_day']]
    return w


def download_batch(tickers):
    raw = yf.download(tickers, period=PERIOD, interval="1d", group_by='ticker', actions=True,
                      auto_adjust=True, repair=True, threads=True, progress=False)
    out = {}
    for t in tickers:
        try:
            d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            d = d.dropna(how='all')
            if getattr(d.index, 'tz', None) is not None:
                d.index = d.index.tz_localize(None)
            if len(d) >= 300:
                out[t] = d
        except KeyError:
            pass
    return out


def main():
    start = time.time()
    os.makedirs(DATA_DIR, exist_ok=True)
    meta = get_jpx_stock_list().rename(columns={'Ticker': 'ticker', '銘柄名': 'name',
                                                '33業種区分': 'sector', 'Market': 'market'})
    meta.to_parquet(os.path.join(DATA_DIR, 'meta.parquet'), index=False)

    frames = []
    tickers = meta['ticker'].tolist()
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i + BATCH]
        for attempt in range(3):
            try:
                got = download_batch(batch)
                break
            except Exception as e:
                print(f"⚠ batch {i} 失敗 ({e})、再試行 {attempt + 1}")
                time.sleep(10)
        else:
            continue
        for t, d in got.items():
            try:
                w = to_weekly(d)
                if len(w) >= 60:
                    frames.append(w.reset_index(names='week').assign(ticker=t))
            except Exception as e:
                print(f"⚠ {t}: {e}")
        print(f"{min(i + BATCH, len(tickers))}/{len(tickers)} 銘柄 ({time.time() - start:.0f}秒)")

    df = pd.concat(frames, ignore_index=True)
    for c in ['Open', 'High', 'Low', 'Close', 'split_factor']:
        df[c] = df[c].astype('float32')
    df['Volume'] = df['Volume'].astype('float64')
    df['Value'] = df['Value'].astype('float64')
    cols = ['ticker', 'week', 'Open', 'High', 'Low', 'Close', 'Volume', 'Value', 'first_day', 'split_factor']
    df = df[cols].sort_values(['ticker', 'week'])
    shard = df['ticker'].map(lambda t: sum(map(ord, t)) % N_SHARDS)  # 実行ごとに変わらない振り分け
    for s in range(N_SHARDS):
        df[shard == s].to_parquet(os.path.join(DATA_DIR, f'weekly_prices_{s}.parquet'),
                                  index=False, compression='zstd')

    mk = []
    for sym in ['^N225', '1306.T']:
        d = yf.download(sym, period=PERIOD, interval="1d", auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        if getattr(d.index, 'tz', None) is not None:
            d.index = d.index.tz_localize(None)
        w = build_weekly(d[['Open', 'High', 'Low', 'Close', 'Volume']].dropna())
        mk.append(w[['Open', 'High', 'Low', 'Close']].reset_index(names='week').assign(symbol=sym))
    pd.concat(mk).to_parquet(os.path.join(DATA_DIR, 'market_weekly.parquet'), index=False)

    print(f"✅ 完了: {df['ticker'].nunique()} 銘柄, {len(df)} 行, {time.time() - start:.0f}秒")
    for f in sorted(os.listdir(DATA_DIR)):
        print(f"  {f}: {os.path.getsize(os.path.join(DATA_DIR, f)) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
