"""
月足・週足・日足を組み合わせた検証用に、全銘柄の日足データを取得して保存するスクリプト。
（GitHub Actions の fetch_daily_data.yml から実行され、結果は data-daily ブランチに置かれる）

出力（data_daily/ 以下、parquet）
- daily_{0..7}.parquet : ticker, date, Open, High, Low, Close, Volume, split_factor
    split_factor: その日より後に起きた株式分割の累積倍率（分割前の実際の株価 ≒ Close × split_factor）
"""
import glob
import os
import time
import warnings
import numpy as np
import pandas as pd
from Weekly_backtest import clean_daily
from fetch_weekly_data import download_batch, BATCH, SLEEP_SEC

warnings.simplefilter('ignore', FutureWarning)

OUT_DIR = "data_daily"
N_SHARDS = 8
MIN_KEEP_RATIO = 0.5   # 日足は専用ブランチに置くので、半分以上取れていれば保存する（欠けた銘柄は分析から外れるだけ）


def to_daily(d):
    splits = d['Stock Splits'].replace(0, np.nan) if 'Stock Splits' in d else pd.Series(dtype=float)
    out = clean_daily(d[['Open', 'High', 'Low', 'Close', 'Volume']].copy())
    sp = splits.reindex(out.index).fillna(1.0)
    # 各日より「後」に起きた分割の累積倍率（逆順の累積積を1日ずらす）
    out['split_factor'] = sp[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    return out


def main():
    start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    meta = pd.read_parquet(os.path.join('data', 'meta.parquet'))
    tickers = meta['ticker'].tolist()
    frames = []
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i + BATCH]
        got = {}
        for attempt in range(3):
            try:
                got = download_batch(batch)
                break
            except Exception as e:
                print(f"⚠ batch {i} 失敗 ({e})、再試行 {attempt + 1}")
                time.sleep(10)
        missing = [t for t in batch if t not in got]
        if missing:
            time.sleep(SLEEP_SEC * 3)
            try:
                got.update(download_batch(missing))
            except Exception as e:
                print(f"⚠ 再取得失敗 ({e})")
        time.sleep(SLEEP_SEC)
        for t, d in got.items():
            try:
                x = to_daily(d)
                if len(x) >= 300:
                    frames.append(x.reset_index(names='date').assign(ticker=t))
            except Exception as e:
                print(f"⚠ {t}: {e}")
        print(f"{min(i + BATCH, len(tickers))}/{len(tickers)} 銘柄 ({time.time() - start:.0f}秒)")

    df = pd.concat(frames, ignore_index=True)
    # 週足データの銘柄数と比べて半分未満なら、アクセス制限とみなして保存しない
    n_weekly = pd.concat([pd.read_parquet(f, columns=['ticker'])
                          for f in glob.glob(os.path.join('data', 'weekly_prices_*.parquet'))])['ticker'].nunique()
    if df['ticker'].nunique() < n_weekly * MIN_KEEP_RATIO:
        raise SystemExit(f"⚠ 日足の銘柄数 {df['ticker'].nunique()} が週足の {n_weekly} の半分未満のため保存しません")
    print(f"日足 {df['ticker'].nunique()} 銘柄 / 週足 {n_weekly} 銘柄")
    for c in ['Open', 'High', 'Low', 'Close', 'Volume', 'split_factor']:
        df[c] = df[c].astype('float32')
    df = df[['ticker', 'date', 'Open', 'High', 'Low', 'Close', 'Volume', 'split_factor']].sort_values(['ticker', 'date'])
    shard = df['ticker'].map(lambda t: sum(map(ord, t)) % N_SHARDS)
    for s in range(N_SHARDS):
        df[shard == s].to_parquet(os.path.join(OUT_DIR, f'daily_{s}.parquet'), index=False, compression='zstd')
    print(f"✅ 完了: {df['ticker'].nunique()} 銘柄, {len(df)} 行, {time.time() - start:.0f}秒")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {f}: {os.path.getsize(os.path.join(OUT_DIR, f)) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
