"""
J-Quants（Lightプラン以上）の日足株価を、リポジトリに保存しながら使うモジュール。

- 保存先: data_jq/prices/prices_YYYYMM.parquet（月ごと。週次の実行では当月のファイルだけが変わる）
          data_jq/prices/done.json（取得済みの営業日。休日など空だった日も記録して取り直さない）
- 保存するのは調整前の四本値・出来高・売買代金と、J-Quants の調整係数（AdjFactor）
  → 調整後の株価は読み込むたびに計算する（後から株式分割があっても正しく調整されるように）
    調整後の株価 = 調整前の株価 × （その日より後の AdjFactor の累積積）
"""
import glob
import json
import os
import time
import numpy as np
import pandas as pd

DIR = os.path.join('data_jq', 'prices')
DONE = os.path.join(DIR, 'done.json')
COLS = {'Date': 'date', 'Code': 'code', 'O': 'Open', 'H': 'High', 'L': 'Low', 'C': 'Close', 'Vo': 'Volume',
        'Va': 'Value', 'AdjFactor': 'AdjFactor'}


def _code_to_ticker(code):
    c = str(code)
    return (c[:4] if len(c) == 5 else c) + '.T'


def update(cli, start, end, wait=1.1):
    """start〜end の営業日のうち、まだ取っていない日の全銘柄の日足を取得して保存する（Lightは60回/分）"""
    os.makedirs(DIR, exist_ok=True)
    done = set(json.load(open(DONE))) if os.path.exists(DONE) else set()
    todo = [d for d in pd.bdate_range(start, end) if d.strftime('%Y%m%d') not in done]
    print(f"J-Quants株価: 取得済み {len(done)} 日 / これから取得 {len(todo)} 日")
    buf, n_err = {}, 0
    for i, d in enumerate(todo):
        ds = d.strftime('%Y%m%d')
        try:
            x = cli.get_eq_bars_daily(date_yyyymmdd=ds)
            n_err = 0
        except Exception as e:
            print(f"⚠ 株価 {ds}: {str(e)[:150]}")
            n_err += 1
            if n_err >= 5:
                print("⚠ 連続してエラーになったため株価の取得を中断します")
                break
            time.sleep(wait)
            continue
        if x is not None and len(x):
            x = x[[c for c in COLS if c in x]].rename(columns=COLS)
            buf.setdefault(d.strftime('%Y%m'), []).append(x)
        if d < pd.Timestamp.now(tz='Asia/Tokyo').tz_localize(None).normalize():
            done.add(ds)   # 当日分はまだ確定していない可能性があるので記録しない
        time.sleep(wait)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(todo)}")
    for ym, parts in buf.items():
        path = os.path.join(DIR, f'prices_{ym}.parquet')
        new = pd.concat(parts, ignore_index=True)
        if os.path.exists(path):
            new = pd.concat([pd.read_parquet(path), new], ignore_index=True)
        new['date'] = pd.to_datetime(new['date'])
        new['code'] = new['code'].astype(str)
        for c in ['Open', 'High', 'Low', 'Close', 'Volume', 'Value', 'AdjFactor']:
            new[c] = pd.to_numeric(new[c], errors='coerce').astype('float64')
        new.drop_duplicates(['code', 'date'], keep='last').sort_values(['code', 'date']).to_parquet(
            path, index=False, compression='zstd')
    json.dump(sorted(done), open(DONE, 'w'))


def load(start):
    """start 以降の日足を読み込み、{ticker: 日足DataFrame（分割調整後の Open/High/Low/Close/Volume と Value）} を返す"""
    files = sorted(glob.glob(os.path.join(DIR, 'prices_*.parquet')))
    files = [f for f in files if os.path.basename(f)[7:13] >= pd.Timestamp(start).strftime('%Y%m')]
    if not files:
        return {}, None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.dropna(subset=['Close']).sort_values(['code', 'date'])
    df['AdjFactor'] = df['AdjFactor'].fillna(1.0)
    # その日より後の調整係数の累積積（逆順の累積積を1日ずらす）
    rev = df.iloc[::-1]
    mult = rev.groupby('code')['AdjFactor'].cumprod().iloc[::-1]
    df['mult'] = mult.groupby(df['code']).shift(-1).fillna(1.0)
    for c in ['Open', 'High', 'Low', 'Close']:
        df[c] = df[c] * df['mult']
    df['Volume'] = df['Volume'] / df['mult']
    df['ticker'] = df['code'].map(_code_to_ticker)
    out = {t: g.set_index('date')[['Open', 'High', 'Low', 'Close', 'Volume', 'Value']] for t, g in df.groupby('ticker')}
    return out, df['date'].max()
