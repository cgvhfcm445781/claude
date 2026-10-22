"""
月足・週足・日足を組み合わせた（マルチタイムフレーム）検証スクリプト。

考え方
- 判定は日足の終値 → 翌営業日の始値でエントリー
- 週足・月足は「その日の終値時点で確定している最新の足」だけを使う（先読みなし）
    例: 水曜日の判定では前週金曜までの週足、前月末までの月足を使う。金曜日（月末日）はその週（月）の足が確定済み
- 日足のエントリー条件 × 上位足（週足・月足）のフィルター の組み合わせを、
  「同じ日のユニバース全体の平均」に対する10日後・20日後の超過リターンで評価する

データ: data_daily/daily_*.parquet（fetch_daily_data.py で取得、data-daily ブランチ）
"""
import glob
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.simplefilter('ignore')
DAILY_DIR = 'data_daily'
MIN_DAILY_VALUE = 50_000_000      # 20日平均の売買代金（日足版バックテストと同じ5,000万円）
PRICE_MIN, PRICE_MAX = 1000, 4000  # 分割前の実際の株価
PERIODS = [('〜2012', None, '2013-01-01'), ('2013-2019', '2013-01-01', '2020-01-01'), ('2020〜', '2020-01-01', None)]


def _stoch(H, L, C, k=9, d=3, sd=3):
    lo, hi = L.rolling(k).min(), H.rolling(k).max()
    fk = 100 * (C - lo) / (hi - lo + 1e-10)
    D = fk.rolling(d).mean()
    return D, D.rolling(sd).mean()


def _macd(C):
    m = C.ewm(span=12, adjust=False).mean() - C.ewm(span=26, adjust=False).mean()
    return m, m.ewm(span=9, adjust=False).mean()


def _rsi(C, n):
    d = C.diff()
    return 100 - 100 / (1 + d.clip(lower=0).rolling(n).mean() / ((-d.clip(upper=0)).rolling(n).mean() + 1e-10))


def _higher_tf(g, rule, prefix, sma_n):
    """週足/月足を作り、各足が確定する日（その期間の最終営業日）に特徴量を置く"""
    b = g.resample(rule, on='date').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
                                          'date': 'last'}).dropna(subset=['Close'])
    C = b['Close']
    f = pd.DataFrame({'avail': b['date']})
    sma = C.rolling(sma_n).mean()
    f[f'{prefix}_above'] = C > sma
    f[f'{prefix}_slope_up'] = sma > sma.shift(4 if prefix == 'w' else 3)
    D, SD = _stoch(b['High'], b['Low'], C)
    f[f'{prefix}_sto'] = D
    f[f'{prefix}_sto_up'] = D > SD
    m, s = _macd(C)
    f[f'{prefix}_macd_up'] = m > s
    f[f'{prefix}_macd_pos'] = m > 0
    f[f'{prefix}_ret1'] = C.pct_change() * 100
    return f


def _per_ticker(g):
    g = g.sort_values('date').reset_index(drop=True)
    O, H, L, C, V = g['Open'], g['High'], g['Low'], g['Close'], g['Volume']
    f = pd.DataFrame({'ticker': g['ticker'], 'date': g['date'], 'Open': O, 'High': H, 'Low': L, 'Close': C})
    D, SD = _stoch(H, L, C)
    f['d_sto'] = D
    f['d_sto_gc'] = (D.shift() <= SD.shift()) & (D > SD)
    f['d_sto_gc20'] = f['d_sto_gc'] & (pd.concat([D.shift(), SD.shift(), D], axis=1).min(axis=1) <= 20)
    f['d_rsi2'] = _rsi(C, 2)
    sma25, sma75 = C.rolling(25).mean(), C.rolling(75).mean()
    f['d_cross25'] = (C.shift() <= sma25.shift()) & (C > sma25)
    f['d_above75'] = C > sma75
    m, s = _macd(C)
    f['d_macd_gc'] = (m.shift() <= s.shift()) & (m > s)
    tr = pd.concat([H - L, (H - C.shift()).abs(), (L - C.shift()).abs()], axis=1).max(axis=1)
    f['atr'] = tr.rolling(14).mean()
    f['value20'] = (C * V).rolling(20).mean()
    f['raw_price'] = C * g['split_factor']
    for h in (10, 20):
        f[f'fwd{h}'] = (C.shift(-h) / O.shift(-1) - 1) * 100
    # 上位足（確定日以降に使える）
    for rule, prefix, n in (('W-FRI', 'w', 26), ('ME', 'm', 12)):
        hf = _higher_tf(g, rule, prefix, n).dropna(subset=['avail']).sort_values('avail')
        f = pd.merge_asof(f.sort_values('date'), hf, left_on='date', right_on='avail', direction='backward').drop(columns='avail')
    return f


def build_panel(daily_dir=DAILY_DIR, cache=None):
    if cache and os.path.exists(cache):
        return pd.read_pickle(cache)
    parts = []
    for fpath in sorted(glob.glob(os.path.join(daily_dir, 'daily_*.parquet'))):
        d = pd.read_parquet(fpath)
        for c in ['Open', 'High', 'Low', 'Close', 'Volume', 'split_factor']:
            d[c] = d[c].astype('float64')
        d['date'] = d['date'].astype('datetime64[ns]')
        x = pd.concat([_per_ticker(g) for _, g in d.groupby('ticker')], ignore_index=True)
        for c in x.columns:
            if x[c].dtype == 'float64':
                x[c] = x[c].astype('float32')
        parts.append(x)
        print(f"  {fpath}: {d['ticker'].nunique()} 銘柄", file=sys.stderr)
    df = pd.concat(parts, ignore_index=True)
    df['universe'] = (df['value20'] >= MIN_DAILY_VALUE) & df['raw_price'].between(PRICE_MIN, PRICE_MAX)
    # 日経平均（地合い）
    mk = pd.read_parquet(os.path.join('data', 'market_weekly.parquet'))
    nk = mk[mk['symbol'] == '^N225'].set_index('week')['Close'].astype(float)
    nkf = pd.DataFrame({'avail': nk.index.astype('datetime64[ns]'), 'mk_ret4w': ((nk / nk.shift(4) - 1) * 100).values})
    df = pd.merge_asof(df.sort_values('date'), nkf, left_on='date', right_on='avail', direction='backward').drop(columns='avail')
    for h in (10, 20):
        df[f'ex{h}'] = df[f'fwd{h}'] - df[f'fwd{h}'].where(df['universe']).groupby(df['date']).transform('mean')
    if cache:
        df.to_pickle(cache)
    return df


def _b(s):
    return s.fillna(False).astype(bool)


ENTRIES = {
    '日足ストキャGC(<=20)': lambda d: _b(d['d_sto_gc20']),
    '日足RSI(2)<10':        lambda d: d['d_rsi2'] < 10,
    '日足25日線上抜け':      lambda d: _b(d['d_cross25']),
    '日足MACD GC':          lambda d: _b(d['d_macd_gc']),
}
FILTERS = {
    'なし':                          lambda d: pd.Series(True, index=d.index),
    '週足↑(26週線上&上向き)':          lambda d: _b(d['w_above']) & _b(d['w_slope_up']),
    '週足↓':                          lambda d: ~_b(d['w_above']) & ~_b(d['w_slope_up']),
    '月足↑(12か月線上)':               lambda d: _b(d['m_above']),
    '月足↓':                          lambda d: ~_b(d['m_above']),
    '月足↑&週足↑':                    lambda d: _b(d['m_above']) & _b(d['w_above']) & _b(d['w_slope_up']),
    '月足↑&週足押し目(週ストキャ<30)':   lambda d: _b(d['m_above']) & (d['w_sto'] < 30),
    '月足MACD↑&週足MACD↑':             lambda d: _b(d['m_macd_up']) & _b(d['w_macd_up']),
    '月足MACD↑&週足MACD↓(押し目)':      lambda d: _b(d['m_macd_up']) & ~_b(d['w_macd_up']),
    '月足↓&週足↓':                    lambda d: ~_b(d['m_above']) & ~_b(d['w_above']) & ~_b(d['w_slope_up']),
}


def evaluate(df):
    u = df[df['universe']].dropna(subset=['ex20'])
    rows = []
    for en, ef in ENTRIES.items():
        e = ef(u)
        for fn, ff in FILTERS.items():
            x = u[e & ff(u)]
            if len(x) < 200:
                continue
            r = {'日足エントリー': en, '上位足フィルター': fn, '件数': len(x)}
            for h in (10, 20):
                w = x.groupby('date')[f'ex{h}'].mean()
                r[f't({h}日)'] = w.mean() / w.std() * np.sqrt(len(w) / h)
            for p, a, b in PERIODS:
                r[p] = x.loc[(x['date'] >= (a or '1900')) & (x['date'] < (b or '2100')), 'ex20'].mean()
            r['20日 素のリターン'] = x['fwd20'].mean()
            r['20日 勝率'] = (x['fwd20'] > 0).mean() * 100
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    df = build_panel(cache=sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"日足パネル: {len(df)} 行, {df['ticker'].nunique()} 銘柄, {df['date'].min():%Y-%m-%d}〜{df['date'].max():%Y-%m-%d}")
    res = evaluate(df)
    pd.set_option('display.width', 230)
    pd.set_option('display.unicode.east_asian_width', True)
    print("\n===== 日足エントリー × 上位足フィルター: 同じ日のユニバース平均に対する20日超過リターン(%) =====")
    print(res.set_index(['日足エントリー', '上位足フィルター']).round(2).to_string())
    os.makedirs('output', exist_ok=True)
    res.to_csv('output/mtf_lab.csv', index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
