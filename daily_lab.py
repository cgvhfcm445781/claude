"""
日足の売買ルールを、往復コスト込みのポートフォリオとして検証するスクリプト。

試す型
1. 短期リバーサル: 直近5日で大きく下げた銘柄を買い、数日〜2週間で売る
2. 押し目買い（Connors型）: 上昇トレンド中に RSI(2) が10未満で買い、RSI(2) が70を超えたら売る
3. 日足の急落検知: 日経平均が1日・5日で急落した日に、20日騰落の下位銘柄を買う
4. 日足ブレイクアウト（順張り）: 60日高値の更新で買う
それぞれ「信用残が重くない」（週足で一番効いた絞り込み）も試す

前提: 終値で判定 → 翌営業日の始値で買い。売りは条件を満たした日の翌営業日の始値、または期限日の終値
      往復コスト0.2%。最大10銘柄、空き枠は現金。前半2008-2016 / 後半2017-2026（信用残は2017年〜）

使い方: python daily_lab.py [日足パネル(mtf_lab.build_panel のpickle)] [出力pickle]
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import mtf_lab as M
import margin_lab as ML
import rule_lab as R

warnings.simplefilter('ignore')
pd.set_option('display.width', 240)
pd.set_option('display.unicode.east_asian_width', True)
SPLIT = '2017-01-01'


def build(d):
    d = d.sort_values(['ticker', 'date']).reset_index(drop=True)
    for c in ['Open', 'High', 'Low', 'Close', 'atr']:
        d[c] = d[c].astype('float64')
    g = d.groupby('ticker')
    C = d['Close']
    d['ret1'] = g['Close'].pct_change() * 100
    d['ret5'] = (C / g['Close'].shift(5) - 1) * 100
    d['ret20'] = (C / g['Close'].shift(20) - 1) * 100
    d['mom12_1'] = g['Close'].shift(20) / g['Close'].shift(250) - 1
    d['sma200'] = g['Close'].transform(lambda s: s.rolling(200).mean())
    d['hi60_prev'] = g['High'].transform(lambda s: s.shift().rolling(60).max())
    d['bad'] = g['ret1'].transform(lambda s: s.abs().rolling(260, min_periods=1).max()) > 50
    d['universe'] = d['universe'] & ~d['bad']
    # 信用残（J-Quants、週次・1週遅れで使える値）→ 日足に
    m = ML.load_margin().rename(columns={'week': 'mweek'}).sort_values('mweek')
    d = pd.merge_asof(d.sort_values('date'), m, left_on='date', right_on='mweek', by='ticker',
                      direction='backward', tolerance=pd.Timedelta(days=10))
    vol_w = d['value20'] / d['Close'] * 5
    lw, sw = d['LongVol'] / vol_w, d['ShrtVol'] / vol_w
    med = lambda s: s.where(d['universe']).groupby(d['date']).transform('median')
    d['heavy_margin'] = ((lw > med(lw) * 2) | (sw > med(sw) * 2)).fillna(False)
    d['has_margin'] = d['LongVol'].notna()
    # 日経平均（日次）
    mk = pd.read_parquet(os.path.join('data', 'macro_daily.parquet'))
    nk = mk[mk['symbol'] == '^N225'].set_index('date')['Close'].astype(float)
    d['nk1'] = d['date'].map(nk.pct_change() * 100)
    d['nk5'] = d['date'].map((nk / nk.shift(5) - 1) * 100)
    d = d.sort_values(['ticker', 'date']).reset_index(drop=True)
    d['rsi2_up70'] = d['d_rsi2'] > 70
    d['above5'] = d['Close'] > d.groupby('ticker')['Close'].transform(lambda s: s.rolling(5).mean())
    d['r5rank'] = d['ret5'].where(d['universe']).groupby(d['date']).rank(pct=True)
    d['r20rank'] = d['ret20'].where(d['universe']).groupby(d['date']).rank(pct=True)
    d['neg_ret5'], d['neg_ret20'], d['neg_rsi2'] = -d['ret5'], -d['ret20'], -d['d_rsi2']
    d['mom_score'] = d['mom12_1']
    return d.rename(columns={'date': 'week'})   # rule_lab は 'week' 列を日付として使う


def seg(eq, a, b):
    e = eq[(eq.index >= a) & (eq.index < b)]
    if len(e) < 50:
        return np.nan, np.nan
    e = e / e.iloc[0]
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    return (e.iloc[-1] ** (1 / yrs) - 1) * 100, (e / e.cummax() - 1).min() * 100


def run(d, name, sig, score, max_pos=10, start='2008-01-01', **kw):
    sig = sig & (d['week'] >= start)
    tr = R.simulate(d, sig, **kw)
    if len(tr) < 30:
        return None
    p, eq, used = R.portfolio_mtm(tr, d, max_pos=max_pos, score=score)
    eq = eq[eq.index >= start]
    c, dd = seg(eq, start, '2100')
    c1, _ = seg(eq, start, SPLIT)
    c2, dd2 = seg(eq, SPLIT, '2100')
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return {'戦略': name, '年率': c, '最大DD': dd, '前半': c1, '後半': c2, '後半DD': dd2, '取引/年': len(p) / yrs,
            '勝率': (p['ret'] > 0).mean() * 100, '1回平均': p['ret'].mean(), '平均保有日': p['weeks'].mean(),
            '投資比率': used.clip(upper=1).mean() * 100}, eq


def main():
    t0 = time.time()
    out = sys.argv[2] if len(sys.argv) > 2 else None
    if out and os.path.exists(out):
        d = pd.read_pickle(out)
    else:
        d = build(M.build_panel(cache=sys.argv[1] if len(sys.argv) > 1 else None))
        if out:
            d.to_pickle(out)
    print(f"準備 {time.time() - t0:.0f}秒  {len(d)}行", file=sys.stderr)
    U = d['universe']
    light = ~d['heavy_margin']
    up = (d['Close'] > d['sma200'])
    cases = []
    # 1. 短期リバーサル
    for q in (0.02, 0.05):
        for h in (5, 10):
            base = U & (d['r5rank'] <= q)
            cases.append((f'短期リバーサル 5日騰落の下位{q:.0%} / {h}日保有', base, 'neg_ret5', dict(max_hold=h)))
            cases.append((f'短期リバーサル 下位{q:.0%} & 200日線上 / {h}日保有', base & up, 'neg_ret5', dict(max_hold=h)))
            cases.append((f'短期リバーサル 下位{q:.0%} & 信用残軽い / {h}日保有', base & light & d['has_margin'], 'neg_ret5',
                          dict(max_hold=h), '2017-01-01'))
        cases.append((f'短期リバーサル 下位{q:.0%} / 5日線を上回ったら売り(最長10日)', U & (d['r5rank'] <= q), 'neg_ret5',
                      dict(max_hold=10, exit_col='above5')))
    # 2. 押し目買い（Connors型）
    for flt, fn in ((up, '200日線上'), (up & light & d['has_margin'], '200日線上&信用残軽い'),
                    (up & (d['mom12_1'] > 0) & light & d['has_margin'], '200日線上&12か月上昇&信用残軽い')):
        st = '2017-01-01' if '信用' in fn else '2008-01-01'
        cases.append((f'押し目 RSI(2)<10 & {fn} / RSI(2)>70で売り(最長10日)', U & flt & (d['d_rsi2'] < 10), 'neg_rsi2',
                      dict(max_hold=10, exit_col='rsi2_up70'), st))
        cases.append((f'押し目 RSI(2)<5 & {fn} / RSI(2)>70で売り(最長10日)', U & flt & (d['d_rsi2'] < 5), 'neg_rsi2',
                      dict(max_hold=10, exit_col='rsi2_up70'), st))
    # 3. 日足の急落検知
    for cond, cn in ((d['nk1'] <= -4, '日経1日-4%以下'), (d['nk5'] <= -7, '日経5日-7%以下'), (d['nk5'] <= -10, '日経5日-10%以下')):
        for h, tp in ((20, None), (40, 0.20)):
            cases.append((f'急落 {cn} & 20日騰落の下位20% / {h}日{" 利確+20%" if tp else ""}', U & cond & (d['r20rank'] <= .2),
                          'neg_ret20', dict(max_hold=h, target=tp)))
    # 4. ブレイクアウト（順張り）
    brk = U & (d['Close'] > d['hi60_prev'])
    cases.append(('ブレイクアウト 60日高値更新 & 信用残軽い / 20日保有', brk & light & d['has_margin'], 'mom_score', dict(max_hold=20), '2017-01-01'))
    cases.append(('ブレイクアウト 60日高値更新 & 12か月上昇 & 信用残軽い / 60日保有', brk & (d['mom12_1'] > 0) & light & d['has_margin'],
                  'mom_score', dict(max_hold=60), '2017-01-01'))
    # 比較: ランダム
    rnd = U & (pd.Series(np.random.default_rng(0).random(len(d)), index=d.index) < 0.01)
    cases.append(('(比較) ランダム / 20日保有', rnd, None, dict(max_hold=20)))
    rows = []
    for cs in cases:
        name, sig, score, kw = cs[:4]
        start = cs[4] if len(cs) > 4 else '2008-01-01'
        r = run(d, name, sig, score, start=start, **kw)
        if r:
            rows.append(r[0])
            print(f"  {name} ({time.time() - t0:.0f}秒)", file=sys.stderr)
    res = pd.DataFrame(rows).set_index('戦略')
    res.to_csv('output/daily_lab.csv', encoding='utf-8-sig')
    print("最大10銘柄・空き枠は現金・往復コスト0.2%込み。信用残を使う条件は2017年〜（前半は空欄）")
    print(res.round(1).to_string())


if __name__ == '__main__':
    main()
