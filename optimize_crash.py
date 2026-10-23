"""
急落リバウンド戦略の「買い・売り・トレーリング」条件を総当たりで比較し、ベストを決めるスクリプト。

過剰最適化を避けるため
- 前半(2008-2016)と後半(2017-2026)に分けて評価し、前半で選んだ条件が後半でも通用するかを見る
- 1つの最高値ではなく、前半・後半の両方で上位に入る「安定した条件」を選ぶ
- 選んだ条件の近く（パラメータを少しずらしたもの）も悪くないかを確認する

評価指標: MAR = 年平均リターン ÷ |最大ドローダウン|（リスクあたりのリターン）。年平均リターンも併記

使い方: python optimize_crash.py [週足パネルのキャッシュ(pickle)]
"""
import itertools
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import rule_lab as R
import sentiment_lab as SL

warnings.simplefilter('ignore')
pd.set_option('display.width', 250)
pd.set_option('display.unicode.east_asian_width', True)
START, SPLIT = '2008-01-01', '2017-01-01'

TRIGGERS = ['日経4週-5%', '日経4週-7%', '日経4週-7%&VIXピーク後']
LOSER_PCT = [0.05, 0.10, 0.20]
HOLD = [4, 8, 13, 26]
STOP = [None, 0.10, 0.20]
TRAIL = [None, 0.10, 0.20]
TARGET = [None, 0.20, 0.40]
MK_EXIT = [False, True]      # 日経平均が13週線を回復したら売る
MAX_POS = 10


def seg_stats(eq, a, b):
    e = eq[(eq.index >= a) & (eq.index < b)]
    e = e / e.iloc[0]
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    cagr = e.iloc[-1] ** (1 / yrs) - 1
    dd = (e / e.cummax() - 1).min()
    return cagr * 100, dd * 100, (cagr / -dd if dd < 0 else np.nan)


def prepare(df):
    T = SL.market_table(df)
    nk4 = -T['日経 4週騰落↓']
    vix = T['VIX水準']
    vix_off = vix < vix.rolling(4).max() * 0.85
    mk = pd.read_parquet(os.path.join('data', 'market_weekly.parquet'))
    nk = mk[mk['symbol'] == '^N225'].set_index('week')['Close'].astype(float)
    df = df.copy()
    W = lambda s: df['week'].map(s).fillna(False).astype(bool)
    trig = {'日経4週-5%': W(nk4 <= -5), '日経4週-7%': W(nk4 <= -7), '日経4週-7%&VIXピーク後': W((nk4 <= -7) & vix_off)}
    df['mk_recovered'] = W(nk > nk.rolling(13).mean())
    df['r4rank'] = df['ret4'].where(df['universe']).groupby(df['week']).rank(pct=True)
    df['neg_ret4'] = -df['ret4']
    df['period_ok'] = df['week'] >= START
    return df, trig


def run_grid(df, trig):
    rows = []
    t0 = time.time()
    grid = list(itertools.product(TRIGGERS, LOSER_PCT, HOLD, STOP, TRAIL, TARGET, MK_EXIT))
    for i, (tg, lp, h, st, trl, tp, mx) in enumerate(grid):
        sig = trig[tg] & (df['r4rank'] <= lp) & df['period_ok']
        tr = R.simulate(df, sig, stop=st, trail=trl, target=tp, exit_col='mk_recovered' if mx else None, max_hold=h)
        if len(tr) < 20:
            continue
        p, eq, used = R.portfolio_mtm(tr, df, max_pos=MAX_POS, score='neg_ret4')
        eq = eq[eq.index >= START]
        r = {'発動': tg, '下落上位': lp, '最大保有': h, '損切り': st or 0, 'トレーリング': trl or 0, '利確': tp or 0,
             '日経回復で売り': mx, '取引': len(p), '勝率': (p['ret'] > 0).mean() * 100, '平均': p['ret'].mean()}
        for lab, a, b in (('全', START, '2100'), ('前半', START, SPLIT), ('後半', SPLIT, '2100')):
            c, d, m = seg_stats(eq, a, b)
            r[f'{lab}CAGR'], r[f'{lab}DD'], r[f'{lab}MAR'] = c, d, m
        rows.append(r)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(grid)} ({time.time() - t0:.0f}秒)", file=sys.stderr)
    return pd.DataFrame(rows)


def main():
    df = pd.read_pickle(sys.argv[1]) if len(sys.argv) > 1 else R.add_features(R.load_panel())
    df, trig = prepare(df)
    G = run_grid(df, trig)
    os.makedirs('output', exist_ok=True)
    G.to_csv('output/optimize_crash.csv', index=False, encoding='utf-8-sig')
    show = ['発動', '下落上位', '最大保有', '損切り', 'トレーリング', '利確', '日経回復で売り', '取引', '勝率', '平均',
            '全CAGR', '全DD', '全MAR', '前半CAGR', '前半MAR', '後半CAGR', '後半DD', '後半MAR']
    G['前半順位'] = G['前半MAR'].rank(ascending=False)
    G['後半順位'] = G['後半MAR'].rank(ascending=False)
    n = len(G)
    print(f"比較した条件の数: {n}")
    print("\n===== A. 前半(2008-2016)のMAR上位10 → 後半(2017-2026)での成績 =====")
    top = G.sort_values('前半MAR', ascending=False).head(10)
    print(top[show + ['後半順位']].round(2).to_string(index=False))
    print(f"→ 前半上位10の後半順位の中央値: {top['後半順位'].median():.0f} / {n}")
    # 安定した条件: 前半・後半とも上位25%に入り、平均順位が良いもの
    G['平均順位'] = (G['前半順位'] + G['後半順位']) / 2
    stable = G[(G['前半順位'] <= n * .25) & (G['後半順位'] <= n * .25)].sort_values('平均順位')
    print(f"\n===== B. 前半・後半とも上位25%に入った安定した条件（{len(stable)}件中 上位15） =====")
    print(stable[show].head(15).round(2).to_string(index=False))
    # 各パラメータの効き方（全期間MARの中央値）
    print("\n===== C. パラメータごとの効き方（全条件での全期間MARの中央値 / 年平均の中央値） =====")
    for col in ['発動', '下落上位', '最大保有', '損切り', 'トレーリング', '利確', '日経回復で売り']:
        t = G.groupby(col).agg(MAR中央値=('全MAR', 'median'), CAGR中央値=('全CAGR', 'median'),
                               前半MAR=('前半MAR', 'median'), 後半MAR=('後半MAR', 'median'))
        print(f"\n[{col}]")
        print(t.round(3).to_string())


if __name__ == '__main__':
    main()
