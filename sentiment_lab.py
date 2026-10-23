"""
「投資家が悲観的で、下値リスクより上値余地が大きい局面（リスクリワードが高い局面）」を探す検証スクリプト。

1. 市場全体の指標を週次で作る
   - VIX（恐怖指数）、金利、ドル円、信用スプレッド（HYG/LQD）… data/macro_daily.parquet（fetch_macro.py）
   - チャート: 日経平均の4週騰落、52週高値からの下落、26週線乖離、RSI
   - 需給（騰落の広がり）: 52週安値の銘柄割合、26週線より下の銘柄割合、4週で-15%以上下げた銘柄割合、出来高急増
2. 各指標を「その時点までの過去データの中での百分位」に変換（先読みなし）。悲観が強いほど1に近い
3. 百分位の帯ごとに、その後13週の
     上昇余地（期間中の最大上昇）、下値リスク（期間中の最大下落）、リスクリワード（上昇余地 ÷ 下値リスク）
   を、日経平均と全銘柄等金額指数で測る
4. 複数指標の平均＝悲観スコア を作り、急落リバウンド戦略の発動条件として使った場合と比べる

使い方: python sentiment_lab.py [週足パネルのキャッシュ(pickle)]
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import rule_lab as R

warnings.simplefilter('ignore')
pd.set_option('display.width', 230)
pd.set_option('display.unicode.east_asian_width', True)
H = 13            # 先行きを測る週数
START = '2008-01-01'


def weekly_macro():
    m = pd.read_parquet(os.path.join('data', 'macro_daily.parquet'))
    w = m.pivot_table(index='date', columns='symbol', values='Close').sort_index()
    return w.resample('W-FRI').last()


def market_table(df):
    """週ごとの市場全体の状態と、先行きの結果"""
    mk = pd.read_parquet(os.path.join('data', 'market_weekly.parquet'))
    nk = mk[mk['symbol'] == '^N225'].set_index('week')[['High', 'Low', 'Close']].astype(float)
    T = pd.DataFrame(index=nk.index)
    c = nk['Close']
    T['日経 4週騰落↓'] = -(c / c.shift(4) - 1) * 100
    T['日経 52週高値からの下落'] = -(c / nk['High'].rolling(52).max() - 1) * 100
    T['日経 26週線乖離↓'] = -(c / c.rolling(26).mean() - 1) * 100
    d = c.diff()
    T['日経 RSI14↓'] = -(100 - 100 / (1 + d.clip(lower=0).rolling(14).mean() / ((-d.clip(upper=0)).rolling(14).mean() + 1e-10)))
    # 騰落の広がり（売買代金の下限だけで選んだ幅広いユニバース）
    u = df[df['value4'] >= R.MIN_WEEKLY_VALUE]
    g = u.groupby('week')
    T['52週安値の銘柄割合'] = g['dist_lo52'].apply(lambda s: (s <= 1).mean() * 100)
    T['26週線より下の銘柄割合'] = g['dev26'].apply(lambda s: (s < 0).mean() * 100)
    T['4週-15%以上の銘柄割合'] = g['ret4'].apply(lambda s: (s <= -15).mean() * 100)
    T['出来高急増(平均倍率)'] = g['vol_ratio'].median()
    # マクロ指標
    M = weekly_macro().reindex(T.index, method='ffill')
    if '^VIX' in M:
        T['VIX水準'] = M['^VIX']
        T['VIX 4週上昇率'] = (M['^VIX'] / M['^VIX'].shift(4) - 1) * 100
    if '^JNIV' in M and M['^JNIV'].notna().sum() > 200:
        T['日経VI水準'] = M['^JNIV']
    if '^TNX' in M:
        T['米10年金利 4週低下幅'] = -(M['^TNX'] - M['^TNX'].shift(4))
        T['米10年金利 4週上昇幅'] = M['^TNX'] - M['^TNX'].shift(4)
    if 'JPY=X' in M:
        T['円高(ドル円4週騰落↓)'] = -(M['JPY=X'] / M['JPY=X'].shift(4) - 1) * 100
    if 'HYG' in M and 'LQD' in M:
        r = M['HYG'] / M['LQD']
        T['信用不安(HYG/LQD 4週↓)'] = -(r / r.shift(4) - 1) * 100
    if '^GSPC' in M:
        T['S&P500 4週騰落↓'] = -(M['^GSPC'] / M['^GSPC'].shift(4) - 1) * 100
    # 先行き: 日経平均
    hi_f = pd.concat([nk['High'].shift(-k) for k in range(1, H + 1)], axis=1).max(axis=1)
    lo_f = pd.concat([nk['Low'].shift(-k) for k in range(1, H + 1)], axis=1).min(axis=1)
    T['nk_fwd'] = (c.shift(-H) / c - 1) * 100
    T['nk_up'] = (hi_f / c - 1) * 100
    T['nk_dn'] = (lo_f / c - 1) * 100
    # 先行き: 全銘柄等金額指数（前週時点のユニバースの週次リターン平均）
    df = df.assign(wret=df.groupby('ticker')['Close'].pct_change().clip(-0.5, 0.5),   # データの誤りによる極端な値を抑える
                   prev_in=df.groupby('ticker')['value4'].shift() >= R.MIN_WEEKLY_VALUE)
    ew = (1 + df[df['prev_in']].groupby('week')['wret'].mean().reindex(T.index).fillna(0)).cumprod()
    hi_e = pd.concat([ew.shift(-k) for k in range(1, H + 1)], axis=1).max(axis=1)
    lo_e = pd.concat([ew.shift(-k) for k in range(1, H + 1)], axis=1).min(axis=1)
    T['ew_fwd'] = (ew.shift(-H) / ew - 1) * 100
    T['ew_up'] = (hi_e / ew - 1) * 100
    T['ew_dn'] = (lo_e / ew - 1) * 100
    return T


INDICATORS = None   # market_table で作った列のうち先行き以外


def to_pct(T):
    """各指標を、その時点までの過去データの中での百分位に（先読みなし）。52週分たまるまでは NaN"""
    cols = [c for c in T.columns if not c.startswith(('nk_', 'ew_'))]
    P = pd.DataFrame(index=T.index)
    for c in cols:
        P[c] = T[c].expanding(min_periods=52).rank(pct=True)
    return P


def episodes(mask):
    w = mask[mask].index
    return int(((w.to_series().diff() > pd.Timedelta(weeks=3)).sum() + 1)) if len(w) else 0


def bucket_table(T, P, cols):
    rows = []
    bins = [(0.95, 1.01, '上位5%'), (0.90, 0.95, '90-95%'), (0.75, 0.90, '75-90%'), (0.25, 0.75, '中間'), (0.0, 0.25, '下位25%')]
    for c in cols:
        for lo, hi, lab in bins:
            m = (P[c] >= lo) & (P[c] < hi) & (P.index >= START) & T['nk_fwd'].notna()
            x = T[m]
            if len(x) < 5:
                continue
            rows.append({'指標': c, '悲観度': lab, '週数': len(x), '局面数': episodes(m),
                         '日経13週後': x['nk_fwd'].mean(), '上昇余地': x['nk_up'].mean(), '下値リスク': x['nk_dn'].mean(),
                         'RR(日経)': x['nk_up'].mean() / -x['nk_dn'].mean(), '13週後プラス率': (x['nk_fwd'] > 0).mean() * 100,
                         '全銘柄13週後': x['ew_fwd'].mean(), 'RR(全銘柄)': x['ew_up'].mean() / -x['ew_dn'].mean()})
    return pd.DataFrame(rows)


def score_strategy(df, T, P, score_cols):
    """悲観スコアを急落リバウンド戦略の発動条件に使った場合の比較"""
    S = P[score_cols].mean(axis=1, skipna=True).expanding(min_periods=52).rank(pct=True)
    df = df.copy()
    df['score'] = df['week'].map(S)
    df['r4rank'] = df['ret4'].where(df['universe']).groupby(df['week']).rank(pct=True)
    df['neg_ret4'] = -df['ret4']
    nk4 = df['week'].map(-T['日経 4週騰落↓'])
    loser = df['r4rank'] <= .1
    vix = T['VIX水準']
    vix_off = df['week'].map(vix < vix.rolling(4).max() * 0.85).fillna(False).astype(bool)   # VIXが直近4週の高値から15%以上低下
    cases = {
        '現行: 日経4週-5%以下': nk4 <= -5,
        '日経4週-7%以下 & VIXがピークから15%以上低下': (nk4 <= -7) & vix_off,
        '現行(厳しめ): 日経4週-7%以下': nk4 <= -7,
        '悲観スコア 上位10%': df['score'] >= .90,
        '悲観スコア 上位5%': df['score'] >= .95,
        '悲観スコア 上位10% & 日経4週-5%以下': (df['score'] >= .90) & (nk4 <= -5),
        '悲観スコア 上位20% & 日経4週-5%以下': (df['score'] >= .80) & (nk4 <= -5),
    }
    rows = []
    for name, cond in cases.items():
        tr = R.simulate(df, cond & loser & (df['week'] >= START), max_hold=8)
        p, eq, used = R.portfolio_mtm(tr, df, max_pos=10, score='neg_ret4')
        eq = eq[eq.index >= START]
        eq = eq / eq.iloc[0]
        yr = eq.resample('YE').last().pct_change().dropna()
        rows.append({'発動条件（下落上位10%を8週保有、最大10銘柄、空き枠現金）': name, '成績': R.perf(eq),
                     '取引': len(p), '平均': round(p['ret'].mean(), 2), '勝率': round((p['ret'] > 0).mean() * 100, 1),
                     '最悪の年': f"{yr.min() * 100:+.0f}%", '投資比率': f"{used.clip(upper=1).mean() * 100:.0f}%"})
    print("\n===== 4. 悲観スコアを急落リバウンド戦略の発動条件に使った場合（2008年〜） =====")
    print(pd.DataFrame(rows).set_index(rows and list(rows[0])[0]).to_string())
    return S


def main():
    df = pd.read_pickle(sys.argv[1]) if len(sys.argv) > 1 else R.add_features(R.load_panel())
    T = market_table(df)
    P = to_pct(T)
    cols = list(P.columns)
    res = bucket_table(T, P, cols)
    print(f"期間 {START}〜（各指標の百分位は、その時点までの過去データだけで計算）  先行き={H}週")
    print("RR = 上昇余地（期間中の最大上昇の平均） ÷ 下値リスク（期間中の最大下落の平均）")
    for c in cols:
        x = res[res['指標'] == c].drop(columns='指標').set_index('悲観度')
        print(f"\n--- {c}")
        print(x.round(2).to_string())
    os.makedirs('output', exist_ok=True)
    res.to_csv('output/sentiment_lab.csv', index=False, encoding='utf-8-sig')
    # 悲観スコア: 恐怖・チャート・需給・信用の代表的な指標の平均
    core = [c for c in ['VIX水準', '日経 52週高値からの下落', '日経 4週騰落↓', '52週安値の銘柄割合',
                        '26週線より下の銘柄割合', '信用不安(HYG/LQD 4週↓)', '円高(ドル円4週騰落↓)'] if c in P]
    # 平均すると値が中央に寄るので、平均値をもう一度「過去の中での百分位」に直す
    S = P[core].mean(axis=1).expanding(min_periods=52).rank(pct=True)
    T['悲観スコア'] = S
    P2 = P.assign(悲観スコア=S)
    print(f"\n===== 悲観スコア（{', '.join(core)} の百分位の平均を、さらに百分位にしたもの） =====")
    print(bucket_table(T, P2, ['悲観スコア']).drop(columns='指標').set_index('悲観度').round(2).to_string())
    score_strategy(df, T, P, core)
    last = S.dropna().index[-1]
    print(f"\n最新週 {last:%Y-%m-%d} の悲観スコア: {S[last]:.2f}（1に近いほど悲観的）")
    print(P.loc[last, core].round(2).to_string())


if __name__ == '__main__':
    main()
