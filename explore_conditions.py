"""
weekly_signals.csv から「ストキャス以外も含めた良い条件」を検証するスクリプト。

やること
1. 候補ルールを、ストキャスのパラメータ × 評価リターン(決済B / 4週後) ごとに年別で評価
   （何年中何年、全シグナルを上回ったか＝安定性）
2. 機械学習(勾配ブースティング)で 〜2023年 を学習 → 2024年〜 で「週内の順位付け」ができるか確認
   （ルールで拾いきれていない情報が残っているかのチェック）

注意
- 日経平均系の特徴量(nk_*)は使わない。暴落週などの少数の局面に依存して、期間が変わると崩れたため
- entry_price は yfinance の分割調整後の価格。後で株式分割した銘柄ほど過去の価格が低く出るため、
  「株価が安いほど良い」という結果には先読みバイアスが混ざっている可能性がある
"""
import argparse
import os
import numpy as np
import pandas as pd

RULES = {
    '全シグナル':                         lambda d: d['entry_price'] > 0,
    '株価<=2400円':                       lambda d: d['entry_price'] <= 2400,
    'シグナル週の上昇<=6%':               lambda d: d['prev_ret_1w'] <= 6,
    '株価<=2400 & 週上昇<=6%':            lambda d: (d['entry_price'] <= 2400) & (d['prev_ret_1w'] <= 6),
    '株価<=2400 & 週上昇<=6% & MACDヒスト-1.8〜0.1':
        lambda d: (d['entry_price'] <= 2400) & (d['prev_ret_1w'] <= 6) & d['macd_hist_pct'].between(-1.8, 0.1),
}
PARAMS = ['5-3-3', '9-3-3', '14-3-3']
RETS = ['exit_B_ret', 'fwd_ret_4w']
TEST_START = '2024-01-01'


def eval_rules(df):
    for ret in RETS:
        for p in PARAMS:
            g0 = df[(df['stoch_params'] == p) & (df['year'] >= 2017)].dropna(subset=[ret])
            base_y = g0.groupby('year')[ret].mean()
            rows = []
            for name, f in RULES.items():
                g = g0[f(g0)]
                r = g[ret]
                y = g.groupby('year')[ret].mean()
                rows.append({'rule': name, 'n': len(g), '平均%': r.mean(), '勝率%': (r > 0).mean() * 100,
                             'PF': r[r > 0].sum() / -r[r < 0].sum(),
                             'テスト期間平均%': g.loc[g['signal_week'] >= TEST_START, ret].mean(),
                             '全体を上回った年': f"{(y > base_y + 1e-9).sum()}/{len(y)}"})
            print(f"\n## {p}  {ret}")
            print(pd.DataFrame(rows).set_index('rule').round(2).to_string())


def ml_check(df):
    from sklearn.ensemble import HistGradientBoostingRegressor
    cols = df.columns
    feats = [c for c in cols[cols.get_loc('entry_price'):cols.get_loc('nk_stoch_d') + 1]
             if df[c].dtype != object and not c.startswith('nk_')]
    print("\n===== 機械学習チェック（〜2023年で学習 → 2024年〜で評価、週内順位で5分割） =====")
    for p in PARAMS:
        for ret in RETS:
            g = df[df['stoch_params'] == p].dropna(subset=[ret]).copy()
            g['ex'] = g[ret] - g.groupby('signal_week')[ret].transform('mean')  # 同じ週の平均との差
            tr, te = g[g['signal_week'] < TEST_START], g[g['signal_week'] >= TEST_START].copy()
            m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.03, max_iter=300,
                                              min_samples_leaf=200, l2_regularization=1.0, random_state=0)
            m.fit(tr[feats], tr['ex'].clip(-30, 30))
            te['pred'] = m.predict(te[feats])
            te['bin'] = pd.cut(te.groupby('signal_week')['pred'].rank(pct=True), [0, .2, .4, .6, .8, 1],
                               labels=['下位20%', '20-40', '40-60', '60-80', '上位20%'])
            ic = te.groupby('signal_week').apply(
                lambda x: x['pred'].corr(x[ret], method='spearman') if len(x) > 5 else np.nan).dropna()
            t = te.groupby('bin', observed=True)[ret].mean().round(2)
            print(f"{p:7s} {ret:11s} IC={ic.mean():+.3f} (t={ic.mean() / ic.std() * np.sqrt(len(ic)):.1f})  "
                  + '  '.join(f"{k}:{v:+.2f}" for k, v in t.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=os.path.join('output', 'weekly_signals.csv'))
    ap.add_argument('--no-ml', action='store_true')
    args = ap.parse_args()
    df = pd.read_csv(args.csv, encoding='utf-8-sig', parse_dates=['signal_week'])
    df['year'] = df['signal_week'].dt.year
    pd.set_option('display.width', 200)
    pd.set_option('display.unicode.east_asian_width', True)
    eval_rules(df)
    if not args.no_ml:
        ml_check(df)


if __name__ == '__main__':
    main()
