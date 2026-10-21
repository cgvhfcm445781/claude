"""
Weekly_backtest.py が出力した weekly_signals.csv を分析するスクリプト。

検証したい仮説
- 「週足が上昇トレンド」を前提条件にすると、ストキャスの売られ過ぎ水準(GC時の水準)の
  最適値が変わり、勝率が上がるのではないか？

やること
- 上昇トレンドの定義(TREND_FILTERS) × 売られ過ぎ閾値(THRESHOLDS) × ストキャスのパラメータ
  の全組み合わせで、勝率・平均リターン・PF・件数を集計する
- 前半(学習期間)/後半(検証期間)に分けて、前半で良かった条件が後半でも通用するかを見る
- --one-position を付けると、1銘柄1ポジション制（保有中の新規シグナルは無視）で間引く

使い方
  python analyze_weekly_signals.py                       # 決済B、閾値=stoch_min3
  python analyze_weekly_signals.py --exit C --level-col gc_level
  python analyze_weekly_signals.py --one-position --min-n 50
"""
import argparse
import os
import numpy as np
import pandas as pd

# 上昇トレンドの定義（すべてシグナル週の終値時点で確定している値）
TREND_FILTERS = {
    'なし(全シグナル)':        lambda d: pd.Series(True, index=d.index),
    '13週線↑':                lambda d: d['sma13_slope'] > 0,
    '26週線↑':                lambda d: d['sma26_slope'] > 0,
    '13週線↑&26週線↑':        lambda d: (d['sma13_slope'] > 0) & (d['sma26_slope'] > 0),
    '終値>26週線':            lambda d: d['sma26_dev'] > 0,
    '終値>52週線':            lambda d: d['sma52_dev'] > 0,
    '26週線↑&終値>52週線':    lambda d: (d['sma26_slope'] > 0) & (d['sma52_dev'] > 0),
    '26週線↑&日経>26週線':    lambda d: (d['sma26_slope'] > 0) & (d['nk_dev26'] > 0),
}
THRESHOLDS = [10, 15, 20, 25, 30, 40, 50, 60]  # 「水準 <= 閾値」を売られ過ぎとみなす


def one_position(df, exit_key):
    """銘柄×パラメータごとに、保有中に出たシグナルを捨てる（決済週の終値以降のシグナルから再エントリー可）"""
    wk_col = f'exit_{exit_key}_weeks'
    keep = []
    for _, g in df.sort_values('signal_week').groupby(['ticker', 'stoch_params'], sort=False):
        free_from = pd.Timestamp.min
        for i, sw, held in zip(g.index, g['signal_week'], g[wk_col]):
            if sw >= free_from:
                keep.append(i)
                free_from = sw + pd.Timedelta(weeks=int(held))
    return df.loc[keep]


def stats(ret, weeks):
    n = len(ret)
    if n == 0:
        return dict(n=0, n_weeks=0, win=np.nan, mean=np.nan, median=np.nan, pf=np.nan)
    gain, loss = ret[ret > 0].sum(), -ret[ret < 0].sum()
    return dict(n=n, n_weeks=weeks.nunique(), win=(ret > 0).mean() * 100,
                mean=ret.mean(), median=ret.median(), pf=gain / loss if loss > 0 else np.inf)


def build_grid(df, ret_col, level_col, exit_key, use_one_pos):
    rows = []
    for params, gp in df.groupby('stoch_params'):
        for fname, f in TREND_FILTERS.items():
            gf = gp[f(gp).fillna(False)]
            for th in THRESHOLDS:
                g = gf[gf[level_col] <= th]
                if use_one_pos:
                    g = one_position(g, exit_key)
                for period, gg in (('全期間', g), ('前半', g[g['is_train']]), ('後半', g[~g['is_train']])):
                    rows.append(dict(params=params, trend=fname, th=th, period=period,
                                     **stats(gg[ret_col], gg['signal_week'])))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=os.path.join('output', 'weekly_signals.csv'))
    ap.add_argument('--exit', default='B', help='決済ルール A/B/C、または fwd4 (4週後の固定リターン)')
    ap.add_argument('--level-col', default='stoch_min3', help='売られ過ぎ判定に使う列 (stoch_min3 / gc_level / stoch_min4w)')
    ap.add_argument('--min-n', type=int, default=30, help='この件数未満の条件は表示しない')
    ap.add_argument('--split', type=float, default=0.7, help='前半(学習)に使う期間の割合')
    ap.add_argument('--one-position', action='store_true')
    args = ap.parse_args()

    df = pd.read_csv(args.csv, encoding='utf-8-sig', parse_dates=['signal_week'])
    if args.exit == 'fwd4':
        ret_col, exit_key = 'fwd_ret_4w', 'B'
    else:
        ret_col, exit_key = f'exit_{args.exit}_ret', args.exit
    df = df.dropna(subset=[ret_col, args.level_col])
    if args.one_position:
        df = df.dropna(subset=[f'exit_{exit_key}_weeks'])

    split_date = df['signal_week'].quantile(args.split)
    df['is_train'] = df['signal_week'] < split_date
    print(f"データ: {len(df)}件  期間 {df['signal_week'].min():%Y-%m-%d} 〜 {df['signal_week'].max():%Y-%m-%d}")
    print(f"前半/後半の境目: {split_date:%Y-%m-%d}   評価: {ret_col}   水準列: {args.level_col}"
          f"   1銘柄1ポジション: {'あり' if args.one_position else 'なし'}\n")

    grid = build_grid(df, ret_col, args.level_col, exit_key, args.one_position)
    out_path = os.path.join(os.path.dirname(args.csv) or '.', f'trend_threshold_grid_{args.exit}.csv')
    grid.to_csv(out_path, index=False, encoding='utf-8-sig')

    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.unicode.east_asian_width', True)
    full = grid[grid['period'] == '全期間']

    for params in sorted(full['params'].unique()):
        g = full[full['params'] == params]
        print(f"===== ストキャス {params} : 勝率(%)  [トレンド条件 × 水準<=閾値]  ※件数{args.min_n}未満は空欄 =====")
        win = g.pivot(index='trend', columns='th', values='win').where(
            g.pivot(index='trend', columns='th', values='n') >= args.min_n)
        print(win.reindex(list(TREND_FILTERS)).round(1).to_string())
        print(f"\n----- 同 件数 -----")
        print(g.pivot(index='trend', columns='th', values='n').reindex(list(TREND_FILTERS)).to_string())
        print(f"\n----- 同 平均リターン(%) -----")
        print(g.pivot(index='trend', columns='th', values='mean').reindex(list(TREND_FILTERS)).round(2).to_string())
        print()

    # 前半で上位だった条件が後半でも良いか（過剰最適化チェック）
    wide = grid.pivot_table(index=['params', 'trend', 'th'], columns='period',
                            values=['n', 'win', 'mean', 'pf'])
    wide.columns = [f'{a}_{b}' for a, b in wide.columns]
    wide = wide[(wide['n_前半'] >= args.min_n) & (wide['n_後半'] >= args.min_n)]
    top = wide.sort_values('win_前半', ascending=False).head(15)
    cols = ['n_前半', 'win_前半', 'mean_前半', 'n_後半', 'win_後半', 'mean_後半', 'pf_後半']
    print("===== 前半の勝率 上位15条件 → 後半(検証期間)での成績 =====")
    print(top[cols].round(2).to_string())

    base = wide.xs('なし(全シグナル)', level='trend')
    print("\n===== 参考: トレンド条件なしの後半成績（これを上回るかが本当の改善） =====")
    print(base[['n_後半', 'win_後半', 'mean_後半', 'pf_後半']].round(2).to_string())
    print(f"\n詳細グリッド: {os.path.abspath(out_path)}")


if __name__ == '__main__':
    main()
