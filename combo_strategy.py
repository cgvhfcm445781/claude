"""
「普段は割安株、急落時は下落上位の銘柄に乗り換える」組み合わせ戦略のバックテスト（2017年〜）。

割安株（普段）   : 低PER(E/P上位) & 12か月線より上 & 信用残が重くない 銘柄を上位 VALUE_N 銘柄、
                   13週ごとに入れ替え（金曜終値で選び翌営業日の始値で買う、13週後に売る）
急落リバウンド   : 日経平均4週騰落 -7%以下の週に、4週騰落の下位20%を下げた順に最大 CRASH_N 銘柄、
                   +20%で利確／最長8週（crash_rebound.py の 'return' モード）
組み合わせ方
- 固定配分   : 資金を割安株と急落用に分け、急落用は出番が来るまで現金
- 乗り換え   : 急落シグナルが出たら、その分の割安株を売って急落銘柄を買い、終わったら割安株に戻す
               （急落銘柄の枠が埋まっている割合だけ資金を移す。乗り換えの売買コストも差し引く）

使い方: python combo_strategy.py [日足パネルのpickle] [週足パネルのpickle]
"""
import sys
import warnings
import numpy as np
import pandas as pd
import mtf_lab as M
import jq_lab as J
import value_lab as V
import rule_lab as R

warnings.simplefilter('ignore')
pd.set_option('display.width', 220)
pd.set_option('display.unicode.east_asian_width', True)
VALUE_N, CRASH_N = 20, 10
START = '2017-01-06'


def value_sleeve(wp, wk):
    """割安株の週次リターン（週足パネル上で、選定週の翌週始値で買い13週保有）"""
    b = lambda s: s.fillna(False).astype(bool)
    filt = b(wp['m_above']) & ~b(wp['heavy_margin']) & (wp['E/P'] > 0) & (wp['B/P'] > 0)
    dates = sorted(wp['date'].unique())[::13]
    picks = pd.concat([wp[(wp['date'] == dt) & filt].nlargest(VALUE_N, 'E/P')[['ticker', 'date']] for dt in dates])
    picks['week'] = picks['date'].dt.to_period('W-FRI').dt.end_time.dt.normalize().astype('datetime64[ns]')
    key = set(zip(picks['ticker'], picks['week']))
    sig = pd.Series([(t, w) in key for t, w in zip(wk['ticker'], wk['week'])], index=wk.index)
    tr = R.simulate(wk.assign(universe=True), sig, max_hold=13, one_position=False)
    p, eq, used = R.portfolio_mtm(tr, wk, max_pos=VALUE_N)
    return eq.pct_change().fillna(0), picks


def crash_sleeve(wk):
    wk = wk.copy()
    wk['r4rank'] = wk['ret4'].where(wk['universe']).groupby(wk['week']).rank(pct=True)
    wk['neg_ret4'] = -wk['ret4']
    sig = (wk['mk_ret4'] <= -7) & (wk['r4rank'] <= .2)
    tr = R.simulate(wk, sig, max_hold=8, target=.20)
    p, eq, used = R.portfolio_mtm(tr, wk, max_pos=CRASH_N, score='neg_ret4')
    r = eq.pct_change().fillna(0)
    frac = used.clip(upper=CRASH_N) / CRASH_N if used.max() > 1 else used.clip(upper=1)
    frac = (used.clip(upper=1)).reindex(r.index).fillna(0)       # 急落銘柄の枠が埋まっている割合
    r_inv = (r / frac.replace(0, np.nan)).fillna(0)               # 投資している分のリターン
    return r, r_inv, frac


def perf_row(name, r):
    r = r[r.index >= START]
    eq = (1 + r).cumprod()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    yr = eq.resample('YE').last().pct_change()
    yr.iloc[0] = eq.resample('YE').last().iloc[0] - 1
    seg = lambda a, b: (1 + r[(r.index >= a) & (r.index < b)]).prod() - 1
    return {'戦略': name, '年率': (eq.iloc[-1] ** (1 / yrs) - 1) * 100, '最大DD': (eq / eq.cummax() - 1).min() * 100,
            '最悪の年': yr.min() * 100, 'マイナスの年': f"{(yr < 0).sum()}/{len(yr)}",
            '2017-21累積': seg('2017', '2022') * 100, '2022-累積': seg('2022', '2100') * 100}, yr


def main():
    fin = J.load_fin()
    d = M.build_panel(cache=sys.argv[1] if len(sys.argv) > 1 else None)
    d = J.add_forward(d[d['date'] >= '2016-10-01'])
    wp = V.build_weekly_picks(d, fin)
    wk = pd.read_pickle(sys.argv[2]) if len(sys.argv) > 2 else R.add_features(R.load_panel())
    rv, picks = value_sleeve(wp, wk)
    rc, rc_inv, frac = crash_sleeve(wk)
    mk = pd.read_parquet('data/market_weekly.parquet')
    nk = mk[mk['symbol'] == '^N225'].set_index('week')['Close'].astype(float).pct_change().fillna(0)
    idx = rv.index[rv.index >= START]
    rv, rc, rc_inv, frac, nk = (s.reindex(idx).fillna(0) for s in (rv, rc, rc_inv, frac, nk))
    switch_cost = frac.diff().abs().fillna(0) * 0.002              # 乗り換えた分の売買コスト(往復0.2%)
    cases = {
        '日経平均の保有': nk,
        '割安株のみ': rv,
        '急落リバウンドのみ(普段は現金)': rc,
        '固定配分 割安株70%・急落用30%': 0.7 * rv + 0.3 * rc,
        '固定配分 割安株50%・急落用50%': 0.5 * rv + 0.5 * rc,
        '乗り換え(急落時は全額を急落銘柄へ)': (1 - frac) * rv + frac * rc_inv - switch_cost,
        '乗り換え(急落時は半分を急落銘柄へ)': (1 - frac / 2) * rv + frac / 2 * rc_inv - switch_cost / 2,
    }
    rows, yrs = [], {}
    for name, r in cases.items():
        row, yr = perf_row(name, r)
        rows.append(row)
        yrs[name] = (yr * 100).round(1)
    print(f"期間 {START}〜{idx[-1]:%Y-%m-%d}（往復コスト0.2%込み）  割安株 {VALUE_N}銘柄・急落 最大{CRASH_N}銘柄")
    print(pd.DataFrame(rows).set_index('戦略').round(1).to_string())
    Y = pd.DataFrame(yrs)
    Y.index = Y.index.year
    print("\n年別リターン(%)")
    print(Y.T.to_string())
    last = picks['date'].max()
    print(f"\n直近の割安株の選定（{last:%Y-%m-%d} 金曜引け時点、次の入れ替えまで保有）")
    cur = wp[(wp['date'] == last)].set_index('ticker').loc[picks[picks['date'] == last]['ticker']]
    print(cur[['raw_price', 'E/P']].assign(PER=lambda x: (1 / x['E/P']).round(1)).drop(columns='E/P').round(1).to_string())


if __name__ == '__main__':
    main()
