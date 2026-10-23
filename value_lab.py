"""
普段の相場で持つ「割安株」の選び方・絞り込み・入れ替え間隔を比較するスクリプト（J-Quants 10年分の決算データ）。

- 割安さ: E/P（予想EPS÷株価）、B/P（BPS÷株価）、配当利回り、3つの順位の合成
- 絞り込み: 長期線（12か月線）より上、26週線が上向き、信用残が重くない、来期の減益予想が出ていない
- 入れ替え: 4週ごと（20営業日保有）／13週ごと（60営業日保有）。金曜の終値で選び、翌営業日の始値で買う
- 前半(2017-2021)・後半(2022-2026)に分けて評価（前半は割安株に逆風だった2018-2020年を含む）

使い方: python value_lab.py [日足パネルのキャッシュ(mtf_lab.build_panel で作るpickle)]
"""
import sys
import warnings
import numpy as np
import pandas as pd
import mtf_lab as M
import jq_lab as J
import margin_lab as ML

warnings.simplefilter('ignore')
pd.set_option('display.width', 240)
pd.set_option('display.unicode.east_asian_width', True)
COST = 0.2
SPLIT = '2022-01-01'


def build_weekly_picks(d, fin):
    f = fin[fin['DocType'].str.contains('FinancialStatements', na=False)][
        ['ticker', 'DiscDate', 'CurPerType', 'FEPS', 'NxFEPS', 'BPS', 'FDivAnn', 'OP', 'NxFOP']].copy()
    f['fwd_eps'] = np.where(f['CurPerType'] == 'FY', f['NxFEPS'], f['FEPS'])
    # 来期の減益予想: 本決算で出た来期営業利益予想が今期実績を下回る（次の本決算まで有効）
    fy = f[f['CurPerType'] == 'FY'][['ticker', 'DiscDate', 'OP', 'NxFOP']].copy()
    fy['guide_down'] = fy['NxFOP'] < fy['OP']
    w = d[d['date'].dt.weekday == 4].copy()
    w = pd.merge_asof(w.sort_values('date'), f[['ticker', 'DiscDate', 'fwd_eps', 'BPS', 'FDivAnn']]
                      .rename(columns={'DiscDate': 'avail'}).sort_values('avail'),
                      left_on='date', right_on='avail', by='ticker', direction='backward', allow_exact_matches=False)
    w = pd.merge_asof(w.sort_values('date'), fy[['ticker', 'DiscDate', 'guide_down']]
                      .rename(columns={'DiscDate': 'avail2'}).sort_values('avail2'),
                      left_on='date', right_on='avail2', by='ticker', direction='backward', allow_exact_matches=False)
    # 信用残（1週遅れで使える値）
    m = ML.load_margin().rename(columns={'week': 'mweek'}).sort_values('mweek')
    w = pd.merge_asof(w.sort_values('date'), m, left_on='date', right_on='mweek', by='ticker', direction='backward',
                      tolerance=pd.Timedelta(days=10))
    w = w[w['universe'].fillna(False).astype(bool)].copy()
    w['E/P'] = w['fwd_eps'] / w['raw_price']
    w['B/P'] = w['BPS'] / w['raw_price']
    w['配当利回り'] = w['FDivAnn'] / w['raw_price']
    r = lambda c: w.groupby('date')[c].rank(pct=True)
    w['合成'] = r('E/P') + r('B/P') + r('配当利回り')
    # 信用残の重さ: 買い残・売り残 ÷ 20日平均売買代金から逆算した概算出来高（日足パネルの value20 / 株価）
    vol_w = w['value20'] / w['Close'] * 5
    w['heavy_margin'] = ((w['LongVol'] / vol_w) > (w['LongVol'] / vol_w).groupby(w['date']).transform('median') * 2) | \
                        ((w['ShrtVol'] / vol_w) > (w['ShrtVol'] / vol_w).groupby(w['date']).transform('median') * 2)
    return w


def backtest(w, score, filt, n, hold):
    fcol, step = ('f20', 4) if hold == 20 else ('f60', 13)
    dates = sorted(w['date'].unique())[::step]
    rets = []
    for dt in dates:
        x = w[(w['date'] == dt) & filt.reindex(w.index, fill_value=False)]
        x = x[(x['E/P'] > 0) & (x['B/P'] > 0)].dropna(subset=[score, fcol])
        rets.append(x.nlargest(n, score)[fcol].mean() - COST if len(x) >= n else np.nan)
    base = [w.loc[w['date'] == dt, fcol].mean() for dt in dates]
    R = pd.DataFrame({'s': rets, 'b': base}, index=pd.to_datetime(dates)).dropna()
    return R, step


def stats(R, step):
    eq = (1 + R['s'] / 100).cumprod()
    yrs = len(R) * step / 52
    out = {'年率': (eq.iloc[-1] ** (1 / yrs) - 1) * 100, '最大DD': ((eq / eq.cummax()) - 1).min() * 100,
           '全体を上回った割合': (R['s'] > R['b']).mean() * 100}
    for lab, m in (('前半', R.index < SPLIT), ('後半', R.index >= SPLIT)):
        e = (1 + R.loc[m, 's'] / 100).cumprod()
        b = (1 + R.loc[m, 'b'] / 100).cumprod()
        y = m.sum() * step / 52
        out[f'{lab}年率'] = (e.iloc[-1] ** (1 / y) - 1) * 100
        out[f'{lab}全体平均'] = (b.iloc[-1] ** (1 / y) - 1) * 100
    return out


def main():
    fin = J.load_fin()
    d = M.build_panel(cache=sys.argv[1] if len(sys.argv) > 1 else None)
    d = J.add_forward(d[d['date'] >= '2016-10-01'])
    w = build_weekly_picks(d, fin)
    b = lambda s: s.fillna(False).astype(bool)
    T = pd.Series(True, index=w.index)
    FILTERS = {
        'なし': T,
        '12か月線より上': b(w['m_above']),
        '26週線が上向き': b(w['w_slope_up']),
        '信用残が重くない': ~b(w['heavy_margin']),
        '減益予想なし': ~b(w['guide_down']),
        '12か月線上&信用残軽い': b(w['m_above']) & ~b(w['heavy_margin']),
        '12か月線上&信用残軽い&減益予想なし': b(w['m_above']) & ~b(w['heavy_margin']) & ~b(w['guide_down']),
    }
    rows = []
    for score in ['E/P', 'B/P', '配当利回り', '合成']:
        for fname, filt in FILTERS.items():
            for n in (10, 20):
                for hold in (20, 60):
                    R, step = backtest(w, score, filt, n, hold)
                    rows.append({'割安さ': score, '絞り込み': fname, '銘柄数': n,
                                 '入れ替え': '4週' if hold == 20 else '13週', **stats(R, step)})
    res = pd.DataFrame(rows)
    res.to_csv('output/value_lab.csv', index=False, encoding='utf-8-sig')
    base = res.iloc[0]
    print(f"ユニバース全体の平均: 前半 年{res['前半全体平均'].iloc[0]:.1f}% / 後半 年{res['後半全体平均'].iloc[0]:.1f}%（4週入れ替えの場合）")
    print("\n===== 前半・後半とも全体平均を上回った条件（後半年率の高い順、上位25） =====")
    ok = res[(res['前半年率'] > res['前半全体平均']) & (res['後半年率'] > res['後半全体平均'])]
    print(f"（{len(ok)} / {len(res)} 条件）")
    print(ok.sort_values('年率', ascending=False).head(25).round(1).to_string(index=False))
    print("\n===== 絞り込みごとの効き方（全条件の年率の中央値） =====")
    print(res.groupby('絞り込み')[['年率', '前半年率', '後半年率', '最大DD', '全体を上回った割合']].median().round(1).to_string())
    print("\n===== 割安さの測り方ごと =====")
    print(res.groupby('割安さ')[['年率', '前半年率', '後半年率', '最大DD', '全体を上回った割合']].median().round(1).to_string())


if __name__ == '__main__':
    main()
