"""
J-Quants の決算データ（data_jq/fin_summary.parquet）を使った検証スクリプト。

1. 業績予想の修正（上方修正・下方修正）後の株価
2. 決算発表: 通期予想に対する進捗率、来期予想（ガイダンス）の伸び
3. 割安さ・質（PER、PBR、ROE、配当利回り）とその後の株価

方針
- 発表日の翌営業日の始値でエントリー（ザラ場中の発表でも先読みしないよう、常に翌営業日）
- 「同じ日のユニバース全体の平均」に対する超過リターンで評価（相場全体の上げ下げを除く）
- データは無料プランの範囲（約2年、12週遅れ）なので、結果は参考程度。t値は日（週）単位で計算

使い方: python jq_lab.py [日足パネルのキャッシュ(mtf_lab.py の build_panel で作るpickle)]
"""
import sys
import warnings
import numpy as np
import pandas as pd
import mtf_lab as M

warnings.simplefilter('ignore')
pd.set_option('display.width', 220)
pd.set_option('display.unicode.east_asian_width', True)
NUM = ['EPS', 'FEPS', 'NxFEPS', 'OP', 'FOP', 'NxFOP', 'Sales', 'FSales', 'NP', 'FNP', 'BPS', 'Eq', 'FDivAnn', 'ROE']
HORIZONS = (5, 20, 60)


def code_to_ticker(code):
    c = str(code)
    return (c[:4] if len(c) == 5 else c) + '.T'


def load_fin():
    f = pd.read_parquet('data_jq/fin_summary.parquet')
    for c in NUM:
        if c in f:
            f[c] = pd.to_numeric(f[c], errors='coerce')
    f['ticker'] = f['Code'].map(code_to_ticker)
    f['DiscDate'] = pd.to_datetime(f['DiscDate']).astype('datetime64[ns]')
    return f.sort_values(['ticker', 'DiscDate', 'DiscTime']).reset_index(drop=True)


def add_forward(d):
    """日足パネルに 5/20/60日後の超過リターン（翌日始値から）を追加"""
    d = d.sort_values(['ticker', 'date']).reset_index(drop=True)
    g = d.groupby('ticker')
    nxt_open = g['Open'].shift(-1)
    for h in HORIZONS:
        d[f'f{h}'] = (g['Close'].shift(-h) / nxt_open - 1) * 100
        d[f'x{h}'] = d[f'f{h}'] - d[f'f{h}'].where(d['universe']).groupby(d['date']).transform('mean')
    return d


def attach(events, d):
    """イベント（ticker, DiscDate）に、発表日（休日なら直前の営業日）の行の先行きリターンを付ける"""
    cols = ['ticker', 'date', 'universe', 'raw_price'] + [f'f{h}' for h in HORIZONS] + [f'x{h}' for h in HORIZONS]
    ev = events.sort_values('DiscDate')
    return pd.merge_asof(ev, d[cols].sort_values('date'), left_on='DiscDate', right_on='date',
                         by='ticker', direction='backward', tolerance=pd.Timedelta(days=5))


def report(ev, groups, title):
    rows = []
    for name, m in groups.items():
        x = ev[m & ev['universe'].fillna(False).astype(bool)].dropna(subset=['x20'])
        if len(x) < 30:
            continue
        r = {'区分': name, '件数': len(x)}
        for h in HORIZONS:
            r[f'超過{h}日'] = x[f'x{h}'].mean()
        # 同じ週の発表をまとめて1標本とみなした t値（20日）
        w = x.groupby(x['date'].dt.to_period('W'))['x20'].mean()
        r['t(20日)'] = w.mean() / w.std() * np.sqrt(len(w)) if len(w) > 2 else np.nan
        r['20日 勝率(素)'] = (x['f20'] > 0).mean() * 100
        rows.append(r)
    print(f"\n===== {title} =====")
    print(pd.DataFrame(rows).set_index('区分').round(2).to_string())


def forecast_revisions(fin, d):
    """業績予想の修正: 同じ会社・同じ決算期の直前の予想と比べた変化率"""
    f = fin.dropna(subset=['CurFYEn']).copy()
    rows = []
    for (t, fy), g in f.groupby(['ticker', 'CurFYEn']):
        g = g.sort_values(['DiscDate', 'DiscTime'])
        prev_eps = prev_op = np.nan
        for _, r in g.iterrows():
            if r['DocType'] == 'EarnForecastRevision':
                rows.append({'ticker': t, 'DiscDate': r['DiscDate'],
                             'rev_eps': (r['FEPS'] - prev_eps) / abs(prev_eps) * 100 if prev_eps else np.nan,
                             'rev_op': (r['FOP'] - prev_op) / abs(prev_op) * 100 if prev_op else np.nan})
            if pd.notna(r['FEPS']):
                prev_eps = r['FEPS']
            if pd.notna(r['FOP']):
                prev_op = r['FOP']
    ev = attach(pd.DataFrame(rows), d)
    rev = ev['rev_op'].fillna(ev['rev_eps'])
    report(ev, {'大幅上方修正(+20%以上)': rev >= 20, '上方修正(+5〜20%)': rev.between(5, 20, inclusive='left'),
                '小幅(±5%未満)': rev.abs() < 5, '下方修正(-5〜-20%)': rev.between(-20, -5, inclusive='right'),
                '大幅下方修正(-20%以下)': rev <= -20}, '1. 業績予想の修正（営業利益、無ければEPSの予想変化率）')


def earnings_releases(fin, d):
    """決算発表: 進捗率（3Q）、来期ガイダンス（本決算）"""
    q = fin[fin['DocType'].str.contains('FinancialStatements', na=False)].copy()
    q3 = q[q['CurPerType'] == '3Q'].copy()
    q3['progress'] = q3['OP'] / q3['FOP'] * 100      # 3Q累計の営業利益 ÷ 通期予想（通常は75%前後）
    ev = attach(q3[['ticker', 'DiscDate', 'progress']], d)
    p = ev['progress']
    report(ev, {'進捗率 90%以上': p >= 90, '80〜90%': p.between(80, 90, inclusive='left'),
                '70〜80%': p.between(70, 80, inclusive='left'), '70%未満': (p < 70) & (p > 0)},
           '2a. 第3四半期決算: 通期営業利益予想に対する進捗率')
    fy = q[q['CurPerType'] == 'FY'].copy()
    fy['guide'] = (fy['NxFOP'] / fy['OP'] - 1) * 100   # 来期予想営業利益 ÷ 今期実績
    fy.loc[fy['OP'] <= 0, 'guide'] = np.nan
    ev = attach(fy[['ticker', 'DiscDate', 'guide']], d)
    gd = ev['guide']
    report(ev, {'来期 +20%以上の増益予想': gd >= 20, '+5〜20%': gd.between(5, 20, inclusive='left'),
                '±5%未満': gd.abs() < 5, '-5%以下の減益予想': gd <= -5},
           '2b. 本決算: 来期の営業利益予想（ガイダンス）の伸び')


def valuation(fin, d):
    """割安さ・質: 各日の時点で公表済みの最新値で PER/PBR/ROE/配当利回り を計算し、週ごとに順位を付ける"""
    f = fin[fin['DocType'].str.contains('FinancialStatements', na=False)][
        ['ticker', 'DiscDate', 'FEPS', 'NxFEPS', 'BPS', 'ROE', 'FDivAnn', 'CurPerType']].copy()
    # 本決算では今期予想が NxFEPS 側に入るので、それを今期予想として使う
    f['fwd_eps'] = np.where(f['CurPerType'] == 'FY', f['NxFEPS'], f['FEPS'])
    f = f.sort_values('DiscDate')
    w = d[d['date'].dt.weekday == 4][['ticker', 'date', 'universe', 'raw_price', 'f20', 'x20', 'f60', 'x60']]
    w = pd.merge_asof(w.sort_values('date'), f.rename(columns={'DiscDate': 'avail'}).sort_values('avail'),
                      left_on='date', right_on='avail', by='ticker', direction='backward',
                      allow_exact_matches=False)   # 金曜引け時点で、前日までに発表済みのもの
    w = w[w['universe'].fillna(False).astype(bool)]
    w['E/P'] = w['fwd_eps'] / w['raw_price']
    w['B/P'] = w['BPS'] / w['raw_price']
    w['配当利回り'] = w['FDivAnn'] / w['raw_price']
    w['ROE'] = pd.to_numeric(w['ROE'], errors='coerce')
    rows = []
    for c in ['E/P', 'B/P', '配当利回り', 'ROE']:
        x = w.dropna(subset=[c, 'x20'])
        x = x.assign(q=x.groupby('date')[c].rank(pct=True))
        top, bot = x[x['q'] >= .8], x[x['q'] <= .2]
        s = top.groupby('date')['x20'].mean() - bot.groupby('date')['x20'].mean()
        ic = x.groupby('date').apply(lambda y: y[c].rank().corr(y['x20'].rank()))
        rows.append({'指標(大きいほど割安/高品質)': c, '週数': x['date'].nunique(),
                     '上位20%の超過20日': top['x20'].mean(), '下位20%の超過20日': bot['x20'].mean(),
                     '上位-下位': s.mean(), 't値(重なり補正)': s.mean() / s.std() * np.sqrt(len(s) / 4),
                     '上位-下位(60日)': top['x60'].mean() - bot['x60'].mean(), 'IC': ic.mean()})
    print("\n===== 3. 割安さ・質: 指標の上位20%と下位20%の、その後20日の超過リターン(%) =====")
    print(pd.DataFrame(rows).set_index('指標(大きいほど割安/高品質)').round(3).to_string())


def value_portfolio(fin, d, cost=0.2):
    """4週ごとに低PBR・低PERの上位銘柄に入れ替えるポートフォリオ（保有20営業日、往復コスト cost% 込み）"""
    f = fin[fin['DocType'].str.contains('FinancialStatements', na=False)][
        ['ticker', 'DiscDate', 'FEPS', 'NxFEPS', 'BPS', 'CurPerType']].copy()
    f['fwd_eps'] = np.where(f['CurPerType'] == 'FY', f['NxFEPS'], f['FEPS'])
    w = d[d['date'].dt.weekday == 4][['ticker', 'date', 'universe', 'raw_price', 'f20']]
    w = pd.merge_asof(w.sort_values('date'), f.rename(columns={'DiscDate': 'avail'}).sort_values('avail'),
                      left_on='date', right_on='avail', by='ticker', direction='backward', allow_exact_matches=False)
    w = w[w['universe'].fillna(False).astype(bool)].dropna(subset=['f20'])
    w['bp'], w['ep'] = w['BPS'] / w['raw_price'], w['fwd_eps'] / w['raw_price']
    dates = sorted(w['date'].unique())[::4]   # 保有期間(20営業日)と重ならないよう4週ごと
    res = {}
    for name, key, n in [('低PBR 上位10銘柄', 'bp', 10), ('低PBR 上位30銘柄', 'bp', 30), ('低PER 上位10銘柄', 'ep', 10)]:
        res[name] = [w[(w['date'] == dt) & (w[key] > 0)].nlargest(n, key)['f20'].mean() - cost for dt in dates]
    res['ユニバース全体の平均'] = [w.loc[w['date'] == dt, 'f20'].mean() for dt in dates]
    R = pd.DataFrame(res, index=pd.to_datetime(dates))
    eq = (1 + R / 100).cumprod()
    yrs = (R.index[-1] - R.index[0]).days / 365.25 + 4 / 52
    out = pd.DataFrame({'累積%': (eq.iloc[-1] - 1) * 100, '年率%': (eq.iloc[-1] ** (1 / yrs) - 1) * 100,
                        '最大DD%': ((eq / eq.cummax()) - 1).min() * 100,
                        '全体平均を上回った回': R.gt(R['ユニバース全体の平均'], axis=0).sum()})
    print(f"\n===== 4. 割安株ポートフォリオ（4週ごとに入れ替え、{len(R)}回、往復コスト{cost}%込み） =====")
    print(out.round(1).to_string())


def main():
    fin = load_fin()
    print(f"決算データ: {len(fin)} 件, {fin['ticker'].nunique()} 銘柄, {fin['DiscDate'].min():%Y-%m-%d}〜{fin['DiscDate'].max():%Y-%m-%d}")
    d = M.build_panel(cache=sys.argv[1] if len(sys.argv) > 1 else None)
    d = add_forward(d[d['date'] >= fin['DiscDate'].min() - pd.Timedelta(days=10)])
    forecast_revisions(fin, d)
    earnings_releases(fin, d)
    valuation(fin, d)
    value_portfolio(fin, d)


if __name__ == '__main__':
    main()
