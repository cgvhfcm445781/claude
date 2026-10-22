"""
ユニバース（市場区分・業種・規模）、季節性、業種ローテーションの検証をまとめたスクリプト。

1. 急落リバウンド戦略をユニバースを変えて比較（プライムのみ、ディフェンシブ除外など）
2. 季節性: 月別、月初の週、Sell in May
3. 年末の節税売り → 1月の反発（前年の下落上位銘柄）
4. 業種ローテーション（業種平均の過去騰落で上位/下位25%の業種）

注意
- 市場区分・業種は現在のもの（2022年の市場再編前にもさかのぼって当てはめている）
- 株価は配当調整済みなので、3月・9月の強さには配当分が含まれる
"""
import warnings
import numpy as np
import pandas as pd
import rule_lab as R

warnings.simplefilter('ignore')
pd.set_option('display.width', 220)
pd.set_option('display.unicode.east_asian_width', True)

DEF = ['食料品', '医薬品', '電気・ガス業', '陸運業', '小売業', '情報・通信業', '水産・農林業']  # 一般的なディフェンシブ業種
CYC = ['電気機器', '機械', '精密機器', '化学', '鉄鋼', '非鉄金属', '輸送用機器', '卸売業', '証券、商品先物取引業',
       '海運業', 'ガラス・土石製品', '金属製品', 'ゴム製品', '繊維製品', 'パルプ・紙']
PER = [('〜2012', None, '2013-01-01'), ('2013-2019', '2013-01-01', '2020-01-01'), ('2020〜', '2020-01-01', None)]


def per_label(w):
    return np.select([w < '2013-01-01', w < '2020-01-01'], ['〜2012', '2013-19'], '2020〜')


def universe_test(df):
    base, wide = df['universe'], df['value4'] >= R.MIN_WEEKLY_VALUE
    prime, defe = df['market'] == 'プライム', df['sector'].isin(DEF)
    univ = {
        '標準(1000-4000円・P+S)': base,
        'プライムのみ': base & prime,
        'スタンダードのみ': base & (df['market'] == 'スタンダード'),
        'ディフェンシブ除外': base & ~defe,
        '景気敏感のみ': base & df['sector'].isin(CYC),
        'プライム&ディフェンシブ除外': base & prime & ~defe,
        '価格制限なし': wide,
        '価格制限なし&プライム&ディフェ除外': wide & prime & ~defe,
        '売買代金上位(週10億以上)': base & (df['value4'] >= 1e9),
    }
    crash = df['mk_ret4'] <= -5
    rows = []
    for name, U in univ.items():
        r4 = df['ret4'].where(U).groupby(df['week']).rank(pct=True)
        d = df.assign(universe=U, neg_ret4=-df['ret4'])
        tr = R.simulate(d, crash & (r4 <= .1), max_hold=8)
        p, eq, _ = R.portfolio_mtm(tr, d, max_pos=10, score='neg_ret4')
        r = {'ユニバース': name, '成績': R.perf(eq), '取引': len(p), '平均': round(p['ret'].mean(), 2)}
        for nm, a, b in PER:
            e = eq[(eq.index >= (a or '1900')) & (eq.index < (b or '2100'))]
            r[nm] = f"{(e.iloc[-1] / e.iloc[0] - 1) * 100:+.0f}%"
        rows.append(r)
    print('===== 1. 急落リバウンド戦略 × ユニバース（日経4週-5%以下で下落上位10%を8週、最大10銘柄、空き枠現金） =====')
    print(pd.DataFrame(rows).set_index('ユニバース').to_string())


def seasonality(df):
    g = df.groupby('ticker')
    df = df.assign(wret=g['Close'].pct_change() * 100, prev_univ=g['universe'].shift().fillna(False).astype(bool))
    u = df[df['prev_univ']]   # 前週時点でユニバースに入っていた銘柄（先読みなし）
    mk = pd.read_parquet('data/market_weekly.parquet')
    nk = mk[mk['symbol'] == '^N225'].set_index('week')['Close'].astype(float).pct_change() * 100
    W = pd.DataFrame({'ew': u.groupby('week')['wret'].mean(), 'nk': nk}).dropna()
    W['m'], W['y'] = W.index.month, W.index.year
    mo = W.groupby(['y', 'm'])[['ew', 'nk']].sum().reset_index()
    t = mo.groupby('m').agg(ew=('ew', 'mean'), ew_pos=('ew', lambda s: (s > 0).sum()),
                            nk=('nk', 'mean'), nk_pos=('nk', lambda s: (s > 0).sum()), 年数=('ew', 'size'))
    t['ew_t'] = mo.groupby('m')['ew'].apply(lambda s: s.mean() / s.std() * np.sqrt(len(s)))
    print('\n===== 2. 月別リターン(%)  ew=ユニバース等金額, nk=日経平均, *_pos=プラスの年数 =====')
    print(t.round(2).to_string())
    first = pd.to_datetime(u.groupby('week')['first_day'].first().reindex(W.index))
    W['turn'] = first.dt.day <= 5
    a, b = W.loc[W['turn'], 'ew'], W.loc[~W['turn'], 'ew']
    print(f"月初を含む週 {a.mean():+.3f}% / その他の週 {b.mean():+.3f}%  差のt値 "
          f"{(a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)):.2f}")
    W['half'] = np.where(W['m'].between(5, 10), '5-10月', '11-4月')
    h = W.groupby(['y', 'half'])['ew'].sum().unstack()
    print(f"Sell in May: 年平均 {h.mean().round(2).to_dict()}  11-4月が上回った年 {(h['11-4月'] > h['5-10月']).sum()}/{len(h)}")


def year_end_reversal(df):
    g = df.groupby('ticker')
    df = df.assign(fwd6=g['Close'].shift(-6) / g['Open'].shift(-1) * 100 - 100)
    u = df[df['universe']]
    print('\n===== 3. 年末の節税売り → 1月の反発（12月に前年の下落上位10%を買い6週保有） =====')
    for md, label in [((12, 6), '12月第1週'), ((12, 13), '12月中旬'), ((12, 20), '12月下旬')]:
        rows = []
        for y in sorted(u['week'].dt.year.unique()):
            wk = u.loc[u['week'] >= pd.Timestamp(y, *md), 'week'].min()
            x = u[u['week'] == wk].dropna(subset=['fwd6', 'ret52'])
            if pd.isna(wk) or len(x) < 50:
                continue
            lo = x[x['ret52'].rank(pct=True) <= .1]
            rows.append({'全体': x['fwd6'].mean(), '下落上位': lo['fwd6'].mean()})
        r = pd.DataFrame(rows)
        print(f"{label}: 全体 {r['全体'].mean():+.2f}%  下落上位 {r['下落上位'].mean():+.2f}%  "
              f"下落上位が上回った年 {(r['下落上位'] > r['全体']).sum()}/{len(r)}")


def sector_rotation(df):
    g = df.groupby('ticker')
    df = df.assign(**{f'fwd{h}': g['Close'].shift(-h) / g['Open'].shift(-1) * 100 - 100 for h in (4, 8, 13)})
    u = df[df['universe'] & df['sector'].notna()]
    S = u.groupby(['week', 'sector']).agg(n=('ret4', 'size'), r4=('ret4', 'mean'), r13=('ret13', 'mean'),
                                          r26=('ret26', 'mean'), r52=('ret52', 'mean'),
                                          f4=('fwd4', 'mean'), f8=('fwd8', 'mean'), f13=('fwd13', 'mean')).reset_index()
    S = S[S['n'] >= 5]
    for c in ('f4', 'f8', 'f13'):
        S['x' + c] = S[c] - S.groupby('week')[c].transform('mean')
    S['per'] = per_label(S['week'])
    rows = []
    for sig in ('r4', 'r13', 'r26', 'r52'):
        q = S.groupby('week')[sig].rank(pct=True)
        for side, m in (('上位25%', q >= .75), ('下位25%', q <= .25)):
            x = S[m]
            for hn in (4, 8, 13):
                w = x.groupby('week')[f'xf{hn}'].mean()
                r = {'業種の過去騰落': sig, '側': side, '保有': f'{hn}週', 't値': w.mean() / w.std() * np.sqrt(len(w) / hn)}
                for p in ('〜2012', '2013-19', '2020〜'):
                    r[p] = x.loc[x['per'] == p, f'xf{hn}'].mean()
                rows.append(r)
    print('\n===== 4. 業種ローテーション: 業種の超過リターン(%)（t値は週単位・重なり補正） =====')
    print(pd.DataFrame(rows).round(2).to_string(index=False))


def main():
    df = R.add_features(R.load_panel())
    universe_test(df)
    seasonality(df)
    year_end_reversal(df)
    sector_rotation(df)


if __name__ == '__main__':
    main()
