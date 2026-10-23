"""
J-Quants の信用残（銘柄別・週次）と業種別空売り比率（日次）を使った検証スクリプト。

先読みを防ぐ扱い
- 信用残は金曜(申込日)時点の値が翌週火曜ごろ公表されるので、1週遅らせて使う
- 空売り比率は日次の値を週平均にし、その週の終値時点で使う

1. 相場全体: 空売り比率・信用買い残の増減・信用倍率 の悲観度（過去の中での百分位）別に、
   その後13週の日経平均の上昇余地／下値リスク（sentiment_lab.py と同じ測り方）
2. 急落リバウンド戦略の候補銘柄を、信用買い残の重さ・信用倍率・業種の空売り比率で分けたときの成績
3. 普段の相場での銘柄選び（信用倍率・買い残の重さと、その後20日の超過リターンの関係）

使い方: python margin_lab.py [週足パネルのキャッシュ(pickle)]
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import rule_lab as R
import sentiment_lab as SL

warnings.simplefilter('ignore')
pd.set_option('display.width', 230)
pd.set_option('display.unicode.east_asian_width', True)
START = '2017-06-01'   # 百分位の計算に必要な助走期間（約半年）を除いた評価開始


def code_to_ticker(code):
    c = str(code)
    return (c[:4] if len(c) == 5 else c) + '.T'


def load_margin():
    m = pd.read_parquet('data_jq/margin.parquet')
    m['ticker'] = m['Code'].map(code_to_ticker)
    for c in ['LongVol', 'ShrtVol']:
        m[c] = pd.to_numeric(m[c], errors='coerce')
    m['week'] = pd.to_datetime(m['Date']).dt.to_period('W-FRI').dt.end_time.dt.normalize().astype('datetime64[ns]')
    m = m.groupby(['ticker', 'week'], as_index=False)[['LongVol', 'ShrtVol']].sum()
    # 公表は翌週なので、1週遅らせた週に「使える値」として置く
    m['week'] = m['week'] + pd.Timedelta(weeks=1)
    return m


def load_short_ratio():
    s = pd.read_parquet('data_jq/short_ratio.parquet')
    for c in ['SellExShortVa', 'ShrtWithResVa', 'ShrtNoResVa']:
        s[c] = pd.to_numeric(s[c], errors='coerce')
    s['short'] = s['ShrtWithResVa'] + s['ShrtNoResVa']
    s['week'] = pd.to_datetime(s['Date']).dt.to_period('W-FRI').dt.end_time.dt.normalize().astype('datetime64[ns]')
    sec = s.groupby(['week', 'S33'])[['short', 'SellExShortVa']].sum()
    sec['ratio'] = sec['short'] / (sec['short'] + sec['SellExShortVa']) * 100
    mkt = s.groupby('week')[['short', 'SellExShortVa']].sum()
    mkt['ratio'] = mkt['short'] / (mkt['short'] + mkt['SellExShortVa']) * 100
    return sec.reset_index(), mkt['ratio']


def attach(df):
    m = load_margin()
    df = df.merge(m, on=['ticker', 'week'], how='left')
    vol13 = df.groupby('ticker')['Volume'].transform(lambda v: v.rolling(13).mean())
    df['買い残の重さ(週)'] = df['LongVol'] / vol13                 # 買い残が平均週間出来高の何週分か
    df['信用倍率'] = df['LongVol'] / df['ShrtVol'].replace(0, np.nan)
    df['買い残4週変化%'] = df.groupby('ticker')['LongVol'].pct_change(4) * 100
    df['売り残の重さ(週)'] = df['ShrtVol'] / vol13
    # 業種の空売り比率（銘柄の33業種コードを master から）
    master = pd.read_parquet('data_jq/master.parquet')
    master['ticker'] = master['Code'].map(code_to_ticker)
    s33 = master.sort_values('Date').groupby('ticker')['S33'].last()
    df['S33'] = df['ticker'].map(s33)
    sec, mkt = load_short_ratio()
    df = df.merge(sec[['week', 'S33', 'ratio']].rename(columns={'ratio': '業種の空売り比率'}), on=['week', 'S33'], how='left')
    return df, mkt


def market_level(df, mkt_short):
    T = SL.market_table(df)
    u = df[df['value4'] >= R.MIN_WEEKLY_VALUE]
    agg = u.groupby('week')[['LongVol', 'ShrtVol']].sum()
    T['空売り比率(市場全体)'] = mkt_short.reindex(T.index)
    T['信用買い残の4週減少率'] = -(agg['LongVol'] / agg['LongVol'].shift(4) - 1).reindex(T.index) * 100
    T['信用倍率の低さ'] = -(agg['LongVol'] / agg['ShrtVol']).reindex(T.index)
    T['信用売り残の4週増加率'] = (agg['ShrtVol'] / agg['ShrtVol'].shift(4) - 1).reindex(T.index) * 100
    cols = ['空売り比率(市場全体)', '信用買い残の4週減少率', '信用倍率の低さ', '信用売り残の4週増加率']
    P = pd.DataFrame({c: T[c].expanding(min_periods=26).rank(pct=True) for c in cols})
    res = SL.bucket_table(T[T.index >= START], P[P.index >= START], cols)
    print(f"===== 1. 相場全体の需給（{START}〜、百分位が高いほど悲観的）: その後13週の日経平均 =====")
    for c in cols:
        x = res[res['指標'] == c].drop(columns='指標').set_index('悲観度')
        print(f"\n--- {c}")
        print(x.round(2).to_string())
    return T, P


def crash_candidates(df):
    """急落リバウンド戦略（日経4週-7%、下落上位20%）の候補銘柄を、需給で分けて8週後までの成績を比較"""
    df = df.copy()
    df['r4rank'] = df['ret4'].where(df['universe']).groupby(df['week']).rank(pct=True)
    g = df.groupby('ticker')
    df['fwd8'] = (g['Close'].shift(-8) / g['Open'].shift(-1) - 1) * 100
    c = df[(df['mk_ret4'] <= -7) & (df['r4rank'] <= .2) & (df['week'] >= START)].dropna(subset=['fwd8'])
    print(f"\n===== 2. 急落リバウンドの候補（{len(c)}件、{c['week'].nunique()}週）: 需給で分けた8週後までのリターン(%) =====")
    rows = []
    for col in ['買い残の重さ(週)', '信用倍率', '買い残4週変化%', '売り残の重さ(週)', '業種の空売り比率']:
        x = c.dropna(subset=[col])
        q = x.groupby('week')[col].rank(pct=True)   # 同じ週の候補の中での順位
        for lab, m in (('少ない/低い 1/3', q <= 1 / 3), ('中間', (q > 1 / 3) & (q <= 2 / 3)), ('多い/高い 1/3', q > 2 / 3)):
            y = x[m]
            rows.append({'指標': col, '区分': lab, '件数': len(y), '8週後 平均': y['fwd8'].mean(),
                         '勝率': (y['fwd8'] > 0).mean() * 100, '+20%到達率(近似)': (y['fwd8'] >= 20).mean() * 100})
    print(pd.DataFrame(rows).set_index(['指標', '区分']).round(2).to_string())


def normal_times(df):
    """普段の相場（急落局面以外）: 需給の指標の上位20%と下位20%の、その後4週の超過リターン"""
    g = df.groupby('ticker')
    df = df.assign(fwd4=(g['Close'].shift(-4) / g['Open'].shift(-1) - 1) * 100)
    u = df[df['universe'] & (df['mk_ret4'] > -5) & (df['week'] >= START)].dropna(subset=['fwd4'])
    u = u.assign(ex4=u['fwd4'] - u.groupby('week')['fwd4'].transform('mean'))
    rows = []
    for col in ['買い残の重さ(週)', '信用倍率', '買い残4週変化%', '売り残の重さ(週)', '業種の空売り比率']:
        x = u.dropna(subset=[col])
        q = x.groupby('week')[col].rank(pct=True)
        top = x[q >= .8].groupby('week')['ex4'].mean()
        bot = x[q <= .2].groupby('week')['ex4'].mean()
        s = (top - bot).dropna()
        per = lambda a, b: (s[(s.index >= a) & (s.index < b)]).mean()
        rows.append({'指標': col, '上位20%': top.mean(), '下位20%': bot.mean(), '上位-下位': s.mean(),
                     't値(重なり補正)': s.mean() / s.std() * np.sqrt(len(s) / 4),
                     '2017-20': per('2017', '2021'), '2021-23': per('2021', '2024'), '2024-': per('2024', '2100')})
    print("\n===== 3. 普段の相場: 需給の指標で上位20%と下位20%の、その後4週の超過リターン(%) =====")
    print(pd.DataFrame(rows).set_index('指標').round(3).to_string())


def main():
    df = pd.read_pickle(sys.argv[1]) if len(sys.argv) > 1 else R.add_features(R.load_panel())
    df, mkt_short = attach(df)
    market_level(df, mkt_short)
    crash_candidates(df)
    normal_times(df)


if __name__ == '__main__':
    main()
