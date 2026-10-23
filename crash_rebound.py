"""
急落リバウンド戦略のバックテスト ＋ 今週のシグナル確認。

ルール（週足、シグナルは金曜終値で判定 → 翌週の始値でエントリー）
- エントリー: 日経平均の4週騰落率が CRASH_TH% 以下（＝相場の急落局面）のとき、
              ユニバース内で4週騰落率が下位 LOSER_PCT の銘柄（よく下げた銘柄）を、下げた順に買う
              MODE='risk' のときは、さらに「VIXが直近4週の高値から15%以上下がっている（ピークアウト）」も条件
- 決済:       買値から +TARGET の指値で利確。届かなければ HOLD_WEEKS 週後の終値で売る
              損切り・トレーリングは入れない（optimize_crash.py の検証で、損切りは成績を悪化させ、
              トレーリングはほぼ効かなかったため）
- 資金:       MAX_POS 枠に等分。空いている枠は IDLE（'cash'=現金 / 'nikkei'=日経平均連動で運用）
- ユニバース: 直近4週の平均週間売買代金2.5億円以上、分割前の実際の株価が1,000〜4,000円

検証結果（optimize_crash.py、1,944通りを前半2008-2016/後半2017-2026に分けて比較、往復コスト0.2%込み）
- MODE='return'（リターン重視）: 年+16.4%、最大DD-39%、勝率69%（10銘柄）。前半+14.9%/後半+17.9%
- MODE='risk'  （リスク重視）  : 年+6.7%、最大DD-17.5%、勝率57%（10銘柄）
- 日経平均の保有: 年+7.1%、最大DD-61%
- 注意: 現在上場している銘柄だけのデータなので、急落後に倒産・上場廃止した銘柄が含まれていない。
  「よく下げた銘柄を買う」戦略はこの影響を受けやすく、特に2008年の成績は実際より良く出ている
"""
import os
import numpy as np
import pandas as pd
import rule_lab as R

MODE = 'return'      # 'return'（リターン重視） or 'risk'（リスク重視）
PRESETS = {
    'return': dict(CRASH_TH=-7, VIX_PEAK=False, HOLD_WEEKS=8, TARGET=0.20),
    'risk':   dict(CRASH_TH=-7, VIX_PEAK=True, HOLD_WEEKS=4, TARGET=0.40),
}
LOSER_PCT = 0.20     # 4週騰落率が下位20%の銘柄が対象（実際には下げた順に MAX_POS 銘柄まで）
MAX_POS = 10
IDLE = 'cash'        # 'cash' or 'nikkei'
OUTPUT_FILE = os.path.join('output', 'crash_rebound_trades.csv')


def vix_peak_off(weeks):
    """VIXが直近4週の高値から15%以上下がっている週（data/macro_daily.parquet が必要）"""
    m = pd.read_parquet(os.path.join(R.DATA_DIR, 'macro_daily.parquet'))
    vix = m[m['symbol'] == '^VIX'].set_index('date')['Close'].astype(float).resample('W-FRI').last()
    off = (vix < vix.rolling(4).max() * 0.85)
    return weeks.map(off).fillna(False).astype(bool)


def main():
    cfg = PRESETS[MODE]
    df = R.add_features(R.load_panel())
    df['r4rank'] = df['ret4'].where(df['universe']).groupby(df['week']).rank(pct=True)
    df['neg_ret4'] = -df['ret4']
    crash = df['mk_ret4'] <= cfg['CRASH_TH']
    if cfg['VIX_PEAK']:
        crash &= vix_peak_off(df['week'])
    signal = crash & (df['r4rank'] <= LOSER_PCT)

    mk = pd.read_parquet(os.path.join(R.DATA_DIR, 'market_weekly.parquet'))
    nk = mk[mk['symbol'] == '^N225'].set_index('week')['Close'].astype(float)
    idle = nk.pct_change() if IDLE == 'nikkei' else None

    tr = R.simulate(df, signal, max_hold=cfg['HOLD_WEEKS'], target=cfg['TARGET'])
    p, eq, used = R.portfolio_mtm(tr, df, max_pos=MAX_POS, score='neg_ret4', idle=idle)
    bench = (1 + nk.pct_change().reindex(eq.index).fillna(0)).cumprod()

    print(f"設定[{MODE}]: 日経4週騰落<={cfg['CRASH_TH']}%{' かつVIXピークアウト' if cfg['VIX_PEAK'] else ''} のとき "
          f"下落上位{LOSER_PCT:.0%}を下げた順に買い、+{cfg['TARGET']:.0%}で利確／最長{cfg['HOLD_WEEKS']}週 "
          f"/ 最大{MAX_POS}銘柄 / 空き枠={IDLE}")
    print(f"期間: {eq.index[0]:%Y-%m-%d} 〜 {eq.index[-1]:%Y-%m-%d}")
    print(f"戦略      : {R.perf(eq)}   取引{len(p)}回  勝率{(p['ret'] > 0).mean() * 100:.0f}%  平均{p['ret'].mean():+.2f}%")
    print(f"日経平均  : {R.perf(bench)}")
    print(f"決済理由  : {p['reason'].map({'target': '利確', 'time': '期間満了'}).value_counts().to_dict()}")
    yr = pd.DataFrame({'戦略': eq.resample('YE').last().pct_change(),
                       '日経平均': bench.resample('YE').last().pct_change()}).dropna() * 100
    yr.index = yr.index.year
    print("\n年別リターン(%)")
    print(yr.round(1).T.to_string())

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    out = p.merge(df[['ticker', 'name']].drop_duplicates('ticker'), on='ticker', how='left')
    out[['week', 'ticker', 'name', 'entry', 'exit', 'ret', 'weeks', 'reason']].to_csv(
        OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\nトレード一覧: {os.path.abspath(OUTPUT_FILE)}")

    # 今週（最新の確定週）の状況
    last = df['week'].max()
    mk_now = df.loc[df['week'] == last, 'mk_ret4'].iloc[0]
    on = bool(signal[df['week'] == last].any())
    print(f"\n===== 最新の確定週 {last:%Y-%m-%d} : 日経平均の4週騰落 {mk_now:+.1f}% "
          f"→ {'シグナルあり（翌営業日の寄り付きで買い）' if on else 'シグナルなし'} =====")
    if on:
        c = df[(df['week'] == last) & signal].sort_values('ret4').head(MAX_POS)
        c = c.assign(利確の目安=(c['raw_price'] * (1 + cfg['TARGET'])).round(0))
        print(c[['ticker', 'name', 'sector', 'raw_price', 'ret4', '利確の目安']].round(1).to_string(index=False))
        print(f"※利確は実際の買値 × {1 + cfg['TARGET']:.2f} に指値。届かなければ{cfg['HOLD_WEEKS']}週後の金曜終値で売り")


if __name__ == '__main__':
    main()
