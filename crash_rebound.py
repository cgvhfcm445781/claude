"""
急落リバウンド戦略のバックテスト ＋ 今週のシグナル確認。

ルール（週足、シグナルは金曜終値で判定 → 翌週の始値でエントリー）
- エントリー: 日経平均の4週騰落率が CRASH_TH% 以下（＝相場の急落局面）のとき、
              ユニバース内で4週騰落率が下位 LOSER_PCT の銘柄（よく下げた銘柄）を、下げた順に買う
- 決済:       HOLD_WEEKS 週後の終値（損切り・利確なし。急落局面では損切りが逆効果だったため）
- 資金:       MAX_POS 枠に等分。空いている枠は IDLE（'cash'=現金 / 'nikkei'=日経平均連動で運用）
- ユニバース: 直近4週の平均週間売買代金2.5億円以上、分割前の実際の株価が1,000〜4,000円

検証結果（rule_lab.py、往復コスト0.2%込み）の要点
- 10年(2016-2026)・20年(2006-2026)のどちらでも、日経平均を保有し続けるより高いリターン
- ただしシグナルは年に数回の急落局面だけ。最大ドローダウンは大きい（20年で-50%前後）
- 現在上場している銘柄だけのデータなので、急落後に倒産・上場廃止した銘柄が含まれていない。
  「よく下げた銘柄を買う」戦略はこのバイアスの影響を受けやすく、特に2008年の成績は実際より良く出ている
"""
import os
import numpy as np
import pandas as pd
import rule_lab as R

CRASH_TH = -5        # 日経平均の4週騰落率(%)がこれ以下で「急落局面」
LOSER_PCT = 0.10     # 4週騰落率が下位10%の銘柄が対象
HOLD_WEEKS = 8
MAX_POS = 10
IDLE = 'cash'        # 'cash' or 'nikkei'
OUTPUT_FILE = os.path.join('output', 'crash_rebound_trades.csv')


def main():
    df = R.add_features(R.load_panel())
    df['r4rank'] = df['ret4'].where(df['universe']).groupby(df['week']).rank(pct=True)
    df['neg_ret4'] = -df['ret4']
    signal = (df['mk_ret4'] <= CRASH_TH) & (df['r4rank'] <= LOSER_PCT)

    mk = pd.read_parquet(os.path.join(R.DATA_DIR, 'market_weekly.parquet'))
    nk = mk[mk['symbol'] == '^N225'].set_index('week')['Close'].astype(float)
    idle = nk.pct_change() if IDLE == 'nikkei' else None

    tr = R.simulate(df, signal, max_hold=HOLD_WEEKS)
    p, eq, used = R.portfolio_mtm(tr, df, max_pos=MAX_POS, score='neg_ret4', idle=idle)
    bench = (1 + nk.pct_change().reindex(eq.index).fillna(0)).cumprod()

    print(f"設定: 日経4週騰落<={CRASH_TH}% のとき 下落上位{LOSER_PCT:.0%} を {HOLD_WEEKS}週保有 / 最大{MAX_POS}銘柄 / 空き枠={IDLE}")
    print(f"期間: {eq.index[0]:%Y-%m-%d} 〜 {eq.index[-1]:%Y-%m-%d}")
    print(f"戦略      : {R.perf(eq)}   取引{len(p)}回  勝率{(p['ret'] > 0).mean() * 100:.0f}%  平均{p['ret'].mean():+.2f}%")
    print(f"日経平均  : {R.perf(bench)}")
    print(f"個別株に投資していた割合: {used.clip(upper=1).mean() * 100:.0f}%")
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
    print(f"\n===== 最新の確定週 {last:%Y-%m-%d} : 日経平均の4週騰落 {mk_now:+.1f}% "
          f"→ {'急落局面（シグナルあり）' if mk_now <= CRASH_TH else 'シグナルなし'} =====")
    if mk_now <= CRASH_TH:
        c = df[(df['week'] == last) & signal].sort_values('ret4')
        print(c[['ticker', 'name', 'sector', 'raw_price', 'ret4']].head(MAX_POS).round(1).to_string(index=False))


if __name__ == '__main__':
    main()
