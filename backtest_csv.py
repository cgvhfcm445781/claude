import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import os
import time

warnings.simplefilter('ignore', FutureWarning)

# --- 設定 ---
OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "nikkei_intraday_backtest.csv")

# 検証する銘柄（市場区分ごと）。いずれも「日経225先物の夜間の動き」を地合いの参照として使う
# （グロース・スタンダードには専用の夜間先物が存在しないための代用）。
TRADE_TICKERS = {
    "プライム（日経225 ETF）": "1329.T",
    "グロース（グロース250 ETF）": "2516.T",
    "スタンダード（スタンダードTOP20 ETF）": "1551.T",
}

# 検証するチェックポイント（日本時間）。
CHECK_POINTS = {
    "10時成行": {"decision_time": "10:00", "exec_time": "10:00", "exec_field": "Close"},
    "12時半成行（前引け11:30時点で判断）": {"decision_time": "11:30", "exec_time": "12:30", "exec_field": "Open"},
}

# エントリー閾値の候補（この%以下に「現物の下げが日経先物の示唆より大きい」場合に買う）
GAP_THRESHOLDS = [-0.3, -0.5, -0.8, -1.0, -1.5]

# 「悪材料なし」判定用：米国11セクターETF、どれもこの%以上下げていなければ「地合い良好」
US_SECTOR_ETFS = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
NO_BAD_NEWS_THRESHOLDS = [-1.0, -1.3]

# 損切り・トレーリング幅の比較パターン
#   ・トレーリングあり：値上がりに応じて利確ラインを切り上げる（日中の監視が必要）
#   ・損切りのみ（trailing=None）：エントリー時に逆指値を置いたら、あとは大引けまで放置でOK
STOP_CONFIGS = [
    {"label": "損切り0.8%/トレーリング0.4%（要監視）", "loss_cut": 0.008, "trailing": 0.004},
    {"label": "損切り0.8%のみ（放置可）", "loss_cut": 0.008, "trailing": None},
    {"label": "損切り1.0%のみ（放置可）", "loss_cut": 0.010, "trailing": None},
    {"label": "損切り1.5%のみ（放置可）", "loss_cut": 0.015, "trailing": None},
]


def to_jst_naive(df):
    """タイムゾーン情報を日本時間に統一し、その後タイムゾーン情報を外す（比較しやすくするため）"""
    if df.index.tz is not None:
        df.index = df.index.tz_convert('Asia/Tokyo').tz_localize(None)
    return df


def get_prev_close(series, date):
    """指定日より前の直近の日足終値を取得"""
    s = series[series.index.normalize() < pd.Timestamp(date)]
    return float(s.iloc[-1]) if len(s) > 0 else np.nan


def get_asof(series, dt):
    """指定日時点で確定している直近の値を取得（未来の値は使わない）"""
    if series is None or len(series) == 0:
        return np.nan
    pos = series.index.searchsorted(dt, side='right') - 1
    if pos < 0:
        return np.nan
    return series.iloc[pos]


def get_bar_price(intraday_df, date, time_str, field='Close'):
    """指定日・指定時刻に最も近い（それ以前の）1時間足の値を取得（field='Close'ならその足の終値）"""
    target = pd.Timestamp(f"{date} {time_str}")
    day_bars = intraday_df[intraday_df.index.normalize() == pd.Timestamp(date)]
    day_bars = day_bars[day_bars.index <= target]
    if len(day_bars) == 0:
        return np.nan, None
    return float(day_bars[field].iloc[-1]), day_bars.index[-1]


def get_bar_price_at_or_after(intraday_df, date, time_str, field='Open'):
    """指定日・指定時刻以降で最初の1時間足の値を取得（field='Open'なら寄り付き＝成行の約定価格に相当）"""
    target = pd.Timestamp(f"{date} {time_str}")
    day_bars = intraday_df[intraday_df.index.normalize() == pd.Timestamp(date)]
    day_bars = day_bars[day_bars.index >= target]
    if len(day_bars) == 0:
        return np.nan, None
    return float(day_bars[field].iloc[0]), day_bars.index[0]


def simulate_exit(intraday_df, date, entry_bar_time, entry_price, loss_cut_rate, trailing_rate):
    """
    エントリー後の1時間足を1本ずつ確認し、損切り・トレーリング・大引けのいずれかで決済する。
    trailing_rate に None を渡すと、トレーリングは一切判定せず「損切りのみ、あとは大引けまで放置」の
    シミュレーションになる（日中ずっと画面を見ていなくても再現できる運用を想定）。
    """
    day_bars = intraday_df[intraday_df.index.normalize() == pd.Timestamp(date)]
    after_entry = day_bars[day_bars.index > entry_bar_time]

    m_price = entry_price
    for bar_time, row in after_entry.iterrows():
        c = float(row['Close'])
        if trailing_rate is not None:
            m_price = max(m_price, c)
            if c <= m_price * (1 - trailing_rate) and c > entry_price:
                return c, bar_time, "トレーリング"
        if c <= entry_price * (1 - loss_cut_rate):
            return c, bar_time, "損切り"

    if len(after_entry) > 0:
        return float(after_entry['Close'].iloc[-1]), after_entry.index[-1], "大引け"
    else:
        return entry_price, entry_bar_time, "大引け"


if __name__ == "__main__":
    start_time = time.time()

    print("日経225先物（日足、地合い判断の参照用、全市場区分で共通）を取得中...")
    df_fut_daily = yf.download("NIY=F", period="2y", interval="1d", progress=False, repair=True)
    if isinstance(df_fut_daily.columns, pd.MultiIndex):
        df_fut_daily.columns = df_fut_daily.columns.get_level_values(0)
    df_fut_daily = to_jst_naive(df_fut_daily)
    fut_daily_close = df_fut_daily['Close'].sort_index()

    print("米国11セクターETFデータ（悪材料なし判定用）を取得中...")
    us_chg_series = {}
    for etf in US_SECTOR_ETFS:
        df_etf = yf.download(etf, period="2y", interval="1d", progress=False, repair=True)
        if isinstance(df_etf.columns, pd.MultiIndex):
            df_etf.columns = df_etf.columns.get_level_values(0)
        df_etf = to_jst_naive(df_etf)
        chg = df_etf['Close'].pct_change() * 100
        us_chg_series[etf] = chg.sort_index()
    print(f"  取得完了: {len(us_chg_series)}銘柄")

    def worst_us_sector_chg(entry_dt):
        """指定日時点で確定している、米国11セクターのうち最も下げていたセクターの変化率(%)"""
        vals = [get_asof(us_chg_series[etf], entry_dt) for etf in US_SECTOR_ETFS]
        vals = [v for v in vals if not pd.isna(v)]
        return min(vals) if vals else np.nan

    all_records = []

    for market_label, ticker in TRADE_TICKERS.items():
        print(f"\n=== {market_label}（{ticker}）===")
        print(f"{ticker}（日足、判断・執行用）を取得中...")
        df_daily = yf.download(ticker, period="2y", interval="1d", progress=False, repair=True)
        if isinstance(df_daily.columns, pd.MultiIndex):
            df_daily.columns = df_daily.columns.get_level_values(0)
        df_daily = to_jst_naive(df_daily)
        if df_daily.empty:
            print(f"  ⚠ {ticker} の日足データが取得できませんでした。スキップします。")
            continue
        daily_close = df_daily['Close'].sort_index()

        print(f"{ticker}（1時間足、実際の売買シミュレーション用）を取得中...")
        df_intraday = yf.download(ticker, period="730d", interval="60m", progress=False, repair=True)
        if isinstance(df_intraday.columns, pd.MultiIndex):
            df_intraday.columns = df_intraday.columns.get_level_values(0)
        df_intraday = to_jst_naive(df_intraday)
        if df_intraday.empty:
            print(f"  ⚠ {ticker} の1時間足データが取得できませんでした。スキップします。")
            continue
        print(f"  取得件数: {len(df_intraday)}件, 期間: {df_intraday.index.min()} 〜 {df_intraday.index.max()}")

        trading_days = sorted(set(df_intraday.index.normalize()))

        for date in trading_days:
            date_str = date.strftime('%Y-%m-%d')
            cash_prev_close = get_prev_close(daily_close, date)
            fut_series_before = fut_daily_close[fut_daily_close.index.normalize() <= date]
            if len(fut_series_before) < 2 or pd.isna(cash_prev_close):
                continue
            fut_now = float(fut_series_before.iloc[-1])
            fut_prev = float(fut_series_before.iloc[-2])
            if fut_prev == 0:
                continue
            futures_overnight_pct = (fut_now - fut_prev) / fut_prev * 100

            day_bars_all = df_intraday[df_intraday.index.normalize() == date]
            if len(day_bars_all) == 0:
                continue
            day_close = float(day_bars_all['Close'].iloc[-1])
            worst_sector_chg = worst_us_sector_chg(date)

            for label, cfg in CHECK_POINTS.items():
                decision_price, decision_bar_time = get_bar_price(df_intraday, date, cfg['decision_time'])
                if pd.isna(decision_price) or decision_bar_time is None:
                    continue
                cash_intraday_pct = (decision_price - cash_prev_close) / cash_prev_close * 100
                gap = cash_intraday_pct - futures_overnight_pct

                exec_price, exec_bar_time = get_bar_price_at_or_after(
                    df_intraday, date, cfg['exec_time'], field=cfg['exec_field']
                )
                if pd.isna(exec_price) or exec_bar_time is None:
                    continue

                for stop_cfg in STOP_CONFIGS:
                    exit_price, exit_bar_time, exit_reason = simulate_exit(
                        df_intraday, date, exec_bar_time, exec_price,
                        stop_cfg['loss_cut'], stop_cfg['trailing']
                    )
                    ret = (exit_price - exec_price) / exec_price * 100

                    all_records.append({
                        'market': market_label, 'ticker': ticker, 'date': date_str,
                        'check_point': label, 'stop_config': stop_cfg['label'], 'gap': round(gap, 3),
                        'worst_us_sector_chg': round(worst_sector_chg, 3) if not pd.isna(worst_sector_chg) else None,
                        'exec_price': exec_price, 'exit_price': exit_price,
                        'exit_reason': exit_reason, 'return_pct': round(ret, 3),
                        'is_win': 1 if ret > 0 else 0,
                    })

    df_out = pd.DataFrame(all_records)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n✅ CSV出力完了: {OUTPUT_FILE}")
    print(f"   行数: {len(df_out)}, 列数: {len(df_out.columns)}")

    print(f"\n=== 市場区分 × チェックポイント × 損切り設定 × gap閾値 別の成績 ===")
    for market_label in TRADE_TICKERS.keys():
        df_m = df_out[df_out['market'] == market_label]
        if df_m.empty:
            print(f"\n########## {market_label}：データなし ##########")
            continue
        print(f"\n########## {market_label} ##########")
        for label in CHECK_POINTS.keys():
            print(f"\n--- {label} ---")
            for stop_cfg in STOP_CONFIGS:
                sub_label = df_m[(df_m['check_point'] == label) & (df_m['stop_config'] == stop_cfg['label'])]
                print(f"\n  [{stop_cfg['label']}]")
                for th in GAP_THRESHOLDS:
                    sub = sub_label[sub_label['gap'] <= th]
                    if len(sub) == 0:
                        print(f"    gap<={th}%: 該当なし")
                        continue
                    win_rate = sub['is_win'].mean() * 100
                    avg_ret = sub['return_pct'].mean()
                    reason_counts = sub['exit_reason'].value_counts().to_dict()
                    print(f"    gap<={th}%: 件数={len(sub)}, 勝率={win_rate:.1f}%, 平均リターン={avg_ret:.2f}%, 決済理由={reason_counts}")

    print(f"\n\n=== 【組み合わせ検証】 gap条件 × 悪材料なし条件（プライム・12時半成行・損切り0.8%のみ） ===")
    df_combo = df_out[
        (df_out['market'] == 'プライム（日経225 ETF）') &
        (df_out['check_point'] == '12時半成行（前引け11:30時点で判断）') &
        (df_out['stop_config'] == '損切り0.8%のみ（放置可）')
    ]
    for gap_th in GAP_THRESHOLDS:
        gap_sub = df_combo[df_combo['gap'] <= gap_th]
        if len(gap_sub) == 0:
            continue
        base_win = gap_sub['is_win'].mean() * 100
        base_ret = gap_sub['return_pct'].mean()
        print(f"\n--- gap<={gap_th}% 単体（件数={len(gap_sub)}, 勝率={base_win:.1f}%, 平均リターン={base_ret:.2f}%） ---")
        for sector_th in NO_BAD_NEWS_THRESHOLDS:
            combo_sub = gap_sub[gap_sub['worst_us_sector_chg'] > sector_th]
            if len(combo_sub) == 0:
                print(f"    + 悪材料なし(>{sector_th}%): 該当なし")
                continue
            win_rate = combo_sub['is_win'].mean() * 100
            avg_ret = combo_sub['return_pct'].mean()
            print(f"    + 悪材料なし(>{sector_th}%): 件数={len(combo_sub)}, 勝率={win_rate:.1f}%, 平均リターン={avg_ret:.2f}%")

    print(f"\n総実行時間: {time.time() - start_time:.1f}秒")