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

# 検証するチェックポイント（日本時間）。
# decision_time: この時刻までのデータで「買うかどうか」を判断する
# exec_time / exec_field: 実際に約定する価格（10時パターンはその場で成行、
#                          12時半パターンはお昼休み明けの成行注文を想定し12:30の始値で約定）
CHECK_POINTS = {
    "10時成行": {"decision_time": "10:00", "exec_time": "10:00", "exec_field": "Close"},
    "12時半成行（前引け11:30時点で判断）": {"decision_time": "11:30", "exec_time": "12:30", "exec_field": "Open"},
}

# エントリー閾値の候補（この%以下に「現物の下げが先物の示唆より大きい」場合に買う）
GAP_THRESHOLDS = [-0.3, -0.5, -0.8, -1.0, -1.5]


def to_jst_naive(df):
    """タイムゾーン情報を日本時間に統一し、その後タイムゾーン情報を外す（比較しやすくするため）"""
    if df.index.tz is not None:
        df.index = df.index.tz_convert('Asia/Tokyo').tz_localize(None)
    return df


if __name__ == "__main__":
    start_time = time.time()

    print("日経平均（日足、参照用）を取得中...")
    df_nk_daily = yf.download("^N225", period="2y", interval="1d", progress=False, repair=True)
    if isinstance(df_nk_daily.columns, pd.MultiIndex):
        df_nk_daily.columns = df_nk_daily.columns.get_level_values(0)
    df_nk_daily = to_jst_naive(df_nk_daily)
    nk_daily_close = df_nk_daily['Close'].sort_index()

    print("日経225先物（日足、参照用）を取得中...")
    df_fut_daily = yf.download("NIY=F", period="2y", interval="1d", progress=False, repair=True)
    if isinstance(df_fut_daily.columns, pd.MultiIndex):
        df_fut_daily.columns = df_fut_daily.columns.get_level_values(0)
    df_fut_daily = to_jst_naive(df_fut_daily)
    fut_daily_close = df_fut_daily['Close'].sort_index()

    print("日経平均（1時間足、直近の日中値動き用）を取得中...")
    # yfinanceの制約：60分足は最大730日分まで遡れる
    df_nk_intraday = yf.download("^N225", period="730d", interval="60m", progress=False, repair=True)
    if isinstance(df_nk_intraday.columns, pd.MultiIndex):
        df_nk_intraday.columns = df_nk_intraday.columns.get_level_values(0)
    df_nk_intraday = to_jst_naive(df_nk_intraday)
    print(f"  取得件数: {len(df_nk_intraday)}件, 期間: {df_nk_intraday.index.min()} 〜 {df_nk_intraday.index.max()}")

    def get_prev_close(series, date):
        """指定日より前の直近の日足終値を取得"""
        s = series[series.index.normalize() < pd.Timestamp(date)]
        return float(s.iloc[-1]) if len(s) > 0 else np.nan

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

    def get_day_close(intraday_df, date):
        """その日の最終バー（大引けに最も近い）の終値を取得"""
        day_bars = intraday_df[intraday_df.index.normalize() == pd.Timestamp(date)]
        if len(day_bars) == 0:
            return np.nan
        return float(day_bars['Close'].iloc[-1])

    trading_days = sorted(set(df_nk_intraday.index.normalize()))
    print(f"対象営業日数: {len(trading_days)}日")

    records = []
    for date in trading_days:
        date_str = date.strftime('%Y-%m-%d')
        cash_prev_close = get_prev_close(nk_daily_close, date)
        fut_now, fut_prev = None, None
        fut_series_before = fut_daily_close[fut_daily_close.index.normalize() <= date]
        if len(fut_series_before) >= 2:
            fut_now = float(fut_series_before.iloc[-1])
            fut_prev = float(fut_series_before.iloc[-2])
        elif len(fut_series_before) == 1:
            fut_now = float(fut_series_before.iloc[-1])

        if pd.isna(cash_prev_close) or fut_now is None or fut_prev is None or fut_prev == 0:
            continue

        futures_overnight_pct = (fut_now - fut_prev) / fut_prev * 100
        day_close = get_day_close(df_nk_intraday, date)
        if pd.isna(day_close):
            continue

        for label, cfg in CHECK_POINTS.items():
            # 「買うかどうか」の判断は decision_time 時点のデータで行う
            decision_price, decision_bar_time = get_bar_price(df_nk_intraday, date, cfg['decision_time'])
            if pd.isna(decision_price) or decision_bar_time is None:
                continue
            cash_intraday_pct = (decision_price - cash_prev_close) / cash_prev_close * 100
            gap = cash_intraday_pct - futures_overnight_pct

            # 実際の約定価格は exec_time 時点（成行の場合はそのバーの始値）
            exec_price, exec_bar_time = get_bar_price_at_or_after(
                df_nk_intraday, date, cfg['exec_time'], field=cfg['exec_field']
            )
            if pd.isna(exec_price) or exec_bar_time is None:
                continue

            # 約定時点から、その日の大引けまで持った場合のリターン
            ret_to_close = (day_close - exec_price) / exec_price * 100

            records.append({
                'date': date_str,
                'check_point': label,
                'decision_bar_time': str(decision_bar_time),
                'exec_bar_time': str(exec_bar_time),
                'cash_prev_close': cash_prev_close,
                'futures_overnight_pct': round(futures_overnight_pct, 3),
                'cash_intraday_pct': round(cash_intraday_pct, 3),
                'gap': round(gap, 3),
                'decision_price': decision_price,
                'exec_price': exec_price,
                'day_close': day_close,
                'return_to_close_pct': round(ret_to_close, 3),
                'is_win': 1 if ret_to_close > 0 else 0,
            })

    df_out = pd.DataFrame(records)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n✅ CSV出力完了: {OUTPUT_FILE}")
    print(f"   行数: {len(df_out)}, 列数: {len(df_out.columns)}")

    print("\n=== チェックポイント × gap閾値 別の成績（参考: 全データでの集計） ===")
    for label in CHECK_POINTS.keys():
        sub_label = df_out[df_out['check_point'] == label]
        print(f"\n--- {label} ---")
        for th in GAP_THRESHOLDS:
            sub = sub_label[sub_label['gap'] <= th]
            if len(sub) == 0:
                print(f"  gap<={th}%: 該当なし")
                continue
            win_rate = sub['is_win'].mean() * 100
            avg_ret = sub['return_to_close_pct'].mean()
            print(f"  gap<={th}%: 件数={len(sub)}, 勝率={win_rate:.1f}%, 平均リターン(大引けまで)={avg_ret:.2f}%")

    print(f"\n総実行時間: {time.time() - start_time:.1f}秒")