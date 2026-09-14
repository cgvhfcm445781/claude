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

# --- 損切り・トレーリング幅の候補（指数向けにタイトな値も含めて比較） ---
STOP_COMBOS = [
    (0.03, 0.02),   # 個別株と同じ値（参考比較用）
    (0.01, 0.005),
    (0.008, 0.004),
    (0.005, 0.003),
    (0.003, 0.002),
]


# 実際に取引する銘柄（1329.T = iシェアーズ・コア 日経225 ETF）
TRADE_TICKER = "1329.T"


def to_jst_naive(df):
    """タイムゾーン情報を日本時間に統一し、その後タイムゾーン情報を外す（比較しやすくするため）"""
    if df.index.tz is not None:
        df.index = df.index.tz_convert('Asia/Tokyo').tz_localize(None)
    return df


if __name__ == "__main__":
    start_time = time.time()

    print(f"{TRADE_TICKER}（日足、判断・執行用）を取得中...")
    df_nk_daily = yf.download(TRADE_TICKER, period="2y", interval="1d", progress=False, repair=True)
    if isinstance(df_nk_daily.columns, pd.MultiIndex):
        df_nk_daily.columns = df_nk_daily.columns.get_level_values(0)
    df_nk_daily = to_jst_naive(df_nk_daily)
    nk_daily_close = df_nk_daily['Close'].sort_index()

    print("日経225先物（日足、地合い判断の参照用）を取得中...")
    df_fut_daily = yf.download("NIY=F", period="2y", interval="1d", progress=False, repair=True)
    if isinstance(df_fut_daily.columns, pd.MultiIndex):
        df_fut_daily.columns = df_fut_daily.columns.get_level_values(0)
    df_fut_daily = to_jst_naive(df_fut_daily)
    fut_daily_close = df_fut_daily['Close'].sort_index()

    print(f"{TRADE_TICKER}（1時間足、実際の売買シミュレーション用）を取得中...")
    # yfinanceの制約：60分足は最大730日分まで遡れる
    df_nk_intraday = yf.download(TRADE_TICKER, period="730d", interval="60m", progress=False, repair=True)
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

    def simulate_exit(intraday_df, date, entry_bar_time, entry_price, loss_cut_rate, trailing_rate):
        """
        エントリー後の1時間足を1本ずつ確認し、個別株バックテストと同じロジックで
        損切り・トレーリングストップ・大引けのいずれかで決済する。
        （個別株版と同様、各足の終値ベースで判定。日をまたいでは保有しない）
        """
        day_bars = intraday_df[intraday_df.index.normalize() == pd.Timestamp(date)]
        after_entry = day_bars[day_bars.index > entry_bar_time]

        m_price = entry_price  # 保有中の最高値（終値ベース）
        for bar_time, row in after_entry.iterrows():
            c = float(row['Close'])
            m_price = max(m_price, c)
            if c <= m_price * (1 - trailing_rate) and c > entry_price:
                return c, bar_time, "トレーリング"
            elif c <= entry_price * (1 - loss_cut_rate):
                return c, bar_time, "損切り"

        # どちらにも該当しなければ、その日の最終バー（大引け）で決済
        if len(after_entry) > 0:
            last_time = after_entry.index[-1]
            last_close = float(after_entry['Close'].iloc[-1])
            return last_close, last_time, "大引け"
        else:
            # エントリー後にバーが無い（最終バーで約定した等）場合はエントリー価格のまま大引け扱い
            return entry_price, entry_bar_time, "大引け"

    trading_days = sorted(set(df_nk_intraday.index.normalize()))
    print(f"対象営業日数: {len(trading_days)}日")

    # --- 1段階目：エントリー候補（日付・チェックポイント・約定価格）を一度だけ抽出 ---
    entries = []
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
            decision_price, decision_bar_time = get_bar_price(df_nk_intraday, date, cfg['decision_time'])
            if pd.isna(decision_price) or decision_bar_time is None:
                continue
            cash_intraday_pct = (decision_price - cash_prev_close) / cash_prev_close * 100
            gap = cash_intraday_pct - futures_overnight_pct

            exec_price, exec_bar_time = get_bar_price_at_or_after(
                df_nk_intraday, date, cfg['exec_time'], field=cfg['exec_field']
            )
            if pd.isna(exec_price) or exec_bar_time is None:
                continue

            entries.append({
                'date': date_str, 'date_ts': date, 'check_point': label,
                'gap': round(gap, 3), 'exec_price': exec_price, 'exec_bar_time': exec_bar_time,
                'day_close': day_close,
            })

    print(f"エントリー候補数: {len(entries)}件（チェックポイント合計）")

    # --- 2段階目：損切り・トレーリングの組み合わせごとに決済シミュレーション ---
    records = []
    for loss_cut_rate, trailing_rate in STOP_COMBOS:
        combo_label = f"損切り{loss_cut_rate*100:.1f}%/トレーリング{trailing_rate*100:.1f}%"
        for e in entries:
            exit_price, exit_bar_time, exit_reason = simulate_exit(
                df_nk_intraday, e['date_ts'], e['exec_bar_time'], e['exec_price'],
                loss_cut_rate, trailing_rate
            )
            ret = (exit_price - e['exec_price']) / e['exec_price'] * 100
            records.append({
                'date': e['date'], 'check_point': e['check_point'], 'stop_combo': combo_label,
                'gap': e['gap'], 'exec_price': e['exec_price'], 'exit_price': exit_price,
                'exit_bar_time': str(exit_bar_time), 'exit_reason': exit_reason,
                'return_pct': round(ret, 3), 'is_win': 1 if ret > 0 else 0,
            })

    df_out = pd.DataFrame(records)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n✅ CSV出力完了: {OUTPUT_FILE}")
    print(f"   行数: {len(df_out)}, 列数: {len(df_out.columns)}")

    print("\n=== チェックポイント × 損切り/トレーリング幅 × gap閾値 別の成績 ===")
    for label in CHECK_POINTS.keys():
        print(f"\n########## {label} ##########")
        for loss_cut_rate, trailing_rate in STOP_COMBOS:
            combo_label = f"損切り{loss_cut_rate*100:.1f}%/トレーリング{trailing_rate*100:.1f}%"
            sub_label = df_out[(df_out['check_point'] == label) & (df_out['stop_combo'] == combo_label)]
            print(f"\n--- {combo_label} ---")
            for th in GAP_THRESHOLDS:
                sub = sub_label[sub_label['gap'] <= th]
                if len(sub) == 0:
                    print(f"  gap<={th}%: 該当なし")
                    continue
                win_rate = sub['is_win'].mean() * 100
                avg_ret = sub['return_pct'].mean()
                reason_counts = sub['exit_reason'].value_counts().to_dict()
                print(f"  gap<={th}%: 件数={len(sub)}, 勝率={win_rate:.1f}%, 平均リターン={avg_ret:.2f}%, 決済理由={reason_counts}")

    print(f"\n総実行時間: {time.time() - start_time:.1f}秒")