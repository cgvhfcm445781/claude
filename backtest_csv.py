import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import os
import time

warnings.simplefilter('ignore', FutureWarning)

OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "no_bad_news_backtest.csv")

TRADE_TICKER = "1329.T"  # iシェアーズ・コア 日経225 ETF

# 米国11セクターETF（daily_alert.py / backtest_csv.py と同じ）
US_SECTOR_ETFS = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']

# 「これ以上下げているセクターが1つでもあれば"地合いが悪い日"」という閾値の候補
BAD_NEWS_THRESHOLDS = [-1.0, -1.3, -1.5, -2.0]


def to_jst_naive(df):
    if df.index.tz is not None:
        df.index = df.index.tz_convert('Asia/Tokyo').tz_localize(None)
    return df


def get_asof(series, dt):
    if series is None or len(series) == 0:
        return np.nan
    pos = series.index.searchsorted(dt, side='right') - 1
    if pos < 0:
        return np.nan
    return series.iloc[pos]


if __name__ == "__main__":
    start_time = time.time()

    print(f"{TRADE_TICKER}（日足）を取得中...")
    df_trade = yf.download(TRADE_TICKER, period="10y", interval="1d", progress=False, repair=True)
    if isinstance(df_trade.columns, pd.MultiIndex):
        df_trade.columns = df_trade.columns.get_level_values(0)
    df_trade = to_jst_naive(df_trade)

    print("米国11セクターETFデータを取得中...")
    us_chg_series = {}
    for etf in US_SECTOR_ETFS:
        df_etf = yf.download(etf, period="10y", interval="1d", progress=False, repair=True)
        if isinstance(df_etf.columns, pd.MultiIndex):
            df_etf.columns = df_etf.columns.get_level_values(0)
        df_etf = to_jst_naive(df_etf)
        chg = df_etf['Close'].pct_change() * 100
        us_chg_series[etf] = chg.sort_index()
        print(f"  {etf}: {len(df_etf)}件")

    records = []
    for date in df_trade.index:
        entry_dt = date
        open_p = float(df_trade.loc[date, 'Open'])
        close_p = float(df_trade.loc[date, 'Close'])
        if pd.isna(open_p) or pd.isna(close_p) or open_p == 0:
            continue
        day_return_pct = (close_p - open_p) / open_p * 100

        # 各セクターETFの「前日比」を取得し、その日の最悪値を採用
        sector_chgs = {}
        for etf in US_SECTOR_ETFS:
            v = get_asof(us_chg_series[etf], entry_dt)
            if not pd.isna(v):
                sector_chgs[etf] = float(v)
        if not sector_chgs:
            continue
        worst_etf = min(sector_chgs, key=sector_chgs.get)
        worst_chg = sector_chgs[worst_etf]

        records.append({
            'date': date.strftime('%Y-%m-%d'),
            'open': open_p, 'close': close_p,
            'day_return_pct': round(day_return_pct, 3),
            'is_win': 1 if day_return_pct > 0 else 0,
            'worst_us_sector': worst_etf,
            'worst_us_sector_chg': round(worst_chg, 3),
        })

    df_out = pd.DataFrame(records)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n✅ CSV出力完了: {OUTPUT_FILE}")
    print(f"   行数: {len(df_out)}, 列数: {len(df_out.columns)}")

    print("\n=== 全体（比較用ベースライン） ===")
    print(f"件数={len(df_out)}, 勝率={df_out['is_win'].mean()*100:.1f}%, "
          f"平均リターン={df_out['day_return_pct'].mean():.2f}%")

    print("\n=== 「地合いが悪くない日」（米国11セクターすべてが閾値より下げていない） vs それ以外 ===")
    for th in BAD_NEWS_THRESHOLDS:
        good = df_out[df_out['worst_us_sector_chg'] > th]
        bad = df_out[df_out['worst_us_sector_chg'] <= th]
        print(f"\n--- 閾値 {th}% ---")
        if len(good) > 0:
            print(f"  地合い良好（悪材料なし）: 件数={len(good)}, 勝率={good['is_win'].mean()*100:.1f}%, "
                  f"平均リターン={good['day_return_pct'].mean():.2f}%")
        if len(bad) > 0:
            print(f"  地合い悪い（どこかのセクターが下落）: 件数={len(bad)}, 勝率={bad['is_win'].mean()*100:.1f}%, "
                  f"平均リターン={bad['day_return_pct'].mean():.2f}%")

    print(f"\n総実行時間: {time.time() - start_time:.1f}秒")