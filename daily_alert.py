import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io
import os
import warnings
import time
from datetime import datetime, timedelta

warnings.simplefilter('ignore', FutureWarning)

# --- 設定（backtest_csv.py と同じ条件） ---
MIN_PRICE_LIMIT = 1000
MIN_VALUE_LIMIT = 50000000
STOCH_K_PERIOD = 9
STOCH_D_PERIOD = 3
STOCH_SD_PERIOD = 3
OVERSOLD_LEVEL = 20

NIKKEI_DEV_THRESHOLD = 50.0  #元-5.0
EXCLUDE_TUESDAY = True
MIN_VOLUME_RATIO = 0.5
MAX_ALERT_COUNT = 20  # Discordに表示する候補数の上限（スコア上位から）

# --- スコアリングモデル（ロジスティック回帰、4,534トレードで学習） ---
# 数値特徴量は標準化してから係数を掛ける
SCORE_FEATURE_MEAN = {'rsi_14': 29.917, 'volume_ratio': 1.309, 'sma25_dev': -8.527}
SCORE_FEATURE_SCALE = {'rsi_14': 10.038, 'volume_ratio': 0.612, 'sma25_dev': 5.431}
SCORE_COEF = {'rsi_14': 0.3888, 'volume_ratio': -0.0139, 'sma25_dev': -0.5889}
SCORE_INTERCEPT = 0.7385
# 業種ボーナス（採用件数が少ない業種は 0 = 「その他」扱い）
SECTOR_SCORE = {
    '不動産業': 0.4547, '建設業': 0.5451, '銀行業': 0.5417, 'その他金融業': 0.2995,
    '食料品': 0.3266, '医薬品': 0.2389, '非鉄金属': 0.2156, 'その他製品': 0.1830,
    '輸送用機器': 0.1002, '陸運業': 0.0591, '卸売業': 0.0285, '化学': 0.0164,
    '繊維製品': -0.0118, '情報・通信業': -0.0146, 'ゴム製品': 0.0242,
    '金属製品': -0.1131, '鉄鋼': -0.1100, 'ガラス・土石製品': -0.1807,
    'サービス業': -0.1693, '機械': -0.1923, '電気機器': -0.2078,
    '小売業': -0.2376, '電気・ガス業': -0.3017, '精密機器': -0.4338,
}


def win_probability_score(rsi_14, volume_ratio, sma25_dev, sector):
    """過去データから学習したロジスティック回帰で、勝率の期待値を算出（0〜1）"""
    z = SCORE_INTERCEPT
    z += (rsi_14 - SCORE_FEATURE_MEAN['rsi_14']) / SCORE_FEATURE_SCALE['rsi_14'] * SCORE_COEF['rsi_14']
    z += (volume_ratio - SCORE_FEATURE_MEAN['volume_ratio']) / SCORE_FEATURE_SCALE['volume_ratio'] * SCORE_COEF['volume_ratio']
    z += (sma25_dev - SCORE_FEATURE_MEAN['sma25_dev']) / SCORE_FEATURE_SCALE['sma25_dev'] * SCORE_COEF['sma25_dev']
    z += SECTOR_SCORE.get(sector, 0.0)
    return 1 / (1 + np.exp(-z))


DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
NOTIFY_ON_EMPTY = True  # 該当銘柄が0件の日も通知するか


def get_jpx_stock_list():
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xlsx"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept": "*/*",
    }
    res = requests.get(url, headers=headers, timeout=30)
    if res.status_code != 200:
        raise RuntimeError(f"JPXへのアクセスに失敗しました (status={res.status_code})")
    content = res.content
    is_xlsx = content[:2] == b'PK'
    df = pd.read_excel(io.BytesIO(content), engine='openpyxl' if is_xlsx else 'xlrd')
    target_markets = ['プライム（内国株式）']
    df = df[df['市場・商品区分'].isin(target_markets)].copy()
    df['Ticker'] = df['コード'].astype(str) + ".T"
    return df[['Ticker', '銘柄名', '33業種区分']]


def calculate_stochastics(df, k_period=9, d_period=3, sd_period=3):
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    fast_k = 100 * ((df['Close'] - low_min) / (high_max - low_min + 1e-10))
    stoch_d = fast_k.rolling(window=d_period).mean()
    slow_d = stoch_d.rolling(window=sd_period).mean()
    return stoch_d, slow_d


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def next_trading_day_weekday(today):
    """簡易版：土日を考慮した翌営業日の曜日（祝日は考慮していない点に注意）"""
    nxt = today + timedelta(days=1)
    while nxt.weekday() >= 5:  # 5=土, 6=日
        nxt += timedelta(days=1)
    return nxt.weekday()


def send_discord_message(content):
    if not DISCORD_WEBHOOK_URL:
        print("⚠ DISCORD_WEBHOOK_URL が設定されていません。通知をスキップします。")
        return
    # Discordは1メッセージ2000文字制限があるため分割して送信
    max_len = 1900
    chunks = [content[i:i + max_len] for i in range(0, len(content), max_len)] or [content]
    for chunk in chunks:
        res = requests.post(DISCORD_WEBHOOK_URL, json={"content": chunk}, timeout=15)
        if res.status_code >= 300:
            print(f"⚠ Discord通知に失敗しました: status={res.status_code}, body={res.text[:300]}")
        time.sleep(0.5)


if __name__ == "__main__":
    start_time = time.time()
    print("日経平均データを取得中...")
    df_nk = yf.download("^N225", period="3mo", interval="1d", progress=False, repair=True)
    if df_nk.empty:
        raise RuntimeError("日経平均データの取得に失敗しました（空のデータが返ってきました）。"
                            "依存パッケージ（特にscikit-learn）が正しくインストールされているか確認してください。")
    if isinstance(df_nk.columns, pd.MultiIndex):
        df_nk.columns = df_nk.columns.get_level_values(0)
    nk_sma25 = df_nk['Close'].rolling(window=25).mean()
    df_nk['nk_sma25_dev'] = (df_nk['Close'] - nk_sma25) / nk_sma25 * 100

    latest_nk_date = df_nk.index[-1]
    latest_nk_dev = float(df_nk['nk_sma25_dev'].iloc[-1])
    print(f"最新営業日: {latest_nk_date.date()}, 日経25日線乖離: {latest_nk_dev:.2f}%")

    next_wd = next_trading_day_weekday(latest_nk_date.to_pydatetime())
    weekday_names = ["月曜", "火曜", "水曜", "木曜", "金曜", "土曜", "日曜"]
    print(f"翌営業日の曜日（推定）: {weekday_names[next_wd]}")

    # --- 地合いフィルタ：日経条件を満たさなければ、個別銘柄を調べるまでもなく終了 ---
    skip_reason = None
    if pd.isna(latest_nk_dev) or latest_nk_dev > NIKKEI_DEV_THRESHOLD:
        skip_reason = f"日経25日線乖離が条件({NIKKEI_DEV_THRESHOLD}%以下)を満たしていません（現在{latest_nk_dev:.2f}%）"
    elif EXCLUDE_TUESDAY and next_wd == 1:
        skip_reason = "翌営業日が火曜のため、条件から除外されます"

    if skip_reason:
        print(f"本日は地合いフィルタで対象外: {skip_reason}")
        if NOTIFY_ON_EMPTY:
            send_discord_message(f"📊 本日({latest_nk_date.date()})の株スクリーニング結果\n\n{skip_reason}\n該当銘柄なし。")
        print(f"総実行時間: {time.time() - start_time:.1f}秒")
        raise SystemExit(0)

    print("地合い条件クリア。個別銘柄をスクリーニング中...")
    stock_list = get_jpx_stock_list()
    hits = []

    for idx, row in stock_list.iterrows():
        ticker, name, sector = row['Ticker'], row['銘柄名'], row['33業種区分']
        try:
            df_d = yf.download(ticker, period="3mo", interval="1d", progress=False, repair=True)
            if df_d.empty or len(df_d) < 30:
                continue
            if isinstance(df_d.columns, pd.MultiIndex):
                df_d.columns = df_d.columns.get_level_values(0)

            price_cols = ['Open', 'High', 'Low', 'Close']
            df_d[price_cols] = df_d[price_cols].where(df_d[price_cols] > 0)
            df_d = df_d.dropna(subset=price_cols)
            if len(df_d) < 30:
                continue

            close = float(df_d['Close'].iloc[-1])
            if close < MIN_PRICE_LIMIT:
                continue

            daily_value = df_d['Close'] * df_d['Volume']
            ma_value_10 = daily_value.rolling(window=10).mean()
            if float(ma_value_10.iloc[-1]) < MIN_VALUE_LIMIT:
                continue

            stoch_d, slow_d = calculate_stochastics(df_d, STOCH_K_PERIOD, STOCH_D_PERIOD, STOCH_SD_PERIOD)
            rsi_14 = calculate_rsi(df_d['Close'], period=14)
            vol_ma20 = df_d['Volume'].rolling(window=20).mean()
            vol_ratio = df_d['Volume'] / (vol_ma20 + 1e-10)
            sma25 = df_d['Close'].rolling(window=25).mean()
            sma25_dev = (df_d['Close'] - sma25) / sma25 * 100

            d_0, sd_0 = stoch_d.iloc[-1], slow_d.iloc[-1]
            d_1, sd_1 = stoch_d.iloc[-2], slow_d.iloc[-2]
            if pd.isna(d_0) or pd.isna(sd_0) or pd.isna(d_1) or pd.isna(sd_1):
                continue

            touch_oversold = (d_1 <= OVERSOLD_LEVEL) or (sd_1 <= OVERSOLD_LEVEL) or (d_0 <= OVERSOLD_LEVEL)
            golden_cross = (d_1 <= sd_1) and (d_0 > sd_0)
            if not (touch_oversold and golden_cross):
                continue

            vr_today = float(vol_ratio.iloc[-1])
            if vr_today < MIN_VOLUME_RATIO:
                continue

            rsi_today = float(rsi_14.iloc[-1])
            sma_dev_today = float(sma25_dev.iloc[-1])
            win_prob = win_probability_score(rsi_today, vr_today, sma_dev_today, sector)

            hits.append({
                'ticker': ticker,
                'name': name,
                'sector': sector,
                'close': close,
                'rsi_14': round(rsi_today, 1),
                'volume_ratio': round(vr_today, 2),
                'sma25_dev': round(sma_dev_today, 2),
                'win_prob': win_prob,
            })
        except Exception:
            continue

        if (idx + 1) % 500 == 0:
            print(f"{idx + 1} 銘柄チェック完了...")

    print(f"\n該当銘柄数: {len(hits)}")

    if hits:
        hits.sort(key=lambda h: h['win_prob'], reverse=True)  # 勝率スコアが高い順に並べ替え
        total_hits = len(hits)
        display_hits = hits[:MAX_ALERT_COUNT]
        lines = [
            f"📈 **買いシグナル検出**（{latest_nk_date.date()}終値時点、翌営業日エントリー候補）",
            f"日経25日線乖離: {latest_nk_dev:.2f}%　※勝率スコア上位{len(display_hits)}件を表示"
            + (f"（全{total_hits}件中）" if total_hits > MAX_ALERT_COUNT else "") + "\n",
        ]
        for h in display_hits:
            lines.append(
                f"・`{h['ticker']}` {h['name']}（{h['sector']}） 推定勝率{h['win_prob']*100:.0f}%\n"
                f"　終値{h['close']:.0f}円 / RSI{h['rsi_14']} / 出来高比{h['volume_ratio']} / 25日線乖離{h['sma25_dev']}%"
            )
        message = "\n".join(lines)
        print(message)
        send_discord_message(message)
    else:
        print("本日は該当銘柄なし。")
        if NOTIFY_ON_EMPTY:
            send_discord_message(f"📊 本日({latest_nk_date.date()})の株スクリーニング結果\n\n該当銘柄なし。")

    print(f"\n総実行時間: {time.time() - start_time:.1f}秒")