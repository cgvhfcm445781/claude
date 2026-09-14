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

NIKKEI_DEV_THRESHOLD = 50.0  # 頻度を優先し実質無効化（backtest_csv.pyと同じ方針）
EXCLUDE_TUESDAY = False
MIN_VOLUME_RATIO = 0.5
US_SECTOR_CHG_THRESHOLD = -1.3  # 対応する米国セクターETFの前日変化率がこれより大きければ見送り
MAX_ALERT_COUNT = 20  # Discordに表示する候補数の上限（スコア上位から）

# --- 日本の33業種区分 → 米国セクターETF（SPDR Select Sector）の近似対応表 ---
SECTOR_ETF_MAP = {
    '水産・農林業': 'XLP', '鉱業': 'XLE', '建設業': 'XLI', '食料品': 'XLP',
    '繊維製品': 'XLY', 'パルプ・紙': 'XLB', '化学': 'XLB', '医薬品': 'XLV',
    '石油・石炭製品': 'XLE', 'ゴム製品': 'XLY', 'ガラス・土石製品': 'XLB',
    '鉄鋼': 'XLB', '非鉄金属': 'XLB', '金属製品': 'XLB', '機械': 'XLI',
    '電気機器': 'XLK', '輸送用機器': 'XLY', '精密機器': 'XLK', 'その他製品': 'XLY',
    '電気・ガス業': 'XLU', '陸運業': 'XLI', '海運業': 'XLI', '空運業': 'XLI',
    '倉庫・運輸関連業': 'XLI', '情報・通信業': 'XLC', '卸売業': 'XLI',
    '小売業': 'XLY', '銀行業': 'XLF', '証券、商品先物取引業': 'XLF',
    '保険業': 'XLF', 'その他金融業': 'XLF', '不動産業': 'XLRE', 'サービス業': 'XLY',
}

# --- スコアリングモデル（ロジスティック回帰、7,227トレードで学習・2026年9月版） ---
# 数値特徴量は標準化してから係数を掛ける
SCORE_FEATURE_MEAN = {'rsi_14': 36.569, 'volume_ratio': 1.154, 'sma25_dev': -4.767}
SCORE_FEATURE_SCALE = {'rsi_14': 12.168, 'volume_ratio': 0.619, 'sma25_dev': 5.018}
SCORE_COEF = {'rsi_14': 0.1928, 'volume_ratio': 0.1244, 'sma25_dev': -0.2837}
SCORE_INTERCEPT = 0.1940
# 業種ボーナス（採用件数が少ない業種は 0 = 「その他」扱い）
SECTOR_SCORE = {
    '不動産業': 0.5022, '銀行業': 0.2851, '卸売業': 0.2231, 'ゴム製品': 0.3287,
    'ガラス・土石製品': 0.1999, '非鉄金属': 0.1869, '建設業': 0.1160, '電気機器': 0.1323,
    '化学': 0.0632, '機械': 0.0311, '繊維製品': 0.0143, '鉄鋼': -0.0224,
    '医薬品': -0.0198, 'その他製品': -0.0322, '輸送用機器': -0.0822, '小売業': -0.0844,
    'その他金融業': -0.0844, '金属製品': -0.0931, '食料品': -0.1184, '陸運業': -0.2020,
    'サービス業': -0.2369, '精密機器': -0.3553, '情報・通信業': -0.4581, '電気・ガス業': -0.7942,
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


def get_asof(series, dt):
    """指定日時点で確定している直近の値を取得（未来の値は使わない、日付のズレにも対応）"""
    if series is None or len(series) == 0:
        return np.nan
    pos = series.index.searchsorted(dt, side='right') - 1
    if pos < 0:
        return np.nan
    return series.iloc[pos]


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

    # --- 地合いフィルタ：日経条件（現在は実質無効）と曜日条件だけ、個別銘柄を調べる前にチェック ---
    skip_reason = None
    if pd.isna(latest_nk_dev) or latest_nk_dev > NIKKEI_DEV_THRESHOLD:
        skip_reason = f"日経25日線乖離が条件({NIKKEI_DEV_THRESHOLD}%以下)を満たしていません（現在{latest_nk_dev:.2f}%）"
    elif EXCLUDE_TUESDAY and next_wd == 1:
        skip_reason = "火曜のため条件から除外されます"

    if skip_reason:
        print(f"本日は地合いフィルタで対象外: {skip_reason}")
        if NOTIFY_ON_EMPTY:
            send_discord_message(f"📊 本日({latest_nk_date.date()})の株スクリーニング結果\n\n{skip_reason}\n該当銘柄なし。")
        print(f"総実行時間: {time.time() - start_time:.1f}秒")
        raise SystemExit(0)

    print("米国セクターETFデータを取得中...")
    us_sector_chg_series = {}
    for etf in sorted(set(SECTOR_ETF_MAP.values())):
        try:
            df_etf = yf.download(etf, period="3mo", interval="1d", progress=False, repair=True)
            if isinstance(df_etf.columns, pd.MultiIndex):
                df_etf.columns = df_etf.columns.get_level_values(0)
            if df_etf.empty:
                raise RuntimeError("空データ")
            chg = df_etf['Close'].pct_change() * 100
            us_sector_chg_series[etf] = chg.sort_index()
        except Exception as e:
            print(f"  ⚠ {etf} の取得に失敗（この業種は米国セクター条件を満たせずスキップされます）: {e}")
            us_sector_chg_series[etf] = None
    print(f"  取得完了: {sum(1 for v in us_sector_chg_series.values() if v is not None)}/{len(us_sector_chg_series)}銘柄")

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

            # --- 米国同業種セクターETFの前日変化率をチェック ---
            us_etf = SECTOR_ETF_MAP.get(sector)
            us_series = us_sector_chg_series.get(us_etf) if us_etf else None
            us_sector_chg = get_asof(us_series, df_d.index[-1]) if us_series is not None else np.nan
            if pd.isna(us_sector_chg) or us_sector_chg > US_SECTOR_CHG_THRESHOLD:
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
                'us_sector_chg': round(float(us_sector_chg), 2),
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
            f"※米国同業種ETF前日比<={US_SECTOR_CHG_THRESHOLD}%を満たす銘柄のみ、勝率スコア上位{len(display_hits)}件を表示"
            + (f"（全{total_hits}件中）" if total_hits > MAX_ALERT_COUNT else "") + "\n",
        ]
        for h in display_hits:
            lines.append(
                f"・`{h['ticker']}` {h['name']}（{h['sector']}） 推定勝率{h['win_prob']*100:.0f}%\n"
                f"　終値{h['close']:.0f}円 / RSI{h['rsi_14']} / 出来高比{h['volume_ratio']} / "
                f"25日線乖離{h['sma25_dev']}% / 米国セクター前日比{h['us_sector_chg']}%"
            )
        message = "\n".join(lines)
        print(message)
        send_discord_message(message)
    else:
        print("本日は該当銘柄なし。")
        if NOTIFY_ON_EMPTY:
            send_discord_message(f"📊 本日({latest_nk_date.date()})の株スクリーニング結果\n\n該当銘柄なし。")

    print(f"\n総実行時間: {time.time() - start_time:.1f}秒")