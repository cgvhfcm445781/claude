import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io
import os
import warnings
import time
import joblib
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
EXCLUDE_TUESDAY = True
MIN_VOLUME_RATIO = 0.5
US_SECTOR_CHG_THRESHOLD = -1.3  # 対応する米国セクターETFの前日変化率がこれより大きければ見送り（ハードな足切り条件）
NAIVE_ETF_CORR_THRESHOLD = 0.08  # 業種マッピング先ETFとの実際の相関がこれ未満なら見送り
CORR_MIN_OVERLAP = 100  # 相関計算に必要な最低データ件数
MAX_ALERT_COUNT = 20  # 足切り後の候補がこれを超えたら、AIスコア上位からこの件数まで絞る
TOP_TIER_WIN_PROB_THRESHOLD = 0.6  # メイン条件を満たした銘柄のうち、AIスコアがこれ以上なら「厳選」として別枠で強調表示

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

# --- ランダムフォレストのスコアリングモデル（8,305トレードで学習・2026年9月版） ---
# rf_model.pkl / rf_features.pkl をリポジトリ直下に置いておく必要がある
RF_MODEL_PATH = "rf_model.pkl"
RF_FEATURES_PATH = "rf_features.pkl"

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
    target_markets = ['プライム（内国株式）', 'スタンダード（内国株式）']  # グロースは検証の結果、成績が明確に劣るため除外
    df = df[df['市場・商品区分'].isin(target_markets)].copy()
    market_label_map = {
        'プライム（内国株式）': 'プライム',
        'スタンダード（内国株式）': 'スタンダード',
    }
    df['Market'] = df['市場・商品区分'].map(market_label_map)
    df['Ticker'] = df['コード'].astype(str) + ".T"
    return df[['Ticker', '銘柄名', '33業種区分', 'Market']]


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


def calculate_rci(series, period=9):
    """RCI（順位相関指数）：価格の順位と時間の順位の相関を-100〜100で表す"""
    def _rci(window):
        price_rank = pd.Series(window).rank(ascending=False)
        time_rank = np.arange(1, len(window) + 1)
        d_sq_sum = ((price_rank.values - time_rank[::-1]) ** 2).sum()
        n = len(window)
        return (1 - 6 * d_sq_sum / (n * (n ** 2 - 1))) * 100
    return series.rolling(window=period).apply(_rci, raw=True)


def calculate_bollinger(series, period=20, num_std=2):
    """ボリンジャーバンド：%b（バンド内でのポジション、0=下限、1=上限）を返す"""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return (series - lower) / ((upper - lower) + 1e-10)


def calculate_macd(series, fast=12, slow=26, signal=9):
    """MACD：MACDライン、シグナルライン、ヒストグラムを返す"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def calculate_dmi_adx(df, period=14):
    """DMI/ADX：+DI, -DI, ADX（トレンドの強さ）を返す"""
    high, low, close = df['High'], df['Low'], df['Close']
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([
        high - low, (high - close.shift()).abs(), (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / (atr + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-10)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return plus_di, minus_di, adx


def calculate_psychological_line(series, period=12):
    """サイコロジカルライン：直近period日のうち上昇した日の割合(%)"""
    up_days = (series.diff() > 0).astype(int)
    return up_days.rolling(window=period).mean() * 100


def calculate_ichimoku(df):
    """一目均衡表：雲との位置関係・転換線基準線の差・雲の色・遅行スパンのシグナルを返す"""
    high, low, close = df['High'], df['Low'], df['Close']
    tenkan = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
    kijun = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
    kumo_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    kumo_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)

    cloud_position = pd.Series(np.where(
        close > kumo_top, (close - kumo_top) / kumo_top * 100,
        np.where(close < kumo_bottom, (close - kumo_bottom) / kumo_bottom * 100, 0.0)
    ), index=df.index)
    tenkan_kijun_diff = (tenkan - kijun) / kijun * 100
    cloud_color = np.where(senkou_a > senkou_b, 1, -1)
    chikou_signal = np.where(close > close.shift(26), 1, -1)
    return cloud_position, tenkan_kijun_diff, pd.Series(cloud_color, index=df.index), pd.Series(chikou_signal, index=df.index)


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


def calc_etf_corr(df_d, us_series, min_overlap=100):
    """この銘柄の日次リターンと、割り当てられた米国セクターETFとの相関係数を計算（先読みなし）"""
    if us_series is None:
        return None
    stock_ret = df_d['Close'].pct_change() * 100
    dates = df_d.index
    us_vals = us_series.reindex(us_series.index.union(dates)).ffill().reindex(dates)
    paired = pd.concat([stock_ret, us_vals], axis=1).dropna()
    if len(paired) < min_overlap:
        return None
    corr = paired.iloc[:, 0].corr(paired.iloc[:, 1])
    return None if pd.isna(corr) else float(corr)


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

    print("ランダムフォレストモデルを読み込み中...")
    if not os.path.exists(RF_MODEL_PATH) or not os.path.exists(RF_FEATURES_PATH):
        raise RuntimeError(
            f"{RF_MODEL_PATH} / {RF_FEATURES_PATH} が見つかりません。"
            f"リポジトリ直下に配置されているか確認してください。"
        )
    rf_model = joblib.load(RF_MODEL_PATH)
    rf_features = joblib.load(RF_FEATURES_PATH)
    print(f"  読み込み完了。使用する特徴量({len(rf_features)}個): {rf_features}")

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
        skip_reason = "翌営業日が火曜のため、条件から除外されます"

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
        ticker, name, sector, market = row['Ticker'], row['銘柄名'], row['33業種区分'], row['Market']
        try:
            # 一目均衡表の雲は52日+26日先行の計算が必要なため、少し長めに取得
            df_d = yf.download(ticker, period="6mo", interval="1d", progress=False, repair=True)
            if df_d.empty or len(df_d) < 80:
                continue
            if isinstance(df_d.columns, pd.MultiIndex):
                df_d.columns = df_d.columns.get_level_values(0)

            price_cols = ['Open', 'High', 'Low', 'Close']
            df_d[price_cols] = df_d[price_cols].where(df_d[price_cols] > 0)
            df_d = df_d.dropna(subset=price_cols)
            if len(df_d) < 80:
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

            # --- 米国同業種セクターETFの前日変化率：ここでハードに足切りする ---
            us_etf = SECTOR_ETF_MAP.get(sector)
            us_series = us_sector_chg_series.get(us_etf) if us_etf else None
            us_sector_chg = get_asof(us_series, df_d.index[-1]) if us_series is not None else np.nan
            if pd.isna(us_sector_chg) or us_sector_chg > US_SECTOR_CHG_THRESHOLD:
                continue

            # --- 業種マッピング先ETFとの相関が弱い銘柄は見送り（シグナルが意味をなさないため） ---
            naive_etf_corr = calc_etf_corr(df_d, us_series, min_overlap=CORR_MIN_OVERLAP)
            if naive_etf_corr is None or naive_etf_corr < NAIVE_ETF_CORR_THRESHOLD:
                continue

            # --- ランダムフォレスト用の特徴量を計算（足切り後の母集団だけで学習したモデルを使う） ---
            i = len(df_d) - 1
            open_today = float(df_d['Open'].iloc[i])
            close_today = close
            close_prev = float(df_d['Close'].iloc[i - 1])
            high_today = float(df_d['High'].iloc[i])
            low_today = float(df_d['Low'].iloc[i])
            day_range = max(high_today - low_today, 1e-6)
            body_top = max(open_today, close_today)
            body_bottom = min(open_today, close_today)
            lower_shadow_ratio = (body_bottom - low_today) / day_range
            upper_shadow_ratio = (high_today - body_top) / day_range
            body_ratio = (body_top - body_bottom) / day_range

            lookback_n = 5
            if i - lookback_n < 0:
                continue
            hi5 = df_d['High'].iloc[i - lookback_n + 1:i + 1].to_numpy(dtype=float)
            lo5 = df_d['Low'].iloc[i - lookback_n + 1:i + 1].to_numpy(dtype=float)
            op5 = df_d['Open'].iloc[i - lookback_n + 1:i + 1].to_numpy(dtype=float)
            cl5 = df_d['Close'].iloc[i - lookback_n + 1:i + 1].to_numpy(dtype=float)
            rng5 = np.maximum(hi5 - lo5, 1e-6)
            body_top5 = np.maximum(op5, cl5)
            body_bottom5 = np.minimum(op5, cl5)
            up_days_last5 = int((cl5 > op5).sum())
            close_prev5 = float(df_d['Close'].iloc[i - lookback_n])
            return_5d = (close_today - close_prev5) / close_prev5 * 100 if close_prev5 != 0 else np.nan
            avg_body_ratio_last5 = float(((body_top5 - body_bottom5) / rng5).mean())
            avg_lower_shadow_last5 = float(((body_bottom5 - lo5) / rng5).mean())
            consecutive_down_days = 0
            for j in range(i, max(i - lookback_n, -1), -1):
                if float(df_d['Close'].iloc[j]) < float(df_d['Close'].iloc[j - 1]):
                    consecutive_down_days += 1
                else:
                    break

            bb_percent_b = calculate_bollinger(df_d['Close'], period=20, num_std=2)
            macd_line, macd_signal, macd_hist = calculate_macd(df_d['Close'])
            rci_9 = calculate_rci(df_d['Close'], period=9)
            plus_di, minus_di, adx = calculate_dmi_adx(df_d, period=14)
            psycho_12 = calculate_psychological_line(df_d['Close'], period=12)
            volatility_20 = df_d['Close'].pct_change().rolling(window=20).std() * 100
            cloud_position, tenkan_kijun_diff, cloud_color, chikou_signal = calculate_ichimoku(df_d)

            feature_values = {
                'rsi_14': float(rsi_14.iloc[i]),
                'volume_ratio': vr_today,
                'sma25_dev': float(sma25_dev.iloc[i]),
                'bb_percent_b': float(bb_percent_b.iloc[i]),
                'macd_hist': float(macd_hist.iloc[i]),
                'rci_9': float(rci_9.iloc[i]),
                'plus_di': float(plus_di.iloc[i]),
                'minus_di': float(minus_di.iloc[i]),
                'adx': float(adx.iloc[i]),
                'psycho_12': float(psycho_12.iloc[i]),
                'volatility_20': float(volatility_20.iloc[i]),
                'ichimoku_cloud_position': float(cloud_position.iloc[i]),
                'ichimoku_tenkan_kijun_diff': float(tenkan_kijun_diff.iloc[i]),
                'ichimoku_cloud_color': float(cloud_color.iloc[i]),
                'ichimoku_chikou_signal': float(chikou_signal.iloc[i]),
                'up_days_last5': up_days_last5,
                'consecutive_down_days': consecutive_down_days,
                'return_5d': return_5d,
                'avg_body_ratio_last5': avg_body_ratio_last5,
                'avg_lower_shadow_last5': avg_lower_shadow_last5,
            }

            # 欠損があるとモデルに通せないのでスキップ（データ不足の銘柄）
            if any(pd.isna(v) for v in feature_values.values()):
                continue

            X_row = pd.DataFrame([[feature_values[f] for f in rf_features]], columns=rf_features)
            win_prob = float(rf_model.predict_proba(X_row)[0, 1])

            hits.append({
                'ticker': ticker,
                'name': name,
                'sector': sector,
                'market': market,
                'close': close,
                'rsi_14': round(feature_values['rsi_14'], 1),
                'volume_ratio': round(vr_today, 2),
                'sma25_dev': round(feature_values['sma25_dev'], 2),
                'us_sector_chg': round(float(us_sector_chg), 2) if not pd.isna(us_sector_chg) else None,
                'win_prob': win_prob,
            })
        except Exception:
            continue

        if (idx + 1) % 500 == 0:
            print(f"{idx + 1} 銘柄チェック完了...")

    print(f"\n該当銘柄数: {len(hits)}")

    if hits:
        hits.sort(key=lambda h: h['win_prob'], reverse=True)
        total_hits = len(hits)
        display_hits = hits[:MAX_ALERT_COUNT]
        top_tier = [h for h in display_hits if h['win_prob'] >= TOP_TIER_WIN_PROB_THRESHOLD]

        def format_hit(h):
            us_txt = f"{h['us_sector_chg']}%" if h['us_sector_chg'] is not None else "N/A"
            return (
                f"・`{h['ticker']}` {h['name']}（{h['sector']}・{h['market']}） 推定勝率{h['win_prob']*100:.0f}%\n"
                f"　終値{h['close']:.0f}円 / RSI{h['rsi_14']} / 出来高比{h['volume_ratio']} / "
                f"25日線乖離{h['sma25_dev']}% / 米国セクター前日比{us_txt}"
            )

        lines = [f"📈 **買いシグナル検出**（{latest_nk_date.date()}終値時点、翌営業日エントリー候補）\n"]

        if top_tier:
            lines.append(f"■ 🌟厳選(AI推定勝率{TOP_TIER_WIN_PROB_THRESHOLD*100:.0f}%以上): {len(top_tier)}件")
            lines.extend(format_hit(h) for h in top_tier)
            lines.append("")

        lines.append(
            f"■ 全候補(ストキャス+GC+出来高+米国セクター{US_SECTOR_CHG_THRESHOLD}%以下)を満たした{total_hits}件"
            + (f"のうちAIスコア上位{len(display_hits)}件" if total_hits > MAX_ALERT_COUNT else "を全て表示")
        )
        lines.extend(format_hit(h) for h in display_hits)

        message = "\n".join(lines)
        print(message)
        send_discord_message(message)
    else:
        print("本日は該当銘柄なし。")
        if NOTIFY_ON_EMPTY:
            send_discord_message(f"📊 本日({latest_nk_date.date()})の株スクリーニング結果\n\n該当銘柄なし。")

    print(f"\n総実行時間: {time.time() - start_time:.1f}秒")