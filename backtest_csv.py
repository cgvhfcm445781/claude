import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io
import warnings
import time
import os

# 警告の抑制
warnings.simplefilter('ignore', FutureWarning)

# --- 設定 ---
INITIAL_CASH = 400000 
LOSS_CUT_RATE = 0.03       # 3%損切り
TRAILING_STOP_RATE = 0.02  # 2%トレーリングストップ
MIN_PRICE_LIMIT = 1000     # 最小株価制限
MIN_VALUE_LIMIT = 50000000 # 売買代金の下限（5,000万円）

# ストキャスティクス パラメータ
STOCH_K_PERIOD = 9
STOCH_D_PERIOD = 3
STOCH_SD_PERIOD = 3
OVERSOLD_LEVEL = 20

# --- 追加フィルタ（過去データ分析から導出） ---
USE_ENTRY_FILTERS = True     # Falseにすれば元の条件のみに戻せる
NIKKEI_DEV_THRESHOLD = 50.0  # 日経平均が25日線からこの%以上下に乖離している日のみエントリー
EXCLUDE_TUESDAY = True       # 火曜エントリーを除外
MIN_VOLUME_RATIO = 0.5       # 出来高比率がこれ未満（閑散銘柄）は除外
FUTURES_BASIS_THRESHOLD = 0.0  # 先物ベーシス（先物-現物）がこれ以上ならエントリー見送り（マイナス＝先物が現物より弱気の時だけ採用）

# --- 既知の暴落期間の除外リスト ---
# 「日経-5%乖離で買う」は通常の調整局面では機能するが、コロナショックのような
# 急激かつ継続的な暴落では逆効果（下げ止まらず損切り連発）だったため、
# 過去に判明している大暴落期間はエントリー対象から除外する。
# 似たような急落が将来起きた場合、ここに期間を手動で追加することを想定。
CRISIS_PERIODS = [
    ("2020-02-20", "2020-05-31"),  # コロナショック
]

# --- ウォークフォワード検証の区切り年 ---
WALKFORWARD_SPLIT_YEAR = 2025  # この年より前を「学習期間」、以降を「検証期間」として比較

# --- CSV出力設定 ---
OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "trade_log.csv")


def get_jpx_stock_list():
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xlsx"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept": "*/*",
    }
    res = requests.get(url, headers=headers, timeout=30)

    # 診断用ログ：常に出力しておく（失敗時の原因特定のため）
    print(f"[JPX] status_code={res.status_code}, content-type={res.headers.get('Content-Type')}, "
          f"content-length={len(res.content)} bytes")

    if res.status_code != 200:
        raise RuntimeError(
            f"JPXへのアクセスに失敗しました (status={res.status_code})。"
            f"アクセス元IPがブロックされている可能性があります。"
            f"レスポンス先頭: {res.content[:300]!r}"
        )

    content = res.content
    # ファイル形式を先頭バイトで判定（xlsxはZIP形式なのでPKで始まる、xlsは別のバイナリシグネチャ）
    is_xlsx = content[:2] == b'PK'

    try:
        if is_xlsx:
            df = pd.read_excel(io.BytesIO(content), engine='openpyxl')
        else:
            df = pd.read_excel(io.BytesIO(content), engine='xlrd')
    except Exception as e:
        raise RuntimeError(
            f"Excelファイルの解析に失敗しました: {e}\n"
            f"先頭バイト: {content[:20]!r}\n"
            f"おそらくJPXからExcelではなくエラーページ(HTML等)が返ってきています。"
        )

    target_markets = ['プライム（内国株式）']
    df = df[df['市場・商品区分'].isin(target_markets)].copy()
    df['Market'] = 'プライム'
    df['Ticker'] = df['コード'].astype(str) + ".T"
    return df[['Ticker', '銘柄名', '33業種区分', 'Market']]


def calculate_stochastics(df, k_period=9, d_period=3, sd_period=3):
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    fast_k = 100 * ((df['Close'] - low_min) / (high_max - low_min + 1e-10))
    stoch_d = fast_k.rolling(window=d_period).mean()
    slow_d = stoch_d.rolling(window=sd_period).mean()
    return stoch_d, slow_d


def is_crisis_period(dt):
    """指定した日付が既知の暴落期間に含まれるかを判定"""
    for start, end in CRISIS_PERIODS:
        if pd.Timestamp(start) <= dt <= pd.Timestamp(end):
            return True
    return False


def compute_msb_ob_series(df_d, zigzag_len=9, fib_factor=0.33):
    """
    TradingViewインジケータ「Market Structure Break & Order Block」（EmreKb氏, MPL-2.0）の
    ロジックをできるだけ忠実にPythonへ移植したもの。このバックテストはロングのみのため、
    強気側（Bullish MSB / Bullish OB）だけを扱う。df_d全体に対して1回だけ計算し、
    日ごとの系列（pandas Series）として返す。

    - to_up/to_down：直近 zigzag_len 本（当日含む）の高値・安値を更新したか（未来は見ない）
    - trend：to_up/to_downの切り替わりで反転するZigZagの状態
    - h1,h0 / l1,l0：直近2つのスイングハイ／スイングロー
    - market：h0がh1を「fib_factor分のマージンを付けて」上抜けたら強気(1)に転換 → これがMSB
    - OB：MSBを起こした「安値側のレグ」(h1〜l0の間)の中で、最後の陰線をOB候補とする
    - OBゾーンは、終値がゾーン下限を割るまで有効とみなす（インジケータのbox無効化条件と同じ）
    """
    n = len(df_d)
    high = df_d['High'].to_numpy(dtype=float)
    low = df_d['Low'].to_numpy(dtype=float)
    open_ = df_d['Open'].to_numpy(dtype=float)
    close = df_d['Close'].to_numpy(dtype=float)

    # 直近zigzag_len本（当日含む）の高値・安値更新判定はrollingで一括計算
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    to_up = (high_s >= high_s.rolling(zigzag_len, min_periods=1).max()).to_numpy()
    to_down = (low_s <= low_s.rolling(zigzag_len, min_periods=1).min()).to_numpy()

    trend = 1
    last_to_up_idx = -1
    last_to_down_idx = -1
    high_pivots, low_pivots = [], []
    h0 = h1 = l0 = l1 = None
    h0i = h1i = l0i = None
    last_h0_for_market = last_l0_for_market = None
    market = 1
    active_ob_low = active_ob_high = None
    last_msb_bar = None

    msb_confirmed = np.zeros(n, dtype=int)
    in_ob_zone = np.zeros(n, dtype=int)
    ob_low_arr = np.full(n, np.nan)
    ob_high_arr = np.full(n, np.nan)
    bars_since_msb = np.full(n, np.nan)

    for i in range(n):
        prev_trend = trend
        if trend == 1 and to_down[i]:
            trend = -1
        elif trend == -1 and to_up[i]:
            trend = 1

        if to_up[i]:
            last_to_up_idx = i
        if to_down[i]:
            last_to_down_idx = i

        if trend != prev_trend:
            if trend == 1:
                start = last_to_up_idx if last_to_up_idx >= 0 else 0
                seg = low[start:i + 1]
                low_val = seg.min()
                low_idx = start + int(seg.argmin())
                low_pivots.append((low_val, low_idx))
                if len(low_pivots) >= 2:
                    l1, l0 = low_pivots[-2][0], low_pivots[-1][0]
                else:
                    l0 = low_pivots[-1][0]
                l0i = low_pivots[-1][1]
            else:
                start = last_to_down_idx if last_to_down_idx >= 0 else 0
                seg = high[start:i + 1]
                high_val = seg.max()
                high_idx = start + int(seg.argmax())
                high_pivots.append((high_val, high_idx))
                if len(high_pivots) >= 2:
                    h1, h0 = high_pivots[-2][0], high_pivots[-1][0]
                    h1i = high_pivots[-2][1]
                else:
                    h0 = high_pivots[-1][0]
                h0i = high_pivots[-1][1]

        if None not in (h0, h1, l0, l1):
            if last_h0_for_market != h0 and last_l0_for_market != l0:
                if market == -1 and h0 > h1 and h0 > h1 + abs(h1 - l0) * fib_factor:
                    market = 1
                    last_h0_for_market, last_l0_for_market = h0, l0
                    msb_confirmed[i] = 1
                    last_msb_bar = i
                    # OB候補：h1i 〜 l0i（安値側のレグ）の中で最後の陰線
                    ob_idx = None
                    if h1i is not None and l0i is not None and h1i <= l0i:
                        for j in range(h1i, l0i + 1):
                            if close[j] < open_[j]:
                                ob_idx = j
                    if ob_idx is not None:
                        active_ob_low = float(low[ob_idx])
                        active_ob_high = float(high[ob_idx])
                    else:
                        active_ob_low = active_ob_high = None
                elif market == 1 and l0 < l1 and l0 < l1 - abs(h0 - l1) * fib_factor:
                    market = -1
                    last_h0_for_market, last_l0_for_market = h0, l0

        # OBゾーンの無効化判定（終値がゾーン下限を割ったら消滅、インジケータと同じ挙動）
        if active_ob_low is not None and close[i] < active_ob_low:
            active_ob_low = active_ob_high = None

        if active_ob_low is not None:
            ob_low_arr[i] = active_ob_low
            ob_high_arr[i] = active_ob_high
            if low[i] <= active_ob_high and high[i] >= active_ob_low:
                in_ob_zone[i] = 1

        if last_msb_bar is not None:
            bars_since_msb[i] = i - last_msb_bar

    idx = df_d.index
    return {
        'msb_confirmed': pd.Series(msb_confirmed, index=idx),
        'in_ob_zone': pd.Series(in_ob_zone, index=idx),
        'ob_low': pd.Series(ob_low_arr, index=idx),
        'ob_high': pd.Series(ob_high_arr, index=idx),
        'bars_since_msb': pd.Series(bars_since_msb, index=idx),
    }


def get_asof(series, dt):
    """指定日時点で確定している直近の値を取得（未来の値は使わない、日付のズレにも対応）"""
    if series is None or len(series) == 0:
        return np.nan
    pos = series.index.searchsorted(dt, side='right') - 1
    if pos < 0:
        return np.nan
    return series.iloc[pos]


def get_asof_with_prev(series, dt):
    """指定日時点の直近値と、その1つ前の値（前営業日相当）をまとめて取得"""
    if series is None or len(series) == 0:
        return np.nan, np.nan
    pos = series.index.searchsorted(dt, side='right') - 1
    if pos < 0:
        return np.nan, np.nan
    curr = series.iloc[pos]
    prev = series.iloc[pos - 1] if pos - 1 >= 0 else np.nan
    return curr, prev


# --- 日本の33業種区分 → 米国セクターETF（SPDR Select Sector）の近似対応表 ---
# 完全に一致する分類ではないため、近いセクターへの大まかな割り当て
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


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def export_to_csv(df, path=OUTPUT_FILE):
    """Pandas DataFrameをCSVへ出力（GitHub Actionsのartifactでも取得しやすい形）"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 文字化け防止のため utf-8-sig（Excelでもそのまま開ける）
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n✅ CSV出力が完了しました: {os.path.abspath(path)}")
        print(f"   行数: {len(df)}, 列数: {len(df.columns)}")
    except Exception as e:
        print(f"\n❌ CSV出力エラー: {e}")
        raise


if __name__ == "__main__":
    start_time = time.time()
    stock_list = get_jpx_stock_list()
    annual_perf = {}
    trade_log_list = []

    weekday_names = ["月曜", "火曜", "水曜", "木曜", "金曜"]
    weekday_perf = {name: {'profit': 0, 'trades': 0, 'wins': 0} for name in weekday_names}
    sector_perf = {}
    market_perf = {'プライム': {'profit': 0, 'trades': 0, 'wins': 0}}

    print("日経平均データを取得中...")
    df_nk = yf.download("^N225", period="10y", interval="1d", progress=False, repair=True)
    if isinstance(df_nk.columns, pd.MultiIndex):
        df_nk.columns = df_nk.columns.get_level_values(0)
    nk_sma25 = df_nk['Close'].rolling(window=25).mean()
    df_nk['nk_sma25_dev'] = (df_nk['Close'] - nk_sma25) / nk_sma25 * 100

    print("日経225先物データを取得中...")
    try:
        df_fut = yf.download("NIY=F", period="10y", interval="1d", progress=False, repair=True)
        if isinstance(df_fut.columns, pd.MultiIndex):
            df_fut.columns = df_fut.columns.get_level_values(0)
        if df_fut.empty:
            raise RuntimeError("先物データが空でした")
        fut_sma25 = df_fut['Close'].rolling(window=25).mean()
        df_fut['fut_sma25_dev'] = (df_fut['Close'] - fut_sma25) / fut_sma25 * 100
        fut_dev_series = df_fut['fut_sma25_dev'].sort_index()
        fut_close_series = df_fut['Close'].sort_index()
        print(f"  取得成功: {len(df_fut)}件, 期間 {df_fut.index.min().date()} 〜 {df_fut.index.max().date()}")
    except Exception as e:
        print(f"  ⚠ 先物データの取得に失敗しました（先物関連の指標はNoneになります）: {e}")
        fut_dev_series = None
        fut_close_series = None

    cash_close_series = df_nk['Close'].sort_index()

    print("米国セクターETFデータを取得中...")
    us_sector_chg_series = {}  # {ETFティッカー: 日次変化率(%)の系列}
    for etf in sorted(set(SECTOR_ETF_MAP.values())):
        try:
            df_etf = yf.download(etf, period="10y", interval="1d", progress=False, repair=True)
            if isinstance(df_etf.columns, pd.MultiIndex):
                df_etf.columns = df_etf.columns.get_level_values(0)
            if df_etf.empty:
                raise RuntimeError("空データ")
            chg = df_etf['Close'].pct_change() * 100
            us_sector_chg_series[etf] = chg.sort_index()
        except Exception as e:
            print(f"  ⚠ {etf} の取得に失敗（この業種のUSセクター指標はNoneになります）: {e}")
            us_sector_chg_series[etf] = None
    print(f"  取得完了: {sum(1 for v in us_sector_chg_series.values() if v is not None)}/{len(us_sector_chg_series)}銘柄")

    print(f"検証開始...")

    for idx, row in stock_list.iterrows():
        ticker, name, sector, market = row['Ticker'], row['銘柄名'], row['33業種区分'], row['Market']
        if pd.isna(sector): sector = "その他"
        if sector not in sector_perf:
            sector_perf[sector] = {'profit': 0, 'trades': 0, 'wins': 0}

        try:
            df_d = yf.download(ticker, period="10y", interval="1d", progress=False, repair=True)
            if df_d.empty or len(df_d) < 30: continue
            if isinstance(df_d.columns, pd.MultiIndex): df_d.columns = df_d.columns.get_level_values(0)

            # --- データクリーニング：明らかな異常値を除去 ---
            # 0以下の価格や欠損はNaN化して除外
            price_cols = ['Open', 'High', 'Low', 'Close']
            df_d[price_cols] = df_d[price_cols].where(df_d[price_cols] > 0)
            df_d = df_d.dropna(subset=price_cols)
            if len(df_d) < 30: continue

            # 前日比±50%を超える異常な急騰急落は、株式分割等でない限り基本的に起こらないためデータエラーとみなす
            daily_return = df_d['Close'].pct_change().abs()
            suspicious = daily_return > 0.5
            if suspicious.any():
                # 疑わしい行はスキップ対象として除外（トレード判定・指標計算から外す）
                df_d = df_d[~suspicious]
            if len(df_d) < 30: continue

            daily_value = df_d['Close'] * df_d['Volume']
            ma_value_10 = daily_value.rolling(window=10).mean()
            stoch_d, slow_d = calculate_stochastics(df_d, STOCH_K_PERIOD, STOCH_D_PERIOD, STOCH_SD_PERIOD)
            rsi_14 = calculate_rsi(df_d['Close'], period=14)
            vol_ma20 = df_d['Volume'].rolling(window=20).mean()
            vol_ratio = df_d['Volume'] / (vol_ma20 + 1e-10)
            sma25 = df_d['Close'].rolling(window=25).mean()
            sma25_dev = (df_d['Close'] - sma25) / sma25 * 100
            sma5 = df_d['Close'].rolling(window=5).mean()
            sma5_dev = (df_d['Close'] - sma5) / sma5 * 100
            msb_series = compute_msb_ob_series(df_d)  # 銘柄ごとに1回だけ計算（日次ループ内では参照するだけ）

            is_holding = False
            e_price, m_price, e_date = 0, 0, None
            entry_meta = {}

            for i in range(25, len(df_d) - 1):
                dt = df_d.index[i]
                close = float(df_d['Close'].iloc[i])
                tomorrow_open = float(df_d['Open'].iloc[i + 1])
                curr_year = dt.year

                if curr_year not in annual_perf:
                    annual_perf[curr_year] = {'profit': 0, 'trades': 0, 'wins': 0}

                d_0, sd_0 = stoch_d.iloc[i], slow_d.iloc[i]
                d_1, sd_1 = stoch_d.iloc[i - 1], slow_d.iloc[i - 1]

                if not is_holding:
                    if close < MIN_PRICE_LIMIT or ma_value_10.iloc[i] < MIN_VALUE_LIMIT: continue
                    if pd.isna(d_0) or pd.isna(sd_0) or pd.isna(d_1) or pd.isna(sd_1): continue

                    touch_oversold = (d_1 <= OVERSOLD_LEVEL) or (sd_1 <= OVERSOLD_LEVEL) or (d_0 <= OVERSOLD_LEVEL)
                    golden_cross = (d_1 <= sd_1) and (d_0 > sd_0)

                    if golden_cross:  # 「売られすぎ」条件はゲートから外し、記録用フラグとして残す（後から両方検証できるように）
                        entry_dt = df_d.index[i]
                        nk_dev = df_nk.loc[entry_dt, 'nk_sma25_dev'] if entry_dt in df_nk.index else np.nan
                        vr_today = float(vol_ratio.iloc[i])
                        next_weekday = df_d.index[i + 1].weekday()  # 0=月, 1=火 ...

                        # --- 陰線／前日比終値判定（検証用フラグ） ---
                        open_today = float(df_d['Open'].iloc[i])
                        close_today = close
                        close_prev = float(df_d['Close'].iloc[i - 1])
                        high_today = float(df_d['High'].iloc[i])
                        low_today = float(df_d['Low'].iloc[i])
                        is_bearish = 1 if close_today < open_today else 0          # 陰線（始値より終値が低い）
                        is_close_down = 1 if close_today < close_prev else 0        # 終値が前日より低い
                        close_chg_prev_day = (close_today - close_prev) / close_prev * 100 if close_prev != 0 else np.nan  # 前日比変化率(%)
                        sma5_dev_today = float(sma5_dev.iloc[i]) if not pd.isna(sma5_dev.iloc[i]) else np.nan  # 5日移動平均線からの乖離率(%)

                        # --- ローソク足の形状（連続値で記録し、閾値は後から自由に検証） ---
                        day_range = max(high_today - low_today, 1e-6)
                        body_top = max(open_today, close_today)
                        body_bottom = min(open_today, close_today)
                        lower_shadow_ratio = (body_bottom - low_today) / day_range   # 下ヒゲの割合（0〜1）
                        upper_shadow_ratio = (high_today - body_top) / day_range     # 上ヒゲの割合（0〜1）
                        body_ratio = (body_top - body_bottom) / day_range           # 実体の割合（0〜1）

                        # --- MSB / OB 判定（銘柄ごとに事前計算済みの系列を参照） ---
                        msb_ob = {
                            'msb_confirmed': int(msb_series['msb_confirmed'].iloc[i]),
                            'in_ob_zone': int(msb_series['in_ob_zone'].iloc[i]),
                            'bars_since_msb': msb_series['bars_since_msb'].iloc[i],
                        }
                        # --- 前日の値幅がOBゾーンに触れていたか（前日時点で有効だったゾーンで判定） ---
                        ob_touched_prev_day = int(msb_series['in_ob_zone'].iloc[i - 1]) if i - 1 >= 0 else 0
                        # 直近3日（GC前日・前々日・3日前）のいずれかで帯（OBゾーン）に触れていたか
                        ob_touched_recent3 = 1 if any(
                            msb_series['in_ob_zone'].iloc[i - k] for k in (1, 2, 3) if i - k >= 0
                        ) else 0

                        # --- 日経225先物関連指標 ---
                        fut_dev_at_entry = get_asof(fut_dev_series, entry_dt) if fut_dev_series is not None else np.nan
                        fut_close_at_entry, fut_close_prev = get_asof_with_prev(fut_close_series, entry_dt) if fut_close_series is not None else (np.nan, np.nan)
                        cash_close_at_entry = get_asof(cash_close_series, entry_dt)
                        if pd.isna(fut_close_at_entry) or pd.isna(cash_close_at_entry) or cash_close_at_entry == 0:
                            futures_basis = np.nan
                        else:
                            futures_basis = (fut_close_at_entry - cash_close_at_entry) / cash_close_at_entry * 100
                        # 先物の「前日比」変化率（一晩でどれだけ動いたか）
                        if pd.isna(fut_close_at_entry) or pd.isna(fut_close_prev) or fut_close_prev == 0:
                            futures_overnight_chg = np.nan
                        else:
                            futures_overnight_chg = (fut_close_at_entry - fut_close_prev) / fut_close_prev * 100

                        # --- 現物(日経)が、夜間先物の示唆した動きに対してどれだけ乖離したか ---
                        cash_close_at_entry_v2, cash_close_prev = get_asof_with_prev(cash_close_series, entry_dt)
                        if pd.isna(cash_close_at_entry_v2) or pd.isna(cash_close_prev) or cash_close_prev == 0:
                            cash_chg_today = np.nan
                        else:
                            cash_chg_today = (cash_close_at_entry_v2 - cash_close_prev) / cash_close_prev * 100
                        if pd.isna(cash_chg_today) or pd.isna(futures_overnight_chg):
                            overnight_gap = np.nan
                        else:
                            # マイナスに大きい＝現物が先物の示唆以上に下げすぎている（理論上は「買い」のサイン）
                            overnight_gap = cash_chg_today - futures_overnight_chg

                        # --- 前日の米国同業種セクターETFの値動き ---
                        us_etf = SECTOR_ETF_MAP.get(sector)
                        us_series = us_sector_chg_series.get(us_etf) if us_etf else None
                        us_sector_chg = get_asof(us_series, entry_dt) if us_series is not None else np.nan

                        # --- 追加フィルタ判定（分析結果に基づく） ---
                        if USE_ENTRY_FILTERS:
                            if pd.isna(nk_dev) or nk_dev > NIKKEI_DEV_THRESHOLD:
                                continue  # 日経が十分売り込まれていない日は見送り
                            if EXCLUDE_TUESDAY and next_weekday == 1:
                                continue  # 火曜エントリーは見送り
                            if vr_today < MIN_VOLUME_RATIO:
                                continue  # 出来高が薄すぎる銘柄は見送り
                            if is_crisis_period(entry_dt):
                                continue  # 既知の暴落期間は見送り
                            if pd.isna(futures_basis) or futures_basis >= FUTURES_BASIS_THRESHOLD:
                                continue  # 先物ベーシスがプラス（先物が現物より強気）の日は見送り

                        if tomorrow_open * 100 <= INITIAL_CASH:
                            is_holding = True
                            e_price = tomorrow_open
                            m_price = tomorrow_open
                            e_date = df_d.index[i + 1]

                            entry_meta = {
                                'rsi_14': float(rsi_14.iloc[i]),
                                'volume_ratio': vr_today,
                                'sma25_dev': float(sma25_dev.iloc[i]),
                                'nikkei_sma25_dev': float(nk_dev) if not pd.isna(nk_dev) else None,
                                'is_bearish': is_bearish,
                                'is_close_down': is_close_down,
                                'close_chg_prev_day': round(close_chg_prev_day, 3) if not pd.isna(close_chg_prev_day) else None,
                                'sma5_dev': round(sma5_dev_today, 3) if not pd.isna(sma5_dev_today) else None,
                                'lower_shadow_ratio': round(lower_shadow_ratio, 3),
                                'upper_shadow_ratio': round(upper_shadow_ratio, 3),
                                'body_ratio': round(body_ratio, 3),
                                'msb_confirmed': msb_ob['msb_confirmed'],
                                'in_ob_zone': msb_ob['in_ob_zone'],
                                'bars_since_msb': msb_ob['bars_since_msb'],
                                'ob_touched_prev_day': ob_touched_prev_day,
                                'ob_touched_recent3': ob_touched_recent3,
                                'touch_oversold': 1 if touch_oversold else 0,
                                'futures_sma25_dev': float(fut_dev_at_entry) if not pd.isna(fut_dev_at_entry) else None,
                                'futures_basis': float(futures_basis) if not pd.isna(futures_basis) else None,
                                'futures_overnight_chg': float(futures_overnight_chg) if not pd.isna(futures_overnight_chg) else None,
                                'cash_chg_today': float(cash_chg_today) if not pd.isna(cash_chg_today) else None,
                                'overnight_gap': float(overnight_gap) if not pd.isna(overnight_gap) else None,
                                'us_sector_chg': float(us_sector_chg) if not pd.isna(us_sector_chg) else None,
                            }
                else:
                    m_price = max(m_price, close)
                    reason = None
                    dead_cross = (d_1 >= sd_1) and (d_0 < sd_0) if not (
                                pd.isna(d_0) or pd.isna(sd_0) or pd.isna(d_1) or pd.isna(sd_1)) else False

                    if close <= m_price * (1 - TRAILING_STOP_RATE) and close > e_price:
                        reason = "トレリ"
                    elif close <= e_price * (1 - LOSS_CUT_RATE):
                        reason = "損切り"
                    elif dead_cross:
                        reason = "デッドC"

                    if reason:
                        exit_dt = df_d.index[i + 1]
                        profit = (float(df_d['Open'].iloc[i + 1]) - e_price) * 100

                        annual_perf[curr_year]['profit'] += profit
                        annual_perf[curr_year]['trades'] += 1
                        if profit > 0: annual_perf[curr_year]['wins'] += 1

                        w_idx_e = e_date.weekday()
                        if w_idx_e < 5:
                            weekday_perf[weekday_names[w_idx_e]]['profit'] += profit
                            weekday_perf[weekday_names[w_idx_e]]['trades'] += 1
                            if profit > 0: weekday_perf[weekday_names[w_idx_e]]['wins'] += 1

                        sector_perf[sector]['profit'] += profit
                        sector_perf[sector]['trades'] += 1
                        if profit > 0: sector_perf[sector]['wins'] += 1

                        if market in market_perf:
                            market_perf[market]['profit'] += profit
                            market_perf[market]['trades'] += 1
                            if profit > 0: market_perf[market]['wins'] += 1

                        if True:  # ダウンロードした全期間を対象にする（年の制限を撤廃）
                            trade_log_list.append({
                                'ticker': ticker, 'name': name, 'sector': sector, 'market': market,
                                'entry': e_date.strftime('%Y-%m-%d'),
                                'exit': exit_dt.strftime('%Y-%m-%d'),
                                'entry_price': e_price,
                                'profit': profit, 'reason': reason,
                                'wday': weekday_names[w_idx_e],
                                'rsi_14': round(entry_meta.get('rsi_14', 0), 2),
                                'volume_ratio': round(entry_meta.get('volume_ratio', 0), 2),
                                'sma25_dev': round(entry_meta.get('sma25_dev', 0), 2),
                                'nikkei_sma25_dev': round(entry_meta.get('nikkei_sma25_dev', 0), 2) if entry_meta.get(
                                    'nikkei_sma25_dev') is not None else None,
                                'is_bearish': entry_meta.get('is_bearish', 0),
                                'is_close_down': entry_meta.get('is_close_down', 0),
                                'close_chg_prev_day': entry_meta.get('close_chg_prev_day'),
                                'sma5_dev': entry_meta.get('sma5_dev'),
                                'lower_shadow_ratio': entry_meta.get('lower_shadow_ratio', 0),
                                'upper_shadow_ratio': entry_meta.get('upper_shadow_ratio', 0),
                                'body_ratio': entry_meta.get('body_ratio', 0),
                                'msb_confirmed': entry_meta.get('msb_confirmed', 0),
                                'in_ob_zone': entry_meta.get('in_ob_zone', 0),
                                'bars_since_msb': entry_meta.get('bars_since_msb'),
                                'ob_touched_prev_day': entry_meta.get('ob_touched_prev_day', 0),
                                'ob_touched_recent3': entry_meta.get('ob_touched_recent3', 0),
                                'touch_oversold': entry_meta.get('touch_oversold', 0),
                                'futures_sma25_dev': round(entry_meta['futures_sma25_dev'], 2) if entry_meta.get('futures_sma25_dev') is not None else None,
                                'futures_basis': round(entry_meta['futures_basis'], 3) if entry_meta.get('futures_basis') is not None else None,
                                'futures_overnight_chg': round(entry_meta['futures_overnight_chg'], 3) if entry_meta.get('futures_overnight_chg') is not None else None,
                                'cash_chg_today': round(entry_meta['cash_chg_today'], 3) if entry_meta.get('cash_chg_today') is not None else None,
                                'overnight_gap': round(entry_meta['overnight_gap'], 3) if entry_meta.get('overnight_gap') is not None else None,
                                'us_sector_chg': round(entry_meta['us_sector_chg'], 3) if entry_meta.get('us_sector_chg') is not None else None,
                            })
                        is_holding = False
        except Exception as e:
            print(f"⚠ {ticker} でエラー: {e}")
            continue

        if (idx + 1) % 400 == 0: print(f"{idx + 1} 銘柄完了...")

    # --- CSV出力実行 ---
    df_log = pd.DataFrame(trade_log_list)
    if not df_log.empty:
        df_log['entry'] = pd.to_datetime(df_log['entry']).dt.strftime('%Y-%m-%d')
        df_log['exit'] = pd.to_datetime(df_log['exit']).dt.strftime('%Y-%m-%d')
        df_log['holding_days'] = (pd.to_datetime(df_log['exit']) - pd.to_datetime(df_log['entry'])).dt.days
        df_log['is_win'] = df_log['profit'].apply(lambda x: 1 if x > 0 else 0)

        export_to_csv(df_log)

        # --- ウォークフォワード検証：学習期間 vs 検証期間で成績を比較 ---
        df_log['entry_year'] = pd.to_datetime(df_log['entry']).dt.year
        train = df_log[df_log['entry_year'] < WALKFORWARD_SPLIT_YEAR]
        test = df_log[df_log['entry_year'] >= WALKFORWARD_SPLIT_YEAR]

        def _summary(g, label):
            if len(g) == 0:
                print(f"  {label}: トレードなし")
                return
            win_rate = g['is_win'].mean() * 100
            avg_profit = g['profit'].mean()
            total_profit = g['profit'].sum()
            print(f"  {label}: 件数={len(g)}, 勝率={win_rate:.1f}%, "
                  f"平均損益={avg_profit:,.0f}円, 合計損益={total_profit:,.0f}円")

        print(f"\n=== ウォークフォワード検証（フィルタ適用: {USE_ENTRY_FILTERS}） ===")
        print(f"条件: 日経25日線乖離 <= {NIKKEI_DEV_THRESHOLD}%, "
              f"火曜除外={EXCLUDE_TUESDAY}, 出来高比率 >= {MIN_VOLUME_RATIO}, "
              f"先物ベーシス < {FUTURES_BASIS_THRESHOLD}")
        _summary(train, f"学習期間（〜{WALKFORWARD_SPLIT_YEAR - 1}年）")
        _summary(test, f"検証期間（{WALKFORWARD_SPLIT_YEAR}年〜、未知データ）")
        print("学習期間と検証期間で勝率・平均損益が大きく変わらなければ、条件が過去データへの")
        print("過剰最適化ではなく、ある程度将来にも通用する可能性が高いと判断できます。")
    else:
        print("\n⚠ トレードが1件も発生しませんでした。CSVは出力されません。")
        print("   （フィルタ条件が厳しすぎる可能性があります。USE_ENTRY_FILTERS = False で確認できます）")

    print(f"\n総実行時間: {time.time() - start_time:.1f}秒")