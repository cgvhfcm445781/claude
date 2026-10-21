"""
週足ストキャスGCの「シグナル研究」用データ収集スクリプト。

考え方
- 日足から週足(W-FRI)を作り、確定済みの週だけでシグナルを判定する（今週の未確定足は使わない）
- シグナル週の金曜終値で判定 → 翌週の月曜寄り付き(=翌週の最初の営業日の始値)でエントリー
- ストキャスのパラメータ(k,d,sd)を複数試し、ゲートは広め(GATE_WIDE)で全部記録する
  → GC水準(gc_level)やその他の条件による絞り込みは analyze_weekly_signals.py 側で行う
- 結果は「固定期間リターン」と「ルール決済(損切り/トレーリング/デッドクロス/時間切れ)」の両方を記録
- 銘柄ごとにシグナルを独立に評価する（1銘柄1ポジション制の重複排除はしない。
  代わりに weeks_since_prev_signal を記録するので、分析側で間引ける）
"""
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io
import os
import time
import warnings

warnings.simplefilter('ignore', FutureWarning)

# ---------------- 設定 ----------------
INITIAL_CASH = 400000          # 100株買える上限判定に使う（日足版と同じ）
USE_CASH_LIMIT = True          # Falseにすると価格帯の制限なしで全銘柄を対象にする
MIN_PRICE_LIMIT = 1000
MIN_WEEKLY_VALUE = 250_000_000  # 直近4週の平均週間売買代金（日足版の5,000万円/日 × 5日相当）

# ストキャスのパラメータ候補 (k期間, d期間, sd期間)。週足なので日足(9,3,3)より短め〜同じを試す
STOCH_PARAM_GRID = [(5, 3, 3), (9, 3, 3), (14, 3, 3)]
GATE_WIDE = 60                 # GC前後3値の最小値がこれ以下ならシグナルとして記録（絞り込みは分析側）

# 決済ルールの候補: {名前: (損切り率, トレーリング率)}。週足なので日足(3%,2%)より広めにしてある
EXIT_GRID = {'A': (0.05, 0.04), 'B': (0.08, 0.06), 'C': (0.12, 0.08)}
MAX_HOLD_WEEKS = 13            # 時間切れ（約3ヶ月）
FWD_WEEKS = (1, 2, 4, 8)       # 固定期間リターンを記録する週数

OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "weekly_signals.csv")


# ---------------- データ取得 ----------------
def get_jpx_stock_list():
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xlsx"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept": "*/*",
    }
    res = requests.get(url, headers=headers, timeout=30)
    print(f"[JPX] status_code={res.status_code}, content-length={len(res.content)} bytes")
    if res.status_code != 200:
        raise RuntimeError(f"JPXへのアクセスに失敗しました (status={res.status_code})")
    content = res.content
    is_xlsx = content[:2] == b'PK'
    df = pd.read_excel(io.BytesIO(content), engine='openpyxl' if is_xlsx else 'xlrd')
    target_markets = ['プライム（内国株式）', 'スタンダード（内国株式）']
    df = df[df['市場・商品区分'].isin(target_markets)].copy()
    df['Market'] = df['市場・商品区分'].map({
        'プライム（内国株式）': 'プライム', 'スタンダード（内国株式）': 'スタンダード'})
    df['Ticker'] = df['コード'].astype(str) + ".T"
    return df[['Ticker', '銘柄名', '33業種区分', 'Market']]


def clean_daily(df_d):
    """日足データのクリーニング（日足版バックテストと同じ方針）"""
    if isinstance(df_d.columns, pd.MultiIndex):
        df_d.columns = df_d.columns.get_level_values(0)
    if getattr(df_d.index, 'tz', None) is not None:
        df_d.index = df_d.index.tz_localize(None)
    price_cols = ['Open', 'High', 'Low', 'Close']
    df_d[price_cols] = df_d[price_cols].where(df_d[price_cols] > 0)
    df_d = df_d.dropna(subset=price_cols)
    suspicious = df_d['Close'].pct_change().abs() > 0.5
    if suspicious.any():
        df_d = df_d[~suspicious]
    return df_d


# ---------------- 指標 ----------------
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


def calculate_bollinger(series, period=20, num_std=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return (series - lower) / ((upper - lower) + 1e-10)


def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def calculate_dmi_adx(df, period=14):
    high, low, close = df['High'], df['Low'], df['Close']
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / (atr + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-10)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return plus_di, minus_di, adx


# ---------------- 週足の作成 ----------------
def build_weekly(df_d, today=None):
    """日足→週足(W-FRI)。まだ終わっていない最新週は除外する（先読み・未確定足の混入防止）"""
    w = df_d.resample('W-FRI').agg(
        {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
    w['Value'] = (df_d['Close'] * df_d['Volume']).resample('W-FRI').sum()
    day_s = pd.Series(df_d.index, index=df_d.index)
    w['first_day'] = day_s.resample('W-FRI').first()
    w['last_day'] = day_s.resample('W-FRI').last()
    w = w.dropna(subset=['Open', 'Close'])
    today = today if today is not None else pd.Timestamp.today().normalize()
    if len(w) and w['last_day'].iloc[-1] < w.index[-1] and w.index[-1] >= today:
        w = w.iloc[:-1]
    return w


def build_nikkei_features(df_nk_d):
    """日経平均の週足ベース指標（地合い）"""
    nk = df_nk_d[['Open', 'High', 'Low', 'Close']].resample('W-FRI').agg(
        {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    out = pd.DataFrame(index=nk.index)
    for n in (13, 26):
        sma = nk['Close'].rolling(n).mean()
        out[f'nk_dev{n}'] = (nk['Close'] - sma) / sma * 100
    out['nk_ret_1w'] = nk['Close'].pct_change() * 100
    out['nk_ret_4w'] = nk['Close'].pct_change(4) * 100
    out['nk_stoch_d'], _ = calculate_stochastics(nk, 9, 3, 3)
    return out


def compute_features(w, nk_feats=None):
    """週足の各種特徴量を一括計算（すべて、その週の終値時点で確定している値のみ）"""
    O, H, L, C, V = w['Open'], w['High'], w['Low'], w['Close'], w['Volume']
    F = pd.DataFrame(index=w.index)
    F['rsi_14w'] = calculate_rsi(C, 14)
    _, _, hist = calculate_macd(C)
    F['macd_hist_pct'] = hist / C * 100
    for n in (13, 26, 52):
        sma = C.rolling(n).mean()
        F[f'sma{n}_dev'] = (C - sma) / sma * 100
        if n in (13, 26):
            F[f'sma{n}_slope'] = (sma - sma.shift(4)) / sma.shift(4) * 100
    F['bb_percent_b'] = calculate_bollinger(C, 20, 2)
    plus_di, minus_di, adx = calculate_dmi_adx(w, 14)
    F['adx_14w'] = adx
    F['di_diff'] = plus_di - minus_di
    F['volume_ratio_w'] = V / (V.rolling(13).mean() + 1e-10)
    F['prev_ret_1w'] = C.pct_change() * 100
    F['prev_ret_4w'] = C.pct_change(4) * 100
    F['prev_ret_13w'] = C.pct_change(13) * 100
    F['dist_52w_high'] = (C / H.rolling(52).max() - 1) * 100
    F['dist_52w_low'] = (C / L.rolling(52).min() - 1) * 100
    tr = pd.concat([H - L, (H - C.shift()).abs(), (L - C.shift()).abs()], axis=1).max(axis=1)
    F['atr_pct_14w'] = tr.rolling(14).mean() / C * 100
    rng = (H - L).clip(lower=1e-6)
    F['w_is_bull'] = (C > O).astype(int)
    F['w_lower_shadow'] = (np.minimum(O, C) - L) / rng
    F['w_body_ratio'] = (np.maximum(O, C) - np.minimum(O, C)) / rng
    down = (C < C.shift()).astype(int)
    F['consec_down_weeks'] = down.groupby((down == 0).cumsum()).cumcount()
    F['up_weeks_last8'] = (C > C.shift()).astype(int).rolling(8).sum()
    if nk_feats is not None:
        nkr = nk_feats.reindex(w.index, method='ffill')
        for col in nk_feats.columns:
            F[col] = nkr[col]
    return F


# ---------------- 決済シミュレーション ----------------
def simulate_exit(t, O, C, dead_cross, n, stop, trail, max_hold):
    """
    t: シグナル週の番号。エントリーは t+1 週の始値。
    各週の終値で判定し、翌週の始値で決済する（日足版と同じ「終値判定→翌寄り決済」）。
    戻り値: (リターン%, 保有週数, 理由)。データ末尾までに決済されなければ None（未決済は除外）
    """
    entry = O[t + 1]
    peak = entry
    for k in range(t + 1, n - 1):
        peak = max(peak, C[k])
        held = k - t
        reason = None
        if C[k] <= peak * (1 - trail) and C[k] > entry:
            reason = 'trail'
        elif C[k] <= entry * (1 - stop):
            reason = 'stop'
        elif dead_cross[k]:
            reason = 'dc'
        elif held >= max_hold:
            reason = 'time'
        if reason:
            return (O[k + 1] / entry - 1) * 100, held, reason
    return None


# ---------------- 銘柄ごとの処理 ----------------
def process_ticker(df_d, meta, nk_feats=None, today=None):
    w = build_weekly(df_d, today)
    if len(w) < 60:
        return []
    F = compute_features(w, nk_feats)
    O, H, L, C = (w[c].to_numpy(dtype=float) for c in ['Open', 'High', 'Low', 'Close'])
    n = len(w)
    val4 = w['Value'].rolling(4).mean().to_numpy()
    rows = []

    for (k, d, sdn) in STOCH_PARAM_GRID:
        sto, slo = calculate_stochastics(w, k, d, sdn)
        gc = (sto.shift(1) <= slo.shift(1)) & (sto > slo)
        dc_arr = ((sto.shift(1) >= slo.shift(1)) & (sto < slo)).fillna(False).to_numpy()
        min3 = pd.concat([sto.shift(1), slo.shift(1), sto], axis=1).min(axis=1)
        sig = (gc & (min3 <= GATE_WIDE) & sto.notna() & slo.notna()).to_numpy()
        min4 = sto.rolling(4).min()
        sto_a, slo_a, min3_a, min4_a = sto.to_numpy(), slo.to_numpy(), min3.to_numpy(), min4.to_numpy()

        prev_t = None
        for t in np.flatnonzero(sig):
            if t + 1 >= n:
                continue  # 翌週がまだ無い（＝最新のシグナル。バックテストでは評価不能）
            if C[t] < MIN_PRICE_LIMIT or not (val4[t] >= MIN_WEEKLY_VALUE):
                continue
            entry = O[t + 1]
            if not np.isfinite(entry) or entry <= 0:
                continue
            if USE_CASH_LIMIT and entry * 100 > INITIAL_CASH:
                continue

            row = dict(meta)
            row.update({
                'stoch_params': f'{k}-{d}-{sdn}',
                'signal_week': w.index[t].strftime('%Y-%m-%d'),
                'entry_date': pd.Timestamp(w['first_day'].iloc[t + 1]).strftime('%Y-%m-%d'),
                'entry_price': entry,
                'gc_level': round(float(sto_a[t]), 2),
                'gc_level_slow': round(float(slo_a[t]), 2),
                'stoch_min3': round(float(min3_a[t]), 2),
                'stoch_min4w': round(float(min4_a[t]), 2) if np.isfinite(min4_a[t]) else np.nan,
                'stoch_gap': round(float(sto_a[t] - slo_a[t]), 2),
                'stoch_d_chg': round(float(sto_a[t] - sto_a[t - 1]), 2) if t >= 1 else np.nan,
                'weeks_since_prev_signal': (t - prev_t) if prev_t is not None else np.nan,
            })
            for col, val in F.iloc[t].items():
                row[col] = round(float(val), 4) if pd.notna(val) else np.nan

            # 固定期間リターン（エントリー週の始値 → N週目の終値）
            for N in FWD_WEEKS:
                row[f'fwd_ret_{N}w'] = round((C[t + N] / entry - 1) * 100, 3) if t + N < n else np.nan
            if t + 4 < n:
                row['fwd_mae_4w'] = round((L[t + 1:t + 5].min() / entry - 1) * 100, 3)
                row['fwd_mfe_4w'] = round((H[t + 1:t + 5].max() / entry - 1) * 100, 3)
            else:
                row['fwd_mae_4w'] = row['fwd_mfe_4w'] = np.nan

            # ルール決済（複数の損切り/トレーリング幅）
            for key, (stop, trail) in EXIT_GRID.items():
                res = simulate_exit(t, O, C, dc_arr, n, stop, trail, MAX_HOLD_WEEKS)
                if res is None:
                    row[f'exit_{key}_ret'] = np.nan
                    row[f'exit_{key}_weeks'] = np.nan
                    row[f'exit_{key}_reason'] = None
                else:
                    row[f'exit_{key}_ret'] = round(res[0], 3)
                    row[f'exit_{key}_weeks'] = res[1]
                    row[f'exit_{key}_reason'] = res[2]

            rows.append(row)
            prev_t = t
    return rows


def export_to_csv(df, path=OUTPUT_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n✅ CSV出力が完了しました: {os.path.abspath(path)}  行数={len(df)}, 列数={len(df.columns)}")


if __name__ == "__main__":
    start = time.time()
    stock_list = get_jpx_stock_list()

    print("日経平均データを取得中...")
    df_nk = yf.download("^N225", period="10y", interval="1d", progress=False, repair=True)
    if isinstance(df_nk.columns, pd.MultiIndex):
        df_nk.columns = df_nk.columns.get_level_values(0)
    nk_feats = build_nikkei_features(df_nk)

    all_rows = []
    print("週足シグナルの収集を開始...")
    for idx, r in stock_list.iterrows():
        meta = {'ticker': r['Ticker'], 'name': r['銘柄名'], 'sector': r['33業種区分'], 'market': r['Market']}
        try:
            df_d = yf.download(r['Ticker'], period="10y", interval="1d", progress=False, repair=True)
            if df_d.empty or len(df_d) < 300:
                continue
            df_d = clean_daily(df_d)
            if len(df_d) < 300:
                continue
            all_rows.extend(process_ticker(df_d, meta, nk_feats))
        except Exception as e:
            print(f"⚠ {r['Ticker']} でエラー: {e}")
            continue
        if (idx + 1) % 400 == 0:
            print(f"{idx + 1} 銘柄完了... (シグナル累計 {len(all_rows)})")

    df_log = pd.DataFrame(all_rows)
    if df_log.empty:
        print("⚠ シグナルが1件も出ませんでした。")
    else:
        export_to_csv(df_log)
        # 簡易サマリ（詳細な分析は analyze_weekly_signals.py で）
        for p, g in df_log.groupby('stoch_params'):
            gg = g.dropna(subset=['exit_B_ret'])
            print(f"  params={p}: シグナル{len(g)}件, 決済済み{len(gg)}件, "
                  f"決済B 勝率={(gg['exit_B_ret'] > 0).mean() * 100:.1f}%, 平均={gg['exit_B_ret'].mean():.2f}%")
    print(f"\n総実行時間: {time.time() - start:.1f}秒")