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
NIKKEI_DEV_THRESHOLD = -5.0  # 日経平均が25日線からこの%以上下に乖離している日のみエントリー
EXCLUDE_TUESDAY = True       # 火曜エントリーを除外
MIN_VOLUME_RATIO = 0.5       # 出来高比率がこれ未満（閑散銘柄）は除外

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


def detect_msb_ob(df_d, i, pivot_window=3, lookback=40):
    """
    簡易版 MSB（Market Structure Break）/ OB（Order Block）判定。
    - スイングハイ：前後 pivot_window 本より高値が高い足（確定には pivot_window 本の経過が必要）
    - MSB：直近のスイングハイが1つ前のスイングハイより低い（下降構造＝安値・高値切り下げ）状態から、
           その後の値動きが直近スイングハイを上抜けたら「構造の転換」とみなす
    - OB：MSBを起こした上昇の起点から遡って最後の陰線を「注文が溜まっていたゾーン」とみなす
    - in_ob_zone：当日の値幅がそのOBのレンジと重なっていれば「押し目でOBに戻ってきた」と判定
    未来の情報は使わず、日 i 時点で確定済みの情報のみで判定する（先読みなし）。
    """
    highs = df_d['High'].values
    lows = df_d['Low'].values
    opens = df_d['Open'].values
    closes = df_d['Close'].values

    start = max(pivot_window, i - lookback)
    end = i - 1 - pivot_window  # 日 i-1 時点で確定しているスイングハイの最終候補
    if end <= start:
        return {'msb_confirmed': 0, 'in_ob_zone': 0, 'bars_since_msb': None}

    swing_highs = [k for k in range(start, end + 1)
                   if highs[k] == highs[k - pivot_window:k + pivot_window + 1].max()]
    if len(swing_highs) < 2:
        return {'msb_confirmed': 0, 'in_ob_zone': 0, 'bars_since_msb': None}

    sh_prior, sh_recent = swing_highs[-2], swing_highs[-1]
    if highs[sh_recent] >= highs[sh_prior]:
        return {'msb_confirmed': 0, 'in_ob_zone': 0, 'bars_since_msb': None}  # 高値切り上げ＝下降構造でない

    # 直近スイングハイを上抜けたポイントを探す（先読みなし：sh_recent より後、i まで）
    break_idx = None
    for j in range(sh_recent + 1, i + 1):
        if highs[j] > highs[sh_recent]:
            break_idx = j
            break
    if break_idx is None:
        return {'msb_confirmed': 0, 'in_ob_zone': 0, 'bars_since_msb': None}

    # 上抜けを起こした上昇の起点から遡り、最後の陰線をOB候補とする
    ob_idx = None
    for j in range(break_idx - 1, sh_recent - 1, -1):
        if closes[j] < opens[j]:
            ob_idx = j
            break

    in_ob_zone = 0
    if ob_idx is not None:
        ob_low, ob_high = lows[ob_idx], highs[ob_idx]
        if lows[i] <= ob_high and highs[i] >= ob_low:
            in_ob_zone = 1

    return {'msb_confirmed': 1, 'in_ob_zone': in_ob_zone, 'bars_since_msb': i - break_idx}


def get_asof(series, dt):
    """指定日時点で確定している直近の値を取得（未来の値は使わない、日付のズレにも対応）"""
    if series is None or len(series) == 0:
        return np.nan
    pos = series.index.searchsorted(dt, side='right') - 1
    if pos < 0:
        return np.nan
    return series.iloc[pos]


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

                    if touch_oversold and golden_cross:
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

                        # --- ローソク足の形状（連続値で記録し、閾値は後から自由に検証） ---
                        day_range = max(high_today - low_today, 1e-6)
                        body_top = max(open_today, close_today)
                        body_bottom = min(open_today, close_today)
                        lower_shadow_ratio = (body_bottom - low_today) / day_range   # 下ヒゲの割合（0〜1）
                        upper_shadow_ratio = (high_today - body_top) / day_range     # 上ヒゲの割合（0〜1）
                        body_ratio = (body_top - body_bottom) / day_range           # 実体の割合（0〜1）

                        # --- MSB / OB 判定 ---
                        msb_ob = detect_msb_ob(df_d, i)

                        # --- 日経225先物関連指標 ---
                        fut_dev_at_entry = get_asof(fut_dev_series, entry_dt) if fut_dev_series is not None else np.nan
                        fut_close_at_entry = get_asof(fut_close_series, entry_dt) if fut_close_series is not None else np.nan
                        cash_close_at_entry = get_asof(cash_close_series, entry_dt)
                        if pd.isna(fut_close_at_entry) or pd.isna(cash_close_at_entry) or cash_close_at_entry == 0:
                            futures_basis = np.nan
                        else:
                            futures_basis = (fut_close_at_entry - cash_close_at_entry) / cash_close_at_entry * 100

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
                                'lower_shadow_ratio': round(lower_shadow_ratio, 3),
                                'upper_shadow_ratio': round(upper_shadow_ratio, 3),
                                'body_ratio': round(body_ratio, 3),
                                'msb_confirmed': msb_ob['msb_confirmed'],
                                'in_ob_zone': msb_ob['in_ob_zone'],
                                'bars_since_msb': msb_ob['bars_since_msb'],
                                'futures_sma25_dev': float(fut_dev_at_entry) if not pd.isna(fut_dev_at_entry) else None,
                                'futures_basis': float(futures_basis) if not pd.isna(futures_basis) else None,
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
                                'lower_shadow_ratio': entry_meta.get('lower_shadow_ratio', 0),
                                'upper_shadow_ratio': entry_meta.get('upper_shadow_ratio', 0),
                                'body_ratio': entry_meta.get('body_ratio', 0),
                                'msb_confirmed': entry_meta.get('msb_confirmed', 0),
                                'in_ob_zone': entry_meta.get('in_ob_zone', 0),
                                'bars_since_msb': entry_meta.get('bars_since_msb'),
                                'futures_sma25_dev': round(entry_meta['futures_sma25_dev'], 2) if entry_meta.get('futures_sma25_dev') is not None else None,
                                'futures_basis': round(entry_meta['futures_basis'], 3) if entry_meta.get('futures_basis') is not None else None,
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
              f"火曜除外={EXCLUDE_TUESDAY}, 出来高比率 >= {MIN_VOLUME_RATIO}")
        _summary(train, f"学習期間（〜{WALKFORWARD_SPLIT_YEAR - 1}年）")
        _summary(test, f"検証期間（{WALKFORWARD_SPLIT_YEAR}年〜、未知データ）")
        print("学習期間と検証期間で勝率・平均損益が大きく変わらなければ、条件が過去データへの")
        print("過剰最適化ではなく、ある程度将来にも通用する可能性が高いと判断できます。")
    else:
        print("\n⚠ トレードが1件も発生しませんでした。CSVは出力されません。")
        print("   （フィルタ条件が厳しすぎる可能性があります。USE_ENTRY_FILTERS = False で確認できます）")

    print(f"\n総実行時間: {time.time() - start_time:.1f}秒")