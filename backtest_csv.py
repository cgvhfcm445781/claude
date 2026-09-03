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

# --- CSV出力設定 ---
OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "trade_log.csv")


def get_jpx_stock_list():
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
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
    df_nk = yf.download("^N225", period="6y", interval="1d", progress=False, repair=True)
    if isinstance(df_nk.columns, pd.MultiIndex):
        df_nk.columns = df_nk.columns.get_level_values(0)
    nk_sma25 = df_nk['Close'].rolling(window=25).mean()
    df_nk['nk_sma25_dev'] = (df_nk['Close'] - nk_sma25) / nk_sma25 * 100

    print(f"検証開始...")

    for idx, row in stock_list.iterrows():
        ticker, name, sector, market = row['Ticker'], row['銘柄名'], row['33業種区分'], row['Market']
        if pd.isna(sector): sector = "その他"
        if sector not in sector_perf:
            sector_perf[sector] = {'profit': 0, 'trades': 0, 'wins': 0}

        try:
            df_d = yf.download(ticker, period="6y", interval="1d", progress=False, repair=True)
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
                        if tomorrow_open * 100 <= INITIAL_CASH:
                            is_holding = True
                            e_price = tomorrow_open
                            m_price = tomorrow_open
                            e_date = df_d.index[i + 1]

                            entry_dt = df_d.index[i]
                            nk_dev = df_nk.loc[entry_dt, 'nk_sma25_dev'] if entry_dt in df_nk.index else np.nan

                            entry_meta = {
                                'rsi_14': float(rsi_14.iloc[i]),
                                'volume_ratio': float(vol_ratio.iloc[i]),
                                'sma25_dev': float(sma25_dev.iloc[i]),
                                'nikkei_sma25_dev': float(nk_dev) if not pd.isna(nk_dev) else None
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

                        if curr_year in [2024, 2025, 2026]:
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
                                    'nikkei_sma25_dev') is not None else None
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
    else:
        print("\n⚠ トレードが1件も発生しませんでした。CSVは出力されません。")

    print(f"\n総実行時間: {time.time() - start_time:.1f}秒")
