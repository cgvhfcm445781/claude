"""
相場の悲観度・リスクリワードの検証に使う、市場全体の指標を取得するスクリプト。
（GitHub Actions の fetch_macro.yml から実行される想定）

出力: data/macro_daily.parquet（date, symbol, Close）
取得できなかったシンボルはスキップして、ログに出す
"""
import os
import pandas as pd
import yfinance as yf

SYMBOLS = {
    '^VIX': '米VIX（S&P500の予想変動率＝恐怖指数）',
    '^JNIV': '日経平均VI（日本版の恐怖指数。取れない場合あり）',
    '^TNX': '米10年金利(%)',
    '^IRX': '米3か月金利(%)',
    'JPY=X': 'ドル円',
    '^GSPC': 'S&P500',
    '^N225': '日経平均',
    '1306.T': 'TOPIX連動ETF',
    'HYG': '米ハイイールド債ETF（信用不安の目安）',
    'LQD': '米投資適格社債ETF',
    'GC=F': '金先物',
    'CL=F': '原油先物',
}


def main():
    out = []
    for sym, desc in SYMBOLS.items():
        try:
            d = yf.download(sym, period='max', interval='1d', auto_adjust=True, progress=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            d = d.dropna(subset=['Close'])
            if d.empty:
                print(f"⚠ {sym} ({desc}): データなし")
                continue
            if getattr(d.index, 'tz', None) is not None:
                d.index = d.index.tz_localize(None)
            out.append(pd.DataFrame({'date': d.index, 'symbol': sym, 'Close': d['Close'].astype(float).values}))
            print(f"{sym} ({desc}): {d.index.min():%Y-%m-%d}〜{d.index.max():%Y-%m-%d} {len(d)}日")
        except Exception as e:
            print(f"⚠ {sym} ({desc}): {e}")
    os.makedirs('data', exist_ok=True)
    pd.concat(out).to_parquet(os.path.join('data', 'macro_daily.parquet'), index=False)


if __name__ == '__main__':
    main()
