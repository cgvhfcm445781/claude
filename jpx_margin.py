"""
JPXが毎営業日無料で公開している「銘柄別信用取引残高」（全銘柄・PDF）を読み取るモジュール。
https://www.jpx.co.jp/markets/statistics-equities/margin/01.html

返す表: Code（5桁）, Date（申込日）, ShrtVol（売残・株数）, LongVol（買残・株数）
J-Quants の信用残（/markets/margin-interest）と同じ数字であることを確認済み（2026-10-22 の 1301 極洋で一致）
"""
import io
import re
import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}
BASE = "https://www.jpx.co.jp"
PAGE = "/markets/statistics-equities/margin/01.html"


def _num(s):
    s = str(s or '').replace(',', '').replace(' ', '')
    neg = s.startswith('▲')
    s = s.lstrip('▲')
    try:
        v = float(s)
    except ValueError:
        return float('nan')
    return -v if neg else v


def latest_files(on_or_before=None):
    """最新の申込日（on_or_before 以前）の PDF の URL 一覧（1日分が _01 などに分かれている場合もまとめて返す）"""
    html = requests.get(BASE + PAGE, headers=UA, timeout=30).text
    links = re.findall(r'href="([^"]+/(\d{8})_mtall[^"]*\.pdf)"', html)
    dates = sorted({d for _, d in links}, reverse=True)
    if on_or_before is not None:
        dates = [d for d in dates if pd.Timestamp(d) <= pd.Timestamp(on_or_before)]
    if not dates:
        return None, []
    d = dates[0]
    return pd.Timestamp(d), [BASE + u for u, dd in links if dd == d]


def parse_pdf(content):
    import pdfplumber
    rows = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            for t in page.extract_tables():
                for r in t:
                    # 「株数」の行: [銘柄, 市場, 種別, コード, 新証券コード, '株数 Shs.', 売残, 前日比, 上場比, 買残, ...]
                    if len(r) >= 10 and str(r[5] or '').startswith('株数') and re.fullmatch(r'[0-9A-Z]{5}', str(r[3] or '')):
                        rows.append({'Code': r[3], 'ShrtVol': _num(r[6]), 'LongVol': _num(r[9])})
    return pd.DataFrame(rows)


def fetch_latest(on_or_before=None):
    date, urls = latest_files(on_or_before)
    if not urls:
        raise RuntimeError("JPXの銘柄別信用取引残高のPDFが見つかりませんでした")
    parts = [parse_pdf(requests.get(u, headers=UA, timeout=120).content) for u in urls]
    df = pd.concat(parts, ignore_index=True).drop_duplicates('Code')
    df['Date'] = date
    print(f"JPX信用残: {date:%Y-%m-%d} 申込み分 {len(df)} 銘柄（{len(urls)}ファイル）")
    return df


if __name__ == '__main__':
    print(fetch_latest().head())
