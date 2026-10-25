"""
JPXが公開している「上場廃止・そのおそれがある銘柄」の一覧を読み取るモジュール。

- 上場廃止銘柄一覧       : https://www.jpx.co.jp/listing/stocks/delisted/index.html （廃止日・理由。これから廃止される銘柄も載る）
- 監理銘柄・整理銘柄     : https://www.jpx.co.jp/listing/market-alerts/supervision/index.html （上場廃止のおそれ／廃止決定）
- 猶予期間・改善期間     : 上場維持基準に適合していない銘柄（注意）

返す表: code（4桁）, 種別（上場廃止予定 / 監理・整理銘柄 / 猶予・改善期間）, 日付, 理由
"""
import re
import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}
BASE = "https://www.jpx.co.jp"
PAGES = [
    ('上場廃止予定', '/listing/stocks/delisted/index.html'),
    ('監理・整理銘柄', '/listing/market-alerts/supervision/index.html'),
    ('猶予・改善期間', '/listing/market-alerts/grace-period/index.html'),
    ('猶予・改善期間', '/listing/market-alerts/improvement-period/index.html'),
]
TOB_WORDS = ('公開買付', 'ＭＢＯ', 'MBO', '買収', '株式交換', '株式移転', '合併', '完全子会社')


def _clean(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s)).strip()


def _tables(html):
    out = []
    for t in re.findall(r"<table.*?</table>", html, flags=re.S):
        rows = [[_clean(c) for c in re.findall(r"<t[hd][^>]*>.*?</t[hd]>", r, flags=re.S)]
                for r in re.findall(r"<tr.*?</tr>", t, flags=re.S)]
        rows = [r for r in rows if r]
        if rows:
            out.append(rows)
    return out


def fetch(today=None):
    today = pd.Timestamp(today or pd.Timestamp.now(tz='Asia/Tokyo').tz_localize(None).normalize())
    recs = []
    for kind, path in PAGES:
        try:
            r = requests.get(BASE + path, headers=UA, timeout=30)
            r.encoding = r.apparent_encoding
        except Exception as e:
            print(f"⚠ JPX {kind} のページを取得できませんでした: {e}")
            continue
        for rows in _tables(r.text):
            head = rows[0]
            if 'コード' not in head:
                continue
            ic = head.index('コード')
            idate = next((i for i, h in enumerate(head) if '年月日' in h or '廃止日' in h), None)
            ireason = next((i for i, h in enumerate(head) if '理由' in h or '備考' in h), None)
            for row in rows[1:]:
                if len(row) <= ic or not re.fullmatch(r"[0-9A-Z]{4}", row[ic]):
                    continue
                date = pd.to_datetime(row[idate], errors='coerce') if idate is not None and idate < len(row) else pd.NaT
                if kind == '上場廃止予定' and pd.notna(date) and date < today:
                    continue   # すでに廃止された銘柄
                recs.append({'code': row[ic], '種別': kind, '日付': date,
                             '理由': row[ireason] if ireason is not None and ireason < len(row) else ''})
    df = pd.DataFrame(recs, columns=['code', '種別', '日付', '理由'])
    print(f"JPX上場廃止関連: {df['種別'].value_counts().to_dict() if len(df) else 0}")
    return df.drop_duplicates()


def is_tob(reason):
    return any(w in str(reason) for w in TOB_WORDS)


if __name__ == '__main__':
    print(fetch().to_string())
