"""
JPXの上場廃止・監理銘柄・整理銘柄などのページの形式を確認する一時的なスクリプト（GitHub Actions で実行）。
"""
import os
import re
import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}
BASE = "https://www.jpx.co.jp"
PAGES = ["/listing/stocks/delisted/index.html", "/listing/market-alerts/supervision/index.html",
         "/listing/market-alerts/surveillance/index.html", "/listing/market-alerts/alert/index.html",
         "/listing/market-alerts/index.html", "/listing/stocks/delisted/01.html"]
OUT = "probe_jpx"


def main():
    os.makedirs(OUT, exist_ok=True)
    rep = []
    for p in PAGES:
        try:
            r = requests.get(BASE + p, headers=UA, timeout=30)
            r.encoding = r.apparent_encoding
            rep.append(f"## {p} status={r.status_code} len={len(r.text)}")
            tables = re.findall(r"<table.*?</table>", r.text, flags=re.S)
            rep.append(f"tables={len(tables)}")
            for t in tables[:2]:
                rows = re.findall(r"<tr.*?</tr>", t, flags=re.S)
                for row in rows[:8]:
                    cells = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", c)).strip()
                             for c in re.findall(r"<t[hd].*?</t[hd]>", row, flags=re.S)]
                    rep.append(" | ".join(cells))
            links = re.findall(r'href="([^"]+\.(?:xls|xlsx|csv|pdf|html))"', r.text)
            rep.append("links: " + " ".join(l for l in links if 'listing' in l)[:1500])
        except Exception as e:
            rep.append(f"## {p} error {e}")
    open(os.path.join(OUT, "report_delist.txt"), "w", encoding="utf-8").write("\n".join(rep))


if __name__ == "__main__":
    main()
