"""
JPXの銘柄別信用取引残高（全銘柄・日次PDF: 01.html の *_mtall.pdf）が読み取れるかを確認するスクリプト（GitHub Actions で実行）。
結果は probe_jpx/ に保存してコミットする。
"""
import os
import re
import requests
import pdfplumber

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}
BASE = "https://www.jpx.co.jp"
PAGE = "/markets/statistics-equities/margin/01.html"
OUT = "probe_jpx"


def main():
    os.makedirs(OUT, exist_ok=True)
    r = requests.get(BASE + PAGE, headers=UA, timeout=30)
    links = re.findall(r'href="([^"]+_mtall[^"]*\.pdf)"', r.text)
    rep = [f"links ({len(links)}):"] + links[:10]
    url = BASE + links[0]
    pdf_bytes = requests.get(url, headers=UA, timeout=120).content
    path = os.path.join(OUT, os.path.basename(url))
    open(path, "wb").write(pdf_bytes)
    rep.append(f"downloaded {url} {len(pdf_bytes)} bytes")
    with pdfplumber.open(path) as pdf:
        rep.append(f"pages {len(pdf.pages)}")
        for i in (0, 1, len(pdf.pages) - 1):
            p = pdf.pages[i]
            rep.append(f"----- page {i + 1} text -----")
            rep.append((p.extract_text() or "")[:3000])
            t = p.extract_table()
            rep.append(f"----- page {i + 1} table rows={len(t) if t else 0} -----")
            for row in (t or [])[:12]:
                rep.append(" | ".join(str(c) for c in row))
    open(os.path.join(OUT, "report_pdf.txt"), "w", encoding="utf-8").write("\n".join(rep))


if __name__ == "__main__":
    main()
