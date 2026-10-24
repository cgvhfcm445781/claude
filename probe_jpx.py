"""
JPXの信用取引残高のページを調べ、公開ファイルの形式とリンクを記録する確認用スクリプト（GitHub Actions で実行）。
結果は probe_jpx/ に保存してコミットする。
"""
import os
import re
import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}
BASE = "https://www.jpx.co.jp"
PAGES = ["/markets/statistics-equities/margin/01.html", "/markets/statistics-equities/margin/03.html",
         "/markets/statistics-equities/margin/06.html", "/markets/statistics-equities/margin/07.html",
         "/markets/statistics-equities/margin/08.html", "/markets/statistics-equities/margin/09.html"]
OUT = "probe_jpx"


def main():
    os.makedirs(OUT, exist_ok=True)
    report = []
    files = []
    for p in PAGES:
        try:
            r = requests.get(BASE + p, headers=UA, timeout=30)
            r.encoding = r.apparent_encoding
            report.append(f"## {p} status={r.status_code} len={len(r.text)}")
            text = re.sub(r"<[^>]+>", " ", r.text)
            text = re.sub(r"\s+", " ", text)
            i = text.rfind("信用取引残高等 ")
            report.append(text[i:i + 2500])
            links = re.findall(r'href="([^"]+\.(?:xls|xlsx|csv|pdf|zip))"', r.text, flags=re.I)
            report.append("links: " + "\n".join(links[:40]))
            files += [l for l in links if re.search(r"\.(xls|xlsx|csv|zip)$", l, re.I)]
        except Exception as e:
            report.append(f"## {p} error {e}")
    # 表計算ファイルを最大4つダウンロードして中身の冒頭を記録
    for l in list(dict.fromkeys(files))[:6]:
        url = l if l.startswith("http") else BASE + l
        try:
            r = requests.get(url, headers=UA, timeout=60)
            name = os.path.basename(url)
            open(os.path.join(OUT, name), "wb").write(r.content)
            report.append(f"## downloaded {url} {len(r.content)} bytes")
            try:
                import pandas as pd
                x = pd.read_excel(os.path.join(OUT, name), header=None, nrows=15) if not name.endswith(".csv") \
                    else pd.read_csv(os.path.join(OUT, name), header=None, nrows=15, encoding="cp932")
                report.append(x.to_string())
            except Exception as e:
                report.append(f"read error {e}")
        except Exception as e:
            report.append(f"## download error {url} {e}")
    open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8").write("\n\n".join(report))


if __name__ == "__main__":
    main()
