"""
J-Quants API (V2) から、決算・信用残・空売り比率のデータを取得して保存するスクリプト。
（GitHub Actions の fetch_jquants.yml から実行される想定。APIキーは Secrets の JQUANTS_API_KEY）

取得するもの（data_jq/ 以下）
- fin_summary.parquet  : 決算短信のサマリ（開示日、EPS、売上、利益、会社予想など）… 営業日ごと
- margin.parquet       : 銘柄別の信用取引残高（買い残・売り残）… 週次（金曜の申込日付）
- short_ratio.parquet  : 33業種別の空売り比率 … 営業日ごと
- master.parquet       : 上場銘柄一覧（業種コードの対応表として使う）
- done.json            : 取得済みの日付（途中で止まっても続きから取れるようにする）

設定は jquants_config.json
- start / end : 取得期間（"auto" なら start=今日の2年前、end=今日。プランの範囲外の日付はスキップ）
- rpm         : 1分あたりのリクエスト数の上限（無料プランは5）
- minutes     : 1回の実行で使う時間の上限（GitHub Actions の6時間制限に収めるため）
"""
import json
import os
import sys
import time
import pandas as pd

OUT_DIR = 'data_jq'
CONFIG = 'jquants_config.json'


def load_config():
    cfg = {'start': 'auto', 'end': 'auto', 'rpm': 4, 'minutes': 330}   # 無料プランの上限5回/分に余裕を持たせて4
    if os.path.exists(CONFIG):
        cfg.update(json.load(open(CONFIG, encoding='utf-8')))
    today = pd.Timestamp.today().normalize()
    cfg['start'] = today - pd.DateOffset(years=2) if cfg['start'] == 'auto' else pd.Timestamp(cfg['start'])
    cfg['end'] = today if cfg['end'] == 'auto' else pd.Timestamp(cfg['end'])
    return cfg


class Fetcher:
    def __init__(self, cfg):
        from jquantsapi import ClientV2
        key = os.environ.get('JQUANTS_API_KEY', '')
        if not key:
            sys.exit("⚠ JQUANTS_API_KEY が設定されていません。GitHubの Settings → Secrets and variables → Actions で登録してください")
        self.cli = ClientV2(api_key=key)
        self.wait = 60.0 / cfg['rpm']
        self.deadline = time.time() + cfg['minutes'] * 60
        os.makedirs(OUT_DIR, exist_ok=True)
        self.done_path = os.path.join(OUT_DIR, 'done.json')
        self.done = json.load(open(self.done_path)) if os.path.exists(self.done_path) else {}
        self.buf = {}

    def call(self, fn, **kw):
        time.sleep(self.wait)   # レート制限を守る
        return fn(**kw)

    def out_of_time(self):
        return time.time() > self.deadline

    def run_dates(self, name, fn, dates, kwname):
        done = set(self.done.get(name, []))
        todo = [d for d in dates if d.strftime('%Y%m%d') not in done]
        print(f"[{name}] 対象 {len(dates)} 日 / 取得済み {len(dates) - len(todo)} / 残り {len(todo)}")
        n_err = 0
        for i, d in enumerate(todo):
            if self.out_of_time():
                print(f"[{name}] 時間切れのため中断（次回の実行で続きから取得）")
                break
            ds = d.strftime('%Y%m%d')
            try:
                df = self.call(fn, **{kwname: ds})
                if df is not None and len(df):
                    self.buf.setdefault(name, []).append(df)
                done.add(ds)
                n_err = 0
            except Exception as e:
                msg = str(e)
                # プランの対象期間外などは「取得済み（データなし）」として扱う
                if any(s in msg for s in ('400', 'subscription', 'Subscription')):
                    done.add(ds)
                else:
                    print(f"[{name}] {ds} 失敗: {msg[:200]}")
                    n_err += 1
                    if any(s in msg for s in ('401', '403')) and len(done) == 0:
                        sys.exit("⚠ 認証エラー。APIキーが正しいか、プランでこのデータが使えるか確認してください")
                    if n_err >= 10:
                        print(f"[{name}] 連続エラーが多いため中断")
                        break
            if (i + 1) % 50 == 0:
                self.done[name] = sorted(done)
                self.flush()
                print(f"[{name}] {i + 1}/{len(todo)}")
        self.done[name] = sorted(done)
        self.flush()

    def flush(self):
        for name, parts in self.buf.items():
            if not parts:
                continue
            path = os.path.join(OUT_DIR, f'{name}.parquet')
            new = pd.concat(parts, ignore_index=True)
            if os.path.exists(path):
                new = pd.concat([pd.read_parquet(path), new], ignore_index=True)
            new = new.astype({c: 'string' for c in new.columns if new[c].dtype == object})
            new.drop_duplicates().to_parquet(path, index=False)
            self.buf[name] = []
        json.dump(self.done, open(self.done_path, 'w'))


def main():
    cfg = load_config()
    print(f"期間 {cfg['start']:%Y-%m-%d} 〜 {cfg['end']:%Y-%m-%d}, {cfg['rpm']} リクエスト/分, 上限 {cfg['minutes']} 分")
    f = Fetcher(cfg)
    master = f.call(f.cli.get_eq_master)
    master.to_parquet(os.path.join(OUT_DIR, 'master.parquet'), index=False)
    bdays = pd.bdate_range(cfg['start'], cfg['end'])
    fridays = [d for d in bdays if d.weekday() == 4]
    # 優先順: 信用残(件数が少ない) → 空売り比率 → 決算
    f.run_dates('margin', f.cli.get_mkt_margin_interest, fridays, 'date_yyyymmdd')
    f.run_dates('short_ratio', f.cli.get_mkt_short_ratio, bdays, 'date_yyyymmdd')
    f.run_dates('fin_summary', f.cli.get_fin_summary, bdays, 'date_yyyymmdd')
    for name in ('margin', 'short_ratio', 'fin_summary'):
        p = os.path.join(OUT_DIR, f'{name}.parquet')
        if os.path.exists(p):
            print(f"  {name}: {len(pd.read_parquet(p))} 行, 取得済み {len(f.done.get(name, []))} 日")


if __name__ == '__main__':
    main()
