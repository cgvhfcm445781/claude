"""
週次シグナル通知（Discord）。iPhoneのショートカットなどから GitHub Actions の weekly_alert.yml を手動実行して使う。

通知する内容（いずれも直近の確定した週の金曜終値時点。週の途中に実行した場合は前週末時点）
1. 急落リバウンド: 日経平均の4週騰落が -7%以下ならシグナル。買う候補（4週騰落の下位20%を下げた順に最大10銘柄）と
   +20%の利確の目安、最長保有の期限（8週後）
2. 割安株: 低PER & 12か月線より上 & 信用残が重くない 上位10銘柄（13週ごとに入れ替え）
3. 順張り: 過去12か月（直近1か月を除く）の上昇率が高い & 信用残が重くない & 減益予想でない 上位10銘柄（13週ごとに入れ替え）
   ※2と3は半分ずつ持つ想定（バックテストでは各10銘柄の半々で年+25.8%、最大DD-19%）

データ
- 株価: yfinance（実行のたびに直近2年分を取得）
- 決算: J-Quants Light（data_jq/ に保存済みのデータに、足りない日の分だけ追加取得してコミット）
- 信用残: JPXが毎営業日公開している銘柄別信用取引残高のPDF（jpx_margin.py）
環境変数: DISCORD_WEBHOOK_URL, JQUANTS_API_KEY
"""
import json
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from Weekly_backtest import get_jpx_stock_list, clean_daily
import jpx_alerts

warnings.simplefilter('ignore')
DISCORD_WEBHOOK_URL = os.environ.get('DISCORD_WEBHOOK_URL', '')
JQ_DIR = 'data_jq'
REBALANCE_ANCHOR = pd.Timestamp('2026-08-14')   # 割安株・順張りの入れ替え基準日（ここから13週ごと）
CRASH_TH, LOSER_PCT, CRASH_N, TARGET, HOLD_WEEKS = -7, 0.20, 10, 0.20, 8
PICK_N = 10
MIN_DAILY_VALUE = 50_000_000
PRICE_MIN, PRICE_MAX = 1000, 4000


# ---------------- 通知 ----------------
def send_discord(content):
    print(content)
    if not DISCORD_WEBHOOK_URL:
        print("⚠ DISCORD_WEBHOOK_URL が未設定のため送信をスキップ")
        return
    lines, chunk = content.split('\n'), ''
    for ln in lines:   # 2000文字制限があるので行単位で分割
        if len(chunk) + len(ln) + 1 > 1900:
            requests.post(DISCORD_WEBHOOK_URL, json={'content': chunk}, timeout=15)
            time.sleep(0.5)
            chunk = ''
        chunk += ln + '\n'
    if chunk.strip():
        requests.post(DISCORD_WEBHOOK_URL, json={'content': chunk}, timeout=15)


# ---------------- J-Quants（足りない分だけ追加取得） ----------------
def code_to_ticker(code):
    c = str(code)
    return (c[:4] if len(c) == 5 else c) + '.T'


def update_fin():
    """決算データ（J-Quants Light）: 保存済みのデータに、足りない日の分だけ追加取得する"""
    key = os.environ.get('JQUANTS_API_KEY', '')
    path = os.path.join(JQ_DIR, 'fin_summary.parquet')
    fin = pd.read_parquet(path)
    if not key:
        print("⚠ JQUANTS_API_KEY が未設定のため、保存済みの決算データだけで計算します")
        return fin
    from jquantsapi import ClientV2
    cli = ClientV2(api_key=key)
    today = pd.Timestamp.now(tz='Asia/Tokyo').tz_localize(None).normalize()
    new = []
    for d in pd.bdate_range(pd.to_datetime(fin['DiscDate']).max() + pd.Timedelta(days=1), today):
        try:
            x = cli.get_fin_summary(date_yyyymmdd=d.strftime('%Y%m%d'))
            if len(x):
                new.append(x)
        except Exception as e:
            print(f"⚠ 決算 {d:%Y-%m-%d}: {str(e)[:120]}")
        time.sleep(1.1)   # Lightプランの上限(60回/分)に収める
    if new:
        fin = pd.concat([fin] + new, ignore_index=True)
        fin = fin.astype({c: 'string' for c in fin.columns if fin[c].dtype == object}).drop_duplicates()
        fin.to_parquet(path, index=False)
    print(f"J-Quants 決算の追加取得: {sum(len(x) for x in new)} 件")
    return fin


def get_margin(asof):
    """信用残（JPXの無料PDF）。取れなければ保存済みの J-Quants データで代用"""
    try:
        import jpx_margin
        return jpx_margin.fetch_latest(on_or_before=asof)
    except Exception as e:
        print(f"⚠ JPXの信用残を取得できませんでした（{e}）。保存済みのデータを使います")
        return pd.read_parquet(os.path.join(JQ_DIR, 'margin.parquet'))


# ---------------- 株価 ----------------
def download_prices(tickers, batch=100):
    out = {}
    for i in range(0, len(tickers), batch):
        b = tickers[i:i + batch]
        for attempt in range(2):
            try:
                raw = yf.download(b, period='2y', interval='1d', group_by='ticker', auto_adjust=True,
                                  threads=True, progress=False)
                break
            except Exception as e:
                print(f"⚠ batch {i}: {e}")
                time.sleep(10)
        else:
            continue
        for t in b:
            try:
                d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                d = d.dropna(how='all')
                if getattr(d.index, 'tz', None) is not None:
                    d.index = d.index.tz_localize(None)
                if len(d) >= 260:
                    out[t] = clean_daily(d[['Open', 'High', 'Low', 'Close', 'Volume']].copy())
            except KeyError:
                pass
        print(f"株価 {min(i + batch, len(tickers))}/{len(tickers)}")
        time.sleep(1)
    return out


def last_completed_week_end(dates):
    """直近の確定した週の最終営業日。今日の足（引け前の可能性がある）は使わない"""
    now = pd.Timestamp.now(tz='Asia/Tokyo').tz_localize(None)
    dates = pd.DatetimeIndex(dates)
    if now.hour < 16:
        dates = dates[dates < now.normalize()]
    last = dates.max()
    fri = last + pd.Timedelta(days=(4 - last.weekday()) % 7)      # その週の金曜
    if last.weekday() == 4 or now.normalize() > fri:              # 金曜の引け後、または週末・翌週
        return last
    return dates[dates <= fri - pd.Timedelta(weeks=1)].max()      # 週の途中なら前週末


# ---------------- 計算 ----------------
def stock_features(prices, asof):
    rows = []
    for t, d in prices.items():
        d = d[d.index <= asof]
        if len(d) < 260 or d.index[-1] < asof - pd.Timedelta(days=7):
            continue
        C = d['Close']
        wk = C.resample('W-FRI').last().dropna()
        me = C.resample('ME').last()
        if (asof + pd.offsets.BDay(1)).month == asof.month:     # 月の途中なら当月の足はまだ確定していない
            me = me[me.index < asof.replace(day=1)]
        val20 = (C * d['Volume']).tail(20).mean()
        # 1日で±50%を超える動き（値幅制限があるので実際にはまず起きない＝株価データの誤り）が過去1年にあれば外す
        bad = bool((C.pct_change().abs().tail(260) > 0.5).any())
        rows.append({'ticker': t, 'price': C.iloc[-1], 'value20': val20, 'bad_data': bad,
                     'ret4': (wk.iloc[-1] / wk.iloc[-5] - 1) * 100 if len(wk) > 5 else np.nan,
                     'mom12_1': C.iloc[-21] / C.iloc[-251] - 1 if len(C) > 251 else np.nan,
                     'm_above': me.iloc[-1] > me.tail(12).mean() if len(me) >= 12 else False})
    f = pd.DataFrame(rows)
    f['universe'] = (f['value20'] >= MIN_DAILY_VALUE) & f['price'].between(PRICE_MIN, PRICE_MAX) & ~f['bad_data']
    return f


def add_fundamentals(f, fin, mar, asof):
    fin = fin.copy()
    for c in ['FEPS', 'NxFEPS', 'OP', 'FOP', 'NxFOP']:
        fin[c] = pd.to_numeric(fin[c], errors='coerce')
    fin['ticker'] = fin['Code'].map(code_to_ticker)
    fin = fin[fin['DocType'].str.contains('FinancialStatements', na=False)]
    fin = fin[pd.to_datetime(fin['DiscDate']) < asof]
    fin['fwd_eps'] = np.where(fin['CurPerType'] == 'FY', fin['NxFEPS'], fin['FEPS'])
    last = fin.sort_values(['DiscDate', 'DiscTime']).groupby('ticker').last()
    f['fwd_eps'] = f['ticker'].map(last['fwd_eps'])
    f['E/P'] = f['fwd_eps'] / f['price']
    # 予想営業利益の伸び（今期の会社予想 ÷ 前期の実績）。順張りでは減益予想の銘柄を外す（検証で成績が上がった）
    fin['fwd_op'] = np.where(fin['CurPerType'] == 'FY', fin['NxFOP'], fin['FOP'])
    fy_op = fin[fin['CurPerType'] == 'FY'].sort_values(['DiscDate', 'DiscTime']).groupby('ticker')['OP'].last()
    f['op_growth'] = f['ticker'].map(last['fwd_op']) / f['ticker'].map(fy_op.where(fy_op > 0)) - 1
    mar = mar.copy()
    mar['ticker'] = mar['Code'].map(code_to_ticker)
    mar['Date'] = pd.to_datetime(mar['Date'])
    mar = mar[mar['Date'] <= asof]
    latest = mar[mar['Date'] == mar['Date'].max()]
    lv = latest.groupby('ticker')['LongVol'].apply(lambda s: pd.to_numeric(s, errors='coerce').sum())
    sv = latest.groupby('ticker')['ShrtVol'].apply(lambda s: pd.to_numeric(s, errors='coerce').sum())
    vol_w = f['value20'] / f['price'] * 5
    f['long_w'] = f['ticker'].map(lv) / vol_w
    f['short_w'] = f['ticker'].map(sv) / vol_w
    u = f['universe']
    f['heavy_margin'] = (f['long_w'] > f.loc[u, 'long_w'].median() * 2) | (f['short_w'] > f.loc[u, 'short_w'].median() * 2)
    return f, mar['Date'].max()


# ---------------- 資金の設定（alert_config.json） ----------------
CONFIG_PATH = 'alert_config.json'


def load_config():
    """資金: 戦略に使う総額（円）。0 なら資金の制限なしで全銘柄を通知
    単元株数: 100（通常の売買）または 1（S株・かぶミニなど1株単位の売買）
    最大銘柄数: 割安株・順張りを合わせて持つ銘柄数の上限 / 急落の最大銘柄数: 急落時に持つ銘柄数の上限"""
    cfg = {'資金': 0, '単元株数': 100, '最大銘柄数': 20, '急落の最大銘柄数': 10}
    if os.path.exists(CONFIG_PATH):
        cfg.update(json.load(open(CONFIG_PATH, encoding='utf-8')))
    return cfg


def plan_shares(price, slot_budget, available, unit):
    """1銘柄あたりの予算で買える株数（単元株数の倍数）。予算に届かなくても残り資金で1単元買えるなら1単元"""
    lot = price * unit
    if lot > available:
        return 0
    n = max(1, int(min(slot_budget, available) // lot))
    return n * unit


# ---------------- 保有銘柄の管理（portfolio.csv） ----------------
PF_PATH, HIST_PATH = 'portfolio.csv', 'trade_history.csv'
PF_COLS = ['種類', 'コード', '銘柄名', 'シグナル日', '買付日', '買値', '株数', '利確価格', '売り期限', '状態']


def load_portfolio():
    if os.path.exists(PF_PATH):
        pf = pd.read_csv(PF_PATH, encoding='utf-8-sig')
        for c in PF_COLS:
            if c not in pf:
                pf[c] = np.nan
        return pf[PF_COLS]
    return pd.DataFrame(columns=PF_COLS)


def save_portfolio(pf, hist):
    pf.to_csv(PF_PATH, index=False, encoding='utf-8-sig')
    if hist:
        h = pd.DataFrame(hist)
        if os.path.exists(HIST_PATH):
            h = pd.concat([pd.read_csv(HIST_PATH, encoding='utf-8-sig'), h], ignore_index=True)
        h.to_csv(HIST_PATH, index=False, encoding='utf-8-sig')


def show(r):
    return f"{str(r['コード']).replace('.T', '')} {str(r['銘柄名'])[:10]}"


def add_row(pf, kind, r, asof, deadline, shares=np.nan):
    row = {'種類': kind, 'コード': r['ticker'], '銘柄名': r['name'], 'シグナル日': f"{asof:%Y-%m-%d}",
           '買付日': np.nan, '買値': np.nan, '株数': shares, '利確価格': np.nan, '売り期限': f"{deadline:%Y-%m-%d}",
           '状態': '買い予定'}
    return pd.concat([pf, pd.DataFrame([row])], ignore_index=True)


def fill_entry_prices(pf, prices):
    """買い予定の銘柄に、シグナル日の翌営業日の始値を買値として記録する"""
    for i, r in pf[pf['状態'] == '買い予定'].iterrows():
        d = prices.get(r['コード'])
        if d is None:
            continue
        nxt = d[d.index > pd.Timestamp(r['シグナル日'])]
        if len(nxt):
            pf.loc[i, ['買付日', '買値', '状態']] = [f"{nxt.index[0]:%Y-%m-%d}", float(nxt['Open'].iloc[0]), '保有']
            if r['種類'] == '急落':
                pf.loc[i, '利確価格'] = round(float(nxt['Open'].iloc[0]) * (1 + TARGET))
    pf['買値'] = pd.to_numeric(pf['買値'], errors='coerce')
    pf['利確価格'] = pd.to_numeric(pf['利確価格'], errors='coerce')
    return pf


def close_row(r, price, date, reason):
    ret = (price / r['買値'] - 1) * 100 if pd.notna(r['買値']) and r['買値'] > 0 else np.nan
    return {'種類': r['種類'], 'コード': r['コード'], '銘柄名': r['銘柄名'], '買付日': r['買付日'], '買値': r['買値'],
            '売却日(目安)': f"{pd.Timestamp(date):%Y-%m-%d}", '売値(目安)': round(float(price), 1) if pd.notna(price) else np.nan,
            '損益%(目安)': round(ret, 2) if pd.notna(ret) else np.nan, '理由': reason}


def main():
    t0 = time.time()
    fin = update_fin()
    meta = get_jpx_stock_list()
    names = dict(zip(meta['Ticker'], meta['銘柄名']))
    prices = download_prices(meta['Ticker'].tolist())
    mk = {s: yf.download(s, period='1y', auto_adjust=True, progress=False)['Close'].squeeze().dropna() for s in ('^N225', '^VIX')}
    asof = last_completed_week_end(mk['^N225'].index)
    nk_w = mk['^N225'][mk['^N225'].index <= asof].resample('W-FRI').last().dropna()
    nk4 = (nk_w.iloc[-1] / nk_w.iloc[-5] - 1) * 100
    vix_w = mk['^VIX'][mk['^VIX'].index <= asof + pd.Timedelta(days=1)].resample('W-FRI').last().dropna()
    vix_off = vix_w.iloc[-1] < vix_w.tail(4).max() * 0.85

    mar = get_margin(asof)
    f = stock_features(prices, asof)
    f, mar_date = add_fundamentals(f, fin, mar, asof)
    f['name'] = f['ticker'].map(names)
    try:
        alerts = jpx_alerts.fetch()
    except Exception as e:
        print(f"⚠ JPXの上場廃止情報を取得できませんでした: {e}")
        alerts = pd.DataFrame(columns=['code', '種別', '日付', '理由'])
    flagged = set(alerts['code'] + '.T')
    u = f[f['universe'] & ~f['ticker'].isin(flagged)].copy()   # 上場廃止予定・監理銘柄などは買わない
    nm = lambda r: f"{r.ticker.replace('.T', '')} {str(r['name'])[:10]}"

    # ---- 銘柄リスト ----
    light = ~u['heavy_margin'].fillna(False)
    v = u[light & u['m_above'] & (u['E/P'] > 0)].nlargest(PICK_N, 'E/P')
    # 割安株と同じ銘柄は順張り側では外し、次の順位の銘柄を使う（同じ銘柄を二重に持たない）
    no_decline = ~(u['op_growth'] < 0)   # 減益予想の銘柄は順張りから外す（予想が無い銘柄は残す）
    m = u[light & no_decline & u['mom12_1'].notna() & ~u['ticker'].isin(v['ticker'])].nlargest(PICK_N, 'mom12_1')
    crash = nk4 <= CRASH_TH
    c = u.assign(r4rank=u['ret4'].rank(pct=True))
    c = c[c['r4rank'] <= LOSER_PCT].sort_values('ret4')
    k = int(np.ceil((asof - REBALANCE_ANCHOR).days / 91))
    next_reb = REBALANCE_ANCHOR + pd.Timedelta(weeks=13 * max(k, 0))
    is_reb = abs((asof - next_reb).days) <= 3
    if not is_reb and next_reb < asof:
        next_reb += pd.Timedelta(weeks=13)

    # ---- 保有銘柄の管理 ----
    cfg = load_config()
    pf = load_portfolio()
    hist = []
    act_sell, act_buy, notes = [], [], []
    pf = fill_entry_prices(pf, prices)
    # 上場廃止・そのおそれ（JPX）: 保有していれば知らせる
    for i, r in pf.iterrows():
        hit = alerts[alerts['code'] + '.T' == r['コード']]
        if hit.empty:
            continue
        a = hit.iloc[0]
        dt = f"{a['日付']:%Y-%m-%d}" if pd.notna(a['日付']) else ''
        if a['種別'] == '上場廃止予定' and jpx_alerts.is_tob(a['理由']):
            act_sell.append(f"・{show(r)}（{r['種類']}）⚠ TOB・買収などで上場廃止予定 {dt}（{a['理由']}）"
                            f"→ 公開買付けに応募するか、廃止日より前に市場で売却")
        elif a['種別'] in ('上場廃止予定', '監理・整理銘柄'):
            act_sell.append(f"・{show(r)}（{r['種類']}）🚨 {a['種別']} {dt} {a['理由']} → 翌営業日の寄り付きで売り")
            d = prices.get(r['コード'])
            hist.append(close_row(r, d['Close'].iloc[-1] if d is not None else np.nan, asof, a['種別']))
            pf = pf.drop(i)
        else:
            notes.append(f"⚠ {show(r)}（{r['種類']}）: 上場維持基準に適合していない（{a['種別']}）。次の入れ替えで外れやすい")
    # 急落銘柄: 利確・期限
    for i, r in pf[(pf['種類'] == '急落') & (pf['状態'] == '保有')].iterrows():
        d = prices.get(r['コード'])
        if d is None:
            continue
        after = d[d.index >= pd.Timestamp(r['買付日'])]
        if (after['High'] >= r['利確価格']).any():
            hist.append(close_row(r, r['利確価格'], after.index[(after['High'] >= r['利確価格']).argmax()], '利確'))
            notes.append(f"✅ 利確（指値が約定しているはず）: {show(r)} {r['利確価格']:,.0f}円")
            pf = pf.drop(i)
            continue
        dl = pd.Timestamp(r['売り期限'])
        if asof >= dl:
            act_sell.append(f"・{show(r)}（急落・保有期限切れ）→ 翌営業日の寄り付きで売り（期限 {dl:%m/%d} の大引けで売っていれば不要）")
            hist.append(close_row(r, d['Close'].iloc[-1], asof, '期限'))
            pf = pf.drop(i)
        elif (dl - asof).days <= 7:
            act_sell.append(f"・{show(r)}（急落・期限）→ {dl:%m/%d}(金) の大引けで売り（引け成行）")
    # 割安株・順張り: 入れ替え週（または保有がまだ無いとき）は、外れた銘柄を売る
    cands = []   # (種類, 行, 売り期限, 補足) を優先順位の順に並べる
    first_time = pf[pf['種類'].isin(['割安株', '順張り'])].empty
    if is_reb or first_time:
        for kind, picks in (('割安株', v), ('順張り', m)):
            held = pf[pf['種類'] == kind]
            new = set(picks['ticker'])
            for i, r in held.iterrows():
                if r['コード'] not in new:
                    act_sell.append(f"・{show(r)}（{kind}・入れ替えで外れた）→ 翌営業日の寄り付きで売り")
                    d = prices.get(r['コード'])
                    hist.append(close_row(r, d['Close'].iloc[-1] if d is not None else np.nan, asof, '入れ替え'))
                    pf = pf.drop(i)
        deadline = next_reb + pd.Timedelta(weeks=13) if is_reb else next_reb
        held_now = set(pf['コード'])
        mm = [r for _, r in m.iterrows() if r['ticker'] not in held_now]
        vv = [r for _, r in v.iterrows() if r['ticker'] not in held_now]
        normal = []
        for j in range(max(len(mm), len(vv))):   # 順張り1位 → 割安株1位 → 順張り2位 → … の順
            if j < len(mm):
                normal.append(('順張り', mm[j], deadline, f"順張り{j + 1}位・12か月{mm[j].mom12_1 * 100:+.0f}%"))
            if j < len(vv):
                normal.append(('割安株', vv[j], deadline, f"割安株{j + 1}位・PER{1 / vv[j]['E/P']:.1f}倍"))
    else:
        normal = []
    # 急落リバウンド: シグナルが出たら、下げた順に空き枠の分だけ（割安株・順張りより優先）
    crash_c = []
    if crash:
        held = set(pf.loc[pf['種類'] == '急落', 'コード'])
        room = cfg['急落の最大銘柄数'] - len(held)
        for j, (_, r) in enumerate(c[~c['ticker'].isin(held)].head(max(room, 0)).iterrows()):
            crash_c.append(('急落', r, asof + pd.Timedelta(weeks=HOLD_WEEKS), f"急落{j + 1}位・4週{r.ret4:+.1f}%"))
    cands = crash_c + normal

    # 資金に合わせて、優先順位の上から買う銘柄と株数を決める
    unit = int(cfg['単元株数'])
    budget = float(cfg['資金'] or 0)
    cur_price = lambda code: prices[code]['Close'].iloc[-1] if code in prices else np.nan
    if budget > 0:
        held_val = sum((r['株数'] if pd.notna(r.get('株数')) else 0) * cur_price(r['コード']) for _, r in pf.iterrows())
        available = budget - held_val
        n_slots = cfg['最大銘柄数']
    funding = []
    for rank, (kind, r, dl, info) in enumerate(cands, start=1):
        lot_cost = r.price * unit
        if budget > 0:
            slot = budget / (cfg['急落の最大銘柄数'] if kind == '急落' else n_slots)
            sh = plan_shares(r.price, slot, available, unit)
            if sh == 0 and kind == '急落':
                funding.append((r, lot_cost))
                sh = unit   # 急落銘柄は割安株・順張りを売ってでも買う
            if sh == 0:
                act_buy.append(f"　{rank}. {nm(r)} {r.price:,.0f}円（{info}）… 資金が足りないため見送り")
                continue
            available -= sh * r.price
            amount = f" {sh:,}株 ≒{sh * r.price / 10000:,.1f}万円"
        else:
            sh = np.nan
            amount = f"（{unit}株 ≒{lot_cost / 10000:,.1f}万円）"
        tail = f" → 約定したら買値の+{TARGET:.0%}に売り指値" if kind == '急落' else ''
        act_buy.append(f"{rank}. {nm(r)} {r.price:,.0f}円{amount}（{info}）{tail}")
        pf = add_row(pf, kind, r, asof, dl, sh)
    if funding:
        need = sum(x for _, x in funding)
        notes.append(f"💴 急落銘柄の資金 約{need / 10000:,.1f}万円は、割安株・順張りの保有を順位の低いものから売って用意")
    save_portfolio(pf, hist)

    # ---- 通知文 ----
    lines = [f"📊 **週次シグナル**（{asof:%Y-%m-%d} 終値時点）", '']
    lines.append("■ **今週やること**")
    if act_sell:
        lines.append("【売り】")
        lines += act_sell
    if act_buy:
        lines.append(f"【買い】翌営業日の寄り付きで（番号が優先順位。{'資金に合わせて上から' if budget > 0 else '買える分だけ上から'}）")
        lines += act_buy
    lines += notes
    if not (act_sell or act_buy or notes):
        lines.append("なし（そのまま保有）")
    lines.append('')
    lines.append(f"■ 急落リバウンド: 日経平均の4週騰落 {nk4:+.1f}%（発動 {CRASH_TH}%以下） / VIX {vix_w.iloc[-1]:.1f}"
                 f"{'（ピークアウト済み）' if vix_off else ''} → {'🚨シグナルあり' if crash else 'シグナルなし'}")
    lines.append(f"■ 割安株・順張りの入れ替え: {'🔁 今週' if is_reb else f'次回 {next_reb:%Y-%m-%d} の引け後'}")
    lines.append('')
    lines.append("■ 保有中（買値は約定した日の始値で記録。実際と違うときは portfolio.csv を編集）")
    for kind in ('急落', '割安株', '順張り'):
        h = pf[pf['種類'] == kind]
        if h.empty:
            continue
        lines.append(f"【{kind}】{len(h)}銘柄")
        for _, r in h.iterrows():
            d = prices.get(r['コード'])
            now = d['Close'].iloc[-1] if d is not None else np.nan
            pl = f"{(now / r['買値'] - 1) * 100:+.1f}%" if pd.notna(r['買値']) and r['買値'] > 0 else '買い予定'
            extra = f" 利確{r['利確価格']:,.0f}円 期限{pd.Timestamp(r['売り期限']):%m/%d}" if kind == '急落' and pd.notna(r['利確価格']) else ''
            lines.append(f"・{show(r)} {now:,.0f}円 {pl}{extra}")
    lines.append('')
    lines.append(f"（株価 {len(f)}銘柄・ユニバース {len(u)}銘柄 / 信用残 {mar_date:%Y-%m-%d}時点 / 処理 {time.time() - t0:.0f}秒）")
    send_discord('\n'.join(lines))


if __name__ == '__main__':
    main()
