"""
週足データ(data/*.parquet)を使って、エントリー条件 × 決済条件 を自由に組み合わせて検証するスクリプト。

前提
- シグナルは週の終値で判定 → 翌週の始値でエントリー（先読みなし）
- 決済は「週中の安値で逆指値の損切り（窓開けは始値約定）」「高値で指値の利確」
  「終値ベースのトレーリング／指標による手仕舞い（翌週始値で約定）」「最大保有週数（その週の終値）」
- 1銘柄1ポジション（保有中の同銘柄シグナルは無視）
- 往復コスト COST を差し引く
- 期間は 探索(〜2021) / 検証(2022-2023) / テスト(2024〜) に分けて表示する
- 価格制限は「分割前の実際の株価」(Close × split_factor) で判定する（分割調整による先読みを避ける）
"""
import glob
import os
import numpy as np
import pandas as pd

DATA_DIR = 'data'
COST = 0.002                    # 往復0.2%（手数料＋スリッページの目安）
MIN_WEEKLY_VALUE = 250_000_000  # 直近4週平均の週間売買代金
PRICE_MIN, PRICE_MAX = 1000, 4000   # 実際の株価（INITIAL_CASH 40万円で100株買える範囲）
PERIODS = [('探索', None, '2022-01-01'), ('検証', '2022-01-01', '2024-01-01'), ('テスト', '2024-01-01', None)]


# ---------------- データと特徴量 ----------------
def load_panel():
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(DATA_DIR, 'weekly_prices_*.parquet')))])
    df = df.sort_values(['ticker', 'week']).reset_index(drop=True)
    for c in ['Open', 'High', 'Low', 'Close']:
        df[c] = df[c].astype('float64')
    meta = pd.read_parquet(os.path.join(DATA_DIR, 'meta.parquet'))
    return df.merge(meta, on='ticker', how='left')


def _rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / (dn + 1e-10))


def _per_ticker(g):
    O, H, L, C, V = g['Open'], g['High'], g['Low'], g['Close'], g['Volume']
    f = pd.DataFrame(index=g.index)
    for n in (5, 13, 26, 52):
        sma = C.rolling(n).mean()
        f[f'sma{n}'] = sma
        f[f'dev{n}'] = (C / sma - 1) * 100
        f[f'slope{n}'] = (sma / sma.shift(4) - 1) * 100
    for n in (1, 4, 13, 26, 52):
        f[f'ret{n}'] = (C / C.shift(n) - 1) * 100
    f['rsi14'] = _rsi(C, 14)
    f['rsi3'] = _rsi(C, 3)
    for k in (5, 9):
        lo, hi = L.rolling(k).min(), H.rolling(k).max()
        fk = 100 * (C - lo) / (hi - lo + 1e-10)
        d = fk.rolling(3).mean()
        sd = d.rolling(3).mean()
        f[f'stoD{k}'], f[f'stoSD{k}'] = d, sd
        f[f'gc{k}'] = (d.shift() <= sd.shift()) & (d > sd)
        f[f'dc{k}'] = (d.shift() >= sd.shift()) & (d < sd)
    tr = pd.concat([H - L, (H - C.shift()).abs(), (L - C.shift()).abs()], axis=1).max(axis=1)
    f['atr'] = tr.rolling(14).mean()
    f['atr_pct'] = f['atr'] / C * 100
    f['hi52'] = H.rolling(52).max()
    f['lo52'] = L.rolling(52).min()
    f['dist_hi52'] = (C / f['hi52'] - 1) * 100
    f['dist_lo52'] = (C / f['lo52'] - 1) * 100
    f['hi26_prev'] = H.shift().rolling(26).max()
    f['vol_ratio'] = V / (V.rolling(13).mean() + 1e-10)
    f['value4'] = g['Value'].rolling(4).mean()
    f['bull'] = C > O
    f['down_streak'] = (C < C.shift()).astype(int).groupby((C >= C.shift()).cumsum()).cumsum()
    return f


def add_features(df):
    feats = df.groupby('ticker', group_keys=False)[['Open', 'High', 'Low', 'Close', 'Volume', 'Value']].apply(_per_ticker)
    df = pd.concat([df, feats], axis=1)
    df['raw_price'] = df['Close'] * df['split_factor']
    # 地合い（TOPIX連動ETF 1306）
    mk = pd.read_parquet(os.path.join(DATA_DIR, 'market_weekly.parquet'))
    tp = mk[mk['symbol'] == '1306.T'].set_index('week')['Close'].astype(float)
    m = pd.DataFrame({'mk_dev26': (tp / tp.rolling(26).mean() - 1) * 100,
                      'mk_slope26': (tp.rolling(26).mean() / tp.rolling(26).mean().shift(4) - 1) * 100,
                      'mk_ret4': (tp / tp.shift(4) - 1) * 100})
    df = df.merge(m, left_on='week', right_index=True, how='left')
    # 同じ週の中での順位（クロスセクション）
    df['ret13_rank'] = df.groupby('week')['ret13'].rank(pct=True)
    df['ret52_rank'] = df.groupby('week')['ret52'].rank(pct=True)
    df['universe'] = (df['value4'] >= MIN_WEEKLY_VALUE) & df['raw_price'].between(PRICE_MIN, PRICE_MAX)
    return df


# ---------------- 決済シミュレーション ----------------
def simulate(df, signal, stop=None, target=None, trail=None, trail_atr=None, stop_atr=None,
             exit_col=None, max_hold=13, one_position=True):
    """
    stop/target/trail: 率（0.08 = 8%）。stop_atr/trail_atr: エントリー時ATRの倍数
    exit_col: 週の終値で True になったら翌週始値で決済する列名（例 'dc5', 'below_sma13'）
    max_hold: 保有週数の上限。その週の終値で決済
    """
    O, H, L, C = (df[c].to_numpy() for c in ['Open', 'High', 'Low', 'Close'])
    atr = df['atr'].to_numpy()
    tick = df['ticker'].to_numpy()
    xs = df[exit_col].to_numpy(dtype=bool) if exit_col else None
    sig_idx = np.flatnonzero(signal.to_numpy(dtype=bool) & df['universe'].to_numpy(dtype=bool))
    n = len(df)
    trades = []
    busy_until = {}
    for t in sig_idx:
        e = t + 1
        if e >= n or tick[e] != tick[t]:
            continue
        if one_position and busy_until.get(tick[t], -1) > t:
            continue
        entry = O[e]
        if not np.isfinite(entry) or entry <= 0:
            continue
        sp = entry * (1 - stop) if stop else -np.inf
        if stop_atr and np.isfinite(atr[t]):
            sp = max(sp, entry - stop_atr * atr[t])
        tp = entry * (1 + target) if target else np.inf
        peak = entry
        px, reason, k = None, None, e
        while k < n and tick[k] == tick[t]:
            # 週中: 損切り（逆指値）→ 利確（指値）の順に判定（同じ週なら損切り優先＝保守的）
            if L[k] <= sp:
                px, reason = min(O[k], sp), 'stop'
            elif H[k] >= tp:
                px, reason = max(O[k], tp), 'target'
            if reason:
                break
            peak = max(peak, C[k])
            held = k - t
            nxt = k + 1 < n and tick[k + 1] == tick[t]
            close_exit = None
            if trail and C[k] <= peak * (1 - trail) and C[k] > entry:
                close_exit = 'trail'
            elif trail_atr and np.isfinite(atr[t]) and C[k] <= peak - trail_atr * atr[t]:
                close_exit = 'trail_atr'
            elif xs is not None and xs[k]:
                close_exit = 'signal'
            if close_exit and nxt:
                px, reason, k = O[k + 1], close_exit, k + 1
                break
            if held >= max_hold:
                px, reason = C[k], 'time'
                break
            k += 1
        if reason is None:
            continue  # データ末尾までに決済されていない
        busy_until[tick[t]] = k
        trades.append((t, e, k, entry, px, reason))
    tr = pd.DataFrame(trades, columns=['sig', 'ent', 'ex', 'entry', 'exit', 'reason'])
    tr['ret'] = (tr['exit'] / tr['entry'] - 1 - COST) * 100
    tr['weeks'] = tr['ex'] - tr['sig']
    tr['week'] = df['week'].to_numpy()[tr['sig']]
    tr['ticker'] = tick[tr['sig']]
    return tr


# ---------------- 集計 ----------------
def summarize(tr, label=''):
    out = {'rule': label}
    for name, a, b in PERIODS:
        m = pd.Series(True, index=tr.index)
        if a:
            m &= tr['week'] >= a
        if b:
            m &= tr['week'] < b
        r = tr.loc[m, 'ret']
        pf = r[r > 0].sum() / -r[r < 0].sum() if (r < 0).any() else np.nan
        out[name] = f"{len(r)}件 {r.mean():+.2f}% 勝{(r > 0).mean() * 100:.0f}% PF{pf:.2f}" if len(r) else '-'
    y = tr.groupby(tr['week'].dt.year)['ret'].mean()
    out['プラスの年'] = f"{(y > 0).sum()}/{len(y)}"
    out['平均保有'] = round(tr['weeks'].mean(), 1)
    return out


def portfolio(tr, df, max_pos=10, score=None):
    """同時保有 max_pos 銘柄までの等金額ポートフォリオ（空き枠に入る順は score の高い順）。週次の資産曲線を返す"""
    weeks = np.sort(df['week'].unique())
    t = tr.copy()
    t['ent_w'] = df['week'].to_numpy()[t['ent']]
    t['ex_w'] = df['week'].to_numpy()[t['ex']]
    t['score'] = df[score].to_numpy()[t['sig']] if score else 0
    t = t.sort_values(['ent_w', 'score'], ascending=[True, False])
    open_until, taken = [], []
    for w, grp in t.groupby('ent_w', sort=True):
        open_until = [x for x in open_until if x > w]
        free = max_pos - len(open_until)
        if free <= 0:
            continue
        pick = grp.head(free)
        taken.append(pick)
        open_until += list(pick['ex_w'])
    p = pd.concat(taken)
    # 決済週にリターンを計上する簡易版（1枠 = 資産の 1/max_pos）
    wk = p.groupby('ex_w')['ret'].sum() / max_pos / 100
    eq = (1 + wk.reindex(weeks, fill_value=0)).cumprod()
    return p, eq


def portfolio_mtm(tr, df, max_pos=5, score=None, idle=None):
    """
    週次で時価評価するポートフォリオ。資産を max_pos 枠に等分し、空いている枠は
    idle（週次リターンのSeries、例: TOPIX ETF）で運用する。None なら現金。
    戻り値: (採用トレード, 資産曲線, 個別株に投資していた割合の週次Series)
    """
    weeks = np.sort(df['week'].unique())
    O, C = df['Open'].to_numpy(), df['Close'].to_numpy()
    wk_arr = df['week'].to_numpy()
    t = tr.copy()
    t['ent_w'] = wk_arr[t['ent']]
    t['ex_w'] = wk_arr[t['ex']]
    t['score'] = df[score].to_numpy()[t['sig']] if score else 0
    t = t.sort_values(['ent_w', 'score'], ascending=[True, False])
    open_until, taken = [], []
    for w, grp in t.groupby('ent_w', sort=True):
        open_until = [x for x in open_until if x > w]
        free = max_pos - len(open_until)
        if free > 0:
            pick = grp.head(free)
            taken.append(pick)
            open_until += list(pick['ex_w'])
    p = pd.concat(taken) if taken else t.iloc[:0]
    contrib = pd.Series(0.0, index=weeks)   # 個別株の週次リターン寄与（枠あたり）
    used = pd.Series(0.0, index=weeks)      # 使用中の枠数
    for e, x, entry, px in zip(p['ent'], p['ex'], p['entry'], p['exit']):
        prev = entry
        for k in range(e, x + 1):
            cur = px * (1 - COST) if k == x else C[k]
            contrib[wk_arr[k]] += cur / prev - 1
            used[wk_arr[k]] += 1
            prev = cur
    idle_r = idle.reindex(weeks).fillna(0) if idle is not None else 0
    port_r = (contrib + (max_pos - used).clip(lower=0) * idle_r) / max_pos
    return p, (1 + port_r).cumprod(), used / max_pos


def perf(eq):
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    return f"CAGR {cagr * 100:+.1f}%  最大DD {dd * 100:.1f}%  最終 {eq.iloc[-1]:.2f}倍"
