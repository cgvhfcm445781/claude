"""
いろいろなテクニカル指標を「エントリーシグナル」として横並びで検証するスクリプト。

対象: MACD / ボリンジャーバンド / RSI / 一目均衡表 / DMI・ADX / OBV /
      ブラックショールズ的な指標（ボラティリティσで正規化した値動き、N(d2)による到達確率、ボラ収縮）

評価方法
- シグナル週の翌週始値で買い、4週後・8週後の終値までのリターンを測る（往復コストは引かない素の値）
- 「同じ週のユニバース全銘柄の平均」との差（超過リターン）で評価する
  → 相場全体の上げ下げの影響を除き、その指標で銘柄を選ぶ意味があるかを見る
- t値は週単位で計算し、保有期間の重なり（4週・8週）を補正
- 期間を3つ（〜2012 / 2013-2019 / 2020〜）に分け、地合い（平常／急落局面）別にも見る

ブラックショールズについて
- 本来はオプションの価格式なので、株そのものの売買シグナルにはならない（オプション価格のデータも無い）
- ここでは中核の考え方「株価は対数正規分布でボラティリティσに従って動く」だけを借りている
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
import rule_lab as R

PERIODS = [('〜2012', None, '2013-01-01'), ('2013-2019', '2013-01-01', '2020-01-01'), ('2020〜', '2020-01-01', None)]
CRASH_TH = -5   # 日経平均4週騰落がこれ以下を「急落局面」とする


def _ind(g):
    O, H, L, C, V = g['Open'], g['High'], g['Low'], g['Close'], g['Volume']
    f = pd.DataFrame(index=g.index)
    up = lambda a, b: (a.shift() <= b.shift()) & (a > b)   # a が b を上抜け
    # --- MACD(12,26,9)
    macd = C.ewm(span=12, adjust=False).mean() - C.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    hist = macd - sig
    f['macd_gc'] = up(macd, sig)
    f['macd_dc'] = up(sig, macd)   # デッドクロス（売りの条件として使う）
    f['macd_gc_below0'] = f['macd_gc'] & (macd < 0)
    f['macd_gc_above0'] = f['macd_gc'] & (macd > 0)
    f['macd_zero_up'] = up(macd, pd.Series(0.0, index=g.index))
    f['macd_hist_turn'] = (hist.shift(2) > hist.shift()) & (hist > hist.shift()) & (hist < 0)  # 下げ止まり
    # --- ボリンジャーバンド(20,2)
    m20, s20 = C.rolling(20).mean(), C.rolling(20).std()
    pb = (C - (m20 - 2 * s20)) / (4 * s20 + 1e-10)
    bw = 4 * s20 / m20
    f['bb_below'] = pb < 0
    f['bb_reenter'] = (pb.shift() < 0) & (pb >= 0)
    f['bb_squeeze_break'] = (bw.shift() <= bw.rolling(52).min().shift() * 1.1) & (pb > 1)
    f['bb_upper_break'] = (pb.shift() <= 1) & (pb > 1)
    # --- RSI(14)
    d = C.diff()
    rsi = 100 - 100 / (1 + d.clip(lower=0).rolling(14).mean() / ((-d.clip(upper=0)).rolling(14).mean() + 1e-10))
    f['rsi_below30'] = rsi < 30
    f['rsi_up30'] = (rsi.shift() < 30) & (rsi >= 30)
    f['rsi_above70'] = rsi > 70
    # --- 一目均衡表(9,26,52)
    tenkan = (H.rolling(9).max() + L.rolling(9).min()) / 2
    kijun = (H.rolling(26).max() + L.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((H.rolling(52).max() + L.rolling(52).min()) / 2).shift(26)
    cloud_top = np.maximum(span_a, span_b)
    f['ichi_cloud_break'] = (C.shift() <= cloud_top.shift()) & (C > cloud_top)
    f['ichi_tk_gc'] = up(tenkan, kijun)
    f['ichi_sanyaku'] = (C > cloud_top) & (tenkan > kijun) & (C > C.shift(26))  # 三役好転（簡易版）
    # --- DMI / ADX(14)
    upm, dnm = H.diff(), -L.diff()
    pdm = pd.Series(np.where((upm > dnm) & (upm > 0), upm, 0.0), index=g.index)
    mdm = pd.Series(np.where((dnm > upm) & (dnm > 0), dnm, 0.0), index=g.index)
    tr = pd.concat([H - L, (H - C.shift()).abs(), (L - C.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=1 / 14, adjust=False).mean() / (atr + 1e-10)
    mdi = 100 * mdm.ewm(alpha=1 / 14, adjust=False).mean() / (atr + 1e-10)
    adx = (100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10)).ewm(alpha=1 / 14, adjust=False).mean()
    f['dmi_gc'] = up(pdi, mdi)
    f['dmi_gc_adx20'] = f['dmi_gc'] & (adx > 20)
    f['adx_strong_up'] = (adx > 30) & (pdi > mdi) & (adx > adx.shift())
    # --- OBV
    obv = (np.sign(C.diff()).fillna(0) * V).cumsum()
    f['obv_div'] = (C < C.shift(8)) & (obv > obv.shift(8))   # 価格は下げ、出来高ベースでは買われている
    # --- ブラックショールズ的な指標
    lr = np.log(C / C.shift())
    sig_w = lr.rolling(26).std()                         # 週次ボラ
    f['vol_ann'] = sig_w * np.sqrt(52)
    z4 = np.log(C / C.shift(4)) / (sig_w.shift(4) * 2)   # 直近4週の値動きが何σか（σは動く前の時点で測る）
    f['bs_z4'] = z4
    f['bs_z4_le_m2'] = z4 <= -2
    f['bs_z4_le_m3'] = z4 <= -3
    f['bs_z4_ge_2'] = z4 >= 2
    hi52 = H.rolling(52).max()
    T = 13 / 52                                          # 13週後に52週高値を上回っている確率 N(d2)（ドリフト0）
    d2 = (np.log(C / hi52) - 0.5 * f['vol_ann'] ** 2 * T) / (f['vol_ann'] * np.sqrt(T) + 1e-10)
    f['bs_prob_hi'] = norm.cdf(d2)
    vr = lr.rolling(8).std() / (lr.rolling(52).std() + 1e-10)
    f['bs_vol_squeeze'] = vr <= 0.5                      # 短期ボラが長期ボラの半分以下（静かな局面）
    f['bs_vol_squeeze_up'] = f['bs_vol_squeeze'] & (C > C.rolling(13).mean())
    return f


def add_indicators(df):
    ind = df.groupby('ticker', group_keys=False)[['Open', 'High', 'Low', 'Close', 'Volume']].apply(_ind)
    df = pd.concat([df, ind], axis=1)
    g = df.groupby('ticker')
    for h in (4, 8):
        df[f'fwd{h}'] = g['Close'].shift(-h) / g['Open'].shift(-1) * 100 - 100
    for h in (4, 8):
        df[f'ex{h}'] = df[f'fwd{h}'] - df[f'fwd{h}'].where(df['universe']).groupby(df['week']).transform('mean')
    # N(d2) の週内順位（上位＝52週高値に届きやすい）
    df['bs_prob_hi_top'] = df['bs_prob_hi'].where(df['universe']).groupby(df['week']).rank(pct=True) >= 0.9
    df['bs_prob_hi_low'] = df['bs_prob_hi'].where(df['universe']).groupby(df['week']).rank(pct=True) <= 0.1
    return df


SIGNALS = {
    'MACD GC': 'macd_gc', 'MACD GC(0未満)': 'macd_gc_below0', 'MACD GC(0以上)': 'macd_gc_above0',
    'MACD 0ライン上抜け': 'macd_zero_up', 'MACDヒスト下げ止まり': 'macd_hist_turn',
    'BB -2σ割れ': 'bb_below', 'BB -2σから戻り': 'bb_reenter', 'BB スクイーズ後ブレイク': 'bb_squeeze_break',
    'BB +2σ上抜け': 'bb_upper_break',
    'RSI<30': 'rsi_below30', 'RSI 30上抜け': 'rsi_up30', 'RSI>70': 'rsi_above70',
    '一目 雲上抜け': 'ichi_cloud_break', '一目 転換線GC': 'ichi_tk_gc', '一目 三役好転': 'ichi_sanyaku',
    'DMI GC': 'dmi_gc', 'DMI GC(ADX>20)': 'dmi_gc_adx20', 'ADX強い上昇': 'adx_strong_up',
    'OBV強気ダイバージェンス': 'obv_div',
    'BS 4週-2σ以下': 'bs_z4_le_m2', 'BS 4週-3σ以下': 'bs_z4_le_m3', 'BS 4週+2σ以上': 'bs_z4_ge_2',
    'BS 高値到達確率 上位10%': 'bs_prob_hi_top', 'BS 高値到達確率 下位10%': 'bs_prob_hi_low',
    'BS ボラ収縮': 'bs_vol_squeeze', 'BS ボラ収縮&13週線上': 'bs_vol_squeeze_up',
    '(比較) 4週-15%以下': None,
}


def evaluate(df):
    u = df[df['universe']].dropna(subset=['ex8'])
    u = u.assign(crash=u['mk_ret4'] <= CRASH_TH)
    rows = []
    for name, col in SIGNALS.items():
        m = (u['ret4'] <= -15) if col is None else u[col].fillna(False).astype(bool)
        for regime, rm in (('平常', ~u['crash']), ('急落局面', u['crash'])):
            x = u[m & rm]
            if len(x) < 50:
                continue
            r = {'指標': name, '地合い': regime, '件数': len(x)}
            for h in (4, 8):
                w = x.groupby('week')[f'ex{h}'].mean()
                r[f't({h}w)'] = w.mean() / w.std() * np.sqrt(len(w) / h) if len(w) > 2 else np.nan
            for p, a, b in PERIODS:
                xx = x[(x['week'] >= (a or '1900')) & (x['week'] < (b or '2100'))]
                r[p] = xx['ex8'].mean()
            r['8週 素のリターン'] = x['fwd8'].mean()
            r['8週 勝率'] = (x['fwd8'] > 0).mean() * 100
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    df = pd.read_pickle(sys.argv[1]) if len(sys.argv) > 1 else R.add_features(R.load_panel())
    df = add_indicators(df)
    res = evaluate(df)
    pd.set_option('display.width', 220)
    pd.set_option('display.unicode.east_asian_width', True)
    for regime in ('平常', '急落局面'):
        t = res[res['地合い'] == regime].drop(columns='地合い').set_index('指標').sort_values('t(8w)', ascending=False)
        print(f"\n===== {regime}: 同じ週のユニバース平均に対する8週超過リターン(%)  ※t値は週単位・重なり補正 =====")
        print(t.round(2).to_string())
    res.to_csv('output/indicator_lab.csv', index=False, encoding='utf-8-sig')
    return df


if __name__ == '__main__':
    main()
