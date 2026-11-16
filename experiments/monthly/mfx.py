"""月頻總經基本面 × 長視野匯率預測：特徵、模型、遞迴樣本外預測與檢定。

時間約定（全部以整數月份索引 t 表示，t=0 為 1986-01）：
  - 預測起點 t：已觀察到 t 月月初的匯率 s_t = log(外幣/美元)
  - 目標：y_{t,h} = s_{t+h} - s_t（h 個月後的 log 變動，跨幣別尺度可比）
  - 特徵只用 t 月月初以前已發布的資料，延遲見 RELEASE_LAG
  - 估計：遞迴擴張視窗；起點 t 的訓練樣本只含 k + h <= t 的 (x_k, y_{k,h})，
    也就是目標在 t 以前已經實現
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "monthly"

CURRENCIES = ["australia", "canada", "denmark", "eurozone", "japan", "south_korea",
              "norway", "sweden", "switzerland", "taiwan", "uk"]

# 以「參考月份」計的發布延遲（月）。起點 t（t 月月初）只能用參考月份 <= t - lag 的值。
#   CPI：多數國家在次月中旬發布 → t 月月初最新的是 t-2 月；
#        澳洲 CPI 是季資料（月值由季資料展開），季末後約 4 週發布 → 保守取 4
#   短率：月度序列不確定是月平均或月底值，一律延後 1 個月（t-1 月的值在 t 月月初已知）
#   貨幣量、IP：韓國 M1、英國與瑞典 IP 約在參考月份後 6 週發布 → 一律取 t-3
RELEASE_LAG = {"cpi": 2, "cpi_australia": 4, "rate": 1, "money": 3, "ip": 3}

# 貨幣模型／含產出缺口 Taylor rule 的子集：貨幣量與 IP 水準值都可得的國家
MACRO_SUBSET = ["japan", "south_korea", "taiwan", "uk", "sweden"]
# 貨幣量的定義斷點：該月的 log 變動以前 12 個月變動的中位數取代（只用過去資料）
MONEY_BREAKS = {"us": ["2020-05"]}  # 美國 M1 重新定義，納入儲蓄存款
GAP_MIN_OBS = 36  # 遞迴二次趨勢至少需要的觀測數

MIN_TRAIN = 60  # 每個貨幣至少 60 筆已實現的訓練樣本才開始預測


# ---------------------------------------------------------------- 資料與特徵
def load_monthly(data_dir=DATA):
    fx = pd.read_csv(Path(data_dir) / "fx.csv", index_col="month")
    cpi = pd.read_csv(Path(data_dir) / "cpi.csv", index_col="month")
    rate = pd.read_csv(Path(data_dir) / "rate.csv", index_col="month")
    months = fx.index.to_numpy()
    return months, fx, cpi, rate


def _shift(a, lag):
    """a[..., t] -> a[..., t-lag]（前面補 NaN）。"""
    out = np.full_like(a, np.nan, dtype=float)
    if lag == 0:
        return a.astype(float).copy()
    out[..., lag:] = a[..., :-lag]
    return out


def build_features(fx, cpi, rate, currencies=CURRENCIES):
    """回傳 dict of [N, T] 陣列：s, z, infl_diff, rdiff。

    z         = (p_i - p_us) - s ：PPP 偏離（> 0 表示外幣相對 PPP 偏強，預期 s 上升）
    infl_diff = 年增率差（log 單位，per year），i 國減美國
    rdiff     = 短率差（小數，per year），i 國減美國
    """
    s = np.log(fx[currencies].to_numpy().T)
    lc_us = RELEASE_LAG["cpi"]
    p_us = _shift(np.log(cpi["us"].to_numpy()), lc_us)
    infl_us = p_us - _shift(p_us, 12)
    r_us = _shift(rate["us"].to_numpy(), RELEASE_LAG["rate"])
    z, infl, rd = [], [], []
    for c in currencies:
        lc = RELEASE_LAG.get(f"cpi_{c}", RELEASE_LAG["cpi"])
        p = _shift(np.log(cpi[c].to_numpy()), lc)
        z.append(p - p_us)
        infl.append((p - _shift(p, 12)) - infl_us)
        rd.append((_shift(rate[c].to_numpy(), RELEASE_LAG["rate"]) - r_us) / 100.0)
    pdiff = np.array(z)
    return {"s": s, "z": pdiff - s, "infl_diff": np.array(infl), "rdiff": np.array(rd)}


def load_macro(data_dir=DATA):
    money = pd.read_csv(Path(data_dir) / "money.csv", index_col="month")
    ip = pd.read_csv(Path(data_dir) / "ip.csv", index_col="month")
    return money, ip


def splice_breaks(logm, months, breaks):
    """把斷點月份的 log 變動換成前 12 個月變動的中位數，之後的水準整體平移。"""
    out = logm.copy()
    for b in breaks:
        k = int(np.where(months == b)[0][0])
        d = np.diff(out)
        prev = d[max(0, k - 13):k - 1]
        prev = prev[~np.isnan(prev)]
        jump = d[k - 1] - np.median(prev)
        out[k:] = out[k:] - jump
    return out


def trailing_mean(a, w):
    """a[t] -> mean(a[t-w+1..t])，任何缺值則為 NaN（只用過去資料，用來去除季節性）。"""
    out = np.full_like(a, np.nan, dtype=float)
    for t in range(w - 1, len(a)):
        seg = a[t - w + 1:t + 1]
        if not np.isnan(seg).any():
            out[t] = seg.mean()
    return out


def recursive_gap(logy, min_obs=GAP_MIN_OBS):
    """即時產出缺口：在每個 k，用 k 以前（含 k）所有可得值配適二次時間趨勢，取 k 的殘差。
    輸入應已依發布延遲位移，因此 gap[k] 只用到參考月份 <= k-lag 的 IP。"""
    out = np.full_like(logy, np.nan, dtype=float)
    idx = np.where(~np.isnan(logy))[0]
    for j, k in enumerate(idx):
        if j + 1 < min_obs:
            continue
        x = idx[:j + 1].astype(float)
        X = np.column_stack([np.ones_like(x), x, x ** 2])
        beta = np.linalg.lstsq(X, logy[idx[:j + 1]], rcond=None)[0]
        out[k] = logy[k] - np.array([1.0, k, k ** 2]) @ beta
    return out


def build_macro_features(fx, cpi, rate, money, ip, months, currencies=MACRO_SUBSET):
    """子集特徵：build_features 的全部欄位，另加
    z_mon    = [(m - m_us) - (y - y_us)] - s ：彈性價格貨幣模型的基本面偏離（Mark 1995）
    gap_diff = 產出缺口差（即時二次趨勢殘差），i 國減美國
    m 為 log 貨幣量的 12 個月移動平均（去季節性），y 為 log IP。"""
    F = build_features(fx, cpi, rate, currencies)
    lm, ly = RELEASE_LAG["money"], RELEASE_LAG["ip"]

    def m_series(c):
        lm_raw = np.log(money[c].to_numpy(dtype=float))
        if c in MONEY_BREAKS:
            lm_raw = splice_breaks(lm_raw, months, MONEY_BREAKS[c])
        return _shift(trailing_mean(lm_raw, 12), lm)

    def y_series(c):
        return _shift(np.log(ip[c].to_numpy(dtype=float)), ly)

    m_us, y_us = m_series("us"), y_series("us")
    gap_us = recursive_gap(y_us)
    zm, gd = [], []
    for i, c in enumerate(currencies):
        y_i = y_series(c)
        zm.append((m_series(c) - m_us) - (y_i - y_us) - F["s"][i])
        gd.append(recursive_gap(y_i) - gap_us)
    F["z_mon"] = np.array(zm)
    F["gap_diff"] = np.array(gd)
    return F


def macro_models():
    return [
        RandomWalk(),
        PPPCalibrated(36, False),
        PanelFE("taylor_panel", ["infl_diff", "rdiff"]),
        OLSEach("mon_each", ["z_mon"]),
        PanelFE("mon_panel", ["z_mon"]),
        OLSEach("taylor_gap_each", ["infl_diff", "gap_diff", "rdiff"]),
        PanelFE("taylor_gap_panel", ["infl_diff", "gap_diff", "rdiff"]),
    ]


def targets(s, h):
    """y[:, t] = s[:, t+h] - s[:, t]；最後 h 期為 NaN。"""
    y = np.full_like(s, np.nan)
    y[:, :-h] = s[:, h:] - s[:, :-h]
    return y


# ---------------------------------------------------------------- 模型
# 每個模型：forecast(F, y, t, h) -> [N] 預測（無法預測者為 NaN）
# F 為特徵 dict，y 為 targets(s, h)；只能使用 y[:, :t-h+1]（k <= t-h 已實現）

def _train_mask(F, y, t, h, cols):
    """k <= t-h 且 y 與所需特徵皆非缺值的訓練樣本 mask，[N, T]。"""
    T = y.shape[1]
    m = np.zeros_like(y, dtype=bool)
    m[:, :max(0, t - h + 1)] = True
    m &= ~np.isnan(y)
    for c in cols:
        m &= ~np.isnan(F[c])
    return m & (np.arange(T) <= t - h)[None, :]


def _current_ok(F, t, cols):
    ok = np.ones(F["s"].shape[0], dtype=bool)
    for c in cols:
        ok &= ~np.isnan(F[c][:, t])
    return ok


class RandomWalk:
    name, cols, nested = "rw", [], True

    def forecast(self, F, y, t, h):
        out = np.zeros(F["s"].shape[0])
        out[np.isnan(F["s"][:, t])] = np.nan
        return out


class Drift:
    """各貨幣過去 h 月變動的遞迴平均。"""
    name, cols, nested = "drift", [], True

    def forecast(self, F, y, t, h):
        m = _train_mask(F, y, t, h, ["s"])
        out = np.full(y.shape[0], np.nan)
        for i in range(y.shape[0]):
            if m[i].sum() >= MIN_TRAIN:
                out[i] = y[i, m[i]].mean()
        return out


class OLSEach:
    """各貨幣分別 OLS：y = a_i + b_i' x。"""
    nested = True

    def __init__(self, name, cols):
        self.name, self.cols = name, cols

    def forecast(self, F, y, t, h):
        m = _train_mask(F, y, t, h, self.cols)
        ok = _current_ok(F, t, self.cols)
        out = np.full(y.shape[0], np.nan)
        for i in range(y.shape[0]):
            if not ok[i] or m[i].sum() < MIN_TRAIN:
                continue
            X = np.column_stack([np.ones(m[i].sum())] + [F[c][i, m[i]] for c in self.cols])
            beta = np.linalg.lstsq(X, y[i, m[i]], rcond=None)[0]
            x0 = np.r_[1.0, [F[c][i, t] for c in self.cols]]
            out[i] = x0 @ beta
        return out


class PanelFE:
    """面板固定效果：y = a_i + b' x（斜率跨貨幣共用，Mark-Sul 2001 式）。"""
    nested = True

    def __init__(self, name, cols):
        self.name, self.cols = name, cols

    def forecast(self, F, y, t, h):
        m = _train_mask(F, y, t, h, self.cols)
        ok = _current_ok(F, t, self.cols)
        N = y.shape[0]
        Xs, ys, means = [], [], {}
        for i in range(N):
            if m[i].sum() < 12:
                continue
            Xi = np.column_stack([F[c][i, m[i]] for c in self.cols])
            yi = y[i, m[i]]
            means[i] = (Xi.mean(0), yi.mean())
            Xs.append(Xi - Xi.mean(0))
            ys.append(yi - yi.mean())
        out = np.full(N, np.nan)
        if not Xs:
            return out
        b = np.linalg.lstsq(np.vstack(Xs), np.concatenate(ys), rcond=None)[0]
        for i in range(N):
            if ok[i] and i in means and m[i].sum() >= MIN_TRAIN:
                xm, ym = means[i]
                x0 = np.array([F[c][i, t] for c in self.cols])
                out[i] = ym + (x0 - xm) @ b
        return out


class PPPCalibrated:
    """不估計參數的 PPP 回復（Ca' Zorzi, Muck & Rubaszek 2016 的精神）：
    實質匯率偏離以半衰期 hl 個月回復，名目匯率另加通膨差（相對 PPP）。

    ŷ = (1 - ρ^h)(z_t - z̄_t) + with_infl · (h/12) · infl_diff_t，ρ = 0.5^(1/hl)
    z̄_t = 該貨幣到 t 為止 z 的平均（z 在 t 已知，不涉及未來資訊）
    """
    nested = False

    def __init__(self, hl=36, with_infl=True):
        self.hl, self.with_infl = hl, with_infl
        self.name = f"ppp_calib_hl{hl}" + ("" if with_infl else "_noinfl")
        self.cols = ["z", "infl_diff"] if with_infl else ["z"]

    def forecast(self, F, y, t, h):
        rho = 0.5 ** (1.0 / self.hl)
        z = F["z"][:, :t + 1]
        out = np.full(y.shape[0], np.nan)
        for i in range(y.shape[0]):
            zi = z[i][~np.isnan(z[i])]
            if len(zi) < MIN_TRAIN or np.isnan(F["z"][i, t]):
                continue
            f = (1 - rho ** h) * (F["z"][i, t] - zi.mean())
            if self.with_infl:
                if np.isnan(F["infl_diff"][i, t]):
                    continue
                f += h / 12.0 * F["infl_diff"][i, t]
            out[i] = f
        return out


class UIPTheory:
    """未拋補利率平價的理論值：預期外幣貶值 = 利差 × h/12（不估計）。"""
    name, cols, nested = "uip_theory", ["rdiff"], False

    def forecast(self, F, y, t, h):
        return F["rdiff"][:, t] * h / 12.0


class Combo:
    """數個模型預測的等權平均（只在所有成員都有預測時才給值）。"""
    nested = True

    def __init__(self, name, members):
        self.name, self.members = name, members
        self.cols = sorted({c for m in members for c in m.cols})

    def forecast(self, F, y, t, h):
        return np.mean([m.forecast(F, y, t, h) for m in self.members], axis=0)


def default_models():
    ppp_e = OLSEach("ppp_each", ["z"])
    ppp_p = PanelFE("ppp_panel", ["z"])
    uip_e = OLSEach("uip_each", ["rdiff"])
    tay_e = OLSEach("taylor_each", ["infl_diff", "rdiff"])
    tay_p = PanelFE("taylor_panel", ["infl_diff", "rdiff"])
    all_p = PanelFE("ppp_taylor_panel", ["z", "infl_diff", "rdiff"])
    return [
        RandomWalk(), Drift(),
        ppp_e, ppp_p, PPPCalibrated(36, True), PPPCalibrated(36, False),
        PPPCalibrated(60, True), PPPCalibrated(60, False),
        UIPTheory(), uip_e, tay_e, tay_p, all_p,
        Combo("combo", [ppp_p, tay_p, uip_e]),
    ]


# ---------------------------------------------------------------- 回測
def run_backtest(F, horizons, models, origin_start, origin_end=None):
    """回傳長表：model, h, nation, t, y, yhat。"""
    rows = []
    T = F["s"].shape[1]
    for h in horizons:
        y = targets(F["s"], h)
        last = T - 1 - h if origin_end is None else min(origin_end, T - 1 - h)
        for t in range(origin_start, last + 1):
            for mdl in models:
                f = mdl.forecast(F, y, t, h)
                for i in range(len(f)):
                    if not np.isnan(f[i]) and not np.isnan(y[i, t]):
                        rows.append((mdl.name, h, i, t, y[i, t], f[i]))
    return pd.DataFrame(rows, columns=["model", "h", "nation", "t", "y", "yhat"])


def common_sample(df, models):
    """只保留所有指定模型都有預測的 (h, nation, t)，確保比較公平。"""
    d = df[df.model.isin(models)]
    cnt = d.groupby(["h", "nation", "t"]).model.nunique()
    keep = cnt[cnt == len(models)].index
    return d.set_index(["h", "nation", "t"]).loc[keep].reset_index()


# ---------------------------------------------------------------- 檢定
def nw_tstat(x, lag):
    """均值的 Newey-West t 統計量（Bartlett kernel）。"""
    x = np.asarray(x, float)
    T = len(x)
    xc = x - x.mean()
    v = (xc ** 2).mean()
    for k in range(1, min(lag, T - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * (xc[k:] * xc[:-k]).mean()
    if v <= 0:
        return 0.0
    return x.mean() / np.sqrt(v / T)


def clark_west(y, yhat, lag):
    """Clark-West (2007) MSPE-adjusted：H0 RW（預測 0）與模型等準，H1 模型較好。
    f_t = y² - [(y - ŷ)² - ŷ²]；單尾 p 值（標準常態）。"""
    f = y ** 2 - ((y - yhat) ** 2 - yhat ** 2)
    t = nw_tstat(f, lag)
    return t, float(stats.norm.sf(t))


def dm_mse(y, yhat, lag):
    """DM 檢定（MSE 損失）vs RW：d = y² - (y-ŷ)²，> 0 表示模型較好；單尾 p 值。"""
    d = y ** 2 - (y - yhat) ** 2
    t = nw_tstat(d, lag)
    return t, float(stats.norm.sf(t))


def evaluate(df, names, lag_fn=lambda h: h):
    """每 (model, h) 的面板與各貨幣結果。

    面板檢定：先在每個起點對貨幣取平均的損失差，再對時間序列做 NW（處理同期跨貨幣相關
    與重疊視野的序列相關）。
    """
    panel_rows, nation_rows = [], []
    for (model, h), g in df[df.model != "rw"].groupby(["model", "h"]):
        lag = lag_fn(h)
        g = g.assign(e_rw=g.y ** 2, e_m=(g.y - g.yhat) ** 2,
                     cw=g.y ** 2 - ((g.y - g.yhat) ** 2 - g.yhat ** 2))
        by_t = g.groupby("t")[["e_rw", "e_m", "cw"]].mean().sort_index()
        d = by_t.e_rw - by_t.e_m
        panel_rows.append({
            "model": model, "h": h, "n_obs": len(g), "n_origins": len(by_t),
            "mspe_ratio": g.e_m.sum() / g.e_rw.sum(),
            "rmse_ratio": np.sqrt(g.e_m.sum() / g.e_rw.sum()),
            "dm_t": nw_tstat(d.values, lag), "cw_t": nw_tstat(by_t.cw.values, lag),
            "hit_rate": float((np.sign(g.yhat) == np.sign(g.y)).mean()),
            "n_nation_better": 0,
        })
        nb = 0
        for i, gi in g.groupby("nation"):
            gi = gi.sort_values("t")
            r = gi.e_m.sum() / gi.e_rw.sum()
            cw_t, cw_p = clark_west(gi.y.values, gi.yhat.values, lag)
            dm_t, dm_p = dm_mse(gi.y.values, gi.yhat.values, lag)
            nb += r < 1
            nation_rows.append({"model": model, "h": h, "nation": names[i], "n": len(gi),
                                "rmse_ratio": np.sqrt(r), "cw_t": cw_t, "cw_p": cw_p,
                                "dm_t": dm_t, "dm_p": dm_p,
                                "hit_rate": float((np.sign(gi.yhat) == np.sign(gi.y)).mean())})
        panel_rows[-1]["n_nation_better"] = int(nb)
    P = pd.DataFrame(panel_rows)
    P["dm_p"] = stats.norm.sf(P.dm_t)
    P["cw_p"] = stats.norm.sf(P.cw_t)
    return P, pd.DataFrame(nation_rows)
