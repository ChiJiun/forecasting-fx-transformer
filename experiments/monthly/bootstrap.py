"""施加虛無假設的 bootstrap：檢查 PPP 模型的長視野可預測性是否只是小樣本假象。

H0：匯率是隨機漫步，與物價差無關（Mark 1995 的 no-predictability null）。
DGP：固定實際的物價差 pdiff = p - p*；匯率報酬以平穩區塊 bootstrap 從實際月報酬重抽
     （11 個貨幣同一組時間索引，保留跨貨幣相關與波動叢聚），累加成 s*；z* = pdiff - s*。
每個複本重跑整個遞迴估計，在與實際評估相同的 (h, nation, t) 上計算面板 RMSE 比與 CW t，
bootstrap p 值 = 複本統計量至少和實際一樣好的比例。

為了速度，PPP 模型以累積和向量化實作（fast_*），測試確認與 mfx 中的原版逐點一致。
"""
import numpy as np

import mfx


def _expanding_stats(z, y, h, min_obs_fe=12):
    """各貨幣、各起點 t 的訓練樣本（k <= t-h，z 與 y 皆非缺值）累積和。

    回傳 n, Sz, Sy, Szz, Szy，形狀 [N, T]；第 t 欄 = 起點 t 可用的訓練樣本統計量。
    """
    ok = ~np.isnan(z) & ~np.isnan(y)
    zz = np.where(ok, z, 0.0)
    yy = np.where(ok, y, 0.0)
    cs = lambda a: np.cumsum(a, axis=1)  # noqa: E731
    stats_k = [cs(ok.astype(float)), cs(zz), cs(yy), cs(zz * zz), cs(zz * yy)]
    out = []
    for a in stats_k:
        b = np.zeros_like(a)
        b[:, h:] = a[:, :-h]  # 起點 t 用到 k <= t-h
        out.append(b)
    return out


def fast_forecasts(z, y, h, hl_list=(36, 60)):
    """回傳 dict model -> [N, T] 預測（NaN 表示不可預測），涵蓋：
    ppp_each、ppp_panel、ppp_calib_hl{hl}_noinfl。"""
    N, T = z.shape
    n, Sz, Sy, Szz, Szy = _expanding_stats(z, y, h)
    with np.errstate(invalid="ignore", divide="ignore"):
        zbar, ybar = Sz / n, Sy / n
        cxx = Szz - Sz * Sz / n
        cxy = Szy - Sz * Sy / n
        b_each = cxy / cxx
    zt = z
    ok_now = ~np.isnan(zt) & (n >= mfx.MIN_TRAIN)
    out = {}
    f = ybar + b_each * (zt - zbar)
    out["ppp_each"] = np.where(ok_now, f, np.nan)

    inc = n >= 12
    num = np.where(inc, cxy, 0.0).sum(0)
    den = np.where(inc, cxx, 0.0).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        b_pan = num / den
    f = ybar + b_pan[None, :] * (zt - zbar)
    out["ppp_panel"] = np.where(ok_now, f, np.nan)

    # 校準 PPP：z̄_t 為到 t 為止（含 t）所有可得 z 的平均
    okz = ~np.isnan(z)
    cnt = np.cumsum(okz, axis=1)
    mean_z = np.cumsum(np.where(okz, z, 0.0), axis=1) / np.maximum(cnt, 1)
    for hl in hl_list:
        rho = 0.5 ** (1.0 / hl)
        f = (1 - rho ** h) * (z - mean_z)
        out[f"ppp_calib_hl{hl}_noinfl"] = np.where(okz & (cnt >= mfx.MIN_TRAIN), f, np.nan)
    return out


def panel_stats(y, yhat, mask, lag):
    """在 mask 上計算面板 RMSE 比、面板 CW t 與面板 DM t（先跨貨幣平均再做 NW）。"""
    e_rw = np.where(mask, y ** 2, 0.0)
    e_m = np.where(mask, (y - yhat) ** 2, 0.0)
    cw = np.where(mask, y ** 2 - ((y - yhat) ** 2 - yhat ** 2), 0.0)
    ratio = np.sqrt(e_m.sum() / e_rw.sum())
    cnt = mask.sum(0)
    tt = cnt > 0
    cw_t = mfx.nw_tstat(cw.sum(0)[tt] / cnt[tt], lag)
    dm_t = mfx.nw_tstat((e_rw - e_m).sum(0)[tt] / cnt[tt], lag)
    return ratio, cw_t, dm_t


def stationary_bootstrap_index(n_out, pool_len, mean_block, rng):
    """Politis-Romano 平穩 bootstrap：產生 n_out 個 [0, pool_len) 內的索引。
    區塊起點在 pool 內均勻抽取，區塊在 pool 尾端循環接回開頭，每個位置被抽中的機率相同。"""
    idx = np.empty(n_out, dtype=int)
    idx[0] = rng.integers(pool_len)
    p = 1.0 / mean_block
    for k in range(1, n_out):
        idx[k] = rng.integers(pool_len) if rng.random() < p else (idx[k - 1] + 1) % pool_len
    return idx


def simulate_s(s, rng, mean_block=12, demean=False):
    """以實際月報酬（各貨幣同一時間索引）重抽，從各貨幣第一個觀測值累加成隨機漫步。
    demean=True 時先扣除各貨幣的報酬平均（無漂移的隨機漫步）。"""
    N, T = s.shape
    r = np.diff(s, axis=1)  # [N, T-1]，可能含 NaN（歐元區 1994 年前）
    valid_t = np.where(~np.isnan(r).any(0))[0]  # 全部貨幣都有報酬的月份
    pool = r[:, valid_t]
    if demean:
        pool = pool - pool.mean(axis=1, keepdims=True)
    k = stationary_bootstrap_index(T - 1, pool.shape[1], mean_block, rng)
    rs = pool[:, k]
    out = np.full_like(s, np.nan)
    for i in range(N):
        first = np.where(~np.isnan(s[i]))[0][0]
        out[i, first] = s[i, first]
        out[i, first + 1:] = s[i, first] + np.cumsum(rs[i, first:])
    return out
