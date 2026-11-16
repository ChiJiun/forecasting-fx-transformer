"""月頻實驗的防洩漏與時點測試。

用法：python -m pytest experiments/tests/test_monthly_noleak.py -q
     （或直接 python experiments/tests/test_monthly_noleak.py）
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "monthly"))
import mfx  # noqa: E402


def _features():
    months, fx, cpi, rate = mfx.load_monthly()
    return months, mfx.build_features(fx, cpi, rate), (fx, cpi, rate)


def test_future_data_does_not_change_forecast():
    """把 t 之後的匯率與特徵換成亂數，所有模型在 t 的預測必須完全相同。"""
    months, F, _ = _features()
    rng = np.random.default_rng(0)
    t = int(np.where(months == "2010-06")[0][0])
    for h in (1, 12):
        y = mfx.targets(F["s"], h)
        G = {k: v.copy() for k, v in F.items()}
        for k in G:
            G[k][:, t + 1:] = rng.normal(size=G[k][:, t + 1:].shape)
        # y 中 t-h 之後的目標都含有 t 之後的匯率，同樣換成亂數
        yg = mfx.targets(G["s"], h)
        assert np.allclose(np.nan_to_num(yg[:, :t - h + 1]), np.nan_to_num(y[:, :t - h + 1]))
        for mdl in mfx.default_models():
            a = mdl.forecast(F, y, t, h)
            b = mdl.forecast(G, yg, t, h)
            assert np.array_equal(np.isnan(a), np.isnan(b)), mdl.name
            assert np.allclose(a[~np.isnan(a)], b[~np.isnan(b)]), (mdl.name, h)


def test_release_lags():
    """t 月的特徵只用到 t-2 月（澳洲 t-4 月）的 CPI 與 t-1 月的利率。"""
    months, F, (fx, cpi, rate) = _features()
    t = int(np.where(months == "2015-03")[0][0])
    j = mfx.CURRENCIES.index("japan")
    a = mfx.CURRENCIES.index("australia")
    p = np.log(cpi.to_numpy(dtype=float))
    us = list(cpi.columns).index("us")
    jp = list(cpi.columns).index("japan")
    au = list(cpi.columns).index("australia")
    s_jp = np.log(fx["japan"].to_numpy()[t])
    assert np.isclose(F["z"][j, t], p[t - 2, jp] - p[t - 2, us] - s_jp)
    s_au = np.log(fx["australia"].to_numpy()[t])
    assert np.isclose(F["z"][a, t], p[t - 4, au] - p[t - 2, us] - s_au)
    r = rate.to_numpy(dtype=float)
    rc = list(rate.columns)
    assert np.isclose(F["rdiff"][j, t], (r[t - 1, rc.index("japan")] - r[t - 1, rc.index("us")]) / 100)


def test_training_targets_realized():
    """訓練 mask 只含 k <= t-h（目標 s_{k+h} 在 t 月月初已觀察到）。"""
    _, F, _ = _features()
    t, h = 300, 12
    y = mfx.targets(F["s"], h)
    m = mfx._train_mask(F, y, t, h, ["z"])
    assert not m[:, t - h + 1:].any()
    assert m[:, :t - h + 1].any()


def _perturb_invisible(df, t, lag, rng, col_lag=None):
    """把 t 月月初看不到的列（參考月份 > t - lag）換成亂數。"""
    out = df.copy()
    for c in out.columns:
        L = (col_lag or {}).get(c, lag)
        rows = out.index[t - L + 1:]
        out.loc[rows, c] = rng.uniform(0.5, 2.0, len(rows)) * np.nanmean(out[c].to_numpy(float))
    return out


def test_raw_invisible_rows_do_not_change_features():
    """原始資料中依發布延遲在 t 還看不到的列全部換成亂數，所有國家 <= t 的特徵（含子集的
    z_mon、gap_diff）都必須完全相同。"""
    months, fx, cpi, rate = mfx.load_monthly()
    money, ip = mfx.load_macro()
    rng = np.random.default_rng(2)
    L = mfx.RELEASE_LAG
    for ym in ("2005-07", "2021-03"):  # 後者在美國 M1 斷點之後
        t = int(np.where(months == ym)[0][0])
        fx2 = _perturb_invisible(fx, t, 0, rng)
        cpi2 = _perturb_invisible(cpi, t, L["cpi"], rng, {"australia": L["cpi_australia"]})
        rate2 = _perturb_invisible(rate, t, L["rate"], rng)
        money2 = _perturb_invisible(money, t, L["money"], rng)
        ip2 = _perturb_invisible(ip, t, L["ip"], rng)
        for build, a, b in (
            (lambda *x: mfx.build_features(*x[:3]), (fx, cpi, rate), (fx2, cpi2, rate2)),
            (lambda *x: mfx.build_macro_features(*x, months),
             (fx, cpi, rate, money, ip), (fx2, cpi2, rate2, money2, ip2)),
        ):
            F1, F2 = build(*a), build(*b)
            for k in F1:
                x1, x2 = F1[k][:, :t + 1], F2[k][:, :t + 1]
                assert np.array_equal(np.isnan(x1), np.isnan(x2)), (ym, k)
                assert np.allclose(x1[~np.isnan(x1)], x2[~np.isnan(x2)]), (ym, k)


def test_bootstrap_index_uniform_within_pool():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "monthly"))
    import bootstrap as bs
    rng = np.random.default_rng(3)
    pool_len = 356
    idx = np.concatenate([bs.stationary_bootstrap_index(452, pool_len, 12, rng) for _ in range(400)])
    assert idx.min() >= 0 and idx.max() < pool_len
    freq = np.bincount(idx, minlength=pool_len) / len(idx)
    assert freq.max() / freq.min() < 1.3  # 各位置機率大致相同（原實作前段約 2 倍）


def test_clark_west_zero_forecast_equals_zero_stat():
    y = np.random.default_rng(1).normal(size=200)
    t, p = mfx.clark_west(y, np.zeros_like(y), lag=3)
    assert t == 0.0 and abs(p - 0.5) < 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
