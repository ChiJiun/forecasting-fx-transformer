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


def test_clark_west_zero_forecast_equals_zero_stat():
    y = np.random.default_rng(1).normal(size=200)
    t, p = mfx.clark_west(y, np.zeros_like(y), lag=3)
    assert t == 0.0 and abs(p - 0.5) < 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
