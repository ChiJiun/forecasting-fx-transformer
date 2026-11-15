"""PPP 模型長視野結果的 null-imposed bootstrap（見 bootstrap.py 說明）。

需先跑 run_monthly.py（使用其 forecasts*.csv 決定評估樣本）。
輸出 experiments/results_monthly/bootstrap_ppp.csv

用法：python experiments/monthly/run_bootstrap.py [--reps 499] [--smoke]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import mfx
import bootstrap as bs

OUT = Path(__file__).resolve().parents[1] / "results_monthly"
MODELS = ["ppp_calib_hl36_noinfl", "ppp_calib_hl60_noinfl", "ppp_panel", "ppp_each"]


def masks_from(forecasts, model_list, N, T, horizons):
    cs = mfx.common_sample(forecasts, model_list)
    out = {}
    for h in horizons:
        g = cs[(cs.h == h) & (cs.model == "rw")]
        m = np.zeros((N, T), dtype=bool)
        m[g.nation.values, g.t.values] = True
        out[h] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=499)
    ap.add_argument("--block", type=float, default=12)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    reps = 20 if args.smoke else args.reps
    horizons = (12, 24) if args.smoke else (6, 12, 24)

    months, fx, cpi, rate = mfx.load_monthly()
    F = mfx.build_features(fx, cpi, rate)
    s = F["s"]
    pdiff = F["z"] + s
    N, T = s.shape

    main_fc = pd.read_csv(OUT / "forecasts.csv")
    long_fc = pd.read_csv(OUT / "forecasts_ppp_long.csv")
    main_models = list(main_fc.model.unique())
    long_models = list(long_fc.model.unique())
    samples = {"main": masks_from(main_fc, main_models, N, T, horizons),
               "ppp_long": masks_from(long_fc, long_models, N, T, horizons)}

    # ---- 實際資料：快速版需與原版預測一致（ppp_each / ppp_panel / ppp_calib_hl36_noinfl）
    for h in horizons:
        y = mfx.targets(s, h)
        fast = bs.fast_forecasts(F["z"], y, h)
        g = main_fc[main_fc.h == h]
        for name in ("ppp_each", "ppp_panel", "ppp_calib_hl36_noinfl"):
            gi = g[g.model == name]
            diff = np.abs(fast[name][gi.nation.values, gi.t.values] - gi.yhat.values)
            assert np.nanmax(diff) < 1e-9 and not np.isnan(diff).any(), (name, h, np.nanmax(diff))
    print("快速版與原版預測一致", flush=True)

    actual = {}
    for h in horizons:
        y = mfx.targets(s, h)
        fast = bs.fast_forecasts(F["z"], y, h)
        for samp, masks in samples.items():
            for name in MODELS:
                actual[(samp, h, name)] = bs.panel_stats(y, fast[name], masks[h], h)

    rng = np.random.default_rng(20261116)
    sims = {k: [] for k in actual}
    t0 = time.time()
    for r in range(reps):
        s_b = bs.simulate_s(s, rng, args.block)
        z_b = pdiff - s_b
        for h in horizons:
            y_b = mfx.targets(s_b, h)
            fast = bs.fast_forecasts(z_b, y_b, h)
            for samp, masks in samples.items():
                for name in MODELS:
                    sims[(samp, h, name)].append(bs.panel_stats(y_b, fast[name], masks[h], h))
        if (r + 1) % 50 == 0:
            print(f"  {r+1}/{reps} ({time.time()-t0:.0f}s)", flush=True)

    rows = []
    for (samp, h, name), (ratio, cw_t) in actual.items():
        sim = np.array(sims[(samp, h, name)])
        rows.append({"sample": samp, "h": h, "model": name,
                     "rmse_ratio": ratio, "cw_t": cw_t,
                     "boot_p_ratio": float((sim[:, 0] <= ratio).mean()),
                     "boot_p_cw": float((sim[:, 1] >= cw_t).mean()),
                     "asym_p_cw": float(1 - __import__("scipy.stats").stats.norm.cdf(cw_t)),
                     "null_ratio_p05": float(np.quantile(sim[:, 0], 0.05)),
                     "null_ratio_median": float(np.median(sim[:, 0])),
                     "null_cw_p95": float(np.quantile(sim[:, 1], 0.95)),
                     "reps": reps})
    df = pd.DataFrame(rows).sort_values(["sample", "h", "rmse_ratio"])
    tag = "_smoke" if args.smoke else ""
    df.to_csv(OUT / f"bootstrap_ppp{tag}.csv", index=False)
    with pd.option_context("display.width", 200):
        print(df.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
