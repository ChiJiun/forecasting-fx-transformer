"""月頻總經基本面 × 長視野預測主實驗。

輸出 experiments/results_monthly/：
  forecasts.csv          所有模型的樣本外預測（長表）
  forecasts_ppp_long.csv PPP 長樣本的樣本外預測
  panel_main.csv         主樣本（所有模型共同樣本）的面板結果
  nation_main.csv        主樣本各貨幣結果
  panel_ppp_long.csv     PPP 長樣本（1995 起，只需 CPI）面板結果
  nation_ppp_long.csv
  panel_subperiod.csv    主樣本前後半段
  panel_lag_sens.csv     HAC lag 敏感度（h-1、h、2h）
  panel_macro.csv / nation_macro.csv / panel_macro_subperiod.csv / sample_span_macro.csv
                         總經子集（5 國）：貨幣模型、含產出缺口的 Taylor rule
  forecasts_macro.csv    總經子集的樣本外預測

用法：python experiments/monthly/run_monthly.py [--smoke]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import mfx

OUT = Path(__file__).resolve().parents[1] / "results_monthly"
HORIZONS = (1, 3, 6, 12, 24)


def idx(months, ym):
    return int(np.where(months == ym)[0][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    horizons = (1, 12) if args.smoke else HORIZONS
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if args.smoke else ""

    months, fx, cpi, rate = mfx.load_monthly()
    F = mfx.build_features(fx, cpi, rate)
    names = mfx.CURRENCIES
    models = mfx.default_models()
    model_names = [m.name for m in models]

    # ---- 主樣本：起點 2000-01 起，所有模型共同樣本
    t0 = time.time()
    start = idx(months, "2000-01")
    end = idx(months, "2001-12") if args.smoke else None
    df = mfx.run_backtest(F, horizons, models, start, end)
    df.to_csv(OUT / f"forecasts{tag}.csv", index=False)
    print(f"主回測完成 {len(df)} 筆 ({time.time()-t0:.0f}s)", flush=True)

    main_df = mfx.common_sample(df, model_names)
    P, Nn = mfx.evaluate(main_df, names)
    P.to_csv(OUT / f"panel_main{tag}.csv", index=False)
    Nn.to_csv(OUT / f"nation_main{tag}.csv", index=False)

    # 共同樣本的實際起訖（每個 h、每個貨幣）
    span = (main_df[main_df.model == "rw"].groupby(["h", "nation"]).t
            .agg(["min", "max", "count"]).reset_index())
    span["nation"] = span.nation.map(lambda i: names[i])
    span["start"] = span["min"].map(lambda t: months[t])
    span["end"] = span["max"].map(lambda t: months[t])
    span.drop(columns=["min", "max"]).to_csv(OUT / f"sample_span{tag}.csv", index=False)

    # ---- 子期間：以 2012-01 切兩段
    cut = idx(months, "2012-01")
    sub = []
    for lab, part in (("2000-2011", main_df[main_df.t < cut]),
                      ("2012-2023", main_df[main_df.t >= cut])):
        if len(part):
            p, _ = mfx.evaluate(part, names)
            sub.append(p.assign(period=lab))
    if sub:
        pd.concat(sub).to_csv(OUT / f"panel_subperiod{tag}.csv", index=False)

    # ---- HAC lag 敏感度
    sens = []
    for lab, fn in (("h-1", lambda h: max(h - 1, 0)), ("h", lambda h: h), ("2h", lambda h: 2 * h)):
        p, _ = mfx.evaluate(main_df, names, lag_fn=fn)
        sens.append(p[["model", "h", "rmse_ratio", "dm_p", "cw_p"]].assign(hac_lag=lab))
    pd.concat(sens).to_csv(OUT / f"panel_lag_sens{tag}.csv", index=False)

    # ---- PPP 長樣本：只需 CPI 與匯率，起點 1995-01
    ppp_models = [m for m in models if m.name in
                  ("rw", "drift", "ppp_each", "ppp_panel", "ppp_calib_hl36",
                   "ppp_calib_hl36_noinfl", "ppp_calib_hl60", "ppp_calib_hl60_noinfl")]
    t0 = time.time()
    start_long = idx(months, "1995-01")
    end_long = idx(months, "1996-12") if args.smoke else None
    dfl = mfx.run_backtest(F, horizons, ppp_models, start_long, end_long)
    dfl.to_csv(OUT / f"forecasts_ppp_long{tag}.csv", index=False)
    long_df = mfx.common_sample(dfl, [m.name for m in ppp_models])
    Pl, Nl = mfx.evaluate(long_df, names)
    Pl.to_csv(OUT / f"panel_ppp_long{tag}.csv", index=False)
    Nl.to_csv(OUT / f"nation_ppp_long{tag}.csv", index=False)
    print(f"PPP 長樣本完成 ({time.time()-t0:.0f}s)", flush=True)

    # ---- 總經子集：貨幣模型與含產出缺口的 Taylor rule（貨幣量與 IP 都可得的 5 國）
    t0 = time.time()
    money, ip = mfx.load_macro()
    Fm = mfx.build_macro_features(fx, cpi, rate, money, ip, months)
    mm = mfx.macro_models()
    dfm = mfx.run_backtest(Fm, horizons, mm, start, end)
    dfm.to_csv(OUT / f"forecasts_macro{tag}.csv", index=False)
    macro_df = mfx.common_sample(dfm, [m.name for m in mm])
    Pm, Nm = mfx.evaluate(macro_df, mfx.MACRO_SUBSET)
    Pm.to_csv(OUT / f"panel_macro{tag}.csv", index=False)
    Nm.to_csv(OUT / f"nation_macro{tag}.csv", index=False)
    subm = []
    for lab, part in (("2000-2011", macro_df[macro_df.t < cut]),
                      ("2012-2023", macro_df[macro_df.t >= cut])):
        if len(part):
            p, _ = mfx.evaluate(part, mfx.MACRO_SUBSET)
            subm.append(p.assign(period=lab))
    if subm:
        pd.concat(subm).to_csv(OUT / f"panel_macro_subperiod{tag}.csv", index=False)
    spm = (macro_df[macro_df.model == "rw"].groupby(["h", "nation"]).t
           .agg(["min", "max", "count"]).reset_index())
    spm["nation"] = spm.nation.map(lambda i: mfx.MACRO_SUBSET[i])
    spm["start"] = spm["min"].map(lambda t: months[t])
    spm["end"] = spm["max"].map(lambda t: months[t])
    spm.drop(columns=["min", "max"]).to_csv(OUT / f"sample_span_macro{tag}.csv", index=False)
    print(f"總經子集完成 ({time.time()-t0:.0f}s)", flush=True)

    meta = {"horizons": list(horizons), "main_origin_start": "2000-01",
            "macro_subset": mfx.MACRO_SUBSET, "money_breaks": mfx.MONEY_BREAKS,
            "ppp_long_origin_start": "1995-01", "subperiod_cut": "2012-01",
            "release_lag": mfx.RELEASE_LAG, "min_train": mfx.MIN_TRAIN,
            "models": model_names, "currencies": names,
            "last_month": str(months[-1])}
    (OUT / f"meta{tag}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                         encoding="utf-8")

    cols = ["model", "h", "n_obs", "rmse_ratio", "dm_p", "cw_p", "hit_rate", "n_nation_better"]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print("\n=== 主樣本（面板）===")
        print(P[cols].sort_values(["h", "rmse_ratio"]).to_string(index=False, float_format="%.3f"))
        print("\n=== PPP 長樣本（面板）===")
        print(Pl[cols].sort_values(["h", "rmse_ratio"]).to_string(index=False, float_format="%.3f"))
        print("\n=== 總經子集（面板）===")
        print(Pm[cols].sort_values(["h", "rmse_ratio"]).to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
