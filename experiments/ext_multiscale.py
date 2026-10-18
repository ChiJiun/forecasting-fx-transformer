"""延伸實驗 2：把不同時間頻率整合到一個模型。

從日資料建立三種尺度的 context（都只用起點 t 以前的資料）：
  d : 日頻，最近 90 個交易日
  w : 週頻，每 5 個交易日取樣一次，最近 52 點（約 1 年）
  m : 月頻，每 21 個交易日取樣一次，最近 24 點（約 2 年）
另有 dl（日頻 480 點 ≈ 月頻的總回看長度）作為「只是看更長」的對照組。

模型族：
  Ridge        : 各尺度落後報酬串接（d20 / w12 / m12 及其組合）
  NLinear      : 各尺度相對 t 的 log 累積報酬串接後一層線性
  PatchTST     : 各尺度各自 patch 化 + 尺度嵌入，共用 encoder
所有變體使用相同起點（起點下限 = 最長回看 483 天），與 RW 比較。
用法：python experiments/ext_multiscale.py [--smoke] [--seeds 2] [--threads 4]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from fxlib.data import load_panel, make_windows
from fxlib.models import NaiveRW, RidgeReturns
from fxlib.multi import (SCALES, scale_lookback, add_multiscale, RidgeMultiScale,
                         MultiScaleForecaster, _MSLinearNet, _MSPatchTSTNet,
                         fold_origins, vs_rw_table, per_nation_ratio)
from fxlib.backtest import metric_rows, mase_scales

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).parent / "results_ext"
L, H = 90, 30
SC = dict(SCALES)
SC["dl"] = (1, 480)  # 長日頻對照
PATCH = {"d": 10, "w": 4, "m": 4, "dl": 48}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    OUT.mkdir(exist_ok=True)
    suffix = "_smoke" if args.smoke else ""

    n_test = 4 if args.smoke else 52
    n_folds = 1 if args.smoke else 2
    seeds = 1 if args.smoke else args.seeds
    train_stride = 20 if args.smoke else 2
    ep = 3 if args.smoke else 40

    names, dates, fx, feats = load_panel(ROOT / "nations")
    fx_log = np.log(fx)
    T, N = fx.shape[1], fx.shape[0]
    lookback = scale_lookback(SC)

    rows = []
    for fold in range(n_folds):
        tr_o, va_o, te_o, cutoff = fold_origins(T, fold, lookback, H, 5, n_test, train_stride)
        wins = {}
        for tag, o in (("train", tr_o), ("val", va_o), ("test", te_o)):
            w = make_windows(fx, feats, o, L, H)
            add_multiscale(w, fx_log, o, N, SC)
            wins[tag] = w
        train_w, val_w, test_w = wins["train"], wins["val"], wins["test"]
        test_w["origin_idx"] = np.repeat(te_o, N)
        scales = mase_scales(fx, cutoff)
        print(f"fold {fold}: train {len(tr_o)} val {len(va_o)} test {len(te_o)} "
              f"{str(dates[te_o[0]])[:10]}~{str(dates[te_o[-1]])[:10]}", flush=True)

        def run(model, seed=0):
            t0 = time.time()
            model.fit(train_w, val_w)
            rs = metric_rows(model.name, seed, test_w, model.predict(test_w), scales, names)
            for r in rs:
                r["fold"] = fold
            rows.extend(rs)
            d = pd.DataFrame(rs)
            print(f"[fold {fold} seed {seed}] {model.name:28s} mase={d.mase.mean():.3f} "
                  f"h1={d.abserr_h1.mean():.4f} ({time.time()-t0:.0f}s)", flush=True)

        run(NaiveRW())
        # ---------- Ridge ----------
        ridge_cfgs = {
            "d20": {"d": 20}, "d100": {"dl": 100}, "w12": {"w": 12}, "m12": {"m": 12},
            "d20_w12": {"d": 20, "w": 12}, "d20_m12": {"d": 20, "m": 12},
            "d20_w12_m12": {"d": 20, "w": 12, "m": 12},
            "w52_m23": {"w": 51, "m": 23},
        }
        for tag, lags in ridge_cfgs.items():
            run(RidgeMultiScale(lags, name=f"ridge_{tag}"))
        for tag in ("d20", "d20_w12_m12"):
            run(RidgeMultiScale(ridge_cfgs[tag], use_feats=True, name=f"ridge_{tag}_feat"))

        # ---------- NLinear / PatchTST ----------
        ms_cfgs = {"d": ["d"], "dl": ["dl"], "w": ["w"], "m": ["m"],
                   "dw": ["d", "w"], "dm": ["d", "m"], "dwm": ["d", "w", "m"]}
        for seed in range(seeds):
            for tag, keys in ms_cfgs.items():
                run(MultiScaleForecaster(f"nlinear_{tag}", keys,
                                         lambda dims, H_: _MSLinearNet(dims, H_),
                                         seed=seed, max_epochs=ep), seed)
            for tag in ("d", "dl", "dw", "dwm"):
                keys = ms_cfgs[tag]
                run(MultiScaleForecaster(
                    f"patchtst_{tag}", keys,
                    lambda dims, H_, k=keys: _MSPatchTSTNet(dims, [PATCH[x] for x in k], H_),
                    seed=seed, max_epochs=ep), seed)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"raw_multiscale{suffix}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\n=== multi-scale vs RW ===")
        print(vs_rw_table(df).round(4).to_string(index=False))
        print("\n=== per-nation MAE ratio ===")
        print(per_nation_ratio(df).round(3).to_string())


if __name__ == "__main__":
    main()
