"""延伸實驗 1：同一個模型在不同匯率的表現是否一致？

兩個層面：
  A. 訓練制度（regime）：
       global : 11 條貨幣一起訓練一個模型（原框架作法）
       local  : 每條貨幣各自訓練
       loco   : 留一貨幣——預測貨幣 n 時只用其餘 10 條訓練（檢驗可轉移性）
     模型：Ridge / Ridge+理論特徵 / NLinear / DLinear / PatchTST。
  B. Ridge 轉移矩陣：只用貨幣 i 訓練、在貨幣 j 測試，得到 11x11 的 MAE 比（vs RW）。

評估設定與 run_experiments.py 相同（2 fold x 52 起點、L=90、H=30）。
用法：python experiments/ext_consistency.py [--smoke] [--seeds 2] [--threads 4]
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
from fxlib.models import (NaiveRW, RidgeReturns, TorchForecaster,
                          _NLinearNet, _DLinearNet, _PatchTSTNet)
from fxlib.multi import (LocalWrapper, LOCOWrapper, RidgeStd, subset_windows, fold_origins,
                         vs_rw_table, per_nation_ratio)
from fxlib.backtest import metric_rows, mase_scales

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).parent / "results_ext"
L, H = 90, 30


def base_models(seed, smoke):
    ep = 3 if smoke else 40
    return {
        "ridge": lambda: RidgeReturns(use_feats=False),
        "ridge_feat": lambda: RidgeReturns(use_feats=True),
        "ridge_std": lambda: RidgeStd(use_feats=False),
        "ridge_std_feat": lambda: RidgeStd(use_feats=True),
        "nlinear": lambda: TorchForecaster("nlinear", lambda L_, H_: _NLinearNet(L_, H_),
                                           seed=seed, max_epochs=ep),
        "dlinear": lambda: TorchForecaster("dlinear", lambda L_, H_: _DLinearNet(L_, H_),
                                           seed=seed, max_epochs=ep),
        "patchtst": lambda: TorchForecaster("patchtst", lambda L_, H_: _PatchTSTNet(L_, H_),
                                            seed=seed, max_epochs=ep),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--skip-loco-patchtst", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    OUT.mkdir(exist_ok=True)
    suffix = "_smoke" if args.smoke else ""

    n_test = 4 if args.smoke else 52
    n_folds = 1 if args.smoke else 2
    seeds = 1 if args.smoke else args.seeds
    train_stride = 20 if args.smoke else 2

    names, dates, fx, feats = load_panel(ROOT / "nations")
    T, N = fx.shape[1], fx.shape[0]

    rows, transfer_rows = [], []
    for fold in range(n_folds):
        tr_o, va_o, te_o, cutoff = fold_origins(T, fold, L - 1, H, 5, n_test, train_stride)
        train_w = make_windows(fx, feats, tr_o, L, H)
        val_w = make_windows(fx, feats, va_o, L, H)
        test_w = make_windows(fx, feats, te_o, L, H)
        test_w["origin_idx"] = np.repeat(te_o, N)
        scales = mase_scales(fx, cutoff)
        print(f"fold {fold}: train {len(tr_o)} val {len(va_o)} test {len(te_o)} "
              f"{str(dates[te_o[0]])[:10]}~{str(dates[te_o[-1]])[:10]}", flush=True)

        def record(name, seed, pred):
            rs = metric_rows(name, seed, test_w, pred, scales, names)
            for r in rs:
                r["fold"] = fold
            rows.extend(rs)
            return pd.DataFrame(rs)

        record("random_walk", 0, NaiveRW().predict(test_w))

        # ---------- A. 訓練制度 ----------
        for seed in range(seeds):
            for key, fn in base_models(seed, args.smoke).items():
                is_torch = key in ("nlinear", "dlinear", "patchtst")
                if seed > 0 and not is_torch:
                    continue
                for regime in ("global", "local", "loco"):
                    if regime == "loco" and key == "patchtst" and args.skip_loco_patchtst:
                        continue
                    t0 = time.time()
                    if regime == "global":
                        m = fn()
                    elif regime == "local":
                        m = LocalWrapper(N, fn, f"{key}__local")
                    else:
                        m = LOCOWrapper(N, fn, f"{key}__loco")
                    m.fit(train_w, val_w)
                    df_m = record(f"{key}__{regime}", seed, m.predict(test_w))
                    print(f"[fold {fold} seed {seed}] {key:10s} {regime:6s} "
                          f"mase={df_m.mase.mean():.3f} ({time.time()-t0:.0f}s)", flush=True)

        # ---------- B. Ridge 轉移矩陣（train on i -> test on j）----------
        rw_pred = NaiveRW().predict(test_w)
        for use_feats in (False, True):
            for i in range(N):
                m = RidgeStd(use_feats=use_feats)
                m.fit(subset_windows(train_w, train_w["nation_id"] == i),
                      subset_windows(val_w, val_w["nation_id"] == i))
                pred = m.predict(test_w)
                for j in range(N):
                    mask = test_w["nation_id"] == j
                    mae_m = np.abs(pred[mask] - test_w["tgt_level"][mask]).mean()
                    mae_rw = np.abs(rw_pred[mask] - test_w["tgt_level"][mask]).mean()
                    transfer_rows.append({"fold": fold, "feat": use_feats,
                                          "train_on": names[i], "test_on": names[j],
                                          "mae_ratio": mae_m / mae_rw})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"raw_consistency{suffix}.csv", index=False)
    tr = pd.DataFrame(transfer_rows)
    tr.to_csv(OUT / f"transfer_matrix{suffix}.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\n=== regimes vs RW ===")
        print(vs_rw_table(df).round(4).to_string(index=False))
        print("\n=== per-nation MAE ratio ===")
        print(per_nation_ratio(df).round(3).to_string())
        print("\n=== Ridge transfer matrix (returns only, mean over folds) ===")
        print(tr[~tr.feat].pivot_table(index="train_on", columns="test_on",
                                       values="mae_ratio").round(3).to_string())
        print("\n=== Ridge transfer matrix (with theory feats) ===")
        print(tr[tr.feat].pivot_table(index="train_on", columns="test_on",
                                      values="mae_ratio").round(3).to_string())


if __name__ == "__main__":
    main()
