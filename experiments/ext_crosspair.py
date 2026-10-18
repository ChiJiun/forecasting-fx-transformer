"""延伸實驗 3：同一時間其他匯率的內涵能否提供資訊？（channel-dependent 預測）

對照組（channel-independent，只看自身）：
  ridge_cross_own（每貨幣一個 Ridge）、nlinear、patchtst（原框架全域版）
跨貨幣模型：
  ridge_cross_dollar : + 橫斷面平均報酬（美元因子）最近 5 日
  ridge_cross_pca    : + 橫斷面報酬前 3 主成分最近 5 日
  ridge_cross_all    : + 其餘 10 條貨幣最近 5 日報酬
  mvlinear           : channel-mixing 線性（11x90 -> 11x30）
  itransformer       : 每貨幣整段 context 為一 token，attention 跨貨幣
  cross_patchtst     : (貨幣, patch) token 全部一起 attention
另輸出診斷：日報酬同期相關、lag-1 跨貨幣相關（lead-lag）、主成分解釋比。
用法：python experiments/ext_crosspair.py [--smoke] [--seeds 2] [--threads 4]
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
from fxlib.models import NaiveRW, RidgeReturns, TorchForecaster, _NLinearNet, _PatchTSTNet
from fxlib.multi import (RidgeCross, RidgeStd, PanelForecaster, _MVLinearNet,
                         _ITransformerNet, _CrossPatchTSTNet, fold_origins, vs_rw_table,
                         per_nation_ratio)
from fxlib.backtest import metric_rows, mase_scales

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).parent / "results_ext"
L, H = 90, 30


def diagnostics(names, fx, train_end):
    """只用訓練期（第 0 fold 的 cutoff 之前）計算相關結構。"""
    r = np.diff(np.log(fx[:, :train_end]), axis=1)
    C0 = pd.DataFrame(np.corrcoef(r), index=names, columns=names)
    LL = np.corrcoef(r[:, 1:], r[:, :-1])[:len(names), len(names):]
    C1 = pd.DataFrame(LL, index=names, columns=names)  # row i at t, col j at t-1
    rc = r - r.mean(1, keepdims=True)
    s = np.linalg.svd(rc, compute_uv=False)
    pcs = pd.Series(s ** 2 / np.sum(s ** 2), index=[f"pc{i+1}" for i in range(len(s))])
    return C0, C1, pcs


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
    T, N = fx.shape[1], fx.shape[0]

    rows = []
    for fold in range(n_folds):
        tr_o, va_o, te_o, cutoff = fold_origins(T, fold, L - 1, H, 5, n_test, train_stride)
        train_w = make_windows(fx, feats, tr_o, L, H)
        val_w = make_windows(fx, feats, va_o, L, H)
        test_w = make_windows(fx, feats, te_o, L, H)
        test_w["origin_idx"] = np.repeat(te_o, N)
        scales = mase_scales(fx, cutoff)
        print(f"fold {fold}: train {len(tr_o)} val {len(va_o)} test {len(te_o)} "
              f"{str(dates[te_o[0]])[:10]}~{str(dates[te_o[-1]])[:10]}", flush=True)
        if fold == 0:
            C0, C1, pcs = diagnostics(names, fx, cutoff)
            C0.to_csv(OUT / f"diag_corr_lag0{suffix}.csv")
            C1.to_csv(OUT / f"diag_corr_lag1{suffix}.csv")
            pcs.to_csv(OUT / f"diag_pca{suffix}.csv")

        def run(model, seed=0):
            t0 = time.time()
            model.fit(train_w, val_w)
            rs = metric_rows(model.name, seed, test_w, model.predict(test_w), scales, names)
            for r in rs:
                r["fold"] = fold
            rows.extend(rs)
            d = pd.DataFrame(rs)
            print(f"[fold {fold} seed {seed}] {model.name:26s} mase={d.mase.mean():.3f} "
                  f"h1={d.abserr_h1.mean():.4f} ({time.time()-t0:.0f}s)", flush=True)

        run(NaiveRW())
        run(RidgeReturns(use_feats=False))  # 全域、只看自身（原框架）
        run(RidgeStd(use_feats=False))  # 全域、只看自身、標準化+驗證選 alpha
        for mode in ("own", "dollar", "pca", "all"):
            m = RidgeCross(N, mode=mode)
            run(m)
            print(f"    chosen alphas: {m.alpha_}", flush=True)
        run(RidgeCross(N, mode="own", use_feats=True))
        run(RidgeCross(N, mode="all", use_feats=True))
        run(RidgeCross(N, mode="all", k_cross=1, name="ridge_cross_all_k1"))
        run(RidgeCross(N, mode="all", k_cross=20, name="ridge_cross_all_k20"))

        for seed in range(seeds):
            run(TorchForecaster("nlinear", lambda L_, H_: _NLinearNet(L_, H_),
                                seed=seed, max_epochs=ep), seed)
            run(PanelForecaster("mvlinear", N, lambda N_, L_, H_: _MVLinearNet(N_, L_, H_),
                                seed=seed, max_epochs=ep), seed)
            run(PanelForecaster("itransformer", N,
                                lambda N_, L_, H_: _ITransformerNet(N_, L_, H_),
                                seed=seed, max_epochs=ep), seed)
            run(TorchForecaster("patchtst", lambda L_, H_: _PatchTSTNet(L_, H_),
                                seed=seed, max_epochs=ep), seed)
            run(PanelForecaster("cross_patchtst", N,
                                lambda N_, L_, H_: _CrossPatchTSTNet(N_, L_, H_),
                                seed=seed, max_epochs=ep), seed)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"raw_crosspair{suffix}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\n=== cross-pair vs RW ===")
        print(vs_rw_table(df).round(4).to_string(index=False))
        print("\n=== per-nation MAE ratio ===")
        print(per_nation_ratio(df).round(3).to_string())
        print("\n=== per-nation h=1 abserr ratio ===")
        print(per_nation_ratio(df, col="abserr_h1").round(3).to_string())


if __name__ == "__main__":
    main()
