"""延伸分析（不需重訓）：用既有 results/raw*.csv 檢視
「同一個模型在不同匯率的表現是否一致」。

輸出（results_ext/）：
  existing_per_nation_ratio.csv   [model x nation] 路徑 MAE / RW MAE
  existing_per_nation_dm.csv      [model x nation] 各貨幣單獨 DM p 值（vs RW）
  existing_consistency.csv        每模型：平均比、跨貨幣標準差、贏 RW 的貨幣數、兩 fold 符號一致的貨幣數
  existing_rank_consistency.csv   Friedman / Kendall's W（模型排名在 11 條貨幣之間是否一致）
  fig_existing_heatmap.png        熱圖
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).parent))
from fxlib.backtest import dm_test

RES = Path(__file__).parent / "results"
OUT = Path(__file__).parent / "results_ext"
OUT.mkdir(exist_ok=True)

KEEP = ["random_walk", "rw_drift", "ridge_returns", "ridge_returns_feat", "tuned_ridge",
        "nlinear", "dlinear", "patchtst", "patchtst_feat", "tuned_nlinear", "tuned_dlinear",
        "tuned_patchtst", "random_forest", "random_forest_feat", "mlp", "mlp_feat",
        "hf_transformer", "chronos_bolt"]


def load():
    frames = [pd.read_csv(RES / f) for f in
              ("raw.csv", "raw_tuned.csv", "raw_sklearn.csv", "raw_hf.csv", "raw_chronos.csv")]
    df = pd.concat(frames, ignore_index=True)
    df = df[df.model.isin(KEEP)]
    per_no = (df.groupby(["model", "nation", "origin", "fold"], as_index=False)
                .mean(numeric_only=True))
    return per_no


def main():
    per_no = load()
    nations = sorted(per_no.nation.unique())
    models = [m for m in KEEP if m in set(per_no.model)]
    base = per_no[per_no.model == "random_walk"].set_index(["nation", "origin"])["mae"]

    ratio = pd.DataFrame(index=models, columns=nations, dtype=float)
    dmp = pd.DataFrame(index=models, columns=nations, dtype=float)
    fold_ratio = {}
    for m in models:
        g = per_no[per_no.model == m].set_index(["nation", "origin"])
        for n in nations:
            a = g.loc[n]["mae"].sort_index()
            b = base.loc[n].reindex(a.index)
            ratio.loc[m, n] = a.mean() / b.mean()
            dmp.loc[m, n] = dm_test(a.values, b.values)[1] if m != "random_walk" else np.nan
        # 每 fold 的比值（符號一致性用）
        for f in sorted(per_no.fold.unique()):
            gf = per_no[(per_no.model == m) & (per_no.fold == f)].set_index(["nation", "origin"])
            bf = per_no[(per_no.model == "random_walk") & (per_no.fold == f)].set_index(
                ["nation", "origin"])["mae"]
            fold_ratio[(m, f)] = {n: gf.loc[n]["mae"].mean() / bf.loc[n].mean() for n in nations}

    ratio.to_csv(OUT / "existing_per_nation_ratio.csv")
    dmp.to_csv(OUT / "existing_per_nation_dm.csv")

    rows = []
    for m in models:
        if m == "random_walk":
            continue
        r = ratio.loc[m]
        f0 = pd.Series(fold_ratio[(m, 0)])
        f1 = pd.Series(fold_ratio[(m, 1)])
        rows.append({
            "model": m,
            "mean_ratio": r.mean(), "min_ratio": r.min(), "max_ratio": r.max(),
            "std_across_nations": r.std(),
            "n_beat_rw": int((r < 1).sum()),
            "n_sig_better_5pct": int(((r < 1) & (dmp.loc[m] < 0.05)).sum()),
            "n_sig_worse_5pct": int(((r > 1) & (dmp.loc[m] < 0.05)).sum()),
            "n_same_sign_both_folds": int(((f0 < 1) == (f1 < 1)).sum()),
            "n_beat_rw_both_folds": int(((f0 < 1) & (f1 < 1)).sum()),
            "spearman_fold0_fold1": stats.spearmanr(f0.values, f1.values)[0],
        })
    cons = pd.DataFrame(rows).sort_values("mean_ratio")
    cons.to_csv(OUT / "existing_consistency.csv", index=False)

    # ---- 模型排名在各貨幣間是否一致：Friedman + Kendall's W ----
    sub = ratio.drop(index="random_walk")
    ranks = sub.rank(axis=0)  # 每貨幣對模型排名
    k, n = sub.shape  # k models, n nations(blocks)
    fried = stats.friedmanchisquare(*[sub.loc[m].values for m in sub.index])
    W = fried.statistic / (n * (k - 1))
    # 各模型排名的跨貨幣標準差
    rank_std = ranks.std(axis=1)
    rc = pd.DataFrame({
        "friedman_chi2": [fried.statistic], "friedman_p": [fried.pvalue],
        "kendall_W": [W], "n_models": [k], "n_nations": [n],
        "mean_rank_std_across_nations": [rank_std.mean()],
    })
    rc.to_csv(OUT / "existing_rank_consistency.csv", index=False)
    ranks.assign(mean_rank=ranks.mean(axis=1), rank_std=rank_std).sort_values(
        "mean_rank").to_csv(OUT / "existing_model_ranks.csv")

    # ---- 熱圖 ----
    plot = ratio.drop(index="random_walk")
    order = plot.mean(axis=1).sort_values().index
    plot = plot.loc[order]
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    v = np.clip(plot.values, 0.9, 1.3)
    im = ax.imshow(v, cmap="RdBu_r", vmin=0.9, vmax=1.3, aspect="auto")
    ax.set_xticks(range(len(nations)))
    ax.set_xticklabels(nations, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    for i in range(plot.shape[0]):
        for j in range(plot.shape[1]):
            val = plot.values[i, j]
            star = "*" if dmp.loc[order[i], nations[j]] < 0.05 else ""
            ax.text(j, i, f"{val:.3f}{star}", ha="center", va="center", fontsize=7,
                    color="white" if abs(val - 1.1) > 0.12 else "black")
    fig.colorbar(im, ax=ax, label="路徑 MAE / Random Walk MAE（<1 優於 RW；* DM p<0.05）")
    ax.set_title("既有模型：各貨幣相對 Random Walk 的誤差比（2 fold × 52 起點）",
                 loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_existing_heatmap.png")

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(ratio.round(3).to_string())
        print()
        print(cons.round(3).to_string(index=False))
        print()
        print(rc.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
