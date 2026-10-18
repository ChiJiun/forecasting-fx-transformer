"""彙整延伸實驗（results_ext/raw_*.csv）為表格與圖，輸出 results_ext/ext_tables.md。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).parent))
from fxlib.multi import vs_rw_table, per_nation_ratio, add_scaled_errors
from fxlib.backtest import dm_test

OUT = Path(__file__).parent / "results_ext"
SURFACE, INK2, MUTED, GRID = "#fcfcfb", "#52514e", "#898781", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8e6bd6", "#d6467a"]


def md(df, floatfmt=".4f"):
    return df.to_markdown(index=False, floatfmt=floatfmt)


def fold_table(df, models=None):
    """各模型在兩個 fold 的 MAE 比（vs RW）。"""
    per = df.groupby(["model", "fold", "nation", "origin"], as_index=False).mean(numeric_only=True)
    t = per.pivot_table(index="model", columns="fold", values="mase", aggfunc="mean")
    t = t.div(t.loc["random_walk"], axis=1)
    t.columns = [f"fold{c}_ratio" for c in t.columns]
    if models is not None:
        t = t.loc[[m for m in models if m in t.index]]
    return t.reset_index()


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def bar_ratio(tab, path, title, highlight=None):
    tab = tab.sort_values("mase_ratio", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(tab) + 1.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.yaxis.grid(False)
    colors = [MUTED if m == "random_walk" else
              (SERIES[1] if highlight and highlight(m) else SERIES[0]) for m in tab.model]
    ax.barh(tab.model, tab.mase_ratio, color=colors, height=0.65)
    ax.axvline(1.0, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
    for i, (v, p) in enumerate(zip(tab.mase_ratio, tab.dm_p_mase)):
        s = f"{v:.3f}" + ("" if np.isnan(p) else f"  (p={p:.2f})")
        ax.annotate(s, xy=(v, i), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=INK2, va="center")
    lo = min(0.95, tab.mase_ratio.min() - 0.01)
    hi = max(1.05, min(tab.mase_ratio.max() + 0.05, 1.4))
    ax.set_xlim(lo, hi)
    ax.set_xlabel("路徑 MASE / Random Walk MASE（尺度無關；<1 優於 RW；p = DM 檢定）", fontsize=9.5, color=INK2)
    ax.set_title(title, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def heat(tab, path, title, vmin=0.9, vmax=1.15, fmt="{:.3f}"):
    fig, ax = plt.subplots(figsize=(10, 0.4 * len(tab) + 1.8), dpi=150)
    v = np.clip(tab.values.astype(float), vmin, vmax)
    im = ax.imshow(v, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(tab.shape[1]))
    ax.set_xticklabels(tab.columns, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(tab.shape[0]))
    ax.set_yticklabels(tab.index, fontsize=8.5)
    mid = (vmin + vmax) / 2
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            val = tab.values[i, j]
            ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=7,
                    color="white" if abs(val - mid) > (vmax - vmin) * 0.3 else "black")
    fig.colorbar(im, ax=ax, label="MAE / RW MAE")
    ax.set_title(title, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parts = []

    # ================= 1. consistency =================
    p = OUT / "raw_consistency.csv"
    if p.exists():
        df = pd.read_csv(p)
        tab = vs_rw_table(df)
        tab[["base", "regime"]] = tab.model.str.split("__", expand=True).reindex(
            columns=[0, 1]).values
        parts.append("## 1. 訓練制度：全域 / 區域 / 留一貨幣\n")
        parts.append(md(tab[["model", "mase", "mase_ratio", "dm_p_mase", "mae_ratio", "dm_p", "h1_ratio",
                             "h5_ratio", "h30_ratio", "n_nation_beat_rw"]]))
        # regime pivot
        piv = tab.dropna(subset=["regime"]).pivot_table(index="base", columns="regime",
                                                        values="mase_ratio")
        piv = piv[["global", "local", "loco"]] if set(["global", "local", "loco"]) <= set(piv.columns) else piv
        parts.append("\n**同一基底模型在三種制度下的 MAE 比（vs RW）**\n")
        parts.append(piv.reset_index().to_markdown(index=False, floatfmt=".4f"))
        parts.append("\n**各 fold**\n")
        parts.append(md(fold_table(df)))
        pn = per_nation_ratio(df)
        parts.append("\n**各貨幣 MAE 比**\n")
        parts.append(pn.reset_index().to_markdown(index=False, floatfmt=".3f"))
        heat(pn.drop(index="random_walk"), OUT / "fig_consistency_heatmap.png",
             "訓練制度 × 貨幣：MAE 比（vs RW）")
        bar_ratio(tab, OUT / "fig_consistency_bar.png", "全域 / 區域 / 留一貨幣 vs Random Walk",
                  highlight=lambda m: m.endswith("loco"))
        tr = pd.read_csv(OUT / "transfer_matrix.csv")
        for feat, tag in ((False, "returns"), (True, "feat")):
            m = tr[tr.feat == feat].pivot_table(index="train_on", columns="test_on",
                                                values="mae_ratio")
            m.to_csv(OUT / f"transfer_matrix_{tag}.csv")
            heat(m, OUT / f"fig_transfer_{tag}.png",
                 f"Ridge 轉移矩陣（{'含理論特徵' if feat else '只用報酬'}）：列=訓練貨幣、欄=測試貨幣",
                 vmin=0.95, vmax=1.10)
            diag = np.diag(m.values)
            off = m.values[~np.eye(len(m), dtype=bool)]
            parts.append(f"\n**Ridge 轉移矩陣（{tag}）**：對角線（自己訓自己）平均 {diag.mean():.4f}、"
                         f"非對角線（跨貨幣轉移）平均 {off.mean():.4f}；"
                         f"非對角線 <1 的比例 {(off < 1).mean():.2f}\n")
            parts.append(m.reset_index().to_markdown(index=False, floatfmt=".3f"))

    # ================= 2. multiscale =================
    p = OUT / "raw_multiscale.csv"
    if p.exists():
        df = pd.read_csv(p)
        tab = vs_rw_table(df)
        parts.append("\n\n## 2. 多頻率輸入\n")
        parts.append(md(tab[["model", "mase", "mase_ratio", "dm_p_mase", "mae_ratio", "dm_p", "h1_ratio",
                             "h5_ratio", "h14_ratio", "h30_ratio", "n_nation_beat_rw"]]))
        parts.append("\n**各 fold**\n")
        parts.append(md(fold_table(df)))
        bar_ratio(tab, OUT / "fig_multiscale_bar.png", "多頻率輸入 vs Random Walk",
                  highlight=lambda m: any(k in m for k in ("_w", "_m", "dw", "dm")))
        pn = per_nation_ratio(df, col="mase")
        heat(pn.drop(index="random_walk"), OUT / "fig_multiscale_heatmap.png",
             "多頻率輸入 × 貨幣：MAE 比（vs RW）")

    # ================= 3. cross-pair =================
    p = OUT / "raw_crosspair.csv"
    if p.exists():
        df = pd.read_csv(p)
        tab = vs_rw_table(df)
        parts.append("\n\n## 3. 跨貨幣資訊\n")
        parts.append(md(tab[["model", "mase", "mase_ratio", "dm_p_mase", "mae_ratio", "dm_p", "h1_ratio",
                             "h5_ratio", "h14_ratio", "h30_ratio", "n_nation_beat_rw"]]))
        parts.append("\n**各 fold**\n")
        parts.append(md(fold_table(df)))
        # h=1 per nation + DM on h1
        pn1 = per_nation_ratio(df, col="abserr_h1")
        parts.append("\n**各貨幣 h=1 絕對誤差比（vs RW）**\n")
        parts.append(pn1.reset_index().to_markdown(index=False, floatfmt=".3f"))
        heat(pn1.drop(index="random_walk"), OUT / "fig_crosspair_h1_heatmap.png",
             "跨貨幣模型 × 貨幣：h=1 誤差比（vs RW）", vmin=0.8, vmax=1.2)
        pn = per_nation_ratio(df, col="mase")
        heat(pn.drop(index="random_walk"), OUT / "fig_crosspair_heatmap.png",
             "跨貨幣模型 × 貨幣：路徑 MAE 比（vs RW）")
        bar_ratio(tab, OUT / "fig_crosspair_bar.png", "跨貨幣資訊 vs Random Walk",
                  highlight=lambda m: any(k in m for k in ("cross", "dollar", "pca", "mv", "itrans")))
        # h=1 DM test per model (pooled) and for KRW/TWD vs others
        per = df.groupby(["model", "nation", "origin"], as_index=False).mean(numeric_only=True)
        per = add_scaled_errors(per)
        base = per[per.model == "random_walk"]
        rows = []
        for m, g in per.groupby("model"):
            if m == "random_walk":
                continue
            row = {"model": m}
            for tag, nats in (("all", None), ("krw_twd", ["south_korea", "taiwan"]),
                              ("others", [n for n in per.nation.unique()
                                          if n not in ("south_korea", "taiwan")])):
                gg = g if nats is None else g[g.nation.isin(nats)]
                bb = base if nats is None else base[base.nation.isin(nats)]
                for col, lab in (("sabserr_h1", "h1"), ("mase", "path")):
                    a = gg.groupby("origin")[col].mean().sort_index()
                    b = bb.groupby("origin")[col].mean().sort_index()
                    row[f"{lab}_ratio_{tag}"] = a.mean() / b.mean()
                    row[f"{lab}_dm_p_{tag}"] = dm_test(a.values, b.values)[1]
            rows.append(row)
        h1 = pd.DataFrame(rows).sort_values("h1_ratio_all")
        h1.to_csv(OUT / "crosspair_h1_dm.csv", index=False)
        parts.append("\n**尺度無關誤差比與 DM p（h=1 與整段路徑 MASE）：全部 / KRW+TWD / 其餘 9 貨幣**\n")
        parts.append(md(h1))
        # diagnostics
        c1 = pd.read_csv(OUT / "diag_corr_lag1.csv", index_col=0)
        parts.append("\n**lag-1 跨貨幣相關（列 = 貨幣 i 在 t 的報酬，欄 = 貨幣 j 在 t-1 的報酬；訓練期）**\n")
        parts.append(c1.round(3).reset_index().to_markdown(index=False))
        pcs = pd.read_csv(OUT / "diag_pca.csv", index_col=0).iloc[:, 0]
        parts.append(f"\n主成分解釋變異比：{', '.join(f'{k}={v:.3f}' for k, v in pcs.head(4).items())}\n")
        heat(c1, OUT / "fig_leadlag.png", "日報酬 lag-1 跨貨幣相關（列：t，欄：t-1）",
             vmin=-0.1, vmax=0.3, fmt="{:.2f}")

    (OUT / "ext_tables.md").write_text("\n".join(parts), encoding="utf-8")
    print("\n".join(parts))


if __name__ == "__main__":
    main()
