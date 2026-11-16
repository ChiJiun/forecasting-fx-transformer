"""產出詳盡表現報告：從全部結果 CSV 組裝完整表格 + 解讀。

輸出：experiments/docs/performance_report.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fxlib.backtest import summarize
from fxlib.reporting import dm_summary_text, feat_lag_note, pstr

RESULTS = Path(__file__).parent / "results"
DOCS = Path(__file__).parent / "docs"

LABELS = {
    "random_walk": "Random Walk", "rw_drift": "RW + drift",
    "ridge_returns": "Ridge (returns)", "ridge_returns_feat": "Ridge + 理論特徵",
    "nlinear": "NLinear", "nlinear_feat": "NLinear + 理論特徵",
    "dlinear": "DLinear", "dlinear_raw_level": "DLinear (raw level 消融)",
    "patchtst": "PatchTST-lite", "patchtst_feat": "PatchTST + 理論特徵",
    "hf_transformer": "HF Transformer (原報告模型)", "chronos_bolt": "Chronos-Bolt (零樣本)",
    "random_forest": "Random Forest", "random_forest_feat": "Random Forest + 理論特徵",
    "mlp": "MLP", "mlp_feat": "MLP + 理論特徵",
    "tuned_ridge": "Ridge (調參後)", "tuned_nlinear": "NLinear (調參後)",
    "tuned_dlinear": "DLinear (調參後)", "tuned_patchtst": "PatchTST (調參後)",
}


def lab(df, col="model"):
    df = df.copy()
    df[col] = df[col].map(lambda m: LABELS.get(m, m))
    return df


def main():
    frames = [pd.read_csv(RESULTS / "raw.csv")]
    for extra in ("raw_hf.csv", "raw_chronos.csv", "raw_tuned.csv", "raw_sklearn.csv"):
        p = RESULTS / extra
        if p.exists():
            frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    meta = json.loads((RESULTS / "meta.json").read_text())
    summary = summarize(df)
    per_no = df.groupby(["model", "nation", "origin"], as_index=False).mean(numeric_only=True)

    S = []  # markdown 片段

    S.append("# 詳盡表現報告\n")
    S.append("本文件完整記錄所有實驗的表現數據。評估設定：2 個年度測試區間"
             f"（{'、'.join(f'{a}~{b}' for a, b in meta['fold_spans'])}）、"
             f"每區間 {meta['n_test_origins']} 個預測起點（stride 5）× "
             f"{meta['n_nations']} 條貨幣序列，預測 30 天路徑；神經模型 3 seeds 平均。\n")
    S.append(feat_lag_note(meta) + "\n")

    # ---- 1. 總表 ----
    S.append(f"## 1. 模型總表（{len(summary)} 個變體，依 MASE 排序）\n")
    cols = ["model", "mase", "mase_ratio", "smape", "mae", "mse", "mase_std",
            "dm_stat", "dm_p", "mae_dm_p"]
    S.append(lab(summary[cols]).round(4).to_markdown(index=False))
    S.append("\n- `dm_p`：與 Random Walk 的整段路徑 MASE 差異之 Diebold-Mariano p 值"
             "（尺度無關；dm_stat<0 表示優於 RW）\n"
             "- `mae_dm_p`：同上但以 level MAE 計算（KRW 等高價位貨幣權重大，僅供對照）\n"
             "- `mase_ratio`：MASE 相對 Random Walk 的比值（<1 優於 RW）\n"
             "- `mase_std`：MASE 的跨起點標準差（衡量表現穩定度）\n")

    # ---- 2. 誤差隨視野 ----
    S.append("## 2. 各視野絕對誤差（跨貨幣×起點平均）\n")
    hor = per_no.groupby("model")[[f"abserr_h{h}" for h in (1, 5, 14, 30)]].mean()
    hor["path_mae"] = per_no.groupby("model")["mae"].mean()
    S.append(lab(hor.reset_index()).round(4).to_markdown(index=False))
    rw_h = hor.loc["random_walk"]
    S.append(f"\nRandom Walk 的絕對誤差從 h=1 的 {rw_h['abserr_h1']:.3f} 增至 h=30 的 "
             f"{rw_h['abserr_h30']:.3f}；各模型 h=30 誤差介於 {hor['abserr_h30'].min():.3f}–"
             f"{hor['abserr_h30'].max():.3f}。此表為 level 尺度，KRW 等高價位貨幣權重大。\n")

    # ---- 3. 跨 fold ----
    S.append("## 3. 跨期穩健性（各年度區間 MASE）\n")
    fm = df.groupby(["model", "fold"])["mase"].mean().unstack()
    fm.columns = [f"{meta['fold_spans'][int(c)][0][:7]} 起一年" for c in fm.columns]
    S.append(lab(fm.reset_index()).round(3).to_markdown(index=False))
    if fm.shape[1] >= 2:
        rho = fm.rank().corr(method="spearman").iloc[0, 1]
        hi = fm.mean().idxmax()
        S.append(f"\n整體平均 MASE 較高的是「{hi}」；兩個區間的模型排名 Spearman 相關為 "
                 f"{rho:.2f}（1 表示排名完全一致）。\n")

    # ---- 4. 各貨幣 ----
    S.append("## 4. 各貨幣 MASE（代表模型）\n")
    pick = ["random_walk", "ridge_returns_feat", "patchtst", "hf_transformer"]
    nat = (per_no[per_no.model.isin(pick)]
           .groupby(["nation", "model"])["mase"].mean().unstack()[pick])
    nat.columns = [LABELS[c] for c in nat.columns]
    S.append(nat.round(3).to_markdown())
    rw_nat = nat[LABELS["random_walk"]].sort_values(ascending=False)
    spread = nat.max(axis=1) - nat.min(axis=1)
    S.append(f"\nRandom Walk MASE 最高的貨幣：{'、'.join(rw_nat.index[:3])}；"
             f"最低：{'、'.join(rw_nat.index[-2:])}。各貨幣上這 {len(pick)} 個模型的 MASE "
             f"最大差距介於 {spread.min():.3f}–{spread.max():.3f}。\n")

    # ---- 5. 調參 ----
    S.append("## 5. 超參數調校結果\n")
    cc = json.loads((RESULTS / "chosen_configs.json").read_text())
    rows = []
    for k, v in cc.items():
        rows.append({"fold_模型": k, "選中配置": json.dumps(v, ensure_ascii=False)})
    S.append(pd.DataFrame(rows).to_markdown(index=False))
    m = summary.set_index("model")
    pairs = {"tuned_ridge": "ridge_returns_feat", "tuned_nlinear": "nlinear",
             "tuned_dlinear": "dlinear", "tuned_patchtst": "patchtst"}
    diffs = "、".join(f"{LABELS[t]} 對 {LABELS[b]} {m.loc[t, 'mase'] - m.loc[b, 'mase']:+.3f}"
                      for t, b in pairs.items() if t in m.index and b in m.index)
    S.append("\n選參只用各 fold 自身驗證區間。調參後 MASE 相對原設定的變動："
             + (diffs or "（無對照）") + "。\n")

    # ---- 6. 敏感度 ----
    sens_p = RESULTS / "raw_sensitivity.csv"
    if sens_p.exists():
        S.append("## 6. 敏感度分析：Ridge + 理論特徵 α × 落後期全網格\n")
        sens = pd.read_csv(sens_p)
        sdf = sens.groupby("model", as_index=False)["mase"].mean()
        sdf[["alpha", "n_lags"]] = sdf.model.str.extract(r"ridge_a([\d.]+)_l(\d+)").astype(float)
        S.append("MASE：\n")
        S.append(sdf.pivot(index="alpha", columns="n_lags", values="mase").round(3).to_markdown())
        S.append(f"\n{len(sdf)} 組參數的 MASE 介於 {sdf.mase.min():.3f}–{sdf.mase.max():.3f}，"
                 "點位結論不依賴超參數。\n")

    # ---- 7. RW challenge ----
    rwc_p = RESULTS / "rwchal_dm.csv"
    if rwc_p.exists():
        S.append("## 7. 「打敗 Random Walk」專項實驗（6 個年度 fold）\n")
        rwc = pd.read_csv(rwc_p)
        S.append(rwc.round(4).to_markdown(index=False))
        b = rwc.sort_values("path_ratio").iloc[0]
        S.append("\n- `*_ratio`：該模型 MAE / RW MAE（<1 為優）；`*_p`：DM p 值。"
                 "**注意：本節（rw_challenge.py）的 DM 以 level MAE／level 絕對誤差為損失，"
                 "KRW 等高價位貨幣權重大，與第 1 節的 MASE 版 dm_p 不同**\n"
                 f"- 整段路徑 MAE 比最低的是 {b.model}：{b.path_ratio:.4f}（{pstr(b.path_p)}）、"
                 f"h=30 為 {b.h30_ratio:.4f}（{pstr(b.h30_p)}）\n"
                 "- 已知問題：rw_challenge.py 的 `fwd_bias` 讀取 `ctx_feat[..., 1]`（rate_diff），"
                 "不是 fwd_premium（索引 2），`fwd_bias` 的結果尚待修正後重跑"
                 "（`combo` 為 shrunk_ridge 與 shrunk_rich 的等權平均，不含 fwd_bias）\n"
                 "- 已提交的 rwchal_dm.csv 產生於 feat_lag 修正之前，其中使用理論特徵的模型"
                 "（ridge_feat_6f、shrunk_ridge、ridge_rich、shrunk_rich、combo）不可採信\n")

    # ---- 8. HF / Chronos 明細 ----
    S.append("## 8. 原報告模型與零樣本模型明細\n")
    hf = pd.read_csv(RESULTS / "raw_hf.csv")
    hfd = hf.groupby(["fold", "seed"])[["mase", "smape", "abserr_h30"]].mean().reset_index()
    S.append("HF TimeSeriesTransformer（不使用理論特徵與未來共變數的重測，各 fold × seed）：\n")
    S.append(hfd.round(3).to_markdown(index=False))
    ch = pd.read_csv(RESULTS / "raw_chronos.csv")
    chd = ch.groupby("fold")[["mase", "smape", "abserr_h30"]].mean().reset_index()
    S.append("\nChronos-Bolt（零樣本）：\n")
    S.append(chd.round(3).to_markdown(index=False))
    seed_rng = hfd.groupby("fold")["mase"].agg(["min", "max"])
    rng_txt = "；".join(f"fold {f}：{r['min']:.2f}–{r['max']:.2f}" for f, r in seed_rng.iterrows())
    sm = summary.set_index("model")
    worse = [LABELS[x] for x in ("hf_transformer", "chronos_bolt")
             if x in sm.index and sm.loc[x, "mase_ratio"] > 1]
    S.append(f"\nHF Transformer 各 seed 的 MASE 範圍（{rng_txt}）。"
             + (f"MASE 高於 RW 的：{'、'.join(worse)}。" if worse else "") + "\n")

    # ---- 9. 結論 ----
    S.append("## 9. 表現總結\n")
    S.append("1. **點位**：" + dm_summary_text(summary, LABELS) + "\n"
             "2. 本節數字由第 1 節總表動態產生；理論特徵的時間戳處理見文件開頭說明。\n")

    DOCS.mkdir(exist_ok=True)
    (DOCS / "performance_report.md").write_text("\n".join(S), encoding="utf-8")
    print("written", DOCS / "performance_report.md")


if __name__ == "__main__":
    main()
