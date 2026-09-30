"""Rolling-origin 回測與評估指標。

- 多個預測起點（stride 抽樣），每起點預測 H 天路徑
- MASE / sMAPE / MAE / MSE，以及 h=1,5,14,30 各視野的絕對誤差
- Diebold-Mariano 檢定（HAC 變異數，起點重疊時仍有效）
"""
import numpy as np
import pandas as pd
from scipy import stats


def metric_rows(model_name, seed, w, pred, mase_scale, names, horizons=(1, 5, 14, 30)):
    """每個 (nation, origin) 一列的指標。w 需含 origin_idx 欄位。"""
    tgt = w["tgt_level"]
    err = pred - tgt
    rows = []
    for i in range(tgt.shape[0]):
        n = w["nation_id"][i]
        row = {
            "model": model_name,
            "seed": seed,
            "nation": names[n],
            "origin": int(w["origin_idx"][i]),
            "mae": float(np.abs(err[i]).mean()),
            "mse": float((err[i] ** 2).mean()),
            "smape": float(np.mean(2 * np.abs(err[i]) / (np.abs(pred[i]) + np.abs(tgt[i])))),
            "mase": float(np.abs(err[i]).mean() / mase_scale[n]),
        }
        for h in horizons:
            row[f"abserr_h{h}"] = float(np.abs(err[i, h - 1]))
        rows.append(row)
    return rows


def mase_scales(fx, train_end):
    """MASE 分母：訓練期 naive 一步預測的平均絕對誤差（每國一個）。"""
    return np.abs(np.diff(fx[:, :train_end], axis=1)).mean(axis=1)


def dm_test(loss_a, loss_b, max_lag=6):
    """Diebold-Mariano：loss_a/loss_b 為依時間排序的每起點損失序列。

    回傳 (dm_stat, p_value)。dm<0 表示 A 優於 B。
    HAC (Bartlett kernel) 變異數處理重疊預測造成的自相關。
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    T = len(d)
    max_lag = min(max_lag, max(1, T // 4))  # 樣本太少時縮小 lag，避免 HAC 退化
    dbar = d.mean()
    dc = d - dbar
    gamma0 = (dc ** 2).mean()
    var = gamma0
    for k in range(1, min(max_lag, T - 1) + 1):
        gk = (dc[k:] * dc[:-k]).mean()
        var += 2 * (1 - k / (max_lag + 1)) * gk
    if var <= 0:
        return 0.0, 1.0
    dm = dbar / np.sqrt(var / T)
    # Harvey-Leybourne-Newbold 小樣本修正
    h = max_lag
    correction = np.sqrt(max(1e-12, (T + 1 - 2 * h + h * (h - 1) / T) / T))
    dm *= correction
    p = 2 * stats.t.sf(np.abs(dm), df=T - 1)
    return float(dm), float(p)


def summarize(df, baseline="random_walk"):
    """彙整成模型層級摘要表，含 DM 檢定 vs baseline。"""
    # 對神經模型多 seed 先取平均（同 nation/origin）
    per_no = (df.groupby(["model", "nation", "origin"], as_index=False)
                .mean(numeric_only=True).drop(columns="seed"))
    base = (per_no[per_no.model == baseline]
            .groupby("origin")["mae"].mean().sort_index())

    out = []
    for model, g in per_no.groupby("model"):
        agg = g.mean(numeric_only=True)
        row = {"model": model,
               "mase": agg["mase"], "smape": agg["smape"],
               "mae": agg["mae"], "mse": agg["mse"]}
        for h in (1, 5, 14, 30):
            row[f"abserr_h{h}"] = agg[f"abserr_h{h}"]
        # MASE 的起點間標準差（跨國平均後）
        by_origin = g.groupby("origin")[["mase", "mae"]].mean().sort_index()
        row["mase_std"] = by_origin["mase"].std()
        if model != baseline:
            dm, p = dm_test(by_origin["mae"].values, base.values)
            row["dm_stat"], row["dm_p"] = dm, p
        else:
            row["dm_stat"], row["dm_p"] = np.nan, np.nan
        out.append(row)
    return pd.DataFrame(out).sort_values("mase").reset_index(drop=True)
