"""遠期溢價時間戳洩漏檢查。

診斷發現：AUD/CAD/DKK/JPY/NOK/CHF/GBP 的 fwd_premium 偏離 CIRP 的部分與「次日」即期報酬
相關 0.43–0.51（EUR/SEK/KRW/TWD 則 ≈0）。本腳本比較三種 Ridge+理論特徵設定：
  feat_orig   : 原特徵（含可能洩漏的 fwd_premium_t）
  feat_nofp   : 拿掉 fwd_premium，只留 rate_diff 與 rate_diff_chg5
  feat_lag1   : 全部理論特徵改用 t-1 的值（保守：任何 t 當日定盤時間差都被排除）
評估：點位（MASE 比、h=1 比、DM）＋ 為了對照既有結論也列出方向準確率與 PT p
（分「洩漏 7 貨幣」與「乾淨 4 貨幣」兩組）。設定與 run_experiments.py 相同。
用法：python experiments/ext_leak_check.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fxlib.data import load_panel, make_windows
from fxlib.models import NaiveRW, RidgeReturns
from fxlib.multi import RidgeStd, fold_origins, vs_rw_table, add_scaled_errors
from fxlib.backtest import metric_rows, mase_scales, pt_test, dm_test

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).parent / "results_ext"
L, H = 90, 30
LEAK = ["australia", "canada", "denmark", "japan", "norway", "switzerland", "uk"]


class RidgeFeatVariant(RidgeReturns):
    """RidgeReturns 的特徵變體：cols = 要用的特徵欄索引（在 ctx_feat 的 1: 之後）；
    lag = 取 context 倒數第 (1+lag) 天的特徵。"""

    def __init__(self, cols, lag, name):
        super().__init__(use_feats=True)
        self.cols, self.lag, self.name = cols, lag, name

    def _feat(self, w):
        return w["ctx_feat"][:, -1 - self.lag, 1:][:, self.cols]

    def _design(self, w):
        logc = np.log(w["ctx_level"])
        rets = np.diff(logc, axis=1)[:, -self.n_lags:]
        f = (self._feat(w) - self._f_mean) / self._f_std
        return np.concatenate([rets, f], axis=1)

    def fit(self, train, val):
        f = self._feat(train)
        self._f_mean, self._f_std = f.mean(0), f.std(0) + 1e-12
        X = self._design(train)
        y = np.log(train["tgt_level"]) - np.log(train["ctx_level"][:, -1:])
        from sklearn.linear_model import Ridge
        self.model = Ridge(alpha=self.alpha).fit(X, y)


def main():
    names, dates, fx, feats = load_panel(ROOT / "nations")
    T, N = fx.shape[1], fx.shape[0]
    rows = []
    for fold in range(2):
        tr_o, va_o, te_o, cutoff = fold_origins(T, fold, L - 1, H, 5, 52, 2)
        train_w = make_windows(fx, feats, tr_o, L, H)
        val_w = make_windows(fx, feats, va_o, L, H)
        test_w = make_windows(fx, feats, te_o, L, H)
        test_w["origin_idx"] = np.repeat(te_o, N)
        scales = mase_scales(fx, cutoff)
        models = [
            NaiveRW(),
            RidgeReturns(use_feats=False),
            RidgeFeatVariant([0, 1, 2], 0, "feat_orig"),
            RidgeFeatVariant([0, 2], 0, "feat_nofp"),
            RidgeFeatVariant([1], 0, "feat_fp_only"),
            RidgeFeatVariant([0, 1, 2], 1, "feat_lag1"),
            RidgeFeatVariant([1], 1, "feat_fp_only_lag1"),
        ]
        for m in models:
            m.fit(train_w, val_w)
            rs = metric_rows(m.name, 0, test_w, m.predict(test_w), scales, names)
            for r in rs:
                r["fold"] = fold
            rows.extend(rs)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "raw_leak_check.csv", index=False)

    tab = vs_rw_table(df)
    per = df.groupby(["model", "nation", "origin"], as_index=False).mean(numeric_only=True)
    per = add_scaled_errors(per)
    base = per[per.model == "random_walk"]
    out = []
    for m, g in per.groupby("model"):
        row = {"model": m}
        for grp, nats in (("leak7", LEAK), ("clean4", [n for n in names if n not in LEAK])):
            gg, bb = g[g.nation.isin(nats)], base[base.nation.isin(nats)]
            a = gg.groupby("origin")["sabserr_h1"].mean().sort_index()
            b = bb.groupby("origin")["sabserr_h1"].mean().sort_index()
            row[f"h1_ratio_{grp}"] = a.mean() / b.mean()
            row[f"h1_dm_p_{grp}"] = dm_test(a.values, b.values)[1] if m != "random_walk" else np.nan
            a = gg.groupby("origin")["mase"].mean().sort_index()
            b = bb.groupby("origin")["mase"].mean().sort_index()
            row[f"mase_ratio_{grp}"] = a.mean() / b.mean()
            row[f"diracc_h5_{grp}"] = gg["diracc_h5"].mean()
            row[f"pt_p_h5_{grp}"] = pt_test(gg["pred_cret_h5"], gg["true_cret_h5"])[1]
        out.append(row)
    grp_tab = pd.DataFrame(out)
    grp_tab.to_csv(OUT / "leak_check_groups.csv", index=False)
    tab.to_csv(OUT / "leak_check_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(tab[["model", "mase", "mase_ratio", "dm_p_mase", "h1_ratio", "h5_ratio",
                   "h30_ratio"]].round(4).to_string(index=False))
        print()
        print(grp_tab.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
