"""報告腳本共用：從 summarize() 的輸出動態產生結論文字，避免報告內文寫死舊數字。

dm_p 的語意以 fxlib.backtest.summarize 為準：MASE 損失、雙尾 p 值，
dm_stat < 0 表示模型優於 baseline。
"""
import numpy as np


def fmt_p(p):
    if p is None or np.isnan(p):
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def dm_findings(summary, labels, baseline="random_walk", alpha=0.05, marginal=0.10):
    """回傳 dict：best（MASE 最低的非 baseline 模型）、better / worse / marginal_worse
    （各為 [(label, mase_ratio, dm_p), ...]）。"""
    lab = lambda m: labels.get(m, m)  # noqa: E731
    s = summary[summary.model != baseline].sort_values("mase")
    if s.empty:
        raise ValueError(f"summary 只有 baseline（{baseline}），沒有可比較的模型")
    best = s.iloc[0]
    pick = lambda mask: [(lab(r.model), r.mase_ratio, r.dm_p)  # noqa: E731
                         for r in s[mask].itertuples()]
    return {
        "best": (lab(best.model), float(best.mase_ratio), float(best.dm_p)),
        "better": pick((s.dm_stat < 0) & (s.dm_p < alpha)),
        "worse": pick((s.dm_stat > 0) & (s.dm_p < alpha)),
        "marginal_worse": pick((s.dm_stat > 0) & (s.dm_p >= alpha) & (s.dm_p < marginal)),
    }


def pstr(p, name="p"):
    """「p=0.123」或「p<0.001」。"""
    s = fmt_p(p)
    return f"{name}{s}" if s.startswith("<") else f"{name}={s}"


def fmt_list(items):
    if not items:
        return "無"
    return "、".join(f"{name}（MASE 比 {r:.3f}，{pstr(p)}）" for name, r, p in items)


def dm_summary_text(summary, labels, baseline="random_walk"):
    """一段描述點位 DM 結果的 markdown 文字（數字全部來自 summary）。"""
    f = dm_findings(summary, labels, baseline)
    name, r, p = f["best"]
    better = ("顯著優於 Random Walk 的模型：" + fmt_list(f["better"])) if f["better"] \
        else "沒有任何模型在 5% 水準顯著優於 Random Walk"
    return (f"{better}。MASE 最低的是 {name}（MASE 比 {r:.3f}，DM {pstr(p)}）。"
            f"在 5% 水準顯著**差於** RW 的模型：{fmt_list(f['worse'])}；"
            f"0.05–0.10 之間差於 RW 的模型：{fmt_list(f['marginal_worse'])}。"
            "（DM 以 MASE 為損失、雙尾檢定。）")


def feat_lag_note(meta):
    """依 meta.json 是否記錄 feat_lag，如實描述理論特徵的時間戳處理。"""
    lag = meta.get("feat_lag")
    if lag is None or lag == 0:
        return ("**注意：本結果產生時理論特徵未延後（feat_lag=0）**。AUD/CAD/DKK/JPY/NOK/CHF/GBP "
                "的遠期匯率取價晚於即期定盤，`fwd_premium` 含次日即期報酬資訊，"
                "所有使用理論特徵的模型結果都不可採信，包括帶 `_feat` 的模型、"
                "選中 use_feats=True 的 tuned_ridge、Ridge + 理論特徵的敏感度網格，"
                "以及 rw_challenge 的 ridge 系列（見 docs/extension_report.md §0）。"
                "目前 `load_panel` 預設 feat_lag=1，重跑實驗即可得到修正後的結果。")
    return (f"理論特徵延後 {lag} 個交易日（feat_lag={lag}），"
            "避免遠期匯率取價晚於即期定盤造成的時間戳洩漏（見 docs/extension_report.md §0）。")
