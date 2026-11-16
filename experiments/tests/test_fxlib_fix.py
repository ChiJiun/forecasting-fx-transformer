"""feat_lag 與 MASE 版 DM 檢定的回歸測試。

用法：python -m pytest experiments/tests -q   （或直接 python experiments/tests/test_fxlib_fix.py）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fxlib.data import load_panel, _lag_features  # noqa: E402
from fxlib.backtest import summarize, dm_test  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def test_lag_features_shift():
    feat = np.arange(12, dtype=float).reshape(6, 2)
    out = _lag_features(feat, 2)
    assert np.array_equal(out[2:], feat[:-2])
    assert np.array_equal(out[:2], np.repeat(feat[:1], 2, axis=0))
    assert _lag_features(feat, 0) is feat


def test_load_panel_lag_alignment():
    n0, d0, fx0, f0 = load_panel(ROOT / "nations", feat_lag=0)
    n1, d1, fx1, f1 = load_panel(ROOT / "nations")  # 預設 feat_lag=1
    assert n0 == n1 and len(n1) == 11 and "eurozone" in n1
    assert np.array_equal(fx0, fx1)  # 即期水準不受影響
    assert np.array_equal(f1[:, 1:], f0[:, :-1])  # feats[t] = 未延後的 feats[t-1]
    assert np.array_equal(f1[:, 0], f0[:, 0])


def _toy_df():
    """兩國、40 起點；KRW 尺度大。model_b 只在 KRW 上較好、在小尺度貨幣上較差。"""
    rng = np.random.default_rng(0)
    rows = []
    for o in range(40):
        for nation, scale in (("south_korea", 1000.0), ("japan", 1.0)):
            base = abs(rng.normal(1.0, 0.1))
            for model, mult in (("random_walk", 1.0),
                                ("model_b", 0.97 if nation == "south_korea" else 1.30)):
                m = base * mult
                rows.append({"model": model, "seed": 0, "nation": nation, "origin": o,
                             "mase": m, "mae": m * scale, "mse": (m * scale) ** 2,
                             "smape": 0.01 * m,
                             **{f"abserr_h{h}": m * scale for h in (1, 5, 14, 30)}})
    return pd.DataFrame(rows)


def test_summarize_uses_mase_for_dm():
    s = summarize(_toy_df()).set_index("model")
    b = s.loc["model_b"]
    # MASE：model_b 平均變差 → DM 統計量 > 0；level MAE 被 KRW 主導 → < 0
    assert b["dm_stat"] > 0 and b["mae_dm_stat"] < 0
    assert b["mase_ratio"] > 1
    assert np.isnan(s.loc["random_walk", "dm_p"])
    assert abs(s.loc["random_walk", "mase_ratio"] - 1) < 1e-12


def test_summarize_aligns_missing_origins():
    """模型缺少部分 origin（長度相同但集合不同）時，DM 只在共同 origin 上計算。"""
    df = _toy_df()
    # model_b 去掉 origin 0–4，另補上 RW 沒有的 origin 100–104
    b = df[(df.model == "model_b") & (df.origin >= 5)]
    extra = df[(df.model == "model_b") & (df.origin < 5)].assign(origin=lambda d: d.origin + 100)
    shifted = pd.concat([df[df.model == "random_walk"], b, extra], ignore_index=True)
    got = summarize(shifted).set_index("model").loc["model_b"]
    ref = summarize(df[df.origin >= 5]).set_index("model").loc["model_b"]
    assert np.isclose(got["dm_stat"], ref["dm_stat"])
    assert np.isclose(got["mae_dm_stat"], ref["mae_dm_stat"])


def test_dm_identical_losses():
    x = np.linspace(1, 2, 30)
    assert dm_test(x, x) == (0.0, 1.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
