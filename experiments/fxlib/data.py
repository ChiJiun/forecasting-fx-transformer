"""載入 nations/*.csv 並建立面板資料與理論導向特徵。

所有特徵都只使用「預測起點當下已知」的資訊，避免原實驗中
把未來 FFX/利率餵給模型的洩漏問題。
"""
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_NAMES = ["log_ret", "rate_diff", "fwd_premium", "rate_diff_chg5"]

# 1999 年後為同一條歐元序列（僅差固定的舊幣換算比率，日報酬率完全相同），
# 合併為單一 eurozone 序列避免面板重複加權
EURO_MEMBERS = {"austria", "belgium", "finland", "france", "germany", "italy", "spain"}


def load_panel(nations_dir, dedup_euro=True, feat_lag=1):
    """回傳 (names, dates, fx, feats)。

    fx:    [N, T]  匯率水準
    feats: [N, T, F] 理論導向特徵
      - log_ret:        log(FX_t / FX_{t-1})，第一天補 0
      - rate_diff:      本國利率 - 美國利率（UIRP 核心變數）
      - fwd_premium:    (FFX - FX) / FX（CIRP 遠期溢價）
      - rate_diff_chg5: 利差的 5 日變化

    feat_lag（預設 1）：特徵整體往後推 feat_lag 個交易日，feats[:, t] 只含 t-feat_lag
    以前的資訊。原因：AUD/CAD/DKK/JPY/NOK/CHF/GBP 的 1M 遠期匯率
    （data/1M_FORWARD_RATE.csv，RFV 來源）取價時間晚於 WM/R 倫敦 16:00 即期定盤，
    同日的 fwd_premium(t) 與 t+1 即期報酬相關 0.43–0.51，等於偷看隔日行情
    （見 docs/extension_report.md §0.1、results_ext/diag_fwd_premium_timing.csv）。
    即期水準 fx 不受影響；feat_lag=0 只供重現修正前的結果。
    """
    paths = sorted(Path(nations_dir).glob("*.csv"))
    names, fx_list, feat_list = [], [], []
    dates = None
    for p in paths:
        if dedup_euro and p.stem in EURO_MEMBERS and p.stem != "germany":
            continue
        df = pd.read_csv(p)
        if dates is None:
            dates = pd.to_datetime(df["Date"]).values
        fx = df["FX"].values.astype(np.float64)
        log_ret = np.zeros_like(fx)
        log_ret[1:] = np.diff(np.log(fx))
        rate_diff = (df["INTEREST"].values - df["US_INTEREST"].values).astype(np.float64)
        fwd_premium = ((df["FFX"].values - fx) / fx).astype(np.float64)
        rd_chg5 = np.zeros_like(rate_diff)
        rd_chg5[5:] = rate_diff[5:] - rate_diff[:-5]
        feat = np.stack([log_ret, rate_diff, fwd_premium, rd_chg5], axis=-1)
        if feat_lag:
            feat = _lag_features(feat, feat_lag)
        names.append("eurozone" if (dedup_euro and p.stem == "germany") else p.stem)
        fx_list.append(fx)
        feat_list.append(feat)
    return names, dates, np.stack(fx_list), np.stack(feat_list)


def _lag_features(feat, lag):
    """把 [T, F] 特徵矩陣往後推 lag 列：out[t] = feat[t-lag]，前 lag 列重複第一列。"""
    if lag < 0:
        raise ValueError("feat_lag 必須 >= 0")
    if lag == 0:
        return feat
    out = np.empty_like(feat)
    out[lag:] = feat[:-lag]
    out[:lag] = feat[0]
    return out


def make_windows(fx, feats, origins, context_len, horizon):
    """由指定 origins（最後一個已知日的索引）切出訓練視窗。

    回傳 dict:
      ctx_level [M, L]、ctx_feat [M, L, F]、tgt_level [M, H]、nation_id [M]
    M = len(origins) * N
    """
    N = fx.shape[0]
    ctx_level, ctx_feat, tgt_level, nation_id = [], [], [], []
    for t in origins:
        sl_ctx = slice(t - context_len + 1, t + 1)
        sl_tgt = slice(t + 1, t + 1 + horizon)
        for n in range(N):
            ctx_level.append(fx[n, sl_ctx])
            ctx_feat.append(feats[n, sl_ctx])
            tgt_level.append(fx[n, sl_tgt])
            nation_id.append(n)
    return {
        "ctx_level": np.asarray(ctx_level),
        "ctx_feat": np.asarray(ctx_feat),
        "tgt_level": np.asarray(tgt_level),
        "nation_id": np.asarray(nation_id),
    }
