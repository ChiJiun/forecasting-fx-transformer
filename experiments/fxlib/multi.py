"""延伸實驗共用工具：多頻率輸入、跨貨幣（channel-dependent）模型、
全域/區域/留一貨幣訓練制度。

與 fxlib.models 相同的介面：model.fit(train, val); model.predict(w) -> [M, H] level。

所有視窗仍由 fxlib.data.make_windows 產生（順序為 origin-major、nation-minor），
因此可用 reshape(-1, N, ...) 直接取得同一起點的全部貨幣（跨貨幣模型需要）。
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge

from .models import TorchForecaster, RidgeReturns


# ====================================================================
# 1. 多頻率（multi-scale）輸入
# ====================================================================
# 每個尺度 = (取樣步長（交易日）, 取樣點數)。 d 為原本的日頻 context。
SCALES = {"d": (1, 90), "w": (5, 52), "m": (21, 24)}


def scale_lookback(scales):
    """所有尺度中需要的最長回看天數（起點索引下限）。"""
    return max(step * (k - 1) for step, k in scales.values())


def add_multiscale(w, fx_log, origins, N, scales=SCALES):
    """在視窗 dict 中加入 ms_<key>: [M, K] 的 log-level 序列，
    取樣點為 t, t-step, t-2*step, ...（由舊到新排序），只用 t 以前的資料。"""
    for key, (step, k) in scales.items():
        offs = -step * np.arange(k)[::-1]  # 由舊到新，最後一個 = t
        idx = origins[:, None] + offs[None, :]  # [n_orig, K]
        arr = fx_log[:, idx]  # [N, n_orig, K]
        arr = np.transpose(arr, (1, 0, 2)).reshape(-1, k)  # origin-major
        w[f"ms_{key}"] = arr
    return w


ALPHA_GRID = (1.0, 10.0, 100.0, 1000.0, 10000.0)


def fit_ridge_auto(X, y, Xv, yv, alpha):
    """alpha='auto' 時以驗證 MAE 從 ALPHA_GRID 選；否則固定。回傳 (model, alpha)。"""
    if alpha != "auto":
        return Ridge(alpha=alpha).fit(X, y), alpha
    best = (np.inf, None, None)
    for a in ALPHA_GRID:
        m = Ridge(alpha=a).fit(X, y)
        v = np.abs(m.predict(Xv) - yv).mean()
        if v < best[0]:
            best = (v, m, a)
    return best[1], best[2]


class RidgeStd:
    """標準化版 Ridge（全域、只看自身落後報酬，可加理論特徵）。

    與 fxlib.models.RidgeReturns 的差異：所有設計矩陣欄位 z-score 標準化
    （原版的日報酬尺度 ~0.005，alpha=10 會把落後報酬係數幾乎壓到 0），
    alpha 預設在驗證集選。作為延伸實驗中「公平的」線性對照組。
    """

    def __init__(self, use_feats=False, n_lags=20, alpha="auto", name=None):
        self.use_feats, self.n_lags, self.alpha = use_feats, n_lags, alpha
        self.name = name or ("ridge_std_feat" if use_feats else "ridge_std")

    def _raw(self, w):
        logc = np.log(w["ctx_level"])
        cols = [np.diff(logc, axis=1)[:, -self.n_lags:]]
        if self.use_feats:
            cols.append(w["ctx_feat"][:, -1, 1:])
        return np.concatenate(cols, axis=1)

    def _design(self, w):
        return (self._raw(w) - self._mu) / self._sd

    @staticmethod
    def _y(w):
        return np.log(w["tgt_level"]) - np.log(w["ctx_level"][:, -1:])

    def fit(self, train, val):
        X = self._raw(train)
        self._mu, self._sd = X.mean(0), X.std(0) + 1e-12
        self.model, self.alpha_ = fit_ridge_auto(self._design(train), self._y(train),
                                                 self._design(val), self._y(val), self.alpha)

    def predict(self, w):
        return np.exp(np.log(w["ctx_level"][:, -1:]) + self.model.predict(self._design(w)))


class RidgeMultiScale(RidgeStd):
    """Ridge：各尺度的落後報酬串接（日/週/月），標準化、alpha 驗證集選，可選理論特徵。

    lags: {'d': 20, 'w': 12, 'm': 12} 每尺度用的落後報酬數。
    """

    def __init__(self, lags, use_feats=False, alpha="auto", name=None):
        super().__init__(use_feats=use_feats, alpha=alpha)
        self.lags = lags
        self.name = name or ("ridge_ms_" + "".join(sorted(lags)) +
                             ("_feat" if use_feats else ""))

    def _raw(self, w):
        cols = []
        for key, n in self.lags.items():
            x = w[f"ms_{key}"]
            cols.append(np.diff(x, axis=1)[:, -n:])
        if self.use_feats:
            cols.append(w["ctx_feat"][:, -1, 1:])
        return np.concatenate(cols, axis=1)


class _MSLinearNet(nn.Module):
    """多尺度 NLinear：各尺度（相對 t 的 log 累積報酬）串接後一層線性。"""

    def __init__(self, dims, H):
        super().__init__()
        self.linear = nn.Linear(sum(dims), H)

    def forward(self, xs):
        return self.linear(torch.cat(xs, dim=1))


class _MSPatchTSTNet(nn.Module):
    """多尺度 PatchTST：每個尺度各自 patch 嵌入 + 尺度嵌入，
    共用一個 Transformer encoder（token 跨尺度互相注意），flatten head。"""

    def __init__(self, dims, patch_lens, H, d_model=64, n_layers=2, n_heads=4):
        super().__init__()
        self.patch_lens = patch_lens
        self.n_patches = [k // p for k, p in zip(dims, patch_lens)]
        self.embeds = nn.ModuleList([nn.Linear(p, d_model) for p in patch_lens])
        self.pos = nn.ParameterList([nn.Parameter(torch.zeros(1, n, d_model))
                                     for n in self.n_patches])
        self.scale_emb = nn.Parameter(torch.zeros(len(dims), 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 4,
                                           dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(sum(self.n_patches) * d_model, H)

    def forward(self, xs):
        toks = []
        for i, x in enumerate(xs):
            B, K = x.shape
            n, p = self.n_patches[i], self.patch_lens[i]
            x = x[:, K - n * p:].reshape(B, n, p)  # 取最新的 n*p 點
            toks.append(self.embeds[i](x) + self.pos[i] + self.scale_emb[i])
        z = self.encoder(torch.cat(toks, dim=1))
        return self.head(z.flatten(1))


class MultiScaleForecaster(TorchForecaster):
    """多尺度神經模型訓練器：輸入為各尺度 log-level 減去 t 時刻 log-level。"""

    def __init__(self, name, keys, net_fn, **kw):
        super().__init__(name, net_fn, use_feats=False, **kw)
        self.keys = keys

    def _prep_ms(self, w):
        xs, last = [], None
        for key in self.keys:
            x = torch.tensor(w[f"ms_{key}"], dtype=torch.float32)
            if last is None:
                last = x[:, -1:]
            xs.append(x - last)
        return xs, last

    def fit(self, train, val):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        H = train["tgt_level"].shape[1]
        dims = [train[f"ms_{k}"].shape[1] for k in self.keys]
        self.net = self.net_fn(dims, H)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.L1Loss()
        xt, lt = self._prep_ms(train)
        yt = torch.log(torch.tensor(train["tgt_level"], dtype=torch.float32)) - lt
        xv, lv = self._prep_ms(val)
        yv = torch.log(torch.tensor(val["tgt_level"], dtype=torch.float32)) - lv
        n = xt[0].shape[0]
        best_val, best_state, bad = np.inf, None, 0
        for _ in range(self.max_epochs):
            self.net.train()
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net([x[idx] for x in xt]), yt[idx])
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vl = loss_fn(self.net(xv), yv).item()
            if vl < best_val - 1e-6:
                best_val, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)

    def predict(self, w):
        xs, last = self._prep_ms(w)
        self.net.eval()
        with torch.no_grad():
            out = self.net(xs)
        return np.exp((out + last).numpy())


# ====================================================================
# 2. 跨貨幣（channel-dependent）模型
# ====================================================================
def to_panel(arr, N):
    """[M*N, ...] -> [M, N, ...]（視窗為 origin-major）。"""
    return arr.reshape(-1, N, *arr.shape[1:])


class RidgeCross:
    """以「目標貨幣自身落後報酬」+ 跨貨幣資訊預測，每個目標貨幣各自一個 Ridge。

    mode:
      own    : 只用自身 n_lags 日報酬（區域版 Ridge，對照組）
      dollar : + 橫斷面平均報酬（美元因子）最近 k_cross 日
      pca    : + 橫斷面報酬前 n_pc 個主成分最近 k_cross 日（在訓練期估 loading）
      all    : + 其餘所有貨幣最近 k_cross 日報酬
    """

    def __init__(self, N, mode="all", n_lags=20, k_cross=5, n_pc=3, alpha="auto",
                 use_feats=False, name=None):
        self.N, self.mode, self.n_lags, self.k_cross, self.n_pc = N, mode, n_lags, k_cross, n_pc
        self.alpha, self.use_feats = alpha, use_feats
        self.name = name or (f"ridge_cross_{mode}" + ("_feat" if use_feats else ""))

    def _rets(self, w):
        logc = np.log(to_panel(w["ctx_level"], self.N))  # [M, N, L]
        return np.diff(logc, axis=2)  # [M, N, L-1]

    def _raw(self, w, n):
        r = self._rets(w)  # [M, N, L-1]
        cols = [r[:, n, -self.n_lags:]]
        if self.mode == "dollar":
            cols.append(r.mean(axis=1)[:, -self.k_cross:])
        elif self.mode == "pca":
            rc = r[:, :, -self.k_cross:]  # [M, N, k]
            pcs = np.einsum("mnk,np->mpk", rc, self.load)  # [M, n_pc, k]
            cols.append(pcs.reshape(rc.shape[0], -1))
        elif self.mode == "all":
            others = [j for j in range(self.N) if j != n]
            cols.append(r[:, others, -self.k_cross:].reshape(r.shape[0], -1))
        if self.use_feats:
            cols.append(to_panel(w["ctx_feat"], self.N)[:, n, -1, 1:])
        return np.concatenate(cols, axis=1)

    def _design(self, w, n):
        return (self._raw(w, n) - self._mu[n]) / self._sd[n]

    def fit(self, train, val):
        r = self._rets(train)
        if self.mode == "pca":
            R = np.transpose(r, (1, 0, 2)).reshape(self.N, -1)  # [N, M*(L-1)]
            evals, evecs = np.linalg.eigh(np.cov(R))
            self.load = evecs[:, ::-1][:, :self.n_pc]  # [N, n_pc]
        logc = np.log(to_panel(train["ctx_level"], self.N))
        tgt = np.log(to_panel(train["tgt_level"], self.N))
        logv = np.log(to_panel(val["ctx_level"], self.N))
        tgtv = np.log(to_panel(val["tgt_level"], self.N))
        self.models, self._mu, self._sd, self.alpha_ = [], [], [], []
        for n in range(self.N):
            X = self._raw(train, n)
            self._mu.append(X.mean(0))
            self._sd.append(X.std(0) + 1e-12)
            m, a = fit_ridge_auto(self._design(train, n), tgt[:, n] - logc[:, n, -1:],
                                  self._design(val, n), tgtv[:, n] - logv[:, n, -1:],
                                  self.alpha)
            self.models.append(m)
            self.alpha_.append(a)

    def predict(self, w):
        logc = np.log(to_panel(w["ctx_level"], self.N))
        M = logc.shape[0]
        out = np.zeros((M, self.N, w["tgt_level"].shape[1]))
        for n in range(self.N):
            out[:, n] = logc[:, n, -1:] + self.models[n].predict(self._design(w, n))
        return np.exp(out.reshape(M * self.N, -1))


class _MVLinearNet(nn.Module):
    """Channel-mixing 線性：所有貨幣的 context 攤平 -> 所有貨幣的 H 步輸出。"""

    def __init__(self, N, L, H):
        super().__init__()
        self.N, self.H = N, H
        self.linear = nn.Linear(N * L, N * H)

    def forward(self, x):  # x: [M, N, L]
        return self.linear(x.flatten(1)).view(-1, self.N, self.H)


class _ITransformerNet(nn.Module):
    """iTransformer-lite（Liu et al., ICLR 2024）：每條貨幣的整段 context 嵌入為一個 token，
    attention 在貨幣之間進行（捕捉跨貨幣相關），每 token 線性輸出 H 步。"""

    def __init__(self, N, L, H, d_model=64, n_layers=2, n_heads=4):
        super().__init__()
        self.embed = nn.Linear(L, d_model)
        self.var_emb = nn.Parameter(torch.zeros(1, N, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 4,
                                           dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, H)

    def forward(self, x):  # [M, N, L]
        z = self.embed(x) + self.var_emb
        return self.head(self.encoder(z))  # [M, N, H]


class _CrossPatchTSTNet(nn.Module):
    """跨貨幣 PatchTST：token = (貨幣, patch)，加貨幣嵌入 + 位置嵌入，
    全部 N*n_patch token 一起做 attention；各貨幣用自己的 token flatten 輸出。"""

    def __init__(self, N, L, H, patch_len=10, d_model=64, n_layers=2, n_heads=4):
        super().__init__()
        self.N, self.p = N, patch_len
        self.n_patch = L // patch_len
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 1, self.n_patch, d_model))
        self.var_emb = nn.Parameter(torch.zeros(1, N, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 4,
                                           dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(self.n_patch * d_model, H)

    def forward(self, x):  # [M, N, L]
        M, N, L = x.shape
        x = x[:, :, L - self.n_patch * self.p:].reshape(M, N, self.n_patch, self.p)
        z = self.embed(x) + self.pos + self.var_emb  # [M, N, P, d]
        z = self.encoder(z.reshape(M, N * self.n_patch, -1))
        z = z.reshape(M, N, -1)
        return self.head(z)  # [M, N, H]


class PanelForecaster(TorchForecaster):
    """跨貨幣模型訓練器：輸入 [M, N, L]（各貨幣 log-level 減去自身 t 時刻值），
    輸出 [M, N, H]。樣本數 = 起點數（非起點×貨幣）。"""

    def __init__(self, name, N, net_fn, **kw):
        kw.setdefault("batch_size", 64)
        super().__init__(name, net_fn, use_feats=False, **kw)
        self.N = N

    def _prep_panel(self, w):
        x = torch.log(torch.tensor(to_panel(w["ctx_level"], self.N), dtype=torch.float32))
        last = x[:, :, -1:]
        return x - last, last

    def fit(self, train, val):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        L = train["ctx_level"].shape[1]
        H = train["tgt_level"].shape[1]
        self.net = self.net_fn(self.N, L, H)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.L1Loss()
        xt, lt = self._prep_panel(train)
        yt = torch.log(torch.tensor(to_panel(train["tgt_level"], self.N),
                                    dtype=torch.float32)) - lt
        xv, lv = self._prep_panel(val)
        yv = torch.log(torch.tensor(to_panel(val["tgt_level"], self.N),
                                    dtype=torch.float32)) - lv
        n = xt.shape[0]
        best_val, best_state, bad = np.inf, None, 0
        for _ in range(self.max_epochs):
            self.net.train()
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net(xt[idx]), yt[idx])
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vl = loss_fn(self.net(xv), yv).item()
            if vl < best_val - 1e-6:
                best_val, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)

    def predict(self, w):
        x, last = self._prep_panel(w)
        self.net.eval()
        with torch.no_grad():
            out = self.net(x) + last
        M, N, H = out.shape
        return np.exp(out.reshape(M * N, H).numpy())


# ====================================================================
# 3. 訓練制度：全域 / 區域 / 留一貨幣
# ====================================================================
def subset_windows(w, mask):
    """依布林 mask 篩選視窗列。"""
    return {k: (v[mask] if isinstance(v, np.ndarray) and v.shape[0] == mask.shape[0] else v)
            for k, v in w.items()}


class LocalWrapper:
    """對每個貨幣各自訓練一個基底模型（region-specific）。"""

    def __init__(self, N, base_fn, name):
        self.N, self.base_fn, self.name = N, base_fn, name

    def fit(self, train, val):
        self.models = []
        for n in range(self.N):
            m = self.base_fn()
            m.fit(subset_windows(train, train["nation_id"] == n),
                  subset_windows(val, val["nation_id"] == n))
            self.models.append(m)

    def predict(self, w):
        out = np.zeros(w["tgt_level"].shape)
        for n in range(self.N):
            mask = w["nation_id"] == n
            out[mask] = self.models[n].predict(subset_windows(w, mask))
        return out


class LOCOWrapper:
    """留一貨幣（leave-one-currency-out）：預測貨幣 n 時，模型只用其餘 N-1 條貨幣訓練，
    用於檢驗模型學到的是否為可跨貨幣轉移的結構。"""

    def __init__(self, N, base_fn, name):
        self.N, self.base_fn, self.name = N, base_fn, name

    def fit(self, train, val):
        self.models = []
        for n in range(self.N):
            m = self.base_fn()
            m.fit(subset_windows(train, train["nation_id"] != n),
                  subset_windows(val, val["nation_id"] != n))
            self.models.append(m)

    def predict(self, w):
        out = np.zeros(w["tgt_level"].shape)
        for n in range(self.N):
            mask = w["nation_id"] == n
            out[mask] = self.models[n].predict(subset_windows(w, mask))
        return out


# ====================================================================
# 4. 共用的 fold 切分（與 run_experiments.py 完全一致）
# ====================================================================
def fold_origins(T, fold, L_lookback, H=30, test_stride=5, n_test_origins=52,
                 train_stride=2):
    """回傳 (train_origins, val_origins, test_origins, train_cutoff)。
    L_lookback = 起點索引下限（需 >= 最長回看天數）。"""
    last_origin = T - 1 - H - fold * n_test_origins * test_stride
    test_origins = np.arange(last_origin - (n_test_origins - 1) * test_stride,
                             last_origin + 1, test_stride)
    train_cutoff = test_origins[0] - H
    val_origins = np.arange(train_cutoff - H - 250, train_cutoff - H, 5)
    train_origins = np.arange(L_lookback, val_origins[0] - H, train_stride)
    return train_origins, val_origins, test_origins, train_cutoff


# ====================================================================
# 5. 快速彙整：各模型 vs Random Walk（路徑 MAE 比、各視野 MAE 比、DM p）
# ====================================================================
def add_scaled_errors(per_no):
    """加入尺度無關的各視野誤差 sabserr_h*（= abserr_h* / 該貨幣 MASE 分母）。
    MASE 分母可由 mae/mase 還原。"""
    scale = (per_no["mae"] / per_no["mase"]).replace([np.inf, -np.inf], np.nan)
    for h in (1, 5, 14, 30):
        per_no[f"sabserr_h{h}"] = per_no[f"abserr_h{h}"] / scale
    return per_no


def vs_rw_table(df, baseline="random_walk", dm_lag=6):
    """以 (nation, origin) 為單位先對 seed 取平均，再對每個模型計算：
    mase、mae_ratio（<1 優於 RW）、各視野 abserr 比、DM 檢定 p 值。"""
    import pandas as pd
    from .backtest import dm_test
    per_no = (df.groupby(["model", "nation", "origin"], as_index=False)
                .mean(numeric_only=True))
    per_no = add_scaled_errors(per_no)
    base = per_no[per_no.model == baseline]
    rows = []
    for model, g in per_no.groupby("model"):
        row = {"model": model, "mase": g["mase"].mean(), "mae": g["mae"].mean()}
        # (a) 原始 level 單位的 MAE（被 KRW/JPY 等大數值貨幣主導）
        a = g.groupby("origin")["mae"].mean().sort_index()
        b = base.groupby("origin")["mae"].mean().sort_index()
        row["mae_ratio"] = a.mean() / b.mean()
        # (b) 尺度無關：MASE（每貨幣除以訓練期 naive 誤差）
        am = g.groupby("origin")["mase"].mean().sort_index()
        bm = base.groupby("origin")["mase"].mean().sort_index()
        row["mase_ratio"] = am.mean() / bm.mean()
        if model != baseline:
            row["dm_stat"], row["dm_p"] = dm_test(a.values, b.values, max_lag=dm_lag)
            row["dm_p_mase"] = dm_test(am.values, bm.values, max_lag=dm_lag)[1]
        else:
            row["dm_stat"], row["dm_p"], row["dm_p_mase"] = np.nan, np.nan, np.nan
        for h in (1, 5, 14, 30):
            ah = g.groupby("origin")[f"sabserr_h{h}"].mean().sort_index()
            bh = base.groupby("origin")[f"sabserr_h{h}"].mean().sort_index()
            row[f"h{h}_ratio"] = ah.mean() / bh.mean()
        row["n_nation_beat_rw"] = int(sum(
            g[g.nation == n]["mae"].mean() < base[base.nation == n]["mae"].mean()
            for n in g.nation.unique()))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mae_ratio").reset_index(drop=True)


def per_nation_ratio(df, baseline="random_walk", col="mae"):
    """[model x nation] 的 MAE 比（相對 RW）。"""
    import pandas as pd
    per_no = (df.groupby(["model", "nation", "origin"], as_index=False)
                .mean(numeric_only=True))
    tab = per_no.pivot_table(index="model", columns="nation", values=col, aggfunc="mean")
    return tab.div(tab.loc[baseline], axis=1)
