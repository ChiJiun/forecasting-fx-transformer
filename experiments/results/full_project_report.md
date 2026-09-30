# 匯率預測實驗改善計畫：完整報告

**基礎專案**：NSTC 大專學生研究計畫「基於評價理論與圖神經網路預測匯率之研究」（NSTC 113-2813-C-008-073-E）
**程式碼位置**：`D:\Code\GitHub Desktop\forcasting-fx-transformer\experiments\`
**完成日期**：2026-08-05

---

## 1. 背景與目標

原計畫以評價理論（PPP、UIRP、CIRP 等）結合 Transformer 系列模型（Transformer / Informer / Autoformer）預測 17 國兌美元匯率，並與 Random Forest、MLP、Random Walk 比較。原報告結論為 Transformer 系列在 MSE/MAE 上表現較佳，但 MASE 均大於 1。

本改善計畫的目標：**讓實驗結果在方法論上站得住腳（可信）、並探索真正能改善預測的做法（有效）**，最終達到可發表的嚴謹度。

---

## 2. 原實驗的問題診斷

改善前先對原程式碼與報告做了完整健檢，找到五個問題，按嚴重程度排列：

### 2.1 未來共變數洩漏（最嚴重）

原 `transformer.py` 把遠期匯率（FFX）、本國利率、美國利率作為 `feat_dynamic_real` 餵給 HF TimeSeriesTransformer。此架構的動態特徵會進入 `future_time_features`——亦即**模型在預測未來 30 天時，直接拿到了那 30 天的遠期匯率與利率**。遠期匯率與即期匯率幾乎同步變動，等同於偷看答案，會系統性高估「加入經濟變數」的效益。

### 2.2 單一測試視窗

原實驗只用最後 30 天（2024-02-26 ~ 2024-04-05）做一次評估。單一視窗的抽樣變異極大：後續實驗證明，原報告中各模型的排名差異在多起點平均後幾乎全部消失。

### 2.3 驗證集與訓練集重疊

原程式的 validation loss 是從與訓練期高度重疊的區段隨機抽窗計算，不是真正的 hold-out 訊號；且固定訓練 120 epochs、無 early stopping、無 checkpoint 選擇、無多 seed。

### 2.4 歐元區重複計算

17 國資料中，奧地利、比利時、芬蘭、法國、德國、義大利、西班牙 7 國自 1999 年起就是同一條歐元序列（僅差固定的舊幣換算比率）。實測 7 國的日 log 報酬率最大差異僅 0.008%（四捨五入雜訊）。原實驗等於把歐元重複加權 7 倍。

### 2.5 缺乏統計檢定

原報告的模型比較沒有任何顯著性檢定，無法區分真實差異與抽樣運氣。

---

## 3. 新評估框架設計

針對上述問題，在 `experiments/` 下建立了全新的評估框架：

| 設計 | 內容 |
|---|---|
| **Rolling-origin 回測** | 2 個年度測試區間（2022-03-04 ~ 2023-02-24、2023-03-03 ~ 2024-02-23），每區間 52 個預測起點（每 5 個交易日一個），每起點預測未來 30 天路徑 |
| **資料切分** | 每個區間各自重訓；驗證集為測試前 250 天（用於 early stopping 與選參）；訓練/驗證視窗的預測目標與測試區之間有 30 天 embargo，嚴格杜絕洩漏 |
| **無洩漏特徵** | 所有特徵只用預測起點當下已知的值：UIRP 利差（本國利率−美國利率）、CIRP 遠期溢價 ((FFX−FX)/FX)、日 log 報酬、利差 5 日變化 |
| **歐元區去重** | 7 國合併為單一 eurozone 序列，面板為 11 條獨立貨幣 |
| **點位檢定** | Diebold-Mariano 檢定 vs Random Walk（HAC 變異數處理重疊視窗的自相關 + Harvey-Leybourne-Newbold 小樣本修正） |
| **訓練規範** | 神經模型 3 seeds 平均、early stopping（patience 6）、L1 loss、log-level instance normalization |

評估指標：MASE、sMAPE、MAE、MSE、各視野絕對誤差（h=1/5/14/30）。

---

## 4. 模型（共 16 個變體）

### 基準模型
1. **Random Walk**（預測=最後價位）
2. **RW + drift**（依 context 平均日變化外推）

### 傳統計量
3. **Ridge 多步報酬回歸**（直接預測 h=1..30 累積 log 報酬）
4. **Ridge + 理論特徵**（加入 UIRP 利差、CIRP 遠期溢價、利差變化）

### 現代線性模型（Zeng et al., AAAI 2023）
5. **NLinear**、6. **NLinear + 理論特徵**
7. **DLinear**（趨勢/季節分解）、8. **DLinear raw level**（消融：不做 normalization）

### Transformer 系列
9. **PatchTST-lite**（Nie et al., ICLR 2023）、10. **PatchTST + 理論特徵**
11. **HF TimeSeriesTransformer**（原報告模型，修洩漏後重測，2 seeds）

### 基礎模型
12. **Chronos-Bolt small**（Amazon，零樣本、不訓練）

### 調參後版本（第 3 輪加入）
13–16. **tuned_ridge / tuned_nlinear / tuned_dlinear / tuned_patchtst**

---

## 5. 實驗歷程（三輪迭代）

- **第 1 輪**：建框架 + 10 模型 × 52 起點 × 17 國（尚未去重）。發現：無模型顯著贏 RW；歐元區重複問題浮現。
- **第 2 輪**：歐元去重、擴為 2 個年度區間、HF Transformer 無洩漏重測、Chronos 零樣本。核心發現成形。
- **第 3 輪**：超參數調校（每 fold 用自身驗證區間 grid search，測試集只評選出的配置）+ 全網格敏感度分析。確認所有結論對超參數穩健。

---

## 6. 最終結果

### 6.1 點位誤差（跨 11 貨幣 × 104 起點）

| 模型 | MASE | DM p 值 vs RW |
|---|---|---|
| Ridge + 理論特徵 | 5.009 | 0.21（不顯著） |
| Ridge（調參後） | 5.016 | 0.83 |
| Ridge（純報酬） | 5.026 | 0.20 |
| **Random Walk** | **5.042** | — |
| PatchTST（調參後） | 5.055 | 0.89 |
| PatchTST-lite | 5.106 | 0.54 |
| DLinear | 5.147 | 0.026（顯著**較差**） |
| NLinear | 5.158 | 0.020（顯著較差） |
| Chronos-Bolt 零樣本 | 5.589 | 0.052（較差） |
| HF Transformer（原模型，無洩漏） | 6.190 | 0.024（顯著**較差**） |
| NLinear + 特徵（消融） | 6.325 | <0.001（顯著較差） |
| DLinear raw level（消融） | 7.061 | <10⁻¹³（顯著較差） |

### 6.2 穩健性

- **跨期**：兩個年度區間（2022 升息年 / 2023 盤整年）模型排名一致。
- **超參數**：Ridge + 特徵在 α ∈ {0.1–1000} × 落後期 ∈ {5–40} 全部 20 組下，MASE 介於 5.008–5.046。
- **調參**：驗證集選參後各模型 MASE 變動 <0.1。

---

## 7. 主要發現（論文核心論點）

1. **點位預測：Random Walk 不可被打敗**——與 Meese-Rogoff (1983) 以降四十年文獻一致。原報告 Transformer 的表面優勢來自未來共變數洩漏與單一測試視窗；修正後其實顯著差於 RW。
2. **零樣本基礎模型不是免費午餐**：Chronos-Bolt 點位差於 RW（p=0.052），通用時序預訓練在匯率點位上沒有優勢。
3. **理論特徵對點位幫助有限**：Ridge 加理論特徵 MASE 只降 0.3%（不顯著）；餵給高容量神經模型反而顯著變差；instance normalization 是必要的（消融 p<10⁻¹³）。

## 8. 限制

- 合併 DM 檢定以 level MAE 計算，KRW 權重過大；尺度無關版本見 `results_ext/existing_vs_rw_scalefree.csv`（結論不變）。
- 7 個貨幣的遠期匯率取價時間晚於即期定盤，`fwd_premium` 有時間戳洩漏（見 `docs/extension_report.md` §0）。
- Chronos 預訓練語料可能與金融序列重疊，zero-shot 結果僅供參考。

## 9. 後續建議

1. 延長回測至 2019–2024（涵蓋疫情、零利率與升息週期）
2. 以正確發布時點併入月頻總經特徵（PPP、Taylor Rule，資料在 FX-predect-v2 專案）
3. 擴寫為論文格式投稿

## 10. 程式碼結構與重現方式

```
experiments/
├── fxlib/
│   ├── data.py            # 資料載入、歐元去重、理論特徵、視窗切割
│   ├── models.py          # 全部模型（統一 fit/predict 介面）
│   └── backtest.py        # 指標、DM 檢定
├── run_experiments.py     # 主實驗矩陣（python experiments/run_experiments.py）
├── hf_transformer_exp.py  # 原報告模型無洩漏重測
├── chronos_exp.py         # Chronos-Bolt 零樣本
├── tune_experiments.py    # 超參數調校 + 敏感度
├── make_report.py         # 彙整結果、產圖、寫 report.md
└── results/               # raw*.csv、summary.csv、report.md、圖表
```

各腳本皆支援 `--smoke` 快速冒煙測試。完整重跑順序：`run_experiments.py` → `hf_transformer_exp.py` → `chronos_exp.py` → `tune_experiments.py` → `make_report.py`。
