# AIoT 垃圾郵件智慧管理

**Live Demo（Streamlit Cloud）** 👉 https://yt2rvxc2kfefpbph3qj2lf.streamlit.app/

> 靈感來源：Packt《Hands-On Artificial Intelligence for Cybersecurity》Ch.3 。重新實作資料預處理、可視化與 CLI/Streamlit 體驗，並以 OpenSpec 記錄需求。

## 特色

- 完整 CLI：`preprocess → train → predict → visualize`
- 預處理可輸出逐步欄位與 JSON 報表，方便 Trace
- TF-IDF + Logistic Regression，支援衍生統計特徵
- 視覺化：類別分佈、Top Tokens、ROC/PR、Threshold Sweep
- Streamlit 儀表板：資料探索 + 模型指標 + 互動推論
- OpenSpec 需求檔：`openspec/changes/spam-visual-pipeline/spec.yaml`

## 快速開始

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

資料集選項：

1. 直接使用 `datasets/sms_spam_demo.csv`（示範檔）
2. 下載 Packt/ UCI SMS Spam Collection -> 放到 `datasets/raw/`

## 指令

### 1) 預處理

```bash
python scripts/preprocess_emails.py ^
  --input datasets/sms_spam_demo.csv ^
  --output datasets/processed/sms_spam_clean.csv ^
  --label-col label ^
  --text-col text ^
  --output-text-col text_clean ^
  --save-step-columns ^
  --steps-out-dir datasets/processed/steps
```

輸出：

- 乾淨資料：`datasets/processed/sms_spam_clean.csv`
- JSON 報告：`datasets/processed/preprocess_report.json`
- （選）每步驟 CSV：`datasets/processed/steps/*.csv`

### 2) 訓練

```bash
python scripts/train_spam_classifier.py ^
  --input datasets/processed/sms_spam_clean.csv ^
  --label-col label ^
  --text-col text_clean ^
  --positive-label spam ^
  --class-weight balanced ^
  --ngram-range 1,2 ^
  --min-df 2 ^
  --sublinear-tf ^
  --eval-threshold 0.5
```

輸出：

- `models/spam_pipeline.joblib`：含 TF-IDF + LR pipeline
- `models/meta.json`：欄位設定與指標
- `reports/train_report.txt`

### 3) 推論

```bash
# 單筆
python scripts/predict_spam.py --text "Free entry in 2 a wkly comp to win cash"

# 批次
python scripts/predict_spam.py ^
  --input datasets/processed/sms_spam_clean.csv ^
  --text-col text_clean ^
  --output predictions.csv
```

### 4) 視覺化

```bash
python scripts/visualize_spam.py ^
  --input datasets/processed/sms_spam_clean.csv ^
  --label-col label ^
  --text-col text_clean ^
  --class-dist --token-freq ^
  --confusion-matrix --roc --pr --threshold-sweep
```

圖檔存於 `reports/visualizations/`。

### 5) Streamlit

```bash
# 直接透過 Streamlit
streamlit run app/streamlit_app.py



功能：

- 讀取本機/上傳資料，並自動挑選合適的 CSV
- 類別分佈圖 × 數值表同列展示，節省空間
- Top Tokens（Plotly）＋ 圖表下方 Top‑N slider（1~100）
- 模型指標（precision/recall/F1）
- 閾值調整、即時推論（Spam/Ham 範例）
- 線上示範：https://yt2rvxc2kfefpbph3qj2lf.streamlit.app/

## OpenSpec

- 規格位置：`openspec/changes/spam-visual-pipeline/spec.yaml`
- 驗證範例：`openspec validate spam-visual-pipeline --path openspec/changes/spam-visual-pipeline/spec.yaml`

## 目錄

```
├─app/                 # Streamlit UI
├─datasets/            # 原始 & 處理後資料
├─docs/                # 預處理步驟、最終報告
├─openspec/            # OpenSpec 需求
├─reports/             # 訓練報告、視覺化輸出
├─scripts/             # CLI 腳本
└─spam_pipeline/       # 可重用的 preprocessing / features / viz 模組
```

## 後續延伸

1. 加入多模型（SVM / XGBoost）並比較 ROC/PR。
2. 導入 MLflow 或 Weights & Biases 追蹤實驗。
3. Streamlit Cloud/Community Cloud 部署後更新 README 連結。

## 開發工具

- **OpenSpec**：以 `openspec/changes/spam-visual-pipeline/spec.yaml` 管理需求並可由 `openspec validate ...` 驗證。
- **AI Coding CLI**：提供互動式指令化開發流程，快速檢查資料、執行腳本與回報結果。
