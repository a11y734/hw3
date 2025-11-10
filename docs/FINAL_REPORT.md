# 專案摘要

## 背景

- 基於 Packt《Hands-On Artificial Intelligence for Cybersecurity》Ch.3 垃圾郵件案例。
- 參考 2025ML-spamEmail 專案設計，加入可重複的 CLI、可視化與 Streamlit 介面。

## 目前成果

1. **預處理 CLI**：可記錄每個步驟，輸出中繼欄位與 JSON 報表。
2. **訓練 CLI**：TF-IDF + Logistic Regression，可自訂向量化與 class weight。
3. **推論 CLI**：單筆或批次 CSV，輸出 spam 機率與標籤。
4. **視覺化 CLI**：產生類別分佈、Top Tokens、ROC/PR、threshold sweep 等圖表。
5. **Streamlit UI**：資料預覽、即時圖表、模型指標、互動推論。
6. **OpenSpec**：`openspec/changes/spam-visual-pipeline/spec.yaml` 描述需求與完成度。

## 待辦建議

- 增加多種模型對比（例如 Linear SVM、XGBoost）。
- 以 MLflow 或 Weights & Biases 追蹤實驗。
- 將 Streamlit 部署至 Cloud 並把 URL 寫回 README。
