# Datasets

| File | Description |
| --- | --- |
| `sms_spam_demo.csv` | 小型示範資料集（含標題列），可立即測試預處理與訓練流程。 |

## 一鍵取得 Packt Chapter03 資料

在專案根目錄執行：

```
python scripts/setup_packt_dataset.py
```

流程：
- 下載 `SMSSpamCollection`（來源：Packt Chapter03/datasets）到 `datasets/raw/`
- 轉成 `datasets/raw/sms_spam_full.csv`（含表頭 `label,text`）
- 預處理成 `datasets/processed/sms_spam_clean.csv`（供 Streamlit/訓練使用）

若要重新下載覆蓋，使用：

```
python scripts/setup_packt_dataset.py --force
```

## 推薦來源

1. Packt《Hands-On Artificial Intelligence for Cybersecurity》範例的 `ch03` Spam SMS 檔案。
2. UCI SMS Spam Collection (https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)。

下載後可放在 `datasets/raw/`，再使用 `scripts/preprocess_emails.py` 轉換成 `datasets/processed/` 供後續步驟使用。
