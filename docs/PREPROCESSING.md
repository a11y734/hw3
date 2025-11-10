# 預處理流程說明

## 1. 輸入

- 預設資料來源：`datasets/sms_spam_demo.csv`
- 支援 `--no-header` 與欄位索引選擇。

## 2. 清理步驟

| 順序 | 說明 |
| --- | --- |
| 00 | 原始文字（補空值、轉字串） |
| 01 | 轉小寫（可用 `--keep-case` 關閉） |
| 02 | 移除 HTML tag |
| 03 | 移除 URL |
| 04 | 移除 email 位址 |
| 05 | 移除數字（需加 `--remove-numbers`） |
| 06 | 移除標點（可用 `--keep-punct` 關閉） |
| 07 | 收斂多餘空白 |
| 08 | 停用詞過濾（`--remove-stopwords` 可啟用，另可加 `--stopwords ...`） |
| 09 | Snowball stemming (`--stem`) |

執行 `--save-step-columns` 時，所有步驟會輸出到 `datasets/processed/steps/`，方便比對。

## 3. 衍生特徵

`spam_pipeline.features.derive_text_features` 會產生下列欄位（預設前綴 `text_clean`）：

- `_char_len`：字元數
- `_token_len`：token 數
- `_digit_count`：數字字元數
- `_uppercase_count`：大寫字元數
- `_percent_caps`：大寫比例
- `_avg_token_len`：平均 token 長度
- `_bang_count`：驚嘆號數量

CLI 可利用 `--numeric-cols` 選擇是否把這些欄位餵給模型。

## 4. 輸出

- 乾淨文字欄位（預設 `text_clean`）
- 衍生特徵欄位
- JSON 報表：`datasets/processed/preprocess_report.json`
