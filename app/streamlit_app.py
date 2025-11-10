from pathlib import Path
from typing import Optional
import sys

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spam_pipeline import artifacts, features, visualizations, preprocessing

st.set_page_config(page_title="📧 垃圾郵件智慧分析", layout="wide")

# 預設改為指向資料夾，會自動尋找最合適的檔案
DEFAULT_DATASET = Path("datasets")
DEFAULT_MODELS = Path("models")
DATASET_SOURCE_URL = (
    "https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity"
    "/tree/master/Chapter03/datasets"
)


@st.cache_data(show_spinner=False)
def load_dataset(path: str, _mtime: float | None = None) -> pd.DataFrame:
    """讀取 CSV，若偵測到沒有表頭則自動補上 label/text。"""
    needs_header = False
    try:
        with open(path, "r", encoding="utf-8") as fh:
            first_line = fh.readline().strip().lower()
        prefix = first_line.replace('"', "").split(",", 1)[0]
        if prefix in {"ham", "spam"}:
            needs_header = True
    except OSError:
        pass

    if needs_header:
        df = pd.read_csv(path, header=None, names=["label", "text"], dtype=str)
    else:
        df = pd.read_csv(path)
    return df


def normalize_text_for_model(text_value: str, target_col: str) -> str:
    """若模型使用 *_clean 欄位，套用預設前處理以貼近訓練資料。"""
    if not text_value:
        return ""
    if "clean" in target_col.lower():
        cleaned, _ = preprocessing.preprocess_text_column(
            pd.Series([text_value]),
            preprocessing.PreprocessConfig(),
        )
        return cleaned.iloc[0]
    return text_value


@st.cache_resource(show_spinner=False)
def load_model(models_dir: str):
    bundle = artifacts.ArtifactBundle(Path(models_dir))
    pipeline = bundle.load("spam_pipeline")
    metadata = bundle.load_metadata()
    return pipeline, metadata


def resolve_dataset_path(path: Path) -> Optional[Path]:
    """允許輸入檔案或資料夾；若為資料夾優先挑選 raw/label+text CSV。"""
    if path.is_file() and path.suffix.lower() == ".csv":
        return path
    if path.is_dir():
        candidates = list(path.rglob("*.csv"))
        if not candidates:
            return None
        def candidate_score(p: Path) -> tuple[int, int]:
            parts_lower = {part.lower() for part in p.parts}
            name_lower = p.name.lower()
            score = 0
            if "processed" in parts_lower:
                score += 4
            if "steps" in parts_lower:
                score += 5
            if "clean" in name_lower:
                score += 2
            if "full" in name_lower:
                score += 1
            if "raw" in parts_lower or "raw" in name_lower:
                score -= 2
            if "no_header" in name_lower or "original" in name_lower:
                score -= 3
            if "spam" in name_lower and "clean" not in name_lower:
                score -= 1
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            return (score, -int(size))

        return min(candidates, key=candidate_score)
    return None


def pick_dataset(local_path: Path) -> tuple[Optional[pd.DataFrame], Optional[Path]]:
    resolved = resolve_dataset_path(local_path)
    if resolved and resolved.exists():
        try:
            mtime = resolved.stat().st_mtime
        except OSError:
            mtime = None
        return load_dataset(str(resolved), mtime), resolved
    return None, None


def main():
    st.title("📧 垃圾郵件智慧分析套件")
    st.write(
        "以 Packt 第三章的垃圾郵件範例為靈感，提供資料預處理、評估與視覺化流程，"
        "並支援 CLI 與 Streamlit 雙介面。"
    )

    st.sidebar.header("資料來源")
    dataset_path = st.sidebar.text_input("資料集路徑", str(DEFAULT_DATASET))
    data, resolved_path = pick_dataset(Path(dataset_path))
    if data is None:
        st.warning("找不到資料集，至少先執行預處理腳本或上傳 CSV。")
        return
    if data.empty:
        st.warning("資料集為空，請提供至少 1 筆樣本。")
        return

    # 自動推測欄位預設
    columns = list(data.columns)
    label_candidates = ["label", "category", "target", "class", "y"]
    text_candidates = ["text_clean", "text", "message", "sms", "v2"]
    def_idx_label = next((i for i, c in enumerate(columns) if c.lower() in label_candidates), 0)
    text_options = [c for c in columns if data[c].dtype == object]
    def_idx_text = next((i for i, c in enumerate(text_options) if c.lower() in text_candidates), 0 if text_options else 0)

    label_col = st.sidebar.selectbox("標籤欄位", options=columns, index=def_idx_label)
    text_col = st.sidebar.selectbox("文字欄位", options=text_options or columns, index=def_idx_text)
    st.sidebar.subheader("資料預覽")
    st.sidebar.caption("固定顯示前 5 筆。")
    preview_rows = min(5, int(len(data)))
    max_top_tokens = int(len(data))

    st.subheader("資料概況")
    table_height = min(700, 38 * preview_rows + 60)
    st.dataframe(data.head(preview_rows), use_container_width=True, height=table_height)
    st.caption(
        f"樣本數：{len(data)} ．資料來源：[Packt Hands-On AI for Cybersecurity]"
        f"({DATASET_SOURCE_URL}) → `{resolved_path or Path(dataset_path)}`"
    )
    st.caption("資料表僅顯示前 N 筆，所有統計與圖表仍使用完整資料集。")

    label_series = data[label_col].astype(str)
    counts = label_series.value_counts()
    label_order = counts.index.tolist()
    spam_label_name = next((lbl for lbl in label_order if lbl.lower() == "spam"), label_order[0])
    ham_label_name = (
        next((lbl for lbl in label_order if lbl.lower() == "ham"), label_order[1] if len(label_order) > 1 else label_order[0])
        if label_order
        else "ham"
    )
    if ham_label_name == spam_label_name and len(label_order) > 1:
        ham_label_name = next((lbl for lbl in label_order if lbl != spam_label_name), ham_label_name)
    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"垃圾郵件（{spam_label_name}）數量", int(counts.get(spam_label_name, 0)))
    with col2:
        st.metric(f"正常郵件（{ham_label_name}）數量", int(counts.get(ham_label_name, 0)))
    st.caption(
        f"分類總數：{int(counts.sum())}（= 全部樣本 {len(data)} 筆）。若不相符請檢查標籤欄位。"
    )

    st.subheader("視覺化")
    fig = visualizations.plot_class_distribution(data, label_col)
    chart_col, info_col = st.columns([3, 2], gap="small")
    with chart_col:
        st.pyplot(fig, use_container_width=True)
    with info_col:
        st.caption("各類別樣本數（可搭配圖表觀察比例）")
        summary_df = counts.rename_axis("標籤").reset_index(name="數量")
        st.dataframe(
            summary_df,
            use_container_width=True,
            height=min(200, 38 * len(summary_df) + 38),
        )

    st.markdown("### Top Tokens by Class")
    st.caption("使用下方 Top-N slider 調整每個類別顯示的 tokens 數量（最多 100）。")
    token_slider_max = max(1, min(100, max_top_tokens))
    token_slider_min = 5 if token_slider_max >= 5 else 1
    token_slider_step = 5 if token_slider_max >= 15 else 1
    default_top_token_limit = min(20, token_slider_max)
    stored_limit = st.session_state.get("top_token_slider", default_top_token_limit)
    top_token_limit = int(
        max(token_slider_min, min(token_slider_max, stored_limit))
    )
    top_tokens = features.top_tokens_by_class(data, label_col, text_col, topn=int(top_token_limit))
    non_empty = {label: items for label, items in top_tokens.items() if items}
    if non_empty:
        columns = st.columns(len(non_empty))
        for (label, items), column in zip(non_empty.items(), columns):
            column.plotly_chart(
                plot_top_tokens_by_label(label, items),
                use_container_width=True,
            )
    else:
        st.info("詞彙統計不足，請確認文字欄位是否存在。")
    st.slider(
        "Top-N tokens（每類別）",
        min_value=token_slider_min,
        max_value=token_slider_max,
        value=top_token_limit,
        step=token_slider_step,
        help="調整每個類別顯示的 tokens 數量，上限 100。",
        key="top_token_slider",
    )

    st.subheader("模型狀態")
    models_dir = st.sidebar.text_input("模型目錄", str(DEFAULT_MODELS))
    model_text_col = None
    try:
        pipeline, metadata = load_model(models_dir)
        metrics = metadata.get("metrics", {})
        model_text_col = metadata.get("text_col")
        st.write(
            {
                "正類別": metadata.get("positive_label"),
                "決策閾值": metrics.get("threshold"),
                "精確率": metrics.get("precision"),
                "召回率": metrics.get("recall"),
                "F1": metrics.get("f1"),
                "文字欄位（模型）": model_text_col,
            }
        )
        if model_text_col and model_text_col != text_col:
            st.info(
                f"模型使用的文字欄位為 `{model_text_col}`，與側欄選取的 `{text_col}` 不同；推論會自動改用模型欄位。"
            )
    except Exception as exc:  # noqa: BLE001
        st.info(f"載入模型失敗或尚未訓練：{exc}")
        pipeline = None
        metadata = None

    st.subheader("即時推論")

    def random_example_from_label(target_label: str) -> Optional[str]:
        matches = data[label_series == target_label]
        if matches.empty or text_col not in matches.columns:
            return None
        return str(matches.sample(1)[text_col].iloc[0])

    col_spam, col_ham = st.columns(2)
    with col_spam:
        if st.button("隨機垃圾郵件範例"):
            example = random_example_from_label(spam_label_name)
            if example:
                st.session_state["candidate_text"] = example
            else:
                st.warning("資料集中找不到垃圾郵件樣本。")
    with col_ham:
        if st.button("隨機正常郵件範例"):
            example = random_example_from_label(ham_label_name)
            if example:
                st.session_state["candidate_text"] = example
            else:
                st.warning("資料集中找不到正常郵件樣本。")

    text_value = st.text_area("輸入訊息", value=st.session_state.get("candidate_text", ""))
    threshold = st.slider(
        "閾值",
        min_value=0.1,
        max_value=0.9,
        value=float(metadata.get("metrics", {}).get("threshold", 0.5) if metadata else 0.5),
        step=0.01,
    )

    if st.button("預測"):
        if pipeline is None:
            st.error("尚未載入模型，請先訓練或放置 artifacts。")
        else:
            numeric_cols = metadata.get("numeric_cols", []) if metadata else []
            inference_text_col = model_text_col or text_col
            inference_df = build_inference_frame(text_value, inference_text_col, numeric_cols)
            prob = float(pipeline.predict_proba(inference_df)[:, 1])
            positive_label = metadata.get("positive_label", "spam")
            prob_pct = prob * 100
            if positive_label.lower() == "spam":
                positive_display = "垃圾郵件"
                negative_display = "正常郵件"
            else:
                positive_display = positive_label
                negative_display = f"非 {positive_label}"
            is_positive = prob >= threshold
            label = positive_display if is_positive else negative_display
            st.success(f"預測結果：{label}")
            st.caption(
                f"模型判為 `{positive_label}` 的機率：{prob_pct:.1f}%（閾值 {threshold:.2f}）"
            )
            st.progress(min(max(prob, 0), 1))


def build_inference_frame(text_value: str, text_col: str, numeric_cols: list[str]) -> pd.DataFrame:
    """建立單筆推論資料，確保欄位符合訓練流程。"""
    normalized_text = normalize_text_for_model(text_value, text_col)
    base = pd.DataFrame({text_col: [normalized_text]})
    if numeric_cols:
        derived = features.derive_text_features(base[text_col], prefix=text_col)
        for col in numeric_cols:
            if col not in derived.columns:
                derived[col] = 0.0
        base = pd.concat([base, derived[numeric_cols]], axis=1)
    return base


def plot_top_tokens_by_label(label: str, items: list[tuple[str, int]]):
    import plotly.express as px

    df = pd.DataFrame(items, columns=["token", "count"])
    fig = px.bar(
        df,
        x="count",
        y="token",
        orientation="h",
        title=f"Class: {label}",
        labels={"count": "次數", "token": "詞彙"},
    )
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
    return fig


if __name__ == "__main__":
    main()
