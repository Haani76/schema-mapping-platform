import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.inference.predictor import SchemaPredictor
from configs.config import config

# Page config
st.set_page_config(
    page_title="Schema Mapping Platform",
    page_icon=None,
    layout="wide",
)

# Load model once
@st.cache_resource
def load_predictor():
    return SchemaPredictor()

# Header
st.title("Schema Mapping Platform")
st.markdown("**AI-powered semantic column inference using BERT NER**")
st.markdown("Upload a CSV or enter column names manually to automatically map them to canonical schema types.")
st.divider()

# Sidebar
with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Confidence Threshold",
        min_value=0.50,
        max_value=0.99,
        value=0.85,
        step=0.01,
        help="Predictions above this threshold are auto-mapped. Below goes to human review."
    )
    st.divider()
    st.header("Supported Labels")
    labels = [l.replace("B-", "") for l in config.LABELS if l != "O"]
    for label in labels:
        st.markdown(f"- `{label}`")
    st.divider()
    st.markdown("[View Source on GitHub](https://github.com/Haani76/schema-mapping-platform)")

# Load predictor
with st.spinner("Loading BERT model..."):
    predictor = load_predictor()
    predictor.confidence_threshold = threshold

st.success("Model loaded and ready.")

# Tabs
tab1, tab2, tab3 = st.tabs(["Upload CSV", "Manual Input", "About"])

# --- Tab 1: CSV Upload ---
with tab1:
    st.subheader("Upload a CSV file")
    st.markdown("The platform will read your column headers and automatically infer their semantic types.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, nrows=5)
        st.markdown("**Preview (first 5 rows):**")
        st.dataframe(df, use_container_width=True)

        columns = list(df.columns)
        st.markdown(f"**Detected {len(columns)} columns:** {', '.join(columns)}")

        if st.button("Run Schema Mapping", key="csv_btn"):
            with st.spinner("Running inference..."):
                results = predictor.predict_dataframe_columns(columns)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Columns", results["total_columns"])
            col2.metric("Auto Mapped", results["auto_mapped_count"])
            col3.metric("Needs Review", results["needs_review_count"])
            col4.metric("Auto Map Rate", f"{results['auto_map_rate']*100:.1f}%")

            st.divider()

            if results["auto_mapped"]:
                st.subheader("Auto Mapped")
                auto_df = pd.DataFrame([{
                    "Column": p["column_name"],
                    "Predicted Label": p["predicted_label"],
                    "Confidence": f"{p['confidence']*100:.1f}%",
                    "Routing": p["routing"],
                } for p in results["auto_mapped"]])
                st.dataframe(auto_df, use_container_width=True)

            if results["needs_review"]:
                st.subheader("Needs Human Review")
                review_df = pd.DataFrame([{
                    "Column": p["column_name"],
                    "Predicted Label": p["predicted_label"],
                    "Confidence": f"{p['confidence']*100:.1f}%",
                    "Reason": "Below confidence threshold",
                } for p in results["needs_review"]])
                st.dataframe(review_df, use_container_width=True)

            all_results = results["auto_mapped"] + results["needs_review"]
            results_df = pd.DataFrame([{
                "column_name": p["column_name"],
                "predicted_label": p["predicted_label"],
                "confidence": p["confidence"],
                "routing": p["routing"],
            } for p in all_results])

            st.download_button(
                label="Download Results as CSV",
                data=results_df.to_csv(index=False),
                file_name="schema_mapping_results.csv",
                mime="text/csv",
            )

# --- Tab 2: Manual Input ---
with tab2:
    st.subheader("Enter column names manually")
    st.markdown("Enter one column name per line. Optionally add a sample value separated by a comma.")

    example = "customer_id, C-1234\nrevenue_q3, 15000.00\norder_date, 2023-01-15\nemail_address, john@example.com\nunknown_col_xyz"
    user_input = st.text_area(
        "Column names (one per line, optional: column_name, sample_value)",
        value=example,
        height=200,
    )

    if st.button("Run Schema Mapping", key="manual_btn"):
        lines = [l.strip() for l in user_input.strip().split("\n") if l.strip()]
        columns_input = []
        for line in lines:
            parts = line.split(",", 1)
            col_name = parts[0].strip()
            sample_val = parts[1].strip() if len(parts) > 1 else ""
            columns_input.append({"column_name": col_name, "sample_value": sample_val})

        with st.spinner("Running inference..."):
            results = predictor.predict_dataframe_columns(
                [c["column_name"] for c in columns_input]
            )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Columns", results["total_columns"])
        col2.metric("Auto Mapped", results["auto_mapped_count"])
        col3.metric("Needs Review", results["needs_review_count"])
        col4.metric("Auto Map Rate", f"{results['auto_map_rate']*100:.1f}%")

        st.divider()

        st.subheader("Detailed Results")
        all_preds = results["auto_mapped"] + results["needs_review"]
        for pred in all_preds:
            routing_label = "Auto Mapped" if pred["routing"] == "auto_map" else "Needs Review"
            with st.expander(f"{pred['column_name']} → {pred['predicted_label']} ({pred['confidence']*100:.1f}%) — {routing_label}"):
                st.markdown(f"**Predicted Label:** `{pred['predicted_label']}`")
                st.markdown(f"**Confidence:** `{pred['confidence']*100:.1f}%`")
                st.markdown(f"**Routing:** `{pred['routing']}`")
                st.markdown("**Top 3 Predictions:**")
                for i, top in enumerate(pred["top3_predictions"]):
                    label = top.get("label", "") if isinstance(top, dict) else str(top)
                    confidence = top.get("confidence", 0) if isinstance(top, dict) else 0
                    st.markdown(f"{i+1}. `{label}` — {float(confidence)*100:.1f}%")

# --- Tab 3: About ---
with tab3:
    st.subheader("About this project")
    st.markdown("""
    ### What is this?
    This platform automatically maps CSV/database column headers to canonical schema types
    using a fine-tuned BERT NER (Named Entity Recognition) model.

    ### How it works
    1. You upload a CSV or enter column names
    2. The BERT model predicts the semantic type of each column
    3. High confidence predictions are **auto-mapped**
    4. Low confidence predictions are flagged for **human review**

    ### Model Performance
    | Metric | Value |
    |---|---|
    | Validation F1 Score | 99.18% |
    | Auto-map Rate | 91.7% |
    | Mean Confidence | 91.4% |
    | Model Parameters | 108.9 Million |

    ### Tech Stack
    - **Model:** BERT-base-uncased (HuggingFace Transformers)
    - **Framework:** PyTorch
    - **API:** FastAPI
    - **Experiment Tracking:** MLflow
    - **Deployment:** Streamlit Cloud
    - **Containerization:** Docker

    ### Supported Semantic Types
    `CUSTOMER_ID` `PRODUCT_ID` `REVENUE` `DATE` `QUANTITY`
    `LOCATION` `EMAIL` `PHONE` `NAME` `STATUS` `CATEGORY`

    ### Links
    - [GitHub Repository](https://github.com/Haani76/schema-mapping-platform)
    - [Live Demo](https://schema-mapping-platform-apbvzvujnb8gthfwdiuf63.streamlit.app/)
    - [Model on HuggingFace](https://huggingface.co/Haani76/schema-mapping-ner)
    """)