from pathlib import Path
import json
import sqlite3

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except ImportError:
    px = None

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "predictions.db"
METRICS_PATH = ROOT / "reports" / "metrics.json"
DRIFT_PATH = ROOT / "reports" / "drift_report.json"

st.set_page_config(
    page_title="Fraud Detection Monitoring",
    page_icon="🛡️",
    layout="wide",
)

st.title("Fraud Detection Monitoring")
st.caption("Transaction risk, model performance, and input-data monitoring.")

with st.sidebar:
    st.subheader("Controls")
    if st.button("Refresh dashboard", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


def modified_time(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(ttl=10)
def load_predictions(db_path: str, db_mtime: float) -> pd.DataFrame:
    del db_mtime
    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(path) as conn:
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table'", conn
            )["name"].tolist()
            if "predictions" not in tables:
                return pd.DataFrame()
            return pd.read_sql_query(
                "SELECT * FROM predictions ORDER BY created_at DESC LIMIT 1000",
                conn,
            )
    except (sqlite3.Error, pd.errors.DatabaseError):
        return pd.DataFrame()


@st.cache_data(ttl=30)
def load_json(path_string: str, mtime: float) -> dict:
    del mtime
    try:
        with open(path_string, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


predictions = load_predictions(str(DB_PATH), modified_time(DB_PATH))
metrics_report = load_json(str(METRICS_PATH), modified_time(METRICS_PATH))
drift_report = load_json(str(DRIFT_PATH), modified_time(DRIFT_PATH))


def number_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


# KPI summary
total = len(predictions)
if total and "predicted_class" in predictions.columns:
    predicted = number_series(predictions, "predicted_class").fillna(0).astype(int)
    fraud_count = int((predicted == 1).sum())
    normal_count = int((predicted == 0).sum())
else:
    predicted = pd.Series(dtype=int)
    fraud_count = 0
    normal_count = 0

alert_rate = (fraud_count / total * 100) if total else 0.0
if total and "latency_ms" in predictions.columns:
    avg_latency = number_series(predictions, "latency_ms").mean()
else:
    avg_latency = None

k1, k2, k3, k4 = st.columns(4)
k1.metric("Transactions scored", f"{total:,}")
k2.metric("Fraud alerts", f"{fraud_count:,}")
k3.metric("Alert rate", f"{alert_rate:.2f}%")
k4.metric(
    "Avg. inference latency",
    f"{avg_latency:.2f} ms" if pd.notna(avg_latency) else "—",
)

st.divider()

# Transaction monitoring
left, right = st.columns([0.85, 1.4], gap="large")

with left:
    st.subheader("Prediction distribution")
    if total:
        chart_data = pd.DataFrame(
            {
                "Classification": ["Not flagged", "Flagged"],
                "Transactions": [normal_count, fraud_count],
            }
        )
        if px is not None:
            fig = px.pie(
                chart_data,
                names="Classification",
                values="Transactions",
                hole=0.62,
                color="Classification",
                color_discrete_map={
                    "Not flagged": "#64748b",
                    "Flagged": "#dc2626",
                },
            )
            fig.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                legend_title_text="",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(chart_data.set_index("Classification"))
    else:
        st.info("No prediction records available.")

with right:
    st.subheader("Recent transactions")
    if total:
        preferred = [
            "prediction_id",
            "created_at",
            "fraud_probability",
            "predicted_class",
            "latency_ms",
        ]
        columns = [column for column in preferred if column in predictions.columns]
        if columns:
            recent = predictions[columns].head(10).copy()
            if "predicted_class" in recent.columns:
                recent["Status"] = number_series(recent, "predicted_class").map(
                    {0: "Not flagged", 1: "Flagged"}
                )
                recent = recent.drop(columns=["predicted_class"])
            if "fraud_probability" in recent.columns:
                recent["fraud_probability"] = number_series(
                    recent, "fraud_probability"
                ).round(4)
            if "latency_ms" in recent.columns:
                recent["latency_ms"] = number_series(recent, "latency_ms").round(2)
            st.dataframe(
                recent,
                use_container_width=True,
                hide_index=True,
                height=380,
            )
        else:
            st.info("Prediction records exist, but expected columns are missing.")
    else:
        st.info("No prediction records available.")

# Model evaluation
st.divider()
st.subheader("Model evaluation")

test_metrics = metrics_report.get("test_metrics", metrics_report)
metric_specs = [
    ("Average precision", "test_average_precision"),
    ("ROC-AUC", "test_roc_auc"),
    ("Precision", "test_precision"),
    ("Recall", "test_recall"),
]
available_metrics = []
for label, key in metric_specs:
    value = test_metrics.get(key)
    if isinstance(value, (int, float)):
        available_metrics.append((label, float(value)))

if available_metrics:
    cols = st.columns(len(available_metrics))
    for col, (label, value) in zip(cols, available_metrics):
        col.metric(label, f"{value:.3f}")

    model_name = metrics_report.get("model_name")
    threshold = test_metrics.get("threshold")
    metadata = []
    if model_name:
        metadata.append(f"Model: {model_name}")
    if isinstance(threshold, (int, float)):
        metadata.append(f"Decision threshold: {threshold:.3f}")
    if metadata:
        st.caption(" · ".join(metadata))
else:
    st.info("Model evaluation report is not available.")

# Data drift
st.divider()
st.subheader("Data drift")

if not drift_report:
    st.info("Drift report is not available.")
else:
    drifted = drift_report.get("drifted_feature_count")
    checked = drift_report.get("features_checked")
    current_rows = drift_report.get("current_rows")
    reference_rows = drift_report.get("reference_rows")

    d1, d2, d3 = st.columns(3)
    d1.metric("Features monitored", str(checked) if checked is not None else "—")
    d2.metric("Features with drift", str(drifted) if drifted is not None else "—")
    d3.metric(
        "Recent sample size",
        f"{current_rows:,}" if isinstance(current_rows, int) else "—",
    )

    feature_results = drift_report.get("features", {})
    rows = []
    if isinstance(feature_results, dict):
        for feature, result in feature_results.items():
            if not isinstance(result, dict):
                continue
            drift_detected = result.get("drift_detected")
            if drift_detected is True:
                status = "Detected"
            elif drift_detected is False:
                status = "Not detected"
            else:
                status = "Insufficient data"
            rows.append(
                {
                    "Feature": feature,
                    "KS statistic": result.get("ks_statistic"),
                    "P-value": result.get("p_value"),
                    "Status": status,
                }
            )

    if rows:
        drift_df = pd.DataFrame(rows)
        drift_df["KS statistic"] = pd.to_numeric(
            drift_df["KS statistic"], errors="coerce"
        )
        chart_data = drift_df.dropna(subset=["KS statistic"])
        if not chart_data.empty:
            if px is not None:
                fig = px.bar(
                    chart_data,
                    x="Feature",
                    y="KS statistic",
                    color="Status",
                    color_discrete_map={
                        "Detected": "#dc2626",
                        "Not detected": "#16a34a",
                        "Insufficient data": "#94a3b8",
                    },
                )
                fig.add_hline(
                    y=0.10,
                    line_dash="dash",
                    line_color="#64748b",
                    annotation_text="Alert threshold (0.10)",
                )
                fig.update_layout(margin=dict(t=20, b=10, l=10, r=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(chart_data.set_index("Feature")["KS statistic"])
        st.dataframe(drift_df, use_container_width=True, hide_index=True)

    if reference_rows is not None and current_rows is not None:
        st.caption(
            f"Reference sample: {reference_rows:,} rows · "
            f"Current sample: {current_rows:,} rows"
        )

st.caption("Fraud alerts are model predictions, not confirmed fraud cases.")
