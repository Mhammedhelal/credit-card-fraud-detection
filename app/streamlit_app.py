"""
Streamlit Dashboard — Credit Card Fraud Detection
=====================================================
Single-page UI that talks to the FastAPI service over HTTP for scoring,
and reads the model bundle directly (same MODEL_PATH env var) for the
threshold-sensitivity chart, since the saved PR curve lives in the model's
metadata and doesn't need a network round-trip to visualize.

Run:
    API_BASE_URL=http://localhost:8000 streamlit run app/streamlit_app.py
"""

import os
import json
import random

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
MODEL_PATH = os.environ.get("MODEL_PATH", "models/trained_model.pkl")

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")


# ──────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model_bundle():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def check_api_health():
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return resp.json()
    except requests.exceptions.RequestException:
        return None


bundle = load_model_bundle()
health = check_api_health()

st.title("🔍 Credit Card Fraud Detection")

col_status_1, col_status_2 = st.columns(2)
with col_status_1:
    if health and health.get("status") == "ok":
        st.success(f"API: {API_BASE_URL} — {health['model_type']} v{health['model_version']}")
    elif health:
        st.warning(f"API reachable but degraded: {health}")
    else:
        st.error(f"API unreachable at {API_BASE_URL}. Is `uvicorn api.main:app` running?")
with col_status_2:
    if bundle is None:
        st.error(f"No model bundle found at MODEL_PATH={MODEL_PATH}")
    else:
        st.info(f"Local bundle loaded for charts: {bundle.get('model_type', 'unknown')}")

st.divider()

# ──────────────────────────────────────────────────────────────────────────
# THRESHOLD CONTROL + LIVE PRECISION/RECALL
# ──────────────────────────────────────────────────────────────────────────

st.header("1. Threshold Sensitivity")

left, right = st.columns([1, 2])

with left:
    threshold = st.slider(
        "Fraud probability threshold",
        min_value=0.10, max_value=0.90, value=0.50, step=0.01,
        help="Transactions scored above this probability are flagged as fraud."
    )

    st.markdown("**Business cost inputs**")
    daily_txns = st.number_input("Estimated transactions / day", min_value=1, value=50000, step=1000)
    fraud_rate = st.number_input("Assumed fraud rate", min_value=0.0001, max_value=0.05,
                                  value=0.0018, step=0.0001, format="%.4f")
    avg_fraud_amount = st.number_input("Avg $ per missed fraud (FN)", min_value=1.0, value=120.0)
    avg_review_cost = st.number_input("Avg $ cost per false alarm (FP review)", min_value=0.1, value=5.0)

pr_curve = None
if bundle is not None:
    threshold_info = bundle.get("metadata", {}).get("threshold_info", {})
    pr_curve = threshold_info.get("pr_curve") if isinstance(threshold_info, dict) else None

with right:
    if pr_curve:
        precisions = np.array(pr_curve["precisions"])
        recalls = np.array(pr_curve["recalls"])
        thresholds = np.array(pr_curve["thresholds"])

        idx = int(np.argmin(np.abs(thresholds - threshold)))
        chosen_precision = float(precisions[idx])
        chosen_recall = float(recalls[idx])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=thresholds, y=precisions, name="Precision",
                                  line=dict(color="#2471A3")))
        fig.add_trace(go.Scatter(x=thresholds, y=recalls, name="Recall",
                                  line=dict(color="#C0392B")))
        fig.add_vline(x=threshold, line_dash="dash", line_color="#EB0303",
                      annotation_text=f"threshold={threshold:.2f}")
        fig.update_layout(
            title="Precision & Recall vs Threshold (from validation PR curve)",
            xaxis_title="Threshold", yaxis_title="Score",
            yaxis_range=[0, 1.05], height=380,
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Precision @ threshold", f"{chosen_precision:.3f}")
        m2.metric("Recall @ threshold", f"{chosen_recall:.3f}")
        f1 = 2 * chosen_precision * chosen_recall / (chosen_precision + chosen_recall + 1e-9)
        m3.metric("F1 @ threshold", f"{f1:.3f}")

        # ── Business cost estimate ─────────────────────────────────────────
        daily_fraud_count = daily_txns * fraud_rate
        daily_normal_count = daily_txns - daily_fraud_count

        caught = daily_fraud_count * chosen_recall
        missed = daily_fraud_count * (1 - chosen_recall)
        flagged_total = caught / chosen_precision if chosen_precision > 0 else 0
        false_alarms = max(flagged_total - caught, 0)

        missed_fraud_cost = missed * avg_fraud_amount
        false_alarm_cost = false_alarms * avg_review_cost
        total_daily_cost = missed_fraud_cost + false_alarm_cost

        st.subheader("💰 Estimated daily cost at this threshold")
        c1, c2, c3 = st.columns(3)
        c1.metric("Missed fraud losses", f"${missed_fraud_cost:,.0f}", f"{missed:.1f} txns")
        c2.metric("False-alarm review cost", f"${false_alarm_cost:,.0f}", f"{false_alarms:.1f} txns")
        c3.metric("Total estimated daily cost", f"${total_daily_cost:,.0f}")
    else:
        st.warning("Model bundle has no stored PR curve (threshold_info.pr_curve). "
                   "Retrain with training.py's automatic threshold selection to populate this chart.")

st.divider()

# ──────────────────────────────────────────────────────────────────────────
# CONFUSION MATRIX AT CHOSEN THRESHOLD
# ──────────────────────────────────────────────────────────────────────────

if pr_curve:
    st.header("2. Confusion Matrix at Chosen Threshold (validation, estimated)")

    val_shape = bundle.get("metadata", {}).get("val_shape", [0, 0])
    n_val = val_shape[0] if val_shape else 0
    scale_pos_weight = bundle.get("scale_pos_weight")

    if n_val and scale_pos_weight:
        # The model bundle doesn't store the validation set's actual fraud count,
        # only its row count (val_shape) and the TRAINING class ratio
        # (scale_pos_weight = n_negative / n_positive). We assume validation
        # prevalence matches training prevalence — a reasonable assumption since
        # both come from the same stratified split, but an assumption nonetheless.
        n_fraud_est = n_val / (1 + scale_pos_weight)
        n_normal_est = n_val - n_fraud_est

        tp = chosen_recall * n_fraud_est
        fn = n_fraud_est - tp
        fp = tp * (1 / chosen_precision - 1) if chosen_precision > 0 else n_normal_est
        fp = min(fp, n_normal_est)  # clamp — the approximation can overshoot as precision→0
        tn = n_normal_est - fp

        matrix = np.array([[tn, fp], [fn, tp]])
        row_sums = matrix.sum(axis=1, keepdims=True)
        norm_matrix = matrix / np.clip(row_sums, 1, None)

        row_names = ["Actual Normal", "Actual Fraud"]
        col_names = ["Predicted Normal", "Predicted Fraud"]
        cell_labels = [["TN", "FP"], ["FN", "TP"]]

        fig_cm = go.Figure(data=go.Heatmap(
            z=norm_matrix,
            x=col_names,
            y=row_names,
            colorscale=[[0, "#EAF2F8"], [1, "#2471A3"]],
            zmin=0, zmax=1,
            showscale=False,
        ))
        fig_cm.update_layout(
            title=f"Estimated Confusion Matrix — {n_val:,} validation transactions "
                  f"(~{n_fraud_est:.0f} fraud at training-set prevalence)",
            height=380,
            annotations=[
                dict(
                    x=col_names[j], y=row_names[i],
                    text=f"{cell_labels[i][j]}<br>{matrix[i, j]:,.0f} ({norm_matrix[i, j]:.1%})",
                    showarrow=False,
                    font=dict(color="white" if norm_matrix[i, j] > 0.6 else "black"),
                )
                for i in range(2) for j in range(2)
            ],
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        cm1, cm2, cm3, cm4 = st.columns(4)
        cm1.metric("True Positives (caught fraud)", f"{tp:,.0f}")
        cm2.metric("False Negatives (missed fraud)", f"{fn:,.0f}")
        cm3.metric("False Positives (false alarms)", f"{fp:,.0f}")
        cm4.metric("True Negatives", f"{tn:,.0f}")
    else:
        st.warning(
            "Model bundle is missing `val_shape` or `scale_pos_weight` in its metadata "
            "— can't estimate a confusion matrix. Retrain with the current training.py "
            "to populate these fields."
        )

    st.caption(
        "Counts are estimated from the stored validation PR curve and training-set "
        "class prevalence, not recomputed by re-running the model on raw validation "
        "data. For exact counts, run `testing.py` against the validation split."
    )

st.divider()

# ──────────────────────────────────────────────────────────────────────────
# SCORE A TRANSACTION
# ──────────────────────────────────────────────────────────────────────────

st.header("3. Score a Transaction")

V_FEATURES = [f"V{i}" for i in range(1, 29)]

tab_form, tab_json = st.tabs(["Quick form", "Paste JSON"])

if "sample_txn" not in st.session_state:
    st.session_state.sample_txn = {f: 0.0 for f in V_FEATURES}
    st.session_state.sample_txn.update({"Time": 0.0, "Amount": 0.0})

with tab_form:
    if st.button("🎲 Fill with random plausible values"):
        rng = random.Random()
        st.session_state.sample_txn = {
            "Time": float(rng.randint(0, 172792)),
            "Amount": round(rng.uniform(0, 500), 2),
            **{f: round(rng.gauss(0, 1.5), 3) for f in V_FEATURES},
        }

    c1, c2 = st.columns(2)
    with c1:
        time_val = st.number_input("Time (seconds elapsed)", value=st.session_state.sample_txn["Time"])
    with c2:
        amount_val = st.number_input("Amount ($)", value=st.session_state.sample_txn["Amount"], min_value=0.0)

    with st.expander("V1 – V28 (PCA components)"):
        v_cols = st.columns(4)
        v_values = {}
        for i, feat in enumerate(V_FEATURES):
            with v_cols[i % 4]:
                v_values[feat] = st.number_input(
                    feat, value=st.session_state.sample_txn[feat], key=f"form_{feat}"
                )

    if st.button("Score transaction", type="primary"):
        payload = {"Time": time_val, "Amount": amount_val, **v_values}
        try:
            resp = requests.post(
                f"{API_BASE_URL}/predict", json=payload,
                params={"threshold": threshold}, timeout=10,
            )
            if resp.status_code == 200:
                st.session_state.last_prediction = resp.json()
            else:
                st.error(f"API error {resp.status_code}: {resp.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach API: {e}")

with tab_json:
    default_json = json.dumps(
        {"Time": 406.0, "Amount": 0.0, **{f: 0.0 for f in V_FEATURES}}, indent=2
    )
    raw_json = st.text_area("Transaction JSON", value=default_json, height=200)
    if st.button("Score JSON transaction"):
        try:
            payload = json.loads(raw_json)
            resp = requests.post(
                f"{API_BASE_URL}/predict", json=payload,
                params={"threshold": threshold}, timeout=10,
            )
            if resp.status_code == 200:
                st.session_state.last_prediction = resp.json()
            else:
                st.error(f"API error {resp.status_code}: {resp.text}")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach API: {e}")

# ──────────────────────────────────────────────────────────────────────────
# PREDICTION RESULT + SHAP WATERFALL
# ──────────────────────────────────────────────────────────────────────────

if "last_prediction" in st.session_state:
    pred = st.session_state.last_prediction
    st.subheader("Result")

    r1, r2, r3 = st.columns(3)
    label = "🚨 FRAUD" if pred["prediction"] == 1 else "✅ Normal"
    r1.metric("Prediction", label)
    r2.metric("Fraud probability", f"{pred['fraud_probability']:.4f}")
    r3.metric("Confidence", pred["confidence"])

    shap_data = pred.get("shap_explanation", [])
    if shap_data:
        st.markdown("**SHAP explanation** — features pushing this prediction toward/away from fraud")
        df_shap = pd.DataFrame(shap_data).sort_values("shap_value")

        fig_shap = go.Figure(go.Bar(
            x=df_shap["shap_value"],
            y=df_shap["feature"],
            orientation="h",
            marker_color=["#C0392B" if v > 0 else "#2471A3" for v in df_shap["shap_value"]],
        ))
        fig_shap.update_layout(
            title="Top feature contributions (red = toward fraud, blue = toward normal)",
            xaxis_title="SHAP value", height=400,
        )
        st.plotly_chart(fig_shap, use_container_width=True)
    else:
        st.caption("No SHAP explanation available (shap not installed on the API, or non-tree model).")