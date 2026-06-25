import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_raw_data
from src.input_validator import validate_transaction_dataframe
from src.agents.fraud_decision_workflow import (
    run_fraud_decision_workflow_with_context,
)


DEMO_TRANSACTIONS = {
    "Low-Risk Example": 0,
    "Medium-Risk Example": 1685,
    "High-Risk Example": 306221,
}


DISPLAY_COLUMNS = [
    "trans_date_trans_time",
    "merchant",
    "category",
    "amt",
    "gender",
    "city",
    "state",
    "city_pop",
    "job",
    "dob",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
]

MODEL_COMPARISON_PATH = PROJECT_ROOT / "reports" / "real_world_model_comparison.csv"
GROUPED_IMPORTANCE_PATH = PROJECT_ROOT / "reports" / "xgboost_feature_importance_grouped.csv"


# ---------------------------------------------------------------------------
# Theming / UI helpers
# ---------------------------------------------------------------------------

def inject_global_styles():
    """
    Inject custom CSS for a modern, dark-mode-friendly fintech dashboard look.
    """
    st.markdown(
        """
        <style>
        /* Tighten the default Streamlit top padding for a denser layout */
        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }

        /* Brand block in the sidebar */
        .brand-wrap {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            padding: 0.25rem 0 0.75rem 0;
        }
        .brand-logo {
            font-size: 1.55rem;
            line-height: 1;
        }
        .brand-name {
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: 0.2px;
        }
        .brand-tag {
            font-size: 0.72rem;
            color: #8b949e;
            margin-top: 1px;
        }

        /* Main header */
        .app-title {
            font-size: 2.0rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            margin: 0 0 0.25rem 0;
        }
        .app-subtitle {
            color: #9aa4b2;
            font-size: 0.98rem;
            max-width: 760px;
            margin-bottom: 0.9rem;
        }

        /* Tech badge row */
        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 0.4rem;
        }
        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.76rem;
            font-weight: 600;
            color: #c9d3e0;
            background: rgba(125, 145, 180, 0.12);
            border: 1px solid rgba(125, 145, 180, 0.25);
            padding: 0.28rem 0.65rem;
            border-radius: 999px;
        }
        .badge .dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #3b82f6;
        }

        /* Section headers */
        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin: 0.2rem 0 0.1rem 0;
        }
        .section-sub {
            color: #9aa4b2;
            font-size: 0.86rem;
            margin-bottom: 0.6rem;
        }

        /* KPI cards */
        .kpi-card {
            border: 1px solid rgba(125, 145, 180, 0.22);
            border-radius: 14px;
            padding: 1.0rem 1.1rem;
            background: rgba(125, 145, 180, 0.06);
            height: 100%;
        }
        .kpi-label {
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            color: #8b949e;
            margin-bottom: 0.35rem;
        }
        .kpi-value {
            font-size: 1.9rem;
            font-weight: 800;
            line-height: 1.1;
        }
        .kpi-sub {
            font-size: 0.78rem;
            color: #9aa4b2;
            margin-top: 0.3rem;
        }
        .kpi-delta-up { color: #22c55e; font-weight: 700; }
        .kpi-delta-down { color: #ef4444; font-weight: 700; }

        /* Status pill used on the decision header */
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.82rem;
            font-weight: 700;
            padding: 0.32rem 0.8rem;
            border-radius: 999px;
        }

        /* Intro / info panel */
        .panel {
            border: 1px solid rgba(125, 145, 180, 0.22);
            border-radius: 14px;
            padding: 1.0rem 1.2rem;
            background: rgba(125, 145, 180, 0.05);
            margin-bottom: 0.4rem;
        }
        .panel h4 {
            margin: 0 0 0.35rem 0;
            font-size: 1.0rem;
            font-weight: 700;
        }
        .panel p {
            color: #9aa4b2;
            font-size: 0.9rem;
            margin: 0;
        }

        /* Action card with a colored accent bar */
        .action-card {
            border: 1px solid rgba(125, 145, 180, 0.22);
            border-left-width: 4px;
            border-radius: 12px;
            padding: 0.9rem 1.1rem;
            background: rgba(125, 145, 180, 0.05);
        }
        .action-card .action-label {
            font-size: 0.74rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            color: #8b949e;
            margin-bottom: 0.25rem;
        }
        .action-card .action-body {
            font-size: 0.96rem;
            font-weight: 600;
        }

        /* About page feature grid */
        .about-card {
            border: 1px solid rgba(125, 145, 180, 0.22);
            border-radius: 14px;
            padding: 1.0rem 1.15rem;
            background: rgba(125, 145, 180, 0.05);
            height: 100%;
        }
        .about-card h4 {
            margin: 0 0 0.4rem 0;
            font-size: 0.98rem;
            font-weight: 700;
        }
        .about-card ul {
            margin: 0;
            padding-left: 1.05rem;
            color: #9aa4b2;
            font-size: 0.88rem;
        }
        .about-card p {
            margin: 0;
            color: #9aa4b2;
            font-size: 0.88rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_badges(items):
    """
    Render a row of small pill-style tech badges.
    """
    badges = "".join(
        f'<span class="badge"><span class="dot"></span>{item}</span>'
        for item in items
    )
    st.markdown(f'<div class="badge-row">{badges}</div>', unsafe_allow_html=True)


def render_main_header():
    """
    Render the product-style main header with title, subtitle and tech badges.
    """
    st.markdown(
        '<div class="app-title">🛡️ Fraud Risk Decisioning Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="app-subtitle">Analyze transactions, explain fraud risk, and '
        "generate policy-guided decisions for approval, review, or escalation.</div>",
        unsafe_allow_html=True,
    )
    render_badges(
        [
            "XGBoost Model",
            "SHAP Explainability",
            "Agentic Workflow",
            "MLflow Tracked",
        ]
    )
    st.divider()


def render_section_header(title: str, subtitle: str = ""):
    """
    Render a consistent section header with an optional subtitle.
    """
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


def render_intro_panel(title: str, body: str):
    """
    Render an intro panel card with a title and short description.
    """
    st.markdown(
        f'<div class="panel"><h4>{title}</h4><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


def get_risk_palette(risk_band: str) -> dict:
    """
    Return accent color and tinted background for a given risk band.
    """
    band = (risk_band or "").lower()

    if "high" in band:
        return {"color": "#ef4444", "tint": "rgba(239, 68, 68, 0.10)"}
    if "medium" in band:
        return {"color": "#f59e0b", "tint": "rgba(245, 158, 11, 0.12)"}
    if "low" in band:
        return {"color": "#22c55e", "tint": "rgba(34, 197, 94, 0.10)"}

    return {"color": "#3b82f6", "tint": "rgba(59, 130, 246, 0.10)"}


def get_recommendation_palette(recommendation: str) -> dict:
    """
    Return accent color and tinted background for a recommendation.
    """
    if recommendation == "Deny":
        return {"color": "#ef4444", "tint": "rgba(239, 68, 68, 0.10)"}
    if recommendation == "Manual Review":
        return {"color": "#f59e0b", "tint": "rgba(245, 158, 11, 0.12)"}
    return {"color": "#22c55e", "tint": "rgba(34, 197, 94, 0.10)"}


def render_kpi_card(label, value, color, tint, sub=""):
    """
    Render a single KPI card with custom risk-based styling.
    """
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="kpi-card" style="background:{tint}; border-color:{color};">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="color:{color};">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label, value, delta_text=None, delta_positive=True, sub=""):
    """
    Render a neutral metric card used on the Model Performance page.
    """
    delta_html = ""
    if delta_text:
        cls = "kpi-delta-up" if delta_positive else "kpi-delta-down"
        arrow = "▲" if delta_positive else "▼"
        delta_html = f'<div class="kpi-sub"><span class="{cls}">{arrow} {delta_text}</span> vs baseline</div>'
    elif sub:
        delta_html = f'<div class="kpi-sub">{sub}</div>'

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_default_data():
    """
    Load the default real-world fraud transaction dataset.
    """
    return load_raw_data()


def reset_results():
    """
    Clear saved workflow results when input changes.
    """
    st.session_state["workflow_result"] = None
    st.session_state["selected_transaction"] = None
    st.session_state["selection_key"] = None


def clear_uploaded_data():
    """
    Clear saved uploaded CSV data.
    """
    st.session_state["uploaded_clean_df"] = None
    st.session_state["uploaded_file_name"] = None
    reset_results()


def initialize_session_state():
    """
    Initialize app session state.
    """
    if "uploaded_clean_df" not in st.session_state:
        st.session_state["uploaded_clean_df"] = None

    if "uploaded_file_name" not in st.session_state:
        st.session_state["uploaded_file_name"] = None

    if "workflow_result" not in st.session_state:
        st.session_state["workflow_result"] = None

    if "selected_transaction" not in st.session_state:
        st.session_state["selected_transaction"] = None

    if "selection_key" not in st.session_state:
        st.session_state["selection_key"] = None

    if "input_mode" not in st.session_state:
        st.session_state["input_mode"] = None


def get_risk_status_icon(recommendation: str) -> str:
    """
    Return a simple icon for the recommendation.
    """
    if recommendation == "Deny":
        return "🚫"
    if recommendation == "Manual Review":
        return "🕵️"
    return "✅"


def display_selected_transaction(transaction: pd.DataFrame):
    """
    Display selected transaction details.
    """
    available_columns = [
        column for column in DISPLAY_COLUMNS if column in transaction.columns
    ]

    st.dataframe(
        transaction[available_columns],
        use_container_width=True,
    )


def escape_markdown_text(text: str) -> str:
    """
    Escape characters that Streamlit Markdown may interpret as formatting.
    This prevents dollar amounts and underscores from breaking summary text.
    """
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace("$", "\\$")
        .replace("_", "\\_")
        .replace("*", "\\*")
        .replace("`", "\\`")
    )


def format_feature_name(feature_name: str) -> str:
    """
    Convert technical feature names into readable labels.
    """
    readable_names = {
        "amt": "Transaction Amount",
        "category": "Merchant Category",
        "gender": "Customer Gender",
        "state": "Customer State",
        "job": "Customer Job",
        "transaction_hour": "Transaction Hour",
        "transaction_day_of_week": "Transaction Day of Week",
        "transaction_month": "Transaction Month",
        "is_weekend": "Weekend Transaction",
        "customer_age": "Customer Age",
        "city_pop": "City Population",
        "distance_from_home_km": "Distance From Home",
        "time_since_last_transaction_minutes": "Time Since Last Transaction",
        "distance_from_last_transaction_km": "Distance From Previous Transaction",
        "implied_travel_speed_kmh": "Implied Travel Speed",
    }

    return readable_names.get(feature_name, feature_name.replace("_", " ").title())


def format_shap_feature_value(feature_name: str, value):
    """
    Format SHAP feature values for display.
    """
    if value == "N/A":
        return value

    if feature_name in ["distance_from_home_km", "distance_from_last_transaction_km"]:
        try:
            return f"{float(value) * 0.621371:,.1f} miles"
        except Exception:
            return value

    if feature_name == "implied_travel_speed_kmh":
        try:
            return f"{float(value) * 0.621371:,.1f} mph"
        except Exception:
            return value

    if feature_name == "amt":
        try:
            return f"${float(value):,.2f}"
        except Exception:
            return value

    return value


def build_shap_driver_sentence(shap_explanation: dict, risk_band: str) -> str:
    """
    Build a natural sentence from local SHAP drivers.
    """
    if not shap_explanation or shap_explanation.get("error"):
        return ""

    risk_increasing = shap_explanation.get("risk_increasing_features", [])
    risk_reducing = shap_explanation.get("risk_reducing_features", [])

    def format_driver_list(drivers):
        driver_names = [
            format_feature_name(driver["base_feature"])
            for driver in drivers[:3]
        ]

        if len(driver_names) == 0:
            return ""

        if len(driver_names) == 1:
            return driver_names[0]

        if len(driver_names) == 2:
            return f"{driver_names[0]} and {driver_names[1]}"

        return f"{driver_names[0]}, {driver_names[1]}, and {driver_names[2]}"

    if risk_band == "Low Risk":
        driver_text = format_driver_list(risk_reducing)

        if driver_text:
            return (
                f"Model attribution indicates that {driver_text} helped keep the "
                "model score within the low-risk range for this transaction."
            )

        return (
            "Model attribution did not identify a dominant risk-increasing driver, "
            "and the overall transaction profile remained low risk."
        )

    driver_text = format_driver_list(risk_increasing)

    if driver_text:
        return (
            f"Model attribution indicates that {driver_text} were the strongest "
            "risk-increasing drivers for this specific transaction."
        )

    return (
        "Model attribution did not identify a dominant risk-increasing driver "
        "for this transaction."
    )


def display_shap_driver_expander(shap_explanation: dict):
    """
    Display local SHAP drivers in an optional expander.
    """
    if not shap_explanation:
        return

    with st.expander("Transaction-specific model drivers (SHAP)", expanded=False):
        if shap_explanation.get("error"):
            st.warning("SHAP explanation could not be generated.")
            st.caption(shap_explanation["error"])
            return

        top_features = shap_explanation.get("top_features", [])

        if not top_features:
            st.info("No transaction-specific model drivers available.")
            return

        st.caption(
            "Local SHAP drivers for this exact transaction. "
            "**Increased fraud risk** means the feature pushed the score higher; "
            "**reduced fraud risk** means it pushed the score lower."
        )

        driver_rows = []

        for feature in top_features:
            base_feature = feature["base_feature"]

            driver_rows.append(
                {
                    "Feature": format_feature_name(base_feature),
                    "Value": format_shap_feature_value(
                        base_feature,
                        feature["feature_value"],
                    ),
                    "Impact": feature["impact"],
                    "Impact Strength": round(feature["abs_shap_value"], 4),
                }
            )

        st.dataframe(
            pd.DataFrame(driver_rows),
            use_container_width=True,
            hide_index=True,
        )


def render_agent_workflow_trace(
    risk_result: dict,
    investigation_result: dict,
    policy_result: dict,
    final_decision: dict,
):
    """
    Render a clean, readable technical trace of each agent's output using
    Streamlit-native components instead of raw JSON blocks.
    """
    render_section_header(
        "Agent Workflow Trace",
        "Technical step-by-step output from each agent in the decision pipeline.",
    )

    with st.expander("1 · Risk Scoring Agent", expanded=False):
        st.caption("Scores the transaction with the deployed XGBoost model.")
        trace_col1, trace_col2, trace_col3 = st.columns(3)
        trace_col1.metric(
            "Fraud Probability",
            f"{risk_result['fraud_probability']:.2%}",
        )
        trace_col2.metric("Risk Band", risk_result["risk_band"])
        trace_col3.metric("Recommendation", risk_result["recommendation"])

    with st.expander("2 · Investigation Agent", expanded=False):
        st.caption("Builds the analyst-style narrative and key risk signals.")
        st.markdown("**Summary**")
        st.markdown(escape_markdown_text(investigation_result["summary"]))

        key_risk_signals = investigation_result.get("key_risk_signals", [])
        if key_risk_signals:
            st.markdown("**Key risk signals**")
            for signal in key_risk_signals:
                st.markdown(f"- {escape_markdown_text(signal)}")

    with st.expander("3 · Policy Agent", expanded=False):
        st.caption("Maps the recommendation to supporting fraud policy guidance.")
        st.markdown(
            f"**Recommendation:** {escape_markdown_text(policy_result['recommendation'])}"
        )
        st.markdown("**Policy guidance**")
        st.markdown(escape_markdown_text(policy_result["policy_guidance"]))

    with st.expander("4 · Decision Agent", expanded=False):
        st.caption("Consolidates every agent into the final decision report.")
        st.text(final_decision["final_report"])


def display_decision_dashboard(workflow_result: dict, show_workflow_details: bool):
    """
    Display fraud risk decision results.
    """
    risk_result = workflow_result["risk_result"]
    investigation_result = workflow_result["investigation_result"]
    policy_result = workflow_result["policy_result"]
    final_decision = workflow_result["final_decision"]

    fraud_probability = risk_result["fraud_probability"]
    risk_band = risk_result["risk_band"]
    recommendation = final_decision["final_recommendation"]
    business_action = final_decision["business_action"]
    status_icon = get_risk_status_icon(recommendation)

    risk_palette = get_risk_palette(risk_band)
    rec_palette = get_recommendation_palette(recommendation)

    st.divider()

    header_col, pill_col = st.columns([0.7, 0.3])
    with header_col:
        render_section_header(
            "Fraud Risk Decision",
            "Model score, risk band and the policy-guided recommendation for this transaction.",
        )
    with pill_col:
        st.markdown(
            f"""
            <div style="display:flex; justify-content:flex-end; padding-top:0.2rem;">
                <span class="status-pill" style="background:{rec_palette['tint']};
                    color:{rec_palette['color']}; border:1px solid {rec_palette['color']};">
                    {status_icon} {recommendation}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        render_kpi_card(
            "Fraud Probability",
            f"{fraud_probability:.2%}",
            risk_palette["color"],
            risk_palette["tint"],
            sub="Model-estimated likelihood of fraud",
        )

    with col2:
        render_kpi_card(
            "Risk Band",
            risk_band,
            risk_palette["color"],
            risk_palette["tint"],
            sub="Calibrated risk tier",
        )

    with col3:
        render_kpi_card(
            "Recommendation",
            recommendation,
            rec_palette["color"],
            rec_palette["tint"],
            sub="Policy-guided next step",
        )

    st.write("")

    with st.container(border=True):
        render_section_header("AI Analyst Summary")
        st.markdown(escape_markdown_text(investigation_result["summary"]))

        shap_sentence = build_shap_driver_sentence(
            workflow_result.get("shap_explanation"),
            risk_band=risk_band,
        )

        if shap_sentence:
            st.markdown(escape_markdown_text(shap_sentence))

        display_shap_driver_expander(
            workflow_result.get("shap_explanation")
        )

    st.write("")

    action_col, policy_col = st.columns(2)

    with action_col:
        st.markdown(
            f"""
            <div class="action-card" style="border-left-color:{rec_palette['color']};
                background:{rec_palette['tint']};">
                <div class="action-label">Recommended Business Action</div>
                <div class="action-body">{escape_markdown_text(business_action)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with policy_col:
        with st.container(border=True):
            render_section_header("Policy Support")
            st.markdown(escape_markdown_text(policy_result["policy_guidance"]))

    if show_workflow_details:
        st.write("")
        render_agent_workflow_trace(
            risk_result=risk_result,
            investigation_result=investigation_result,
            policy_result=policy_result,
            final_decision=final_decision,
        )


def render_feature_importance_chart(top_importance: pd.DataFrame):
    """
    Render a clean horizontal bar chart of global feature importance using Plotly.
    """
    chart_df = top_importance.sort_values("Importance", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=chart_df["Importance"],
            y=chart_df["Feature"],
            orientation="h",
            marker=dict(
                color=chart_df["Importance"],
                colorscale="Blues",
                line=dict(width=0),
            ),
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Importance", showgrid=True, gridcolor="rgba(125,145,180,0.15)"),
        yaxis=dict(title=""),
        font=dict(size=13),
    )

    st.plotly_chart(fig, use_container_width=True)


def display_model_performance_page():
    """
    Display model comparison metrics and global feature importance.
    """
    render_section_header(
        "Model Performance & Explainability",
        "Why XGBoost is the deployed model, and which features it relies on most.",
    )

    render_intro_panel(
        "Model evaluation overview",
        "This page benchmarks the deployed XGBoost model against a Logistic Regression "
        "baseline on recall, precision, PR-AUC and false positives, then shows the "
        "model's global feature importance.",
    )

    if not MODEL_COMPARISON_PATH.exists():
        st.warning(
            "Model comparison report not found. Run `python -m src.train_model` first."
        )
        return

    comparison_df = pd.read_csv(MODEL_COMPARISON_PATH)

    logistic = comparison_df[comparison_df["model"] == "Logistic Regression"].iloc[0]
    xgboost = comparison_df[comparison_df["model"] == "XGBoost"].iloc[0]

    st.write("")
    render_section_header("XGBoost vs Logistic Regression")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        recall_delta = xgboost["recall"] - logistic["recall"]
        render_metric_card(
            "Recall",
            f"{xgboost['recall']:.2%}",
            delta_text=f"{recall_delta:.2%}",
            delta_positive=recall_delta >= 0,
        )

    with col2:
        precision_delta = xgboost["precision"] - logistic["precision"]
        render_metric_card(
            "Precision",
            f"{xgboost['precision']:.2%}",
            delta_text=f"{precision_delta:.2%}",
            delta_positive=precision_delta >= 0,
        )

    with col3:
        pr_auc_delta = xgboost["pr_auc"] - logistic["pr_auc"]
        render_metric_card(
            "PR-AUC",
            f"{xgboost['pr_auc']:.3f}",
            delta_text=f"{pr_auc_delta:.3f}",
            delta_positive=pr_auc_delta >= 0,
        )

    with col4:
        false_positive_reduction = (
            logistic["false_positives"] - xgboost["false_positives"]
        )
        render_metric_card(
            "False Positives Reduced",
            f"{int(false_positive_reduction):,}",
            sub="Fewer legitimate transactions flagged",
        )

    st.write("")
    render_section_header("Model comparison")

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
    )

    render_intro_panel(
        "Why XGBoost was selected",
        "XGBoost was deployed because it improved recall, increased PR-AUC, caught more "
        "fraud cases, and substantially reduced false positives compared with the "
        "Logistic Regression baseline — a better balance of catching fraud while "
        "keeping legitimate customers unaffected.",
    )

    st.write("")
    render_section_header(
        "Global feature importance",
        "Which feature groups XGBoost relies on most across the full dataset. This is "
        "different from SHAP, which explains a single selected transaction.",
    )

    if not GROUPED_IMPORTANCE_PATH.exists():
        st.warning(
            "Feature importance report not found. Run `python -m src.feature_importance` first."
        )
        return

    importance_df = pd.read_csv(GROUPED_IMPORTANCE_PATH)

    readable_importance = importance_df.copy()
    readable_importance["Feature"] = readable_importance["base_feature"].apply(
        format_feature_name
    )
    readable_importance["Importance"] = readable_importance["importance"]

    top_importance = readable_importance[["Feature", "Importance"]].head(10)

    render_feature_importance_chart(top_importance)

    with st.expander("View global feature importance table", expanded=False):
        st.dataframe(
            readable_importance[["Feature", "Importance"]],
            use_container_width=True,
            hide_index=True,
        )


def display_about_page():
    """
    Display a polished portfolio-style overview of the project.
    """
    render_section_header(
        "About This App",
        "An agentic AI fraud risk decisioning dashboard for financial transactions.",
    )

    render_intro_panel(
        "What it does",
        "This dashboard reviews financial transactions for fraud risk. It combines "
        "XGBoost fraud scoring, SHAP explainability, MLflow experiment tracking and a "
        "multi-agent decision workflow to produce a fraud probability, risk band, "
        "business recommendation, analyst summary, policy support and "
        "transaction-specific model drivers.",
    )

    st.write("")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="about-card">
                <h4>Problem statement</h4>
                <p>Fraud teams must catch fraudulent transactions without blocking
                legitimate customers. Naive models flood analysts with false positives.
                This app prioritises high recall while sharply cutting false positives,
                and explains every decision.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="about-card">
                <h4>Dataset</h4>
                <p>A real-world-style fraud transaction dataset with merchant, category,
                amount, location and customer attributes. Engineered features include
                transaction time, customer age, distance from home, time since the
                previous transaction and implied travel speed.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown(
            """
            <div class="about-card">
                <h4>Multi-agent workflow</h4>
                <ul>
                    <li><b>Risk Scoring Agent</b> — scores fraud probability and risk band</li>
                    <li><b>Investigation Agent</b> — writes the analyst summary</li>
                    <li><b>Policy Agent</b> — applies policy guidance</li>
                    <li><b>Decision Agent</b> — issues the final recommendation</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
            <div class="about-card">
                <h4>Model &amp; explainability</h4>
                <p>The deployed model is XGBoost, selected over Logistic Regression after
                improving recall, PR-AUC and false-positive performance. SHAP provides
                local, transaction-specific attributions, while grouped feature
                importance shows global model behaviour.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    col5, col6 = st.columns(2)

    with col5:
        st.markdown(
            """
            <div class="about-card">
                <h4>Tech stack</h4>
                <ul>
                    <li>Python, pandas, NumPy</li>
                    <li>XGBoost &amp; scikit-learn</li>
                    <li>SHAP explainability</li>
                    <li>MLflow experiment tracking</li>
                    <li>Streamlit &amp; Plotly</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col6:
        st.markdown(
            """
            <div class="about-card">
                <h4>Limitations &amp; future work</h4>
                <ul>
                    <li>Trained on a static, offline dataset</li>
                    <li>No real-time streaming or feedback loop yet</li>
                    <li>Future: live scoring API and analyst feedback</li>
                    <li>Future: drift monitoring and auto-retraining</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_transaction_review_page(input_mode, show_transaction, show_workflow_details):
    """
    Render the full Transaction Review analyst workflow page.
    """
    render_section_header(
        "Transaction Review",
        "Review a transaction, run the fraud risk workflow, and understand the reasoning behind the decision.",
    )

    render_intro_panel(
        "How it works",
        "Choose a demo transaction or upload a CSV, select the row you want to review, and "
        "run the fraud risk analysis to generate a score, explanation, and recommended action.",
    )

    if (
        st.session_state["input_mode"] is not None
        and input_mode != st.session_state["input_mode"]
    ):
        st.session_state["workflow_result"] = None
        st.session_state["selected_transaction"] = None
        st.session_state["selection_key"] = None

    st.session_state["input_mode"] = input_mode

    transactions = None
    selected_index = None
    selected_transaction = None

    st.write("")

    if input_mode == "Upload CSV":
        render_section_header("Upload transaction dataset")

        uploaded_file = st.file_uploader(
            "Upload a CSV file using the real-world fraud transaction schema",
            type=["csv"],
            key="csv_uploader",
        )

        clean_df = None

        if uploaded_file is not None:
            try:
                uploaded_file.seek(0)
                uploaded_df = pd.read_csv(uploaded_file)

                is_valid, message, validated_df = validate_transaction_dataframe(
                    uploaded_df
                )

                if not is_valid:
                    reset_results()
                    st.error(message)
                    return

                st.session_state["uploaded_clean_df"] = validated_df
                st.session_state["uploaded_file_name"] = uploaded_file.name

                clean_df = validated_df
                st.success(message)

            except Exception as error:
                st.error(f"Could not read uploaded file: {error}")

                if st.session_state["uploaded_clean_df"] is not None:
                    st.info(
                        f"Using previously uploaded file: "
                        f"{st.session_state['uploaded_file_name']}"
                    )
                    clean_df = st.session_state["uploaded_clean_df"]
                else:
                    reset_results()
                    return

        elif st.session_state["uploaded_clean_df"] is not None:
            clean_df = st.session_state["uploaded_clean_df"]

            st.info(
                f"Using previously uploaded file: "
                f"{st.session_state['uploaded_file_name']}"
            )

            if st.button("Clear uploaded CSV"):
                clear_uploaded_data()
                st.rerun()

        else:
            reset_results()
            st.warning(
                "Upload a transaction CSV to begin, or choose Demo Transaction "
                "from the sidebar."
            )

        if clean_df is not None:
            with st.expander("Uploaded Data Preview", expanded=False):
                st.dataframe(clean_df.head(20), use_container_width=True)

            transactions = clean_df.drop(columns=["is_fraud"], errors="ignore")

            row_position = st.number_input(
                "Select row number to review",
                min_value=0,
                max_value=len(transactions) - 1,
                value=0,
                step=1,
            )

            selected_index = transactions.index[int(row_position)]
            selected_transaction = transactions.loc[[selected_index]]

    elif input_mode == "Demo Transaction":
        render_section_header("Demo transaction")

        try:
            default_df = load_default_data()

            is_valid, message, clean_df = validate_transaction_dataframe(default_df)

            if not is_valid:
                reset_results()
                st.error(message)
                return

            transactions = clean_df.drop(columns=["is_fraud"], errors="ignore")

            demo_choice = st.selectbox(
                "Choose a demo case",
                list(DEMO_TRANSACTIONS.keys()),
            )

            selected_index = DEMO_TRANSACTIONS[demo_choice]
            selected_transaction = transactions.loc[[selected_index]]

            st.caption(
                f"Selected demo index: {selected_index} "
                f"({demo_choice})"
            )

        except Exception as error:
            reset_results()
            st.error(f"Could not load demo data: {error}")
            return

    if selected_transaction is not None and transactions is not None:
        current_selection_key = f"{input_mode}-{selected_index}"

        if (
            st.session_state["selection_key"] is not None
            and current_selection_key != st.session_state["selection_key"]
        ):
            st.session_state["workflow_result"] = None

        st.session_state["selection_key"] = current_selection_key
        st.session_state["selected_transaction"] = selected_transaction

        if show_transaction:
            with st.expander("Selected Transaction", expanded=True):
                display_selected_transaction(selected_transaction)

        if st.button("Run Fraud Risk Review", type="primary"):
            with st.spinner("Running fraud decision workflow..."):
                workflow_result = run_fraud_decision_workflow_with_context(
                    transactions=transactions,
                    selected_index=selected_index,
                )

                st.session_state["workflow_result"] = workflow_result

    if st.session_state["workflow_result"] is not None:
        display_decision_dashboard(
            workflow_result=st.session_state["workflow_result"],
            show_workflow_details=show_workflow_details,
        )


def render_sidebar():
    """
    Render the sidebar brand, navigation and Transaction Review controls.
    Returns the selected page and Transaction Review options.
    """
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-wrap">
                <div class="brand-logo">🛡️</div>
                <div>
                    <div class="brand-name">FraudGuard</div>
                    <div class="brand-tag">Risk Decisioning</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        page = st.radio(
            "Navigation",
            ["Transaction Review", "Model Performance", "About This App"],
            index=0,
        )

        input_mode = "Upload CSV"
        show_transaction = True
        show_workflow_details = False

        if page == "Transaction Review":
            st.divider()
            st.caption("Transaction Controls")

            input_mode = st.radio(
                "Input source",
                ["Upload CSV", "Demo Transaction"],
                index=0,
            )

            show_transaction = st.checkbox(
                "Show selected transaction",
                value=True,
            )

            show_workflow_details = st.checkbox(
                "Show agent workflow trace",
                value=False,
            )

        st.divider()
        st.caption("Built with XGBoost, SHAP, MLflow, and Streamlit.")

    return page, input_mode, show_transaction, show_workflow_details


def main():
    st.set_page_config(
        page_title="Fraud Risk Decisioning Dashboard",
        page_icon="🛡️",
        layout="wide",
    )

    inject_global_styles()
    initialize_session_state()

    render_main_header()

    page, input_mode, show_transaction, show_workflow_details = render_sidebar()

    if page == "Model Performance":
        display_model_performance_page()
        return

    if page == "About This App":
        display_about_page()
        return

    render_transaction_review_page(
        input_mode=input_mode,
        show_transaction=show_transaction,
        show_workflow_details=show_workflow_details,
    )


if __name__ == "__main__":
    main()
