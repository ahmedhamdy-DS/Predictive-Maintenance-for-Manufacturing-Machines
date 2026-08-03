import os
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import shap
import streamlit as st

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Page config & Global Card Styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    layout="wide",
)

# Force Plotly dark theme as base
pio.templates.default = "plotly_dark"

# Custom CSS for the deep dark background, specific card colors, and larger tabs
st.markdown("""
<style>
    /* Main App Background */
    .stApp { background-color: #121418; }
    
    /* Adjust spacing */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 95%; }
    
    /* Enlarge Tab Fonts */
    button[data-baseweb="tab"] p { 
        font-size: 22px !important; 
        font-weight: 600 !important; 
        color: #a0aab2;
    }
    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #ffffff !important;
    }
    
    /* Custom Metric Card (Overview Tab) */
    .custom-metric-card {
        background-color: #1e2129;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
    }
    .custom-metric-value {
        font-size: 38px;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        line-height: 1.2;
    }
    .custom-metric-label {
        font-size: 14px;
        color: #a0aab2;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Color Palette
# ----------------------------------------------------------------------------
PALETTE = {
    "blue": "#2d87e0",
    "orange": "#f49d37",
    "teal": "#20b2aa",
    "purple": "#7a61ba",
    "card_bg": "#1e2129",
    "text_muted": "#a0aab2"
}

# ----------------------------------------------------------------------------
# Helper: Standard Layout for Plotly Cards
# ----------------------------------------------------------------------------
def apply_card_layout(fig, title=""):
    fig.update_layout(
        title={"text": title, "font": {"color": "#ffffff", "size": 16}},
        plot_bgcolor=PALETTE["card_bg"],
        paper_bgcolor=PALETTE["card_bg"],
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color=PALETTE["text_muted"])),
        yaxis=dict(showgrid=True, gridcolor="#2c303a", zeroline=False, tickfont=dict(color=PALETTE["text_muted"])),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=PALETTE["text_muted"]))
    )
    return fig

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
PIPELINE_PATH = os.path.join(MODEL_DIR, "preprocessing_pipeline.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_predictive_maintenance.pkl")

NUM_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CAT_COLS = ["Type"]
REQUIRED_COLS = NUM_COLS + CAT_COLS

# ----------------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        pipeline = joblib.load(PIPELINE_PATH)
        model = joblib.load(MODEL_PATH)
        return pipeline, model
    except FileNotFoundError:
        st.error("Model artifacts not found. Please ensure `models/` directory exists with the required `.pkl` files.")
        st.stop()

@st.cache_resource
def build_explainer(_model):
    return shap.TreeExplainer(_model)

def get_clean_feature_names(pipeline):
    cat_encoder = pipeline.named_transformers_["cat"].named_steps["ohe"]
    cat_features = cat_encoder.get_feature_names_out(CAT_COLS)
    names = NUM_COLS + list(cat_features)
    return [n.replace("[", "(").replace("]", ")") for n in names]


# ----------------------------------------------------------------------------
# Sidebar — data upload & settings
# ----------------------------------------------------------------------------
st.sidebar.title("Predictive Maintenance")
st.sidebar.caption("Machine Learning Performance Metrics")

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file", type=["csv"], help="Must contain the core sensor columns"
)

use_sample = st.sidebar.checkbox("Use sample data (predictive_maintenance.csv)", value=uploaded_file is None)

st.sidebar.markdown("---")
st.sidebar.subheader("Prediction Settings")
risk_threshold = st.sidebar.slider(
    "Risk threshold (probability)",
    min_value=0.05, max_value=0.95, value=0.5, step=0.05,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Assumptions")
cost_unplanned = st.sidebar.number_input("Unplanned failure cost", value=10000, step=500)
cost_preventive = st.sidebar.number_input("Preventive maint. cost", value=1000, step=100)


# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
sample_path = os.path.join(os.path.dirname(__file__), "predictive_maintenance.csv")

if uploaded_file is not None and not use_sample:
    df_raw = pd.read_csv(uploaded_file)
    data_source = uploaded_file.name
elif os.path.exists(sample_path):
    df_raw = pd.read_csv(sample_path)
    data_source = "predictive_maintenance.csv (sample)"
else:
    st.warning("Please upload a CSV file to get started.")
    st.stop()

missing_cols = [c for c in REQUIRED_COLS if c not in df_raw.columns]
if missing_cols:
    st.error(f"Missing columns: {', '.join(missing_cols)}")
    st.stop()

has_target = "Target" in df_raw.columns
has_failure_type = "Failure Type" in df_raw.columns
has_id = "Product ID" in df_raw.columns

st.title("Predictive Maintenance Dashboard")
st.caption(f"Source: **{data_source}**  |  Rows: **{len(df_raw):,}**")

# ----------------------------------------------------------------------------
# Load model & run predictions
# ----------------------------------------------------------------------------
pipeline, model = load_artifacts()

X = df_raw[REQUIRED_COLS].copy()
X_transformed = pipeline.transform(X)
feature_names = get_clean_feature_names(pipeline)
X_df = pd.DataFrame(X_transformed, columns=feature_names)

proba = model.predict_proba(X_transformed)[:, 1]
pred = (proba >= risk_threshold).astype(int)

results = df_raw.copy()
results["Failure Probability"] = proba
results["Predicted Risk"] = np.select(
    [proba >= 0.7, proba >= risk_threshold],
    ["High", "Medium"],
    default="Low",
)
results["Predicted Failure"] = pred

id_col = "Product ID" if has_id else None

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_overview, tab_eda, tab_predictions, tab_shap, tab_cost = st.tabs(
    ["Overview", "Data Analysis", "Predictions", "Explainability", "Cost Impact"]
)

# ---- Overview (20-Minute Dashboard Style) ------------------------------
with tab_overview:
    # ROW 1
    r1_col1, r1_col2 = st.columns([1.5, 1])
    
    with r1_col1:
        # Multi-line chart
        df_trend = results.sort_values('Tool wear [min]').reset_index(drop=True)
        window = min(50, len(df_trend))
        
        y1 = (df_trend['Air temperature [K]'].rolling(window).mean() - 290) * 1.5
        y2 = (df_trend['Process temperature [K]'].rolling(window).mean() - 300) * 1.2
        y3 = (df_trend['Torque [Nm]'].rolling(window).mean()) * 0.5
        
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(y=y1, name="Air Temp Trend", line=dict(color=PALETTE["blue"], width=2.5)))
        fig_line.add_trace(go.Scatter(y=y2, name="Process Temp Trend", line=dict(color=PALETTE["teal"], width=2.5)))
        fig_line.add_trace(go.Scatter(y=y3, name="Torque Trend", line=dict(color=PALETTE["orange"], width=2.5)))
        
        fig_line = apply_card_layout(fig_line, "Sensor Trends vs Tool Wear")
        fig_line.update_layout(height=320, xaxis=dict(showticklabels=False))
        st.plotly_chart(fig_line, use_container_width=True, config={'displayModeBar': False})

    with r1_col2:
        # Donut Chart
        risk_counts = results["Predicted Risk"].value_counts().reset_index()
        risk_counts.columns = ["Risk", "Count"]
        color_map = {"Low": PALETTE["blue"], "Medium": PALETTE["orange"], "High": PALETTE["teal"]}
        
        fig_donut = px.pie(
            risk_counts, values='Count', names='Risk', hole=0.6,
            color='Risk', color_discrete_map=color_map
        )
        
        high_risk_rows = risk_counts[risk_counts['Risk'] == "High"]
        high_risk_pct = int((high_risk_rows['Count'].sum() / len(results)) * 100) if not high_risk_rows.empty else 0
        fig_donut.add_annotation(text=f"{high_risk_pct}%", x=0.5, y=0.5, font_size=32, font_color="white", showarrow=False)
        
        fig_donut = apply_card_layout(fig_donut, "Risk Distribution")
        fig_donut.update_layout(height=320, showlegend=False)
        fig_donut.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})

    # ROW 2
    r2_col1, r2_col2, r2_col3 = st.columns([1.2, 1, 1.2])
    
    with r2_col1:
        # Horizontal Bar
        type_risk = results.groupby("Type")["Failure Probability"].mean().reset_index()
        fig_hbar = px.bar(
            type_risk, x="Failure Probability", y="Type", orientation='h',
            color="Type", color_discrete_sequence=[PALETTE["teal"], PALETTE["blue"], PALETTE["orange"]]
        )
        fig_hbar = apply_card_layout(fig_hbar, "Avg Risk by Machine Type")
        fig_hbar.update_layout(height=350, showlegend=False, yaxis=dict(showgrid=False))
        st.plotly_chart(fig_hbar, use_container_width=True, config={'displayModeBar': False})

    with r2_col2:
        # Top: Metric Card
        flagged_count = int(pred.sum())
        st.markdown(f"""
            <div class="custom-metric-card">
                <p class="custom-metric-value">{flagged_count:,}</p>
                <p class="custom-metric-label">Machines Flagged for Review</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Bottom: Small Scatter Plot
        scatter_sample = results.sample(min(30, len(results)))
        fig_scatter = px.scatter(
            scatter_sample, x="Rotational speed [rpm]", y="Failure Probability",
            color_discrete_sequence=[PALETTE["purple"]]
        )
        fig_scatter = apply_card_layout(fig_scatter, "Risk vs Speed (Sample)")
        fig_scatter.update_layout(
            height=190, margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False)
        )
        fig_scatter.update_traces(marker=dict(size=8, opacity=0.8))
        st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False})

    with r2_col3:
        # Vertical Bar
        results['Wear Bin'] = pd.cut(results['Tool wear [min]'], bins=8, labels=[f"B{i}" for i in range(1, 9)])
        flagged_bins = results[results['Predicted Failure'] == 1]['Wear Bin'].value_counts().sort_index().reset_index()
        flagged_bins.columns = ['Wear Bin', 'Count']
        
        fig_vbar = px.bar(
            flagged_bins, x="Wear Bin", y="Count",
            color="Wear Bin", color_discrete_sequence=[PALETTE["blue"]]
        )
        fig_vbar = apply_card_layout(fig_vbar, "Flagged Machines by Wear")
        fig_vbar.update_layout(height=350, showlegend=False, xaxis=dict(showgrid=False))
        st.plotly_chart(fig_vbar, use_container_width=True, config={'displayModeBar': False})

# ---- EDA -----------------------------------------------------------------
with tab_eda:
    st.subheader("Sensor Reading Distributions")
    sel_feature = st.selectbox("Choose a variable to view", NUM_COLS)
    colA, colB = st.columns(2)

    with colA:
        fig = px.histogram(df_raw, x=sel_feature, nbins=50, title=f"Distribution of {sel_feature}", color_discrete_sequence=[PALETTE["teal"]])
        fig = apply_card_layout(fig, f"Distribution of {sel_feature}")
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        if has_target:
            fig = px.box(df_raw, x="Target", y=sel_feature, color="Target", title=f"{sel_feature} by failure occurrence", color_discrete_sequence=[PALETTE["orange"], PALETTE["blue"]])
        else:
            fig = px.box(results, x="Predicted Risk", y=sel_feature, color="Predicted Risk", title=f"{sel_feature} by predicted risk level", color_discrete_map={"Low": PALETTE["blue"], "Medium": PALETTE["orange"], "High": PALETTE["teal"]})
        fig = apply_card_layout(fig, f"{sel_feature} by Risk/Target")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Correlation Between Variables")
    corr_cols = NUM_COLS + (["Target"] if has_target else [])
    corr = df_raw[corr_cols].corr()
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="Blues", title="Correlation Heatmap")
    fig_corr = apply_card_layout(fig_corr, "Correlation Heatmap")
    st.plotly_chart(fig_corr, use_container_width=True)


# ---- Predictions -----------------------------------------------------
with tab_predictions:
    st.subheader("Highest-Risk Machines")
    display_cols = []
    if id_col:
        display_cols.append(id_col)
    display_cols += REQUIRED_COLS + ["Failure Probability", "Predicted Risk"]

    risk_filter = st.multiselect(
        "Filter by risk level", ["High", "Medium", "Low"],
        default=["High", "Medium"],
    )
    filtered = results[results["Predicted Risk"].isin(risk_filter)].sort_values(
        "Failure Probability", ascending=False
    )
    
    st.dataframe(
        filtered[display_cols].style.format({"Failure Probability": "{:.1%}"}),
        use_container_width=True, height=450,
    )


# ---- SHAP --------------------------------------------------------------
with tab_shap:
    st.subheader("Why Did the Model Flag This Machine?")
    explainer = build_explainer(model)
    shap_values = explainer.shap_values(X_df)

    st.markdown("**Global feature importance across all data:**")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    imp_df = pd.DataFrame({"Feature": feature_names, "Mean |SHAP|": mean_abs_shap}).sort_values(
        "Mean |SHAP|", ascending=True
    )
    fig_imp = px.bar(imp_df, x="Mean |SHAP|", y="Feature", orientation="h", title="SHAP Feature Importance", color_discrete_sequence=[PALETTE["blue"]])
    fig_imp = apply_card_layout(fig_imp, "Feature Importance (Impact on decision)")
    st.plotly_chart(fig_imp, use_container_width=True)


# ---- Cost impact ---------------------------------------------------------
with tab_cost:
    st.subheader("Financial Estimate")
    n_flagged = int(pred.sum())
    st.write(f"Based on the model, **{n_flagged:,}** machines would be flagged for preventive maintenance (out of {len(results):,} total).")

    if has_target:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(df_raw["Target"], pred)
        tn, fp, fn, tp = cm.ravel()

        cost_baseline = (tp + fn) * cost_unplanned 
        cost_with_model = tp * cost_preventive + fn * cost_unplanned + fp * cost_preventive
        savings = cost_baseline - cost_with_model

        c1, c2, c3 = st.columns(3)
        c1.metric("Cost without the model", f"${cost_baseline:,.0f}")
        c2.metric("Cost with the model", f"${cost_with_model:,.0f}")
        c3.metric("Estimated savings", f"${savings:,.0f}", delta=f"{savings/cost_baseline:.0%}" if cost_baseline else None)
    else:
        estimated_savings = n_flagged * (cost_unplanned - cost_preventive)
        st.metric("Rough estimated savings", f"${estimated_savings:,.0f}")