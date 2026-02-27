# ============================================================
# 🌾 Dynamic Crop Recommendation Dashboard — Redesigned
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import cloudpickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb

# ============================================================
# 1. Page Setup + Global CSS Styling
# ============================================================

st.set_page_config(
    page_title="CropSense — Intelligent Crop Advisor",
    layout="wide",
    page_icon="🌾",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:          #0f1a10;
    --surface:     #172518;
    --card:        #1e3320;
    --border:      #2e4d30;
    --accent:      #6fcf57;
    --accent-soft: #4a9e35;
    --gold:        #e0b84a;
    --text:        #e8f0e9;
    --muted:       #8aab8c;
    --danger:      #e07070;
    --info:        #70b8e0;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1400px; }

.hero {
    background: linear-gradient(135deg, #1b3a1d 0%, #0f2410 50%, #162e18 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3rem 3.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "🌾";
    position: absolute;
    font-size: 18rem;
    right: -2rem;
    top: -3rem;
    opacity: 0.05;
    line-height: 1;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: var(--accent) !important;
    margin: 0 0 0.5rem;
    line-height: 1.2;
}
.hero p {
    color: var(--muted);
    font-size: 1.05rem;
    font-weight: 300;
    max-width: 620px;
    line-height: 1.7;
    margin: 0;
}
.hero .badge {
    display: inline-block;
    background: rgba(111,207,87,0.12);
    border: 1px solid rgba(111,207,87,0.3);
    color: var(--accent);
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    margin-bottom: 1rem;
}

.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 2.5rem 0 1.2rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
}
.section-header .icon {
    font-size: 1.5rem;
    width: 2.5rem;
    height: 2.5rem;
    background: rgba(111,207,87,0.1);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.section-header h2 {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text) !important;
    margin: 0;
}

.result-box {
    background: linear-gradient(135deg, #1e3d20, #162d18);
    border: 1.5px solid var(--accent-soft);
    border-radius: 18px;
    padding: 2rem 2.5rem;
    text-align: center;
    margin: 1.5rem 0;
}
.result-box .crop-name {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    color: var(--accent);
    font-weight: 700;
    margin: 0.5rem 0;
}
.result-box .result-label {
    font-size: 0.8rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
}

.conf-bar-bg {
    background: var(--border);
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 999px;
}
.conf-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: var(--muted);
    margin-bottom: 0.4rem;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label {
    color: var(--muted) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #4a9e35, #6fcf57) !important;
    color: #0f1a10 !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.7rem 1.5rem !important;
    transition: all 0.2s ease !important;
    font-family: 'DM Sans', sans-serif !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(111,207,87,0.3) !important;
}

div[data-testid="stSuccess"] {
    background: rgba(111,207,87,0.08) !important;
    border: 1px solid rgba(111,207,87,0.3) !important;
    border-radius: 12px !important;
    color: var(--accent) !important;
}
div[data-testid="stWarning"] {
    background: rgba(224,184,74,0.08) !important;
    border: 1px solid rgba(224,184,74,0.3) !important;
    border-radius: 12px !important;
    color: var(--gold) !important;
}
div[data-testid="stInfo"] {
    background: rgba(112,184,224,0.08) !important;
    border: 1px solid rgba(112,184,224,0.3) !important;
    border-radius: 12px !important;
    color: var(--info) !important;
}
div[data-testid="stError"] {
    background: rgba(224,112,112,0.08) !important;
    border: 1px solid rgba(224,112,112,0.3) !important;
    border-radius: 12px !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

div[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    background: var(--card) !important;
}

div[data-testid="metric-container"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 1rem 1.25rem !important;
}
div[data-testid="metric-container"] label {
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 1.8rem !important;
}

hr { border-color: var(--border) !important; margin: 2.5rem 0 !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ─────────────────────────────────
mpl.rcParams.update({
    'figure.facecolor':  '#1e3320',
    'axes.facecolor':    '#172518',
    'axes.edgecolor':    '#2e4d30',
    'axes.labelcolor':   '#8aab8c',
    'xtick.color':       '#8aab8c',
    'ytick.color':       '#8aab8c',
    'text.color':        '#e8f0e9',
    'grid.color':        '#2e4d30',
    'grid.linestyle':    '--',
    'grid.alpha':        0.5,
    'font.family':       'sans-serif',
})

def section(icon, title):
    st.markdown(f"""
    <div class="section-header">
        <div class="icon">{icon}</div>
        <h2>{title}</h2>
    </div>""", unsafe_allow_html=True)

# ── Feature constants ─────────────────────────────────────
NUMERIC_FEATURES     = ["NDVI", "Rainfall", "Soil_pH"]
CATEGORICAL_FEATURES = ["Season"]
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ============================================================
# Hero Banner
# ============================================================
st.markdown("""
<div class="hero">
    <div class="badge">AI-Powered Advisory</div>
    <h1>CropSense</h1>
    <p>Satellite intelligence meets soil science. Enter your district's environmental profile
    to receive a data-driven crop recommendation — with full model transparency and a farmer
    feedback loop.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 2. Load Model & Data
# ============================================================

@st.cache_resource
def load_model():
    with open("best_tuned_model.pkl", "rb") as f:
        return cloudpickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv("final_dataset1.csv")
    if "Soil_pH" in df.columns and df["Soil_pH"].max() > 20:
        df["Soil_pH"] = df["Soil_pH"] / 10.0
    return df

@st.cache_resource
def build_preprocessor(_df):
    """Fit a ColumnTransformer on the base dataset once and cache it."""
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(),                              NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore",
                              sparse_output=False),            CATEGORICAL_FEATURES),
    ])
    preprocessor.fit(_df[ALL_FEATURES])
    return preprocessor

model = load_model()
if model is None:
    st.stop()

df          = load_data()
preprocessor = build_preprocessor(df)          # fitted once, cached

districts   = sorted(df["ADM2_NAME_clean"].unique()) if "ADM2_NAME_clean" in df.columns else []
seasons     = sorted(df["Season"].dropna().unique())
crops       = sorted(df["dominant_crop"].unique())

# ============================================================
# 3. Input Section
# ============================================================
section("🧩", "Environmental Parameters")

col1, col2, col3 = st.columns([1.2, 1.2, 1], gap="large")

with col1:
    st.markdown('<p style="color:#8aab8c;font-size:0.75rem;font-weight:600;'
                'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem">Location</p>',
                unsafe_allow_html=True)
    district = st.selectbox("District", districts, label_visibility="collapsed")

    st.markdown('<p style="color:#8aab8c;font-size:0.75rem;font-weight:600;'
                'letter-spacing:0.1em;text-transform:uppercase;'
                'margin-bottom:0.5rem;margin-top:1rem">Season</p>',
                unsafe_allow_html=True)
    season = st.selectbox("Season", seasons, label_visibility="collapsed")
    season = str(season).strip().title()

with col2:
    ndvi     = st.slider("🛰 NDVI — Vegetation Index", 0.0, 1.0, 0.5, 0.01)
    rainfall = st.slider("🌧 Rainfall (Normalized)",   0.0, 1.5, 0.5, 0.01)
    soil_ph  = st.slider("🧪 Soil pH",                 4.0, 8.0, 6.5, 0.1)

with col3:
    st.markdown("""
    <div style="background:#172518;border:1px solid #2e4d30;border-radius:14px;
                padding:1.25rem;margin-bottom:1rem">
        <div style="font-size:0.7rem;color:#8aab8c;text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:0.4rem">Vegetation Health</div>
        <div style="font-size:0.85rem;color:#e8f0e9">
            NDVI near <b style="color:#6fcf57">0.8+</b> signals dense, healthy canopy.
            Values below <b style="color:#e0b84a">0.3</b> suggest sparse or stressed vegetation.
        </div>
    </div>
    <div style="background:#172518;border:1px solid #2e4d30;border-radius:14px;
                padding:1.25rem;margin-bottom:1rem">
        <div style="font-size:0.7rem;color:#8aab8c;text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:0.4rem">Ideal Soil pH</div>
        <div style="font-size:0.85rem;color:#e8f0e9">
            Most crops thrive at pH <b style="color:#6fcf57">6.0–7.0</b>.
            Acidic below 5.5 may limit nutrient uptake.
        </div>
    </div>
    """, unsafe_allow_html=True)
    predict_button = st.button("🚀 Predict Dominant Crop", use_container_width=True)

# ── Build input dataframe (only the 4 features the model needs) ──
input_data = pd.DataFrame({
    "NDVI":     [ndvi],
    "Rainfall": [rainfall],
    "Soil_pH":  [soil_ph],
    "Season":   [season],
})

# ============================================================
# 4. Prediction
# ============================================================
if predict_button:
    section("🌱", "Prediction Result")

    try:
        X_transformed = preprocessor.transform(input_data[ALL_FEATURES])
        prediction    = model.predict(X_transformed)[0]
        proba         = model.predict_proba(X_transformed)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # ── Decode prediction to crop name ───────────────────
    try:
        label_encoder = joblib.load("label_encoder.joblib")
        crop_name     = label_encoder.inverse_transform([prediction])[0]
        class_names   = label_encoder.classes_
    except Exception:
        # Fallback: use integer index mapped to sorted crop list
        crop_name   = crops[int(prediction)] if int(prediction) < len(crops) else str(prediction)
        class_names = np.array(crops)

    st.session_state["predicted_crop"]  = crop_name
    st.session_state["last_proba"]      = proba
    st.session_state["last_class_names"] = class_names

    confidence = float(np.max(proba))
    conf_color = "#6fcf57" if confidence >= 0.75 else "#e0b84a" if confidence >= 0.5 else "#e07070"
    conf_label = ("High Confidence"   if confidence >= 0.75 else
                  "Medium Confidence" if confidence >= 0.5  else
                  "Low Confidence")

    rc, cc = st.columns([1.4, 1], gap="large")

    with rc:
        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Recommended Dominant Crop</div>
            <div class="crop-name">{crop_name}</div>
            <div style="margin-top:1.2rem">
                <div class="conf-label">
                    <span>Model Confidence</span>
                    <span style="color:{conf_color};font-weight:700">
                        {conf_label} · {confidence:.1%}
                    </span>
                </div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill"
                         style="width:{confidence*100:.1f}%;background:{conf_color}"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with cc:
        st.markdown("**Top Crop Probabilities**")
        prob_df = (
            pd.DataFrame({"Crop": class_names, "Probability": proba})
            .sort_values("Probability", ascending=False)
            .head(6)
            .reset_index(drop=True)
        )
        fig, ax = plt.subplots(figsize=(5, 3))
        colors  = ["#6fcf57" if c == crop_name else "#2e4d30" for c in prob_df["Crop"]]
        ax.barh(prob_df["Crop"][::-1], prob_df["Probability"][::-1],
                color=colors[::-1], height=0.6)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, bbox_inches="tight")

    st.session_state["last_prediction"] = {
        "district":   district,
        "season":     season,
        "crop_name":  crop_name,
        "prob_df":    prob_df,
        "confidence": confidence,
        "conf_color": conf_color,
        "conf_label": conf_label,
    }

# ── Persistent last-prediction banner ────────────────────
if "last_prediction" in st.session_state and not predict_button:
    lp = st.session_state["last_prediction"]
    st.markdown(f"""
    <div style="background:#1e3320;border:1px solid #2e4d30;border-radius:14px;
                padding:1.25rem 1.75rem;display:flex;align-items:center;
                gap:1.5rem;flex-wrap:wrap;margin:1rem 0">
        <div>
            <div style="font-size:0.7rem;color:#8aab8c;text-transform:uppercase;
                        letter-spacing:0.1em">Last Prediction</div>
            <div style="font-family:'Georgia',serif;font-size:1.6rem;
                        color:#6fcf57;font-weight:700">{lp['crop_name']}</div>
        </div>
        <div style="flex:1">
            <span style="background:rgba(111,207,87,0.1);border:1px solid rgba(111,207,87,0.25);
                         color:#8aab8c;font-size:0.78rem;padding:0.25rem 0.7rem;
                         border-radius:999px;margin-right:0.5rem">📍 {lp['district']}</span>
            <span style="background:rgba(111,207,87,0.1);border:1px solid rgba(111,207,87,0.25);
                         color:#8aab8c;font-size:0.78rem;padding:0.25rem 0.7rem;
                         border-radius:999px">🗓 {lp['season']}</span>
        </div>
        <div style="color:{lp['conf_color']};font-size:0.85rem;font-weight:600">
            {lp['conf_label']} · {lp['confidence']:.1%}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 5. SHAP Explainability
# ============================================================
if predict_button:
    with st.expander("🔍 Model Explanation — SHAP Feature Importance"):
        try:
            X_trans       = preprocessor.transform(input_data[ALL_FEATURES])
            feature_names = preprocessor.get_feature_names_out()
            explainer     = shap.TreeExplainer(model)
            shap_values   = explainer.shap_values(X_trans)

            if isinstance(shap_values, list):
                pred_class          = int(np.argmax(model.predict_proba(X_trans), axis=1)[0])
                shap_values_to_plot = shap_values[pred_class]
                class_label         = crop_name
            else:
                shap_values_to_plot = shap_values
                class_label         = crop_name

            st.markdown(
                f"<p style='color:#8aab8c;font-size:0.85rem'>Feature contribution for: "
                f"<strong style='color:#6fcf57'>{class_label}</strong></p>",
                unsafe_allow_html=True
            )
            fig, ax = plt.subplots(figsize=(7, 4))
            shap.summary_plot(shap_values_to_plot, X_trans,
                              feature_names=feature_names, plot_type="bar", show=False)
            st.pyplot(fig, bbox_inches="tight", dpi=120)

        except Exception as e:
            st.warning(f"SHAP visualization skipped: {e}")

# ============================================================
# 6. Farmer Feedback
# ============================================================
section("🗣️", "Farmer Feedback")
st.markdown(
    "<p style='color:#8aab8c;font-size:0.9rem;margin-top:-0.5rem'>"
    "Share your on-ground results to help improve future recommendations.</p>",
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3, gap="large")
with col1:
    actual_crop = st.selectbox("Actual Crop Grown", crops)
with col2:
    yield_kg = st.number_input("Yield (kg / hectare)", 500, 10000, 3000, 100)
with col3:
    satisfaction = st.select_slider(
        "Farmer Satisfaction",
        options=[1, 2, 3, 4, 5],
        value=3,
        format_func=lambda x: {
            1: "😞 Very Poor", 2: "😐 Poor",
            3: "😊 Fair",      4: "😄 Good",
            5: "🤩 Excellent"
        }[x]
    )

save_feedback = st.button("💾 Submit Feedback", use_container_width=True)

if save_feedback:
    feedback_record = pd.DataFrame({
        "timestamp":      [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "district":       [district],
        "season":         [season],
        "NDVI":           [ndvi],
        "Rainfall":       [rainfall],
        "Soil_pH":        [soil_ph],
        "predicted_crop": [st.session_state.get("predicted_crop", "Not Predicted")],
        "actual_crop":    [actual_crop],
        "yield_kg":       [yield_kg],
        "satisfaction":   [satisfaction],
    })
    try:
        existing = pd.read_csv("feedback_data.csv")
        updated  = pd.concat([existing, feedback_record], ignore_index=True)
    except FileNotFoundError:
        updated = feedback_record
    updated.to_csv("feedback_data.csv", index=False)
    st.success(f"✅ Feedback recorded — {actual_crop} · {yield_kg:,} kg/ha · {'⭐' * satisfaction}")
    st.session_state["last_feedback"] = feedback_record.iloc[0].to_dict()

if "last_feedback" in st.session_state:
    fb = st.session_state["last_feedback"]
    st.markdown(f"""
    <div style="background:#172518;border:1px solid #2e4d30;border-radius:12px;
                padding:1rem 1.5rem;font-size:0.875rem;color:#8aab8c;margin-top:0.5rem">
        📬 <strong style="color:#e8f0e9">Latest submission:</strong>
        &nbsp;{fb['district']} / {fb['season']}
        &nbsp;·&nbsp;<span style="color:#6fcf57">{fb['actual_crop']}</span>
        &nbsp;·&nbsp;{int(fb['yield_kg']):,} kg/ha
        &nbsp;·&nbsp;{'⭐' * int(fb['satisfaction'])}
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 7. Feedback Analytics
# ============================================================
section("📊", "Feedback Analytics & Insights")

try:
    feedback_df = pd.read_csv("feedback_data.csv")

    k1, k2, k3, k4 = st.columns(4, gap="medium")
    k1.metric("Avg Satisfaction",  f"{feedback_df['satisfaction'].mean():.2f} ⭐")
    k2.metric("Avg Yield",         f"{feedback_df['yield_kg'].mean():,.0f} kg/ha")
    k3.metric("Total Responses",   f"{len(feedback_df):,}")
    k4.metric("Districts Covered", f"{feedback_df['district'].nunique()}")

    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

    ch1, ch2 = st.columns(2, gap="large")

    with ch1:
        st.markdown("**Satisfaction by Predicted Crop**")
        crop_sat = (feedback_df.groupby("predicted_crop")["satisfaction"]
                    .mean().sort_values(ascending=True))
        fig, ax  = plt.subplots(figsize=(6, max(2.5, len(crop_sat) * 0.45)))
        bars     = ax.barh(crop_sat.index, crop_sat.values, color="#4a9e35", height=0.55)
        for bar, val in zip(bars, crop_sat.values):
            ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}", va='center', fontsize=8, color="#8aab8c")
        ax.set_xlim(0, 5.5)
        ax.set_xlabel("Average Satisfaction")
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    with ch2:
        with st.expander("📈 Yield vs Satisfaction", expanded=True):
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            palette = sns.color_palette("YlGn",
                                        n_colors=max(1, feedback_df["predicted_crop"].nunique()))
            sns.scatterplot(data=feedback_df, x="yield_kg", y="satisfaction",
                            hue="predicted_crop", palette=palette,
                            s=80, alpha=0.85, ax=ax)
            ax.set_xlabel("Yield (kg/ha)")
            ax.set_ylabel("Satisfaction")
            ax.spines[['top', 'right']].set_visible(False)
            ax.legend(fontsize=7, framealpha=0.2, loc="lower right")
            plt.tight_layout()
            st.pyplot(fig)

    st.markdown("**Trends Over Time**")
    feedback_df["timestamp"] = pd.to_datetime(feedback_df["timestamp"], errors="coerce")

    t1, t2 = st.columns(2, gap="large")

    with t1:
        trend   = feedback_df.groupby("timestamp")["yield_kg"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(5.5, 3))
        ax.fill_between(trend["timestamp"], trend["yield_kg"], alpha=0.15, color="#6fcf57")
        ax.plot(trend["timestamp"], trend["yield_kg"],
                marker="o", color="#6fcf57", linewidth=2, markersize=5)
        ax.set_title("Average Yield Over Time", fontsize=10)
        ax.set_ylabel("kg/ha")
        plt.xticks(rotation=35, fontsize=7)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    with t2:
        sat_trend = feedback_df.groupby("timestamp")["satisfaction"].mean().reset_index()
        fig, ax   = plt.subplots(figsize=(5.5, 3))
        ax.fill_between(sat_trend["timestamp"], sat_trend["satisfaction"],
                        alpha=0.15, color="#e0b84a")
        ax.plot(sat_trend["timestamp"], sat_trend["satisfaction"],
                color="#e0b84a", marker="o", linewidth=2, markersize=5)
        ax.set_title("Average Satisfaction Over Time", fontsize=10)
        ax.set_ylim(0, 5.5)
        plt.xticks(rotation=35, fontsize=7)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("**🏆 District Leaderboard**")
    lb1, lb2 = st.columns(2, gap="large")

    with lb1:
        top_yield = (feedback_df.groupby("district")["yield_kg"]
                     .mean().sort_values(ascending=False).head(5).reset_index())
        top_yield.columns = ["District", "Avg Yield (kg/ha)"]
        top_yield["Avg Yield (kg/ha)"] = top_yield["Avg Yield (kg/ha)"].round(0).astype(int)
        top_yield.index = ["🥇", "🥈", "🥉", "4th", "5th"]
        st.dataframe(top_yield, use_container_width=True)

    with lb2:
        top_sat = (feedback_df.groupby("district")["satisfaction"]
                   .mean().sort_values(ascending=False).head(5).reset_index())
        top_sat.columns = ["District", "Avg Satisfaction"]
        top_sat["Avg Satisfaction"] = top_sat["Avg Satisfaction"].round(2)
        top_sat.index = ["🥇", "🥈", "🥉", "4th", "5th"]
        st.dataframe(top_sat, use_container_width=True)

except FileNotFoundError:
    st.markdown("""
    <div style="background:#172518;border:1px dashed #2e4d30;border-radius:14px;
                padding:2.5rem;text-align:center;color:#8aab8c">
        <div style="font-size:2.5rem;margin-bottom:0.75rem">📭</div>
        <div style="font-size:1rem;color:#e8f0e9;margin-bottom:0.4rem">No feedback yet</div>
        <div style="font-size:0.85rem">Submit your first entry above to see analytics here.</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 8. Adaptive Retraining
# ============================================================
section("🧠", "Adaptive Retraining")
st.markdown(
    "<p style='color:#8aab8c;font-size:0.9rem;margin-top:-0.5rem'>"
    "Incorporate farmer feedback into the model to improve future predictions.</p>",
    unsafe_allow_html=True
)

if st.button("🔁 Retrain Model Using Latest Feedback", use_container_width=True):
    try:
        feedback_df = pd.read_csv("feedback_data.csv")
        base_df     = df.copy()
        feedback_df.rename(columns={"actual_crop": "dominant_crop"}, inplace=True)

        retrain_df = pd.concat([base_df, feedback_df], ignore_index=True)
        X = retrain_df[ALL_FEATURES]
        y = retrain_df["dominant_crop"].astype(str)

        le_crop = LabelEncoder()
        y_enc   = le_crop.fit_transform(y)
        joblib.dump(le_crop, "label_encoder.joblib")

        new_preprocessor = ColumnTransformer([
            ("num", StandardScaler(),                         NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"),   CATEGORICAL_FEATURES),
        ])
        clf_new      = lgb.LGBMClassifier(n_estimators=300, random_state=42)
        retrain_pipe = Pipeline(steps=[("preproc", new_preprocessor), ("clf", clf_new)])
        retrain_pipe.fit(X, y_enc)

        with open("best_tuned_model.pkl", "wb") as f:
            cloudpickle.dump(retrain_pipe, f)

        st.success("✅ Model retrained and saved! Restart the app to load the new model.")
        st.balloons()

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        from sklearn.utils.multiclass import unique_labels

        _, counts = np.unique(y_enc, return_counts=True)
        if counts.min() < 2:
            st.warning("⚠️ Some crops appear only once — skipping stratified split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_enc, test_size=0.2, random_state=42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

        y_pred = retrain_pipe.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, average="weighted")

        m1, m2 = st.columns(2)
        m1.metric("Accuracy (Validation)", f"{acc * 100:.2f}%")
        m2.metric("F1-Score (Weighted)",   f"{f1 * 100:.2f}%")

        present_labels = unique_labels(y_test, y_pred)
        active_classes = [le_crop.classes_[i] for i in present_labels]

        with st.expander("📋 Full Classification Report"):
            try:
                report = pd.DataFrame(
                    classification_report(
                        y_test, y_pred,
                        labels=present_labels,
                        target_names=active_classes,
                        output_dict=True
                    )
                ).T
                st.dataframe(report.style.format({
                    "precision": "{:.2f}",
                    "recall":    "{:.2f}",
                    "f1-score":  "{:.2f}"
                }))
            except Exception as e:
                st.warning(f"Report skipped: {e}")

        st.success("🌱 Updated model is ready. Restart the app to use it for predictions.")

    except Exception as e:
        st.error(f"Retraining failed: {e}")

# ============================================================
# 9. Footer
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:1rem 0 0.5rem;color:#4a6b4c;font-size:0.82rem">
    <span style="font-family:'Georgia',serif;font-size:1rem;color:#8aab8c">CropSense</span>
    &nbsp;·&nbsp; Adaptive Crop Recommendation with Real-Time Farmer Feedback
    &nbsp;·&nbsp; Developed by <strong style="color:#6fcf57">Namansh Singh Maurya</strong>
</div>
""", unsafe_allow_html=True)
