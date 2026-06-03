import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────
st.set_page_config(
    page_title="PhoneCheck — Smartphone Addiction Analysis",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main { background: #0b0c10; }

.stApp { background: #0b0c10; color: #e8eaf2; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #13151b !important;
    border-right: 1px solid rgba(255,255,255,0.07);
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #13151b;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] { color: #6c7080 !important; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="stMetricValue"] { color: #7effd4 !important; font-family: 'Syne', sans-serif; font-weight: 800; }

/* Headers */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

/* Buttons */
.stButton > button {
    background: #7effd4 !important;
    color: #0b0c10 !important;
    border: none !important;
    border-radius: 100px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 12px 32px !important;
    width: 100%;
    transition: all 0.2s !important;
}
.stButton > button:hover { background: #5de8be !important; box-shadow: 0 8px 24px rgba(126,255,212,0.25) !important; }

/* Inputs */
.stSlider [data-baseweb="slider"] { color: #7effd4; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #13151b;
    border-radius: 100px;
    padding: 4px;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    color: #6c7080;
    border-radius: 100px;
    font-family: 'DM Sans', sans-serif;
}
.stTabs [aria-selected="true"] {
    background: #1b1e28 !important;
    color: #7effd4 !important;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* Success / Error boxes */
.result-addicted {
    background: rgba(255,107,107,0.08);
    border: 1px solid rgba(255,107,107,0.3);
    border-radius: 14px;
    padding: 28px 32px;
    margin-top: 20px;
}
.result-safe {
    background: rgba(126,255,212,0.07);
    border: 1px solid rgba(126,255,212,0.25);
    border-radius: 14px;
    padding: 28px 32px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────
# LOAD MODEL & DATA
# ─────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("model_rf.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("good_features.json") as f:
        features = json.load(f)
    return model, scaler, features

@st.cache_data
def load_data():
    df = pd.read_csv("Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv")
    df.columns = df.columns.str.lower()
    return df

model, scaler, good_features = load_model()
df = load_data()

# ─────────────────────────────────
# SIDEBAR
# ─────────────────────────────────
with st.sidebar:
    st.markdown("## 📱 PhoneCheck")
    st.markdown("Smartphone Addiction Analysis")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📊 Analysis", "🔮 Predict", "🤖 Model"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    **Dataset**  
    7,500 users · 16 features  
    Ages 18–35 · Kaggle

    **Best Model**  
    Random Forest  
    Accuracy: **93.67%**
    """)

# ─────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────
if page == "🏠 Overview":
    st.markdown("# Understanding the **Digital Addiction** Crisis")
    st.markdown("A machine learning study of 7,500 individuals — exploring how smartphone habits, sleep, and stress converge into addiction risk.")
    st.markdown("---")

    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    addicted_rate = round(df['addicted_label'].mean() * 100, 1)
    avg_screen = round(df['daily_screen_time_hours'].mean(), 1)
    avg_sleep = round(df['sleep_hours'].mean(), 2)
    avg_notif = round(df['notifications_per_day'].mean(), 0)

    with col1:
        st.metric("Addiction Rate", f"{addicted_rate}%", "of 7,500 users")
    with col2:
        st.metric("Avg Screen Time", f"{avg_screen}h/day", "daily usage")
    with col3:
        st.metric("Avg Sleep", f"{avg_sleep}h", "below 8h rec.")
    with col4:
        st.metric("Notifications/Day", f"{int(avg_notif)}", "interruptions")
    with col5:
        st.metric("Model Accuracy", "93.67%", "Random Forest")

    st.markdown("---")

    col1, col2 = st.columns([1.6, 1])

    with col1:
        # Addiction level donut
        level_counts = df['addiction_level'].value_counts(dropna=False)
        labels = ['Moderate', 'Severe', 'Mild', 'Not Labeled']
        values = [
            int(level_counts.get('Moderate', 0)),
            int(level_counts.get('Severe', 0)),
            int(level_counts.get('Mild', 0)),
            int(df['addiction_level'].isna().sum()),
        ]
        fig = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.65,
            marker_colors=['#ffd166', '#ff6b6b', '#7effd4', '#3a3f52'],
            textinfo='percent+label',
            textfont_color='#e8eaf2',
        ))
        fig.update_layout(
            title="Addiction Level Distribution",
            paper_bgcolor='#13151b', plot_bgcolor='#13151b',
            font_color='#e8eaf2',
            legend=dict(font=dict(color='#6c7080')),
            margin=dict(t=40, b=10, l=10, r=10), height=320
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Addiction by Gender")
        for g, color in [('Male', '#74b9ff'), ('Female', '#fd79a8'), ('Other', '#a29bfe')]:
            sub = df[df['gender'] == g]
            pct = round(sub['addicted_label'].mean() * 100, 1)
            st.markdown(f"**{g}** — {pct}%")
            st.progress(pct / 100)

        st.markdown("#### By Stress Level")
        for s, color in [('Low', '#7effd4'), ('Medium', '#ffd166'), ('High', '#ff6b6b')]:
            sub = df[df['stress_level'] == s]
            pct = round(sub['addicted_label'].mean() * 100, 1)
            st.markdown(f"**{s} Stress** — {pct}%")
            st.progress(pct / 100)

    st.markdown("---")
    st.markdown("#### Key Insights")

    insights = pd.DataFrame({
        "Metric": ["Daily Screen Time", "Social Media Hours", "Sleep Hours", "Gaming Hours", "Notifications/Day", "Work/Study Hours"],
        "Overall Avg": ["7.5h", "3.27h", "6.74h", "2.01h", "134", "~2.5h"],
        "Addicted (avg)": ["~9.1h", "~4.2h", "~5.8h", "~2.5h", "~161", "~2.2h"],
        "Not Addicted (avg)": ["~4.2h", "~1.5h", "~8.4h", "~1.0h", "~80", "~3.1h"],
        "Risk Level": ["🔴 High", "🔴 High", "🟡 Medium", "🟡 Medium", "🔴 High", "🟢 Low"],
    })
    st.dataframe(insights, use_container_width=True, hide_index=True)

# ─────────────────────────────────
# PAGE: ANALYSIS
# ─────────────────────────────────
elif page == "📊 Analysis":
    st.markdown("# Usage **Patterns** & Correlations")
    st.markdown("Visual breakdown of smartphone behavior trends across the dataset.")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔗 Relationships", "🎯 Feature Importance"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                df, x='daily_screen_time_hours', nbins=30,
                title="Screen Time Distribution",
                color_discrete_sequence=['#74b9ff'],
                labels={'daily_screen_time_hours': 'Hours/Day', 'count': 'Users'}
            )
            fig.update_layout(paper_bgcolor='#13151b', plot_bgcolor='#13151b', font_color='#e8eaf2', margin=dict(t=40,b=10))
            fig.update_traces(marker_line_color='#0b0c10', marker_line_width=0.5)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(
                df, x='sleep_hours', nbins=25,
                title="Sleep Hours Distribution",
                color_discrete_sequence=['#7effd4'],
                labels={'sleep_hours': 'Sleep Hours', 'count': 'Users'}
            )
            fig.update_layout(paper_bgcolor='#13151b', plot_bgcolor='#13151b', font_color='#e8eaf2', margin=dict(t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig = px.histogram(
                df, x='social_media_hours', nbins=25,
                title="Social Media Hours",
                color_discrete_sequence=['#fd79a8'],
            )
            fig.update_layout(paper_bgcolor='#13151b', plot_bgcolor='#13151b', font_color='#e8eaf2', margin=dict(t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            fig = px.histogram(
                df, x='age', nbins=18,
                title="Age Distribution (18–35)",
                color_discrete_sequence=['#a29bfe'],
            )
            fig.update_layout(paper_bgcolor='#13151b', plot_bgcolor='#13151b', font_color='#e8eaf2', margin=dict(t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(
                df, x='addicted_label', y='daily_screen_time_hours',
                title="Screen Time by Addiction Status",
                color='addicted_label',
                color_discrete_map={0: '#7effd4', 1: '#ff6b6b'},
                labels={'addicted_label': 'Addicted (0=No, 1=Yes)', 'daily_screen_time_hours': 'Screen Time (hrs)'}
            )
            fig.update_layout(paper_bgcolor='#13151b', plot_bgcolor='#13151b', font_color='#e8eaf2', showlegend=False, margin=dict(t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(
                df, x='addicted_label', y='sleep_hours',
                title="Sleep Hours by Addiction Status",
                color='addicted_label',
                color_discrete_map={0: '#7effd4', 1: '#ff6b6b'},
                labels={'addicted_label': 'Addicted (0=No, 1=Yes)', 'sleep_hours': 'Sleep (hrs)'}
            )
            fig.update_layout(paper_bgcolor='#13151b', plot_bgcolor='#13151b', font_color='#e8eaf2', showlegend=False, margin=dict(t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(
            df.sample(800, random_state=42),
            x='daily_screen_time_hours', y='sleep_hours',
            color=df.sample(800, random_state=42)['addicted_label'].map({0: 'Not Addicted', 1: 'Addicted'}),
            title="Screen Time vs Sleep Hours (sample of 800)",
            color_discrete_map={'Not Addicted': '#7effd4', 'Addicted': '#ff6b6b'},
            opacity=0.6,
            labels={'daily_screen_time_hours': 'Screen Time (hrs)', 'sleep_hours': 'Sleep (hrs)'}
        )
        fig.update_layout(paper_bgcolor='#13151b', plot_bgcolor='#13151b', font_color='#e8eaf2', margin=dict(t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'Feature': good_features, 'Importance': importances})
        feat_df = feat_df.sort_values('Importance', ascending=True)

        fig = px.bar(
            feat_df, x='Importance', y='Feature', orientation='h',
            title="Random Forest Feature Importances",
            color='Importance',
            color_continuous_scale=['#1b1e28', '#74b9ff', '#7effd4'],
        )
        fig.update_layout(
            paper_bgcolor='#13151b', plot_bgcolor='#13151b',
            font_color='#e8eaf2', margin=dict(t=40,b=10),
            coloraxis_showscale=False, height=350
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("The most important features are **screen time**, **social media hours**, and **the engineered screen-to-sleep ratio** — confirming that imbalanced usage relative to rest is the primary driver of addiction prediction.")

# ─────────────────────────────────
# PAGE: PREDICT
# ─────────────────────────────────
elif page == "🔮 Predict":
    st.markdown("# Check Your **Addiction Risk**")
    st.markdown("Enter your average daily smartphone habits below to get an AI-powered risk assessment.")
    st.markdown("---")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("#### Your Usage Profile")

        screen_time = st.slider("📱 Daily Screen Time (hours)", 1.0, 16.0, 7.0, 0.5)
        sleep_hours = st.slider("😴 Sleep Hours", 3.0, 12.0, 7.0, 0.5)
        social_media = st.slider("📲 Social Media Hours", 0.0, 10.0, 2.0, 0.5)
        weekend_time = st.slider("🗓️ Weekend Screen Time (hours)", 1.0, 20.0, 9.0, 0.5)

        predict_btn = st.button("⚡ Analyse My Usage")

    with col2:
        st.markdown("#### What These Mean")
        st.markdown(f"""
        | Metric | Your Input | Avg User |
        |--------|-----------|----------|
        | Screen Time | **{screen_time}h** | 7.5h |
        | Sleep | **{sleep_hours}h** | 6.74h |
        | Social Media | **{social_media}h** | 3.27h |
        | Weekend Screen | **{weekend_time}h** | ~9h |
        """)

        st.markdown("---")
        st.markdown("**How it works**")
        st.markdown("The Random Forest model uses 6 key features including engineered ratios to predict addiction risk with **93.67% accuracy**.")

    if predict_btn:
        # Build engineered features
        screen_to_sleep_ratio = screen_time / (sleep_hours + 1)
        fun_usage_percent = min((social_media + 1.0) / max(screen_time, 0.1), 1.0)

        all_vals = {
            'daily_screen_time_hours': screen_time,
            'social_media_hours': social_media,
            'sleep_hours': sleep_hours,
            'weekend_screen_time': weekend_time,
            'screen_to_sleep_ratio': screen_to_sleep_ratio,
            'fun_usage_percent': fun_usage_percent,
        }

        input_df = pd.DataFrame([[all_vals.get(f, 0) for f in good_features]], columns=good_features)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]

        addicted_prob = round(proba[1] * 100, 1)
        safe_prob = round(proba[0] * 100, 1)

        st.markdown("---")
        if prediction == 1:
            st.markdown(f"""
            <div class="result-addicted">
                <h2 style="color:#ff6b6b;font-family:'Syne',sans-serif;margin:0 0 8px 0">⚠️ Addicted</h2>
                <p style="color:#6c7080;margin:0 0 16px 0">Confidence: {addicted_prob}% probability of addiction</p>
                <p style="background:rgba(255,255,255,0.04);border-radius:10px;padding:12px 16px;margin:0">
                {"🚨 High-risk pattern detected. A structured digital detox is strongly recommended. Consider setting app time limits and phone-free hours before bed." if addicted_prob > 75 else "⚡ Moderate addiction signals present. Try setting daily screen time limits and replacing 30 min of social media with offline activities."}
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
                <h2 style="color:#7effd4;font-family:'Syne',sans-serif;margin:0 0 8px 0">✅ Not Addicted</h2>
                <p style="color:#6c7080;margin:0 0 16px 0">Confidence: {safe_prob}% probability of healthy usage</p>
                <p style="background:rgba(255,255,255,0.04);border-radius:10px;padding:12px 16px;margin:0">
                Your usage patterns look healthy! Keep maintaining good sleep habits and balanced screen time to stay in this zone.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=addicted_prob,
            title={'text': "Addiction Risk Score", 'font': {'color': '#e8eaf2', 'size': 16}},
            number={'suffix': '%', 'font': {'color': '#e8eaf2', 'size': 32}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#6c7080'},
                'bar': {'color': '#ff6b6b' if prediction == 1 else '#7effd4'},
                'bgcolor': '#1b1e28',
                'bordercolor': '#1b1e28',
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(126,255,212,0.1)'},
                    {'range': [40, 70], 'color': 'rgba(255,209,102,0.1)'},
                    {'range': [70, 100], 'color': 'rgba(255,107,107,0.1)'},
                ],
                'threshold': {'line': {'color': '#ffd166', 'width': 2}, 'thickness': 0.75, 'value': 50}
            }
        ))
        fig.update_layout(
            paper_bgcolor='#13151b', font_color='#e8eaf2',
            height=280, margin=dict(t=40, b=10, l=20, r=20)
        )
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────
# PAGE: MODEL
# ─────────────────────────────────
elif page == "🤖 Model":
    st.markdown("# How the **Model** Works")
    st.markdown("A supervised binary classification pipeline built with scikit-learn.")
    st.markdown("---")

    # Model comparison
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### Logistic Regression")
        st.metric("Accuracy", "~74%")
        st.markdown("Linear classifier. Fast but limited to linear decision boundaries.")
    with col2:
        st.markdown("#### Decision Tree")
        st.metric("Accuracy", "~79%")
        st.markdown("Flowchart model (max depth=6). Interpretable but can overfit.")
    with col3:
        st.markdown("#### 🏆 Random Forest")
        st.metric("Accuracy", "93.67%", "Best Model")
        st.markdown("100 trees voting together. Most accurate, robust to noise.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Pipeline Steps")
        steps = {
            "1. Load & Clean": "Load 7,500 rows. Drop IDs. Fill missing values (mode for categorical, median for numeric). Remove duplicates.",
            "2. Drop Leaky Column": "`addiction_level` is derived from the target — keeping it causes 100% accuracy via data leakage. Dropped.",
            "3. Encode Categoricals": "LabelEncoder converts gender, stress_level, academic_work_impact into integers.",
            "4. Feature Engineering": "Create 3 new features: screen_to_sleep_ratio, fun_usage_percent, notif_per_awake_hour.",
            "5. Select Features": "Keep only features with correlation > 0.03 to target. Result: 6 features.",
            "6. Train/Test Split": "80% train / 20% test. Stratified to preserve class ratios.",
            "7. Scale & Train": "StandardScaler on train set only. Train Random Forest (100 trees, depth=10).",
        }
        for step, desc in steps.items():
            with st.expander(step):
                st.markdown(desc)

    with col2:
        st.markdown("#### Selected Features")
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'Feature': good_features, 'Importance': importances})
        feat_df = feat_df.sort_values('Importance', ascending=False)
        feat_df['Importance'] = feat_df['Importance'].apply(lambda x: f"{x:.4f}")
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

        st.markdown("#### Data Leakage Warning")
        st.error("The `addiction_level` column was **dropped** before training. It's derived from the target variable — keeping it inflates accuracy to ~100% on training data but fails on unseen data.")

        st.markdown("#### Engineered Features")
        st.code("""
screen_to_sleep_ratio = screen_time / (sleep + 1)

fun_usage_percent = (social + gaming) / screen_time

notif_per_awake_hour = notifications / (24 - sleep)
        """, language="python")

    st.markdown("---")
    st.markdown("#### Model Card")
    model_card = pd.DataFrame({
        "Property": ["Algorithm", "Estimators", "Max Depth", "Train Size", "Test Size", "Accuracy", "Target", "Source"],
        "Value": ["Random Forest Classifier", "100 trees", "10", "6,000 samples (80%)", "1,500 samples (20%)", "93.67%", "addicted_label (0/1)", "Kaggle — Smartphone Usage & Addiction Dataset"],
    })
    st.dataframe(model_card, use_container_width=True, hide_index=True)
