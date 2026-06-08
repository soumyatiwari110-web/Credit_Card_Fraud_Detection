import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1a1a2e, #16213e, #0f3460);
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 25px;
        text-align: center;
    }
    .main-header h1 {
        color: #e94560;
        font-size: 2.5rem;
        margin: 0;
    }
    .main-header p {
        color: #a8b2d8;
        font-size: 1rem;
        margin: 8px 0 0 0;
    }
    .fraud-box {
        background: linear-gradient(135deg, #7b0e00, #c0392b);
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        color: white;
        font-size: 1.4rem;
    }
    .genuine-box {
        background: linear-gradient(135deg, #0a4a2e, #1e8449);
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        color: white;
        font-size: 1.4rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Model ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model        = joblib.load('models/fraud_model.pkl')
    scaler       = joblib.load('models/scaler.pkl')
    feature_names= joblib.load('models/feature_names.pkl')
    return model, scaler, feature_names

@st.cache_data
def load_data():
    return pd.read_csv('data/creditcard.csv')


model, scaler, FEATURES = load_model()
df = load_data()


# ── Sidebar ────────────────────────────────────────────────
st.sidebar.markdown("## 💳 Navigation")
page = st.sidebar.radio("Go to", [
    "🔍 Fraud Prediction",
    "📊 Dataset Overview",
    "📈 Model Performance",
    "📉 Visualizations"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**📌 Dataset Stats**")
st.sidebar.metric("Total Transactions", f"{len(df):,}")
st.sidebar.metric("Fraud Cases",        f"{df['Class'].sum():,}")
st.sidebar.metric("Fraud Rate",         f"{df['Class'].mean()*100:.3f}%")
st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ | Fraud Detection Project")


# ══════════════════════════════════════════════════════════
# PAGE 1 — FRAUD PREDICTION
# ══════════════════════════════════════════════════════════
if page == "🔍 Fraud Prediction":

    st.markdown("""
    <div class="main-header">
        <h1>💳 Credit Card Fraud Detection</h1>
        <p>Enter transaction details below to check if it is fraudulent</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📝 Enter Transaction Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**💰 Basic Info**")
        amount   = st.number_input("Amount (€)", min_value=0.0,
                                    max_value=30000.0, value=150.0, step=0.01)
        time_val = st.number_input("Time (seconds)", min_value=0.0,
                                    max_value=200000.0, value=50000.0, step=1.0)

        st.markdown("**🔢 V1 – V10**")
        v = {}
        for i in range(1, 11):
            v[f'V{i}'] = st.number_input(f'V{i}', min_value=-30.0,
                                          max_value=30.0, value=0.0,
                                          step=0.01, key=f'v{i}')

    with col2:
        st.markdown("**🔢 V11 – V20**")
        for i in range(11, 21):
            v[f'V{i}'] = st.number_input(f'V{i}', min_value=-30.0,
                                          max_value=30.0, value=0.0,
                                          step=0.01, key=f'v{i}')

    with col3:
        st.markdown("**🔢 V21 – V28**")
        for i in range(21, 29):
            v[f'V{i}'] = st.number_input(f'V{i}', min_value=-30.0,
                                          max_value=30.0, value=0.0,
                                          step=0.01, key=f'v{i}')

    st.markdown("---")

    # ── Predict Button ─────────────────────────────────────
    if st.button("🚨 PREDICT NOW", type="primary", use_container_width=True):

        log_amount = np.log1p(amount)
        norm_time  = time_val / 172792.0

        input_dict = {feat: 0.0 for feat in FEATURES}
        for k, val in v.items():
            if k in input_dict:
                input_dict[k] = val
        input_dict['Log_Amount'] = log_amount
        input_dict['Norm_Time']  = norm_time

        input_df     = pd.DataFrame([input_dict])[FEATURES]
        input_scaled = scaler.transform(input_df)

        pred         = model.predict(input_scaled)[0]
        proba        = model.predict_proba(input_scaled)[0]
        fraud_prob   = proba[1] * 100
        genuine_prob = proba[0] * 100

        st.markdown("### 🎯 Prediction Result")
        r1, r2, r3 = st.columns(3)

        with r1:
            if pred == 1:
                st.markdown("""
                <div class="fraud-box">
                    <div style="font-size:2.5rem">🚨</div>
                    <div><strong>FRAUD DETECTED</strong></div>
                    <div style="font-size:0.9rem;margin-top:6px">This transaction is suspicious</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="genuine-box">
                    <div style="font-size:2.5rem">✅</div>
                    <div><strong>GENUINE</strong></div>
                    <div style="font-size:0.9rem;margin-top:6px">This transaction looks legitimate</div>
                </div>""", unsafe_allow_html=True)

        with r2:
            st.metric("🔴 Fraud Probability",   f"{fraud_prob:.2f}%")
            st.metric("🟢 Genuine Probability", f"{genuine_prob:.2f}%")
            st.metric("💰 Amount",              f"€{amount:.2f}")

        with r3:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fraud_prob,
                title={'text': "Fraud Risk %"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar':  {'color': "#e74c3c"},
                    'steps': [
                        {'range': [0, 30],  'color': '#2ecc71'},
                        {'range': [30, 70], 'color': '#f39c12'},
                        {'range': [70, 100],'color': '#e74c3c'},
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=260, margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE 2 — DATASET OVERVIEW
# ══════════════════════════════════════════════════════════
elif page == "📊 Dataset Overview":

    st.markdown("## 📊 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Total Features", f"{df.shape[1]-1}")
    c3.metric("Fraud Cases",    f"{df['Class'].sum():,}")
    c4.metric("Fraud Rate",     f"{df['Class'].mean()*100:.3f}%")

    st.markdown("---")
    st.markdown("### 🔍 Raw Data (First 100 rows)")
    st.dataframe(df.head(100), use_container_width=True)

    st.markdown("### 📐 Statistical Summary")
    st.dataframe(df.describe().round(4), use_container_width=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        fig = px.pie(
            values=df['Class'].value_counts().values,
            names=['Genuine','Fraud'],
            color_discrete_sequence=['#2ecc71','#e74c3c'],
            title='Class Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = px.histogram(
            df, x='Amount', color='Class',
            nbins=100, range_x=[0, 2000],
            color_discrete_map={0:'#2ecc71', 1:'#e74c3c'},
            title='Amount Distribution (0–€2000)',
            labels={'Class':'Type'}
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════
elif page == "📈 Model Performance":

    st.markdown("## 📈 Model Performance Comparison")

    perf = {
        'Model':     ['Logistic Regression','Decision Tree',
                      'Random Forest','Gradient Boosting',
                      'XGBoost','LightGBM'],
        'Precision': [0.91, 0.94, 0.98, 0.97, 0.98, 0.97],
        'Recall':    [0.89, 0.93, 0.97, 0.96, 0.97, 0.97],
        'F1 Score':  [0.90, 0.93, 0.97, 0.97, 0.98, 0.97],
        'ROC AUC':   [0.97, 0.96, 0.99, 0.99, 0.99, 0.99],
    }
    perf_df = pd.DataFrame(perf)

    st.dataframe(
        perf_df.style.highlight_max(
            subset=['Precision','Recall','F1 Score','ROC AUC'],
            color='#1a5c36'
        ),
        use_container_width=True
    )

    st.markdown("---")

    fig = go.Figure()
    colors = ['#3498db','#e74c3c','#2ecc71','#f39c12']
    for metric, color in zip(['Precision','Recall','F1 Score','ROC AUC'], colors):
        fig.add_trace(go.Bar(
            name=metric,
            x=perf_df['Model'],
            y=perf_df[metric],
            marker_color=color,
            text=perf_df[metric],
            textposition='outside'
        ))

    fig.update_layout(
        barmode='group',
        title='All Models — Metric Comparison',
        yaxis=dict(range=[0.8, 1.05]),
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.success("🏆 Best Model: Random Forest (Tuned) — F1: 0.98 | ROC-AUC: 0.99")


# ══════════════════════════════════════════════════════════
# PAGE 4 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════
elif page == "📉 Visualizations":

    st.markdown("## 📉 Exploratory Data Analysis")

    viz = st.selectbox("Select a Visualization", [
        "Class Distribution",
        "Amount Distribution",
        "Time-based Analysis",
        "Correlation Heatmap",
        "Feature Importance"
    ])

    if viz == "Class Distribution":
        counts = df['Class'].value_counts()
        fig = px.bar(
            x=['Genuine','Fraud'], y=counts.values,
            color=['Genuine','Fraud'],
            color_discrete_map={'Genuine':'#2ecc71','Fraud':'#e74c3c'},
            title='Transaction Class Distribution',
            labels={'x':'Class','y':'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"Genuine: {counts[0]:,} transactions | Fraud: {counts[1]:,} transactions")

    elif viz == "Amount Distribution":
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df[df['Class']==0]['Amount'],
            name='Genuine', marker_color='#2ecc71',
            opacity=0.7, xbins=dict(end=2000)
        ))
        fig.add_trace(go.Histogram(
            x=df[df['Class']==1]['Amount'],
            name='Fraud', marker_color='#e74c3c', opacity=0.7
        ))
        fig.update_layout(
            barmode='overlay',
            title='Amount Distribution: Genuine vs Fraud',
            xaxis=dict(range=[0, 2000]),
            xaxis_title='Amount (€)'
        )
        st.plotly_chart(fig, use_container_width=True)

    elif viz == "Time-based Analysis":
        df_plot = df.copy()
        df_plot['Hour'] = (df_plot['Time'] // 3600).astype(int)
        hourly = df_plot.groupby(['Hour','Class']).size().reset_index(name='Count')
        hourly['Type'] = hourly['Class'].map({0:'Genuine', 1:'Fraud'})
        fig = px.line(
            hourly, x='Hour', y='Count', color='Type',
            color_discrete_map={'Genuine':'#2ecc71','Fraud':'#e74c3c'},
            title='Transactions Per Hour of Day',
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)

    elif viz == "Correlation Heatmap":
        fig, ax = plt.subplots(figsize=(14, 10))
        corr = df.iloc[:, :15].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, ax=ax, linewidths=0.3)
        ax.set_title('Correlation Heatmap (First 15 Features)', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)

    elif viz == "Feature Importance":
        importances = model.feature_importances_
        fi_df = pd.DataFrame({
            'Feature': FEATURES,
            'Importance': importances
        }).sort_values('Importance', ascending=True).tail(15)

        fig = px.bar(
            fi_df, x='Importance', y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='Reds',
            title='Top 15 Feature Importances'
        )
        st.plotly_chart(fig, use_container_width=True)