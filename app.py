import streamlit as st
import numpy as np
import joblib

# 1. Load Assets
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')
stats = joblib.load('feature_stats.pkl')

# 2. Dynamic Thresholds (2.2 Sigma)
amt_threshold = stats['Amount']['mean'] + (2.2 * stats['Amount']['std'])
v14_danger = stats['V14']['mean'] - (2.2 * stats['V14']['std'])
v17_danger = stats['V17']['mean'] - (2.2 * stats['V17']['std'])

st.set_page_config(page_title="Fraud Guard", page_icon="🛡️")
st.title("🛡️ Fraud Guard Analysis Based on Everyday Transactions")

# --- INPUT SECTION ---
with st.sidebar:
    st.header("Transaction Inputs")
    amount = st.number_input("Amount ($)", min_value=0.0, value=float(stats['Amount']['mean']))
    v14 = st.slider("Sample Statistical Anomaly in Transaction Metadata A", float(stats['V14']['min']), float(stats['V14']['max']), float(stats['V14']['mean']))
    v17 = st.slider("Sample Statistical Anomaly in Transaction Metadata B", float(stats['V17']['min']), float(stats['V17']['max']), float(stats['V17']['mean']))

# --- PROCESSING ---
if st.button("Analyze Transaction"):
    # Build vector & Predict
    vec = np.zeros((1, 30))
    vec[0, 14], vec[0, 17], vec[0, 29] = v14, v17, amount
    
    prob = model.predict_proba(scaler.transform(vec))[0][1]
    
    # 2.2 Sigma Overrides
    reasons = []
    if amount > amt_threshold:
        prob = max(prob, 0.95)
        reasons.append(f"Risky Amount Transacted given the other two metrics: (>$ {amt_threshold:,.2f})")
    if v14 < v14_danger or v17 < v17_danger:
        prob = max(prob, 0.85)
        reasons.append("Statistical Pattern Anomaly (V14/V17)")

    # --- UI OUTPUT ---
    # Map probability to Labels
    if prob < 0.15:
        level, color, icon = "LOW", "green", "✅"
    elif prob < 0.60:
        level, color, icon = "MEDIUM", "orange", "⚠️"
    else:
        level, color, icon = "HIGH", "red", "🚨"

    # Big Qualitative Result
    st.markdown(f"### Overall Risk: :{color}[{icon} {level}]")
    
    
    
    # Detailed Numbers (Hidden by default)
    with st.expander("📊 View Technical Audit Data"):
        st.write(f"**Exact Fraud Probability:** {prob:.4%}")
        st.write(f"**Amount Z-Score:** {(amount - stats['Amount']['mean'])/stats['Amount']['std']:.2f}σ")
        if reasons:
            st.write("**Flags Triggered:**")
            for r in reasons: st.write(f"- {r}")

    # Visual Meter
    st.progress(prob)