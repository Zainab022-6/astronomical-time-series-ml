import streamlit as st
import numpy as np
import pandas as pd
import lightkurve as lk
from astropy.timeseries import LombScargle
from scipy.stats import skew, kurtosis
import plotly.graph_objects as go
import joblib

st.set_page_config(page_title="Stellar Variability Classifier", layout="wide")

st.title("Stellar Variability Detection Dashboard")
st.markdown("Automated classification of stellar light curves using real Kepler & TESS telemetry.")

# Load pre-trained model
@st.cache_resource
def load_model():
    return joblib.load("stellar_classifier.joblib")

model = load_model()

# Sidebar configuration
st.sidebar.header("Target Configuration")
target_id = st.sidebar.text_input("Enter Target ID / Star Name", value="RR Lyr")
mission = st.sidebar.selectbox("Select Mission", ["Kepler", "TESS"])
analyze_btn = st.sidebar.button("Fetch & Analyze Telemetry")

def process_target(target, mission_name):
    search_result = lk.search_lightcurve(target, mission=mission_name)
    if len(search_result) == 0:
        return None, None, None, None
    
    # Download first sector/quarter
    lc = search_result[0].download()
    clean_lc = lc.remove_nans().remove_outliers(sigma=4).normalize()
    
    time = clean_lc.time.value
    flux = clean_lc.flux.value
    
    # Statistical features
    std_val = float(np.std(flux))
    iqr_val = float(np.percentile(flux, 95) - np.percentile(flux, 5)) / 2.0
    skew_val = float(skew(flux))
    kurt_val = float(kurtosis(flux))
    
    # Lomb-Scargle Periodogram
    ls = LombScargle(time, flux)
    freq, power = ls.autopower(minimum_frequency=1/20.0, maximum_frequency=1/0.05)
    best_freq = freq[np.argmax(power)]
    best_period = float(1.0 / best_freq)
    peak_power = float(np.max(power))
    
    features = {
        "period_days": best_period,
        "peak_power": peak_power,
        "amplitude": iqr_val,
        "flux_std": std_val,
        "skewness": skew_val,
        "kurtosis": kurt_val
    }
    
    return clean_lc, features, freq, power

if analyze_btn:
    with st.spinner(f"Querying NASA MAST servers for '{target_id}'..."):
        clean_lc, features, freq, power = process_target(target_id, mission)
        
    if clean_lc is None:
        st.error(f"No light curve observations found for '{target_id}' in mission {mission}.")
    else:
        feature_order = ["period_days", "peak_power", "amplitude", "flux_std", "skewness", "kurtosis"]
        input_data = pd.DataFrame([features])[feature_order]
        
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        st.markdown("---")
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            if prediction == 1:
                st.success("### Verdict: Variable Star")
            else:
                st.info("### Verdict: Stable Star")
            
            prob_var = probabilities[1] if len(probabilities) > 1 else 1.0
            st.metric("Variable Probability", f"{prob_var * 100:.1f}%")
            st.metric("Detected Period", f"{features['period_days']:.4f} days")
            st.metric("Lomb-Scargle Power", f"{features['peak_power']:.3f}")

        with col_res2:
            st.write("#### Extracted Time-Series Features")
            st.dataframe(pd.DataFrame([features]), use_container_width=True)

        st.markdown("---")
        st.subheader("Interactive Telemetry Visualizations")
        tab1, tab2, tab3 = st.tabs(["Raw Normalized Light Curve", "Phase-Folded Curve", "Lomb-Scargle Spectrum"])
        
        with tab1:
            fig_raw = go.Figure()
            fig_raw.add_trace(go.Scattergl(x=clean_lc.time.value, y=clean_lc.flux.value, mode="markers",
                                          marker=dict(size=2, color="#1f77b4")))
            fig_raw.update_layout(title=f"Normalized Flux vs Time ({target_id})",
                                  xaxis_title="Time (Days)", yaxis_title="Normalized Flux")
            st.plotly_chart(fig_raw, use_container_width=True)
            
        with tab2:
            period = features["period_days"]
            phase = np.mod((clean_lc.time.value - clean_lc.time.value[0]) / period, 1.0)
            
            fig_fold = go.Figure()
            fig_fold.add_trace(go.Scattergl(x=phase, y=clean_lc.flux.value, mode="markers",
                                           marker=dict(size=2, color="#ff7f0e")))
            fig_fold.update_layout(title=f"Phase-Folded at P = {period:.4f} days",
                                   xaxis_title="Phase (0.0 to 1.0)", yaxis_title="Normalized Flux")
            st.plotly_chart(fig_fold, use_container_width=True)
            
        with tab3:
            fig_pg = go.Figure()
            fig_pg.add_trace(go.Scatter(x=1.0 / freq, y=power, mode="lines", line=dict(color="#2ca02c")))
            fig_pg.update_layout(title="Lomb-Scargle Periodogram",
                                 xaxis_title="Period (Days)", yaxis_title="Spectral Power")
            st.plotly_chart(fig_pg, use_container_width=True)