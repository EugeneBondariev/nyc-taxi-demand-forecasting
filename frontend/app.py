import os
from datetime import datetime

import httpx
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("NYC Taxi Demand Forecast")

zone_id = st.slider("NYC Zone ID", 1, 265, 166)
prediction_time = st.datetime_input(
    "Prediction time",
    value=datetime.now().replace(minute=0, second=0, microsecond=0),
)
payload = {
    "zone_id": zone_id,
    "prediction_time": prediction_time.isoformat(),
}

col1, col2 = st.columns(2)

if col1.button("Get prediction"):
    r = httpx.post(f"{API_URL}/v1/predict", json=payload)
    if r.is_success:
        data = r.json()
        st.metric("Predicted trips", data["predicted_trips"])
        if data.get("lower_bound") is not None:
            st.markdown(
                f"**80% interval:** {data['lower_bound']} – {data['upper_bound']} trips"
            )
        st.badge("Success", icon=":material/check:", color="green")
    else:
        st.error(f"Error {r.status_code}: {r.text}")
        st.badge("Failure", color="red")

if col2.button("Explain prediction"):
    r = httpx.post(f"{API_URL}/v1/explain", json=payload)
    if r.is_success:
        contributions = r.json()["feature_contributions"]
        st.subheader("SHAP feature contributions (XGBoost)")
        sorted_items: list[tuple[str, float]] = sorted(
            contributions.items(), key=lambda x: abs(x[1]), reverse=True
        )
        features: list[str] = [k for k, _ in sorted_items]
        values: list[float] = [v for _, v in sorted_items]
        colors: list[str] = ["#d62728" if v > 0 else "#1f77b4" for v in values]
        df_shap = pd.DataFrame({"SHAP value": values}, index=features)
        st.bar_chart(df_shap, color=colors)
        st.caption("Red = increases predicted demand · Blue = decreases it")
    else:
        st.error(f"Error {r.status_code}: {r.text}")

st.divider()
st.subheader("System Status")
tab_models, tab_drift, tab_ab = st.tabs(
    ["Model Versions", "Feature Drift", "A/B Results"]
)

with tab_models:
    if st.button("Load models"):
        r = httpx.get(f"{API_URL}/v1/models")
        if r.is_success:
            rows = r.json()
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("No model versions in database yet.")
        else:
            st.error(f"Error {r.status_code}: {r.text}")

with tab_drift:
    if st.button("Check drift"):
        r = httpx.get(f"{API_URL}/v1/drift")
        if r.is_success:
            data = r.json()
            if not data["baseline_exists"]:
                st.warning("No feature baseline found — run training first.")
            elif data["rows_analyzed"] == 0:
                st.warning("No demand history rows to analyse.")
            else:
                st.metric("Rows analysed", data["rows_analyzed"])
                st.metric("Features checked", len(data["features_checked"]))
                if data["drifted_features"]:
                    st.error(f"Drifted features: {', '.join(data['drifted_features'])}")
                else:
                    st.success("No drift detected.")
        else:
            st.error(f"Error {r.status_code}: {r.text}")

with tab_ab:
    if st.button("Load A/B results"):
        r = httpx.get(f"{API_URL}/v1/ab-results")
        if r.is_success:
            rows = r.json()
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("No model versions in database yet.")
        else:
            st.error(f"Error {r.status_code}: {r.text}")
