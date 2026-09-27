import os
from datetime import datetime

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("NYC Taxi Demand Forecast")

zone_id = st.slider("NYC Zone ID", 1, 265, 166)
prediction_time = st.datetime_input(
    "Prediction time",
    value=datetime.now().replace(minute=0, second=0, microsecond=0),
)
is_holiday = st.checkbox("Public holiday")
snowfall = st.slider("Snowfall (cm)", min_value=0.0, max_value=20.0, value=0.0, step=0.5)

payload = {
    "zone_id": zone_id,
    "prediction_time": prediction_time.isoformat(),
    "is_holiday": int(is_holiday),
    "snowfall": snowfall,
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
        sorted_items = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        features = [k for k, _ in sorted_items]
        values = [v for _, v in sorted_items]
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in values]
        chart_data = {
            "Feature": features,
            "SHAP value": values,
        }
        import pandas as pd
        df_shap = pd.DataFrame(chart_data).set_index("Feature")
        st.bar_chart(df_shap, color=colors)
        st.caption("Red = increases predicted demand · Blue = decreases it")
    else:
        st.error(f"Error {r.status_code}: {r.text}")
