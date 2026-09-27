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

if st.button("Get prediction"):
    payload = {
        "zone_id": zone_id,
        "prediction_time": prediction_time.isoformat(),
        "is_holiday": int(is_holiday),
        "snowfall": snowfall,
    }
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
