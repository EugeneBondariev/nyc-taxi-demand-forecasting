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

if st.button("Get prediction"):
    payload = {
        "zone_id": zone_id,
        "prediction_time": prediction_time.isoformat(),
        "is_holiday": int(is_holiday),
    }
    r = httpx.post(f"{API_URL}/v1/predict", json=payload)

    if r.is_success:
        st.markdown(f"Trips prediction: **{r.json()['predicted_trips']}**")
        st.badge("Success", icon=":material/check:", color="green")
    else:
        st.error(f"Error {r.status_code}: {r.text}")
        st.badge("Failure", color="red")
