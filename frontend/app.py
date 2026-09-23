import streamlit as st
import httpx
from datetime import datetime

zone_id = st.slider("NYC Zone Id", 1, 265, 166)

hour = st.selectbox(
    "Hour",
    range(0, 24),
    index=16,
    placeholder="Select hour in range from 0 to 23",
)

days_of_week = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

day_of_week = st.selectbox(
    "Day of Week",
    days_of_week,
    index=2,
    placeholder="Select the day of week",
)

week = st.slider("Week Number", 1, 53, datetime.now().isocalendar().week)

if st.button("Get prediction"):
    st.markdown("The request has been sent")
    r = httpx.post(
        "http://localhost:8000/predict",
        json={
            "zone_id": zone_id,
            "hour": hour,
            "day_of_week": days_of_week.index(day_of_week),
            "week": week,
        },
    )

    st.markdown(f"Trips prediction: **{r.json()["predicted_trips"]}**")

    if r.is_success:
        st.badge("Success", icon=":material/check:", color="green")
    else:
        st.badge("Failure", color="red")
