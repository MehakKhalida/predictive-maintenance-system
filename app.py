import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Page settings
st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide"
)

# Title
st.title("⚙️ Predictive Maintenance System")
st.write(
    "A Machine Learning based system for monitoring "
    "equipment temperature and vibration."
)

# -----------------------------
# Generate simulated sensor data
# -----------------------------

np.random.seed(42)

temperature = np.random.normal(60, 5, 100)
vibration = np.random.normal(3, 0.5, 100)

# Create some abnormal readings
temperature[85:90] = np.random.normal(90, 3, 5)
vibration[85:90] = np.random.normal(8, 0.5, 5)

data = pd.DataFrame({
    "Temperature": temperature,
    "Vibration": vibration
})

# -----------------------------
# Machine Learning Model
# -----------------------------

model = IsolationForest(
    contamination=0.10,
    random_state=42
)

data["Anomaly"] = model.fit_predict(
    data[["Temperature", "Vibration"]]
)

data["Status"] = data["Anomaly"].apply(
    lambda x: "⚠️ Anomaly" if x == -1 else "✅ Normal"
)

# -----------------------------
# Dashboard
# -----------------------------

st.subheader("📊 Latest Sensor Readings")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Temperature",
        f"{data['Temperature'].iloc[-1]:.2f} °C"
    )

with col2:
    st.metric(
        "Vibration",
        f"{data['Vibration'].iloc[-1]:.2f}"
    )

with col3:
    st.metric(
        "Total Readings",
        len(data)
    )

# -----------------------------
# Alert
# -----------------------------

anomalies = data[data["Anomaly"] == -1]

if len(anomalies) > 0:
    st.error(
        f"⚠️ WARNING: {len(anomalies)} abnormal "
        "sensor readings detected. Equipment may require inspection."
    )
else:
    st.success(
        "✅ Equipment is operating normally."
    )

# -----------------------------
# Temperature Graph
# -----------------------------

st.subheader("🌡️ Temperature Monitoring")

fig1, ax1 = plt.subplots()

ax1.plot(
    data["Temperature"],
    label="Temperature"
)

ax1.set_xlabel("Sensor Reading")
ax1.set_ylabel("Temperature (°C)")
ax1.set_title("Temperature Sensor Data")
ax1.legend()

st.pyplot(fig1)

# -----------------------------
# Vibration Graph
# -----------------------------

st.subheader("📳 Vibration Monitoring")

fig2, ax2 = plt.subplots()

ax2.plot(
    data["Vibration"],
    label="Vibration"
)

ax2.set_xlabel("Sensor Reading")
ax2.set_ylabel("Vibration")
ax2.set_title("Vibration Sensor Data")
ax2.legend()

st.pyplot(fig2)

# -----------------------------
# Anomaly Results
# -----------------------------

st.subheader("🚨 Detected Anomalies")

if len(anomalies) > 0:

    st.dataframe(
        anomalies[
            ["Temperature", "Vibration", "Status"]
        ],
        use_container_width=True
    )

else:

    st.write("No anomalies detected.")
