# predictive-maintenance-system

# ⚙️ Predictive Maintenance System

## 📌 Project Overview

This project implements a simple predictive maintenance system that monitors equipment using temperature and vibration sensor data.

Machine Learning is used to identify abnormal sensor readings that may indicate possible equipment problems.

## 🎯 Objective

The main objective is to:

- Monitor temperature data
- Monitor vibration data
- Detect abnormal equipment conditions
- Use Machine Learning for anomaly detection
- Display the results through a Python dashboard
- Alert the user when abnormal readings are detected

## 🔧 Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit

## 🤖 Machine Learning

The project uses the **Isolation Forest** Machine Learning algorithm to detect abnormal temperature and vibration readings.

## 📊 Dashboard Features

The dashboard displays:

- Latest temperature reading
- Latest vibration reading
- Total sensor readings
- Temperature graph
- Vibration graph
- Detected anomalies
- Equipment warning alert

## 📡 Sensor Data

For this mini project, simulated temperature and vibration sensor data are used instead of physical sensors.

Abnormal readings are also simulated to demonstrate how the predictive maintenance system identifies possible equipment problems.

## 🚨 Alert System

When abnormal sensor readings are detected, the dashboard displays a warning indicating that the equipment may require inspection.

## 📁 Project Files

```text
predictive-maintenance-week2
│
├── app.py
├── requirements.txt
└── README.md
