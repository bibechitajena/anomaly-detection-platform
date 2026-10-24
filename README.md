# 🛡️ Real-Time Anomaly Detection Platform

A modular, beginner-friendly **Real-Time Anomaly Detection Platform** built with **Python**, **Streamlit**, **scikit-learn**, **pandas**, and **Plotly**.

The platform simulates server telemetry (CPU usage, memory usage, response time, and request count) and utilizes an **Isolation Forest** machine learning model trained on healthy baseline operations to flag abnormal behavior, resource exhaustion, and traffic surges in real time.

---

## 🏗️ Architecture Overview

The system is designed with a clean, decoupled architecture:

```
┌────────────────────────────────────────────────────────────────────────┐
│                            Streamlit Web UI                            │
│ (app.py - Live Stream, Offline Log Upload, Plotly, Incident Management)│
└───────────────────────┬───────────────────────┬────────────────────────┘
                        │                       │
         ┌──────────────┴──────────────┐        │
         ▼                             ▼        ▼
┌──────────────────────────┐  ┌──────────────────────────┐  ┌──────────────────────────┐
│   Data Generator Module  │  │ Anomaly Detector Module  │  │ Incident Store (SQLite)  │
│    (data_generator.py)   │  │       (detector.py)      │  │    (incident_store.py)   │
├──────────────────────────┤  ├──────────────────────────┤  ├──────────────────────────┤
│ - Baseline Generator     │  │ - IsolationForest (seed) │  │ - Persistent Incidents   │
│ - Streaming Telemetry    │  │ - Contamination tuning   │  │ - Cooldown deduplication │
│ - Anomaly Injection      │  │ - Severity & Z-scores    │  │ - Status: Open/Ack/Res   │
└──────────────────────────┘  └─────────────┬────────────┘  └──────────────────────────┘
                                            │
                                            ▼
                              ┌──────────────────────────┐
                              │  Offline Analyzer Module │
                              │   (offline_analyzer.py)  │
                              ├──────────────────────────┤
                              │ - Safe CSV Validation    │
                              │ - Flexible Col Mapping   │
                              │ - Dual Baseline Modes    │
                              │ - Isolated Offline Plots │
                              └──────────────────────────┘
```

### Module Responsibilities

1. **`data_generator.py`**:
   - **`generate_baseline_data`**: Generates healthy telemetry samples used strictly to fit the baseline Isolation Forest model.
   - **`generate_streaming_observation`**: Simulates incoming real-time telemetry events using continuous, mean-reverting AR(1) stochastic drift.
   - **`generate_anomaly_observation`**: Injects extreme, out-of-distribution values (CPU starvation, traffic surges, cascade failures).

2. **`detector.py`**:
   - Encapsulates `sklearn.ensemble.IsolationForest` with reproducible seeds and dynamic calibration controls (`update_parameters`).
   - Fits on healthy baseline data, avoiding model poisoning from live data.
   - Computes decision scores, normalized 0–100 **Risk Index**, and classifies severity (`Critical`, `High`, `Medium`, `Normal`) with explainable Z-score diagnostic triggers.

3. **`incident_store.py`**:
   - Provides local SQLite persistence for detected anomaly incidents across page reloads and restarts.
   - Features automated cooldown deduplication to prevent alert fatigue when an anomaly persists across successive observations.
   - Manages lifecycle statuses (`Open`, `Acknowledged`, `Resolved`, `Ignored`) and operator notes.

4. **`offline_analyzer.py`**:
   - Validates and parses uploaded CSV files safely (size limits, empty checks, malformed row recovery).
   - Provides flexible column mapping to supported telemetry metrics (CPU, Memory, Latency, Requests, Error Rate).
   - Executes offline detection in dual baseline modes (`unsupervised` or `operational_baseline`).
   - Isolates uploaded data from the live stream and provides one-click promotion to the SQLite incident database.

5. **`app.py`**:
   - Dual-tab Streamlit dashboard providing **Live Streaming & Telemetry** and **Offline Log Analysis & CSV Upload**.

---

## 🚀 Quickstart Guide for Windows

Follow these exact commands in Windows PowerShell to set up and launch the platform.

### Prerequisites
- Python 3.10+ installed on your system.

### Step 1: Open PowerShell and Navigate to the Project
```powershell
cd c:\Users\USER\OneDrive\Desktop\anomaly-detection-platform
```

### Step 2: Create a Virtual Environment
```powershell
python -m venv .venv
```

### Step 3: Activate the Virtual Environment
In Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If you see an execution policy error in PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` or use `.\.venv\Scripts\activate.bat` in Command Prompt).*

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 5: Launch the Dashboard
```powershell
streamlit run app.py
```

The Streamlit dashboard will automatically open in your default browser at:
👉 **`http://localhost:8501`**

---

## 🧪 How to Test and Verify

1. **Observe the Initial Dashboard**:
   - Upon launch, the platform pre-populates 25 normal historical readings so you can immediately inspect the UI.
   - Check the metric cards: **Total Observations**, **Detected Anomalies**, **Anomaly Rate**, and **Latest CPU Usage**.

2. **Start Live Monitoring**:
   - Click the green **▶ Start** button in the sidebar.
   - Notice the status badge switch to **● LIVE STREAMING ACTIVE**.
   - Watch the Plotly time series chart update automatically at the selected refresh interval (default: 2.0s).

3. **Inject a Synthetic Anomaly**:
   - In the sidebar under **Synthetic Anomaly Injection**, select a scenario:
     - *Traffic Surge / DDoS Spike*
     - *CPU Starvation (99% Lockup)*
     - *Cascade Failure (Memory + Latency)*
   - Click the **⚡ Inject Synthetic Anomaly** button.
   - **Verification Results**:
     - A red notification banner will confirm the detected anomaly.
     - The Plotly chart will plot a distinct **red diamond marker** on the spike.
     - The **Detected Anomalies** counter will increment.
     - The **Telemetry & Alerts Log** table will display a new row classified with `Critical` or `High` severity, along with the diagnostic explanation (e.g., `CPU Usage: 97.4% (+8.1σ)`).

4. **Test Filtering & CSV Export**:
   - Select **Anomalies Only** from the Anomaly Filter dropdown to view only flagged incidents.
   - Click **📥 Export All Telemetry (CSV)** or **🚨 Export Detected Anomalies Only (CSV)** to download the dataset for external audits.

5. **Stop Monitoring**:
   - Click **⏹ Stop** in the sidebar to pause the data stream.

---

## 📁 Offline Log Analysis & CSV Upload Guide

The platform provides a standalone **Offline Log Analysis** tab to analyze historical server logs and external CSV exports without corrupting live streaming data.

### Expected CSV Format & Guidelines

1. **File Format**: Standard comma-separated values (`.csv`).
2. **File Size Limit**: Up to 20 MB.
3. **Timestamp Column (Optional)**:
   - Supports standard ISO-8601 timestamps (`2026-10-24 14:30:00` or `2026-10-24T14:30:00Z`).
   - If omitted, sequential 10-second interval timestamps are generated automatically.
4. **Metric Columns**: At least one numeric metric column must be present.
5. **Supported Target Metrics**:
   - **CPU Utilization**: `cpu_usage` (0–100%)
   - **Memory Utilization**: `memory_usage` (0–100%)
   - **Response Latency**: `response_time` (ms)
   - **Request Count / Throughput**: `request_count` (req/s)
   - **Error Rate**: `error_rate` (%)

### Sample CSV Input

```csv
timestamp,server_cpu,server_ram,latency_ms,requests_per_sec,error_rate
2026-10-24 10:00:00,34.5,51.2,115.0,210.0,0.45
2026-10-24 10:00:10,36.1,51.8,118.2,215.4,0.50
2026-10-24 10:00:20,38.0,52.1,120.5,220.1,0.52
2026-10-24 10:00:30,98.2,88.4,2850.0,1850.0,12.40
2026-10-24 10:00:40,99.1,91.0,3100.0,2100.0,15.20
```

### Analysis Workflow

1. Navigate to the **📁 Offline Log Analysis & CSV Upload** tab.
2. Drag and drop your `.csv` file into the uploader.
3. Under **🗺️ Column Mapping Configuration**, map your CSV header names to the corresponding target metrics.
4. Select your **Baseline Profile**:
   - **Unsupervised**: Fits Isolation Forest directly on the uploaded data to isolate extreme outliers within the file.
   - **Platform Production Baseline**: Compares the metrics against strict healthy production baseline bounds.
5. Click **🔍 Run Offline Anomaly Analysis**.
6. Inspect summary KPIs, the interactive Plotly time series with red diamond markers, and the filtered anomaly table.
7. Download:
   - **📥 Validated Input CSV**
   - **📊 Full Analyzed CSV** (including anomaly flags, scores, and explanations)
   - **🚨 Anomalies Only CSV**
8. (Optional) Click **💾 Save Flagged Anomalies to Persistent Incident Database** to explicitly promote detected offline anomalies into the SQLite incident database (tagged as `IsolationForest-Offline`).

---

## 🧪 Automated Test Suite

Run the full automated test suite containing 21 tests across calibration, incident store, and offline analyzer:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

### Test Coverage Breakdown:
- **`test_calibration.py` (6 tests)**: Verifies Isolation Forest training, contamination sensitivity adjustments, Z-score explanation triggers, and buffer re-scoring.
- **`test_incident_store.py` (7 tests)**: Verifies SQLite schema creation, incident field persistence, status lifecycle (`Open` -> `Acknowledged` -> `Resolved` -> `Ignored`), cooldown deduplication, and persistence across server restarts.
- **`test_offline_analyzer.py` (8 tests)**: Verifies safe CSV parsing, empty/large file rejection, unparseable and duplicate timestamp healing, NaN imputation, detector integration (unsupervised & baseline), export generation, and incident store promotion.

---

## ⚠️ Machine Learning Limitations & Disclaimers

> [!IMPORTANT]
> This platform is an **educational proof-of-concept (MVP)** designed for learning and local demonstration. It is **not** a production-ready enterprise monitoring system (like Prometheus, Datadog, or Grafana).

### Critical Limitations of Isolation Forest in this Context:
1. **Unsupervised Assumptions & Contamination**:
   - The Isolation Forest assumes anomalies are few and structurally isolated. Setting the `contamination` parameter too high can lead to false positives during normal traffic peaks.
2. **Temporal & Seasonal Blindness**:
   - Isolation Forest treats each observation as an independent point in multidimensional space. It does not account for time-of-day seasonality (e.g., higher traffic at 2 PM vs 3 AM) or sequential time-series autocorrelation without manual lag feature engineering.
3. **Concept Drift & Static Baselines**:
   - The model is trained on a static baseline. In real production systems, hardware upgrades or application updates change the baseline profile, requiring managed model re-calibration.
4. **Synthetic vs. Real Noise**:
   - Real-world telemetry contains network dropouts, clock skews, and missing packets that require robust distributed stream processing (e.g., Apache Kafka, Flink).
