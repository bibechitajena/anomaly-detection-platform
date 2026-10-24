"""
Real-Time Anomaly Detection Platform
====================================
A modern, modular Streamlit dashboard for real-time telemetry monitoring
and machine-learning-based anomaly detection using scikit-learn's IsolationForest.

Author: Antigravity Team
Platform: Streamlit + scikit-learn + pandas + plotly
"""

from datetime import datetime, timedelta
import io
import time
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data_generator import (
    FEATURE_COLUMNS,
    generate_baseline_data,
    generate_streaming_observation,
    generate_anomaly_observation,
)
from detector import AnomalyDetector
from incident_store import IncidentStore
from offline_analyzer import (
    MAX_UPLOAD_SIZE_BYTES,
    SUPPORTED_METRIC_TARGETS,
    validate_and_parse_csv,
    run_offline_detection,
    save_offline_incidents_to_store,
)
from analytics import (
    get_incident_severity_summary,
    get_incident_status_summary,
    get_incident_source_summary,
    get_incident_time_distribution,
    get_anomaly_rate_trend,
    get_metric_correlation_matrix,
    get_system_health_summary,
)

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Real-Time Anomaly Detection Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a sleek, modern observability UI
st.markdown(
    """
    <style>
    /* Metric Card Enhancements */
    div[data-testid="stMetric"] {
        background-color: rgba(240, 243, 246, 0.4);
        border: 1px solid rgba(220, 226, 235, 0.8);
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: transform 0.15s ease-in-out;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
    }
    
    /* Header & Badge Styling */
    .status-badge-live {
        background: #10B981;
        color: white;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-badge-paused {
        background: #6B7280;
        color: white;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .anomaly-alert-banner {
        background-color: #FEE2E2;
        border-left: 5px solid #EF4444;
        color: #991B1B;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. Session State Initialization
# -----------------------------------------------------------------------------
def initialize_session_state():
    """Initializes persistent application state across Streamlit reruns.

    Maintains:
    - Trained Isolation Forest model instance.
    - Historical observations stream.
    - Streaming control flags (monitoring active / paused).
    """
    if "detector" not in st.session_state:
        # Initialize detector and train once on baseline data
        detector = AnomalyDetector(contamination=0.04, random_state=42)
        baseline_df = generate_baseline_data(n_samples=500, random_seed=42)
        detector.train(baseline_df)
        st.session_state.detector = detector
        st.session_state.baseline_df = baseline_df

    if "history" not in st.session_state:
        # Seed the platform with 25 realistic initial telemetry points
        # so the user sees populated charts and metrics immediately on launch
        detector: AnomalyDetector = st.session_state.detector
        initial_history: List[Dict[str, object]] = []

        now = datetime.now()
        start_time = now - timedelta(seconds=25 * 3)

        prev_obs = None
        for i in range(25):
            t = start_time + timedelta(seconds=i * 3)
            obs = generate_streaming_observation(prev_obs, current_time=t)
            analyzed = detector.predict_point(obs)
            initial_history.append(analyzed)
            prev_obs = obs

        st.session_state.history = initial_history

    if "is_monitoring" not in st.session_state:
        st.session_state.is_monitoring = False

    if "refresh_interval" not in st.session_state:
        st.session_state.refresh_interval = 2.0

    if "last_injection_message" not in st.session_state:
        st.session_state.last_injection_message = None

    if "incident_store" not in st.session_state:
        st.session_state.incident_store = IncidentStore(
            db_path="incidents.db", cooldown_seconds=60
        )

    if "offline_results" not in st.session_state:
        st.session_state.offline_results = None


initialize_session_state()


# -----------------------------------------------------------------------------
# 3. Sidebar Controls & Model Metadata
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Platform Controls")

    # Monitoring Status Indicator
    if st.session_state.is_monitoring:
        st.markdown(
            '<div class="status-badge-live">● LIVE STREAMING ACTIVE</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-badge-paused">⏸ MONITORING PAUSED</div>',
            unsafe_allow_html=True,
        )

    st.write("")

    # Start / Stop Monitoring Control
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button(
            "▶ Start",
            disabled=st.session_state.is_monitoring,
            use_container_width=True,
            help="Start real-time streaming data ingestion",
        ):
            st.session_state.is_monitoring = True
            st.rerun()

    with col_btn2:
        if st.button(
            "⏹ Stop",
            disabled=not st.session_state.is_monitoring,
            use_container_width=True,
            help="Pause streaming data ingestion",
        ):
            st.session_state.is_monitoring = False
            st.rerun()

    # Configurable Refresh Interval
    refresh_sec = st.slider(
        "⏱ Refresh Interval (seconds)",
        min_value=1.0,
        max_value=5.0,
        value=float(st.session_state.refresh_interval),
        step=0.5,
        help="How frequently new telemetry is captured when monitoring is active.",
    )
    st.session_state.refresh_interval = refresh_sec

    # Manual Single Step (Useful for testing without auto-stream)
    if st.button("➕ Generate 1 Normal Reading", use_container_width=True):
        detector: AnomalyDetector = st.session_state.detector
        prev_obs = st.session_state.history[-1] if st.session_state.history else None
        obs = generate_streaming_observation(prev_obs)
        predicted = detector.predict_point(obs)
        st.session_state.history.append(predicted)
        if predicted.get("is_anomaly"):
            inc_id, is_new = st.session_state.incident_store.record_anomaly(predicted)
            action = "New incident logged" if is_new else "Incident grouped"
            st.session_state.last_injection_message = (
                f"Generated reading flagged as anomaly! {action}: `{inc_id}`"
            )
        else:
            st.session_state.last_injection_message = (
                "Added 1 normal telemetry observation."
            )
        st.rerun()

    st.markdown("---")
    st.markdown("### 🎯 Detection Calibration")
    st.caption("Interactively tune Isolation Forest sensitivity and diagnostic thresholds.")

    current_contam_pct = float(st.session_state.detector.contamination * 100.0)
    current_z_thresh = float(st.session_state.detector.z_threshold)

    slider_contam = st.slider(
        "Sensitivity / Contamination (%)",
        min_value=1.0,
        max_value=10.0,
        value=current_contam_pct,
        step=0.5,
        help=(
            "Expected proportion of baseline outliers. Higher values make the model "
            "more aggressive and sensitive to subtle shifts. Lower values demand stronger isolation."
        ),
    )
    new_contamination = slider_contam / 100.0

    new_z_threshold = st.slider(
        "Diagnostic Trigger (σ)",
        min_value=1.5,
        max_value=4.5,
        value=current_z_thresh,
        step=0.1,
        help=(
            "Number of baseline standard deviations required for an anomalous metric "
            "to trigger explainability details and elevate severity to High/Critical."
        ),
    )

    re_evaluate_buffer = st.checkbox(
        "Re-evaluate active buffer",
        value=True,
        help="Instantly re-score previously captured telemetry with the new sensitivity settings.",
    )

    # Apply calibration updates if either parameter changed
    params_changed = (
        round(new_contamination, 4) != round(st.session_state.detector.contamination, 4)
        or round(new_z_threshold, 2) != round(st.session_state.detector.z_threshold, 2)
    )

    if params_changed:
        st.session_state.detector.update_parameters(
            contamination=new_contamination,
            z_threshold=new_z_threshold,
        )
        if re_evaluate_buffer and st.session_state.history:
            st.session_state.history = st.session_state.detector.rescore_history(
                st.session_state.history
            )
            st.session_state.last_injection_message = (
                f"🎯 Calibration updated: Contamination = {new_contamination*100:.1f}%, "
                f"Trigger = {new_z_threshold:.1f}σ. Ingested buffer re-scored."
            )
        else:
            st.session_state.last_injection_message = (
                f"🎯 Calibration updated: Contamination = {new_contamination*100:.1f}%, "
                f"Trigger = {new_z_threshold:.1f}σ. Applied to subsequent observations."
            )
        st.rerun()

    st.markdown("---")
    st.markdown("### 🧪 Synthetic Anomaly Injection")
    st.caption("Demonstrate Isolation Forest detection by injecting extreme spikes.")

    scenario_option = st.selectbox(
        "Anomaly Scenario",
        options=[
            ("traffic_surge", "Traffic Surge / DDoS Spike"),
            ("cpu_starvation", "CPU Starvation (99% Lockup)"),
            ("cascade_failure", "Cascade Failure (Memory + Latency)"),
        ],
        format_func=lambda x: x[1],
    )

    if st.button(
        "⚡ Inject Synthetic Anomaly", type="primary", use_container_width=True
    ):
        detector: AnomalyDetector = st.session_state.detector
        anomaly_obs = generate_anomaly_observation(
            anomaly_scenario=scenario_option[0]
        )
        analyzed_anomaly = detector.predict_point(anomaly_obs)
        st.session_state.history.append(analyzed_anomaly)

        # Persist incident to SQLite store with cooldown grouping
        inc_id, is_new = st.session_state.incident_store.record_anomaly(analyzed_anomaly)
        sev = analyzed_anomaly.get("severity", "High")
        action_desc = "New Incident Logged" if is_new else "Incident Grouped / Updated"

        st.session_state.last_injection_message = (
            f"🚨 Injected synthetic {scenario_option[1]}! "
            f"IsolationForest flagged it as **{sev.upper()} ANOMALY** "
            f"(Score: {analyzed_anomaly['anomaly_score']}). "
            f"[{action_desc}: `{inc_id}`]"
        )
        st.rerun()

    st.markdown("---")

    # Clear / Reset Buffer
    if st.button("🗑️ Reset Telemetry Buffer", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_injection_message = "Telemetry buffer reset."
        initialize_session_state()
        st.rerun()

    # Model Summary Expander
    with st.expander("ℹ️ ML Model Details"):
        model_info = st.session_state.detector.get_summary()
        st.markdown(f"**Algorithm:** `{model_info['model_type']}`")
        st.markdown(f"**Baseline Samples:** `{model_info['trained_samples']}`")
        st.markdown(f"**Contamination:** `{model_info['contamination'] * 100:.1f}%`")
        st.markdown(f"**Diagnostic Trigger:** `{model_info.get('z_threshold', 2.5):.1f}σ`")
        st.markdown(f"**Random Seed:** `{model_info['random_state']}`")
        st.markdown(f"**Features:** {', '.join(model_info['features'])}")
        st.markdown(
            "Baseline data was trained independently. "
            "No incoming test/streaming data is used for training."
        )


# -----------------------------------------------------------------------------
# 4. Main Dashboard Header
# -----------------------------------------------------------------------------
col_title, col_actions = st.columns([3, 1])
with col_title:
    st.title("🛡️ Real-Time Anomaly Detection Platform")
    st.markdown(
        "Live multivariate telemetry monitoring & automated outlier detection "
        "powered by scikit-learn's **Isolation Forest**."
    )

with col_actions:
    st.write("")
    if st.session_state.is_monitoring:
        st.success("🟢 Engine: Ingesting Telemetry")
    else:
        st.info("⏸ Engine: Paused")

# Temporary injection banner
if st.session_state.last_injection_message:
    if "Injected synthetic" in st.session_state.last_injection_message:
        st.markdown(
            f'<div class="anomaly-alert-banner">{st.session_state.last_injection_message}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info(st.session_state.last_injection_message)
    st.session_state.last_injection_message = None


def render_overview_tab():
    """Renders the executive observability overview, health status, and KPI summaries."""
    store: IncidentStore = st.session_state.incident_store
    history = st.session_state.history
    health = get_system_health_summary(history, store)

    # Health status banner
    status_label = health["health_status"]
    status_color = health["health_color"]
    st.markdown(
        f"""
        <div style="background-color: rgba(240, 243, 246, 0.5); border-left: 6px solid {status_color}; padding: 14px 18px; border-radius: 8px; margin-bottom: 1.2rem;">
            <div style="font-size: 1.15rem; font-weight: 700; color: {status_color};">
                ● SYSTEM HEALTH STATUS: {status_label}
            </div>
            <div style="font-size: 0.95rem; color: #4B5563; margin-top: 4px;">
                {health["health_message"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Overview KPIs
    ov1, ov2, ov3, ov4 = st.columns(4)
    with ov1:
        st.metric(
            "Engine Status",
            "ACTIVE" if st.session_state.is_monitoring else "PAUSED",
            delta=f"{st.session_state.refresh_interval}s interval" if st.session_state.is_monitoring else "Standby",
        )
    with ov2:
        open_cnt = health["open_incidents"]
        st.metric(
            "Active Incidents",
            f"{open_cnt}",
            delta=f"{open_cnt} unresolved" if open_cnt > 0 else "All Clear",
            delta_color="inverse",
        )
    with ov3:
        crit_cnt = health["critical_incidents"]
        st.metric(
            "Critical Incidents",
            f"{crit_cnt}",
            delta=f"{crit_cnt} critical" if crit_cnt > 0 else "0",
            delta_color="inverse",
        )
    with ov4:
        st.metric(
            "Live Anomaly Rate",
            f"{health['anomaly_rate']:.1f}%",
            delta=f"{health['anomalies_count']} flagged / {health['total_observations']} total",
            delta_color="inverse",
        )

    # Real-time Telemetry Snapshot
    st.markdown("#### ⚡ Real-Time Telemetry Snapshot")
    if history:
        t1, t2, t3, t4 = st.columns(4)
        latest = history[-1]
        prev = history[-2] if len(history) > 1 else latest

        c_val = float(latest["cpu_usage"])
        c_delta = round(c_val - float(prev["cpu_usage"]), 1)
        with t1:
            st.metric("CPU Utilization", f"{c_val:.1f}%", f"{c_delta:+.1f}%", delta_color="inverse")

        m_val = float(latest["memory_usage"])
        m_delta = round(m_val - float(prev["memory_usage"]), 1)
        with t2:
            st.metric("Memory Utilization", f"{m_val:.1f}%", f"{m_delta:+.1f}%", delta_color="inverse")

        r_val = float(latest["response_time"])
        r_delta = round(r_val - float(prev["response_time"]), 1)
        with t3:
            st.metric("Response Latency", f"{r_val:.0f} ms", f"{r_delta:+.0f} ms", delta_color="inverse")

        q_val = float(latest["request_count"])
        q_delta = round(q_val - float(prev["request_count"]), 1)
        with t4:
            st.metric("Throughput", f"{q_val:.0f} req/s", f"{q_delta:+.0f} req/s")
    else:
        st.info("No live telemetry ingested yet. Click Start in the sidebar.")

    # Rolling Anomaly Rate Trend Sparkline
    st.markdown("#### 📉 Recent Anomaly Rate Trend")
    trend_df = get_anomaly_rate_trend(history, window_size=10)
    if not trend_df.empty:
        fig_trend = go.Figure()
        fig_trend.add_trace(
            go.Scatter(
                x=trend_df["timestamp"],
                y=trend_df["rolling_anomaly_rate"],
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(239, 68, 68, 0.12)",
                line=dict(color="#EF4444", width=2),
                name="Rolling Anomaly Rate (%)",
            )
        )
        fig_trend.update_layout(
            template="plotly_white",
            height=220,
            xaxis_title="Time",
            yaxis_title="Anomaly Rate (%)",
            margin=dict(l=30, r=30, t=20, b=30),
        )
        st.plotly_chart(fig_trend, width="stretch")
    else:
        st.caption("Awaiting observations to compute anomaly trend.")

    # Recent Incidents Snapshot Table
    st.markdown("#### 🚨 Latest Persistent Incidents")
    df_recent = store.get_incidents(limit=5)
    if not df_recent.empty:
        display_rec = df_recent.copy()
        display_rec["Last Seen"] = pd.to_datetime(display_rec["last_seen_timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        show_cols = ["incident_id", "Last Seen", "metric", "observed_value", "severity", "status", "occurrence_count"]
        table_rec = display_rec[show_cols].rename(
            columns={
                "incident_id": "Incident ID",
                "metric": "Trigger Metric",
                "observed_value": "Peak Value",
                "severity": "Severity",
                "status": "Status",
                "occurrence_count": "Occurrences",
            }
        )
        st.dataframe(table_rec, width="stretch", hide_index=True)
    else:
        st.info("No incidents currently recorded in the SQLite database.")


def render_live_monitoring_tab():
    """Renders the live telemetry streaming dashboard, visualizations, and alerts log."""
    # -----------------------------------------------------------------------------
    # 5. Core Metric Cards
    # -----------------------------------------------------------------------------
    df_history = pd.DataFrame(st.session_state.history)

    if not df_history.empty:
        total_obs = len(df_history)
        anomalies_count = int(df_history["is_anomaly"].sum())
        anomaly_rate = (anomalies_count / total_obs) * 100.0

        latest_row = df_history.iloc[-1]
        prev_row = df_history.iloc[-2] if len(df_history) > 1 else latest_row

        cpu_val = float(latest_row["cpu_usage"])
        cpu_delta = round(cpu_val - float(prev_row["cpu_usage"]), 1)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(label="Total Observations", value=f"{total_obs:,}")
        with m2:
            st.metric(
                label="Detected Anomalies",
                value=f"{anomalies_count:,}",
                delta=f"{anomalies_count} flagged" if anomalies_count > 0 else "0",
                delta_color="inverse",
            )
        with m3:
            st.metric(
                label="Anomaly Rate",
                value=f"{anomaly_rate:.1f}%",
                delta="Normal" if anomaly_rate <= 6.0 else "Elevated",
                delta_color="inverse",
            )
        with m4:
            st.metric(
                label="Latest CPU Usage",
                value=f"{cpu_val:.1f}%",
                delta=f"{cpu_delta:+.1f}%",
                delta_color="inverse",
            )

        # Secondary metrics row for Memory, Response Time, and Requests
        sm1, sm2, sm3 = st.columns(3)
        with sm1:
            mem_val = float(latest_row["memory_usage"])
            mem_delta = round(mem_val - float(prev_row["memory_usage"]), 1)
            st.metric(
                label="Latest Memory Usage",
                value=f"{mem_val:.1f}%",
                delta=f"{mem_delta:+.1f}%",
                delta_color="inverse",
            )
        with sm2:
            resp_val = float(latest_row["response_time"])
            resp_delta = round(resp_val - float(prev_row["response_time"]), 1)
            st.metric(
                label="Latest Response Time",
                value=f"{resp_val:.0f} ms",
                delta=f"{resp_delta:+.0f} ms",
                delta_color="inverse",
            )
        with sm3:
            req_val = float(latest_row["request_count"])
            req_delta = round(req_val - float(prev_row["request_count"]), 1)
            st.metric(
                label="Latest Request Count",
                value=f"{req_val:.0f} req/s",
                delta=f"{req_delta:+.0f} req/s",
            )
    else:
        st.warning("No observations collected yet. Use the sidebar to start streaming.")

    # -----------------------------------------------------------------------------
    # 6. Interactive Visualizations with Plotly
    # -----------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📈 Telemetry Time Series & Anomaly Highlights")

    if not df_history.empty:
        chart_col1, chart_col2 = st.columns([2, 1])

        with chart_col1:
            selected_metric = st.selectbox(
                "Primary Visualized Metric",
                options=[
                    ("cpu_usage", "CPU Usage (%)"),
                    ("memory_usage", "Memory Usage (%)"),
                    ("response_time", "Response Time (ms)"),
                    ("request_count", "Request Count (req/sec)"),
                ],
                format_func=lambda x: x[1],
                index=0,
            )

        with chart_col2:
            chart_view_mode = st.radio(
                "Display Mode",
                options=["Focus View", "Multi-Metric Grid"],
                horizontal=True,
            )

        metric_col_name = selected_metric[0]
        metric_label = selected_metric[1]

        # Baseline statistics for visual reference
        baseline_stats = st.session_state.detector.baseline_stats.get(
            metric_col_name, {"mean": 40.0, "p95": 60.0}
        )

        if chart_view_mode == "Focus View":
            df_normal = df_history[~df_history["is_anomaly"]]
            df_anomalies = df_history[df_history["is_anomaly"]]

            fig = go.Figure()

            if "p95" in baseline_stats:
                fig.add_hrect(
                    y0=0,
                    y1=baseline_stats["p95"],
                    fillcolor="#10B981",
                    opacity=0.08,
                    layer="below",
                    line_width=0,
                    annotation_text="Normal Baseline Zone",
                    annotation_position="top left",
                )

            fig.add_hline(
                y=baseline_stats["mean"],
                line_dash="dot",
                line_color="rgba(100, 116, 139, 0.6)",
                annotation_text=f"Baseline Mean ({baseline_stats['mean']:.1f})",
                annotation_position="bottom right",
            )

            fig.add_trace(
                go.Scatter(
                    x=df_history["timestamp"],
                    y=df_history[metric_col_name],
                    mode="lines+markers",
                    name=f"{metric_label} (Normal)",
                    line=dict(color="#3B82F6", width=2),
                    marker=dict(size=5, color="#1D4ED8"),
                    hovertemplate="<b>%{x|%H:%M:%S}</b><br>Value: %{y:.2f}<extra></extra>",
                )
            )

            if not df_anomalies.empty:
                hover_text = [
                    f"<b>ANOMALY DETECTED</b><br>"
                    f"Time: {t.strftime('%H:%M:%S')}<br>"
                    f"{metric_label}: {v:.2f}<br>"
                    f"Severity: <b>{sev}</b><br>"
                    f"Score: {score:.4f}<br>"
                    f"Cause: {exp}"
                    for t, v, sev, score, exp in zip(
                        df_anomalies["timestamp"],
                        df_anomalies[metric_col_name],
                        df_anomalies["severity"],
                        df_anomalies["anomaly_score"],
                        df_anomalies["explanation"],
                    )
                ]

                fig.add_trace(
                    go.Scatter(
                        x=df_anomalies["timestamp"],
                        y=df_anomalies[metric_col_name],
                        mode="markers",
                        name="Detected Anomaly",
                        marker=dict(
                            symbol="diamond",
                            size=14,
                            color="#EF4444",
                            line=dict(color="#7F1D1D", width=2),
                        ),
                        text=hover_text,
                        hoverinfo="text",
                    )
                )

            fig.update_layout(
                title=f"<b>{metric_label}</b> over Time with Isolation Forest Anomaly Detection",
                xaxis_title="Time",
                yaxis_title=metric_label,
                template="plotly_white",
                hovermode="closest",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                margin=dict(l=40, r=40, t=60, b=40),
                height=420,
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            fig_multi = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "CPU Usage (%)",
                    "Memory Usage (%)",
                    "Response Time (ms)",
                    "Request Count (req/s)",
                ),
            )

            metrics_map = [
                ("cpu_usage", 1, 1, "#3B82F6"),
                ("memory_usage", 1, 2, "#8B5CF6"),
                ("response_time", 2, 1, "#F59E0B"),
                ("request_count", 2, 2, "#06B6D4"),
            ]

            df_anomalies = df_history[df_history["is_anomaly"]]

            for col_name, r, c, color in metrics_map:
                fig_multi.add_trace(
                    go.Scatter(
                        x=df_history["timestamp"],
                        y=df_history[col_name],
                        mode="lines",
                        line=dict(color=color, width=2),
                        name=col_name.replace("_", " ").title(),
                        showlegend=False,
                    ),
                    row=r,
                    col=c,
                )
                if not df_anomalies.empty:
                    fig_multi.add_trace(
                        go.Scatter(
                            x=df_anomalies["timestamp"],
                            y=df_anomalies[col_name],
                            mode="markers",
                            marker=dict(
                                symbol="diamond",
                                size=10,
                                color="#EF4444",
                                line=dict(color="#991B1B", width=1.5),
                            ),
                            name="Anomaly",
                            showlegend=False,
                        ),
                        row=r,
                        col=c,
                    )

            fig_multi.update_layout(
                template="plotly_white",
                height=520,
                margin=dict(l=30, r=30, t=50, b=30),
            )
            st.plotly_chart(fig_multi, use_container_width=True)

    # -----------------------------------------------------------------------------
    # 7. Telemetry & Alerts Log Table with Filters
    # -----------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📋 Telemetry & Alerts Log")

    if not df_history.empty:
        f_col1, f_col2, f_col3 = st.columns(3)

        with f_col1:
            status_filter = st.selectbox(
                "Anomaly Filter",
                options=["All Observations", "Anomalies Only", "Normal Only"],
                index=0,
            )

        with f_col2:
            severity_filter = st.selectbox(
                "Severity Filter",
                options=["All Severities", "Critical", "High", "Medium", "Normal"],
                index=0,
            )

        with f_col3:
            time_range_filter = st.selectbox(
                "Time Range Filter",
                options=[
                    ("all", "All Available Records"),
                    ("5m", "Last 5 Minutes"),
                    ("15m", "Last 15 Minutes"),
                    ("30m", "Last 30 Minutes"),
                ],
                format_func=lambda x: x[1],
                index=0,
            )

        filtered_df = df_history.copy()

        if time_range_filter[0] != "all":
            cutoff_minutes = int(time_range_filter[0].replace("m", ""))
            time_cutoff = datetime.now() - timedelta(minutes=cutoff_minutes)
            filtered_df = filtered_df[filtered_df["timestamp"] >= time_cutoff]

        if status_filter == "Anomalies Only":
            filtered_df = filtered_df[filtered_df["is_anomaly"]]
        elif status_filter == "Normal Only":
            filtered_df = filtered_df[~filtered_df["is_anomaly"]]

        if severity_filter != "All Severities":
            filtered_df = filtered_df[filtered_df["severity"] == severity_filter]

        if filtered_df.empty:
            st.info(
                "🔍 No observations match your selected filter criteria. "
                "Try adjusting the filters above."
            )
        else:
            display_df = filtered_df.sort_values(
                by="timestamp", ascending=False
            ).copy()

            display_df["Time"] = display_df["timestamp"].dt.strftime("%H:%M:%S")

            columns_to_show = [
                "Time",
                "anomaly_status",
                "severity",
                "cpu_usage",
                "memory_usage",
                "response_time",
                "request_count",
                "anomaly_risk_index",
                "explanation",
            ]

            table_view = display_df[columns_to_show].rename(
                columns={
                    "anomaly_status": "Status",
                    "severity": "Severity",
                    "cpu_usage": "CPU (%)",
                    "memory_usage": "Memory (%)",
                    "response_time": "Latency (ms)",
                    "request_count": "Requests/s",
                    "anomaly_risk_index": "Risk Index (0-100)",
                    "explanation": "Diagnostic Trigger",
                }
            )

            st.dataframe(
                table_view,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Status": st.column_config.TextColumn(
                        "Status",
                        help="Anomaly detection status (Anomaly vs Normal)",
                    ),
                    "Severity": st.column_config.TextColumn(
                        "Severity",
                        help="Calculated severity based on isolation score and metric deviation",
                    ),
                    "Risk Index (0-100)": st.column_config.ProgressColumn(
                        "Risk Index",
                        format="%.0f",
                        min_value=0,
                        max_value=100,
                    ),
                },
            )

            export_col1, export_col2 = st.columns(2)
            with export_col1:
                csv_all = df_history.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Export All Telemetry (CSV)",
                    data=csv_all,
                    file_name=f"telemetry_stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with export_col2:
                anomalies_only_df = df_history[df_history["is_anomaly"]]
                if not anomalies_only_df.empty:
                    csv_anomalies = anomalies_only_df.to_csv(index=False).encode(
                        "utf-8"
                    )
                    st.download_button(
                        label="🚨 Export Detected Anomalies Only (CSV)",
                        data=csv_anomalies,
                        file_name=f"anomalies_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                else:
                    st.button(
                        "🚨 No Anomalies to Export",
                        disabled=True,
                        use_container_width=True,
                    )

    # -----------------------------------------------------------------------------
    # 8. Persistent Incident Management (SQLite)
    # -----------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🚨 Persistent Incident Management (SQLite)")
    st.caption(
        "Detected anomalies are automatically persisted in a local SQLite database with "
        "cooldown grouping and deduplication. Records persist across page reloads and restarts."
    )

    store: IncidentStore = st.session_state.incident_store
    stats = store.get_incident_stats()

    ikpi1, ikpi2, ikpi3, ikpi4 = st.columns(4)
    with ikpi1:
        st.metric("Total Incidents", f"{stats['total']}")
    with ikpi2:
        st.metric(
            "Open Incidents",
            f"{stats['open']}",
            delta=f"{stats['open']} active" if stats["open"] > 0 else "Clear",
            delta_color="inverse",
        )
    with ikpi3:
        st.metric("Acknowledged", f"{stats['acknowledged']}")
    with ikpi4:
        resolved_or_ignored = stats["resolved"] + stats["ignored"]
        st.metric("Resolved / Ignored", f"{resolved_or_ignored}")

    if stats["total"] > 0:
        icol1, icol2 = st.columns(2)
        with icol1:
            inc_status_filter = st.selectbox(
                "Filter Incidents by Status",
                options=["All", "Open", "Acknowledged", "Resolved", "Ignored"],
                index=0,
            )
        with icol2:
            inc_sev_filter = st.selectbox(
                "Filter Incidents by Severity",
                options=["All", "Critical", "High", "Medium"],
                index=0,
            )

        df_incidents = store.get_incidents(
            status=inc_status_filter, severity=inc_sev_filter, limit=100
        )

        if not df_incidents.empty:
            display_inc = df_incidents.copy()
            display_inc["First Seen"] = pd.to_datetime(
                display_inc["timestamp"]
            ).dt.strftime("%Y-%m-%d %H:%M:%S")
            display_inc["Last Seen"] = pd.to_datetime(
                display_inc["last_seen_timestamp"]
            ).dt.strftime("%Y-%m-%d %H:%M:%S")

            inc_columns = [
                "incident_id",
                "First Seen",
                "Last Seen",
                "metric",
                "observed_value",
                "severity",
                "status",
                "occurrence_count",
                "notes",
            ]
            table_inc = display_inc[inc_columns].rename(
                columns={
                    "incident_id": "Incident ID",
                    "metric": "Trigger Metric",
                    "observed_value": "Peak Value",
                    "severity": "Severity",
                    "status": "Status",
                    "occurrence_count": "Occurrences",
                    "notes": "Operator Notes",
                }
            )

            st.dataframe(
                table_inc,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### 🛠️ Manage Incident Lifecycle")
            act_col1, act_col2, act_col3 = st.columns([2, 1, 2])

            incident_ids = display_inc["incident_id"].tolist()
            with act_col1:
                target_inc_id = st.selectbox(
                    "Select Incident ID to Update",
                    options=incident_ids,
                    key="target_inc_id",
                )
            with act_col2:
                new_status = st.selectbox(
                    "New Status",
                    options=["Acknowledged", "Resolved", "Ignored", "Open"],
                    key="new_status_select",
                )
            with act_col3:
                notes_input = st.text_input(
                    "Operator Notes (Optional)",
                    placeholder="e.g., Scaling pod replicas / False positive test",
                    key="notes_input",
                )

            if st.button(
                "💾 Apply Incident Status Update",
                type="secondary",
                use_container_width=True,
            ):
                success = store.update_incident_status(
                    target_inc_id,
                    new_status,
                    notes=notes_input if notes_input else None,
                )
                if success:
                    st.success(
                        f"Updated incident `{target_inc_id}` to status: **{new_status}**"
                    )
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Failed to update incident status in SQLite.")

            csv_inc = df_incidents.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Persistent Incidents (CSV)",
                data=csv_inc,
                file_name=f"incidents_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("No incidents match the selected status / severity filter.")
    else:
        st.info(
            "ℹ️ No incidents recorded in the persistent SQLite database yet. "
            "Use '⚡ Inject Synthetic Anomaly' or activate streaming to record incidents."
        )


def render_offline_analysis_tab():
    """Renders the offline CSV upload, column mapping, and outlier analysis workflow."""
    st.markdown("### 📁 Custom Telemetry CSV Upload & Offline Outlier Analysis")
    st.caption(
        "Upload historical telemetry logs (CSV) for standalone anomaly detection. "
        "Uploaded data is evaluated in strict isolation and will not pollute live streaming "
        "or overwrite your active incident database unless explicitly requested."
    )

    uploaded_file = st.file_uploader(
        "Upload Telemetry CSV",
        type=["csv"],
        help="Max file size 20 MB. Must contain at least one numeric telemetry column.",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_size_kb = len(file_bytes) / 1024.0

        if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
            st.error(
                f"File size exceeds the 20 MB limit ({file_size_kb / 1024.0:.1f} MB uploaded)."
            )
        else:
            try:
                peek_df = pd.read_csv(io.BytesIO(file_bytes), nrows=5)
                csv_cols = [str(c).strip() for c in peek_df.columns]
            except Exception as e:
                st.error(f"Failed to read CSV header: {e}")
                csv_cols = []

            if csv_cols:
                st.info(
                    f"📄 Loaded file: **{uploaded_file.name}** "
                    f"({file_size_kb:.1f} KB, {len(csv_cols)} columns detected)"
                )

                # Column Mapping Section
                with st.expander("🗺️ Column Mapping Configuration", expanded=True):
                    st.markdown(
                        "Map your CSV columns to standard telemetry metrics. "
                        "Unmapped metrics will be excluded from analysis."
                    )

                    col_options = ["(Skip / Not Present)"] + csv_cols
                    map_col1, map_col2 = st.columns(2)
                    mapping: Dict[str, str] = {}

                    with map_col1:
                        # Timestamp mapping
                        default_ts_idx = 0
                        for idx, c in enumerate(col_options):
                            if c.lower() in [
                                "timestamp",
                                "time",
                                "datetime",
                                "date",
                                "ts",
                                "created_at",
                            ]:
                                default_ts_idx = idx
                                break

                        selected_ts = st.selectbox(
                            "Timestamp Column",
                            options=col_options,
                            index=default_ts_idx,
                            help="If skipped, sequential 10-second timestamps will be automatically generated.",
                        )

                        # CPU mapping
                        default_cpu_idx = 0
                        for idx, c in enumerate(col_options):
                            if "cpu" in c.lower():
                                default_cpu_idx = idx
                                break
                        sel_cpu = st.selectbox(
                            "CPU Utilization (%)",
                            options=col_options,
                            index=default_cpu_idx,
                        )
                        if sel_cpu != "(Skip / Not Present)":
                            mapping[sel_cpu] = "cpu_usage"

                        # Memory mapping
                        default_mem_idx = 0
                        for idx, c in enumerate(col_options):
                            if "mem" in c.lower() or "ram" in c.lower():
                                default_mem_idx = idx
                                break
                        sel_mem = st.selectbox(
                            "Memory Utilization (%)",
                            options=col_options,
                            index=default_mem_idx,
                        )
                        if sel_mem != "(Skip / Not Present)":
                            mapping[sel_mem] = "memory_usage"

                    with map_col2:
                        # Latency mapping
                        default_resp_idx = 0
                        for idx, c in enumerate(col_options):
                            if (
                                "lat" in c.lower()
                                or "resp" in c.lower()
                                or "duration" in c.lower()
                            ):
                                default_resp_idx = idx
                                break
                        sel_resp = st.selectbox(
                            "Response Latency (ms)",
                            options=col_options,
                            index=default_resp_idx,
                        )
                        if sel_resp != "(Skip / Not Present)":
                            mapping[sel_resp] = "response_time"

                        # Request count mapping
                        default_req_idx = 0
                        for idx, c in enumerate(col_options):
                            if (
                                "req" in c.lower()
                                or "rpm" in c.lower()
                                or "rps" in c.lower()
                                or "count" in c.lower()
                            ):
                                default_req_idx = idx
                                break
                        sel_req = st.selectbox(
                            "Request Count / Throughput (req/s)",
                            options=col_options,
                            index=default_req_idx,
                        )
                        if sel_req != "(Skip / Not Present)":
                            mapping[sel_req] = "request_count"

                        # Error rate mapping
                        default_err_idx = 0
                        for idx, c in enumerate(col_options):
                            if "err" in c.lower() or "fail" in c.lower():
                                default_err_idx = idx
                                break
                        sel_err = st.selectbox(
                            "Error Rate (%)",
                            options=col_options,
                            index=default_err_idx,
                        )
                        if sel_err != "(Skip / Not Present)":
                            mapping[sel_err] = "error_rate"

                # Detection Settings Section
                with st.expander("⚙️ Offline Detection Settings", expanded=False):
                    cfg_col1, cfg_col2 = st.columns(2)
                    with cfg_col1:
                        baseline_mode = st.radio(
                            "Baseline Profile",
                            options=[
                                (
                                    "unsupervised",
                                    "Unsupervised Outlier Isolation (Fit on Uploaded Data)",
                                ),
                                (
                                    "operational_baseline",
                                    "Platform Production Baseline (Compare to Production SLA)",
                                ),
                            ],
                            format_func=lambda x: x[1],
                            index=0,
                            help="Unsupervised identifies relative outliers within the file. Production Baseline flags deviations from standard healthy server baselines.",
                        )[0]

                        offline_contam = (
                            st.slider(
                                "Offline Contamination / Sensitivity (%)",
                                min_value=1.0,
                                max_value=15.0,
                                value=5.0,
                                step=0.5,
                                help="Expected proportion of anomalies. Higher = more sensitive.",
                            )
                            / 100.0
                        )

                    with cfg_col2:
                        offline_z = st.slider(
                            "Offline Diagnostic Trigger (σ)",
                            min_value=1.5,
                            max_value=4.5,
                            value=2.5,
                            step=0.1,
                            help="Z-score deviation required to trigger explanation hints and elevated severity.",
                        )

                # Run Analysis Button
                if st.button(
                    "🔍 Run Offline Anomaly Analysis",
                    type="primary",
                    use_container_width=True,
                ):
                    ts_param = (
                        selected_ts
                        if selected_ts != "(Skip / Not Present)"
                        else None
                    )
                    with st.spinner(
                        "Validating telemetry data and running Isolation Forest..."
                    ):
                        cleaned_df, warnings, errors = validate_and_parse_csv(
                            file_bytes,
                            column_mapping=mapping,
                            timestamp_col=ts_param,
                        )

                        if errors:
                            for err in errors:
                                st.error(f"❌ {err}")
                        else:
                            mapped_features = [
                                m
                                for m in mapping.values()
                                if m in cleaned_df.columns
                            ]
                            if not mapped_features:
                                mapped_features = [
                                    c
                                    for c in cleaned_df.columns
                                    if c != "timestamp"
                                ]

                            analyzed_df = run_offline_detection(
                                cleaned_df=cleaned_df,
                                features=mapped_features,
                                contamination=offline_contam,
                                z_threshold=offline_z,
                                baseline_mode=baseline_mode,
                            )

                            st.session_state.offline_results = {
                                "file_name": uploaded_file.name,
                                "analyzed_df": analyzed_df,
                                "warnings": warnings,
                                "mapped_features": mapped_features,
                                "baseline_mode": baseline_mode,
                                "contamination": offline_contam,
                                "z_threshold": offline_z,
                                "timestamp": datetime.now(),
                            }
                            st.success(
                                f"Analysis complete! Evaluated {len(analyzed_df)} records across {len(mapped_features)} metrics."
                            )
                            st.rerun()

    # Display Analysis Results if available
    if st.session_state.offline_results is not None:
        res = st.session_state.offline_results
        df_offline: pd.DataFrame = res["analyzed_df"]
        warnings: List[str] = res["warnings"]
        mapped_feats: List[str] = res["mapped_features"]

        st.markdown("---")
        st.markdown(f"### 📊 Offline Analysis Results: `{res['file_name']}`")
        st.caption(
            f"Evaluated with **Isolation Forest** ({res['baseline_mode']} mode, "
            f"contamination: {res['contamination']*100:.1f}%, trigger: {res['z_threshold']:.1f}σ) "
            f"at {res['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
        )

        if warnings:
            with st.expander(
                "⚠️ Data Cleansing & Validation Notices", expanded=False
            ):
                for w in warnings:
                    st.write(f"- {w}")

        total_rows = len(df_offline)
        total_anom = int(df_offline["is_anomaly"].sum())
        anom_rate = (
            (total_anom / total_rows * 100.0) if total_rows > 0 else 0.0
        )

        okpi1, okpi2, okpi3, okpi4 = st.columns(4)
        with okpi1:
            st.metric("Total Records", f"{total_rows:,}")
        with okpi2:
            st.metric("Analyzed Metrics", f"{len(mapped_feats)}")
        with okpi3:
            st.metric(
                "Flagged Anomalies",
                f"{total_anom:,}",
                delta=f"{total_anom} detected" if total_anom > 0 else "0",
                delta_color="inverse",
            )
        with okpi4:
            st.metric("Anomaly Rate", f"{anom_rate:.1f}%")

        # Interactive Plotly Chart
        st.markdown("#### 📈 Telemetry Time Series with Detected Outliers")
        if mapped_feats:
            sel_plot_metric = st.selectbox(
                "Metric to Graph",
                options=mapped_feats,
                format_func=lambda x: SUPPORTED_METRIC_TARGETS.get(
                    x, x.replace("_", " ").title()
                ),
                key="offline_plot_metric",
            )

            fig_offline = go.Figure()

            # Normal telemetry line
            fig_offline.add_trace(
                go.Scatter(
                    x=df_offline["timestamp"],
                    y=df_offline[sel_plot_metric],
                    mode="lines",
                    name=SUPPORTED_METRIC_TARGETS.get(
                        sel_plot_metric, sel_plot_metric
                    ),
                    line=dict(color="#3B82F6", width=2),
                )
            )

            # Anomaly markers
            df_anom_subset = df_offline[df_offline["is_anomaly"]]
            if not df_anom_subset.empty:
                hover_text = [
                    f"<b>OFFLINE ANOMALY</b><br>"
                    f"Time: {t}<br>"
                    f"{sel_plot_metric}: {v:.2f}<br>"
                    f"Severity: <b>{sev}</b><br>"
                    f"Score: {score:.4f}<br>"
                    f"Trigger: {exp}"
                    for t, v, sev, score, exp in zip(
                        df_anom_subset["timestamp"],
                        df_anom_subset[sel_plot_metric],
                        df_anom_subset["severity"],
                        df_anom_subset["anomaly_score"],
                        df_anom_subset["explanation"],
                    )
                ]
                fig_offline.add_trace(
                    go.Scatter(
                        x=df_anom_subset["timestamp"],
                        y=df_anom_subset[sel_plot_metric],
                        mode="markers",
                        name="Detected Anomaly",
                        marker=dict(
                            symbol="diamond",
                            size=12,
                            color="#EF4444",
                            line=dict(color="#7F1D1D", width=1.5),
                        ),
                        text=hover_text,
                        hoverinfo="text",
                    )
                )

            fig_offline.update_layout(
                template="plotly_white",
                height=420,
                xaxis_title="Timestamp",
                yaxis_title=SUPPORTED_METRIC_TARGETS.get(
                    sel_plot_metric, sel_plot_metric
                ),
                hovermode="closest",
                margin=dict(l=40, r=40, t=40, b=40),
            )
            st.plotly_chart(fig_offline, use_container_width=True)

        # Filters & Detailed Table
        st.markdown("#### 📋 Analyzed Telemetry & Anomaly Log")
        of_col1, of_col2 = st.columns(2)
        with of_col1:
            offline_status_filter = st.selectbox(
                "Filter Rows",
                options=["All Observations", "Anomalies Only", "Normal Only"],
                key="offline_status_filter",
            )
        with of_col2:
            offline_sev_filter = st.selectbox(
                "Severity Filter",
                options=[
                    "All Severities",
                    "Critical",
                    "High",
                    "Medium",
                    "Normal",
                ],
                key="offline_sev_filter",
            )

        filtered_offline = df_offline.copy()
        if offline_status_filter == "Anomalies Only":
            filtered_offline = filtered_offline[filtered_offline["is_anomaly"]]
        elif offline_status_filter == "Normal Only":
            filtered_offline = filtered_offline[
                ~filtered_offline["is_anomaly"]
            ]

        if offline_sev_filter != "All Severities":
            filtered_offline = filtered_offline[
                filtered_offline["severity"] == offline_sev_filter
            ]

        cols_to_display = (
            ["timestamp", "anomaly_status", "severity", "anomaly_risk_index"]
            + mapped_feats
            + ["explanation"]
        )
        valid_cols_to_display = [
            c for c in cols_to_display if c in filtered_offline.columns
        ]

        st.dataframe(
            filtered_offline[valid_cols_to_display].sort_values(
                by="timestamp", ascending=False
            ),
            use_container_width=True,
            hide_index=True,
        )

        # Export Controls
        st.markdown("#### 📥 Export Offline Results")
        exp1, exp2, exp3 = st.columns(3)
        with exp1:
            csv_validated = (
                df_offline[["timestamp"] + mapped_feats]
                .to_csv(index=False)
                .encode("utf-8")
            )
            st.download_button(
                "📥 Validated Input CSV",
                data=csv_validated,
                file_name=f"validated_{res['file_name']}",
                mime="text/csv",
                use_container_width=True,
            )
        with exp2:
            csv_analyzed = df_offline.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📊 Full Analyzed CSV",
                data=csv_analyzed,
                file_name=f"analyzed_{res['file_name']}",
                mime="text/csv",
                use_container_width=True,
            )
        with exp3:
            df_anom_only = df_offline[df_offline["is_anomaly"]]
            if not df_anom_only.empty:
                csv_anom_only = df_anom_only.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "🚨 Anomalies Only CSV",
                    data=csv_anom_only,
                    file_name=f"anomalies_only_{res['file_name']}",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.button(
                    "🚨 No Anomalies to Export",
                    disabled=True,
                    use_container_width=True,
                )

        # Save to SQLite Incident Store (Explicit Operator Action)
        st.markdown("---")
        st.markdown("#### 🚨 Promote Detections to Incident Database")
        st.caption(
            "Explicitly archive offline detected anomalies into the persistent SQLite incident database. "
            "Records will be tagged with detector `IsolationForest-Offline`."
        )

        if st.button(
            "💾 Save Flagged Anomalies to Persistent Incident Database",
            type="secondary",
            use_container_width=True,
        ):
            saved_cnt = save_offline_incidents_to_store(
                df_offline, st.session_state.incident_store
            )
            if saved_cnt > 0:
                st.success(
                    f"Successfully recorded/grouped {saved_cnt} incidents into the persistent database!"
                )
                time.sleep(0.5)
                st.rerun()
            else:
                st.info("No anomalous records found to save.")


# -----------------------------------------------------------------------------
# 5. Tab Layout: Live Stream vs. Offline Log Analysis
# -----------------------------------------------------------------------------
tab_live, tab_offline = st.tabs([
    "📡 Live Streaming & Telemetry",
    "📁 Offline Log Analysis & CSV Upload",
])

with tab_live:
    render_live_monitoring_tab()

with tab_offline:
    render_offline_analysis_tab()


# -----------------------------------------------------------------------------
# 9. Live Streaming Auto-Refresh Mechanism
# -----------------------------------------------------------------------------
if st.session_state.is_monitoring:
    # Pause for the specified refresh interval
    time.sleep(float(st.session_state.refresh_interval))

    # Generate the next continuous telemetry event
    detector: AnomalyDetector = st.session_state.detector
    prev_obs = st.session_state.history[-1] if st.session_state.history else None
    next_observation = generate_streaming_observation(prev_obs)

    # Classify observation with Isolation Forest
    analyzed_observation = detector.predict_point(next_observation)

    # Persist incident to SQLite if flagged as anomaly
    if analyzed_observation.get("is_anomaly"):
        st.session_state.incident_store.record_anomaly(analyzed_observation)

    # Append to memory buffer (keep up to 300 points to prevent memory overflow)
    st.session_state.history.append(analyzed_observation)
    if len(st.session_state.history) > 300:
        st.session_state.history = st.session_state.history[-300:]

    # Rerun the Streamlit app to update visualizations and metrics in real-time
    st.rerun()
