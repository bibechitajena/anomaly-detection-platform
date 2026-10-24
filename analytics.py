"""
Analytics Module for Observability and Anomaly Detection Platform.
==================================================================
Provides statistical aggregations, time-series binning, incident distribution
metrics, and system health status calculations.

All calculations use actual stored and observed telemetry; no synthetic or
fabricated metrics are invented.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from incident_store import IncidentStore


def get_incident_severity_summary(incident_store: IncidentStore) -> pd.DataFrame:
    """Calculates distribution of incidents by severity level."""
    df = incident_store.get_incidents(limit=1000)
    if df.empty:
        return pd.DataFrame({"Severity": ["Critical", "High", "Medium"], "Count": [0, 0, 0]})

    counts = df["severity"].value_counts()
    severities = ["Critical", "High", "Medium"]
    res = pd.DataFrame({
        "Severity": severities,
        "Count": [int(counts.get(s, 0)) for s in severities],
    })
    return res


def get_incident_status_summary(incident_store: IncidentStore) -> pd.DataFrame:
    """Calculates distribution of incidents by lifecycle status."""
    df = incident_store.get_incidents(limit=1000)
    if df.empty:
        return pd.DataFrame({"Status": ["Open", "Acknowledged", "Resolved", "Ignored"], "Count": [0, 0, 0, 0]})

    counts = df["status"].value_counts()
    statuses = ["Open", "Acknowledged", "Resolved", "Ignored"]
    res = pd.DataFrame({
        "Status": statuses,
        "Count": [int(counts.get(s, 0)) for s in statuses],
    })
    return res


def get_incident_source_summary(incident_store: IncidentStore) -> pd.DataFrame:
    """Calculates distribution of incidents by origin (Live Streaming vs Offline Upload)."""
    df = incident_store.get_incidents(limit=1000)
    if df.empty:
        return pd.DataFrame({"Source": ["Live Streaming", "Offline Upload"], "Count": [0, 0]})

    live_count = int((df["detector_name"] == "IsolationForest").sum())
    offline_count = int((df["detector_name"] == "IsolationForest-Offline").sum())
    other_count = len(df) - (live_count + offline_count)

    sources = ["Live Streaming", "Offline Upload"]
    counts = [live_count, offline_count]
    if other_count > 0:
        sources.append("Other")
        counts.append(other_count)

    return pd.DataFrame({"Source": sources, "Count": counts})


def get_incident_time_distribution(
    incident_store: IncidentStore,
    bin_size_minutes: int = 15,
) -> pd.DataFrame:
    """Bins incident counts by time intervals.

    Parameters:
        incident_store (IncidentStore): Database store instance.
        bin_size_minutes (int): Width of each time bucket in minutes.

    Returns:
        pd.DataFrame: Binned timeline with 'Time Bucket' and 'Incident Count'.
    """
    df = incident_store.get_incidents(limit=1000)
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame(columns=["Time Bucket", "Incident Count"])

    timestamps = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
    if timestamps.empty:
        return pd.DataFrame(columns=["Time Bucket", "Incident Count"])

    # Floor timestamps to interval
    freq = f"{bin_size_minutes}min"
    binned = timestamps.dt.floor(freq).value_counts().sort_index()

    timeline_df = pd.DataFrame({
        "Time Bucket": binned.index.strftime("%Y-%m-%d %H:%M"),
        "Incident Count": binned.values,
    })
    return timeline_df


def get_anomaly_rate_trend(
    history: List[Dict[str, Any]],
    window_size: int = 10,
) -> pd.DataFrame:
    """Calculates rolling anomaly rate over a telemetry sequence."""
    if not history:
        return pd.DataFrame(columns=["timestamp", "rolling_anomaly_rate"])

    df = pd.DataFrame(history)
    if "is_anomaly" not in df.columns or "timestamp" not in df.columns:
        return pd.DataFrame(columns=["timestamp", "rolling_anomaly_rate"])

    df = df.sort_values(by="timestamp").reset_index(drop=True)
    anom_numeric = df["is_anomaly"].astype(float)
    rolling_rate = anom_numeric.rolling(window=window_size, min_periods=1).mean() * 100.0

    return pd.DataFrame({
        "timestamp": df["timestamp"],
        "rolling_anomaly_rate": np.round(rolling_rate, 1),
    })


def get_metric_correlation_matrix(history: List[Dict[str, Any]]) -> pd.DataFrame:
    """Computes correlation coefficients between numeric telemetry features."""
    if not history:
        return pd.DataFrame()

    df = pd.DataFrame(history)
    feature_candidates = ["cpu_usage", "memory_usage", "response_time", "request_count", "error_rate"]
    active_features = [f for f in feature_candidates if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]

    if len(active_features) < 2 or len(df) < 3:
        return pd.DataFrame()

    corr = df[active_features].corr().fillna(0.0)
    return np.round(corr, 2)


def get_system_health_summary(
    history: List[Dict[str, Any]],
    incident_store: IncidentStore,
) -> Dict[str, Any]:
    """Calculates system health status, active incidents, and recent trend."""
    stats = incident_store.get_incident_stats()
    open_incidents = stats.get("open", 0)
    critical_incidents = stats.get("critical", 0)
    high_incidents = stats.get("high", 0)

    total_obs = len(history)
    anomalies_count = 0
    anomaly_rate = 0.0

    latest_cpu = 0.0
    latest_memory = 0.0
    latest_response_time = 0.0
    latest_request_count = 0.0

    if history:
        df = pd.DataFrame(history)
        if "is_anomaly" in df.columns:
            anomalies_count = int(df["is_anomaly"].sum())
            anomaly_rate = round((anomalies_count / total_obs) * 100.0, 1)

        latest = history[-1]
        latest_cpu = float(latest.get("cpu_usage", 0.0))
        latest_memory = float(latest.get("memory_usage", 0.0))
        latest_response_time = float(latest.get("response_time", 0.0))
        latest_request_count = float(latest.get("request_count", 0.0))

    # Determine health classification
    if critical_incidents > 0 or latest_cpu >= 90.0 or anomaly_rate > 15.0:
        health_status = "CRITICAL"
        health_color = "#EF4444"
        health_message = "Critical degradation detected. Urgent response recommended."
    elif high_incidents > 0 or latest_cpu >= 75.0 or anomaly_rate > 6.0:
        health_status = "WARNING"
        health_color = "#F59E0B"
        health_message = "Elevated metric deviations observed. Investigation advised."
    else:
        health_status = "HEALTHY"
        health_color = "#10B981"
        health_message = "All telemetry metrics operating within baseline bounds."

    return {
        "health_status": health_status,
        "health_color": health_color,
        "health_message": health_message,
        "total_observations": total_obs,
        "anomalies_count": anomalies_count,
        "anomaly_rate": anomaly_rate,
        "open_incidents": open_incidents,
        "critical_incidents": critical_incidents,
        "high_incidents": high_incidents,
        "latest_cpu": latest_cpu,
        "latest_memory": latest_memory,
        "latest_response_time": latest_response_time,
        "latest_request_count": latest_request_count,
    }
