"""
Data Generator Module for Real-Time Anomaly Detection Platform.

This module provides functions to:
1. Generate a clean baseline training dataset representing normal system operations.
2. Generate single real-time simulated streaming observations.
3. Inject synthetic anomalous observations with high resource spikes to test anomaly detection.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

# The core feature columns used across training, streaming, and ML prediction
FEATURE_COLUMNS = ["cpu_usage", "memory_usage", "response_time", "request_count"]


def generate_baseline_data(
    n_samples: int = 600, random_seed: int = 42
) -> pd.DataFrame:
    """Generates synthetic baseline data representing normal server operation.

    This baseline dataset is strictly used to train the Isolation Forest model.
    It contains normal fluctuations without extreme spikes or system failures.

    Parameters:
        n_samples (int): Number of normal historical observations to generate.
        random_seed (int): Seed for reproducible baseline generation.

    Returns:
        pd.DataFrame: DataFrame containing normal metric observations.
    """
    rng = np.random.default_rng(random_seed)

    # Generate sequential timestamps going back in time
    base_time = datetime.now() - timedelta(minutes=n_samples)
    timestamps = [base_time + timedelta(minutes=i) for i in range(n_samples)]

    # Normal metric distributions (healthy system operation)
    # CPU usage: typically 25% to 55%, mean around 38%
    cpu = rng.normal(loc=38.0, scale=7.0, size=n_samples)
    cpu = np.clip(cpu, 12.0, 65.0)

    # Memory usage: stable between 45% to 65%, mean around 52%
    memory = rng.normal(loc=52.0, scale=5.0, size=n_samples)
    memory = np.clip(memory, 35.0, 72.0)

    # Response time: fast and stable around 110ms with mild variance
    response_time = rng.normal(loc=110.0, scale=20.0, size=n_samples)
    response_time = np.clip(response_time, 50.0, 220.0)

    # Request count: typical traffic around 220 requests/sec
    request_count = rng.normal(loc=220.0, scale=35.0, size=n_samples)
    request_count = np.clip(request_count, 100.0, 400.0)

    df_baseline = pd.DataFrame(
        {
            "timestamp": timestamps,
            "cpu_usage": np.round(cpu, 2),
            "memory_usage": np.round(memory, 2),
            "response_time": np.round(response_time, 2),
            "request_count": np.round(request_count, 1),
        }
    )

    return df_baseline


def generate_streaming_observation(
    prev_observation: Optional[Dict[str, float]] = None,
    current_time: Optional[datetime] = None,
) -> Dict[str, object]:
    """Generates a single incoming real-time telemetry observation.

    Simulates a live streaming metric event with realistic autocorrelated fluctuations.
    Does not use the baseline random seed so live data mimics genuine real-world inputs.

    Parameters:
        prev_observation (dict, optional): The previous metric record for smooth continuous drift.
        current_time (datetime, optional): Timestamp for this event (defaults to now).

    Returns:
        dict: A dictionary containing timestamp and numeric metric values.
    """
    if current_time is None:
        current_time = datetime.now()

    # If previous observation exists, use slight random walk for realism
    if prev_observation:
        prev_cpu = float(prev_observation.get("cpu_usage", 38.0))
        prev_mem = float(prev_observation.get("memory_usage", 52.0))
        prev_resp = float(prev_observation.get("response_time", 110.0))
        prev_req = float(prev_observation.get("request_count", 220.0))

        # AR(1) style mean-reverting drift
        cpu = 0.75 * prev_cpu + 0.25 * 38.0 + np.random.normal(0, 3.5)
        mem = 0.85 * prev_mem + 0.15 * 52.0 + np.random.normal(0, 1.8)
        resp = 0.70 * prev_resp + 0.30 * 110.0 + np.random.normal(0, 12.0)
        req = 0.75 * prev_req + 0.25 * 220.0 + np.random.normal(0, 18.0)
    else:
        cpu = np.random.normal(38.0, 7.0)
        mem = np.random.normal(52.0, 5.0)
        resp = np.random.normal(110.0, 20.0)
        req = np.random.normal(220.0, 35.0)

    # Bound within realistic normal operational boundaries
    cpu = float(np.clip(cpu, 12.0, 65.0))
    mem = float(np.clip(mem, 35.0, 72.0))
    resp = float(np.clip(resp, 50.0, 230.0))
    req = float(np.clip(req, 90.0, 420.0))

    return {
        "timestamp": current_time,
        "cpu_usage": round(cpu, 2),
        "memory_usage": round(mem, 2),
        "response_time": round(resp, 2),
        "request_count": round(req, 1),
        "is_injected": False,
    }


def generate_anomaly_observation(
    current_time: Optional[datetime] = None,
    anomaly_scenario: Optional[str] = None,
) -> Dict[str, object]:
    """Generates an intentional synthetic anomaly observation.

    Creates an observation that lies substantially outside the training distribution,
    simulating realistic outage patterns such as CPU exhaustion, memory leak,
    or DDoS surges.

    Parameters:
        current_time (datetime, optional): Timestamp for this event (defaults to now).
        anomaly_scenario (str, optional): Specific scenario to simulate.
            Options: 'traffic_surge', 'cpu_starvation', 'cascade_failure'.
            If None, randomly chooses one.

    Returns:
        dict: A dictionary containing the anomalous metric values and an injection marker.
    """
    if current_time is None:
        current_time = datetime.now()

    scenarios = ["traffic_surge", "cpu_starvation", "cascade_failure"]
    if not anomaly_scenario or anomaly_scenario not in scenarios:
        anomaly_scenario = np.random.choice(scenarios)

    if anomaly_scenario == "traffic_surge":
        # Sudden traffic spike (DDoS or flash crowd) leading to degraded response
        cpu = np.random.uniform(88.0, 97.0)
        mem = np.random.uniform(78.0, 89.0)
        resp = np.random.uniform(1400.0, 2800.0)  # >10x baseline
        req = np.random.uniform(1200.0, 2500.0)   # >6x baseline
    elif anomaly_scenario == "cpu_starvation":
        # Runaway process locking CPU cores
        cpu = np.random.uniform(94.0, 99.5)
        mem = np.random.uniform(82.0, 91.0)
        resp = np.random.uniform(1200.0, 2400.0)
        req = np.random.uniform(180.0, 350.0)
    else:  # cascade_failure
        # Catastrophic memory leak and thread pileup
        cpu = np.random.uniform(92.0, 99.0)
        mem = np.random.uniform(92.0, 99.2)
        resp = np.random.uniform(2200.0, 4200.0)
        req = np.random.uniform(800.0, 1800.0)

    return {
        "timestamp": current_time,
        "cpu_usage": round(float(cpu), 2),
        "memory_usage": round(float(mem), 2),
        "response_time": round(float(resp), 2),
        "request_count": round(float(req), 1),
        "is_injected": True,
        "scenario": anomaly_scenario,
    }
