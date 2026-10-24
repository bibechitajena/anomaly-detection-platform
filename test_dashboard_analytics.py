"""
Unit and integration tests for dashboard analytics module (analytics.py).
Validates:
1. Empty incident store aggregation handling.
2. Incident severity and status distributions.
3. Live vs Offline incident source attribution.
4. Timeline bucket aggregation.
5. Rolling anomaly rate calculation.
6. Metric correlation matrix calculations.
7. System health status calculation and transitions (Healthy -> Warning -> Critical).
"""

from datetime import datetime, timedelta
import tempfile
import unittest
import pandas as pd

from incident_store import IncidentStore
from analytics import (
    get_incident_severity_summary,
    get_incident_status_summary,
    get_incident_source_summary,
    get_incident_time_distribution,
    get_anomaly_rate_trend,
    get_metric_correlation_matrix,
    get_system_health_summary,
)


class TestDashboardAnalytics(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = f"{self.temp_dir.name}/test_analytics.db"
        self.store = IncidentStore(db_path=self.db_path, cooldown_seconds=60)

    def tearDown(self):
        del self.store
        self.temp_dir.cleanup()

    def test_empty_incident_store_summaries(self):
        """Test summaries handle empty database gracefully without crashing."""
        sev_df = get_incident_severity_summary(self.store)
        self.assertEqual(len(sev_df), 3)
        self.assertEqual(sev_df["Count"].sum(), 0)

        status_df = get_incident_status_summary(self.store)
        self.assertEqual(len(status_df), 4)
        self.assertEqual(status_df["Count"].sum(), 0)

        source_df = get_incident_source_summary(self.store)
        self.assertEqual(source_df["Count"].sum(), 0)

        time_df = get_incident_time_distribution(self.store)
        self.assertTrue(time_df.empty)

    def test_populated_incident_summaries(self):
        """Test accurate counts for severities, statuses, and sources."""
        now = datetime.now()

        # Add 1 Critical Open incident (Live)
        self.store.record_anomaly({
            "timestamp": now,
            "cpu_usage": 98.0,
            "is_anomaly": True,
            "severity": "Critical",
            "explanation": "Cpu Usage: 98.0 (+8.2σ)",
        }, detector_name="IsolationForest")

        # Add 1 High Resolved incident (Offline)
        inc_id2, _ = self.store.record_anomaly({
            "timestamp": now + timedelta(seconds=70),
            "response_time": 2500.0,
            "is_anomaly": True,
            "severity": "High",
            "explanation": "Response Time: 2500.0 (+7.1σ)",
        }, detector_name="IsolationForest-Offline")
        self.store.update_incident_status(inc_id2, "Resolved")

        # Severity breakdown
        sev_df = get_incident_severity_summary(self.store)
        sev_dict = dict(zip(sev_df["Severity"], sev_df["Count"]))
        self.assertEqual(sev_dict["Critical"], 1)
        self.assertEqual(sev_dict["High"], 1)
        self.assertEqual(sev_dict["Medium"], 0)

        # Status breakdown
        stat_df = get_incident_status_summary(self.store)
        stat_dict = dict(zip(stat_df["Status"], stat_df["Count"]))
        self.assertEqual(stat_dict["Open"], 1)
        self.assertEqual(stat_dict["Resolved"], 1)

        # Source breakdown
        src_df = get_incident_source_summary(self.store)
        src_dict = dict(zip(src_df["Source"], src_df["Count"]))
        self.assertEqual(src_dict["Live Streaming"], 1)
        self.assertEqual(src_dict["Offline Upload"], 1)

    def test_rolling_anomaly_rate(self):
        """Test rolling rate calculation over a mock telemetry stream."""
        base_time = datetime(2026, 10, 24, 10, 0, 0)
        history = []
        for i in range(20):
            history.append({
                "timestamp": base_time + timedelta(seconds=i * 5),
                "is_anomaly": (i >= 15),  # Last 5 observations are anomalies
            })

        trend_df = get_anomaly_rate_trend(history, window_size=5)
        self.assertEqual(len(trend_df), 20)
        # Final rolling rate should be 100% since last 5 are all anomalies
        self.assertEqual(trend_df["rolling_anomaly_rate"].iloc[-1], 100.0)

    def test_metric_correlation_matrix(self):
        """Test correlation matrix calculation."""
        history = [
            {"cpu_usage": 20.0, "memory_usage": 40.0, "response_time": 100.0, "request_count": 200.0},
            {"cpu_usage": 40.0, "memory_usage": 50.0, "response_time": 150.0, "request_count": 250.0},
            {"cpu_usage": 60.0, "memory_usage": 60.0, "response_time": 200.0, "request_count": 300.0},
            {"cpu_usage": 80.0, "memory_usage": 70.0, "response_time": 250.0, "request_count": 350.0},
        ]
        corr = get_metric_correlation_matrix(history)
        self.assertFalse(corr.empty)
        # Perfect positive correlation in synthetic linear test
        self.assertAlmostEqual(corr.loc["cpu_usage", "memory_usage"], 1.0, places=1)

    def test_system_health_transitions(self):
        """Test system health state transitions (HEALTHY, WARNING, CRITICAL)."""
        # Case 1: Healthy
        healthy_history = [
            {"timestamp": datetime.now(), "cpu_usage": 35.0, "memory_usage": 50.0, "response_time": 110.0, "request_count": 220.0, "is_anomaly": False}
        ]
        h_summary = get_system_health_summary(healthy_history, self.store)
        self.assertEqual(h_summary["health_status"], "HEALTHY")

        # Case 2: Warning (High incident added)
        self.store.record_anomaly({
            "timestamp": datetime.now(),
            "cpu_usage": 78.0,
            "is_anomaly": True,
            "severity": "High",
            "explanation": "Cpu Usage: 78.0 (+4.6σ)",
        })
        w_summary = get_system_health_summary(healthy_history, self.store)
        self.assertEqual(w_summary["health_status"], "WARNING")

        # Case 3: Critical (Critical incident added)
        self.store.record_anomaly({
            "timestamp": datetime.now() + timedelta(seconds=70),
            "cpu_usage": 98.0,
            "is_anomaly": True,
            "severity": "Critical",
            "explanation": "Cpu Usage: 98.0 (+8.2σ)",
        })
        c_summary = get_system_health_summary(healthy_history, self.store)
        self.assertEqual(c_summary["health_status"], "CRITICAL")


if __name__ == "__main__":
    unittest.main()
