"""
Unit and integration tests for IncidentStore.
Validates:
1. Database schema initialization.
2. Recording an anomaly into a persistent incident.
3. Proper storage of fields (ID, timestamp, metric, observed value, severity, detector, status, evidence).
4. Status transitions: Open -> Acknowledged -> Resolved -> Ignored.
5. Duplicate suppression and grouping within cooldown window.
6. Cooldown expiration generating new incident.
7. Independent metrics generating distinct incidents.
8. Persistence across separate IncidentStore instances (simulating restart).
9. Statistics aggregation.
"""

from datetime import datetime, timedelta
import os
import tempfile
import unittest

import gc
from incident_store import IncidentStore, VALID_STATUSES


class TestIncidentStore(unittest.TestCase):
    def setUp(self):
        # Create a unique temporary directory for isolation
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = os.path.join(self.temp_dir.name, "test_incidents.db")
        self.store = IncidentStore(db_path=self.db_path, cooldown_seconds=60)

    def tearDown(self):
        del self.store
        gc.collect()
        self.temp_dir.cleanup()

    def test_init_db(self):
        """Verify database and table are properly initialized."""
        self.assertTrue(os.path.exists(self.db_path))
        incidents = self.store.get_incidents()
        self.assertTrue(incidents.empty)

    def test_record_anomaly_fields(self):
        """Test recording an anomaly persists all expected fields."""
        now = datetime.now()
        obs = {
            "timestamp": now,
            "cpu_usage": 98.4,
            "memory_usage": 85.0,
            "response_time": 2400.0,
            "request_count": 350.0,
            "is_anomaly": True,
            "severity": "Critical",
            "anomaly_score": -0.18,
            "anomaly_risk_index": 92.5,
            "explanation": "Cpu Usage: 98.4 (+8.3σ)",
            "scenario": "cpu_starvation",
        }

        inc_id, is_new = self.store.record_anomaly(obs, detector_name="IsolationForest")
        self.assertTrue(is_new)
        self.assertTrue(inc_id.startswith("INC-"))

        # Retrieve and verify stored record
        record = self.store.get_incident_by_id(inc_id)
        self.assertIsNotNone(record)
        self.assertEqual(record["incident_id"], inc_id)
        self.assertEqual(record["metric"], "cpu_usage")
        self.assertAlmostEqual(record["observed_value"], 98.4)
        self.assertEqual(record["severity"], "Critical")
        self.assertEqual(record["detector_name"], "IsolationForest")
        self.assertEqual(record["status"], "Open")
        self.assertEqual(record["occurrence_count"], 1)
        self.assertIn("cpu_starvation", record["evidence"])
        self.assertIn("98.4", record["evidence"])

    def test_status_lifecycle_and_notes(self):
        """Test transitioning statuses through Open, Acknowledged, Resolved, Ignored."""
        obs = {
            "timestamp": datetime.now(),
            "cpu_usage": 95.0,
            "memory_usage": 80.0,
            "response_time": 1800.0,
            "request_count": 250.0,
            "is_anomaly": True,
            "severity": "High",
            "explanation": "Cpu Usage: 95.0 (+7.8σ)",
        }
        inc_id, _ = self.store.record_anomaly(obs)

        # 1. Acknowledge
        success = self.store.update_incident_status(
            inc_id, "Acknowledged", notes="Investigating CPU lockup"
        )
        self.assertTrue(success)
        rec = self.store.get_incident_by_id(inc_id)
        self.assertEqual(rec["status"], "Acknowledged")
        self.assertEqual(rec["notes"], "Investigating CPU lockup")

        # 2. Resolve
        success = self.store.update_incident_status(
            inc_id, "Resolved", notes="Worker process restarted"
        )
        self.assertTrue(success)
        rec = self.store.get_incident_by_id(inc_id)
        self.assertEqual(rec["status"], "Resolved")
        self.assertEqual(rec["notes"], "Worker process restarted")

        # 3. Ignore
        success = self.store.update_incident_status(inc_id, "Ignored")
        self.assertTrue(success)
        rec = self.store.get_incident_by_id(inc_id)
        self.assertEqual(rec["status"], "Ignored")

        # 4. Reject invalid status
        success = self.store.update_incident_status(inc_id, "InvalidStatus")
        self.assertFalse(success)

    def test_duplicate_suppression_and_cooldown(self):
        """Verify grouping of repeated anomalies within the cooldown period."""
        base_time = datetime(2026, 10, 24, 12, 0, 0)

        obs1 = {
            "timestamp": base_time,
            "cpu_usage": 94.0,
            "memory_usage": 70.0,
            "response_time": 1200.0,
            "request_count": 200.0,
            "is_anomaly": True,
            "severity": "High",
            "explanation": "Cpu Usage: 94.0 (+7.5σ)",
        }

        # First occurrence creates new incident
        inc_id1, is_new1 = self.store.record_anomaly(obs1, cooldown_seconds=60)
        self.assertTrue(is_new1)

        # Second occurrence 10 seconds later (within 60s cooldown)
        obs2 = {
            "timestamp": base_time + timedelta(seconds=10),
            "cpu_usage": 97.5,  # higher spike
            "memory_usage": 72.0,
            "response_time": 1400.0,
            "request_count": 210.0,
            "is_anomaly": True,
            "severity": "Critical",
            "explanation": "Cpu Usage: 97.5 (+8.2σ)",
        }
        inc_id2, is_new2 = self.store.record_anomaly(obs2, cooldown_seconds=60)

        # Must be grouped into existing incident, not creating a duplicate
        self.assertFalse(is_new2)
        self.assertEqual(inc_id1, inc_id2)

        rec = self.store.get_incident_by_id(inc_id1)
        self.assertEqual(rec["occurrence_count"], 2)
        self.assertAlmostEqual(rec["observed_value"], 97.5)  # Peak value tracked
        self.assertEqual(rec["severity"], "Critical")

        # Third occurrence 75 seconds later (exceeds 60s cooldown)
        obs3 = {
            "timestamp": base_time + timedelta(seconds=75),
            "cpu_usage": 96.0,
            "memory_usage": 70.0,
            "response_time": 1300.0,
            "request_count": 200.0,
            "is_anomaly": True,
            "severity": "High",
            "explanation": "Cpu Usage: 96.0 (+7.9σ)",
        }
        inc_id3, is_new3 = self.store.record_anomaly(obs3, cooldown_seconds=60)

        # Must create a new incident because cooldown expired
        self.assertTrue(is_new3)
        self.assertNotEqual(inc_id1, inc_id3)

        # Total incidents should now be 2
        all_incidents = self.store.get_incidents()
        self.assertEqual(len(all_incidents), 2)

    def test_distinct_metrics_create_distinct_incidents(self):
        """Anomalies in different metrics within the same time window create distinct incidents."""
        now = datetime.now()
        obs_cpu = {
            "timestamp": now,
            "cpu_usage": 98.0,
            "memory_usage": 50.0,
            "response_time": 100.0,
            "request_count": 200.0,
            "is_anomaly": True,
            "severity": "Critical",
            "explanation": "Cpu Usage: 98.0 (+8.2σ)",
        }
        obs_latency = {
            "timestamp": now + timedelta(seconds=2),
            "cpu_usage": 40.0,
            "memory_usage": 52.0,
            "response_time": 2900.0,
            "request_count": 220.0,
            "is_anomaly": True,
            "severity": "Critical",
            "explanation": "Response Time: 2900.0 (+10.5σ)",
        }

        id_cpu, new_cpu = self.store.record_anomaly(obs_cpu)
        id_latency, new_lat = self.store.record_anomaly(obs_latency)

        self.assertTrue(new_cpu)
        self.assertTrue(new_lat)
        self.assertNotEqual(id_cpu, id_latency)

        df = self.store.get_incidents()
        self.assertEqual(len(df), 2)
        metrics = set(df["metric"])
        self.assertEqual(metrics, {"cpu_usage", "response_time"})

    def test_persistence_across_restarts(self):
        """Simulate application restart by creating a new store pointing to the same file."""
        obs = {
            "timestamp": datetime.now(),
            "cpu_usage": 96.0,
            "memory_usage": 50.0,
            "response_time": 100.0,
            "request_count": 200.0,
            "is_anomaly": True,
            "severity": "High",
            "explanation": "Cpu Usage: 96.0 (+7.8σ)",
        }
        inc_id, _ = self.store.record_anomaly(obs)
        self.store.update_incident_status(inc_id, "Acknowledged", notes="Saved before restart")

        # Instantiate a second store on the same DB file
        restarted_store = IncidentStore(db_path=self.db_path)
        record = restarted_store.get_incident_by_id(inc_id)

        self.assertIsNotNone(record)
        self.assertEqual(record["incident_id"], inc_id)
        self.assertEqual(record["status"], "Acknowledged")
        self.assertEqual(record["notes"], "Saved before restart")

    def test_stats_calculation(self):
        """Verify aggregated counts by status and severity."""
        now = datetime.now()
        # Incident 1: Critical, Open
        id1, _ = self.store.record_anomaly({
            "timestamp": now,
            "cpu_usage": 99.0,
            "is_anomaly": True,
            "severity": "Critical",
            "explanation": "Cpu Usage: 99.0 (+8.5σ)",
        })
        # Incident 2: High, Resolved
        id2, _ = self.store.record_anomaly({
            "timestamp": now + timedelta(seconds=70),
            "response_time": 2500.0,
            "is_anomaly": True,
            "severity": "High",
            "explanation": "Response Time: 2500.0 (+6.5σ)",
        })
        self.store.update_incident_status(id2, "Resolved")

        stats = self.store.get_incident_stats()
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["open"], 1)
        self.assertEqual(stats["resolved"], 1)
        self.assertEqual(stats["critical"], 1)
        self.assertEqual(stats["high"], 1)


if __name__ == "__main__":
    unittest.main()
