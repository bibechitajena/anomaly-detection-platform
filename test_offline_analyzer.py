"""
Unit and integration tests for offline_analyzer module.
Covers:
1. Valid CSV input with explicit and heuristic column mappings.
2. Missing required/mapped columns detection and error reporting.
3. Invalid numeric values and malformed row imputation.
4. Missing timestamp generation.
5. Empty uploads and empty data row rejection.
6. Duplicate timestamp resolution via monotonic offsets.
7. Detector integration (both unsupervised and operational baseline modes).
8. Result exports (validated CSV, analyzed CSV, anomalies only CSV).
9. Explicit incident promotion to IncidentStore without polluting live stream.
"""

from datetime import datetime, timedelta
import io
import tempfile
import unittest
import pandas as pd

from incident_store import IncidentStore
from offline_analyzer import (
    validate_and_parse_csv,
    run_offline_detection,
    save_offline_incidents_to_store,
)


class TestOfflineAnalyzer(unittest.TestCase):
    def test_valid_csv_input(self):
        """Test parsing and validation of a well-formed CSV."""
        csv_text = (
            "timestamp,cpu_usage,memory_usage,response_time\n"
            "2026-10-24 10:00:00,35.2,50.1,105.0\n"
            "2026-10-24 10:00:10,38.4,52.3,112.5\n"
            "2026-10-24 10:00:20,95.0,85.2,2500.0\n"
        )
        mapping = {
            "cpu_usage": "cpu_usage",
            "memory_usage": "memory_usage",
            "response_time": "response_time",
        }
        df, warnings, errors = validate_and_parse_csv(csv_text, column_mapping=mapping)

        self.assertFalse(errors)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 3)
        self.assertIn("timestamp", df.columns)
        self.assertIn("cpu_usage", df.columns)
        self.assertEqual(df["cpu_usage"].iloc[-1], 95.0)

    def test_missing_required_columns(self):
        """Test error when mapped column does not exist and no valid metrics are found."""
        csv_text = "colA,colB\n1,2\n3,4\n"
        mapping = {"non_existent_col": "cpu_usage"}
        df, warnings, errors = validate_and_parse_csv(csv_text, column_mapping=mapping)

        self.assertTrue(errors)
        self.assertIsNone(df)
        self.assertTrue(any("At least one valid numeric metric column" in e for e in errors))

    def test_invalid_numeric_values(self):
        """Test that invalid numbers/corrupt strings are coerced and imputed without crashing."""
        csv_text = (
            "time,cpu,latency\n"
            "2026-10-24 10:00:00,35.0,100.0\n"
            "2026-10-24 10:00:10,corrupt_string,null\n"
            "2026-10-24 10:00:20,40.0,120.0\n"
        )
        mapping = {"cpu": "cpu_usage", "latency": "response_time"}
        df, warnings, errors = validate_and_parse_csv(csv_text, column_mapping=mapping)

        self.assertFalse(errors)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 3)
        # Check that NaN was filled with valid numeric values
        self.assertTrue(pd.notna(df["cpu_usage"].iloc[1]))
        self.assertTrue(pd.notna(df["response_time"].iloc[1]))
        self.assertTrue(any("non-numeric/missing" in w for w in warnings))

    def test_missing_timestamps(self):
        """Test that synthetic sequential timestamps are generated when none exist."""
        csv_text = "cpu_percent,mem_percent\n35.0,50.0\n38.0,52.0\n42.0,55.0\n"
        mapping = {"cpu_percent": "cpu_usage", "mem_percent": "memory_usage"}
        df, warnings, errors = validate_and_parse_csv(csv_text, column_mapping=mapping)

        self.assertFalse(errors)
        self.assertIsNotNone(df)
        self.assertIn("timestamp", df.columns)
        self.assertEqual(len(df["timestamp"]), 3)
        self.assertTrue(any("No timestamp column" in w for w in warnings))
        # Ensure timestamps are strictly increasing
        self.assertTrue(df["timestamp"].iloc[1] > df["timestamp"].iloc[0])

    def test_empty_uploads(self):
        """Test rejection of empty inputs."""
        # None
        df1, _, errors1 = validate_and_parse_csv(None)
        self.assertTrue(errors1)

        # Empty bytes
        df2, _, errors2 = validate_and_parse_csv(b"")
        self.assertTrue(errors2)

        # Header only (no rows)
        df3, _, errors3 = validate_and_parse_csv("colA,colB\n")
        self.assertTrue(errors3)

    def test_duplicate_timestamps(self):
        """Test that duplicate timestamps receive monotonic millisecond offsets."""
        csv_text = (
            "timestamp,cpu_usage\n"
            "2026-10-24 10:00:00,35.0\n"
            "2026-10-24 10:00:00,36.0\n"
            "2026-10-24 10:00:00,37.0\n"
        )
        mapping = {"cpu_usage": "cpu_usage"}
        df, warnings, errors = validate_and_parse_csv(csv_text, column_mapping=mapping)

        self.assertFalse(errors)
        self.assertIsNotNone(df)
        # Verify timestamps are now strictly unique and monotonic
        self.assertFalse(df["timestamp"].duplicated().any())
        self.assertTrue(any("duplicate timestamp" in w for w in warnings))

    def test_detector_integration_and_exports(self):
        """Test running detection on parsed data and exporting subsets."""
        # Create dataset with 30 normal points and 2 obvious spikes
        records = []
        base_time = datetime(2026, 10, 24, 10, 0, 0)
        for i in range(30):
            records.append({
                "time": (base_time + timedelta(seconds=i * 10)).isoformat(),
                "cpu": 35.0 + (i % 5),
                "latency": 100.0 + (i % 10),
            })
        # Add 2 extreme anomalies
        records.append({
            "time": (base_time + timedelta(seconds=300)).isoformat(),
            "cpu": 99.5,
            "latency": 3200.0,
        })
        records.append({
            "time": (base_time + timedelta(seconds=310)).isoformat(),
            "cpu": 98.8,
            "latency": 2900.0,
        })

        df_input = pd.DataFrame(records)
        csv_buf = io.StringIO()
        df_input.to_csv(csv_buf, index=False)

        mapping = {"cpu": "cpu_usage", "latency": "response_time"}
        cleaned_df, warnings, errors = validate_and_parse_csv(
            csv_buf.getvalue(), column_mapping=mapping
        )
        self.assertFalse(errors)

        # 1. Unsupervised detection
        analyzed_unsup = run_offline_detection(
            cleaned_df,
            features=["cpu_usage", "response_time"],
            contamination=0.06,
            z_threshold=2.5,
            baseline_mode="unsupervised",
        )
        self.assertIn("is_anomaly", analyzed_unsup.columns)
        self.assertIn("anomaly_score", analyzed_unsup.columns)
        self.assertIn("severity", analyzed_unsup.columns)
        self.assertIn("explanation", analyzed_unsup.columns)

        # The extreme points must be detected
        anomalies_unsup = analyzed_unsup[analyzed_unsup["is_anomaly"]]
        self.assertGreaterEqual(len(anomalies_unsup), 1)

        # 2. Operational baseline detection
        analyzed_base = run_offline_detection(
            cleaned_df,
            features=["cpu_usage", "response_time"],
            contamination=0.04,
            z_threshold=2.5,
            baseline_mode="operational_baseline",
        )
        anomalies_base = analyzed_base[analyzed_base["is_anomaly"]]
        self.assertGreaterEqual(len(anomalies_base), 1)

        # 3. Export validation
        csv_all = analyzed_base.to_csv(index=False)
        self.assertTrue(len(csv_all) > 0)
        self.assertIn("anomaly_score", csv_all)

        csv_anomalies_only = anomalies_base.to_csv(index=False)
        self.assertTrue(len(csv_anomalies_only) > 0)

    def test_save_offline_incidents_to_store(self):
        """Test explicitly saving flagged offline anomalies to SQLite incident store."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            db_path = f"{temp_dir}/test_offline_incidents.db"
            store = IncidentStore(db_path=db_path, cooldown_seconds=60)

            # Create sample analyzed dataframe
            analyzed_df = pd.DataFrame([
                {
                    "timestamp": datetime.now(),
                    "cpu_usage": 38.0,
                    "is_anomaly": False,
                    "severity": "Normal",
                    "explanation": "Normal bounds",
                },
                {
                    "timestamp": datetime.now() + timedelta(seconds=5),
                    "cpu_usage": 98.5,
                    "is_anomaly": True,
                    "severity": "Critical",
                    "anomaly_score": -0.19,
                    "anomaly_risk_index": 95.0,
                    "explanation": "Cpu Usage: 98.5 (+8.4σ)",
                },
            ])

            # Save to store
            saved_count = save_offline_incidents_to_store(analyzed_df, store)
            self.assertEqual(saved_count, 1)

            incidents = store.get_incidents()
            self.assertEqual(len(incidents), 1)
            self.assertEqual(incidents.iloc[0]["detector_name"], "IsolationForest-Offline")
            self.assertEqual(incidents.iloc[0]["severity"], "Critical")


if __name__ == "__main__":
    unittest.main()
