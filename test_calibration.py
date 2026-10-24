"""
Unit and integration tests for AnomalyDetector sensitivity & calibration.
Verifies Phase 1 functionality:
1. Baseline training and baseline metrics caching.
2. Contamination parameter updates and sensitivity impact.
3. Diagnostic Z-score threshold updates and explanation/severity impact.
4. Telemetry buffer rescoring consistency and data preservation.
"""

import unittest
from datetime import datetime
import pandas as pd
import numpy as np

from data_generator import generate_baseline_data, generate_streaming_observation, generate_anomaly_observation
from detector import AnomalyDetector


class TestDetectorCalibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reproducible baseline data
        cls.baseline_df = generate_baseline_data(n_samples=500, random_seed=42)

    def setUp(self):
        self.detector = AnomalyDetector(contamination=0.04, random_state=42, z_threshold=2.5)
        self.detector.train(self.baseline_df)

    def test_initial_state(self):
        """Test initial detector configuration and metadata."""
        summary = self.detector.get_summary()
        self.assertTrue(summary["is_trained"])
        self.assertEqual(summary["contamination"], 0.04)
        self.assertEqual(summary["z_threshold"], 2.5)
        self.assertEqual(summary["trained_samples"], 500)
        self.assertIn("cpu_usage", self.detector.baseline_stats)

    def test_normal_observation(self):
        """Test prediction on normal observation within baseline bounds."""
        normal_obs = {
            "timestamp": datetime.now(),
            "cpu_usage": 38.0,
            "memory_usage": 52.0,
            "response_time": 110.0,
            "request_count": 220.0,
        }
        res = self.detector.predict_point(normal_obs)
        self.assertFalse(res["is_anomaly"])
        self.assertEqual(res["anomaly_status"], "Normal")
        self.assertEqual(res["severity"], "Normal")
        self.assertLess(res["anomaly_risk_index"], 50.0)

    def test_anomaly_observation(self):
        """Test prediction on severe anomaly observation."""
        anomaly_obs = generate_anomaly_observation(anomaly_scenario="cpu_starvation")
        res = self.detector.predict_point(anomaly_obs)
        self.assertTrue(res["is_anomaly"])
        self.assertIn(res["severity"], ["Critical", "High"])
        self.assertIn("Cpu Usage", res["explanation"])

    def test_contamination_sensitivity_impact(self):
        """Test that updating contamination actually alters the decision threshold.
        Higher contamination = more sensitive (lower decision_function threshold).
        """
        # Create a borderline observation (slightly elevated, e.g. CPU 58%, Mem 65%, Latency 190ms)
        borderline_obs = {
            "timestamp": datetime.now(),
            "cpu_usage": 58.0,
            "memory_usage": 67.0,
            "response_time": 195.0,
            "request_count": 320.0,
        }

        # With low contamination (conservative, e.g. 1%), borderline is treated as normal
        self.detector.update_parameters(contamination=0.01)
        res_conservative = self.detector.predict_point(borderline_obs)

        # With high contamination (sensitive, e.g. 15%), borderline is flagged as anomaly
        self.detector.update_parameters(contamination=0.15)
        res_sensitive = self.detector.predict_point(borderline_obs)

        # Verify that score is lower/more critical and/or status flipped
        self.assertNotEqual(res_conservative["anomaly_score"], res_sensitive["anomaly_score"])
        self.assertTrue(
            res_sensitive["anomaly_risk_index"] > res_conservative["anomaly_risk_index"]
        )

    def test_z_threshold_impact(self):
        """Test that changing z_threshold changes diagnostic triggers and severity."""
        # Moderately elevated observation (+2.8 sigma on CPU)
        mean_cpu = self.detector.baseline_stats["cpu_usage"]["mean"]
        std_cpu = self.detector.baseline_stats["cpu_usage"]["std"]
        elevated_cpu = round(mean_cpu + 2.8 * std_cpu, 1)

        obs = {
            "timestamp": datetime.now(),
            "cpu_usage": elevated_cpu,
            "memory_usage": 52.0,
            "response_time": 110.0,
            "request_count": 220.0,
        }

        # With relaxed z_threshold=3.5, 2.8 sigma is NOT triggered in explanation
        self.detector.update_parameters(z_threshold=3.5)
        _, exp_relaxed = self.detector._determine_severity_and_explanation(obs, raw_score=-0.08, is_anomaly=True)
        self.assertNotIn("Cpu Usage", exp_relaxed)

        # With strict z_threshold=2.0, 2.8 sigma IS triggered in explanation
        self.detector.update_parameters(z_threshold=2.0)
        _, exp_strict = self.detector._determine_severity_and_explanation(obs, raw_score=-0.08, is_anomaly=True)
        self.assertIn("Cpu Usage", exp_strict)

    def test_rescore_history(self):
        """Test that rescore_history re-evaluates all observations while preserving raw telemetry."""
        history = [
            {
                "timestamp": datetime(2026, 1, 1, 12, 0, 0),
                "cpu_usage": 35.0,
                "memory_usage": 50.0,
                "response_time": 100.0,
                "request_count": 200.0,
                "is_injected": False,
            },
            {
                "timestamp": datetime(2026, 1, 1, 12, 0, 5),
                "cpu_usage": 98.0,
                "memory_usage": 95.0,
                "response_time": 3000.0,
                "request_count": 1500.0,
                "is_injected": True,
                "scenario": "cascade_failure",
            },
        ]

        # Initial scoring with standard calibration
        scored = self.detector.rescore_history(history)
        self.assertEqual(len(scored), 2)
        self.assertEqual(scored[0]["timestamp"], history[0]["timestamp"])
        self.assertEqual(scored[1]["timestamp"], history[1]["timestamp"])
        self.assertEqual(scored[0]["cpu_usage"], 35.0)
        self.assertEqual(scored[1]["cpu_usage"], 98.0)
        self.assertFalse(scored[0]["is_anomaly"])
        self.assertTrue(scored[1]["is_anomaly"])


if __name__ == "__main__":
    unittest.main()
