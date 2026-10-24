"""
Anomaly Detector Module using scikit-learn's IsolationForest.

This module encapsulates the Machine Learning logic:
- Trains strictly on baseline normal observations.
- Never retrains on incoming streaming or test observations (prevents model poisoning).
- Predicts anomaly status, normalized anomaly scores, severity levels, and explainability hints.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from data_generator import FEATURE_COLUMNS


class AnomalyDetector:
    """Isolation Forest based anomaly detector for server telemetry metrics.

    Attributes:
        features (list): Names of feature columns used for detection.
        contamination (float): Expected proportion of outliers in the baseline training data.
        random_state (int): Reproducible seed for tree generation.
        model (IsolationForest): The trained scikit-learn model instance.
        is_trained (bool): Whether the model has been fitted.
        baseline_stats (dict): Mean and std of features in baseline data for deviation calculation.
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        contamination: float = 0.04,
        random_state: int = 42,
        z_threshold: float = 2.5,
    ):
        self.features = features or FEATURE_COLUMNS
        self.contamination = contamination
        self.random_state = random_state
        self.z_threshold = z_threshold
        self.is_trained = False
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        self.trained_sample_count: int = 0
        self._baseline_df: Optional[pd.DataFrame] = None

        # Initialize scikit-learn IsolationForest with reproducible seed
        self.model = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=self.random_state,
            bootstrap=False,
            n_jobs=-1,
        )

    def train(self, baseline_df: pd.DataFrame) -> None:
        """Trains the Isolation Forest strictly on baseline operational data.

        Important Machine Learning Rule:
        We ONLY fit this model on baseline data representing known normal system behavior.
        We do NOT train or update the model on live incoming monitoring points.

        Parameters:
            baseline_df (pd.DataFrame): DataFrame containing baseline historical metrics.
        """
        # Validate that required features exist in baseline dataframe
        missing_cols = [col for col in self.features if col not in baseline_df.columns]
        if missing_cols:
            raise ValueError(
                f"Baseline data is missing required feature columns: {missing_cols}"
            )

        self._baseline_df = baseline_df.copy()
        X_train = baseline_df[self.features].copy()

        # Fit the Isolation Forest model
        self.model.fit(X_train)
        self.is_trained = True
        self.trained_sample_count = len(baseline_df)

        # Store baseline statistics (mean, std) for rule-based severity and explainability
        self.baseline_stats = {}
        for col in self.features:
            col_mean = float(baseline_df[col].mean())
            col_std = float(baseline_df[col].std())
            # Avoid division by zero
            if col_std == 0:
                col_std = 1.0
            self.baseline_stats[col] = {
                "mean": col_mean,
                "std": col_std,
                "p95": float(baseline_df[col].quantile(0.95)),
            }

    def _determine_severity_and_explanation(
        self, features_dict: Dict[str, float], raw_score: float, is_anomaly: bool
    ) -> Tuple[str, str]:
        """Calculates severity and human-readable explanation based on feature z-scores."""
        if not is_anomaly:
            return "Normal", "Operating within baseline bounds"

        deviations = []
        max_z = 0.0

        for col in self.features:
            val = float(features_dict.get(col, 0.0))
            stats = self.baseline_stats.get(col, {"mean": 0.0, "std": 1.0})
            z = (val - stats["mean"]) / stats["std"]
            if z > self.z_threshold:
                deviations.append(f"{col.replace('_', ' ').title()}: {val} (+{z:.1f}σ)")
                if z > max_z:
                    max_z = z

        explanation = (
            ", ".join(deviations)
            if deviations
            else "Multivariate pattern deviated from baseline"
        )

        # Assign severity based on deviation magnitude and IsolationForest score
        # Scaled dynamically based on the configured z_threshold
        crit_z = self.z_threshold * 3.2
        high_z = self.z_threshold * 1.8

        if raw_score < -0.15 or max_z > crit_z:
            severity = "Critical"
        elif raw_score < -0.07 or max_z > high_z:
            severity = "High"
        else:
            severity = "Medium"

        return severity, explanation

    def predict_point(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Predicts anomaly status for a single incoming observation.

        Parameters:
            observation (dict): Dictionary containing metric feature values.

        Returns:
            dict: Enriched observation with 'is_anomaly', 'severity',
                  'anomaly_score', and 'explanation'.
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model is not trained yet! Call train(baseline_df) first."
            )

        features_df = pd.DataFrame([{col: observation[col] for col in self.features}])

        # scikit-learn predict returns 1 for inlier, -1 for outlier
        pred = self.model.predict(features_df)[0]
        # decision_function returns negative values for anomalies, positive for normal
        raw_score = float(self.model.decision_function(features_df)[0])

        is_anomaly = bool(pred == -1)
        severity, explanation = self._determine_severity_and_explanation(
            observation, raw_score, is_anomaly
        )

        # Convert raw_score into an intuitive 0.0-100.0 Anomaly Risk Index
        # Raw score typically spans from -0.35 (severe anomaly) to +0.25 (typical normal)
        # We map more negative scores to higher risk indices
        risk_index = float(np.clip(round((0.15 - raw_score) * 150.0, 1), 0.0, 100.0))

        result = dict(observation)
        result["is_anomaly"] = is_anomaly
        result["anomaly_status"] = "Anomaly" if is_anomaly else "Normal"
        result["severity"] = severity
        result["anomaly_score"] = round(raw_score, 4)
        result["anomaly_risk_index"] = risk_index
        result["explanation"] = explanation

        return result

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Runs predictions over a batch DataFrame without retraining.

        Parameters:
            df (pd.DataFrame): Incoming observations.

        Returns:
            pd.DataFrame: Augmented DataFrame with anomaly detection columns.
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model is not trained yet! Call train(baseline_df) first."
            )

        if df.empty:
            out = df.copy()
            out["is_anomaly"] = False
            out["anomaly_status"] = "Normal"
            out["severity"] = "Normal"
            out["anomaly_score"] = 0.0
            out["anomaly_risk_index"] = 0.0
            out["explanation"] = ""
            return out

        X = df[self.features].copy()
        preds = self.model.predict(X)
        scores = self.model.decision_function(X)

        result_df = df.copy()
        result_df["is_anomaly"] = preds == -1
        result_df["anomaly_status"] = np.where(
            result_df["is_anomaly"], "Anomaly", "Normal"
        )
        result_df["anomaly_score"] = np.round(scores, 4)

        # Calculate risk index
        result_df["anomaly_risk_index"] = np.clip(
            np.round((0.15 - scores) * 150.0, 1), 0.0, 100.0
        )

        severities = []
        explanations = []

        for idx, row in result_df.iterrows():
            obs_dict = {col: row[col] for col in self.features}
            sev, exp = self._determine_severity_and_explanation(
                obs_dict, row["anomaly_score"], row["is_anomaly"]
            )
            severities.append(sev)
            explanations.append(exp)

        result_df["severity"] = severities
        result_df["explanation"] = explanations

        return result_df

    def update_parameters(
        self,
        contamination: Optional[float] = None,
        z_threshold: Optional[float] = None,
    ) -> bool:
        """Dynamically recalibrates the detector sensitivity and threshold.

        Re-fits the Isolation Forest model on the baseline data if contamination changes.
        Updates internal thresholds for explainability and severity calculations.

        Parameters:
            contamination (float, optional): New contamination rate (e.g., 0.01 to 0.15).
            z_threshold (float, optional): New Z-score deviation trigger (e.g., 1.5 to 4.5).

        Returns:
            bool: True if parameters changed and calibration succeeded, False otherwise.
        """
        changed = False

        if z_threshold is not None and round(float(z_threshold), 2) != round(self.z_threshold, 2):
            self.z_threshold = round(float(z_threshold), 2)
            changed = True

        if contamination is not None and round(float(contamination), 4) != round(self.contamination, 4):
            self.contamination = float(contamination)
            if self.is_trained and self._baseline_df is not None:
                # Re-fit IsolationForest with updated contamination rate
                self.model = IsolationForest(
                    n_estimators=100,
                    contamination=self.contamination,
                    random_state=self.random_state,
                    bootstrap=False,
                    n_jobs=-1,
                )
                X_train = self._baseline_df[self.features]
                self.model.fit(X_train)
            changed = True

        return changed

    def rescore_history(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Re-evaluates a list of historical observations with current calibration settings.

        Preserves original raw metric features and timestamps while recomputing
        anomaly scores, severities, and diagnostic explanations.

        Parameters:
            history (list of dict): Previous observation history.

        Returns:
            list of dict: Rescored observations reflecting the new calibration.
        """
        if not self.is_trained or not history:
            return list(history)

        rescored = []
        for item in history:
            raw_obs = {
                "timestamp": item["timestamp"],
                "cpu_usage": float(item["cpu_usage"]),
                "memory_usage": float(item["memory_usage"]),
                "response_time": float(item["response_time"]),
                "request_count": float(item["request_count"]),
                "is_injected": item.get("is_injected", False),
            }
            if "scenario" in item:
                raw_obs["scenario"] = item["scenario"]

            rescored_item = self.predict_point(raw_obs)
            rescored.append(rescored_item)

        return rescored

    def get_summary(self) -> Dict[str, Any]:
        """Returns summary metadata about the trained model."""
        return {
            "model_type": "IsolationForest",
            "features": self.features,
            "trained_samples": self.trained_sample_count,
            "contamination": self.contamination,
            "z_threshold": self.z_threshold,
            "random_state": self.random_state,
            "is_trained": self.is_trained,
        }

