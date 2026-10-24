"""
Incident Store Module for Real-Time Anomaly Detection Platform.
==============================================================
Provides SQLite-based persistence for detected anomaly incidents across
Streamlit reruns and server restarts.

Features:
- Parameterized SQL queries for safety and resilience against injection.
- Automatic table initialization with indices for fast querying.
- Grouping and cooldown deduplication: prevents alarm storms when the
  same anomaly persists across successive streaming observations.
- Lifecycle status management: Open -> Acknowledged -> Resolved / Ignored.
- Diagnostic evidence capturing (Z-scores, anomaly scores, full metric snapshots).
"""

from contextlib import contextmanager
from datetime import datetime
import json
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

# Valid incident lifecycle statuses
VALID_STATUSES = ("Open", "Acknowledged", "Resolved", "Ignored")
DEFAULT_COOLDOWN_SECONDS = 60


class IncidentStore:
    """Manages persistent storage, grouping, and retrieval of anomaly incidents using SQLite."""

    def __init__(
        self,
        db_path: str = "incidents.db",
        cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
    ):
        """Initializes the SQLite incident database.

        Parameters:
            db_path (str): File path to the SQLite database.
            cooldown_seconds (int): Minimum interval before creating a new incident
                                    for the same recurring metric anomaly.
        """
        self.db_path = db_path
        self.cooldown_seconds = cooldown_seconds
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Yields a configured SQLite connection and guarantees connection closure."""
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Creates the incidents schema and indices if they do not exist."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS incidents (
                        incident_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        last_seen_timestamp TEXT NOT NULL,
                        metric TEXT NOT NULL,
                        observed_value REAL NOT NULL,
                        severity TEXT NOT NULL,
                        detector_name TEXT NOT NULL,
                        status TEXT NOT NULL,
                        evidence TEXT NOT NULL,
                        occurrence_count INTEGER NOT NULL DEFAULT 1,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        notes TEXT DEFAULT ''
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_incidents_metric_status 
                    ON incidents (metric, status);
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_incidents_last_seen 
                    ON incidents (last_seen_timestamp DESC);
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_incidents_status 
                    ON incidents (status);
                    """
                )
        except sqlite3.Error as e:
            logger.error("Failed to initialize SQLite database: %s", e)
            raise

    def _determine_primary_metric(self, observation: Dict[str, Any]) -> Tuple[str, float]:
        """Identifies the primary contributing metric and its observed value."""
        explanation = str(observation.get("explanation", ""))
        metric_candidates = [
            ("cpu_usage", "Cpu Usage"),
            ("memory_usage", "Memory Usage"),
            ("response_time", "Response Time"),
            ("request_count", "Request Count"),
        ]

        # Check explanation string for first mentioned feature
        for key, label in metric_candidates:
            if label.lower() in explanation.lower() and key in observation:
                return key, float(observation[key])

        # Fallback to finding metric with highest value or default to multivariate
        for key, _ in metric_candidates:
            if key in observation and observation.get("is_anomaly", False):
                return key, float(observation[key])

        return "multivariate", float(observation.get("anomaly_risk_index", 0.0))

    def record_anomaly(
        self,
        observation: Dict[str, Any],
        detector_name: str = "IsolationForest",
        cooldown_seconds: Optional[int] = None,
    ) -> Tuple[str, bool]:
        """Records an anomaly event with duplicate suppression and cooldown grouping.

        If an active incident (Open or Acknowledged) exists for the same primary metric
        within the cooldown window, the existing incident's occurrence count and
        last_seen_timestamp are updated instead of creating a noisy duplicate.

        Parameters:
            observation (dict): Analyzed telemetry observation with anomaly metadata.
            detector_name (str): Identifier of the detector that flagged the anomaly.
            cooldown_seconds (int, optional): Override for incident grouping window.

        Returns:
            Tuple[str, bool]: (incident_id, is_new_incident)
        """
        cooldown = cooldown_seconds if cooldown_seconds is not None else self.cooldown_seconds
        metric, observed_value = self._determine_primary_metric(observation)
        severity = str(observation.get("severity", "High"))

        obs_time = observation.get("timestamp")
        if isinstance(obs_time, datetime):
            timestamp_str = obs_time.isoformat()
            current_dt = obs_time
        elif isinstance(obs_time, str):
            timestamp_str = obs_time
            try:
                current_dt = datetime.fromisoformat(obs_time)
            except ValueError:
                current_dt = datetime.now()
        else:
            current_dt = datetime.now()
            timestamp_str = current_dt.isoformat()

        now_str = datetime.now().isoformat()

        # Build evidence snapshot
        evidence_dict = {
            "explanation": observation.get("explanation", ""),
            "anomaly_score": observation.get("anomaly_score", 0.0),
            "anomaly_risk_index": observation.get("anomaly_risk_index", 0.0),
            "metrics": {
                k: observation.get(k)
                for k in ["cpu_usage", "memory_usage", "response_time", "request_count"]
                if k in observation
            },
            "scenario": observation.get("scenario"),
        }
        evidence_json = json.dumps(evidence_dict)

        try:
            with self._get_connection() as conn:
                # Check for active incident for this metric
                cursor = conn.execute(
                    """
                    SELECT incident_id, last_seen_timestamp, occurrence_count, observed_value
                    FROM incidents
                    WHERE metric = ? AND status IN ('Open', 'Acknowledged')
                    ORDER BY last_seen_timestamp DESC
                    LIMIT 1;
                    """,
                    (metric,),
                )
                row = cursor.fetchone()

                if row:
                    last_seen_str = row["last_seen_timestamp"]
                    try:
                        last_seen_dt = datetime.fromisoformat(last_seen_str)
                        delta_sec = abs((current_dt - last_seen_dt).total_seconds())
                    except ValueError:
                        delta_sec = cooldown + 1

                    # If within cooldown period, group and update existing incident
                    if delta_sec <= cooldown:
                        incident_id = row["incident_id"]
                        new_count = row["occurrence_count"] + 1
                        # Track peak observed value
                        peak_val = max(float(row["observed_value"]), observed_value)

                        conn.execute(
                            """
                            UPDATE incidents
                            SET last_seen_timestamp = ?,
                                occurrence_count = ?,
                                observed_value = ?,
                                severity = ?,
                                evidence = ?,
                                updated_at = ?
                            WHERE incident_id = ?;
                            """,
                            (
                                timestamp_str,
                                new_count,
                                peak_val,
                                severity,
                                evidence_json,
                                now_str,
                                incident_id,
                            ),
                        )
                        return incident_id, False

                # Otherwise create a new incident
                timestamp_prefix = current_dt.strftime("%Y%m%d%H%M%S")
                # Generate unique ID based on timestamp and microsecond
                short_uid = f"{current_dt.microsecond:06d}"[:4]
                incident_id = f"INC-{timestamp_prefix}-{short_uid}"

                conn.execute(
                    """
                    INSERT INTO incidents (
                        incident_id, timestamp, last_seen_timestamp, metric,
                        observed_value, severity, detector_name, status,
                        evidence, occurrence_count, created_at, updated_at, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'Open', ?, 1, ?, ?, '');
                    """,
                    (
                        incident_id,
                        timestamp_str,
                        timestamp_str,
                        metric,
                        observed_value,
                        severity,
                        detector_name,
                        evidence_json,
                        now_str,
                        now_str,
                    ),
                )
                return incident_id, True

        except sqlite3.Error as e:
            logger.error("Database error while recording anomaly: %s", e)
            raise

    def get_incidents(
        self,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        metric: Optional[str] = None,
        detector_name: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Retrieves incidents filtered by status, severity, metric, detector, and time range.

        Parameters:
            status (str, optional): Filter by status ('Open', 'Acknowledged', 'Resolved', 'Ignored').
            severity (str, optional): Filter by severity ('Critical', 'High', 'Medium').
            metric (str, optional): Filter by primary metric ('cpu_usage', 'memory_usage', etc.).
            detector_name (str, optional): Filter by detector ('IsolationForest', 'IsolationForest-Offline').
            since (datetime, optional): Cutoff timestamp for time-range filtering.
            limit (int): Maximum records to retrieve.

        Returns:
            pd.DataFrame: DataFrame containing incident records.
        """
        query = "SELECT * FROM incidents WHERE 1=1"
        params: List[Any] = []

        if status and status != "All":
            query += " AND status = ?"
            params.append(status)

        if severity and severity != "All":
            query += " AND severity = ?"
            params.append(severity)

        if metric and metric != "All":
            query += " AND metric = ?"
            params.append(metric)

        if detector_name and detector_name != "All":
            query += " AND detector_name = ?"
            params.append(detector_name)

        if since is not None:
            query += " AND last_seen_timestamp >= ?"
            params.append(since.isoformat())

        query += " ORDER BY last_seen_timestamp DESC LIMIT ?"
        params.append(limit)

        try:
            with self._get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
                return df
        except Exception as e:
            logger.error("Failed to query incidents: %s", e)
            return pd.DataFrame()

    def get_incident_by_id(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single incident by its ID."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM incidents WHERE incident_id = ?;", (incident_id,)
                )
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None
        except sqlite3.Error as e:
            logger.error("Failed to retrieve incident %s: %s", incident_id, e)
            return None

    def update_incident_status(
        self,
        incident_id: str,
        new_status: str,
        notes: Optional[str] = None,
    ) -> bool:
        """Updates the status and optional notes for an incident.

        Parameters:
            incident_id (str): Target incident ID.
            new_status (str): Must be one of 'Open', 'Acknowledged', 'Resolved', 'Ignored'.
            notes (str, optional): Operator annotations.

        Returns:
            bool: True if updated successfully, False otherwise.
        """
        if new_status not in VALID_STATUSES:
            logger.warning("Invalid incident status '%s' requested.", new_status)
            return False

        now_str = datetime.now().isoformat()
        try:
            with self._get_connection() as conn:
                if notes is not None:
                    cursor = conn.execute(
                        """
                        UPDATE incidents
                        SET status = ?, notes = ?, updated_at = ?
                        WHERE incident_id = ?;
                        """,
                        (new_status, notes, now_str, incident_id),
                    )
                else:
                    cursor = conn.execute(
                        """
                        UPDATE incidents
                        SET status = ?, updated_at = ?
                        WHERE incident_id = ?;
                        """,
                        (new_status, now_str, incident_id),
                    )
                return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.error("Failed to update status for incident %s: %s", incident_id, e)
            return False

    def get_incident_stats(self) -> Dict[str, int]:
        """Calculates incident counts by status and severity."""
        stats = {
            "total": 0,
            "open": 0,
            "acknowledged": 0,
            "resolved": 0,
            "ignored": 0,
            "critical": 0,
            "high": 0,
        }
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT status, severity, COUNT(*) as cnt
                    FROM incidents
                    GROUP BY status, severity;
                    """
                )
                rows = cursor.fetchall()
                for r in rows:
                    cnt = int(r["cnt"])
                    st = r["status"].lower()
                    sev = r["severity"].lower()

                    stats["total"] += cnt
                    if st in stats:
                        stats[st] += cnt
                    if sev in stats:
                        stats[sev] += cnt
        except sqlite3.Error as e:
            logger.error("Failed to compute incident statistics: %s", e)

        return stats

    def clear_all_incidents(self) -> bool:
        """Clears all records from the incidents table (primarily for testing)."""
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM incidents;")
                return True
        except sqlite3.Error as e:
            logger.error("Failed to clear incidents: %s", e)
            return False
