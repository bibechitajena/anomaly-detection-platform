"""
Offline Telemetry Log Analyzer Module.
======================================
Provides safe parsing, validation, column mapping, and offline anomaly detection
for uploaded telemetry CSV datasets without corrupting live streaming or
overwriting active monitoring history.

Features:
- Configurable maximum upload size (default: 20 MB).
- Empty file and invalid format rejection.
- Automated and configurable timestamp parsing with duplicate/missing handling.
- Graceful missing value, NaN, and malformed row imputation.
- Flexible column mapping to supported telemetry metrics (CPU, Memory, Latency, Requests, Error Rate).
- Dual detection baseline modes:
    1. 'unsupervised': Fits Isolation Forest directly on the uploaded dataset.
    2. 'operational_baseline': Compares against platform healthy operational baseline.
- Isolated CSV export helpers (Validated data, Analyzed results, Anomalies only).
- Explicit, user-driven promotion of offline detections to the SQLite incident store.
"""

from datetime import datetime, timedelta
import io
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from detector import AnomalyDetector
from incident_store import IncidentStore

logger = logging.getLogger(__name__)

# Max allowed upload size: 20 MB
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024

# Supported target telemetry metrics with human-readable labels
SUPPORTED_METRIC_TARGETS: Dict[str, str] = {
    "cpu_usage": "CPU Utilization (%)",
    "memory_usage": "Memory Utilization (%)",
    "response_time": "Response Latency (ms)",
    "request_count": "Request Count / Throughput (req/s)",
    "error_rate": "Error Rate (%)",
}

# Common column name heuristics for automatic timestamp discovery
TIMESTAMP_CANDIDATE_NAMES = [
    "timestamp",
    "time",
    "datetime",
    "date",
    "ts",
    "created_at",
    "event_time",
    "log_time",
]


def validate_and_parse_csv(
    file_or_content: Union[bytes, io.BytesIO, io.StringIO, str],
    column_mapping: Optional[Dict[str, str]] = None,
    timestamp_col: Optional[str] = None,
    max_size_bytes: int = MAX_UPLOAD_SIZE_BYTES,
) -> Tuple[Optional[pd.DataFrame], List[str], List[str]]:
    """Safely validates, parses, and cleans an uploaded telemetry CSV.

    Parameters:
        file_or_content: File bytes, StringIO, or raw CSV string.
        column_mapping (dict, optional): Mapping of CSV column name -> target metric name.
        timestamp_col (str, optional): Explicit name of the timestamp column in the CSV.
        max_size_bytes (int): Maximum allowable upload size in bytes.

    Returns:
        Tuple[Optional[pd.DataFrame], List[str], List[str]]:
            - Cleaned and validated DataFrame (None if critical errors occur).
            - List of non-fatal validation warnings.
            - List of critical errors causing validation failure.
    """
    warnings: List[str] = []
    errors: List[str] = []

    # 1. Check for empty input
    if file_or_content is None:
        errors.append("Uploaded file is empty or missing.")
        return None, warnings, errors

    # 2. Check byte size if bytes or BytesIO
    raw_bytes: Optional[bytes] = None
    if isinstance(file_or_content, bytes):
        raw_bytes = file_or_content
    elif isinstance(file_or_content, io.BytesIO):
        raw_bytes = file_or_content.getvalue()
    elif isinstance(file_or_content, str):
        raw_bytes = file_or_content.encode("utf-8")

    if raw_bytes is not None:
        if len(raw_bytes) == 0:
            errors.append("Uploaded file is 0 bytes (empty).")
            return None, warnings, errors
        if len(raw_bytes) > max_size_bytes:
            errors.append(
                f"File size ({len(raw_bytes) / (1024*1024):.1f} MB) exceeds maximum allowed limit of {max_size_bytes / (1024*1024):.0f} MB."
            )
            return None, warnings, errors

    # 3. Parse CSV safely with pandas
    df: Optional[pd.DataFrame] = None
    parse_stream = io.BytesIO(raw_bytes) if raw_bytes is not None else file_or_content

    # Try utf-8 first, fallback to latin-1
    for encoding in ["utf-8", "latin-1"]:
        try:
            if isinstance(parse_stream, io.BytesIO):
                parse_stream.seek(0)
            df = pd.read_csv(parse_stream, encoding=encoding)
            break
        except pd.errors.EmptyDataError:
            errors.append("CSV file contains no readable data or header.")
            return None, warnings, errors
        except Exception as e:
            if encoding == "latin-1":
                errors.append(f"Failed to parse CSV file: {str(e)}")
                return None, warnings, errors

    if df is None or df.empty:
        errors.append("Parsed CSV contains 0 data rows.")
        return None, warnings, errors

    # Strip whitespace from column headers
    df.columns = [str(c).strip() for c in df.columns]

    # 4. Identify and clean timestamp column
    parsed_timestamps: Optional[pd.Series] = None
    target_ts_col = timestamp_col

    if not target_ts_col:
        for candidate in TIMESTAMP_CANDIDATE_NAMES:
            for col in df.columns:
                if col.lower() == candidate:
                    target_ts_col = col
                    break
            if target_ts_col:
                break

    if target_ts_col and target_ts_col in df.columns:
        converted = pd.to_datetime(df[target_ts_col], errors="coerce")
        nat_count = int(converted.isna().sum())
        if nat_count == len(df):
            warnings.append(
                f"Column '{target_ts_col}' could not be parsed as valid timestamps. Generated sequential timestamps instead."
            )
        else:
            if nat_count > 0:
                warnings.append(
                    f"Found {nat_count} unparseable timestamp values in '{target_ts_col}'. Forward-filled missing timestamps."
                )
                converted = converted.ffill().bfill()
            parsed_timestamps = converted
    else:
        warnings.append(
            "No timestamp column identified. Generated sequential 10-second interval timestamps."
        )

    # Fallback timestamp generation if absent or unparseable
    if parsed_timestamps is None:
        now = datetime.now()
        n_rows = len(df)
        base_time = now - timedelta(seconds=(n_rows - 1) * 10)
        parsed_timestamps = pd.Series(
            [base_time + timedelta(seconds=i * 10) for i in range(n_rows)],
            index=df.index,
        )

    # Handle duplicate timestamps by adding monotonic millisecond offsets
    if parsed_timestamps.duplicated().any():
        dup_count = int(parsed_timestamps.duplicated().sum())
        warnings.append(
            f"Found {dup_count} duplicate timestamp records; applied sequential millisecond offsets to guarantee strict chronological ordering."
        )
        cum_dups = parsed_timestamps.groupby(parsed_timestamps).cumcount()
        parsed_timestamps = parsed_timestamps + pd.to_timedelta(cum_dups * 10, unit="ms")

    cleaned_df = pd.DataFrame({"timestamp": parsed_timestamps})

    # 5. Process and validate metric columns
    active_mapping = column_mapping or {}
    valid_mapped_metrics = 0

    for csv_col, target_metric in active_mapping.items():
        if csv_col not in df.columns:
            warnings.append(
                f"Mapped column '{csv_col}' not found in CSV. Skipped."
            )
            continue

        # Convert to numeric, coercing errors to NaN
        raw_series = pd.to_numeric(df[csv_col], errors="coerce")
        nan_count = int(raw_series.isna().sum())

        if nan_count == len(df):
            errors.append(
                f"Mapped metric '{target_metric}' (column '{csv_col}') contains no valid numeric data."
            )
            continue

        if nan_count > 0:
            median_val = float(raw_series.median())
            raw_series = raw_series.ffill().bfill().fillna(median_val)
            warnings.append(
                f"Column '{csv_col}' had {nan_count} non-numeric/missing values; repaired using forward fill and median."
            )

        cleaned_df[target_metric] = np.round(raw_series, 2)
        valid_mapped_metrics += 1

    # If no mapping was provided, automatically detect numeric columns
    if not active_mapping:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            # Attempt to convert object columns to numeric
            for col in df.columns:
                if col == target_ts_col:
                    continue
                coerced = pd.to_numeric(df[col], errors="coerce")
                if coerced.notna().sum() > len(df) * 0.5:
                    numeric_cols.append(col)

        if not numeric_cols:
            errors.append("No numeric metric columns could be automatically identified in the CSV.")
            return None, warnings, errors

        for col in numeric_cols:
            target_name = col.lower().replace(" ", "_")
            raw_num = pd.to_numeric(df[col], errors="coerce").ffill().bfill()
            if raw_num.notna().sum() > 0:
                cleaned_df[target_name] = np.round(raw_num.fillna(0.0), 2)
                valid_mapped_metrics += 1

    if valid_mapped_metrics == 0:
        errors.append("At least one valid numeric metric column must be mapped and verified.")
        return None, warnings, errors

    # Sort strictly by timestamp ascending
    cleaned_df = cleaned_df.sort_values(by="timestamp").reset_index(drop=True)

    return cleaned_df, warnings, errors


def generate_custom_baseline(
    features: List[str], n_samples: int = 500, random_seed: int = 42
) -> pd.DataFrame:
    """Generates a synthetic baseline dataset matching the requested feature names."""
    rng = np.random.default_rng(random_seed)
    data: Dict[str, np.ndarray] = {}

    for f in features:
        if f == "cpu_usage":
            vals = rng.normal(loc=38.0, scale=7.0, size=n_samples)
            vals = np.clip(vals, 12.0, 65.0)
        elif f == "memory_usage":
            vals = rng.normal(loc=52.0, scale=5.0, size=n_samples)
            vals = np.clip(vals, 35.0, 72.0)
        elif f == "response_time":
            vals = rng.normal(loc=110.0, scale=20.0, size=n_samples)
            vals = np.clip(vals, 50.0, 220.0)
        elif f == "request_count":
            vals = rng.normal(loc=220.0, scale=35.0, size=n_samples)
            vals = np.clip(vals, 100.0, 400.0)
        elif f == "error_rate":
            vals = rng.normal(loc=0.8, scale=0.3, size=n_samples)
            vals = np.clip(vals, 0.05, 3.0)
        else:
            vals = rng.normal(loc=50.0, scale=10.0, size=n_samples)
            vals = np.clip(vals, 0.0, 100.0)

        data[f] = np.round(vals, 2)

    return pd.DataFrame(data)


def run_offline_detection(
    cleaned_df: pd.DataFrame,
    features: List[str],
    contamination: float = 0.04,
    z_threshold: float = 2.5,
    baseline_mode: str = "unsupervised",
    random_seed: int = 42,
) -> pd.DataFrame:
    """Executes anomaly detection on a cleaned telemetry dataset.

    Parameters:
        cleaned_df (pd.DataFrame): Validated telemetry DataFrame.
        features (List[str]): List of metric feature column names to evaluate.
        contamination (float): Expected outlier ratio (0.01 - 0.15).
        z_threshold (float): Z-score deviation trigger for diagnostic hints.
        baseline_mode (str):
            - 'unsupervised': Fits Isolation Forest directly on the uploaded dataset.
            - 'operational_baseline': Compares against platform healthy operational baseline.
        random_seed (int): Reproducibility seed.

    Returns:
        pd.DataFrame: Enriched dataset containing anomaly flags, severities,
                      risk scores, and diagnostic explanations.
    """
    if cleaned_df.empty or not features:
        return cleaned_df.copy()

    detector = AnomalyDetector(
        features=features,
        contamination=contamination,
        random_state=random_seed,
        z_threshold=z_threshold,
    )

    if baseline_mode == "operational_baseline":
        baseline_df = generate_custom_baseline(features=features, n_samples=500, random_seed=random_seed)
        detector.train(baseline_df)
    else:  # unsupervised mode
        # In unsupervised mode, Isolation Forest is fitted directly on the uploaded features
        detector.train(cleaned_df[features].copy())

    # Execute predictions across all records
    analyzed_df = detector.predict_dataframe(cleaned_df)
    return analyzed_df


def save_offline_incidents_to_store(
    analyzed_df: pd.DataFrame,
    incident_store: IncidentStore,
    selected_indices: Optional[List[int]] = None,
) -> int:
    """Explicitly persists selected offline anomaly detections to the SQLite incident database.

    Guarantees requirement 11: Uploaded historical data does not silently pollute
    the live incident database; it is only saved when an operator explicitly requests it.

    Parameters:
        analyzed_df (pd.DataFrame): The analyzed telemetry DataFrame.
        incident_store (IncidentStore): Active incident database instance.
        selected_indices (List[int], optional): Indices of specific rows to persist.
                                                If None, all flagged anomalies are saved.

    Returns:
        int: Number of incidents recorded/updated.
    """
    if analyzed_df.empty or "is_anomaly" not in analyzed_df.columns:
        return 0

    target_rows = analyzed_df[analyzed_df["is_anomaly"]]
    if selected_indices is not None:
        target_rows = target_rows.loc[target_rows.index.intersection(selected_indices)]

    count = 0
    for idx, row in target_rows.iterrows():
        obs_dict = row.to_dict()
        incident_store.record_anomaly(obs_dict, detector_name="IsolationForest-Offline")
        count += 1

    return count
