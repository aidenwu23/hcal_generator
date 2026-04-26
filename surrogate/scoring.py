"""
Shared scoring helpers for BO proposal ranking and best-observed selection.
Reads separate calibration files for the baseline anchor and the upper reference anchor.
"""

from __future__ import annotations

import csv
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
TOP_TAIL_SLOPE = 0.2


def safe_eval_expr(expr: str, local_vars: dict[str, object]) -> float:
    # Evaluate simple configured score expressions using only row values.
    return float(eval(expr, {"__builtins__": {}}, local_vars))


def _parse_float(value: object) -> float | None:
    # Convert CSV-style values into floats and treat blanks as missing.
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    value_text = str(value).strip()
    if not value_text:
        return None
    try:
        return float(value_text)
    except ValueError:
        return None


def _quantile(values: list[float], fraction: float) -> float:
    # Compute a linearly interpolated quantile for the normalization anchor.
    if not values:
        raise ValueError("Cannot compute a quantile from an empty value list.")
    if fraction <= 0.0:
        return min(values)
    if fraction >= 1.0:
        return max(values)

    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * fraction
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    if lower_index == upper_index:
        return lower_value
    weight = position - lower_index
    return lower_value + weight * (upper_value - lower_value)


def _clip01(value: float) -> float:
    # Clamp a value to the unit interval.
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def _normalize_metric(value: float, baseline_value: float, upper_value: float) -> float:
    # Map a metric to baseline=0 and upper anchor=1, with a weak tail above 1.
    denominator = upper_value - baseline_value
    if denominator <= 0.0:
        return 1.0 if value > baseline_value else 0.0
    normalized_value = (value - baseline_value) / denominator
    if normalized_value <= 0.0:
        return 0.0
    if normalized_value <= 1.0:
        return normalized_value
    return 1.0 + TOP_TAIL_SLOPE * (normalized_value - 1.0)


def _normalized_metric_name(metric_name: str) -> str:
    # Name the derived diagnostic column for a normalized metric.
    return f"{metric_name}_normalized"


@lru_cache(maxsize=None)
def _load_metric_values(
    csv_path_text: str,
    metric_names_key: tuple[str, ...],
) -> dict[str, list[float]]:
    # Read one CSV and collect the values available for each metric.
    csv_path = Path(csv_path_text)
    if not csv_path.exists():
        raise FileNotFoundError(f"Normalization CSV not found: {csv_path}")

    metric_values: dict[str, list[float]] = {metric_name: [] for metric_name in metric_names_key}

    with csv_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} does not contain a CSV header.")
        missing_columns = [
            column_name
            for column_name in metric_names_key
            if column_name not in reader.fieldnames
        ]
        if missing_columns:
            raise ValueError(f"{csv_path} is missing normalization columns: {missing_columns}")

        # Skip missing values so partially filled rows do not poison other metrics.
        for row in reader:
            for metric_name in metric_names_key:
                parsed_value = _parse_float(row.get(metric_name))
                if parsed_value is None:
                    continue
                metric_values[metric_name].append(parsed_value)

    return metric_values


@lru_cache(maxsize=None)
def _load_baseline_values(
    baseline_csv_path_text: str,
    baseline_geometry_id: str,
    metric_names_key: tuple[str, ...],
) -> dict[str, float]:
    # Read the baseline CSV and keep only the rows for the baseline geometry.
    baseline_csv_path = Path(baseline_csv_path_text)
    if not baseline_csv_path.exists():
        raise FileNotFoundError(f"Baseline CSV not found: {baseline_csv_path}")

    baseline_values: dict[str, list[float]] = {metric_name: [] for metric_name in metric_names_key}

    # Multiple baseline rows are averaged to reduce run-to-run noise in the anchor.
    with baseline_csv_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"{baseline_csv_path} does not contain a CSV header.")
        missing_columns = [
            column_name
            for column_name in ("geometry_id", *metric_names_key)
            if column_name not in reader.fieldnames
        ]
        if missing_columns:
            raise ValueError(f"{baseline_csv_path} is missing baseline columns: {missing_columns}")

        for row in reader:
            geometry_id = str(row.get("geometry_id", "")).strip()
            if geometry_id != baseline_geometry_id:
                continue
            for metric_name in metric_names_key:
                parsed_value = _parse_float(row.get(metric_name))
                if parsed_value is not None:
                    baseline_values[metric_name].append(parsed_value)

    # Build one fixed anchor pair per metric.
    baseline_means: dict[str, float] = {}
    for metric_name in metric_names_key:
        if not baseline_values[metric_name]:
            raise ValueError(
                f"Baseline geometry {baseline_geometry_id!r} has no values for {metric_name!r} "
                f"in {baseline_csv_path}."
            )
        baseline_means[metric_name] = (
            sum(baseline_values[metric_name]) / len(baseline_values[metric_name])
        )

    return baseline_means


def _parse_normalized_weighted_metrics(scoring: dict[str, Any]) -> list[dict[str, Any]]:
    # Read the metric columns and weights from the scoring block.
    metrics = scoring.get("metrics", []) or []
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("normalized_weighted scoring requires a non-empty scoring.metrics list.")

    parsed_metrics: list[dict[str, Any]] = []
    for metric in metrics:
        if not isinstance(metric, dict):
            raise ValueError("Each normalized_weighted metric entry must be a mapping.")
        column_name = str(metric.get("column", "")).strip()
        if not column_name:
            raise ValueError("Each normalized_weighted metric entry must define column.")
        parsed_metrics.append({
            "column": column_name,
            "weight": float(metric.get("weight", 1.0)),
        })
    return parsed_metrics


def _load_normalized_weighted_anchors(
    scoring: dict[str, Any],
    relative_to: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    # Resolve the scoring block into metric definitions and fixed normalization anchors.
    normalization = scoring.get("normalization", {}) or {}
    if not isinstance(normalization, dict):
        raise ValueError("normalized_weighted scoring requires scoring.normalization to be a mapping.")

    baseline_csv_text = str(normalization.get("baseline_csv", "")).strip()
    reference_csv_text = str(normalization.get("reference_csv", "")).strip()
    baseline_geometry_id = str(normalization.get("baseline_geometry_id", "")).strip()
    percentile = float(normalization.get("percentile", 0.9))

    if not baseline_csv_text:
        raise ValueError("normalized_weighted scoring requires scoring.normalization.baseline_csv.")
    if not reference_csv_text:
        raise ValueError("normalized_weighted scoring requires scoring.normalization.reference_csv.")
    if not baseline_geometry_id:
        raise ValueError("normalized_weighted scoring requires scoring.normalization.baseline_geometry_id.")

    # Configured paths are project-root relative unless already absolute.
    baseline_csv_path = Path(baseline_csv_text)
    if not baseline_csv_path.is_absolute():
        baseline_csv_path = (PROJECT_DIRECTORY / baseline_csv_path).resolve()

    reference_csv_path = Path(reference_csv_text)
    if not reference_csv_path.is_absolute():
        reference_csv_path = (PROJECT_DIRECTORY / reference_csv_path).resolve()

    parsed_metrics = _parse_normalized_weighted_metrics(scoring)
    metric_names_key = tuple(metric["column"] for metric in parsed_metrics)
    baseline_values = _load_baseline_values(
        str(baseline_csv_path),
        baseline_geometry_id,
        metric_names_key,
    )
    reference_values = _load_metric_values(str(reference_csv_path), metric_names_key)

    anchors: dict[str, dict[str, float]] = {}
    # Use the baseline as zero and the configured reference percentile as one.
    for metric_name in metric_names_key:
        if not reference_values[metric_name]:
            raise ValueError(
                f"Reference CSV {reference_csv_path} has no values for {metric_name!r}."
            )
        anchors[metric_name] = {
            "baseline": baseline_values[metric_name],
            "upper": _quantile(reference_values[metric_name], percentile),
        }
    return parsed_metrics, anchors


def score_row(
    row_values: dict[str, object],
    scoring: dict[str, Any],
    relative_to: Path,
) -> tuple[float, dict[str, float]]:
    # Score one observed row and return any derived normalized metric columns.
    mode = str(scoring.get("mode", "metric")).lower()

    if mode == "normalized_weighted":
        parsed_metrics, anchors = _load_normalized_weighted_anchors(scoring, relative_to)
        scored_values: dict[str, float] = {}
        total_score = 0.0

        # Normalize each configured metric onto the shared 0 to 1 scale before weighting.
        for metric in parsed_metrics:
            column_name = metric["column"]
            parsed_value = _parse_float(row_values.get(column_name))
            if parsed_value is None:
                raise ValueError(f"Missing value for normalized scoring metric {column_name!r}.")
            normalized_value = _normalize_metric(
                parsed_value,
                anchors[column_name]["baseline"],
                anchors[column_name]["upper"],
            )
            scored_values[_normalized_metric_name(column_name)] = normalized_value
            total_score += float(metric["weight"]) * normalized_value
        if not bool(scoring.get("maximize", True)):
            total_score = -total_score
        return total_score, scored_values

    if mode == "tradeoff":
        # Tradeoff mode supports a configured arithmetic expression over row columns.
        expr = scoring.get("expr")
        if not expr:
            raise ValueError("scoring.mode=tradeoff requires scoring.expr.")
        return safe_eval_expr(str(expr), row_values), {}

    # Metric mode scores directly from one configured column.
    metric_name = str(scoring.get("metric", "")).strip()
    if not metric_name:
        raise ValueError("scoring.mode=metric requires scoring.metric.")
    metric_value = _parse_float(row_values.get(metric_name))
    if metric_value is None:
        raise ValueError(f"Missing value for scoring.metric {metric_name!r}.")
    maximize = bool(scoring.get("maximize", True))
    if not maximize:
        metric_value = -metric_value
    return metric_value, {}


def score_prediction_dict(
    pred: dict[str, Any],
    scoring: dict[str, Any],
    relative_to: Path,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    # Score a surrogate prediction batch and keep any derived normalized metric arrays.
    mode = str(scoring.get("mode", "metric")).lower()
    extra_outputs: dict[str, Any] = {}

    if mode == "normalized_weighted":
        parsed_metrics, anchors = _load_normalized_weighted_anchors(scoring, relative_to)
        if not parsed_metrics:
            raise ValueError("normalized_weighted scoring requires at least one metric.")

        candidate_count = len(next(iter(pred.values())))
        scores = np.zeros((candidate_count,), dtype=float)

        # Normalize each predicted metric against the fixed anchors before weighting.
        for metric in parsed_metrics:
            column_name = metric["column"]
            if column_name not in pred:
                raise ValueError(
                    f"Predicted targets do not include {column_name!r} required by normalized_weighted scoring."
                )
            baseline_value = anchors[column_name]["baseline"]
            upper_value = anchors[column_name]["upper"]
            denominator = upper_value - baseline_value
            values = pred[column_name].astype(float)
            if denominator <= 0.0:
                normalized_values = (values > baseline_value).astype(float)
            else:
                normalized_values = np.clip((values - baseline_value) / denominator, 0.0, 1.0)
            extra_outputs[_normalized_metric_name(column_name)] = normalized_values
            scores += float(metric["weight"]) * normalized_values
        if not bool(scoring.get("maximize", True)):
            scores = -scores
        return scores, extra_outputs

    if mode == "tradeoff":
        # Evaluate the configured expression once per predicted candidate.
        expr = scoring.get("expr")
        if not expr:
            raise ValueError("scoring.mode=tradeoff requires scoring.expr.")
        scores = np.zeros((len(next(iter(pred.values()))),), dtype=float)
        for index in range(scores.shape[0]):
            row_values = {key: float(values[index]) for key, values in pred.items()}
            scores[index] = safe_eval_expr(str(expr), row_values)
        if not bool(scoring.get("maximize", True)):
            scores = -scores
        return scores, extra_outputs

    # Metric mode uses one predicted target as the score array.
    metric_name = str(scoring.get("metric", "")).strip()
    if not metric_name:
        raise ValueError("scoring.mode=metric requires scoring.metric.")
    if metric_name not in pred:
        raise ValueError(f"Unknown scoring.metric {metric_name!r}. Available: {list(pred.keys())}")
    scores = pred[metric_name].astype(float).copy()
    if not bool(scoring.get("maximize", True)):
        scores = -scores
    return scores, extra_outputs
