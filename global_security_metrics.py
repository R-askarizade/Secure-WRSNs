"""Common security metrics for the Real-World WRSN comparison.

This module keeps the method-independent final-classification ground truth used
by the global runner, but delegates ALL IDS metric formulas to
``MetricsCollector.calculate_ids_metrics()``.

Why this matters
----------------
* Baseline implementations are not modified.
* TP/TN/FP/FN are built from the shared final attacker-vs-blacklist comparison.
* TPR/FPR/F1/Accuracy/MCC and zero-denominator behavior are exactly those of
  ``metric_collector.py``.
* Common ``..._%`` columns are presentation aliases only: native collector
  fractions multiplied by 100. They do not have independent formulas.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from metric_collector import MetricsCollector


COMMON_SECURITY_METRICS_VERSION = (
    "WRSN_Global_Common_Security_Metrics_v2_MetricsCollectorFormula_FinalClassification"
)

LEGACY_SECURITY_METRICS_VERSIONS = frozenset({
    "WRSN_Global_Common_Security_Metrics_v1_FinalClassification",
})

# Keep the thesis-facing percentage columns stable for downstream tables.
# Their values are direct *100 aliases of MetricsCollector's native [0,1] rates.
PRIMARY_SECURITY_COMPARISON_METRICS = (
    "Common_Security_TPR_%",
    "Common_Security_FPR_%",
    "Common_Security_Precision_%",
    "Common_Security_F1_%",
    "Common_Security_Accuracy_%",
    "Common_Security_Specificity_%",
    "Common_Security_Balanced_Accuracy_%",
    "Common_Security_MCC",
)

# Native MetricsCollector columns are exported unchanged so the security CSV
# can be checked directly against MetricsCollector.export_summary().
NATIVE_IDS_METRIC_COLUMNS = (
    "TP",
    "TN",
    "FP",
    "FN",
    "TPR",
    "Recall",
    "FPR",
    "TNR",
    "Specificity",
    "FNR",
    "Accuracy",
    "Balanced_Accuracy",
    "Precision",
    "NPV",
    "F1_Score",
    "MCC",
)

SECURITY_METRIC_COLUMNS = (
    "Security_Total_Sensor_Nodes",
    "Security_Actual_Attackers",
    "Security_Actual_Benign",
    "Security_Detected_Blacklisted",
    *NATIVE_IDS_METRIC_COLUMNS,
    "Common_Security_TPR_%",
    "Common_Security_FNR_%",
    "Common_Security_FPR_%",
    "Common_Security_Specificity_%",
    "Common_Security_Precision_%",
    "Common_Security_NPV_%",
    "Common_Security_F1_%",
    "Common_Security_Accuracy_%",
    "Common_Security_Balanced_Accuracy_%",
    "Common_Security_MCC",
)

_NATIVE_RATE_COLUMNS = {
    "TPR",
    "Recall",
    "FPR",
    "TNR",
    "Specificity",
    "FNR",
    "Accuracy",
    "Balanced_Accuracy",
    "Precision",
    "NPV",
    "F1_Score",
}

_PERCENT_COLUMNS = {
    "Common_Security_TPR_%",
    "Common_Security_FNR_%",
    "Common_Security_FPR_%",
    "Common_Security_Specificity_%",
    "Common_Security_Precision_%",
    "Common_Security_NPV_%",
    "Common_Security_F1_%",
    "Common_Security_Accuracy_%",
    "Common_Security_Balanced_Accuracy_%",
}


def _detected_blacklisted_nodes(simulator) -> set[int]:
    """Read only the method's final BLACKLISTED decision set."""
    bs = getattr(simulator, "bs", None)
    bs_id = int(getattr(simulator, "BS_ID", 0))
    if bs is None:
        return set()

    getter = getattr(bs, "get_blacklisted_nodes", None)
    if callable(getter):
        try:
            return {
                int(node_id)
                for node_id in getter()
                if int(node_id) != bs_id
            }
        except Exception:
            pass

    detected = set()
    for node_id, state in (getattr(bs, "node_states", {}) or {}).items():
        try:
            nid = int(node_id)
        except (TypeError, ValueError):
            continue
        if nid == bs_id:
            continue
        if str((state or {}).get("status", "")).strip().upper() == "BLACKLISTED":
            detected.add(nid)
    return detected


def _calculate_ids_with_metric_collector(
    *, tp: int, tn: int, fp: int, fn: int
) -> Dict[str, Any]:
    """Use the exact MetricsCollector implementation as the formula source.

    A temporary collector is intentional.  It prevents the common evaluator
    from mutating the completed simulator's native metrics while guaranteeing
    identical formulas, including MetricsCollector's zero-denominator and MCC
    policies.
    """
    total = int(tp) + int(tn) + int(fp) + int(fn)
    collector = MetricsCollector(total_nodes=total, total_rounds_or_time=0.0)
    collector.update_ids_matrix(
        int(tp),
        int(tn),
        int(fp),
        int(fn),
    )
    return dict(collector.calculate_ids_metrics())


def _pct(native_value: Any) -> float:
    """Presentation-only conversion from MetricsCollector fraction to percent."""
    return float(native_value) * 100.0


def export_common_security_row(
    simulator,
    *,
    seed: int,
    method_name: str,
    scenario_name: str,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Export one common final-classification security row.

    Classification counts are method-independent:
      actual positive   = simulator.malicious_nodes
      predicted positive = final BLACKLISTED set

    All rate formulas then come directly from
    MetricsCollector.calculate_ids_metrics().
    """
    bs_id = int(getattr(simulator, "BS_ID", 0))
    nodes = getattr(simulator, "nodes", {}) or {}
    universe = {
        int(node_id)
        for node_id in nodes.keys()
        if int(node_id) != bs_id
    }

    actual_attackers = {
        int(node_id)
        for node_id in (getattr(simulator, "malicious_nodes", []) or [])
        if int(node_id) != bs_id and int(node_id) in universe
    }
    detected_attackers = _detected_blacklisted_nodes(simulator) & universe

    actual_benign = universe - actual_attackers
    predicted_benign = universe - detected_attackers

    tp = len(actual_attackers & detected_attackers)
    fn = len(actual_attackers & predicted_benign)
    fp = len(actual_benign & detected_attackers)
    tn = len(actual_benign & predicted_benign)

    ids = _calculate_ids_with_metric_collector(
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
    )

    row: Dict[str, Any] = {
        "Common_Security_Metrics_Version": COMMON_SECURITY_METRICS_VERSION,
        "Method": str(method_name),
        "Scenario": str(scenario_name),
        "Seed": int(seed),
        "Common_Security_Ground_Truth": "simulator.malicious_nodes",
        "Common_Security_Prediction_Definition": "final_BLACKLISTED_node_set",
        "Common_Security_Final_Classification_Flag": 1,
        "Common_Security_Formula_Source": (
            "metric_collector.py::MetricsCollector.calculate_ids_metrics"
        ),
        "Security_Total_Sensor_Nodes": int(len(universe)),
        "Security_Actual_Attackers": int(len(actual_attackers)),
        "Security_Actual_Benign": int(len(actual_benign)),
        "Security_Detected_Blacklisted": int(len(detected_attackers)),

        # Native MetricsCollector names/values: no reimplementation.
        "TP": int(ids["TP"]),
        "TN": int(ids["TN"]),
        "FP": int(ids["FP"]),
        "FN": int(ids["FN"]),
        "Confusion_Matrix": str(ids["Confusion_Matrix"]),
        "TPR": float(ids["TPR"]),
        "Recall": float(ids["Recall"]),
        "FPR": float(ids["FPR"]),
        "TNR": float(ids["TNR"]),
        "Specificity": float(ids["Specificity"]),
        "FNR": float(ids["FNR"]),
        "Accuracy": float(ids["Accuracy"]),
        "Balanced_Accuracy": float(ids["Balanced_Accuracy"]),
        "Precision": float(ids["Precision"]),
        "NPV": float(ids["NPV"]),
        "F1_Score": float(ids["F1_Score"]),
        "MCC": float(ids["MCC"]),

        # Backward-compatible count aliases for existing analysis scripts.
        "Security_TP": int(ids["TP"]),
        "Security_TN": int(ids["TN"]),
        "Security_FP": int(ids["FP"]),
        "Security_FN": int(ids["FN"]),

        # Thesis-facing percent aliases. Formula remains MetricsCollector's.
        "Common_Security_TPR_%": _pct(ids["TPR"]),
        "Common_Security_FNR_%": _pct(ids["FNR"]),
        "Common_Security_FPR_%": _pct(ids["FPR"]),
        "Common_Security_Specificity_%": _pct(ids["Specificity"]),
        "Common_Security_Precision_%": _pct(ids["Precision"]),
        "Common_Security_NPV_%": _pct(ids["NPV"]),
        "Common_Security_F1_%": _pct(ids["F1_Score"]),
        "Common_Security_Accuracy_%": _pct(ids["Accuracy"]),
        "Common_Security_Balanced_Accuracy_%": _pct(ids["Balanced_Accuracy"]),
        "Common_Security_MCC": float(ids["MCC"]),
    }

    if extra_metadata:
        row.update({str(k): v for k, v in extra_metadata.items()})

    validate_common_security_row(row)
    return row



def upgrade_legacy_common_security_row(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Upgrade a saved v1 security row without rerunning the simulation.

    The v1 row already contains the final-classification confusion counts.
    Those counts are preserved exactly; only derived IDS metrics are rebuilt
    through MetricsCollector.calculate_ids_metrics().
    """
    version = str(record.get("Common_Security_Metrics_Version", ""))
    if version == COMMON_SECURITY_METRICS_VERSION:
        row = dict(record)
        validate_common_security_row(row)
        return row
    if version not in LEGACY_SECURITY_METRICS_VERSIONS:
        raise ValueError(f"Unsupported legacy security metric version: {version!r}")

    tp = int(record["Security_TP"])
    tn = int(record["Security_TN"])
    fp = int(record["Security_FP"])
    fn = int(record["Security_FN"])
    ids = _calculate_ids_with_metric_collector(tp=tp, tn=tn, fp=fp, fn=fn)

    row = dict(record)
    row.update({
        "Common_Security_Metrics_Version": COMMON_SECURITY_METRICS_VERSION,
        "Common_Security_Formula_Source": (
            "metric_collector.py::MetricsCollector.calculate_ids_metrics"
        ),
        "TP": int(ids["TP"]),
        "TN": int(ids["TN"]),
        "FP": int(ids["FP"]),
        "FN": int(ids["FN"]),
        "Confusion_Matrix": str(ids["Confusion_Matrix"]),
        "TPR": float(ids["TPR"]),
        "Recall": float(ids["Recall"]),
        "FPR": float(ids["FPR"]),
        "TNR": float(ids["TNR"]),
        "Specificity": float(ids["Specificity"]),
        "FNR": float(ids["FNR"]),
        "Accuracy": float(ids["Accuracy"]),
        "Balanced_Accuracy": float(ids["Balanced_Accuracy"]),
        "Precision": float(ids["Precision"]),
        "NPV": float(ids["NPV"]),
        "F1_Score": float(ids["F1_Score"]),
        "MCC": float(ids["MCC"]),
        "Security_TP": int(ids["TP"]),
        "Security_TN": int(ids["TN"]),
        "Security_FP": int(ids["FP"]),
        "Security_FN": int(ids["FN"]),
        "Common_Security_TPR_%": _pct(ids["TPR"]),
        "Common_Security_FNR_%": _pct(ids["FNR"]),
        "Common_Security_FPR_%": _pct(ids["FPR"]),
        "Common_Security_Specificity_%": _pct(ids["Specificity"]),
        "Common_Security_Precision_%": _pct(ids["Precision"]),
        "Common_Security_NPV_%": _pct(ids["NPV"]),
        "Common_Security_F1_%": _pct(ids["F1_Score"]),
        "Common_Security_Accuracy_%": _pct(ids["Accuracy"]),
        "Common_Security_Balanced_Accuracy_%": _pct(ids["Balanced_Accuracy"]),
        "Common_Security_MCC": float(ids["MCC"]),
    })
    validate_common_security_row(row)
    return row

def validate_common_security_row(row: Mapping[str, Any]) -> None:
    total = int(row["Security_Total_Sensor_Nodes"])
    tp = int(row["TP"])
    tn = int(row["TN"])
    fp = int(row["FP"])
    fn = int(row["FN"])
    attackers = int(row["Security_Actual_Attackers"])
    benign = int(row["Security_Actual_Benign"])

    if tp + tn + fp + fn != total:
        raise RuntimeError(
            "Security confusion matrix does not partition all sensor nodes: "
            f"TP={tp}, TN={tn}, FP={fp}, FN={fn}, total={total}"
        )
    if tp + fn != attackers:
        raise RuntimeError("TP+FN does not equal actual attacker count.")
    if tn + fp != benign:
        raise RuntimeError("TN+FP does not equal actual benign count.")

    for native_count, alias in (
        ("TP", "Security_TP"),
        ("TN", "Security_TN"),
        ("FP", "Security_FP"),
        ("FN", "Security_FN"),
    ):
        if int(row[native_count]) != int(row[alias]):
            raise RuntimeError(
                f"Security alias mismatch: {native_count} != {alias}"
            )

    for column in _NATIVE_RATE_COLUMNS:
        value = float(row[column])
        if not (0.0 - 1e-12 <= value <= 1.0 + 1e-12):
            raise RuntimeError(f"{column} outside [0,1]: {value}")

    mcc = float(row["MCC"])
    if not (-1.0 - 1e-12 <= mcc <= 1.0 + 1e-12):
        raise RuntimeError(f"MCC outside [-1,1]: {mcc}")

    # Prove the row still matches the exact collector implementation.
    expected = _calculate_ids_with_metric_collector(tp=tp, tn=tn, fp=fp, fn=fn)
    for column in NATIVE_IDS_METRIC_COLUMNS:
        actual = float(row[column])
        target = float(expected[column])
        if not math.isclose(actual, target, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(
                f"MetricCollector parity failure for {column}: "
                f"row={actual}, expected={target}"
            )

    percent_aliases = {
        "Common_Security_TPR_%": "TPR",
        "Common_Security_FNR_%": "FNR",
        "Common_Security_FPR_%": "FPR",
        "Common_Security_Specificity_%": "Specificity",
        "Common_Security_Precision_%": "Precision",
        "Common_Security_NPV_%": "NPV",
        "Common_Security_F1_%": "F1_Score",
        "Common_Security_Accuracy_%": "Accuracy",
        "Common_Security_Balanced_Accuracy_%": "Balanced_Accuracy",
    }
    for percent_column, native_column in percent_aliases.items():
        value = float(row[percent_column])
        expected_percent = float(row[native_column]) * 100.0
        if not math.isclose(value, expected_percent, rel_tol=0.0, abs_tol=1e-10):
            raise RuntimeError(
                f"Percent alias mismatch for {percent_column}: "
                f"{value} != {expected_percent}"
            )
        if not (-1e-9 <= value <= 100.0 + 1e-9):
            raise RuntimeError(f"{percent_column} outside [0,100]: {value}")

    if not math.isclose(
        float(row["Common_Security_MCC"]),
        float(row["MCC"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Common_Security_MCC is not identical to MCC.")


def _ci95_numeric(values: Iterable[Any]) -> Tuple[int, float, float, float]:
    arr = pd.to_numeric(
        pd.Series(list(values)), errors="coerce"
    ).dropna().to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return 0, math.nan, math.nan, math.nan

    mean = float(np.mean(arr))
    if n == 1:
        return 1, mean, mean, mean

    try:
        from scipy import stats

        sem = float(stats.sem(arr))
        if not math.isfinite(sem) or sem <= 0.0:
            return n, mean, mean, mean
        low, high = stats.t.interval(
            0.95,
            n - 1,
            loc=mean,
            scale=sem,
        )
        return n, mean, float(low), float(high)
    except Exception:
        std = float(np.std(arr, ddof=1))
        margin = 1.96 * std / math.sqrt(n)
        return n, mean, mean - margin, mean + margin


def summarize_common_security_results(
    df: pd.DataFrame,
    out_csv: Optional[Path | str] = None,
) -> pd.DataFrame:
    required = {"Method", "Seed", "Common_Security_Metrics_Version"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Security result dataframe missing: "
            + ", ".join(sorted(missing))
        )

    versions = set(
        df["Common_Security_Metrics_Version"].dropna().astype(str).unique()
    )
    if versions != {COMMON_SECURITY_METRICS_VERSION}:
        raise ValueError(
            f"Mixed security metric versions: {sorted(versions)}"
        )

    numeric_candidates = [
        column
        for column in SECURITY_METRIC_COLUMNS
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column])
    ]

    rows = []
    for method, group in df.groupby("Method", sort=False):
        row: Dict[str, Any] = {
            "Method": method,
            "N_Runs": int(len(group)),
            "Seed_Min": int(group["Seed"].min()),
            "Seed_Max": int(group["Seed"].max()),
            "Common_Security_Metrics_Version": COMMON_SECURITY_METRICS_VERSION,
        }

        for column in numeric_candidates:
            n, mean, low, high = _ci95_numeric(group[column])

            if column in _PERCENT_COLUMNS:
                mean = min(100.0, max(0.0, mean)) if math.isfinite(mean) else mean
                low = max(0.0, low) if math.isfinite(low) else low
                high = min(100.0, high) if math.isfinite(high) else high
            elif column in _NATIVE_RATE_COLUMNS:
                mean = min(1.0, max(0.0, mean)) if math.isfinite(mean) else mean
                low = max(0.0, low) if math.isfinite(low) else low
                high = min(1.0, high) if math.isfinite(high) else high
            elif column in {"MCC", "Common_Security_MCC"}:
                mean = min(1.0, max(-1.0, mean)) if math.isfinite(mean) else mean
                low = max(-1.0, low) if math.isfinite(low) else low
                high = min(1.0, high) if math.isfinite(high) else high

            row[f"{column}_N_Valid"] = n
            row[f"{column}_Mean"] = mean
            row[f"{column}_CI95_Low"] = low
            row[f"{column}_CI95_High"] = high

        rows.append(row)

    summary = pd.DataFrame(rows)
    if out_csv is not None:
        path = Path(out_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".tmp")
        summary.to_csv(tmp, index=False)
        os.replace(tmp, path)
    return summary


def select_primary_security_table(summary: pd.DataFrame) -> pd.DataFrame:
    columns = ["Method", "N_Runs"]
    for metric in PRIMARY_SECURITY_COMPARISON_METRICS:
        for suffix in ("_N_Valid", "_Mean", "_CI95_Low", "_CI95_High"):
            column = metric + suffix
            if column in summary.columns:
                columns.append(column)
    return summary.loc[:, columns].copy()
