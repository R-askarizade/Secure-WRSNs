"""
global_comparison_metrics.py
============================

Canonical, method-independent metric contract for fair WRSN baseline comparison.

Purpose
-------
This module is the ONLY place where the main cross-method comparison metrics
are defined and calculated. Individual methods may keep their own native
diagnostics, but the thesis comparison table should be produced from this
module.

Design principles
-----------------
1. No physical/scenario numerical constants are duplicated here.
2. Setup/bootstrap overhead is excluded from the main operational-overhead
   denominator.
3. Workload delivery is evaluated against the shared exogenous workload trace,
   so a method cannot improve its denominator merely because a sensor died and
   stopped generating packets.
4. "Critical workload" is taken from the shared workload trace, not from a
   method-specific internal criticality classifier.
5. Charging response/deadline metrics are measured by a common observer:
      native request -> first positive physical energy delivery.
   The evaluation deadline is injected from the canonical scenario; it is not
   duplicated as a number in this file.
   regardless of whether the delivery comes from a mobile charger, an RF
   cluster, or a fixed charger.
6. Undefined/not-applicable ratios are NaN, never silently converted to zero.
7. Method-specific diagnostics (QIFL, DQN, fuzzy, tomography, NJNP internal
   counters, etc.) are intentionally excluded from the main comparison schema.
8. FND/HND/LND are exported with censoring-aware restricted-time metrics.
9. The module does not modify routing, charging, security, deployment, learning,
   or any method decision. All hooks are observer-only.

Typical use in global_real_world.py
-----------------------------------
    from global_comparison_metrics import (
        attach_global_comparison_metrics,
        summarize_global_comparison_results,
    )

    simulator = spec.manager_class(...)
    common = attach_global_comparison_metrics(
        simulator,
        workload_trace_path=trace_path,
        method_name=spec.label,
        scenario_name=scenario_name,
    )

    simulator.run_simulation()

    row = common.export_row(seed=seed)

At the end:
    summary = summarize_global_comparison_results(df, summary_csv)

Future baseline adapters
------------------------
If a baseline does not use SimulatorManager._trace_packet_path for application
delivery, call:
    observer.record_workload_delivery(...)

If its native charging-request creation does not call
MetricsCollector.record_charging_request, call:
    observer.record_charging_request(node_id, time)

The current NJNP adapter is automatically observed through its
_njnp_sync_service_pool() return value, without changing NJNP decisions.
"""

from __future__ import annotations

import inspect
import json
import math
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


COMMON_METRICS_VERSION = "WRSN_Global_Common_Metrics_v3_ChargeStart"


# ============================================================================
# MAIN THESIS COMPARISON SCHEMA
# ============================================================================

PRIMARY_COMPARISON_METRICS = (
    # Application / QoS
    "Common_Workload_Delivery_Ratio_%",
    "Common_Critical_Workload_Delivery_Ratio_%",
    "Packet_Level_PDR_%",
    "Throughput_bps",
    "Common_Avg_EndToEnd_Delay_s",
    "Common_P95_EndToEnd_Delay_s",

    # Operational overhead -- setup/bootstrap excluded
    "Common_Operational_Control_Overhead_Packet_%",
    "Common_Operational_Control_Overhead_Bit_%",

    # Energy / lifetime
    "Mean_Residual_Sensor_Energy_End_J",
    "Residual_Sensor_Energy_Fraction_End_%",
    "Alive_Nodes_End",
    "Ever_Dead_Node_Rate_%",
    "Sensor_Operational_Availability_%",
    "FND_Restricted_s",
    "HND_Restricted_s",
    "LND_Restricted_s",

    # Charging / fleet outcome
    "Mobile_Charger_Count",
    "MC_Travel_Distance_m",
    "MC_Travel_Energy_J",
    "Total_Injected_Energy_To_Sensors_J",
    "MC_Total_Mission_Energy_J",
    "Total_Charge_Service_Events",
    "Common_Charging_First_Response_Avg_s",
    "Common_Charging_First_Response_P95_s",
    "Common_Charging_First_Response_Max_s",
    "Common_Charging_Request_Service_Rate_%",
    "Common_Charging_Deadline_Failure_Rate_%",

    # Computational implementation cost
    "Sensor_Execution_Time_ms",
    "BS_Execution_Time_ms",
)


METRIC_GROUPS = {
    "qos": (
        "Common_Workload_Events",
        "Common_Workload_Delivered_Events",
        "Common_Workload_Delivery_Ratio_%",
        "Common_Critical_Workload_Events",
        "Common_Critical_Workload_Delivered_Events",
        "Common_Critical_Workload_Delivery_Ratio_%",
        "Generated_Data_Packets",
        "Received_Data_Packets",
        "Packet_Level_PDR_%",
        "Generated_Critical_Packets_Native",
        "Received_Critical_Packets_Native",
        "Native_Critical_PDR_%",
        "Throughput_bps",
        "Common_Avg_EndToEnd_Delay_s",
        "Common_P95_EndToEnd_Delay_s",
        "Common_Max_EndToEnd_Delay_s",
    ),
    "overhead": (
        "Operational_Control_Packets",
        "Setup_Control_Packets_Excluded",
        "Common_Operational_Control_Overhead_Packet_%",
        "Common_Operational_Control_Bits",
        "Common_Setup_Control_Bits_Excluded",
        "Common_Generated_Data_Bits",
        "Common_Operational_Control_Overhead_Bit_%",
    ),
    "lifetime_energy": (
        "Alive_Nodes_End",
        "Ever_Dead_Node_Count",
        "Ever_Dead_Node_Rate_%",
        "Node_Revival_Count",
        "Sensor_Operational_Availability_%",
        "Mean_Residual_Sensor_Energy_End_J",
        "Total_Residual_Sensor_Energy_End_J",
        "Total_Sensor_Energy_Capacity_J",
        "Residual_Sensor_Energy_Fraction_End_%",
        "FND_Observed_s",
        "FND_Censored_Flag",
        "FND_Restricted_s",
        "HND_Observed_s",
        "HND_Censored_Flag",
        "HND_Restricted_s",
        "LND_Observed_s",
        "LND_Censored_Flag",
        "LND_Restricted_s",
    ),
    "charging": (
        "Mobile_Charger_Count",
        "MC_Travel_Distance_m",
        "MC_Travel_Energy_J",
        "MC_RF_Transmit_Energy_J",
        "MC_Contact_Charging_Energy_J",
        "MC_Total_Mission_Energy_J",
        "MC_Injected_Energy_J",
        "Fixed_Charger_Injected_Energy_J",
        "Total_Injected_Energy_To_Sensors_J",
        "MC_Useful_Energy_Efficiency",
        "MC_Charge_Events",
        "Fixed_Charger_Charge_Events",
        "Total_Charge_Service_Events",
        "Common_Charging_Requests",
        "Common_Charging_Requests_Served",
        "Common_Charging_Requests_Failed_Before_First_Response",
        "Common_Charging_Requests_Unresolved_End",
        "Common_Charging_Request_Service_Rate_%",
        "Common_Charging_Deadline_s",
        "Common_Charging_Deadline_Eligible_Requests",
        "Common_Charging_Deadline_Failures",
        "Common_Charging_Deadline_Censored_End",
        "Common_Charging_Deadline_Failure_Rate_%",
        "Common_Charging_First_Response_Avg_s",
        "Common_Charging_First_Response_P95_s",
        "Common_Charging_First_Response_Max_s",
        "MC_Injected_Energy_Per_Charge_Event_J",
        "MC_Travel_Per_Injected_Joule_m_per_J",
    ),
    "compute": (
        "Sensor_Execution_Time_ms",
        "BS_Execution_Time_ms",
        "Wall_Clock_Runtime_s",
    ),
}


# ============================================================================
# BASIC HELPERS
# ============================================================================

def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _safe_ratio(
    numerator: Any,
    denominator: Any,
    *,
    scale: float = 1.0,
    undefined: float = math.nan,
) -> float:
    try:
        numerator = float(numerator)
        denominator = float(denominator)
    except (TypeError, ValueError):
        return float(undefined)
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return float(undefined)
    if denominator <= 0.0:
        return float(undefined)
    return float(numerator / denominator * scale)


def _mean(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else math.nan


def _percentile(values: Sequence[float], q: float) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, q)) if arr.size else math.nan


def _max_or_nan(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if arr.size else math.nan


def _event_time_key(time_value: Any) -> float:
    # Replay times are read from JSON and then assigned directly to
    # SimulatorManager.current_time. Rounding only protects against harmless
    # floating formatting differences.
    return round(float(time_value), 9)


def _method_signature(method) -> inspect.Signature:
    return inspect.signature(method)


def _bound_arguments(method, args, kwargs) -> Dict[str, Any]:
    try:
        bound = _method_signature(method).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except Exception:
        return dict(kwargs)


# ============================================================================
# CENSORING-AWARE LIFETIME
# ============================================================================

def _lifetime_event(
    death_times: Sequence[float],
    rank: int,
    horizon: float,
) -> Tuple[float, int, float]:
    """Return observed time, censor flag, and restricted time.

    observed:
        NaN when the event did not occur within the evaluation horizon.
    censored:
        1 when not observed; 0 otherwise.
    restricted:
        min(T_event, horizon). If the event was not observed, this is horizon.
        The mean of this field across seeds is a horizon-restricted
        time-to-event quantity and does not suffer from the historical -1
        sentinel problem.
    """
    sorted_times = sorted(
        t for t in (_finite_float(v) for v in death_times)
        if math.isfinite(t) and t >= 0.0
    )
    if rank <= 0:
        raise ValueError("rank must be >= 1")
    if len(sorted_times) >= rank:
        observed = min(float(sorted_times[rank - 1]), float(horizon))
        return observed, 0, observed
    return math.nan, 1, float(horizon)


# ============================================================================
# SHARED WORKLOAD TRACE
# ============================================================================

@dataclass
class SharedWorkloadTruth:
    total_events: int
    critical_events: Optional[int]
    event_truth: MutableMapping[Tuple[float, int], deque]

    @classmethod
    def from_json(cls, path: Path | str) -> "SharedWorkloadTruth":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            events = list(json.load(handle))

        truth: MutableMapping[Tuple[float, int], deque] = defaultdict(deque)
        critical_known = True
        critical_count = 0

        for index, event in enumerate(events):
            if "time" not in event or "source_id" not in event:
                raise ValueError(
                    f"Workload event #{index} is missing time/source_id: {event!r}"
                )
            critical = event.get("critical", None)
            if critical is None:
                critical_known = False
            else:
                critical = bool(critical)
                critical_count += int(critical)

            key = (
                _event_time_key(event["time"]),
                int(event["source_id"]),
            )
            truth[key].append(
                {
                    "event_index": int(index),
                    "critical": critical,
                }
            )

        return cls(
            total_events=len(events),
            critical_events=(critical_count if critical_known else None),
            event_truth=truth,
        )

    def match(self, time_value: float, source_id: int) -> Optional[Dict[str, Any]]:
        key = (_event_time_key(time_value), int(source_id))
        queue = self.event_truth.get(key)
        if not queue:
            return None
        return dict(queue.popleft())


# ============================================================================
# COMMON OBSERVER
# ============================================================================

class GlobalComparisonMetricsObserver:
    """Observer-only common metric layer attached to one simulator run."""

    def __init__(
        self,
        simulator,
        *,
        workload_trace_path: Path | str,
        method_name: str,
        scenario_name: str,
        charging_deadline_s: Optional[float] = None,
    ) -> None:
        self.simulator = simulator
        self.metrics = simulator.metrics
        self.method_name = str(method_name)
        self.scenario_name = str(scenario_name)
        self.workload_trace_path = str(Path(workload_trace_path))
        self.workload = SharedWorkloadTruth.from_json(workload_trace_path)
        self.common_charging_deadline_s = (
            None
            if charging_deadline_s is None
            else max(0.0, float(charging_deadline_s))
        )

        self.workload_delivered = 0
        self.critical_workload_delivered = 0
        self.common_delays: list[float] = []
        self.workload_match_failures = 0

        self.operational_control_bits = 0.0
        self.setup_control_bits = 0.0
        self.generated_data_bits = 0.0

        # Request -> beginning of physical charging service.
        self.active_charging_requests: Dict[int, float] = {}
        self.charging_request_count = 0
        self.charging_request_served = 0
        self.charging_request_failed = 0
        self.charging_first_response_times: list[float] = []
        # Resolved deadline outcomes: True = deadline failure, False = success.
        # Requests still active at the end are classified in export_row:
        # age >= common deadline -> failure; younger -> right-censored.
        self.charging_deadline_outcomes: list[bool] = []
        self._request_generation = defaultdict(int)
        self._resolved_request_keys = set()

        # Avoid double-observing one NJNP request if the adapter later adds an
        # explicit MetricsCollector.record_charging_request call.
        self._request_seen_at = set()

        self._hooks_installed = False
        self._install_hooks()

    # ------------------------------------------------------------------
    # Public adapter hooks
    # ------------------------------------------------------------------
    def record_charging_request(self, node_id: int, request_time: float) -> None:
        node_id = int(node_id)
        request_time = float(request_time)
        dedupe_key = (node_id, round(request_time, 9))
        if dedupe_key in self._request_seen_at:
            return
        self._request_seen_at.add(dedupe_key)

        if node_id in self.active_charging_requests:
            return
        self._request_generation[node_id] += 1
        self.active_charging_requests[node_id] = request_time
        self.charging_request_count += 1

    def record_charge_start(
        self,
        node_id: int,
        service_start_time: float,
    ) -> None:
        """Resolve one request when charging physically begins.

        This is the common cross-method charging-latency definition and matches
        the INMA paper's "request sent -> MC begins to charge" metric. It also
        applies consistently to the proposed method, NJNP, RF clusters and
        fixed chargers.
        """
        node_id = int(node_id)
        start = self.active_charging_requests.pop(node_id, None)
        if start is None:
            # Proactive charging has no request-response latency.
            return

        generation = int(self._request_generation[node_id])
        key = (node_id, generation)
        if key in self._resolved_request_keys:
            return
        self._resolved_request_keys.add(key)

        response = max(0.0, float(service_start_time) - float(start))
        self.charging_first_response_times.append(response)
        self.charging_request_served += 1
        if self.common_charging_deadline_s is not None:
            self.charging_deadline_outcomes.append(
                bool(response > self.common_charging_deadline_s + 1e-12)
            )

    def record_node_death(self, node_id: int, death_time: float) -> None:
        node_id = int(node_id)
        start = self.active_charging_requests.pop(node_id, None)
        if start is None:
            return

        generation = int(self._request_generation[node_id])
        key = (node_id, generation)
        if key in self._resolved_request_keys:
            return
        self._resolved_request_keys.add(key)
        self.charging_request_failed += 1
        if self.common_charging_deadline_s is not None:
            # Physical node death before charging begins is a failed charging
            # response regardless of whether the nominal deadline elapsed.
            self.charging_deadline_outcomes.append(True)

    def record_workload_delivery(
        self,
        *,
        common_critical: Optional[bool],
        delay_s: Optional[float],
    ) -> None:
        self.workload_delivered += 1
        if common_critical is True:
            self.critical_workload_delivered += 1
        if delay_s is not None:
            value = _finite_float(delay_s)
            if math.isfinite(value) and value >= 0.0:
                self.common_delays.append(value)

    # ------------------------------------------------------------------
    # Automatic observer-only hooks for the current simulator stack
    # ------------------------------------------------------------------
    def _install_hooks(self) -> None:
        if self._hooks_installed:
            return

        self._hook_packet_accounting()
        self._hook_charging_accounting()
        self._hook_mobile_charger_service_start()
        self._hook_workload_delivery()
        self._hook_current_njnp_request_source()

        self._hooks_installed = True
        setattr(
            self.simulator,
            "_global_comparison_metrics_observer",
            self,
        )

    def _hook_packet_accounting(self) -> None:
        original = self.metrics.update_packet_transmission

        def wrapped(*args, **kwargs):
            values = _bound_arguments(original, args, kwargs)
            is_control = bool(values.get("is_control", False))
            size_bits = values.get("size_bits", None)
            size_bytes = values.get("size_bytes", 0)
            bit_count = (
                _finite_float(size_bits, math.nan)
                if size_bits is not None
                else _finite_float(size_bytes, 0.0) * 8.0
            )
            if not math.isfinite(bit_count):
                bit_count = 0.0
            bit_count = max(0.0, bit_count)

            if is_control:
                phase = str(values.get("overhead_phase", "operational")).strip().lower()
                if phase == "setup":
                    self.setup_control_bits += bit_count
                else:
                    self.operational_control_bits += bit_count
            else:
                count_generated = bool(values.get("count_generated", True))
                if count_generated:
                    self.generated_data_bits += bit_count

            return original(*args, **kwargs)

        self.metrics.update_packet_transmission = wrapped

    def _hook_charging_accounting(self) -> None:
        # Native request recorder used by the proposed method and adapters.
        original_request = self.metrics.record_charging_request

        def wrapped_request(*args, **kwargs):
            values = _bound_arguments(original_request, args, kwargs)
            result = original_request(*args, **kwargs)
            if bool(values.get("validated", True)):
                node_id = values.get("node_id")
                request_time = values.get("request_time")
                if node_id is not None and request_time is not None:
                    self.record_charging_request(node_id, request_time)
            return result

        self.metrics.record_charging_request = wrapped_request

        # Fixed-charger injection is already an instantaneous positive physical
        # delivery event, so it is also the service-start event.
        original_fixed_charge = self.metrics.record_fixed_charge

        def wrapped_fixed_charge(*args, **kwargs):
            values = _bound_arguments(original_fixed_charge, args, kwargs)
            result = original_fixed_charge(*args, **kwargs)
            node_id = values.get("node_id")
            injected = max(
                0.0, _finite_float(values.get("injected_energy", 0.0), 0.0)
            )
            if node_id is not None and injected > 0.0:
                service_start_time = float(
                    getattr(self.metrics, "current_sim_time", 0.0)
                )
                self.record_charge_start(node_id, service_start_time)
            return result

        self.metrics.record_fixed_charge = wrapped_fixed_charge

        original_death = self.metrics.record_node_death

        def wrapped_death(*args, **kwargs):
            values = _bound_arguments(original_death, args, kwargs)
            result = original_death(*args, **kwargs)
            node_id = values.get("node_id")
            death_time = values.get(
                "current_time",
                getattr(self.metrics, "current_sim_time", 0.0),
            )
            if node_id is not None:
                self.record_node_death(node_id, death_time)
            return result

        self.metrics.record_node_death = wrapped_death

    def _hook_mobile_charger_service_start(self) -> None:
        """Observe the exact start of each MC charging session.

        ``MetricsCollector.record_mc_charge`` is called when a session finishes,
        which is too late for a charging-response/latency metric. Wrapping
        ``begin_charging_session`` is observer-only and records the instant the
        MC actually begins delivering charge.
        """
        chargers = list(
            getattr(self.simulator, "mobile_chargers", []) or []
        )
        for charger in chargers:
            original_begin = getattr(charger, "begin_charging_session", None)
            if not callable(original_begin):
                continue

            def wrapped_begin(*args, __original=original_begin, **kwargs):
                values = _bound_arguments(__original, args, kwargs)
                session = __original(*args, **kwargs)
                if not session:
                    return session

                service_start_time = _finite_float(
                    session.get(
                        "start_time",
                        values.get(
                            "start_time",
                            getattr(self.metrics, "current_sim_time", 0.0),
                        ),
                    ),
                    getattr(self.metrics, "current_sim_time", 0.0),
                )

                allocations = dict(session.get("allocations", {}) or {})
                powers = dict(session.get("receive_powers", {}) or {})
                for node in list(session.get("nodes", []) or []):
                    node_id = int(getattr(node, "node_id"))
                    allocation = max(
                        0.0, _finite_float(allocations.get(node_id, 0.0), 0.0)
                    )
                    power = max(
                        0.0, _finite_float(powers.get(node_id, 0.0), 0.0)
                    )
                    if allocation > 0.0 and power > 0.0:
                        self.record_charge_start(
                            node_id,
                            float(service_start_time),
                        )
                return session

            charger.begin_charging_session = wrapped_begin

    def _hook_workload_delivery(self) -> None:
        if not hasattr(self.simulator, "_trace_packet_path"):
            return

        original = self.simulator._trace_packet_path

        def wrapped(*args, **kwargs):
            values = _bound_arguments(original, args, kwargs)
            count_generated = bool(values.get("count_generated", True))
            source_id = values.get("source_id", None)

            truth = None
            if count_generated and source_id is not None:
                truth = self.workload.match(
                    float(getattr(self.simulator, "current_time", 0.0)),
                    int(source_id),
                )
                if truth is None:
                    self.workload_match_failures += 1

            before_received = int(
                getattr(self.metrics, "received_data_packets", 0)
            )
            before_delay_len = len(
                getattr(self.metrics, "end_to_end_delays", [])
            )

            result = original(*args, **kwargs)

            after_received = int(
                getattr(self.metrics, "received_data_packets", 0)
            )

            if count_generated and after_received > before_received:
                delays = getattr(self.metrics, "end_to_end_delays", [])
                delay = None
                if len(delays) > before_delay_len:
                    delay = float(delays[-1])

                self.record_workload_delivery(
                    common_critical=(
                        None if truth is None
                        else truth.get("critical", None)
                    ),
                    delay_s=delay,
                )

            return result

        self.simulator._trace_packet_path = wrapped

    def _hook_current_njnp_request_source(self) -> None:
        """Compatibility observer for the current NJNP adapter.

        NJNP's method-native service pool returns the set of newly arrived
        requests. The wrapper observes that return value only; it does not
        change selection, preemption, queue order, charging, or request timing.

        Future adapters should preferably call MetricsCollector.record_charging_request
        or this observer's record_charging_request directly.
        """
        if not hasattr(self.simulator, "_njnp_sync_service_pool"):
            return

        original = self.simulator._njnp_sync_service_pool

        def wrapped(*args, **kwargs):
            newly_arrived = original(*args, **kwargs)
            now = float(getattr(self.simulator, "current_time", 0.0))
            for node_id in sorted(newly_arrived or []):
                self.record_charging_request(int(node_id), now)
            return newly_arrived

        self.simulator._njnp_sync_service_pool = wrapped

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _sensor_nodes(self) -> list:
        bs_id = int(getattr(self.simulator, "BS_ID", 0))
        nodes = getattr(self.simulator, "nodes", {}) or {}
        return [
            node
            for node_id, node in nodes.items()
            if int(node_id) != bs_id
        ]

    def _sensor_energy_snapshot(self) -> Tuple[float, float, float, float]:
        nodes = self._sensor_nodes()
        if not nodes:
            return math.nan, math.nan, math.nan, math.nan

        residual = np.asarray(
            [max(0.0, _finite_float(getattr(node, "E_current", 0.0), 0.0))
             for node in nodes],
            dtype=float,
        )
        capacity = np.asarray(
            [max(0.0, _finite_float(getattr(node, "E_max", 0.0), 0.0))
             for node in nodes],
            dtype=float,
        )

        residual_total = float(np.sum(residual))
        capacity_total = float(np.sum(capacity))
        residual_mean = float(np.mean(residual))
        residual_fraction = _safe_ratio(
            residual_total,
            capacity_total,
            scale=100.0,
        )
        return (
            residual_mean,
            residual_total,
            capacity_total,
            residual_fraction,
        )

    def _sensor_operational_availability(self, sim_time: float) -> float:
        n = max(1, int(getattr(self.metrics, "total_nodes",
                               getattr(self.simulator, "num_nodes", 1))))
        completed_dead = sum(
            max(0.0, _finite_float(v, 0.0))
            for v in getattr(self.metrics, "node_dead_durations", [])
        )

        open_dead = 0.0
        for start in (
            getattr(self.metrics, "_current_death_start_times", {}) or {}
        ).values():
            start = _finite_float(start, math.nan)
            if math.isfinite(start):
                open_dead += max(0.0, float(sim_time) - start)

        total_node_time = float(n) * max(float(sim_time), 0.0)
        if total_node_time <= 0.0:
            return math.nan
        available = max(
            0.0,
            total_node_time - completed_dead - open_dead,
        )
        return _safe_ratio(
            available,
            total_node_time,
            scale=100.0,
        )

    def export_row(
        self,
        *,
        seed: int,
        sim_time: Optional[float] = None,
        wall_clock_runtime_s: Optional[float] = None,
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Export ONLY method-independent comparison metrics + audit metadata."""
        if sim_time is None:
            sim_time = float(getattr(self.simulator, "sim_time", 0.0))
        sim_time = float(sim_time)

        m = self.metrics

        generated_data = int(getattr(m, "generated_data_packets", 0))
        received_data = int(getattr(m, "received_data_packets", 0))
        generated_critical_native = int(
            getattr(m, "generated_critical_packets", 0)
        )
        received_critical_native = int(
            getattr(m, "received_critical_packets", 0)
        )

        packet_pdr = _safe_ratio(
            received_data,
            generated_data,
            scale=100.0,
        )
        native_critical_pdr = _safe_ratio(
            received_critical_native,
            generated_critical_native,
            scale=100.0,
        )

        workload_ratio = _safe_ratio(
            self.workload_delivered,
            self.workload.total_events,
            scale=100.0,
        )
        common_critical_ratio = (
            _safe_ratio(
                self.critical_workload_delivered,
                self.workload.critical_events,
                scale=100.0,
            )
            if self.workload.critical_events is not None
            else math.nan
        )

        total_bits_received = max(
            0.0,
            _finite_float(getattr(m, "total_bits_received", 0.0), 0.0),
        )
        throughput = _safe_ratio(
            total_bits_received,
            sim_time,
            scale=1.0,
        )

        operational_control_packets = int(
            getattr(m, "operational_control_packets", 0)
        )
        setup_control_packets = int(
            getattr(m, "setup_control_packets", 0)
        )
        operational_packet_overhead = _safe_ratio(
            operational_control_packets,
            generated_data + operational_control_packets,
            scale=100.0,
        )
        operational_bit_overhead = _safe_ratio(
            self.operational_control_bits,
            self.generated_data_bits + self.operational_control_bits,
            scale=100.0,
        )

        death_times = list(
            (getattr(m, "node_death_times", {}) or {}).values()
        )
        n = max(
            1,
            int(getattr(m, "total_nodes",
                        getattr(self.simulator, "num_nodes", 1))),
        )
        fnd_obs, fnd_censored, fnd_restricted = _lifetime_event(
            death_times, 1, sim_time
        )
        hnd_rank = max(1, int(math.ceil(n / 2.0)))
        hnd_obs, hnd_censored, hnd_restricted = _lifetime_event(
            death_times, hnd_rank, sim_time
        )
        lnd_obs, lnd_censored, lnd_restricted = _lifetime_event(
            death_times, n, sim_time
        )

        residual_mean, residual_total, capacity_total, residual_fraction = (
            self._sensor_energy_snapshot()
        )

        alive_end = int(getattr(m, "alive_nodes", n))
        ever_dead = int(len(getattr(m, "node_death_times", {}) or {}))
        ever_dead_rate = _safe_ratio(
            ever_dead,
            n,
            scale=100.0,
        )

        mc_count = int(
            getattr(
                m,
                "mobile_charger_count",
                len(getattr(self.simulator, "mobile_chargers", []) or []) or 1,
            )
        )
        mc_travel_distance = max(
            0.0,
            _finite_float(getattr(m, "mc_travel_distance", 0.0), 0.0),
        )
        mc_travel_energy = max(
            0.0,
            _finite_float(getattr(m, "mc_travel_energy", 0.0), 0.0),
        )
        mc_rf_energy = max(
            0.0,
            _finite_float(getattr(m, "mc_rf_transmit_energy", 0.0), 0.0),
        )
        mc_contact_energy = max(
            0.0,
            _finite_float(getattr(m, "mc_contact_charging_energy", 0.0), 0.0),
        )
        mc_mission_energy = mc_travel_energy + mc_rf_energy + mc_contact_energy

        mc_injected = max(
            0.0,
            _finite_float(getattr(m, "mc_total_injected_energy", 0.0), 0.0),
        )
        fixed_injected = max(
            0.0,
            _finite_float(
                getattr(m, "fixed_charger_total_injected_energy", 0.0),
                0.0,
            ),
        )
        total_injected = mc_injected + fixed_injected

        mc_charge_events = int(getattr(m, "mc_charge_events", 0))
        fixed_charge_events = int(getattr(m, "fixed_charger_charge_events", 0))
        total_charge_events = mc_charge_events + fixed_charge_events

        charging_service_rate = _safe_ratio(
            self.charging_request_served,
            self.charging_request_count,
            scale=100.0,
        )

        deadline_failures = 0
        deadline_eligible = 0
        deadline_censored_end = 0
        if self.common_charging_deadline_s is not None:
            deadline_failures = int(sum(self.charging_deadline_outcomes))
            deadline_eligible = int(len(self.charging_deadline_outcomes))

            # An unresolved request whose common evaluation deadline has
            # already elapsed is an observed deadline failure. A request born
            # too close to the end of the run is right-censored instead.
            for start in self.active_charging_requests.values():
                age = max(0.0, sim_time - float(start))
                if age + 1e-12 >= self.common_charging_deadline_s:
                    deadline_failures += 1
                    deadline_eligible += 1
                else:
                    deadline_censored_end += 1

            charging_deadline_failure_rate = _safe_ratio(
                deadline_failures,
                deadline_eligible,
                scale=100.0,
            )
        else:
            charging_deadline_failure_rate = math.nan

        computational_edge = _mean(
            getattr(m, "edge_processing_times", []) or []
        )
        computational_bs = _mean(
            getattr(m, "bs_processing_times", []) or []
        )

        row: Dict[str, Any] = {
            # Contract / audit
            "Common_Metrics_Version": COMMON_METRICS_VERSION,
            "Method": self.method_name,
            "Scenario": self.scenario_name,
            "Seed": int(seed),
            "Simulation_Time_s": sim_time,
            "Workload_Trace_Path": self.workload_trace_path,
            "Common_Metric_Observer_Only_Flag": 1,
            "Common_Charging_Response_Definition": "request_to_charge_start",
            "Common_Workload_Match_Failures": int(
                self.workload_match_failures
            ),

            # Shared workload and QoS
            "Common_Workload_Events": int(self.workload.total_events),
            "Common_Workload_Delivered_Events": int(
                self.workload_delivered
            ),
            "Common_Workload_Delivery_Ratio_%": workload_ratio,
            "Common_Critical_Workload_Events": (
                math.nan
                if self.workload.critical_events is None
                else int(self.workload.critical_events)
            ),
            "Common_Critical_Workload_Delivered_Events": int(
                self.critical_workload_delivered
            ),
            "Common_Critical_Workload_Delivery_Ratio_%": common_critical_ratio,
            "Generated_Data_Packets": generated_data,
            "Received_Data_Packets": received_data,
            "Packet_Level_PDR_%": packet_pdr,
            "Generated_Critical_Packets_Native": generated_critical_native,
            "Received_Critical_Packets_Native": received_critical_native,
            "Native_Critical_PDR_%": native_critical_pdr,
            "Throughput_bps": throughput,
            "Common_Avg_EndToEnd_Delay_s": _mean(self.common_delays),
            "Common_P95_EndToEnd_Delay_s": _percentile(
                self.common_delays, 95.0
            ),
            "Common_Max_EndToEnd_Delay_s": _max_or_nan(
                self.common_delays
            ),

            # Operational overhead. Bootstrap/setup is deliberately excluded.
            "Operational_Control_Packets": operational_control_packets,
            "Setup_Control_Packets_Excluded": setup_control_packets,
            "Common_Operational_Control_Overhead_Packet_%": (
                operational_packet_overhead
            ),
            "Common_Operational_Control_Bits": float(
                self.operational_control_bits
            ),
            "Common_Setup_Control_Bits_Excluded": float(
                self.setup_control_bits
            ),
            "Common_Generated_Data_Bits": float(
                self.generated_data_bits
            ),
            "Common_Operational_Control_Overhead_Bit_%": (
                operational_bit_overhead
            ),

            # Lifetime / energy
            "Alive_Nodes_End": alive_end,
            "Ever_Dead_Node_Count": ever_dead,
            "Ever_Dead_Node_Rate_%": ever_dead_rate,
            "Node_Revival_Count": int(
                getattr(m, "node_revival_count", 0)
            ),
            "Sensor_Operational_Availability_%": (
                self._sensor_operational_availability(sim_time)
            ),
            "Mean_Residual_Sensor_Energy_End_J": residual_mean,
            "Total_Residual_Sensor_Energy_End_J": residual_total,
            "Total_Sensor_Energy_Capacity_J": capacity_total,
            "Residual_Sensor_Energy_Fraction_End_%": residual_fraction,

            # Censor-aware lifetime events
            "FND_Observed_s": fnd_obs,
            "FND_Censored_Flag": int(fnd_censored),
            "FND_Restricted_s": fnd_restricted,
            "HND_Observed_s": hnd_obs,
            "HND_Censored_Flag": int(hnd_censored),
            "HND_Restricted_s": hnd_restricted,
            "LND_Observed_s": lnd_obs,
            "LND_Censored_Flag": int(lnd_censored),
            "LND_Restricted_s": lnd_restricted,

            # Charging / fleet
            "Mobile_Charger_Count": mc_count,
            "MC_Travel_Distance_m": mc_travel_distance,
            "MC_Travel_Energy_J": mc_travel_energy,
            "MC_RF_Transmit_Energy_J": mc_rf_energy,
            "MC_Contact_Charging_Energy_J": mc_contact_energy,
            "MC_Total_Mission_Energy_J": mc_mission_energy,
            "MC_Injected_Energy_J": mc_injected,
            "Fixed_Charger_Injected_Energy_J": fixed_injected,
            "Total_Injected_Energy_To_Sensors_J": total_injected,
            "MC_Useful_Energy_Efficiency": _safe_ratio(
                mc_injected,
                mc_mission_energy,
            ),
            "MC_Charge_Events": mc_charge_events,
            "Fixed_Charger_Charge_Events": fixed_charge_events,
            "Total_Charge_Service_Events": total_charge_events,
            "Common_Charging_Requests": int(
                self.charging_request_count
            ),
            "Common_Charging_Requests_Served": int(
                self.charging_request_served
            ),
            "Common_Charging_Requests_Failed_Before_First_Response": int(
                self.charging_request_failed
            ),
            "Common_Charging_Requests_Unresolved_End": int(
                len(self.active_charging_requests)
            ),
            "Common_Charging_Request_Service_Rate_%": charging_service_rate,
            "Common_Charging_Deadline_s": (
                math.nan
                if self.common_charging_deadline_s is None
                else float(self.common_charging_deadline_s)
            ),
            "Common_Charging_Deadline_Eligible_Requests": int(
                deadline_eligible
            ),
            "Common_Charging_Deadline_Failures": int(
                deadline_failures
            ),
            "Common_Charging_Deadline_Censored_End": int(
                deadline_censored_end
            ),
            "Common_Charging_Deadline_Failure_Rate_%": (
                charging_deadline_failure_rate
            ),
            "Common_Charging_First_Response_Avg_s": _mean(
                self.charging_first_response_times
            ),
            "Common_Charging_First_Response_P95_s": _percentile(
                self.charging_first_response_times, 95.0
            ),
            "Common_Charging_First_Response_Max_s": _max_or_nan(
                self.charging_first_response_times
            ),
            "MC_Injected_Energy_Per_Charge_Event_J": _safe_ratio(
                mc_injected,
                mc_charge_events,
            ),
            "MC_Travel_Per_Injected_Joule_m_per_J": _safe_ratio(
                mc_travel_distance,
                mc_injected,
            ),

            # Computational implementation cost
            "Sensor_Execution_Time_ms": computational_edge,
            "BS_Execution_Time_ms": computational_bs,
            "Wall_Clock_Runtime_s": (
                math.nan
                if wall_clock_runtime_s is None
                else max(0.0, _finite_float(wall_clock_runtime_s, math.nan))
            ),
        }

        if extra_metadata:
            for key, value in extra_metadata.items():
                row[str(key)] = value

        validate_global_comparison_row(row)
        return row


# ============================================================================
# ATTACHMENT API
# ============================================================================

def attach_global_comparison_metrics(
    simulator,
    *,
    workload_trace_path: Path | str,
    method_name: str,
    scenario_name: str,
    charging_deadline_s: Optional[float] = None,
) -> GlobalComparisonMetricsObserver:
    """Attach the common observer BEFORE simulator.run_simulation()."""
    existing = getattr(
        simulator,
        "_global_comparison_metrics_observer",
        None,
    )
    if existing is not None:
        return existing

    return GlobalComparisonMetricsObserver(
        simulator,
        workload_trace_path=workload_trace_path,
        method_name=method_name,
        scenario_name=scenario_name,
        charging_deadline_s=charging_deadline_s,
    )


# ============================================================================
# VALIDATION
# ============================================================================

_PERCENT_COLUMNS = {
    "Common_Workload_Delivery_Ratio_%",
    "Common_Critical_Workload_Delivery_Ratio_%",
    "Packet_Level_PDR_%",
    "Native_Critical_PDR_%",
    "Common_Operational_Control_Overhead_Packet_%",
    "Common_Operational_Control_Overhead_Bit_%",
    "Residual_Sensor_Energy_Fraction_End_%",
    "Ever_Dead_Node_Rate_%",
    "Sensor_Operational_Availability_%",
    "Common_Charging_Request_Service_Rate_%",
    "Common_Charging_Deadline_Failure_Rate_%",
}


def validate_global_comparison_row(row: Mapping[str, Any]) -> None:
    """Fail closed on impossible common-metric values."""
    for column in _PERCENT_COLUMNS:
        value = _finite_float(row.get(column, math.nan))
        if math.isfinite(value) and not (-1e-9 <= value <= 100.0 + 1e-9):
            raise RuntimeError(
                f"Common metric {column} is outside [0,100]: {value}"
            )

    generated = int(row.get("Generated_Data_Packets", 0))
    received = int(row.get("Received_Data_Packets", 0))
    if received > generated:
        raise RuntimeError(
            f"Received_Data_Packets ({received}) > Generated_Data_Packets ({generated})"
        )

    workload = int(row.get("Common_Workload_Events", 0))
    workload_delivered = int(
        row.get("Common_Workload_Delivered_Events", 0)
    )
    if workload_delivered > workload:
        raise RuntimeError(
            "Common workload deliveries exceed scheduled workload events."
        )

    common_critical = row.get("Common_Critical_Workload_Events", math.nan)
    common_critical_delivered = int(
        row.get("Common_Critical_Workload_Delivered_Events", 0)
    )
    critical_value = _finite_float(common_critical)
    if (
        math.isfinite(critical_value)
        and common_critical_delivered > int(critical_value)
    ):
        raise RuntimeError(
            "Delivered common critical events exceed scheduled critical events."
        )

    requests = int(row.get("Common_Charging_Requests", 0))
    served = int(row.get("Common_Charging_Requests_Served", 0))
    failed = int(
        row.get(
            "Common_Charging_Requests_Failed_Before_First_Response",
            0,
        )
    )
    unresolved = int(
        row.get("Common_Charging_Requests_Unresolved_End", 0)
    )
    if served + failed + unresolved != requests:
        raise RuntimeError(
            "Charging-request partition is inconsistent: "
            f"requests={requests}, served={served}, failed={failed}, "
            f"unresolved={unresolved}."
        )

    deadline_eligible = int(
        row.get("Common_Charging_Deadline_Eligible_Requests", 0)
    )
    deadline_failures = int(
        row.get("Common_Charging_Deadline_Failures", 0)
    )
    deadline_censored = int(
        row.get("Common_Charging_Deadline_Censored_End", 0)
    )
    if deadline_failures > deadline_eligible:
        raise RuntimeError(
            "Charging deadline failures exceed eligible requests."
        )
    if deadline_eligible + deadline_censored > requests:
        raise RuntimeError(
            "Charging deadline eligible+censored requests exceed requests."
        )

    alive = int(row.get("Alive_Nodes_End", 0))
    ever_dead = int(row.get("Ever_Dead_Node_Count", 0))
    if alive < 0 or ever_dead < 0:
        raise RuntimeError("Negative lifetime node counters detected.")

    for column in (
        "MC_Travel_Distance_m",
        "MC_Travel_Energy_J",
        "MC_RF_Transmit_Energy_J",
        "MC_Contact_Charging_Energy_J",
        "MC_Total_Mission_Energy_J",
        "MC_Injected_Energy_J",
        "Fixed_Charger_Injected_Energy_J",
        "Total_Injected_Energy_To_Sensors_J",
    ):
        value = _finite_float(row.get(column, math.nan))
        if math.isfinite(value) and value < -1e-12:
            raise RuntimeError(
                f"Negative physical metric {column}: {value}"
            )


# ============================================================================
# SUMMARY / 95% CI
# ============================================================================

def _ci95_numeric(values: Iterable[Any]) -> Tuple[int, float, float, float]:
    arr = pd.to_numeric(
        pd.Series(list(values)),
        errors="coerce",
    ).dropna().to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]

    n = int(arr.size)
    if n == 0:
        return 0, math.nan, math.nan, math.nan

    mean = float(np.mean(arr))
    if n == 1:
        return 1, mean, mean, mean

    # Avoid importing scipy at module import time if only raw export is used.
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
        # Normal approximation fallback for portability.
        std = float(np.std(arr, ddof=1))
        sem = std / math.sqrt(n)
        margin = 1.96 * sem
        return n, mean, mean - margin, mean + margin


def summarize_global_comparison_results(
    df: pd.DataFrame,
    out_csv: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Produce per-method mean + 95% CI using only common numeric metrics.

    NaN values are excluded from that metric's CI and the number of valid seeds
    is explicitly exported as <metric>_N_Valid.
    """
    required = {"Method", "Seed", "Common_Metrics_Version"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Global comparison dataframe is missing: "
            + ", ".join(sorted(missing))
        )

    versions = set(
        df["Common_Metrics_Version"].dropna().astype(str).unique()
    )
    if versions != {COMMON_METRICS_VERSION}:
        raise ValueError(
            f"Mixed common metric versions detected: {sorted(versions)}"
        )

    numeric_candidates = [
        column
        for column in df.columns
        if column in {
            metric
            for group in METRIC_GROUPS.values()
            for metric in group
        }
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    rows = []
    for method, group in df.groupby("Method", sort=False):
        row: Dict[str, Any] = {
            "Method": method,
            "N_Runs": int(len(group)),
            "Seed_Min": int(group["Seed"].min()),
            "Seed_Max": int(group["Seed"].max()),
            "Common_Metrics_Version": COMMON_METRICS_VERSION,
        }

        for column in numeric_candidates:
            n_valid, mean, low, high = _ci95_numeric(group[column])
            if column in _PERCENT_COLUMNS:
                if math.isfinite(mean):
                    mean = min(100.0, max(0.0, mean))
                if math.isfinite(low):
                    low = max(0.0, low)
                if math.isfinite(high):
                    high = min(100.0, high)

            row[f"{column}_N_Valid"] = n_valid
            row[f"{column}_Mean"] = mean
            row[f"{column}_CI95_Low"] = low
            row[f"{column}_CI95_High"] = high

        rows.append(row)

    summary = pd.DataFrame(rows)

    if out_csv is not None:
        path = Path(out_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(str(path) + ".tmp")
        summary.to_csv(temporary, index=False)
        os.replace(temporary, path)

    return summary


def select_primary_comparison_table(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Return a compact thesis table containing only primary common metrics."""
    columns = ["Method", "N_Runs"]
    for metric in PRIMARY_COMPARISON_METRICS:
        mean_col = f"{metric}_Mean"
        low_col = f"{metric}_CI95_Low"
        high_col = f"{metric}_CI95_High"
        n_col = f"{metric}_N_Valid"
        for column in (n_col, mean_col, low_col, high_col):
            if column in summary.columns:
                columns.append(column)
    return summary.loc[:, columns].copy()
