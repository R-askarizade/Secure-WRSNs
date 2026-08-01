from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ParameterSpec:
    """One tunable method parameter represented on a normalized [0, 1] axis."""

    name: str
    group: str
    kind: str  # float | int | choice
    default: float | int
    low: float | int | None = None
    high: float | int | None = None
    choices: tuple[Any, ...] = ()
    scale: str = "linear"  # linear | log
    sobol_eligible: bool = True
    description: str = ""

    def decode(self, unit_value: float) -> float | int | Any:
        u = float(np.clip(unit_value, 0.0, 1.0))
        if self.kind == "choice":
            if not self.choices:
                raise ValueError(f"{self.name}: choice parameter has no choices")
            idx = min(len(self.choices) - 1, int(math.floor(u * len(self.choices))))
            return self.choices[idx]
        if self.low is None or self.high is None:
            raise ValueError(f"{self.name}: numeric bounds are missing")
        low = float(self.low)
        high = float(self.high)
        if self.scale == "log":
            if low <= 0 or high <= 0:
                raise ValueError(f"{self.name}: log bounds must be positive")
            value = math.exp(math.log(low) + u * (math.log(high) - math.log(low)))
        else:
            value = low + u * (high - low)
        if self.kind == "int":
            return int(round(value))
        return float(value)

    def encode_default(self) -> float:
        if self.kind == "choice":
            try:
                idx = list(self.choices).index(self.default)
            except ValueError as exc:
                raise ValueError(f"Default {self.default!r} not in choices for {self.name}") from exc
            return (idx + 0.5) / len(self.choices)
        if self.low is None or self.high is None:
            raise ValueError(f"{self.name}: numeric bounds are missing")
        low = float(self.low)
        high = float(self.high)
        default = float(self.default)
        if self.scale == "log":
            return (math.log(default) - math.log(low)) / (math.log(high) - math.log(low))
        return (default - low) / (high - low)


# These are experimental/network/physical conditions. Stage 2 never changes them.
LOCKED_CONFIGURATION_KEYS: frozenset[str] = frozenset({
    "experiment_family",
    "enable_energy_experiment",
    "initial_energy_override",
    "event_rate_lambda",
    "idle_sensing_power_j_per_s",
    "attack_percentage",
    "critical_event_probability",
    "transmission_range",
    "data_packet_bits",
    "control_packet_bits",
    "radio_e_elec_j_per_bit",
    "radio_eps_fs_j_per_bit_m2",
    "radio_eps_mp_j_per_bit_m4",
    "radio_e_da_j_per_bit",
    "num_mobile_chargers",
    "mobile_charger_selection_mode",
    "mobile_charger_capacity_mode",
    "mobile_charger_capacity",
    "mobile_charger_capacities",
    "mobile_charger_sensor_equivalent_capacity",
    "mobile_charger_sensor_equivalent_capacities",
    "mobile_charger_travel_cost_mode",
    "mobile_charger_travel_energy_cost_per_m",
    "mobile_charger_travel_energy_costs_per_m",
    "mobile_charger_travel_sensor_equivalent_cost_per_m",
    "mobile_charger_speed",
    "mobile_charger_speeds",
    "charging_rate",
    "rf_charging_radius",
    "rf_path_loss_exponent",
    "rf_reference_distance_m",
    "rf_energy_conversion_efficiency",
    "contact_charging_efficiency",
    "charging_energy_efficiency",
    "fixed_charger_count",
})

# Method-presence switches belong to ablation studies, not parameter tuning.
LOCKED_METHOD_SWITCHES: frozenset[str] = frozenset({
    "use_adaptive_quantum_charging",
    "use_dqn_charger_routing",
    "enable_qifl_learning",
    "use_adaptive_charge_guards",
    "use_charging_hysteresis",
    "use_travel_aware_admission",
    "use_hybrid_fixed_chargers",
    "use_rf_multi_node_charging",
    "use_dynamic_partial_charging",
    "use_hybrid_hierarchical_routing",
    "use_data_plane_hybrid_routing",
    "use_adaptive_greedy_weights",
    "use_ifbe_fuzzy_ch_selection",
    "use_fcbe_soft_balancing",
    "use_mst_backbone",
    "use_secure_charging_request_validation",
    "use_mc_hmac_authentication",
    "use_mc_location_verification",
    "use_sed_filter",
    "use_zero_overhead_spoof_detection",
})


PARAMETERS: tuple[ParameterSpec, ...] = (
    # QIFL/DQN learner profile.
    ParameterSpec("qifl_learning_rate", "qifl_dqn", "float", 5e-3, 1e-4, 2e-2, scale="log",
                  description="Adam learning rate of the quantum-inspired fuzzy oracle."),
    ParameterSpec("qifl_threshold_lower_bound", "qifl_dqn", "float", 0.20, 0.14, 0.30,
                  description="Lower clamp of the adaptive charging threshold."),
    ParameterSpec("qifl_threshold_span", "qifl_dqn", "float", 0.32, 0.14, 0.42,
                  description="Virtual parameter; upper bound equals lower bound plus this span."),
    ParameterSpec("qifl_center_shift", "qifl_dqn", "float", 0.0, -0.05, 0.05,
                  description="Common shift applied to the five QIFL output centers."),
    ParameterSpec("qifl_center_scale", "qifl_dqn", "float", 1.0, 0.70, 1.30,
                  description="Scale of QIFL output-center spread around their mean."),
    ParameterSpec("dqn_learning_rate", "qifl_dqn", "float", 1e-3, 1e-5, 5e-3, scale="log",
                  description="DQN Adam learning rate."),
    ParameterSpec("dqn_gamma", "qifl_dqn", "float", 0.80, 0.70, 0.995,
                  description="DQN temporal discount factor."),
    ParameterSpec("dqn_hidden_dim", "qifl_dqn", "choice", 64, choices=(32, 64, 128, 256),
                  description="Width of both hidden layers."),
    ParameterSpec("dqn_batch_size", "qifl_dqn", "choice", 16, choices=(8, 16, 32, 64),
                  description="Replay minibatch size."),
    ParameterSpec("dqn_replay_to_batch_ratio", "qifl_dqn", "choice", 32, choices=(8, 16, 32, 64),
                  description="Virtual parameter; replay capacity is batch size times this ratio."),
    ParameterSpec("dqn_target_sync_interval", "qifl_dqn", "choice", 10, choices=(5, 10, 20, 40),
                  description="Gradient updates between target-network synchronizations."),
    ParameterSpec("dqn_warmup_competitive_decisions", "qifl_dqn", "choice", 24,
                  choices=(8, 16, 24, 48, 96), description="Competitive decisions before DQN control."),
    # Four independent log-ratios determine five reward weights.  The deadline
    # component is the reference and the original total reward scale (14.25) is
    # restored after decoding.  This removes the multiplicative
    # non-identifiability of sampling five raw weights and normalizing them.
    ParameterSpec("dqn_reward_energy_to_deadline_ratio", "qifl_dqn", "float",
                  1.5 / 4.0, 0.10, 1.50, scale="log",
                  description="Energy/deadline reward-weight ratio."),
    ParameterSpec("dqn_reward_travel_to_deadline_ratio", "qifl_dqn", "float",
                  0.75 / 4.0, 0.025, 0.75, scale="log",
                  description="Travel/deadline penalty-weight ratio."),
    ParameterSpec("dqn_reward_revisit_to_deadline_ratio", "qifl_dqn", "float",
                  2.0 / 4.0, 0.10, 2.00, scale="log",
                  description="Revisit/deadline penalty-weight ratio."),
    ParameterSpec("dqn_reward_death_to_deadline_ratio", "qifl_dqn", "float",
                  6.0 / 4.0, 0.50, 4.00, scale="log",
                  description="Death/deadline penalty-weight ratio."),

    # Admission, service, and partial charging policy profile.
    ParameterSpec("minimum_charge_energy_ratio", "charging_policy", "float", 0.08, 0.03, 0.18,
                  description="Minimum useful energy allocation per admitted request."),
    ParameterSpec("post_charge_release_ratio", "charging_policy", "float", 0.88, 0.72, 0.98,
                  description="Energy ratio at which a serviced node is released."),
    ParameterSpec("charging_reentry_cooldown_s", "charging_policy", "float", 180.0, 45.0, 420.0,
                  description="Minimum interval before ordinary re-admission."),
    ParameterSpec("travel_admission_lambda", "charging_policy", "float", 2.0, 0.4, 5.0,
                  description="Strength of the travel-aware admission penalty."),
    ParameterSpec("travel_admission_gate", "charging_policy", "float", 0.10, 0.01, 0.35,
                  description="Minimum travel-aware admission score."),
    ParameterSpec("minimum_cluster_charge_gap_ratio", "charging_policy", "float", 0.05, 0.0, 0.18,
                  description="Minimum energy gap preserved within a charging cluster."),
    ParameterSpec("max_cluster_stay_time_s", "charging_policy", "float", 1.5, 0.4, 5.0,
                  description="Maximum RF cluster dwell time."),
    # Two independent log-ratios determine three fuzzy weights.  NOW is the
    # reference and the decoded weights are normalized to the original sum 3.
    ParameterSpec("dynamic_charging_wt_to_now_ratio", "charging_policy", "float",
                  1.0, 0.10, 10.0, scale="log",
                  description="Waiting-time/NOW fuzzy-weight ratio."),
    ParameterSpec("dynamic_charging_eps_to_now_ratio", "charging_policy", "float",
                  1.0, 0.10, 10.0, scale="log",
                  description="Depletion-rate/NOW fuzzy-weight ratio."),
    ParameterSpec("request_plus_fraction", "charging_policy", "float", 0.50, 0.15, 0.85,
                  description="Fraction controlling admissions to the plus pool."),
    ParameterSpec("plus_pool_target_ratio", "charging_policy", "float", 0.45, 0.25, 0.70,
                  description="Target energy ratio for plus-pool service."),
    ParameterSpec("plus_pool_max_members_per_cluster", "charging_policy", "choice", 2,
                  choices=(1, 2, 3, 4), description="Maximum plus-pool members in one RF cluster."),
    ParameterSpec("service_horizon_safety_multiplier", "charging_policy", "float", 1.12, 1.0, 1.5,
                  description="Safety multiplier applied to predicted service horizon."),
    ParameterSpec("service_latency_safety_margin_s", "charging_policy", "float", 75.0, 20.0, 180.0,
                  description="Additional latency margin in deadline feasibility checks."),
    ParameterSpec("adaptive_request_threshold_hard_cap", "charging_policy", "float", 0.68, 0.50, 0.85,
                  description="Hard upper cap on adaptive charging-request thresholds."),
    ParameterSpec("sos_threshold_margin_ratio", "charging_policy", "float", 0.03, 0.005, 0.10,
                  description="SOS threshold margin above estimated critical energy."),

    # DATA-plane routing profile. Control/charging routing remains structurally fixed.
    ParameterSpec("data_route_recompute_period", "routing", "choice", 5, choices=(2, 3, 5, 8, 10),
                  description="Control epochs between DATA overlay recomputations."),
    ParameterSpec("data_route_min_improvement", "routing", "float", 0.0025, 2.5e-4, 2e-2, scale="log",
                  description="Minimum projected improvement required to replace a reference route."),
    ParameterSpec("data_route_energy_tolerance", "routing", "float", 0.0, 0.0, 0.05,
                  description="Permitted projected energy increase in the dominance guard."),
    ParameterSpec("data_route_stress_tolerance", "routing", "float", 0.0, 0.0, 0.05,
                  description="Permitted bottleneck-stress increase in the dominance guard."),
    ParameterSpec("data_route_distance_stretch", "routing", "float", 1.0, 1.0, 1.30,
                  description="Maximum accepted route-distance stretch."),
    ParameterSpec("data_route_hop_slack", "routing", "choice", 0, choices=(0, 1, 2, 3),
                  description="Maximum accepted hop-count increase."),
    ParameterSpec("data_secondary_k_paths", "routing", "choice", 24, choices=(8, 12, 24, 36, 48),
                  description="Candidate path budget for secondary DATA routes."),
    ParameterSpec("data_secondary_max_energy_stretch", "routing", "float", 1.0, 1.0, 1.30,
                  description="Maximum energy stretch accepted for a secondary path."),
    ParameterSpec("data_routing_weight_temperature", "routing", "float", 0.75, 0.25, 1.75,
                  description="Softmax temperature for adaptive DATA routing weights."),
    ParameterSpec("data_routing_weight_floor", "routing", "float", 0.10, 0.02, 0.20,
                  description="Minimum contribution of each adaptive routing criterion."),
)


PROFILE_GROUPS: dict[str, tuple[str, ...]] = {
    "qifl_dqn": ("qifl_dqn",),
    "charging_policy": ("charging_policy",),
    "routing": ("routing",),
    "charging_all": ("qifl_dqn", "charging_policy"),
    "all": ("qifl_dqn", "charging_policy", "routing"),
}


def get_specs(profile: str = "qifl_dqn", names: Sequence[str] | None = None) -> list[ParameterSpec]:
    by_name = {p.name: p for p in PARAMETERS}
    if names is not None:
        missing = sorted(set(names) - set(by_name))
        if missing:
            raise KeyError(f"Unknown parameter(s): {', '.join(missing)}")
        return [by_name[name] for name in names]
    if profile not in PROFILE_GROUPS:
        raise KeyError(f"Unknown profile {profile!r}; choose from {sorted(PROFILE_GROUPS)}")
    groups = set(PROFILE_GROUPS[profile])
    return [p for p in PARAMETERS if p.group in groups]


def unit_problem(specs: Sequence[ParameterSpec]) -> dict[str, Any]:
    """SALib problem in normalized coordinates.

    Decoding is deliberately external so Morris and Sobol effects are directly
    comparable on a common [0, 1] input scale even when physical parameters use
    logarithmic or discrete domains.
    """
    return {
        "num_vars": len(specs),
        "names": [p.name for p in specs],
        "bounds": [[0.0, 1.0] for _ in specs],
    }


def decode_unit_row(specs: Sequence[ParameterSpec], row: Sequence[float]) -> dict[str, Any]:
    if len(specs) != len(row):
        raise ValueError(f"Expected {len(specs)} values, received {len(row)}")
    return {spec.name: spec.decode(float(u)) for spec, u in zip(specs, row)}


def _renormalize(values: Mapping[str, float], keys: Sequence[str], target_sum: float) -> dict[str, float]:
    raw = np.asarray([max(float(values[key]), 1e-12) for key in keys], dtype=float)
    raw *= float(target_sum) / float(raw.sum())
    return {key: float(value) for key, value in zip(keys, raw)}


def method_values_to_overrides(method_values: Mapping[str, Any]) -> dict[str, Any]:
    """Convert sampled method values to valid simulator configuration overrides."""
    values = dict(method_values)
    overrides = dict(values)

    # Virtual QIFL upper bound and ordered output centers.
    lower = float(values.get("qifl_threshold_lower_bound", 0.20))
    if "qifl_threshold_span" in values:
        upper = min(0.80, lower + float(values.pop("qifl_threshold_span")))
        upper = max(lower + 0.05, upper)
        overrides.pop("qifl_threshold_span", None)
        overrides["qifl_threshold_upper_bound"] = float(upper)
    base_centers = np.asarray((0.18, 0.26, 0.34, 0.42, 0.50), dtype=float)
    if "qifl_center_shift" in values or "qifl_center_scale" in values:
        shift = float(values.get("qifl_center_shift", 0.0))
        scale = float(values.get("qifl_center_scale", 1.0))
        centered = base_centers.mean() + scale * (base_centers - base_centers.mean()) + shift
        centered = np.clip(centered, 0.08, 0.90)
        centered = np.maximum.accumulate(centered + np.arange(5) * 1e-6)
        overrides.pop("qifl_center_shift", None)
        overrides.pop("qifl_center_scale", None)
        overrides["qifl_threshold_centers"] = tuple(float(x) for x in centered)

    # Replay capacity is represented by a scale-free ratio.
    if "dqn_replay_to_batch_ratio" in values:
        batch = int(values.get("dqn_batch_size", 16))
        ratio = int(values["dqn_replay_to_batch_ratio"])
        overrides.pop("dqn_replay_to_batch_ratio", None)
        overrides["dqn_replay_capacity"] = int(max(batch, batch * ratio))

    # Decode four independent reward ratios into five actual simulator
    # weights.  Deadline is the reference component.  The total scale is fixed
    # to the original 14.25 so sensitivity concerns relative emphasis rather
    # than an arbitrary common multiplier.
    reward_ratio_keys = (
        "dqn_reward_energy_to_deadline_ratio",
        "dqn_reward_travel_to_deadline_ratio",
        "dqn_reward_revisit_to_deadline_ratio",
        "dqn_reward_death_to_deadline_ratio",
    )
    if any(key in values for key in reward_ratio_keys):
        ratios = {
            "dqn_reward_energy_weight": float(
                values.get("dqn_reward_energy_to_deadline_ratio", 1.5 / 4.0)
            ),
            "dqn_reward_deadline_weight": 1.0,
            "dqn_reward_travel_weight": float(
                values.get("dqn_reward_travel_to_deadline_ratio", 0.75 / 4.0)
            ),
            "dqn_reward_revisit_penalty": float(
                values.get("dqn_reward_revisit_to_deadline_ratio", 2.0 / 4.0)
            ),
            "dqn_reward_death_penalty": float(
                values.get("dqn_reward_death_to_deadline_ratio", 6.0 / 4.0)
            ),
        }
        for key in reward_ratio_keys:
            overrides.pop(key, None)
        reward_keys = tuple(ratios)
        overrides.update(_renormalize(ratios, reward_keys, target_sum=14.25))
    else:
        # Backward-compatible handling for an explicitly supplied set of raw
        # weights; these names are not exposed by the Stage-2 registry.
        reward_keys = (
            "dqn_reward_energy_weight",
            "dqn_reward_deadline_weight",
            "dqn_reward_travel_weight",
            "dqn_reward_revisit_penalty",
            "dqn_reward_death_penalty",
        )
        if any(key in values for key in reward_keys):
            defaults = {
                "dqn_reward_energy_weight": 1.5,
                "dqn_reward_deadline_weight": 4.0,
                "dqn_reward_travel_weight": 0.75,
                "dqn_reward_revisit_penalty": 2.0,
                "dqn_reward_death_penalty": 6.0,
            }
            merged = {
                key: float(values.get(key, defaults[key])) for key in reward_keys
            }
            overrides.update(
                _renormalize(merged, reward_keys, target_sum=sum(defaults.values()))
            )

    # Decode two independent dynamic-fuzzy ratios into three actual weights.
    dynamic_ratio_keys = (
        "dynamic_charging_wt_to_now_ratio",
        "dynamic_charging_eps_to_now_ratio",
    )
    if any(key in values for key in dynamic_ratio_keys):
        ratios = {
            "dynamic_charging_weight_wt": float(
                values.get("dynamic_charging_wt_to_now_ratio", 1.0)
            ),
            "dynamic_charging_weight_eps": float(
                values.get("dynamic_charging_eps_to_now_ratio", 1.0)
            ),
            "dynamic_charging_weight_now": 1.0,
        }
        for key in dynamic_ratio_keys:
            overrides.pop(key, None)
        dynamic_keys = tuple(ratios)
        overrides.update(_renormalize(ratios, dynamic_keys, target_sum=3.0))
    else:
        # Backward-compatible handling for raw weights.
        dynamic_keys = (
            "dynamic_charging_weight_wt",
            "dynamic_charging_weight_eps",
            "dynamic_charging_weight_now",
        )
        if any(key in values for key in dynamic_keys):
            merged = {key: float(values.get(key, 1.0)) for key in dynamic_keys}
            overrides.update(_renormalize(merged, dynamic_keys, target_sum=3.0))

    # Defensive cross-parameter constraints.
    if "minimum_charge_energy_ratio" in overrides and "post_charge_release_ratio" in overrides:
        minimum = float(overrides["minimum_charge_energy_ratio"])
        overrides["post_charge_release_ratio"] = max(
            float(overrides["post_charge_release_ratio"]), minimum + 0.25)
        overrides["post_charge_release_ratio"] = min(0.99, overrides["post_charge_release_ratio"])

    forbidden = (set(overrides) & LOCKED_CONFIGURATION_KEYS) | (set(overrides) & LOCKED_METHOD_SWITCHES)
    if forbidden:
        raise ValueError(f"Stage-2 attempted to change locked setting(s): {sorted(forbidden)}")
    return overrides


def apply_method_values(base_config: Mapping[str, Any], method_values: Mapping[str, Any]) -> dict[str, Any]:
    base = dict(base_config)
    locked_before = {key: base.get(key) for key in LOCKED_CONFIGURATION_KEYS | LOCKED_METHOD_SWITCHES}
    overrides = method_values_to_overrides(method_values)
    base.update(overrides)
    changed_locked = {
        key: (locked_before[key], base.get(key))
        for key in locked_before
        if locked_before[key] != base.get(key)
    }
    if changed_locked:
        raise AssertionError(f"Locked configuration changed: {changed_locked}")
    return base


def config_hash(overrides: Mapping[str, Any]) -> str:
    payload = json.dumps(overrides, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def registry_records(specs: Iterable[ParameterSpec] = PARAMETERS) -> list[dict[str, Any]]:
    records = []
    for spec in specs:
        record = asdict(spec)
        record["choices"] = json.dumps(record["choices"], ensure_ascii=False)
        records.append(record)
    return records
