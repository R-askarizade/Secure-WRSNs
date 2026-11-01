"""Canonical Real-World comparison runner for the WRSN project.

Design goals
------------
1. The Real-World physical/scenario configuration has exactly ONE source of
   truth: ``Article_Scaled_Absolute_Single`` in ``sim_phase.py``.
2. Baseline files do not repeat Real-World numbers.
3. Each method may override only explicitly declared *method-native* keys.
4. External evaluation conditions are shared whenever the method architecture
   permits it: node count, field size, operational horizon, seeds, workload
   trace, attack realization, packet/radio environment, and common metric contract.
5. The proposed method is not simplified to match a baseline. Its only changes
   relative to the canonical Real-World scenario are the five Adaptive Multi-MC
   extensions agreed for the final method.

Add future baselines only in ``build_method_registry()``. Their adapters should
expose a Manager subclass and a config-builder; do not create separate Real-World
runners with duplicated scenario values.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Type

import pandas as pd

import sim_phase
from simulator_manager import SimulatorManager
from global_comparison_metrics import (
    COMMON_METRICS_VERSION,
    METRIC_GROUPS,
    PRIMARY_COMPARISON_METRICS,
    attach_global_comparison_metrics,
    select_primary_comparison_table,
    summarize_global_comparison_results,
)

from njnp_baseline import (
    NJNPRealWorldSimulatorManager,
    NJNP_METHOD_NATIVE_KEYS,
    build_njnp_real_world_config,
)

from inma_realworld_adapter import (
    INMARealWorldSimulatorManager,
    INMA_METHOD_NATIVE_KEYS,
    build_inma_real_world_config,
)

from fuzzy2019_realworld_adapter import (
    Fuzzy2019RealWorldSimulatorManager,
    FUZZY2019_METHOD_NATIVE_KEYS,
    build_fuzzy2019_real_world_config,
)

from bcv2025_realworld_adapter import (
    BCV2025RealWorldSimulatorManager,
    BCV2025_METHOD_NATIVE_KEYS,
    build_bcv2025_real_world_config,
)

from abe_tbsra2025_realworld_adapter import (
    ABETBSRA2025RealWorldSimulatorManager,
    ABE_TBSRA_METHOD_NATIVE_KEYS,
    build_abe_tbsra2025_real_world_config,
)

from aref_trust_mcr2026_realworld_adapter import (
    ArefTrustMCR2026RealWorldSimulatorManager,
    AREF2026_METHOD_NATIVE_KEYS,
    build_aref_trust_mcr2026_real_world_config,
)

from tfsm_dpc2024_realworld_adapter import (
    TFSMDPC2024RealWorldSimulatorManager,
    TFSM_DPC_METHOD_NATIVE_KEYS,
    build_tfsm_dpc2024_real_world_config,
)

from fdtm_wwsn2024_realworld_adapter import (
    FDTMWWSN2024RealWorldSimulatorManager,
    FDTM_WWSN2024_METHOD_NATIVE_KEYS,
    build_fdtm_wwsn2024_real_world_config,
)

from mtaaha2024_realworld_adapter import (
    MTAHA2024RealWorldSimulatorManager,
    M_TAAHA2024_METHOD_NATIVE_KEYS,
    build_mtaaha2024_real_world_config,
)

from ac_dcs2026_realworld_adapter import (
    ACDCS2026RealWorldSimulatorManager,
    AC_DCS2026_METHOD_NATIVE_KEYS,
    build_ac_dcs2026_real_world_config,
)

from adec_spfc2026_realworld_adapter import (
    ADECSPFC2026RealWorldSimulatorManager,
    ADEC_SPFC2026_METHOD_NATIVE_KEYS,
    build_adec_spfc2026_real_world_config,
)

from mmcc2026_realworld_adapter import (
    MMCC2026RealWorldSimulatorManager,
    MMCC2026_METHOD_NATIVE_KEYS,
    build_mmcc2026_real_world_config,
)

from rl_ogsa_eeosp2026_realworld_adapter import (
    RL_OGSA_EEOSP2026RealWorldSimulatorManager,
    RL_OGSA_EEOSP2026_METHOD_NATIVE_KEYS,
    build_rl_ogsa_eeosp2026_real_world_config,
)

from csgp_realworld_adapter import (
    CSGPRealWorldSimulatorManager,
    CSGP_METHOD_NATIVE_KEYS,
    build_csgp_real_world_config,
)

from mmccs_realworld_adapter import (
    MMCCSRealWorldSimulatorManager,
    MMCCS_METHOD_NATIVE_KEYS,
    build_mmccs_real_world_config,
)

from medqn_realworld_adapter import (
    MEDQNRealWorldSimulatorManager,
    MEDQN_METHOD_NATIVE_KEYS,
    build_medqn_real_world_config,
)

from tarf_realworld_adapter import (
    TARFRealWorldSimulatorManager,
    TARF_METHOD_NATIVE_KEYS,
    build_tarf_real_world_config,
)

from flsga2025_realworld_adapter import (
    FLSGA2025RealWorldSimulatorManager,
    FLSGA2025_METHOD_NATIVE_KEYS,
    build_flsga2025_real_world_config,
)


from gtfsc_realworld_adapter import (
    GTFSCRealWorldSimulatorManager,
    GTFSC_METHOD_NATIVE_KEYS,
    build_gtfsc_real_world_config,
)


from qifl_dqn2026_realworld_adapter import (
    QIFLDQN2026RealWorldSimulatorManager,
    QIFL_DQN2026_METHOD_NATIVE_KEYS,
    build_qifl_dqn2026_real_world_config,
)

from fmgcp_realworld_adapter import (
    FMGCPRealWorldSimulatorManager,
    FMGCP_METHOD_NATIVE_KEYS,
    build_fmgcp_real_world_config,
)


from mcdm_fis2025_realworld_adapter import (
    MCDMFIS2025RealWorldSimulatorManager,
    MCDM_FIS2025_METHOD_NATIVE_KEYS,
    build_mcdm_fis2025_real_world_config,
)

from fldc2024_realworld_adapter import (
    FLDC2024RealWorldSimulatorManager,
    FLDC2024_METHOD_NATIVE_KEYS,
    build_fldc2024_real_world_config,
)

from nf_mnca2024_realworld_adapter import (
    NFMNCA2024RealWorldSimulatorManager,
    NF_MNCA2024_METHOD_NATIVE_KEYS,
    build_nf_mnca2024_real_world_config,
)

from etq2026_realworld_adapter import (
    ETQ2026RealWorldSimulatorManager,
    ETQ2026_METHOD_NATIVE_KEYS,
    build_etq2026_real_world_config,
)

from mfgml2026_realworld_adapter import (
    MFGML2026RealWorldSimulatorManager,
    MFGML2026_METHOD_NATIVE_KEYS,
    build_mfgml2026_real_world_config,
)

from mcss_osed2025_realworld_adapter import (
    MCSSOSED2025RealWorldSimulatorManager,
    MCSS_OSED2025_METHOD_NATIVE_KEYS,
    build_mcss_osed2025_real_world_config,
)

from fcbe2025_realworld_component import (
    FCBE2025_COMPONENT_PROTOCOL,
    run_fcbe2025_component_track,
)

# ============================================================================
# CENTRAL EVALUATION PROTOCOL
# ============================================================================
# These are evaluation-protocol settings, not duplicated physical parameters.
# The physical/radio/charging/attack values are loaded from sim_phase.py.

DEFAULT_NUM_NODES = 100
DEFAULT_NETWORK_SIZE = (100.0, 100.0)
DEFAULT_SEEDS = tuple(range(1, 21))

CANONICAL_MODE = "article_scaled"
CANONICAL_SCENARIO_NAME = "Article_Scaled_Absolute_Single"
PROTOCOL_VERSION = "RealWorld_CommonEnvironment_v3_CommonMetrics"

# The final proposed-method extension agreed in the thesis workflow.
PROPOSED_ADAPTIVE_MULTIMC_OVERRIDES: Mapping[str, object] = {
    "mobile_charger_selection_mode": "dynamic",
    "use_stable_charger_regions": True,
    "use_interference_safe_multi_mc_dispatch": True,
    "mobile_charger_safe_distance_m": 2.0,
    "max_adaptive_charge_queue": 18,
}
PROPOSED_METHOD_NATIVE_KEYS = frozenset(PROPOSED_ADAPTIVE_MULTIMC_OVERRIDES)


# ============================================================================
# METHOD CONTRACT
# ============================================================================

@dataclass(frozen=True)
class MethodSpec:
    """A method registered in the common Real-World evaluation.

    ``native_keys`` are the only canonical-config keys the method is allowed to
    change. This gives the global runner a fail-closed audit: accidental changes
    to physical/scenario parameters stop the run before simulation.

    ``deployment_policy`` is metadata only. ``common_random`` means this method
    can use the common random placement produced by SimulatorManager with the
    shared seed. A future method with a paper-specific placement should use
    ``method_native`` and implement that placement in its Manager subclass.

    ``not_applicable_metric_groups`` does not change any common metric formula.
    It only masks a metric family to NaN when a paper does not define the
    corresponding subsystem at all. For example, TARF is a WSN routing/security
    framework and publishes no charging architecture, so charging outcomes are
    N/A rather than artificial zeros.
    """

    label: str
    manager_class: Type[SimulatorManager]
    config_builder: Callable[[Mapping[str, object]], Dict[str, object]]
    native_keys: frozenset[str]
    deployment_policy: str = "common_random"
    comparison_domain: str = "full_system"
    not_applicable_metric_groups: Tuple[str, ...] = ()
    notes: str = ""


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _config_digest(config: Mapping[str, object]) -> str:
    payload = json.dumps(
        _jsonable(dict(config)), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_canonical_real_world() -> Tuple[str, Dict[str, object]]:
    """Load the single canonical Real-World scenario from the user's code."""
    scenarios = list(sim_phase._scenarios_for_mode(CANONICAL_MODE))
    matches = [(name, cfg) for name, cfg in scenarios if name == CANONICAL_SCENARIO_NAME]
    if len(matches) != 1:
        available = [name for name, _ in scenarios]
        raise RuntimeError(
            f"Expected exactly one {CANONICAL_SCENARIO_NAME!r}; available={available}"
        )

    name, cfg = matches[0]
    cfg = dict(cfg)

    # Structural checks only. No duplicated physical numbers are hard-coded here.
    required_markers = {
        "article_energy_profile_enabled",
        "article_scaled_single_scenario",
        "simulation_time_override_s",
        "initial_energy_override",
        "mobile_charger_capacity_mode",
        "mobile_charger_capacity",
        "mobile_charger_travel_cost_mode",
        "mobile_charger_travel_energy_cost_per_m",
        "charging_rate",
        "rf_charging_radius",
        "mobile_charger_speed",
        "radio_e_elec_j_per_bit",
        "radio_eps_fs_j_per_bit_m2",
        "radio_eps_mp_j_per_bit_m4",
        "event_rate_lambda",
        "critical_event_probability",
        "attack_percentage",
        "transmission_range",
    }
    missing = sorted(required_markers - set(cfg))
    if missing:
        raise RuntimeError(
            "Canonical Real-World scenario is missing expected fields: "
            + ", ".join(missing)
        )
    if cfg.get("article_energy_profile_enabled") is not True:
        raise RuntimeError("Canonical scenario is not marked as article energy profile.")
    if cfg.get("article_scaled_single_scenario") is not True:
        raise RuntimeError("Canonical scenario is not the article-scaled single profile.")

    return name, cfg


def build_proposed_final_real_world_config(
    canonical: Mapping[str, object],
) -> Dict[str, object]:
    """Final proposed method: canonical Real-World + five agreed MC extensions."""
    cfg = dict(canonical)
    cfg.update(PROPOSED_ADAPTIVE_MULTIMC_OVERRIDES)
    return cfg


def changed_keys(
    canonical: Mapping[str, object], method_config: Mapping[str, object]
) -> Set[str]:
    keys = set(canonical) | set(method_config)
    sentinel = object()
    return {
        key
        for key in keys
        if canonical.get(key, sentinel) != method_config.get(key, sentinel)
    }


def audit_method_config(
    spec: MethodSpec,
    canonical: Mapping[str, object],
    method_config: Mapping[str, object],
) -> Set[str]:
    """Fail if a method changes a non-native canonical parameter."""
    observed = changed_keys(canonical, method_config)
    unauthorized = observed - set(spec.native_keys)
    if unauthorized:
        details = []
        for key in sorted(unauthorized):
            details.append(
                f"{key}: canonical={canonical.get(key)!r}, method={method_config.get(key)!r}"
            )
        raise RuntimeError(
            f"[{spec.label}] unauthorized common-environment override(s):\n  "
            + "\n  ".join(details)
        )
    return observed


def apply_metric_applicability(
    row: Dict[str, object],
    spec: MethodSpec,
) -> Dict[str, object]:
    """Mask non-applicable subsystem metrics without changing formulas."""
    for group_name in spec.not_applicable_metric_groups:
        if group_name not in METRIC_GROUPS:
            raise RuntimeError(
                f"[{spec.label}] unknown common metric group: {group_name!r}"
            )
        for metric_name in METRIC_GROUPS[group_name]:
            if metric_name in row:
                row[metric_name] = float("nan")
    row["Global_Comparison_Domain"] = str(spec.comparison_domain)
    row["Global_Not_Applicable_Metric_Groups"] = "|".join(
        spec.not_applicable_metric_groups
    )
    return row


def build_method_registry() -> List[MethodSpec]:
    """The ONLY central list of methods executed under this Real-World protocol."""
    return [
        MethodSpec(
            label="Proposed_Final_RealWorld_AdaptiveMultiMC",
            manager_class=SimulatorManager,
            config_builder=build_proposed_final_real_world_config,
            native_keys=PROPOSED_METHOD_NATIVE_KEYS,
            deployment_policy="common_random",
            notes="Canonical Real-World final method plus the five agreed Adaptive Multi-MC extensions.",
        ),
        MethodSpec(
            label="Baseline_NJNP",
            manager_class=NJNPRealWorldSimulatorManager,
            config_builder=build_njnp_real_world_config,
            native_keys=NJNP_METHOD_NATIVE_KEYS,
            deployment_policy="common_random",
            notes="Paper-native NJNP scheduler/request/routing/charging architecture under the canonical external Real-World environment.",
        ),
        MethodSpec(
            label="Baseline_INMA",
            manager_class=INMARealWorldSimulatorManager,
            config_builder=build_inma_real_world_config,
            native_keys=INMA_METHOD_NATIVE_KEYS,
            deployment_policy="common_random",
            notes=(
                "Paper-native INMA request threshold, dynamic energy-rate "
                "estimator, Algorithm-1 scheduling, one-MC full contact charging "
                "under the canonical external Real-World environment."
            ),
        ),
        MethodSpec(
            label="Baseline_Fuzzy2019",
            manager_class=Fuzzy2019RealWorldSimulatorManager,
            config_builder=build_fuzzy2019_real_world_config,
            native_keys=FUZZY2019_METHOD_NATIVE_KEYS,
            deployment_policy="common_random",
            notes=(
                "Tomar-Muduli-Jana Fuzzy-2019: one MC, paper-native request "
                "threshold, 27-rule Mamdani scheduling, full simultaneous "
                "multi-node WET around the selected anchor, finite-MC return "
                "feasibility, and no proposed QIFL/DQN/security-defense stack."
            ),
        ),
        # MethodSpec(
        #     label="Baseline_BCV2025",
        #     manager_class=BCV2025RealWorldSimulatorManager,
        #     config_builder=build_bcv2025_real_world_config,
        #     native_keys=BCV2025_METHOD_NATIVE_KEYS,
        #     deployment_policy="common_random",
        #     comparison_domain="charging_security",
        #     notes=(
        #         "Yang-Lin-Deng et al. BCV-2025: single-MC full contact charging, "
        #         "native Em/Ec=0.4 requests, 3/2 metric-TSP scheduling with "
        #         "critical-node insertion, Algorithms 1/2 consensus validation, "
        #         "greedy validation-node selection, and 20-byte multi-layer "
        #         "blockchain storage. Canonical horizon/deployment/workload/attack/"
        #         "radio/sensor-MC absolute physics are retained; the paper's "
        #         "Section-VII absolute hardware values are not injected. Missing "
        #         "e/Delta-t/Ev values, exact cited TSP executable, first-block and "
        #         "in-flight replan details are explicit audited source gaps. No "
        #         "Proposed QIFL/DQN/fixed/RF/adaptive-partial/Adaptive-Multi-MC or "
        #         "Proposed security stack is transferred into BCV."
        #     ),
        # ),
        MethodSpec(
            label="Baseline_ABE_TBSRA2025",
            manager_class=ABETBSRA2025RealWorldSimulatorManager,
            config_builder=build_abe_tbsra2025_real_world_config,
            native_keys=ABE_TBSRA_METHOD_NATIVE_KEYS,
            deployment_policy="common_random",
            comparison_domain="routing_security",
            not_applicable_metric_groups=("charging", "overhead"),
            notes=(
                "Selvi et al. ABE-TBSRA-2025: LEACH-style clustered secure DATA "
                "routing, Eqs.5-11 direct/indirect/overall trust with the published "
                "OT>=0.65 gate, Eq.12 CH eligibility, trust/energy/centre-aware CH "
                "selection and symbolic Algorithm-1 ABE + bilinear-DH + RSA flow. "
                "The paper publishes no charging architecture, so common charging "
                "metrics are N/A. It also omits crypto/control packet sizes, so "
                "common operational packet/bit overhead is N/A rather than an "
                "invented value. Canonical horizon/deployment/workload/attacks/"
                "radio/energy remain unchanged; Table-1 200x200 m, 2 J, node-count "
                "sweep and Random-Waypoint settings are not forced into the common "
                "Real-World run. Unpublished alpha/f/lambda/weights/P/clustering "
                "details are explicit audited source-gap assumptions. The paper's "
                "contradictory columnar-transposition/RNS+ paragraph is not added "
                "because Algorithm 1 and the method definition specify ABE/DH/RSA."
            ),
        ),
        MethodSpec(
            label="Baseline_ArefTrustMCR2026",
            manager_class=ArefTrustMCR2026RealWorldSimulatorManager,
            config_builder=build_aref_trust_mcr2026_real_world_config,
            native_keys=AREF2026_METHOD_NATIVE_KEYS,
            deployment_policy="common_random",
            comparison_domain="charging_trust",
            notes=(
                "Aref-Omar-Aloqaily-Hassan Trust-MCR-2026: distributed local "
                "direct trust over sensor/robot charging interactions, gamma=0.5, "
                "30% request threshold, W=10 recent-history window, trust filtering, "
                "temporary Eq.2 exponential-backoff blacklisting, max-trust choice "
                "with random ties, per-robot FCFS queues, one-node-at-a-time full "
                "charging by reliable robots and paper-native virtually-infinite "
                "robot energy. The k=8 value is an explicit reference anchor from "
                "the paper's fixed-k experiments, while canonical absolute MC "
                "capacity/speed/travel cost are merely replicated across that fleet. "
                "The paper's fully-connected/no-location one-step environment and "
                "synthetic 50% unreliable-robot scenario are not injected into the "
                "Global Real-World run. Tmin, timeout and blacklist base B are "
                "unpublished; Tmin=0.5 is fail-visible and timeout/B are derived "
                "from canonical physics. Algorithm 1 line 19 conflicts with Sec.4.3 "
                "and Fig.2; the primary profile follows max-trust selection stated "
                "in Sec.4.3/Fig.2, with the literal random profile retained for "
                "sensitivity tests. The common security-resume track scores "
                "malicious SENSOR nodes versus the BS BLACKLISTED node set; this "
                "paper instead maintains sensor-local trust/blacklists over MCRs, "
                "so that sensor-IDS security table is intentionally N/A rather "
                "than populated with a misleading zero-detection row. No Proposed "
                "QIFL/DQN/fixed/RF-multinode/adaptive-partial/security-defense "
                "features are transferred."
            ),
        ),
        MethodSpec(
            label="Baseline_TFSM_DPC2024",
            manager_class=TFSMDPC2024RealWorldSimulatorManager,
            config_builder=build_tfsm_dpc2024_real_world_config,
            native_keys=TFSM_DPC_METHOD_NATIVE_KEYS,
            deployment_policy="common_random",
            comparison_domain="sensor_security",
            not_applicable_metric_groups=("charging", "overhead"),
            notes=(
                "Han-Wang-Li-Zhang TFSM-DPC-2024: sensor trust/security model "
                "with Eq.4 Beta-style initial direct trust, Eqs.5-7 "
                "volatilization plus adaptive penalty, Eqs.8-9 weighted indirect "
                "trust, Eqs.10-11 comprehensive trust, and improved K=3 DPC "
                "recommendation filtering using artificial upper/lower "
                "benchmarks (Eqs.12-14). Published initial trust=0.5, CTth=0.35, "
                "eta=0.5 and K=3 are retained. The paper publishes no charging "
                "architecture and no routing contribution; charging metrics are "
                "N/A and a neutral host BaselineRouting substrate carries the "
                "shared workload. Byte-level recommendation/control traffic is "
                "unpublished, so common operational packet/bit overhead is N/A. "
                "Canonical horizon/deployment/workload/attack/radio/energy are "
                "inherited; the paper's 100x100 m, 100-node, 25% attacker and "
                "cycle-20 attack settings are paper-reference only. The global "
                "run does not inject extra bad-mouth/collusion/on-off attacks; "
                "those are exercised only in paper-fidelity tests. Unpublished "
                "lambda, Eq.7 parameter a, exact benchmark values and cutoff "
                "percentile are explicit fail-visible source-gap assumptions. "
                "The common security-resume track is applicable because TFSM-DPC "
                "classifies malicious SENSOR nodes; its distributed exclusions "
                "are projected into the common final BLACKLISTED observer set. "
                "No Proposed QIFL/DQN/charging/security-risk features are transferred."
            ),
        ),
        MethodSpec(
            label="Baseline_FDTM_WWSN2024",
            manager_class=FDTMWWSN2024RealWorldSimulatorManager,
            config_builder=build_fdtm_wwsn2024_real_world_config,
            native_keys=FDTM_WWSN2024_METHOD_NATIVE_KEYS,
            deployment_policy="common_random",
            comparison_domain="sensor_security",
            not_applicable_metric_groups=("charging", "overhead"),
            notes=(
                "Wang-Liu-Jiang FDTM (TMC-2024): malicious SENSOR-node "
                "detection for WWSNs using communication/correct-forwarding/"
                "flow trust evidence (Eqs.10-12), interval type-2 fuzzy "
                "membership functions (Figs.2-3), FAHP weights (Eqs.15-25), "
                "initial/dynamic trust (Eqs.26-34), communication-condition "
                "perception (Eqs.35-38), and N=5 K-means trust-history "
                "features (Eq.39). The paper publishes no charging architecture, "
                "so common charging metrics are N/A; byte-level trust/control "
                "packet formats are not published, so common operational "
                "packet/bit overhead is N/A rather than invented. Canonical "
                "horizon/deployment/workload/attack/radio/energy remain the "
                "source of truth; the paper's 100x100-unit, 100-node, 20-unit "
                "communication range, central-sink and Table-II absolute "
                "radio/energy values are paper-reference only. Numeric fuzzy "
                "breakpoints are graph-digitized because the paper provides "
                "figures but no breakpoint table. FAHP priorities, dynamic "
                "delta/epsilon/alpha and complete K-means outlier details are "
                "explicit source gaps. Eq.28 prints v=(1.5,0.5,-1) for "
                "(low,medium,high), contradicting its text and the high-trust "
                "K-means semantics; the primary profile retains the printed "
                "coefficients but applies them to (high,medium,low), equivalent "
                "to (-1,0.5,1.5) on the published (low,medium,high) columns; "
                "the literal low/medium/high profile is retained for sensitivity tests. FDTM detects malicious SENSOR nodes, "
                "therefore the integrated common security-resume track applies. "
                "No Proposed QIFL/DQN/charging/security-fusion feature is transferred."
            ),
        ),
        MethodSpec(
            label="Baseline_M_TAAHA2024",
            manager_class=MTAHA2024RealWorldSimulatorManager,
            config_builder=build_mtaaha2024_real_world_config,
            native_keys=M_TAAHA2024_METHOD_NATIVE_KEYS,
            deployment_policy="common_random",
            comparison_domain="routing_clustering_security_mobile_sink",
            not_applicable_metric_groups=("charging", "overhead"),
            notes=(
                "Jemla-Naik-Parameswarappa-Ramachandra M-TAAHA (ETRI-2024): "
                "trust-aware Artificial Hummingbird Algorithm secure CH selection "
                "and secure multihop routing toward a paper-native mobile sink. "
                "Published trust equations DT/IDT/RT (Eqs.13-16), alpha=0.3, "
                "residual-energy/distance/CH-balancing objectives (Eqs.17-21), "
                "Eq.22 cluster potential, and trust/energy/sink-distance route "
                "objective (Eq.23) are retained. The paper publishes no charging "
                "architecture, therefore common charging metrics are N/A; it also "
                "does not publish byte-level trust/clustering control packets, so "
                "common operational packet/bit overhead is N/A rather than invented. "
                "Canonical horizon, common random sensor deployment, workload, attack "
                "realization, radio/energy physics and seeds remain source-of-truth; "
                "Table-1 200x200 m, 0.55 J, 4000-bit packet and absolute radio values "
                "are paper-reference only. The MS trajectory/speed, SCH count, AHA "
                "population/iterations, delta/mu objective weights, trust threshold, "
                "and discrete AHA projection are unpublished and remain explicit "
                "source gaps. Eq.20 is zero-sum for a complete partition and Eqs.21/23 "
                "place favorable trust/energy terms in a minimization objective; the "
                "primary profile performs direction-only normalized repairs while a "
                "literal printed-equation profile is retained for sensitivity. The "
                "method detects/avoids malicious SENSOR nodes, so the integrated "
                "common security-resume track applies. No Proposed QIFL/DQN/charging/"
                "security-fusion feature is transferred."
            ),
        ),
        MethodSpec(
            label="Baseline_AC_DCS2026",
            manager_class=ACDCS2026RealWorldSimulatorManager,
            config_builder=build_ac_dcs2026_real_world_config,
            native_keys=AC_DCS2026_METHOD_NATIVE_KEYS,
            deployment_policy="common_random",
            comparison_domain="charging_routing_heterogeneous_sensors",
            not_applicable_metric_groups=("overhead",),
            notes=(
                "Lin-Huang-Sung-Tsai AC-DCS (IoT-J 2026): heterogeneous WRSN "
                "dual-mode charging with a single one-to-one MC selected by the "
                "online DARS neural-UCB policy and deploy/recycle directional "
                "portable chargers for fixed-sensor clusters. Mean-shift "
                "clustering (Alg.1), urgency/cluster valuation (Eqs.12-14), "
                "PCDP geometric deployment plus RSDEA/DE (Eqs.15-20, Algs.2-4), "
                "DARS two-layer ReLU/UCB and sliding-window online update "
                "(Algs.5-6), and energy-aware adaptive multihop routing (Eqs.5-7) "
                "are method-native. Canonical horizon/total-node field/workload/"
                "attacks/radio/sensor energy/MC capacity-speed-travel cost remain "
                "source-of-truth. Table-II 200x200 m, 100-J, 3-W, 25-J/m and "
                "absolute 50-mW/3-m PC hardware are paper-reference only; the "
                "Global PC rate preserves only the published 50mW:3W ratio and "
                "uses the canonical RF radius. The paper fixes 30 dynamic sensors "
                "in its main experiments and states random-walk mobility but omits "
                "the step distribution; the adapter retains a 30-node paper "
                "reference partition and derives random-walk step scale from the "
                "canonical transmission range. Because the paper itself uses uniform-"
                "random initial placement, the common master-seed deployment is reused; "
                "only the FS/DS partition and subsequent DS random walk are method-native. "
                "Eth, Tsafe, DARS reward/H/W/mu/LR, "
                "cluster-value weights and DE F/CR are explicit source gaps. "
                "PC receive power preserves Eq.(3) loss via a scale-free receive/source "
                "fraction rather than renormalizing away distance/angle loss. DRV travel "
                "uses canonical mobility physics with a finite trip budget; its unpublished "
                "modified-TSP details are mapped to deterministic NN+2-opt. "
                "Algorithm-1's printed convergence test is tautological after its "
                "assignment; the primary implementation follows the prose until "
                "no position change, with a literal profile retained. AC-DCS has "
                "no malicious-node detection contribution and is therefore not "
                "entered in the common security table. Control-message byte sizes "
                "are unpublished, so common operational packet/bit overhead is N/A. "
                "No Proposed QIFL/DQN/fixed-charger/adaptive-partial/Adaptive-Multi-MC "
                "or Proposed security stack is transferred."
            ),
        ),
        # MethodSpec(
        #     label="Baseline_ADEC_SPFC2026",
        #     manager_class=ADECSPFC2026RealWorldSimulatorManager,
        #     config_builder=build_adec_spfc2026_real_world_config,
        #     native_keys=ADEC_SPFC2026_METHOD_NATIVE_KEYS,
        #     deployment_policy="common_random",
        #     comparison_domain="charging_directional_partial",
        #     not_applicable_metric_groups=("overhead",),
        #     notes=(
        #         "Lin-Yu-Sung-Tsai ADEC/SPFC (Sensors-J 2026): one finite-energy "
        #         "MC with directional one-to-many WPT, Eq.5 distance-aware dynamic "
        #         "request thresholds, Eq.6 demand-aware partial charging, mean-shift "
        #         "charging points (Alg.1), angle-set-cover greedy directional dwell "
        #         "scheduling (Alg.2), SPFC nearest-cluster maintenance (Alg.3), ADEC "
        #         "adaptive critical-set refinement (Alg.4), and permutation DE with "
        #         "order-preserving crossover/swap mutation plus the printed battery "
        #         "fitness simulation (Algs.5-6). Regime selection follows Eq.11: a "
        #         "causal current-state projection of SPFC availability activates ADEC "
        #         "when RA<1 and SPFC when RA=1; no future workload trace is inspected. "
        #         "Canonical horizon/deployment/workload/attacks/radio/sensor-energy/"
        #         "MC capacity-speed-travel cost/absolute charging scale remain the "
        #         "source of truth. Table-I 200x200 m, 10.8-kJ sensors, 2000-kJ MC, "
        #         "5-W source, 250-W movement power, 5-m radius and 0.01-0.05-W synthetic "
        #         "drain are paper-reference only. The paper's power-law/ER/BA topologies "
        #         "are evaluation scenarios, not a required deployment, so the common "
        #         "master-seed deployment is retained. Published directional mu/beta/c, "
        #         "epsilon=0.1, angle grid 30..180 deg step 15 deg and ADEC 500/500, "
        #         "F=0.3, CR=0.8 are retained. Unpublished mean-shift kernel K, rank "
        #         "probability formula, exact OPC construction, hi value, pre-dispatch "
        #         "RA estimator and SS recharge time are explicit audited source gaps. "
        #         "Eq.4's Powercast-calibrated output is treated on its empirical mW scale "
        #         "before conversion to a receive/source fraction, avoiding false saturation; "
        #         "finite MC battery and return-to-SS feasibility are applied in both path "
        #         "fitness projection and physical execution. "
        #         "The method has no malicious-node detector and no published byte-level "
        #         "charging-control formats, so common sensor-security is N/A and common "
        #         "operational packet/bit overhead is N/A. No Proposed QIFL/DQN/fixed "
        #         "charger/dynamic-partial/Adaptive-Multi-MC/security stack is transferred."
        #     ),
        # ),
        # MethodSpec(
        #     label="Baseline_MMCC2026",
        #     manager_class=MMCC2026RealWorldSimulatorManager,
        #     config_builder=build_mmcc2026_real_world_config,
        #     native_keys=MMCC2026_METHOD_NATIVE_KEYS,
        #     deployment_policy="common_random",
        #     comparison_domain="charging_multi_mc_hybrid_cooperative",
        #     not_applicable_metric_groups=("overhead",),
        #     notes=(
        #         "Li-Sha-Xia-Malekian-Lin-Wang MMCC (IoT-J 2026): paper-native "
        #         "multi-MC cooperative charging with DoI/load/distance-balanced node "
        #         "groups and exactly three coordinated roles per constructed group: "
        #         "MCP periodic service, MCR on-demand FCFS service, and MCO always-online "
        #         "request-first/DoI service. Eqs.6-11 DoI, Eqs.12-16 MCR capacity/group "
        #         "radius, Algorithm 1/Eqs.17-20 grouping, Eq.24 request threshold, Eq.25 "
        #         "finite service+return feasibility, the two-pass MCP loop construction "
        #         "with the explicit 50%-capacity per-visit replenishment amount, MCR "
        #         "latest-departure/Eq.29 filtering and Eq.30 passer-by charging, MCO "
        #         "dynamic request promotion, and Eqs.31-32 public node locking are retained. "
        #         "Canonical horizon, common random deployment, workload, attack realization, "
        #         "radio/sensor energy and per-MC capacity-speed-travel-cost/charging scale "
        #         "remain source-of-truth; the runtime fleet merely replicates those canonical "
        #         "per-MC physics to three roles per paper-constructed group. Table-IV 200x200 m, "
        #         "N=150, 0.3-1 W drain, 2000/30000-mAh batteries, 2 m/s, 5-W movement and "
        #         "eta=0.8 are paper-reference only. Eq.20 mu1/mu2/mu3, the exact Rg choice "
        #         "inside Eq.16, TSP solver, PPR epsilon, unlock delta and Eq.1 WPT constants "
        #         "are unpublished; the primary profile uses explicit audited mappings: "
        #         "published experimental mu=4 for all load tiers, midpoint Rg, deterministic "
        #         "NN+2-opt, epsilon=0.5 and delta=0.75. Eq.24 mixes mAh and J/s in the paper; "
        #         "Global keeps the dimensionally consistent continuous-joule form rather than "
        #         "inventing a voltage/rounding conversion. MMCC has no malicious-node detector, "
        #         "so common sensor-security is N/A. Control/lock message byte sizes are not "
        #         "published, so common operational packet/bit overhead is N/A. No Proposed "
        #         "QIFL/DQN/fixed/RF-multinode/adaptive-partial/Adaptive-Multi-MC/security stack "
        #         "is transferred."
        #     ),
        # ),
        MethodSpec(
            label="Baseline_RL_OGSA_EEOSP2026",
            manager_class=RL_OGSA_EEOSP2026RealWorldSimulatorManager,
            config_builder=build_rl_ogsa_eeosp2026_real_world_config,
            native_keys=RL_OGSA_EEOSP2026_METHOD_NATIVE_KEYS,
            deployment_policy="method_native",
            comparison_domain="routing_clustering_mobile_sink_trust",
            not_applicable_metric_groups=("charging",),
            notes=(
                "Askarizade-Kuchaki-Rafsanjani-Fanian RL-OGSA-EEOSP (IJSWIS 2026): "
                "paper-native Voronoi/Lloyd deployment followed by independent Q-learning "
                "edge-node tuning, adaptive cluster-count Eq.10, discrete OGSA+opposition "
                "cluster-head selection, five-trigger re-clustering, Monte-Carlo EEOSP "
                "mobile-sink placement Eq.17, and hybrid one-/multi-hop greedy routing with "
                "Eq.18 distance/residual-energy/load/trust cost. Published Table-5 values "
                "Wd/We/Wl/Wt=0.5/0.2/0.15/0.15, Kmin=2, Kmax=10, OGSA alpha/beta=0.6/0.4, "
                "15 iterations, G0=50, population=10, recluster period=75, energy threshold "
                "0.1, load threshold=10, sink-move threshold=10 and visit period=5 are retained. "
                "The official simulator defines one native algorithmic round as 50 s; therefore "
                "these round-indexed scheduler periods and the 0.5 m/s mobile-sink speed are mapped "
                "to that 50-s method cadence, not to the unrelated Global attack/control epoch. "
                "The supplied official implementation is archived in reference_sources and "
                "audited against the article. Stale/inconsistent source behavior is repaired rather "
                "than silently copied: OGSA population initialization and Q-learning action choice "
                "reseed their RNG on every call; the top-level Simulation wrapper defaults routing "
                "weights to 0.5/0.3/0.1/0.1, cluster bounds to 8/20 and sink-move threshold to 20 m "
                "despite Table 5 specifying 0.5/0.2/0.15/0.15, 2/10 and 10 m. EEOSP target sampling "
                "is also moved to the independent policy RNG so successive visits remain stochastic "
                "but reproducible. Paper Table 5 is internally inconsistent for EEOSP weights "
                "(it prints 0.6 and 0.6); the official code's complementary 0.4 fairness/0.6 distance "
                "pair is retained and audit-tagged. The re-clustering prose also says CH-to-sink "
                "distance while Table 5 and official code define a sink-move threshold; the primary "
                "profile follows the Table-5/official-code sink-displacement semantics and records "
                "the prose conflict. "
                "Canonical random coordinates are the reproducible starting deployment, after "
                "which the paper-native Voronoi/RL placement is applied; paper experiment field "
                "sizes, node counts, 0.5-J batteries, 4000-bit packets, 50-m communication range "
                "and corner/top-right BS are not forced into Global. The method is an MWSN data-"
                "collection/routing framework and publishes no wireless-charging architecture, "
                "so charging metrics are N/A. The official source does define 64-bit clustering/"
                "re-clustering control packets, therefore operational overhead is measured by "
                "the unchanged common observer instead of masked or invented. Trust affects route "
                "cost but the paper/code provides no comparable malicious-sensor final blacklist "
                "classifier, so the common sensor-IDS security table is N/A. No Proposed charging, "
                "QIFL/DQN, multipath, or security-fusion feature is transferred."
            ),
        ),
        MethodSpec(
            label="Baseline_CSGP2021",
            manager_class=CSGPRealWorldSimulatorManager,
            config_builder=build_csgp_real_world_config,
            native_keys=CSGP_METHOD_NATIVE_KEYS,
            deployment_policy="common_random",
            notes=(
                "Priyadarshani-Tomar-Jana CSGP-2021: paper-native multi-MCV/"
                "multi-depot charging tours, NSGA-II non-dominated fronts, "
                "AHP-TOPSIS ordering, multi-node dynamic partial charging, "
                "normalized heterogeneous sensor capacities, and no proposed "
                "QIFL/DQN/security-defense stack. The paper's missing dynamic "
                "request-threshold formula is explicitly handled by a common "
                "generic E_critical proxy rather than an invented equation."
            ),
        ),
        MethodSpec(
            label="Baseline_MMCCS2022",
            manager_class=MMCCSRealWorldSimulatorManager,
            config_builder=build_mmccs_real_world_config,
            native_keys=MMCCS_METHOD_NATIVE_KEYS,
            deployment_policy="common_random",
            notes=(
                "Jia-Wang-Ji-Peng MMCCS-2022: four-MC static service "
                "partitioning plus dynamic collaboration, improved K-means, "
                "three concentric request-threshold zones, RCA/CEM assignment, "
                "energy-hole-aware online next-node selection, one-to-one "
                "full charging, return feasibility, and dead-node regression. "
                "Published Eq.7/Eq.18/Eq.19 inconsistencies use the explicit "
                "paper-text-consistent profile; undefined dc and missing E2 "
                "threshold equation are recorded as source limitations."
            ),
        ),
        # MethodSpec(
        #     label="Baseline_MEDQN2024",
        #     manager_class=MEDQNRealWorldSimulatorManager,
        #     config_builder=build_medqn_real_world_config,
        #     native_keys=MEDQN_METHOD_NATIVE_KEYS,
        #     deployment_policy="method_native",
        #     notes=(
        #         "Li-Wang-Jiang-Xiao MEDQN-2024: one-MC online full charging, "
        #         "paper request threshold, paper-specific sensing deployment, "
        #         "QSC/SRNR reward, full-network state, experience replay, target "
        #         "network and multistage epsilon exploration. Unpublished "
        #         "epsilon decay/activation/optimizer/normalization/CS coordinate "
        #         "choices are explicit in the adapter. Offline training cycles "
        #         "begin at the first native request, matching the paper's "
        #         "charging-cycle definition rather than a trivial return-only reset."
        #     ),
        # ),
        MethodSpec(
            label="Baseline_TARF2012",
            manager_class=TARFRealWorldSimulatorManager,
            config_builder=build_tarf_real_world_config,
            native_keys=TARF_METHOD_NATIVE_KEYS,
            deployment_policy="common_random",
            comparison_domain="routing_security",
            not_applicable_metric_groups=("charging",),
            notes=(
                "Zhan-Shi-Deng TARF-2012: trust-aware single-path DATA routing "
                "with periodic authenticated-root delivery feedback, "
                "EnergyWatcher, TrustManager and Fig.5 trust/energy next-hop "
                "selection. TARF is a WSN routing/security framework and "
                "publishes no charging architecture; charging service is "
                "therefore disabled and common charging metrics are N/A. "
                "Published period/trust/table limits are retained; unpublished "
                "Fig.5 thresholds and control-payload sizes remain explicit "
                "audited source-gap assumptions."
            ),
        ),
        MethodSpec(
            label="Baseline_FLSGA2025",
            manager_class=FLSGA2025RealWorldSimulatorManager,
            config_builder=build_flsga2025_real_world_config,
            native_keys=FLSGA2025_METHOD_NATIVE_KEYS,
            deployment_policy="common_random",
            notes=(
                "Punriboon et al. FLSGA-2025: one-MC centralized adaptive "
                "request/pooling (FLS_RT), GA-evolved fuzzy route selection "
                "(FLS_MC), proportional-fair adaptive partial contact charging "
                "(FLS_CD), finite shared MC battery and CTP-style data routing. "
                "The common Real-World physical scale is inherited unchanged; "
                "the paper's Table-6 hardware numbers are not injected. "
                "Unpublished GA/normalization/DRR/COG/EPS details and the "
                "Eq.9/Eq.10/Eq.14 publication ambiguities are explicitly "
                "audited in the adapter and fidelity notes."
            ),
        ),
        MethodSpec(
            label="Baseline_GTFSC2024",
            manager_class=GTFSCRealWorldSimulatorManager,
            config_builder=build_gtfsc_real_world_config,
            native_keys=GTFSC_METHOD_NATIVE_KEYS,
            deployment_policy="common_random",
            comparison_domain="routing_security",
            not_applicable_metric_groups=("charging", "overhead"),
            notes=(
                "Ravale-Borkar GTFSC-2024: five-factor trust, Fuzzy C-Means "
                "malicious-node classification, EGT-labelled max-fitness/"
                "proximity clustering and GWO single-path source-to-sink "
                "routing. GTFSC is a MANET routing/security protocol and "
                "publishes no charging architecture, so charging is N/A. "
                "Its paper reports overhead but does not publish packet-level "
                "control-message semantics compatible with the common packet/"
                "bit observer, so common overhead is N/A rather than host-WRSN "
                "traffic or artificial zero. Canonical attacks are inherited; "
                "the paper's 30% four-type attack mix is reserved for a "
                "paper-reference/security-suite run. Mobility is described but "
                "not specified algorithmically, so no motion model is invented."
            ),
        ),
        # MethodSpec(
        #     label="Baseline_QIFL_DQN2026",
        #     manager_class=QIFLDQN2026RealWorldSimulatorManager,
        #     config_builder=build_qifl_dqn2026_real_world_config,
        #     native_keys=QIFL_DQN2026_METHOD_NATIVE_KEYS,
        #     deployment_policy="common_random",
        #     notes=(
        #         "Goyal-Tomar QIFL-DQN-2026: EBHO load-aware multi-MC "
        #         "partitioning/initial positioning, QIFL adaptive request "
        #         "thresholds, per-partition multi-agent DQNs, proximity-aware "
        #         "AGA cooperation and full simultaneous Friis multi-node WET. "
        #         "The common Real-World sensor/MC hardware scale is retained; "
        #         "paper Table-3 hardware numbers are not injected. M=3 is the "
        #         "lower published endpoint of the paper's 3..7 range because "
        #         "no node-count-to-M rule is published. Source gaps in EBHO "
        #         "fitness, QIFL scaling, DQN synchronization/epsilon, AGA "
        #         "constants and Friis constants remain explicit audit metadata. "
        #         "Dead-node revival is disabled because the paper does not "
        #         "publish a resurrection mechanism."
        #     ),
        # ),
        MethodSpec(
            label="Baseline_FMGCP2025",
            manager_class=FMGCPRealWorldSimulatorManager,
            config_builder=build_fmgcp_real_world_config,
            native_keys=FMGCP_METHOD_NATIVE_KEYS,
            deployment_policy="common_random",
            comparison_domain="routing_clustering",
            not_applicable_metric_groups=("charging",),
            notes=(
                "Mohan et al. FMGCP-2025: fuzzy-median-graph cluster-head "
                "selection, Prim MST conversion, all end-node median triples, "
                "ceil(d/2) candidate expansion, high-residual-energy/low-"
                "neighbor-distance CH selection, least-hop cluster joining, "
                "induced-MST intra/inter-cluster multi-hop DATA routing, "
                "10% CH re-clustering/relay rerouting and paper control-packet "
                "accounting. FMGCP publishes no charging architecture, so "
                "canonical idle depletion is retained while charging service "
                "is disabled and common charging metrics are N/A. The paper "
                "does not publish concrete f(d,RE), h(RE), CH scalarization or "
                "disconnected-network handling; these remain explicit audited "
                "source-gap assumptions rather than hidden paper constants."
            ),
        ),
        MethodSpec(
            label="Baseline_MCDM_FIS2025",
            manager_class=MCDMFIS2025RealWorldSimulatorManager,
            config_builder=build_mcdm_fis2025_real_world_config,
            native_keys=MCDM_FIS2025_METHOD_NATIVE_KEYS,
            deployment_policy="common_random",
            notes=(
                "Abdel Aziz et al. MCDM-FIS-2025: on-demand multi-WCV "
                "dynamic clustering, full one-to-multi modified-Friis WPT, "
                "five-input Mamdani FIS (D/RE/ND/TR/CR), centroid priority, "
                "finite-WCV charge-and-return feasibility and no Proposed "
                "QIFL/DQN/fixed-charger/dynamic-partial/security stack. "
                "Canonical hardware/workload/attack values are inherited. "
                "The PDF prints only eight of 243 rules; the main profile "
                "preserves those rows and fills only the unprinted rows from "
                "the paper's condition-count prose. Unpublished Friis/CR/ND/"
                "clustering constants are explicit audit metadata."
            ),
        ),
        MethodSpec(
            label="Baseline_FLDC2024",
            manager_class=FLDC2024RealWorldSimulatorManager,
            config_builder=build_fldc2024_real_world_config,
            native_keys=FLDC2024_METHOD_NATIVE_KEYS,
            deployment_policy="common_random",
            notes=(
                "Ma-Sha-Wang-Wang-Wang FLDC-2024: periodic residual-lifetime "
                "Qnode admission, paper-formula multi-MC fleet sizing, regular "
                "hexagonal NB-VG/B-VG service regions, two-layer Mamdani FIS, "
                "OLSP staying-point optimization, DCD/DCD_A directional cover "
                "sets, full NB one-to-many/B zero-distance one-to-one charging, "
                "and finite-MC energy/time return guards. Canonical workload, "
                "attack, sensor/MC hardware scale, charging radius and common "
                "data-routing layer are retained where direct counterparts "
                "exist. The paper has no single default theta, OLSP numerics, "
                "rotation cost or exact concurrent multi-MC tie policy; these "
                "remain explicit source-gap audit metadata. FLDC consumes "
                "control-channel reported energy rather than oracle battery "
                "truth and receives no Proposed DQN/QIFL/security stack."
            ),
        ),
        # MethodSpec(
        #     label="Baseline_NF_MNCA2024",
        #     manager_class=NFMNCA2024RealWorldSimulatorManager,
        #     config_builder=build_nf_mnca2024_real_world_config,
        #     native_keys=NF_MNCA2024_METHOD_NATIVE_KEYS,
        #     deployment_policy="common_random",
        #     notes=(
        #         "Hingoliwala-Kumar-Nayyar-Swain NF-MNCA-2024: on-demand "
        #         "adaptive thresholds, optimal FCM multi-region allocation, "
        #         "regional MCV depots, FOCS-style visiting-point reduction, "
        #         "full simultaneous multi-node charging, 4-input/81-rule "
        #         "Gaussian first-order-Sugeno ANFIS with PSO tuning, AWF "
        #         "tie-breaking, and full charge-and-return feasibility. "
        #         "Canonical workload/attack/radio/sensor-MCV hardware scale, "
        #         "charging radius/efficiency and common DATA-routing layer are "
        #         "retained. Table-6 duplicate antecedents are repaired only to "
        #         "restore the required 3^4 Cartesian rule base while preserving "
        #         "the published CSP sequence. The ANFIS dataset/consequents, "
        #         "exact FOCS pseudocode, validity-index combination, PSO "
        #         "population/exact acceleration coefficients, and regional-MCV "
        #         "recharge time are unpublished and remain explicit audited "
        #         "source gaps. Decisions use control-channel reported energy, "
        #         "not oracle battery truth, and no Proposed DQN/QIFL/security "
        #         "stack is transferred."
        #     ),
        # ),
        MethodSpec(
            label="Baseline_ETQ2026",
            manager_class=ETQ2026RealWorldSimulatorManager,
            config_builder=build_etq2026_real_world_config,
            native_keys=ETQ2026_METHOD_NATIVE_KEYS,
            deployment_policy="common_random",
            notes=(
                "Hou-Guo ETQ-2026: one-MC on-demand complete RF charging, "
                "uneven energy/distance/density-aware clustering, greedy "
                "intra-cluster energy trading with rollback, trade-dependent "
                "request priority, normalized urgency/consumption/distance "
                "reward and tabular epsilon-greedy Q-learning. Table-3 absolute "
                "hardware values are not injected; 10 J/30 J thresholds are "
                "preserved as 10%/30% battery ratios on canonical Emax. The "
                "paper does not publish candidate probability t, epsilon/M/"
                "convergence details, numeric priority mapping, full Eq.-4 RF "
                "constants, exact CH-to-BS route selection, buffer API or "
                "control payload sizes. Those remain explicit source gaps. "
                "Eq.9 is repaired only in the direction required by its prose "
                "buyer-overcharge constraint. Scheduling uses reported control "
                "energy and receives no Proposed DQN/QIFL/security stack."
            ),
        ),
        MethodSpec(
            label="Baseline_MFGML2026",
            manager_class=MFGML2026RealWorldSimulatorManager,
            config_builder=build_mfgml2026_real_world_config,
            native_keys=MFGML2026_METHOD_NATIVE_KEYS,
            deployment_policy="common_random",
            notes=(
                "Fanian-Kuchaki-Rafsanjani-Borumand-Saeid MFGML-2026: "
                "centralized cluster-based multi-MC WRSN with dedicated fixed "
                "zones, three stage-specific ML priority models for clustering, "
                "CH forwarding and charging scheduling, on-demand requests, "
                "multi-node WPT and AHA-derived partial-charge semantics. "
                "The paper varies MC count as a scenario feature and publishes "
                "no unseen-scenario fleet-sizing equation, so canonical fleet "
                "availability/capacity/speed/travel cost are inherited. Table-2 "
                "absolute hardware values are not injected. Author SVR model "
                "artifacts and the 10,000-sample stage datasets are unavailable; "
                "the executable uses three deterministic offline RBF-SVR "
                "surrogates trained from the published Eq.5/Eq.8/Eq.9 priority "
                "directions and Table-4 AHA anchor profiles. Exact unseen-NTest "
                "zoning/control mapping, fuzzy breakpoints, feature formulas and "
                "interference rule remain explicit source gaps. Decisions use "
                "reported control energy and no Proposed DQN/QIFL/security or "
                "interference-safe-dispatch implementation is transferred."
            ),
        ),
        MethodSpec(
            label="Baseline_MCSS_OSED_MEQA2025",
            manager_class=MCSSOSED2025RealWorldSimulatorManager,
            config_builder=build_mcss_osed2025_real_world_config,
            native_keys=MCSS_OSED2025_METHOD_NATIVE_KEYS,
            deployment_policy="method_native",
            notes=(
                "Li-Wang-Zhang-Kong-Xiao MCSS-OSED/MEQA-2025: one-MC offline "
                "full-network charging sequence triggered by the native 30% "
                "request ratio, paper-specific no-four-way-overlap deployment, "
                "probabilistic sensing/PDEO SCR+CDP QED objective, tabular "
                "Q-learning with multistage exploration and periodic experience "
                "strengthening, one-to-one full contact charging and return to CS. "
                "Canonical sensor/MC hardware, workload, attacks, horizon and "
                "radio/data environment are retained; the paper's absolute Table-4 "
                "hardware values are not injected. Numerical PDEO heatmaps and "
                "epsilon/error/experience details are unpublished and therefore "
                "explicit deterministic reconstructions. Eq.5, Algorithm-1, Eq.12 "
                "and Eq.14 publication inconsistencies use text/figure-consistent "
                "profiles. Policy input uses reported control energy and causal "
                "cycle-start depletion-rate estimates only; no future workload and "
                "no Proposed QIFL/DQN/security/multi-MC features are transferred."
            ),
        ),
    ]


# ============================================================================
# SHARED EXOGENOUS WORKLOAD
# ============================================================================

def generate_shared_workload_trace(
    path: Path,
    *,
    seed: int,
    num_nodes: int,
    sim_time: float,
    canonical_config: Mapping[str, object],
) -> None:
    """Generate a method-independent sensing workload for one scenario/seed.

    The RNG stream and event semantics mirror the original SimulatorManager,
    except the source ID is selected from the full deployed node-ID set instead
    of a method-dependent alive-set. This is required for a paired common
    workload: one method's lifetime must not alter another method's input trace.
    """
    event_rate = max(float(canonical_config["event_rate_lambda"]), 1e-12)
    critical_probability = float(canonical_config["critical_event_probability"])
    rng = random.Random(int(seed) + 100_003)

    events: List[Dict[str, object]] = []
    event_time = rng.expovariate(event_rate)
    while event_time < float(sim_time) - 1e-12:
        source_id = int(rng.randint(1, int(num_nodes)))
        is_critical = rng.random() < critical_probability
        events.append(
            {
                "time": float(event_time),
                "source_id": source_id,
                "raw_data": float(80.0 if is_critical else 25.0),
                # Kept as explicit metadata for audit/future adapters. The
                # current SimulatorManager consumes raw_data as before.
                "critical": bool(is_critical),
            }
        )
        event_time += rng.expovariate(event_rate)

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + ".tmp")
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(events, handle, indent=2)
    os.replace(temp, path)


# ============================================================================
# EXECUTION
# ============================================================================

def _env_int(name: str, default: int) -> int:
    return int(float(os.environ.get(name, default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_seed_list() -> Tuple[int, ...]:
    raw = os.environ.get("WRSN_EVAL_SEEDS", "").strip()
    if raw:
        values = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
        if not values:
            raise ValueError("WRSN_EVAL_SEEDS is empty after parsing.")
        return values
    count = _env_int("WRSN_NUM_SEEDS", len(DEFAULT_SEEDS))
    return tuple(range(1, count + 1))


def _atomic_write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + ".tmp")
    df.to_csv(temp, index=False)
    os.replace(temp, path)


def _safe_make_plots(df: pd.DataFrame, plot_dir: Path) -> None:
    if str(os.environ.get("WRSN_SKIP_PLOTS", "0")).strip().lower() in {
        "1", "true", "yes", "on"
    }:
        return
    try:
        sim_phase.make_plots(df, str(plot_dir))
    except Exception as exc:
        print(f"[PLOT WARNING] {exc}")


def run_registered_methods():
    scenario_name, canonical = load_canonical_real_world()

    # Scientific evaluation-contract values are locked HERE, not in PowerShell
    # and not inside individual baseline files. Change them only in this global
    # protocol when intentionally defining a different experiment.
    num_nodes = int(DEFAULT_NUM_NODES)
    network_size = (float(DEFAULT_NETWORK_SIZE[0]), float(DEFAULT_NETWORK_SIZE[1]))
    seeds = tuple(int(seed) for seed in DEFAULT_SEEDS)

    # Operational horizon is read ONLY from the canonical scenario.
    sim_time = int(float(canonical["simulation_time_override_s"]))

    output_dir = Path(
        os.environ.get(
            "WRSN_BASELINE_OUTPUT_DIR",
            "realworld_common_baseline_results",
        )
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = output_dir / "shared_workload_traces"
    results_csv = output_dir / "realworld_common_metrics_results.csv"
    summary_csv = output_dir / "realworld_common_metrics_summary_ci.csv"
    primary_csv = output_dir / "realworld_common_primary_comparison.csv"
    manifest_json = output_dir / "realworld_common_manifest.json"

    registry = build_method_registry()

    # Temporary baseline-only execution mode.  Keep the proposed method in
    # build_method_registry() so the contract tests continue to validate the
    # complete scientific registry, but remove it from the actual experiment
    # when WRSN_SKIP_PROPOSED is enabled by the PowerShell runner.
    skip_proposed = str(os.environ.get("WRSN_SKIP_PROPOSED", "0")).strip().lower() in {
        "1", "true", "yes", "on"
    }
    if skip_proposed:
        proposed_label = "Proposed_Final_RealWorld_AdaptiveMultiMC"
        registry = [spec for spec in registry if spec.label != proposed_label]
        print(f"[MODE] Baseline-only run: SKIP {proposed_label}")

    labels = [spec.label for spec in registry]
    if len(labels) != len(set(labels)):
        raise RuntimeError("Duplicate method labels in global registry.")

    built_configs: Dict[str, Dict[str, object]] = {}
    audit_changes: Dict[str, List[str]] = {}
    for spec in registry:
        cfg = spec.config_builder(canonical)
        observed = audit_method_config(spec, canonical, cfg)
        built_configs[spec.label] = cfg
        audit_changes[spec.label] = sorted(observed)

    canonical_digest = _config_digest(canonical)

    # Resume only exact Method+Seed rows from this protocol/scenario/geometry/config.
    rows: List[Dict[str, object]] = []
    completed: Set[Tuple[str, int]] = set()
    if results_csv.exists() and results_csv.stat().st_size > 0:
        existing = pd.read_csv(results_csv)
        required = {
            "Method",
            "Seed",
            "Global_Protocol_Version",
            "Global_Source_Scenario",
            "Global_Node_Count",
            "Global_Network_Width_m",
            "Global_Network_Height_m",
            "Common_Metrics_Version",
            "Global_Canonical_Config_SHA256",
            "Method_Config_SHA256",
        }
        if not required.issubset(existing.columns):
            raise RuntimeError(
                f"Existing result file does not match global protocol: {results_csv}"
            )
        valid_rows = []
        for _, record in existing.iterrows():
            try:
                label = str(record["Method"])
                seed = int(record["Seed"])
                if label not in labels or seed not in seeds:
                    continue
                if str(record["Global_Protocol_Version"]) != PROTOCOL_VERSION:
                    continue
                if str(record["Common_Metrics_Version"]) != COMMON_METRICS_VERSION:
                    continue
                if str(record["Global_Source_Scenario"]) != scenario_name:
                    continue
                if int(record["Global_Node_Count"]) != num_nodes:
                    continue
                if float(record["Global_Network_Width_m"]) != network_size[0]:
                    continue
                if float(record["Global_Network_Height_m"]) != network_size[1]:
                    continue
                if str(record["Global_Canonical_Config_SHA256"]) != canonical_digest:
                    continue
                expected_method_digest = _config_digest(built_configs[label])
                if str(record["Method_Config_SHA256"]) != expected_method_digest:
                    continue
            except (TypeError, ValueError, KeyError):
                continue
            resumed_row = record.to_dict()
            resumed_spec = next(spec for spec in registry if spec.label == label)
            resumed_row = apply_metric_applicability(resumed_row, resumed_spec)
            valid_rows.append(resumed_row)
            completed.add((label, seed))
        rows = valid_rows
        print(f"[RESUME] valid completed Method×Seed runs: {len(completed)}")

    print("=" * 92)
    print("GLOBAL REAL-WORLD COMMON-ENVIRONMENT EVALUATION")
    print("=" * 92)
    print("Source scenario :", scenario_name)
    print("Protocol        :", PROTOCOL_VERSION)
    print("Common metrics  :", COMMON_METRICS_VERSION)
    print("Nodes           :", num_nodes)
    print("Network         :", network_size)
    print("Seeds           :", list(seeds))
    print("Operational time:", sim_time, "s (from canonical code)")
    print("Methods         :", labels)
    print("Canonical SHA256:", canonical_digest)
    print("Physical/scenario numerical values are NOT duplicated in this runner.")
    print("=" * 92)

    for seed in seeds:
        trace_path = traces_dir / f"realworld_workload_seed_{seed:03d}.json"
        if not trace_path.exists():
            generate_shared_workload_trace(
                trace_path,
                seed=seed,
                num_nodes=num_nodes,
                sim_time=sim_time,
                canonical_config=canonical,
            )

        for spec in registry:
            pair = (spec.label, int(seed))
            if pair in completed:
                print(f"[RESUME] SKIP method={spec.label}; seed={seed}")
                continue

            # Fresh copy every run. Runtime trace paths are common-injected and
            # therefore are not part of the method-native difference audit.
            config = dict(built_configs[spec.label])
            config["workload_trace_record_path"] = None
            config["workload_trace_replay_path"] = str(trace_path)

            print("\n" + "-" * 92)
            print(f"[RUN] method={spec.label}; seed={seed}; nodes={num_nodes}; network={network_size}")
            print(f"[AUDIT] method-native changed keys: {audit_changes[spec.label]}")
            print("-" * 92)

            simulator = spec.manager_class(
                num_nodes=num_nodes,
                sim_time=sim_time,
                ablation_config=config,
                seed=int(seed),
                network_size=network_size,
            )

            # ONE method-independent metric observer for every method.
            # The deadline is read from the canonical scenario and used only
            # for evaluation; it does not alter any method's scheduling logic.
            common_metrics = attach_global_comparison_metrics(
                simulator,
                workload_trace_path=trace_path,
                method_name=spec.label,
                scenario_name=scenario_name,
                charging_deadline_s=canonical.get("hard_deadline_margin_s"),
            )

            wall_start = time.perf_counter()
            simulator.run_simulation()
            wall_runtime = time.perf_counter() - wall_start

            row = common_metrics.export_row(
                seed=int(seed),
                sim_time=sim_time,
                wall_clock_runtime_s=wall_runtime,
                extra_metadata={
                    "Global_Protocol_Version": PROTOCOL_VERSION,
                    "Global_Source_Scenario": scenario_name,
                    "Global_Canonical_Config_SHA256": canonical_digest,
                    "Global_Node_Count": int(num_nodes),
                    "Global_Network_Width_m": float(network_size[0]),
                    "Global_Network_Height_m": float(network_size[1]),
                    "Global_Operational_Time_s": int(sim_time),
                    "Global_Shared_Workload_Trace": str(trace_path),
                    "Global_Deployment_Policy": spec.deployment_policy,
                    "Method_Native_Changed_Keys": "|".join(audit_changes[spec.label]),
                    "Method_Config_SHA256": _config_digest(built_configs[spec.label]),
                },
            )
            row = apply_metric_applicability(row, spec)
            rows.append(row)
            completed.add(pair)

            checkpoint = pd.DataFrame(rows)
            _atomic_write_dataframe(checkpoint, results_csv)
            print(f"[CHECKPOINT] saved method={spec.label}; seed={seed}")

    df = pd.DataFrame(rows)

    expected = {(spec.label, int(seed)) for spec in registry for seed in seeds}
    observed = set(zip(df["Method"].astype(str), df["Seed"].astype(int)))
    missing = expected - observed
    if missing:
        raise RuntimeError(f"Incomplete global result set; missing {len(missing)} Method×Seed rows.")
    duplicates = df.duplicated(subset=["Method", "Seed"], keep=False)
    if duplicates.any():
        raise RuntimeError(
            "Duplicate Method×Seed rows detected:\n"
            + df.loc[duplicates, ["Method", "Seed"]].to_string(index=False)
        )

    _atomic_write_dataframe(df, results_csv)
    summary = summarize_global_comparison_results(df, summary_csv)
    primary = select_primary_comparison_table(summary)
    _atomic_write_dataframe(primary, primary_csv)

    # FCBE-2025 is a clustering-only paper, not an end-to-end WRSN method.
    # Run it as a separate controlled component track using the SAME global
    # runner and SAME common metric observer, but do not mix it into the
    # full-system primary comparison table.
    fcbe2025_component = run_fcbe2025_component_track(
        scenario_name=scenario_name,
        canonical=canonical,
        num_nodes=num_nodes,
        network_size=network_size,
        seeds=seeds,
        sim_time=sim_time,
        output_dir=output_dir,
        traces_dir=traces_dir,
    )

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "common_metrics_version": COMMON_METRICS_VERSION,
        "primary_comparison_metrics": list(PRIMARY_COMPARISON_METRICS),
        "canonical_mode": CANONICAL_MODE,
        "canonical_scenario": scenario_name,
        "canonical_config_sha256": canonical_digest,
        "physical_values_source": "sim_phase.py -> Article_Scaled_Absolute_Single",
        "num_nodes": num_nodes,
        "network_size": list(network_size),
        "seeds": list(seeds),
        "simulation_time_s": sim_time,
        "evaluation_contract_locked_in_global_real_world_py": True,
        "shared_workload_per_scenario_seed": True,
        "component_tracks": [fcbe2025_component],
        "simulator_rng_streams_are_seeded_by_master_seed": True,
        "methods": [
            {
                "label": spec.label,
                "deployment_policy": spec.deployment_policy,
                "comparison_domain": spec.comparison_domain,
                "not_applicable_metric_groups": list(
                    spec.not_applicable_metric_groups
                ),
                "allowed_method_native_keys": sorted(spec.native_keys),
                "observed_changed_keys": audit_changes[spec.label],
                "config_sha256": _config_digest(built_configs[spec.label]),
                "notes": spec.notes,
            }
            for spec in registry
        ],
        "results_csv": str(results_csv),
        "summary_csv": str(summary_csv),
        "primary_comparison_csv": str(primary_csv),
    }
    temp_manifest = Path(str(manifest_json) + ".tmp")
    with open(temp_manifest, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    os.replace(temp_manifest, manifest_json)

    print("\n" + "=" * 92)
    print("GLOBAL REAL-WORLD EVALUATION COMPLETE")
    print("Results :", results_csv)
    print("Summary :", summary_csv)
    print("Primary :", primary_csv)
    print("Manifest:", manifest_json)
    print("FCBE component:", fcbe2025_component["results_csv"])
    print("=" * 92)
    return df, summary


def main():
    return run_registered_methods()


if __name__ == "__main__":
    main()
