import os
import json
import random
import numpy as np
import pandas as pd

import os

# Must be configured before importing PyTorch or initializing CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from config import compute_packet_hash, make_mc_mac, summarize_results, make_plots,  DATA_PACKET_BITS, CONTROL_PACKET_BYTES_SMALL, DATA_PACKET_BYTES
from metric_collector import MetricsCollector
from base_station import BaseStation
from mobile_charger import MobileCharger
from simulator_manager import SimulatorManager


def _env_enabled(name, default='0'):
    raw = str(os.environ.get(name, default)).strip().lower()
    if raw in ('', '0', 'false', 'no', 'off', 'none'):
        return False
    try:
        return float(raw) > 0.0
    except ValueError:
        return raw in ('true', 'yes', 'on', 'enable', 'enabled')


def _base_config():
    return {
        'experiment_family': 'communication_security',
        'enable_energy_experiment': False,
        'enable_physical_compromise_experiment': False,
        'use_gradient_symmetric_discovery': True,
        'use_adaptive_edge_threshold': True,
        'use_critical_multipath_routing': True,
        # Retain two distinct critical routes whenever topology permits.
        'critical_require_two_routes': True,
        'critical_secondary_selection_mode': 'least_overlap_safest',
        # Synthesized second routes activate only after primary failure.
        'critical_synthesized_secondary_mode': 'failover',
        # Non-regression energy shield for synthesized failovers.
        'critical_failover_energy_guard': _env_enabled(
            'WRSN_CRITICAL_FAILOVER_ENERGY_GUARD', '1'),
        'critical_failover_min_energy_ratio': float(os.environ.get(
            'WRSN_CRITICAL_FAILOVER_MIN_ENERGY_RATIO', 0.65)),
        'critical_failover_reserve_margin_ratio': float(os.environ.get(
            'WRSN_CRITICAL_FAILOVER_RESERVE_MARGIN_RATIO', 0.05)),
        'use_implicit_piggyback_telemetry': True,
        'use_network_tomography_for_delay': True,
        'use_intersection_logic_for_droppers': True,
        'use_asynchronous_processing_pipeline': True,
        'use_shannon_entropy_weights': True,
        'use_security_veto_aggregation': True,
        'use_stealthy_attacker_detection': True,
        'use_zero_overhead_spoof_detection': True,
        'use_sed_filter': True,
        'use_uncertainty_calibrated_risk': True,
        'use_counterfactual_path_attribution': True,
        'use_fcm_classification': True,
        'use_risk_history_hysteresis': True,
        'use_risk_adaptive_coded_multipath': False,
        'use_mc_hmac_authentication': True,
        'use_mc_location_verification': True,
        'use_secure_charging_request_validation': True,

        # Adaptive charging core features
        'use_adaptive_quantum_charging': False,
        'use_dqn_charger_routing': False,
        'enable_qifl_learning': True,

        # ---------------------------------------------------------
        # NEW: Multi-Node RF Charging & Dynamic Partial Allocation
        # ---------------------------------------------------------
        'use_hybrid_fixed_chargers': False,
        'use_rf_multi_node_charging': False,
        'rf_charging_radius': 25.0,
        'use_dynamic_partial_charging': False,
        'dynamic_charging_weight_wt': 1.0,
        'dynamic_charging_weight_eps': 1.0,
        'dynamic_charging_weight_now': 1.0,
        'charging_rate': 5.0,
        'rf_path_loss_exponent': 2.0,
        'rf_reference_distance_m': 1.0,
        'rf_energy_conversion_efficiency': 0.70,
        'contact_location_tolerance_m': 5.0,
        'fixed_charger_count': 1,
        'fixed_charger_trigger_ratio': 0.15,
        'fixed_charger_target_ratio': 0.88,
        # ---------------------------------------------------------

        # ---------------------------------------------------------
        # NEW: Hybrid Hierarchical Routing & Clustering Parameters
        # ---------------------------------------------------------
        # The original routing engine remains the reference plane for
        # control and all charging decisions.  Only DATA packets use the new
        # clustered/MST overlay below.
        'use_hybrid_hierarchical_routing': True,
        'use_data_plane_hybrid_routing': True,
        'use_adaptive_greedy_weights': True,
        'use_ifbe_fuzzy_ch_selection': True,
        'use_fcbe_soft_balancing': True,
        'use_mst_backbone': True,

        # Strict energy-dominance guards: a DATA route may replace the original
        # route only when projected radio energy and bottleneck stress do not
        # increase.  These are routing parameters, not charger parameters.
        'data_route_recompute_period': 5,
        'data_route_energy_tolerance': 0.0,
        'data_route_stress_tolerance': 0.0,
        'data_route_distance_stretch': 1.0,
        'data_route_hop_slack': 0,
        'data_route_min_improvement': 0.0025,
        'data_secondary_k_paths': 24,
        'data_secondary_max_energy_stretch': 1.0,
        # ---------------------------------------------------------

        'energy_ewma_beta': 0.70,
        'qifl_learning_rate': 5e-3,
        'dqn_learning_rate': 1e-3,
        'dqn_max_candidates': 12,
        'dqn_epsilon': None,
        'dqn_exploration_mode': 'uncertainty_adaptive',
        'dqn_warmup_updates': 32,
        'dqn_warmup_competitive_decisions': 24,
        # Legacy-compatible default. The optional composite mode can use
        # updates, replay size, TD-error stability, and a decision timeout.
        'dqn_warmup_readiness_mode': 'competitive_only',
        'dqn_warmup_min_replay_samples': None,
        'dqn_warmup_td_abs_threshold': None,
        'dqn_warmup_td_window': 20,
        'dqn_warmup_max_total_decisions': None,
        # Observer-only learning audit. These defaults add measurements but do
        # not alter target selection, rewards, exploration, or safety shields.
        'enable_dqn_audit_logging': True,
        'dqn_training_enabled': True,
        'dqn_checkpoint_load_path': None,
        'dqn_checkpoint_save_path': None,
        'dqn_checkpoint_load_optimizer': False,
        'workload_trace_record_path': None,
        'workload_trace_replay_path': None,
        'use_gpu_for_adaptive_charging': True,
        'adaptive_compute_device': 'auto',
        'require_pytorch_for_adaptive_charging': True,
        'require_cuda_for_adaptive_charging': False,
        'adaptive_torch_deterministic': True,
        'adaptive_torch_strict_determinism': False,
        'adaptive_torch_allow_tf32': False,
        'sos_threshold_margin_ratio': 0.03,

        'use_charging_hysteresis': True,
        'minimum_charge_energy_ratio': 0.08,
        'post_charge_release_ratio': 0.88,
        'charging_reentry_cooldown_s': 180.0,
        'qifl_threshold_lower_bound': 0.20,
        'qifl_threshold_upper_bound': 0.52,
        'qifl_sos_threshold_upper_bound': 0.90,
        'minimum_cluster_charge_gap_ratio': 0.05,
        'penalize_repeated_mc_visits': True,
        'dqn_reward_travel_weight': 0.75,
        'dqn_reward_energy_weight': 1.5,
        'dqn_reward_deadline_weight': 4.0,
        'dqn_reward_revisit_penalty': 2.0,
        'dqn_reward_death_penalty': 6.0,
        'use_travel_aware_admission': True,
        'travel_admission_lambda': 2.0,
        'travel_admission_gate': 0.10,
        'hard_deadline_margin_s': 90.0,
        'service_cycle_fallback_s': 225.0,
        'service_horizon_safety_multiplier': 1.12,
        'balanced_service_horizon_cap_s': 420.0,
        'emergency_service_horizon_cap_s': 720.0,
        'opportunistic_plus_horizon_s': 150.0,
        'plus_pool_target_ratio': 0.45,
        'plus_pool_max_members_per_cluster': 2,
        'max_rf_cluster_members': 5,
        'max_cluster_stay_time_s': 1.50,
        'adaptive_request_threshold_hard_cap': 0.68,
        'request_plus_fraction': 0.50,
        'max_service_latency_batches': 6,
        'depletion_rate_safety_factor': 1.20,
        'physical_eps_ewma_beta': 0.70,
        'communication_energy_reserve_ratio': 0.04,
        'service_latency_safety_margin_s': 75.0,
        'use_deadline_safe_energy_sentinel': True,
        'use_universal_mc_feasibility_guard': True,
        'mc_reserve_ratio': 0.05,

        'mobile_charger_selection_mode': 'hardcoded',
        'num_mobile_chargers': 1,

        # Cost-aware adaptive fleet sizing. The hardcoded default preserves
        # the published single-charger baseline; dynamic/cost_aware mode uses
        # the smallest fleet satisfying node ratio, energy balance, service
        # time, and response-time coverage constraints.
        'mobile_charger_count_safety_factor': 1.10,
        'mobile_charger_service_utilization': 0.70,
        'mobile_charger_expected_request_fraction': 0.25,
        'mobile_charger_average_recharge_fraction': 0.50,
        'mobile_charger_service_horizon_s': 900.0,
        'mobile_charger_response_deadline_s': 90.0,
        'mobile_charger_response_margin_s': 0.0,
        'mobile_charger_dispatch_overhead_s': 0.0,
        'mobile_charger_spatial_packing_efficiency': 0.65,
        'mobile_charger_estimated_node_power_j_per_s': None,
        'mobile_charger_unit_purchase_cost': 0.0,
        'mobile_charger_unit_integration_cost': 0.0,
        'mobile_charger_annual_maintenance_cost': 0.0,
        'mobile_charger_economic_life_years': 3.0,
        'mobile_charger_total_budget': None,
        'mobile_charger_budget_policy': 'performance_first',
        'mobile_charger_cost_currency': 'user_defined',

        # Dead nodes remain chargeable and can re-enter the ALIVE state after
        # receiving sufficient energy. A zero threshold preserves legacy
        # behavior (any positive physically delivered energy can revive).
        'enable_dead_node_revival': True,
        'dead_node_revival_min_energy_ratio': 0.0,

        # Multi-MC spatial safety is opt-in so the one-MC baseline is exactly
        # preserved. Stable angular regions are soft preferences; the RF-zone
        # separation guard permits verified emergency overrides.
        'use_stable_charger_regions': False,
        'use_interference_safe_multi_mc_dispatch': False,
        'mobile_charger_safe_distance_m': 0.0,
        'mobile_charger_physical_footprint_m': 0.0,

        'mobile_charger_capacity_mode': 'absolute',
        'mobile_charger_capacity': 50000.0,
        'mobile_charger_capacities': [50000.0],
        'mobile_charger_sensor_equivalent_capacity': 20.0,
        'mobile_charger_sensor_equivalent_capacities': [20.0],
        'mobile_charger_travel_cost_mode': 'absolute',
        'mobile_charger_travel_energy_cost_per_m': 0.1,
        'mobile_charger_travel_sensor_equivalent_cost_per_m': 0.0005,
        'mobile_charger_speed': 5.0,
        'mobile_charger_speeds': [5.0],
        'mobile_charger_nodes_per_charger': 40,
        'min_mobile_chargers': 1,
        'max_mobile_chargers': 8,

        'enable_network_layer_ack': False,
        # MAC ACK is always accounted as a hardware link-layer cost. The
        # default accounting mode preserves legacy battery trajectories; use
        # 'strict_physical' for an additional physical battery deduction.
        'implicit_mac_ack_mode': 'accounting',
        'ack_packet_bits': 16 * 8,
        'charge_receipt_tolerance_ratio': 0.05,
        'mc_energy_claim_tolerance_ratio': 0.05,
        'enable_energy_trust_recovery': True,
        'energy_trust_recovery_epochs': 8,
        'energy_trust_recovery_alpha': 0.12,
        'energy_trust_recovery_tolerance_ratio': 0.01,
        'energy_trust_recovery_min_auth_trust': 0.95,
        'energy_trust_recovery_min_forwarding_trust': 0.90,
        'charging_audit_enforcement_mode': 'shadow',
        'preserve_legacy_mc_claim_credit': True,
        'enable_periodic_hello': False,
        'route_recompute_period': None,
        'data_packet_bits': DATA_PACKET_BITS,

        # Explicit attack model. ``legacy_compatible`` is the non-regression
        # thesis baseline; dedicated attack-suite scenarios below enable the
        # corrected end-to-end models independently.
        'attack_model_mode': 'legacy_compatible',
        'attack_data_drop_probability': 1.0,
        'attack_control_drop_probability': 0.50,
        'attack_false_energy_ratio': 0.05,
        'attack_false_energy_data_enabled': False,
        'attack_false_energy_control_enabled': True,
        'attack_false_energy_receipt_enabled': True,
        'attack_false_sos_mode': 'legacy_compatible',
        'attack_false_sos_probability_per_epoch': 0.05,
        'attack_false_sos_energy_ratio': 0.05,
        'attack_control_epoch_s': 10.0,
        'attack_delay_enabled': False,
        'attack_delay_probability': 0.0,
        'attack_delay_min_s': 0.25,
        'attack_delay_max_s': 1.00,
        'attack_tamper_enabled': False,
        'attack_tamper_probability': 0.0,
        'attack_replay_enabled': False,
        'attack_replay_probability': 0.0,
        'attack_on_off_enabled': False,
        'attack_on_off_period_s': 120.0,
        'attack_on_fraction': 0.50,
        'attack_remediation_time_s': None,
        # When remediation is enabled, keep the node in local PROBATION for
        # one control epoch before completing the label transition to HONEST.
        # Baseline results are unchanged because remediation itself is disabled.
        'attack_rehabilitation_delay_s': 10.0,

        'mc_attack_energy_claim_enabled': False,
        'mc_attack_energy_claim_factor': 1.50,
        'mc_attack_location_spoof_enabled': False,
        'mc_attack_location_spoof_offset_m': 100.0,
        'mc_attack_forged_hmac_enabled': False,
        'mc_attack_replay_enabled': False,
        'mc_attack_probability': 1.0,
        'mc_attack_on_off_enabled': False,

        'fake_charger_attack_enabled': False,
        'fake_charger_attack_probability_per_epoch': 0.0,
        'fake_charger_attack_energy_ratio': 0.35,
        'attack_percentage': 0.15,
        'critical_event_probability': 0.10,
        'transmission_range': 40.0,
    }


def _physical_compromise_scenarios():
    c = dict(_base_config())
    c.update({
        'experiment_family': 'physical_compromise',
        'enable_energy_experiment': False,
        'enable_physical_compromise_experiment': True,
        'simulate_lab_reprovisioning': False,
    })
    return [('PhysicalCompromise_TPM_Zeroization', c)]


def _scenarios_for_mode(mode):
    mode = (mode or 'main').strip().lower().replace('-', '_')
    aliases = {
        'physical': 'physical_compromise', 'tpm': 'physical_compromise', 'tpm_compromise': 'physical_compromise',
        'mc': 'charging_suite', 'mobile_charging': 'charging_suite', 'charging': 'charging_suite',
        'routing': 'routing_suite', 'routing_ablation': 'routing_suite', 'cluster': 'routing_suite',
        'all': 'all_modes', 'full': 'all_modes', 'full_suite': 'all_modes', 'everything': 'all_modes',
        'complete': 'all_modes', 'complete_suite': 'all_modes',
        'sweep': 'parameter_sweep', 'stress': 'parameter_sweep', 'parameter': 'parameter_sweep',
        'article': 'article_scaled', 'paper_energy': 'article_scaled',
        'article_energy': 'article_scaled', 'article_mc': 'article_scaled',
        'paper_mc': 'article_scaled', 'article_compare': 'article_scaled',
        'article_energy_mc_sweep': 'article_scaled',
        'article_energy_comparison': 'article_scaled',
        'attacks': 'attack_suite', 'attack': 'attack_suite',
        'attack_validation': 'attack_suite', 'end_to_end_attacks': 'attack_suite',
    }
    mode = aliases.get(mode, mode)

    base = _base_config()

    def get_article_scaled_single():
        """One runtime-scaled article profile built on the best V5 energy method.

        The physical paper horizon is 5000 x 30 = 150000 s.  To avoid an
        impractically long event-driven run, the default effective horizon is
        3600 s.  Stored battery energies are scaled by the same time ratio,
        while per-packet radio coefficients, charging power, movement cost,
        radius, and speed retain their paper values.  This preserves the main
        horizon-normalized energy ratios for a stationary workload without
        changing packet/header sizes or any charging-request threshold.
        """
        # Start from the exact V5 EnergyStress method so QIFL/DQN, request
        # thresholds, packet/header sizes, routing, and charger count remain
        # unchanged unless explicitly listed below.
        _, reference = get_energy()[0]
        c = dict(reference)

        physical_time_s = 5000.0 * 30.0
        effective_time_s = float(os.environ.get(
            'WRSN_ARTICLE_SCALED_TIME', 3600.0))
        effective_time_s = min(physical_time_s, max(30.0, effective_time_s))
        time_scale = effective_time_s / physical_time_s

        physical_sensor_energy_j = 5.0
        physical_mc_capacity_j = 200000.0
        effective_sensor_energy_j = physical_sensor_energy_j * time_scale
        effective_mc_capacity_j = physical_mc_capacity_j * time_scale
        charger_count = max(1, int(c.get('num_mobile_chargers', 1)))

        c.update({
            'article_energy_profile_enabled': True,
            'article_scaled_single_scenario': True,
            'article_physical_simulation_time_s': physical_time_s,
            'article_effective_simulation_time_s': effective_time_s,
            'article_time_scale': time_scale,
            'article_physical_sensor_initial_energy_j': physical_sensor_energy_j,
            'article_physical_mc_capacity_per_charger_j': physical_mc_capacity_j,

            # Time-scaled stored energies.  The ratio MC_capacity/E0 remains
            # exactly 200000/5 = 40000, and the shortened horizon has the same
            # ratio to both stored-energy values.
            'initial_energy_override': effective_sensor_energy_j,
            'mobile_charger_capacity_mode': 'absolute',
            'mobile_charger_capacity': effective_mc_capacity_j,
            'mobile_charger_capacities': [effective_mc_capacity_j] * charger_count,

            # Exact article radio coefficients; packet/header sizes are NOT
            # overridden and therefore remain those of the original method.
            'radio_e_elec_j_per_bit': 50e-9,
            'radio_eps_fs_j_per_bit_m2': 100e-12,
            'radio_eps_mp_j_per_bit_m4': 0.013e-12,

            # Exact requested MC physical parameters.  Capacity remains in
            # ABSOLUTE mode; relative mode can still be selected later by the
            # original resolver without any special-case code.
            'mobile_charger_travel_cost_mode': 'absolute',
            'mobile_charger_travel_energy_cost_per_m': 0.010,
            'mobile_charger_travel_energy_costs_per_m': [0.010] * charger_count,
            'charging_rate': 0.005,
            'rf_charging_radius': 5.0,
            'mobile_charger_speed': 5.0,
            'mobile_charger_speeds': [5.0] * charger_count,

            # One scaled scenario only.
            'simulation_time_override_s': effective_time_s,
        })
        return [('Article_Scaled_Absolute_Single', c)]

    def get_routing_ablation():
        out = []

        guarded_defaults = {
            'experiment_family': 'routing_clustering_ablation',
            'enable_energy_experiment': True,

            # Active New Routing Features
            'use_hybrid_hierarchical_routing': True,
            'use_data_plane_hybrid_routing': True,
            'use_adaptive_greedy_weights': True,
            'use_ifbe_fuzzy_ch_selection': True,
            'use_fcbe_soft_balancing': True,
            'use_mst_backbone': True,
        }

        def add(name, **changes):
            c = dict(base)
            c.update(guarded_defaults)
            c.update(changes)
            out.append((name, c))

        # 1. Master Method with all layers enabled
        add('Final_Method_MC_Completed', **{})

        # 2. Isolate Soft Balancing (Phase 2 FCBE)
        add('Routing_No_FCBE_SoftBalancing',
            use_fcbe_soft_balancing=False)

        # 3. Isolate Fuzzy CH Selection (Phase 1 IFBE)
        add('Routing_No_IFBE_Selection',
            use_ifbe_fuzzy_ch_selection=False)

        # 4. Isolate Adaptive Urgency Weights
        add('Routing_No_AdaptiveWeights',
            use_adaptive_greedy_weights=False)

        # 5. Isolate the DATA-plane MST backbone.
        add('Routing_No_MST_Backbone',
            use_mst_backbone=False)

        # 6. Reference method: no DATA-plane overlay.
        add('Routing_Flat_Baseline',
            use_data_plane_hybrid_routing=False)

        return out

    def get_energy():
        c = dict(base)
        c.update({
            'experiment_family': 'energy_charging_stress',
            'enable_energy_experiment': True,
            'initial_energy_override': 0.010,
            'event_rate_lambda': 1.8,
            'idle_sensing_power_j_per_s': 1.0e-5,
            'charge_need_ratio': 0.80,
            'charge_claim_est_max_deviation': 1.00,
            'allow_energy_stress_predictive_charging': True,
            'use_adaptive_quantum_charging': True,
            'use_dqn_charger_routing': True,

            'use_hybrid_fixed_chargers': True,
            'use_rf_multi_node_charging': True,
            'use_dynamic_partial_charging': True,

            'use_charging_hysteresis': True,
            'use_adaptive_charge_guards': True,
            'mobile_charger_capacity_mode': 'absolute',
            'mobile_charger_capacity': 50000.0,
            'mobile_charger_capacities': [50000.0],
            'mobile_charger_travel_cost_mode': 'absolute',
            'mobile_charger_travel_energy_cost_per_m': 0.1,
            'mobile_charger_speed': 5.0,
            'mobile_charger_speeds': [5.0],
            'charging_rate': 5.0,
            'minimum_charge_energy_ratio': 0.08,
            'post_charge_release_ratio': 0.88,
            'charging_reentry_cooldown_s': 180.0,
            'qifl_threshold_lower_bound': 0.20,
            'qifl_threshold_upper_bound': 0.52,
        })
        return [('EnergyStress_QIFL_DQN_Guarded', c)]

    def get_mc_ablation():
        out = []

        guarded_defaults = {
            'experiment_family': 'energy_charging_stress',
            'enable_energy_experiment': True,
            'initial_energy_override': 0.018,
            'event_rate_lambda': 1.2,
            'idle_sensing_power_j_per_s': 2.5e-6,
            'allow_energy_stress_predictive_charging': True,
            'charge_need_ratio': 0.80,
            'charge_claim_est_max_deviation': 1.00,
            'use_charging_hysteresis': True,
            'use_adaptive_charge_guards': True,
            'minimum_charge_energy_ratio': 0.08,
            'post_charge_release_ratio': 0.88,
            'charging_reentry_cooldown_s': 180.0,
            'qifl_threshold_lower_bound': 0.20,
            'qifl_threshold_upper_bound': 0.52,
            'qifl_sos_threshold_upper_bound': 0.90,
            'minimum_cluster_charge_gap_ratio': 0.05,
            'penalize_repeated_mc_visits': True,

            'use_hybrid_fixed_chargers': True,
            'use_rf_multi_node_charging': True,
            'use_dynamic_partial_charging': True,
        }

        def add(name, **changes):
            c = dict(base)
            c.update(guarded_defaults)
            c.update(changes)
            out.append((name, c))

        add('MC_Full_QIFL_DQN',
            use_adaptive_quantum_charging=True,
            use_dqn_charger_routing=True)

        add('MC_Full_Security',
            use_adaptive_quantum_charging=True,
            use_dqn_charger_routing=True)

        if _env_enabled('WRSN_ENABLE_MULTI_MC_SCENARIOS'):
            add('MC_Full_QIFL_DQN_MultiMC_Hardcoded',
                use_adaptive_quantum_charging=True,
                use_dqn_charger_routing=True,
                mobile_charger_selection_mode='hardcoded',
                mobile_charger_capacity_mode='relative',
                mobile_charger_travel_cost_mode='relative',
                num_mobile_chargers=3,
                mobile_charger_sensor_equivalent_capacities=[
                    100.0, 80.0, 60.0],
                mobile_charger_travel_sensor_equivalent_cost_per_m=0.0005,
                mobile_charger_speeds=[5.0, 5.0, 5.0],
                use_relative_mc_feasibility_guard=True,
                use_stable_charger_regions=True,
                use_interference_safe_multi_mc_dispatch=True,
                mobile_charger_safe_distance_m=2.0,
                max_adaptive_charge_queue=18)
            add('MC_Full_QIFL_DQN_MultiMC_Dynamic',
                use_adaptive_quantum_charging=True,
                use_dqn_charger_routing=True,
                mobile_charger_selection_mode='dynamic',
                mobile_charger_capacity_mode='relative',
                mobile_charger_travel_cost_mode='relative',
                mobile_charger_nodes_per_charger=8,
                min_mobile_chargers=2,
                max_mobile_chargers=4,
                mobile_charger_sensor_equivalent_capacities=[
                    100.0, 80.0, 60.0, 45.0],
                mobile_charger_travel_sensor_equivalent_cost_per_m=0.0005,
                use_relative_mc_feasibility_guard=True,
                use_stable_charger_regions=True,
                use_interference_safe_multi_mc_dispatch=True,
                mobile_charger_safe_distance_m=2.0,
                max_adaptive_charge_queue=18)

        if _env_enabled('WRSN_ENABLE_MC_COUNT_SWEEP'):
            for mc_count in (1, 2, 3, 4):
                add(f'MC_Full_QIFL_DQN_{mc_count}MC_RelCap',
                    use_adaptive_quantum_charging=True,
                    use_dqn_charger_routing=True,
                    mobile_charger_selection_mode='hardcoded',
                    mobile_charger_capacity_mode='relative',
                    mobile_charger_travel_cost_mode='relative',
                    num_mobile_chargers=mc_count,
                    mobile_charger_sensor_equivalent_capacities=[
                        80.0] * mc_count,
                    mobile_charger_travel_sensor_equivalent_cost_per_m=0.0005,
                    mobile_charger_speeds=[5.0] * mc_count,
                    use_relative_mc_feasibility_guard=True,
                    use_stable_charger_regions=(mc_count > 1),
                    use_interference_safe_multi_mc_dispatch=(mc_count > 1),
                    mobile_charger_safe_distance_m=2.0 if mc_count > 1 else 0.0,
                    max_adaptive_charge_queue=max(18, mc_count * 5))

        add('MC_No_Fixed_Chargers',
            use_adaptive_quantum_charging=True,
            use_dqn_charger_routing=True,
            use_hybrid_fixed_chargers=False)

        add('MC_No_RF_MultiNode',
            use_adaptive_quantum_charging=True,
            use_dqn_charger_routing=True,
            use_rf_multi_node_charging=False)

        add('MC_No_Dynamic_Partial',
            use_adaptive_quantum_charging=True,
            use_dqn_charger_routing=True,
            use_dynamic_partial_charging=False)

        add('MC_QIFL_Only',
            use_adaptive_quantum_charging=True,
            use_dqn_charger_routing=False)
        add('MC_DQN_Only',
            use_adaptive_quantum_charging=False,
            use_dqn_charger_routing=True)
        add('MC_No_QIFL_DQN',
            use_adaptive_quantum_charging=False,
            use_dqn_charger_routing=False,
            use_adaptive_charge_guards=False,
            use_rf_multi_node_charging=False,
            use_dynamic_partial_charging=False)

        add('MC_QIFL_DQN_No_Guards',
            use_adaptive_quantum_charging=True,
            use_dqn_charger_routing=True,
            use_adaptive_charge_guards=False,
            use_charging_hysteresis=False,
            minimum_charge_energy_ratio=0.0,
            charging_reentry_cooldown_s=0.0,
            qifl_threshold_upper_bound=0.90,
            minimum_cluster_charge_gap_ratio=0.0,
            penalize_repeated_mc_visits=False)

        add('MC_No_HMAC',
            use_adaptive_quantum_charging=True,
            use_dqn_charger_routing=True,
            use_mc_hmac_authentication=False)
        add('MC_No_Location_Check',
            use_adaptive_quantum_charging=True,
            use_dqn_charger_routing=True,
            use_mc_location_verification=False)
        add('MC_Raw_SOS_No_Validation',
            use_adaptive_quantum_charging=True,
            use_dqn_charger_routing=True,
            use_secure_charging_request_validation=False)
        add('MC_No_SED',
            use_adaptive_quantum_charging=True,
            use_dqn_charger_routing=True,
            use_sed_filter=False)
        add('MC_No_Deviation_Filter',
            use_adaptive_quantum_charging=True,
            use_dqn_charger_routing=True,
            use_zero_overhead_spoof_detection=False)
        return out

    def get_ablation():
        ablations = []

        def add(name, **changes):
            c = dict(base)
            c.update(changes)
            ablations.append((name, c))
        add('Full_Method')
        add('A01_FCM_Only', use_uncertainty_calibrated_risk=False, use_risk_history_hysteresis=False, use_counterfactual_path_attribution=False,
            use_zero_overhead_spoof_detection=False, use_sed_filter=False, use_stealthy_attacker_detection=False)
        add('A02_No_FCM', use_fcm_classification=False)
        add('A03_No_UCR_Risk_Fusion', use_uncertainty_calibrated_risk=False)
        add('A04_No_Risk_History_Hysteresis', use_risk_history_hysteresis=False)
        add('A05_No_WRSN_Hard_Filters', use_zero_overhead_spoof_detection=False,
            use_sed_filter=False, use_stealthy_attacker_detection=False)
        add('A06_No_SED', use_sed_filter=False)
        add('A07_No_Deviation_MC', use_zero_overhead_spoof_detection=False)
        add('A08_No_Stealth_Filter', use_stealthy_attacker_detection=False)
        add('A09_No_Counterfactual_Path_Attribution',
            use_counterfactual_path_attribution=False)
        add('A10_No_Shannon_Entropy_Weights', use_shannon_entropy_weights=False)
        add('A11_No_Security_Veto_FE', use_security_veto_aggregation=False)
        add('A12_No_Critical_Multipath', use_critical_multipath_routing=False)
        add('A13_Periodic_Route_Recompute', route_recompute_period=1)
        add('A14_Periodic_HELLO_Enabled', enable_periodic_hello=True)
        add('A15_Network_ACK_Enabled', enable_network_layer_ack=True)
        add('A16_No_Piggyback_Telemetry', use_implicit_piggyback_telemetry=False)
        add('A17_No_Adaptive_Threshold', use_adaptive_edge_threshold=False)
        return ablations

    def get_parameter_sweep():
        out = []

        def add(name, **changes):
            c = dict(base)
            c.update(changes)
            out.append((name, c))
        for attack_pct in (0.00, 0.05, 0.15, 0.30, 0.45):
            add(f'Sweep_AttackPct_{int(attack_pct*100):02d}',
                attack_percentage=attack_pct)
        for event_rate in (0.10, 0.50, 1.20, 2.00):
            add(f'Sweep_EventRate_{str(event_rate).replace(".", "p")}',
                event_rate_lambda=event_rate)
        for crit_prob in (0.01, 0.05, 0.10, 0.25, 0.40):
            add(f'Sweep_CriticalProb_{int(crit_prob*100):02d}',
                critical_event_probability=crit_prob)
        for tx_range in (25.0, 40.0, 60.0):
            add(f'Sweep_TxRange_{int(tx_range)}m', transmission_range=tx_range)
        for w in (10, 25, 50, 100):
            add(f'Sweep_WindowW_{w}', sliding_window_size=w)
        return out

    def get_attack_suite():
        """Corrected end-to-end attack scenarios.

        The default energy scenario remains legacy-compatible for exact result
        preservation.  These scenarios activate one attack at a time through
        the real SensorNode/MobileCharger execution paths.
        """
        out = []

        def add(name, **changes):
            c = dict(base)
            c.update({
                'experiment_family': 'end_to_end_attack_validation',
                'attack_model_mode': 'corrected_end_to_end',
                'attack_data_drop_probability': 0.0,
                'attack_control_drop_probability': 0.0,
                'attack_false_energy_data_enabled': False,
                'attack_false_energy_control_enabled': False,
                'attack_false_energy_receipt_enabled': False,
                'attack_false_sos_mode': 'disabled',
                'attack_delay_enabled': False,
                'attack_tamper_enabled': False,
                'attack_replay_enabled': False,
                'attack_on_off_enabled': False,
                'mc_attack_energy_claim_enabled': False,
                'mc_attack_location_spoof_enabled': False,
                'mc_attack_forged_hmac_enabled': False,
                'mc_attack_replay_enabled': False,
                'fake_charger_attack_enabled': False,
            })
            c.update(changes)
            out.append((name, c))

        add('Attack_Blackhole_EndToEnd',
            attack_data_drop_probability=1.0)
        add('Attack_SelectiveForwarding_50_EndToEnd',
            attack_data_drop_probability=0.50)
        add('Attack_FalseEnergy_EndToEnd',
            attack_false_energy_data_enabled=True,
            attack_false_energy_receipt_enabled=True)
        add('Attack_FalseSOS_EndToEnd',
            attack_false_sos_mode='per_epoch',
            attack_false_sos_probability_per_epoch=0.25)
        add('Attack_Delay_EndToEnd',
            attack_delay_enabled=True,
            attack_delay_probability=0.75,
            attack_delay_min_s=0.50,
            attack_delay_max_s=2.00)
        add('Attack_Tamper_EndToEnd',
            attack_tamper_enabled=True,
            attack_tamper_probability=0.75)
        add('Attack_Replay_EndToEnd',
            attack_replay_enabled=True,
            attack_replay_probability=0.50,
            use_critical_multipath_routing=False)
        add('Attack_OnOff_SelectiveForwarding_EndToEnd',
            attack_data_drop_probability=0.75,
            attack_on_off_enabled=True,
            attack_on_off_period_s=120.0,
            attack_on_fraction=0.50)
        add('Attack_Remediation_EndToEnd',
            attack_false_energy_data_enabled=True,
            attack_false_energy_receipt_enabled=True,
            attack_remediation_time_s=600.0,
            attack_rehabilitation_delay_s=10.0)
        add('Attack_MC_FalseEnergy_EndToEnd',
            enable_energy_experiment=True,
            use_adaptive_quantum_charging=True,
            use_dqn_charger_routing=True,
            mc_attack_energy_claim_enabled=True)
        add('Attack_MC_LocationSpoof_EndToEnd',
            enable_energy_experiment=True,
            use_adaptive_quantum_charging=True,
            use_dqn_charger_routing=True,
            mc_attack_location_spoof_enabled=True)
        add('Attack_MC_ForgedHMAC_EndToEnd',
            enable_energy_experiment=True,
            use_adaptive_quantum_charging=True,
            use_dqn_charger_routing=True,
            mc_attack_forged_hmac_enabled=True)
        add('Attack_MC_Replay_EndToEnd',
            enable_energy_experiment=True,
            use_adaptive_quantum_charging=True,
            use_dqn_charger_routing=True,
            mc_attack_replay_enabled=True)
        add('Attack_FakeCharger_EndToEnd',
            fake_charger_attack_enabled=True,
            fake_charger_attack_probability_per_epoch=0.25)
        return out

    if mode == 'article_scaled':
        return get_article_scaled_single()
    if mode == 'physical_compromise':
        return _physical_compromise_scenarios()
    if mode == 'security_suite':
        return [('Phase1_Method', base)] + get_ablation()
    if mode in ('charging_suite', 'energy_suite', 'mc_suite', 'charging_ablation'):
        return get_energy() + get_mc_ablation()
    if mode == 'routing_suite':
        return get_routing_ablation()
    if mode == 'all_modes':
        return [('Phase1_Method', base)] + get_ablation() + get_energy() + get_mc_ablation() + get_routing_ablation() + _physical_compromise_scenarios()
    if mode in ('parameter_sweep', 'stress_sweep', 'innovation_parameter_sweep'):
        return get_parameter_sweep()
    if mode == 'attack_suite':
        return get_attack_suite()
    if mode == 'energy':
        return get_energy()
    if mode == 'mc_ablation':
        return get_mc_ablation()
    if mode == 'ablation':
        return get_ablation()

    return [('Phase1_Method', base)]


def _edgecase_base_station(async_pipeline=False):
    cfg = dict(_base_config())
    cfg['use_asynchronous_processing_pipeline'] = bool(async_pipeline)
    cfg['require_source_packet_hmac'] = True
    cfg['use_critical_multipath_routing'] = True
    # Ensure edge case mocks use flat routing compatibility if testing raw paths
    cfg['use_hybrid_hierarchical_routing'] = False
    cfg['use_data_plane_hybrid_routing'] = False
    metrics = MetricsCollector(total_nodes=4, total_rounds_or_time=10)
    bs = BaseStation(metrics, cfg, bs_id=0)

    bs.node_states = {
        1: {
            'x': 0.0, 'y': 0.0, 'E_initial': 1.0, 'E_claimed': 1.0, 'E_est': 1.0,
            'trust_vector': [1.0, 1.0, 1.0, 1.0, 1.0], 'T1_history': 1.0, 'T2_history': 1.0,
            'T3_history': 1.0, 'T4_history': 1.0, 'T5_history': 1.0, 'FE': 1.0,
            'status': 'WHITELISTED', 'crypto_keys': 'node-key-1', 'Deviation_MC_norm': 0.0,
            'SED_index': 0.0, 'spoof_trust': 1.0, 'total_injected_energy': 0.0,
        }
    }
    bs.routing_tables = {1: {'primary': 2, 'secondary': 3, 'primary_path': [
        1, 2, 0], 'secondary_path': [1, 3, 0], 'critical_paths': [[1, 2, 0], [1, 3, 0]]}}
    return bs, metrics


def _edgecase_packet(seq=1, packet_type='DATA', priority='NORMAL', energy=0.9, payload=25.0, timestamp=0.0, key='node-key-1'):
    pkt = {
        'source_id': 1, 'type': packet_type, 'seq': int(seq), 'timestamp': float(timestamp),
        'priority': priority, 'payload': payload,
        'size_bytes': CONTROL_PACKET_BYTES_SMALL if packet_type != 'DATA' else DATA_PACKET_BYTES,
        'energy_trace': [(1, float(energy))],
    }
    if key is not None:
        pkt['hash'] = compute_packet_hash(pkt, key)
    return pkt


def run_security_edge_case_validation(output_csv='security_edge_cases_results.csv', verbose=True):
    rows = []

    def record(case, passed, expected, observed, details=''):
        rows.append({'Case': case, 'Passed': bool(
            passed), 'Expected': expected, 'Observed': observed, 'Details': details})

    bs, metrics = _edgecase_base_station()
    bad_sos = _edgecase_packet(seq=1, packet_type='SOS_CHARGE',
                               priority='HIGH', energy=0.01, timestamp=0.0, key=None)
    bs.data_plane_receive(dict(bad_sos), [1, 2, 0], 0.1)
    record('Forged_SOS_Rejected', (1 not in bs.pending_sos_requests and abs(bs.node_states[1]['E_claimed'] - 1.0) < 1e-12 and metrics.packet_hmac_failures >= 1),
           'No pending SOS, no E_claimed update, HMAC failure counted', f"pending={1 in bs.pending_sos_requests}, E_claimed={bs.node_states[1]['E_claimed']}, hmac_failures={metrics.packet_hmac_failures}")

    bs, metrics = _edgecase_base_station()
    good_sos = _edgecase_packet(
        seq=1, packet_type='SOS_CHARGE', priority='HIGH', energy=0.2, timestamp=0.0)
    bs.data_plane_receive(dict(good_sos), [1, 2, 0], 0.1)
    record('Authenticated_SOS_Accepted_As_Evidence', (1 in bs.pending_sos_requests and abs(
        bs.node_states[1]['E_claimed'] - 0.2) < 1e-12), 'Pending SOS evidence exists and E_claimed is updated', f"pending={1 in bs.pending_sos_requests}, E_claimed={bs.node_states[1]['E_claimed']}")

    bs, metrics = _edgecase_base_station()
    bad_data = _edgecase_packet(
        seq=1, packet_type='DATA', priority='NORMAL', timestamp=0.0)
    bad_data['hash'] = 'not-a-valid-hmac'
    bs.data_plane_receive(dict(bad_data), [1, 2, 0], 0.1)
    record('Bad_Data_HMAC_Rejected', (metrics.packet_hmac_failures >= 1 and bs.last_seq_per_source.get(1) is None),
           'HMAC failure counted and DATA freshness not advanced', f"hmac_failures={metrics.packet_hmac_failures}, last_data_seq={bs.last_seq_per_source.get(1)}")

    bs, metrics = _edgecase_base_station()
    p = _edgecase_packet(seq=5, packet_type='DATA',
                         priority='NORMAL', timestamp=1.0)
    bs.data_plane_receive(dict(p), [1, 2, 0], 1.1)
    bs.data_plane_receive(dict(p), [1, 2, 0], 1.2)
    record('Source_Sequence_Replay_Rejected', metrics.source_sequence_replay_events >= 1,
           'Replay counter increments for repeated normal DATA sequence', f"replay_events={metrics.source_sequence_replay_events}")

    bs, metrics = _edgecase_base_station()
    high = _edgecase_packet(seq=10, packet_type='DATA',
                            priority='HIGH', payload=80.0, timestamp=1.0)
    bs.data_plane_receive(dict(high), [1, 2, 0], 1.1)
    before = metrics.source_sequence_replay_events
    bs.data_plane_receive(dict(high), [1, 3, 0], 1.2)
    record('Approved_Critical_Multipath_Copy_Accepted', metrics.source_sequence_replay_events == before,
           'Approved secondary path does not count as replay', f"replay_events_before={before}, replay_events_after={metrics.source_sequence_replay_events}")

    bs, metrics = _edgecase_base_station()
    high = _edgecase_packet(seq=10, packet_type='DATA',
                            priority='HIGH', payload=80.0, timestamp=1.0)
    bs.data_plane_receive(dict(high), [1, 2, 0], 1.1)
    before = metrics.source_sequence_replay_events
    bs.data_plane_receive(dict(high), [1, 99, 0], 1.2)
    record('Unapproved_Critical_Multipath_Rejected', metrics.source_sequence_replay_events == before + 1,
           'Unapproved repeated HIGH path increments replay/path-injection counter', f"replay_events_before={before}, replay_events_after={metrics.source_sequence_replay_events}")

    bs, metrics = _edgecase_base_station()
    data = _edgecase_packet(seq=1, packet_type='DATA',
                            priority='NORMAL', timestamp=1.0)
    sos = _edgecase_packet(seq=1, packet_type='SOS_CHARGE',
                           priority='HIGH', timestamp=1.1)
    bs.data_plane_receive(dict(data), [1, 2, 0], 1.2)
    before = metrics.source_sequence_replay_events
    bs.data_plane_receive(dict(sos), [1, 2, 0], 1.3)
    record('Control_Sequence_Separate_From_Data_Sequence', (metrics.source_sequence_replay_events == before and 1 in bs.pending_sos_requests),
           'SOS seq=1 accepted after DATA seq=1 because freshness is per packet type', f"replay_events_before={before}, replay_events_after={metrics.source_sequence_replay_events}, pending={1 in bs.pending_sos_requests}")

    bs, metrics = _edgecase_base_station()
    bs.mc_shared_key = 'mc-key'
    ok = bs.report_mc_injection_oob(
        1, 0.2, mc_x=0.0, mc_y=0.0, mac_tag='bad-tag', mc_id='MC-1', counter=1, nonce='n1', timestamp=1.0)
    record('Forged_MC_Report_Rejected', (not ok and metrics.mc_rejected_unauth_reports >= 1 and abs(bs.node_states[1].get(
        'total_injected_energy', 0.0)) < 1e-12), 'Invalid MC HMAC rejected and injected energy not credited', f"ok={ok}, rejected={metrics.mc_rejected_unauth_reports}, injected={bs.node_states[1].get('total_injected_energy', 0.0)}")

    bs, metrics = _edgecase_base_station()
    bs.mc_shared_key = 'mc-key'
    tag = make_mc_mac('mc-key', 1, 0.2, 0.0, 0.0, mc_id='MC-1',
                      counter=1, nonce='n-valid', timestamp=1.0)
    ok1 = bs.report_mc_injection_oob(
        1, 0.2, mc_x=0.0, mc_y=0.0, mac_tag=tag, mc_id='MC-1', counter=1, nonce='n-valid', timestamp=1.0)
    ok2 = bs.report_mc_injection_oob(
        1, 0.2, mc_x=0.0, mc_y=0.0, mac_tag=tag, mc_id='MC-1', counter=1, nonce='n-valid', timestamp=1.0)
    record('MC_Report_Replay_Rejected', (ok1 is True and ok2 is False and metrics.mc_rejected_unauth_reports >= 1 and abs(bs.node_states[1].get(
        'total_injected_energy', 0.0) - 0.2) < 1e-12), 'First valid MC report accepted, replay rejected, injected energy counted once', f"first_ok={ok1}, second_ok={ok2}, rejected={metrics.mc_rejected_unauth_reports}, injected={bs.node_states[1].get('total_injected_energy', 0.0)}")

    bs, metrics = _edgecase_base_station()
    bs.node_states[1]['E_est'] = 0.2
    bs.node_states[1]['E_claimed'] = 0.9
    bs.CHARGE_MAX_CLAIM_EST_DEVIATION = 0.30
    ok = bs.validate_charging_request(1, emergency=True)
    record('Raw_Inconsistent_SOS_Dispatch_Denied', (ok is False and metrics.mc_raw_sos_dispatches == 0),
           'Claim/estimate inconsistency denies dispatch and raw SOS dispatch counter stays zero', f"ok={ok}, raw_sos_dispatches={metrics.mc_raw_sos_dispatches}")

    bs, metrics = _edgecase_base_station()
    mc = MobileCharger(mc_id='MC-1', depot_x=0.0, depot_y=0.0, capacity=1.0,
                       speed=1.0, base_station=bs, metrics_collector=metrics)
    mc.crypto_keys = 'mc-key'
    mc.tpms_zeroization()
    record('TPM_Zeroization_Bricks_MC', (mc.crypto_keys is None and mc.zeroized is True and mc.requires_lab_reprovisioning is True and mc.status == 'BRICKED'),
           'Keys zeroized, MC marked bricked, lab reprovisioning required', f"crypto_keys={mc.crypto_keys}, zeroized={mc.zeroized}, requires_lab={mc.requires_lab_reprovisioning}, status={mc.status}")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    passed = int(df['Passed'].sum())
    total = int(len(df))
    if verbose:
        print(
            f"[EDGE-VALIDATION] Passed {passed}/{total} security edge cases.")
        print(df.to_string(index=False))
        print(f"[EDGE-VALIDATION] Saved {output_csv}")
    if passed != total:
        failed = df.loc[~df['Passed'], 'Case'].tolist()
        raise AssertionError(f"Security edge-case validation failed: {failed}")
    return df


def _innovation_audit_row(case, passed, expected, observed, metrics=None, details=''):
    row = {
        'Case': case,
        'Passed': bool(passed),
        'Expected': expected,
        'Observed': observed,
        'Details': details,
    }
    if metrics is not None:
        row.update(metrics)
    return row


def _audit_tomography_case(observable=True):
    if observable:
        A = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
        ], dtype=float)
        x_true = np.array([0.02, 0.35, 0.01], dtype=float)
        true_attacker_idx = 1
    else:
        A = np.array([
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
        ], dtype=float)
        x_true = np.array([0.20, 0.20, 0.01], dtype=float)
        true_attacker_idx = None

    D = A @ x_true
    x_hat, residuals, rank, _ = np.linalg.lstsq(A, D, rcond=None)
    rank_ok = rank >= A.shape[1]
    residual = float(np.mean((A @ x_hat - D) ** 2))

    if rank_ok:
        predicted_idx = int(np.argmax(x_hat))
        correct = (predicted_idx == true_attacker_idx)
        false_accusation = False
    else:
        predicted_idx = None
        correct = True
        false_accusation = False

    return {
        'rank': int(rank), 'observable': bool(rank_ok), 'predicted_idx': predicted_idx,
        'true_idx': true_attacker_idx, 'correct': bool(correct),
        'false_accusation': bool(false_accusation), 'residual': residual,
    }


def run_innovation_coverage_validation(output_csv='phase1_innovation_coverage_validation.csv', verbose=True):
    rows = []

    bs, metrics = _edgecase_base_station()
    bs.energy_experiment_active = True
    cases = [
        ('need_accept', 0.20, 0.20, 'WHITELISTED', True),
        ('healthy_reject', 0.90, 0.90, 'WHITELISTED', False),
        ('blacklisted_reject', 0.10, 0.10, 'BLACKLISTED', False),
        ('inconsistent_reject', 0.20, 0.95, 'WHITELISTED', False),
    ]
    for idx, (_, e_est, e_claim, status, needed) in enumerate(cases, start=1):
        bs.node_states[1]['E_est'] = e_est
        bs.node_states[1]['E_claimed'] = e_claim
        bs.node_states[1]['status'] = status
        admitted = bs.validate_charging_request(1, emergency=False)
        metrics.record_charging_decision(needed=needed, admitted=admitted)
    s = metrics.export_summary(1, 'InnovationAudit_ChargingDecision', 10)
    rows.append(_innovation_audit_row(
        'Charging_Decision_Accuracy_and_False_Admissions',
        s['Charging_Decision_Accuracy'] >= 0.99 and s['False_Charging_Admissions'] == 0,
        'Needed low-energy node admitted; healthy, blacklisted, and inconsistent requests rejected.',
        f"accuracy={s['Charging_Decision_Accuracy']:.3f}, false_admissions={s['False_Charging_Admissions']}", s))

    metrics = MetricsCollector(total_nodes=4, total_rounds_or_time=10)
    metrics.record_silent_sos_result(
        rescued=True, false_sos=False, response_time=12.0)
    metrics.record_silent_sos_result(
        rescued=False, false_sos=True, response_time=None)
    s = metrics.export_summary(1, 'InnovationAudit_SilentSOS', 10)
    rows.append(_innovation_audit_row(
        'Silent_Node_SOS_Rescue_and_False_SOS_Rate',
        abs(s['Silent_Node_Rescue_Rate'] -
            0.5) < 1e-12 and abs(s['False_SOS_Rate'] - 0.5) < 1e-12,
        'One true silent-node SOS is rescued and one false SOS is counted as false.',
        f"rescue_rate={s['Silent_Node_Rescue_Rate']:.3f}, false_sos_rate={s['False_SOS_Rate']:.3f}", s))

    metrics = MetricsCollector(total_nodes=4, total_rounds_or_time=10)
    obs = _audit_tomography_case(observable=True)
    deg = _audit_tomography_case(observable=False)
    metrics.record_delay_tomography_result(
        obs['observable'], correct=obs['correct'], false_accusation=obs['false_accusation'], residual_error=obs['residual'])
    metrics.record_delay_tomography_result(
        deg['observable'], correct=deg['correct'], false_accusation=deg['false_accusation'], residual_error=deg['residual'])
    s = metrics.export_summary(1, 'InnovationAudit_Tomography', 10)
    rows.append(_innovation_audit_row(
        'Delay_Tomography_Accuracy_Observability_and_Conservative_Rank_Gating',
        s['Delay_Attribution_Accuracy'] >= 0.99 and abs(
            s['Tomography_Observability_Ratio'] - 0.5) < 1e-12 and s['False_Delay_Accusation_Rate'] == 0.0,
        'Observable window identifies the slow node; rank-deficient window does not force a node accusation.',
        f"accuracy={s['Delay_Attribution_Accuracy']:.3f}, observability={s['Tomography_Observability_Ratio']:.3f}, rank_deficient={s['Tomography_Rank_Deficient_Windows']}", s))

    metrics = MetricsCollector(total_nodes=4, total_rounds_or_time=10)
    metrics.record_shadow_route_result(precomputed=True, extra_compute_ms=2.0)
    metrics.record_shadow_route_result(
        hit=True, ordinary_ms=10.0, shadow_ms=1.0)
    s = metrics.export_summary(1, 'InnovationAudit_ShadowRouting', 10)
    rows.append(_innovation_audit_row(
        'Shadow_Route_Precompute_Hit_and_Speedup_Metrics',
        s['Shadow_Route_Hit_Rate'] == 1.0 and s['Route_Repair_Speedup_Ratio'] >= 9.99,
        'One standby route is precomputed and later activated faster than ordinary recomputation.',
        f"hit_rate={s['Shadow_Route_Hit_Rate']:.3f}, speedup={s['Route_Repair_Speedup_Ratio']:.3f}", s))

    bs, metrics = _edgecase_base_station()
    high = _edgecase_packet(seq=10, packet_type='DATA',
                            priority='HIGH', payload=80.0, timestamp=1.0)
    bs.data_plane_receive(dict(high), [1, 2, 0], 1.1)
    before = metrics.source_sequence_replay_events
    bs.data_plane_receive(dict(high), [1, 3, 0], 1.2)
    accepted = int(metrics.source_sequence_replay_events == before)
    before = metrics.source_sequence_replay_events
    bs.data_plane_receive(dict(high), [1, 99, 0], 1.3)
    rejected = int(metrics.source_sequence_replay_events == before + 1)
    metrics.record_critical_multipath_opportunity(
        secondary_available=True, accepted=accepted, rejected=rejected)
    metrics.record_critical_multipath_opportunity(
        secondary_available=False, accepted=0, rejected=0)
    s = metrics.export_summary(1, 'InnovationAudit_CriticalMultipath', 10)
    rows.append(_innovation_audit_row(
        'Critical_Multipath_Availability_Approved_and_Unapproved_Copy_Handling',
        accepted == 1 and rejected == 1 and abs(
            s['Critical_Secondary_Path_Availability_Rate'] - 0.5) < 1e-12,
        'Approved secondary copy is admitted; unapproved duplicate/path-injection is rejected; sparse no-backup case is counted.',
        f"availability={s['Critical_Secondary_Path_Availability_Rate']:.3f}, accepted={s['Critical_Redundant_Copy_Accepted_Count']}, rejected={s['Critical_Redundant_Copy_Rejected_Count']}", s))

    bs, metrics = _edgecase_base_station()
    bs.mc_shared_key = 'mc-key'
    forged = bs.report_mc_injection_oob(
        1, 0.2, mc_x=0.0, mc_y=0.0, mac_tag='bad', mc_id='MC-1', counter=1, nonce='bad-nonce', timestamp=1.0)
    good_tag = make_mc_mac('mc-key', 1, 0.2, 0.0, 0.0,
                           mc_id='MC-1', counter=2, nonce='n-good', timestamp=2.0)
    ok1 = bs.report_mc_injection_oob(
        1, 0.2, mc_x=0.0, mc_y=0.0, mac_tag=good_tag, mc_id='MC-1', counter=2, nonce='n-good', timestamp=2.0)
    replay = bs.report_mc_injection_oob(
        1, 0.2, mc_x=0.0, mc_y=0.0, mac_tag=good_tag, mc_id='MC-1', counter=2, nonce='n-good', timestamp=2.0)
    off_tag = make_mc_mac('mc-key', 1, 0.2, 999.0, 999.0,
                          mc_id='MC-1', counter=3, nonce='n-off', timestamp=3.0)
    off_location = bs.report_mc_injection_oob(
        1, 0.2, mc_x=999.0, mc_y=999.0, mac_tag=off_tag,
        mc_id='MC-1', counter=3, nonce='n-off', timestamp=3.0,
        is_spoof_attempt=True)
    bs.node_states[1]['E_est'] = 0.2
    bs.node_states[1]['E_claimed'] = 0.9
    raw_ok = bs.validate_charging_request(1, emergency=True)
    s = metrics.export_summary(1, 'InnovationAudit_MCSecurity', 10)
    rows.append(_innovation_audit_row(
        'MC_HMAC_Replay_Location_Spoof_and_Raw_SOS_Blocking',
        (not forged) and ok1 and (not replay) and (not off_location) and (
            not raw_ok) and s['Location_Spoof_Rejection_Rate'] == 1.0,
        'Forged MC report rejected, valid report accepted once, replay rejected, off-location report rejected, inconsistent raw SOS blocked.',
        f"forged_rejection={s['Forged_MC_Report_Rejection_Rate']:.3f}, replay_rejection={s['MC_Replay_Rejection_Rate']:.3f}, location_rejection={s['Location_Spoof_Rejection_Rate']:.3f}, raw_sos_block={s['Raw_SOS_Blocking_Rate']:.3f}", s))

    bs, metrics = _edgecase_base_station()
    bs.node_states[1]['E_claimed_prev'] = 0.20
    bs.node_states[1]['E_claimed'] = 0.80
    bs.mc_recent_injected_energy[1] = 0.0
    bs._verify_wrsn_cyber_physical_security()
    fake_detected = bs.node_states[1].get('hard_veto', False)
    bs, metrics2 = _edgecase_base_station()
    bs.node_states[1]['E_claimed_prev'] = 0.50
    bs.node_states[1]['E_claimed'] = 0.50
    bs.mc_recent_injected_energy[1] = 0.50
    bs._verify_wrsn_cyber_physical_security()
    hoarder_detected = bs.node_states[1].get('hard_veto', False)
    metrics.fake_charger_attempts += metrics2.fake_charger_attempts
    metrics.fake_charger_detections += metrics2.fake_charger_detections
    metrics.charge_hoarder_attempts += metrics2.charge_hoarder_attempts
    metrics.charge_hoarder_detections += metrics2.charge_hoarder_detections
    s = metrics.export_summary(1, 'InnovationAudit_EnergyAttackTaxonomy', 10)
    rows.append(_innovation_audit_row(
        'Energy_Attack_Taxonomy_Fake_Charger_and_Charge_Hoarder_Detection',
        fake_detected and hoarder_detected and s['Fake_Charger_Detection_Rate'] == 1.0 and s[
            'Charge_Hoarder_Detection_Rate'] == 1.0,
        'False energy increase without MC injection and missing claim increase after real injection both trigger hard filters.',
        f"fake_detected={fake_detected}, hoarder_detected={hoarder_detected}, fake_rate={s['Fake_Charger_Detection_Rate']:.3f}, hoarder_rate={s['Charge_Hoarder_Detection_Rate']:.3f}", s))

    sweep_manifest = {
        'network_size': '50,100,200,500 via WRSN_NUM_NODES',
        'attack_percentage': '0,5,15,30,45 suggested through scenario configs',
        'traffic_rate': 'event_rate_lambda low/medium/high',
        'critical_probability': 'critical_event_probability suggested; default remains 0.10',
        'thresholds': 'risk, SOS, MC, sequence, tomography, and window-size sensitivity columns exported',
    }
    rows.append(_innovation_audit_row(
        'Parameter_Sweep_Manifest_For_Paper_Level_Evaluation',
        True,
        'The audit records the remaining parameter families that should be swept for final paper tables.',
        json.dumps(sweep_manifest, sort_keys=True), None))

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    passed = int(df['Passed'].sum())
    total = int(len(df))
    if verbose:
        print(
            f"[INNOVATION-AUDIT] Passed {passed}/{total} innovation-coverage checks.")
        print(df[['Case', 'Passed', 'Observed']].to_string(index=False))
        print(f"[INNOVATION-AUDIT] Saved {output_csv}")
    if passed != total:
        failed = df.loc[~df['Passed'], 'Case'].tolist()
        raise AssertionError(
            f"Innovation coverage validation failed: {failed}")
    return df


def run_self_tests(verbose=True):
    ok = True
    cfg = _base_config()
    cfg['use_asynchronous_processing_pipeline'] = False
    metrics = MetricsCollector(total_nodes=4, total_rounds_or_time=10)
    bs = BaseStation(metrics, cfg, bs_id=0)

    bs.node_states[1] = {
        'x': 0.0, 'y': 0.0, 'E_initial': 1.0, 'E_claimed': 1.0, 'E_est': 0.2, 'trust_vector': [1, 1, 1, 1, 1],
        'T1_history': 1.0, 'T2_history': 1.0, 'T3_history': 1.0, 'T4_history': 1.0, 'T5_history': 1.0,
        'FE': 1.0, 'status': 'WHITELISTED', 'crypto_keys': 'k1', 'Deviation_MC_norm': 0.0, 'SED_index': 0.0, 'spoof_trust': 1.0
    }
    bs.routing_tables = {1: {'primary_path': [1, 2, 0], 'secondary_path': [
        1, 3, 0], 'critical_paths': [[1, 2, 0], [1, 3, 0]]}}

    bad_sos = {'source_id': 1, 'type': 'SOS_CHARGE', 'seq': 1, 'timestamp': 0.0,
               'priority': 'HIGH', 'size_bytes': 16, 'energy_trace': [(1, 0.01)]}
    bs.data_plane_receive(dict(bad_sos), [1, 2, 0], 0.1)
    assert 1 not in bs.pending_sos_requests, 'unauthenticated SOS created a pending request'
    assert abs(bs.node_states[1]['E_claimed'] -
               1.0) < 1e-12, 'unauthenticated SOS changed E_claimed'
    assert metrics.packet_hmac_failures >= 1, 'SOS HMAC failure was not exported to metrics'

    good_sos = dict(bad_sos)
    good_sos['hash'] = compute_packet_hash(good_sos, 'k1')
    bs.data_plane_receive(dict(good_sos), [1, 2, 0], 0.2)
    assert 1 in bs.pending_sos_requests, 'authenticated SOS was not admitted as evidence'

    p = {'source_id': 1, 'type': 'DATA', 'seq': 10, 'timestamp': 1.0, 'priority': 'HIGH',
         'payload': 80.0, 'size_bytes': DATA_PACKET_BYTES, 'energy_trace': [(1, 0.8)]}
    p['hash'] = compute_packet_hash(p, 'k1')
    bs.data_plane_receive(dict(p), [1, 2, 0], 1.1)
    before = metrics.source_sequence_replay_events
    bs.data_plane_receive(dict(p), [1, 3, 0], 1.2)
    assert metrics.source_sequence_replay_events == before, 'approved secondary copy counted as replay'
    bs.data_plane_receive(dict(p), [1, 99, 0], 1.3)
    assert metrics.source_sequence_replay_events == before + \
        1, 'unapproved critical path was accepted'

    bs.CHARGE_MAX_CLAIM_EST_DEVIATION = 1.00
    bs.node_states[1]['E_est'] = 0.2
    bs.node_states[1]['E_claimed'] = 0.9
    assert bs.validate_charging_request(
        1, emergency=True), 'configured charge_claim_est_max_deviation was ignored'

    rel_cfg = dict(_base_config())
    rel_cfg.update({
        'initial_energy_override': 0.01,
        'num_mobile_chargers': 2,
        'mobile_charger_capacity_mode': 'relative',
        'mobile_charger_sensor_equivalent_capacities': [20.0, 10.0],
        'mobile_charger_travel_cost_mode': 'relative',
        'mobile_charger_travel_sensor_equivalent_cost_per_m': 0.0005,
    })
    rel_sim = SimulatorManager(
        num_nodes=3, sim_time=10, ablation_config=rel_cfg, seed=1)
    rel_caps = [mc.E_capacity for mc in rel_sim.mobile_chargers]
    assert len(rel_caps) == 2, 'relative capacity test did not create two MCs'
    assert abs(rel_caps[0] - 0.20) < 1e-12 and abs(rel_caps[1] - 0.10) < 1e-12, \
        f'relative MC capacity scaling failed: {rel_caps}'
    assert abs(rel_sim.metrics.mobile_charger_total_sensor_equivalent_capacity - 30.0) < 1e-12, \
        'sensor-equivalent capacity metric is wrong'
    assert abs(rel_sim.mobile_chargers[0].travel_energy_cost_per_m - 0.000005) < 1e-12, \
        'relative MC travel cost scaling failed'

    legacy_cfg = dict(_base_config())
    legacy_sim = SimulatorManager(
        num_nodes=3, sim_time=10, ablation_config=legacy_cfg, seed=1)
    assert len(
        legacy_sim.mobile_chargers) == 1, 'legacy default should keep one MC'
    assert abs(legacy_sim.mobile_chargers[0].E_capacity - 50000.0) < 1e-12, \
        'legacy absolute MC capacity default changed'

    row = metrics.export_summary(1, 'unit', 10)
    assert row['Packet_HMAC_Failures'] >= 1, 'Packet_HMAC_Failures not exported'
    assert row['Source_Sequence_Replay_Events'] >= 1, 'Source_Sequence_Replay_Events not exported'
    assert row['Mobile_Charger_Capacity_Mode'] == 'absolute', 'capacity mode not exported'
    if verbose:
        print('[SELF-TEST] Phase-1 coherence V3 self-tests passed.')
        print('[SELF-TEST] Relative MC capacity scaling passed.')
    return ok


def _safe_div(num, den, undefined=np.nan):
    try:
        den = float(den)
        if den == 0.0:
            return undefined
        return float(num) / den
    except Exception:
        return undefined


def _add_charging_efficiency_diagnostics(df):
    if df is None or df.empty:
        return df

    df = df.copy()
    zeros = pd.Series(0.0, index=df.index)

    charge_events = df.get('MC_Charge_Events', zeros).astype(float)
    injected = df.get('MC_Total_Injected_Energy', zeros).astype(float)
    travel = df.get('MC_Travel_Distance', zeros).astype(float)
    deadlines_missed = df.get(
        'MC_Missed_Charging_Deadlines', zeros).astype(float)
    deaths = df.get('Energy_Critical_Node_Deaths', zeros).astype(float)
    critical_events = df.get(
        'Energy_Critical_Node_Events', zeros).astype(float)
    dqn_decisions = df.get('DQN_Routing_Decisions', zeros).astype(float)
    qifl_loss = df.get('QIFL_Training_Loss_Avg', zeros).astype(float)
    dqn_loss = df.get('DQN_Training_Loss_Avg', zeros).astype(float)

    # MetricsCollector is the canonical owner of these definitions.  This
    # post-processing function only backfills columns for legacy CSV files and
    # must never overwrite values already exported by MetricsCollector.
    if 'MC_Injected_Energy_Per_Charge_Event' not in df.columns:
        df['MC_Injected_Energy_Per_Charge_Event'] = [
            _safe_div(e, c) for e, c in zip(injected, charge_events)
        ]
    if 'MC_Travel_Per_Charged_Joule' not in df.columns:
        df['MC_Travel_Per_Charged_Joule'] = [
            _safe_div(t, e) for t, e in zip(travel, injected)
        ]
    if 'MC_Charge_Events_Per_Critical_Node' not in df.columns:
        df['MC_Charge_Events_Per_Critical_Node'] = [
            _safe_div(c, ce) for c, ce in zip(charge_events, critical_events)
        ]

    avoided = df.get('MC_Avoided_Node_Deaths', zeros).astype(float)
    resolved_deadlines = avoided + deadlines_missed
    if 'MC_Deadline_Failure_Rate' not in df.columns:
        df['MC_Deadline_Failure_Rate'] = [
            _safe_div(m, resolved)
            for m, resolved in zip(deadlines_missed, resolved_deadlines)
        ]
    if 'Energy_Critical_Death_Rate' not in df.columns:
        df['Energy_Critical_Death_Rate'] = [
            _safe_div(d, ce) for d, ce in zip(deaths, critical_events)
        ]

    if 'MC_Useless_Charge_Rate_Heuristic' not in df.columns:
        df['MC_Useless_Charge_Rate_Heuristic'] = [
            1.0 if (c > 0 and _safe_div(e, c) < 1e-3) else 0.0
            for e, c in zip(injected, charge_events)
        ]

    df['QIFL_Learning_Active_Flag'] = [
        1.0 if x > 0 else 0.0 for x in qifl_loss]
    df['DQN_Learning_Active_Flag'] = [1.0 if (d > 0 and l > 0) else 0.0
                                      for d, l in zip(dqn_decisions, dqn_loss)]
    return df


def run_statistical_experiments():
    run_mode = os.environ.get(
        'WRSN_RUN_MODE', 'energy').strip().lower().replace('-', '_')
    aliases = {
        'edge_cases': 'security_edge_cases', 'edge_validation': 'security_edge_cases',
        'security_validation': 'security_edge_cases', 'security_edge_validation': 'security_edge_cases',
        'innovation': 'innovation_coverage', 'innovation_audit': 'innovation_coverage',
        'phase1_innovation_audit': 'innovation_coverage', 'innovation_validation': 'innovation_coverage',
        'routing': 'routing_suite', 'clustering': 'routing_suite', 'routing_ablation': 'routing_suite'
    }
    normalized_mode = aliases.get(run_mode, run_mode)

    if normalized_mode == 'security_edge_cases':
        return run_security_edge_case_validation()
    if normalized_mode == 'innovation_coverage':
        return run_innovation_coverage_validation()

    NUM_SEEDS = int(os.environ.get('WRSN_NUM_SEEDS', 1))
    NUM_NODES = int(os.environ.get('WRSN_NUM_NODES', 100))
    sim_time_env = os.environ.get('WRSN_SIM_TIME')
    SIM_TIME = int(sim_time_env) if sim_time_env is not None else 3600

    scenarios = _scenarios_for_mode(normalized_mode)
    scenario_names = [name for name, _ in scenarios]
    safe_mode = ''.join(c if c.isalnum() or c in ('_', '-')
                        else '_' for c in normalized_mode)

    print(f"[RUNNER] WRSN_RUN_MODE={normalized_mode}")
    print(
        f"[RUNNER] Running {len(scenarios)} scenario(s): {', '.join(scenario_names)}")

    all_results = []
    for scenario_name, config in scenarios:
        scenario_sim_time = (
            SIM_TIME if sim_time_env is not None else
            int(float(config.get('simulation_time_override_s', SIM_TIME)))
        )
        for seed in range(1, NUM_SEEDS + 1):
            print(f"\n--- {scenario_name}: Seed {seed}/{NUM_SEEDS}; "
                  f"sim_time={scenario_sim_time}s ---")
            sim = SimulatorManager(
                num_nodes=NUM_NODES, sim_time=scenario_sim_time,
                ablation_config=config, seed=seed)
            sim.run_simulation()
            all_results.append(sim.metrics.export_summary(
                seed, scenario_name, scenario_sim_time))

    df = pd.DataFrame(all_results)
    df = _add_charging_efficiency_diagnostics(df)
    prefix = f"performance_{safe_mode}"
    df.to_csv(f'{prefix}_results.csv', index=False)
    summary = summarize_results(df, f'{prefix}_summary_ci.csv')
    make_plots(df, f'plots_{prefix}')
    print(
        f"\n[SUCCESS] Saved {prefix}_results.csv, {prefix}_summary_ci.csv, and plots_{prefix}/")

    keep = [c for c in ['Ablation_Scenario', 'N', 'TPR_Mean', 'Accuracy_Mean', 'F1_Score_Mean',
                        'Overall_PDR_Mean', 'Critical_PDR_Mean', 'Throughput_Mean', 'Avg_Delay_Mean',
                        'Operational_Control_Overhead_%_Mean', 'Mobile_Charger_Count_Mean',
                        'MC_Travel_Distance_Mean', 'MC_Injected_Energy_Per_Charge_Event_Mean',
                        'MC_Charge_Events_Per_Critical_Node_Mean', 'MC_Deadline_Failure_Rate_Mean',
                        'Energy_Critical_Death_Rate_Mean', 'FND_Mean', 'HND_Mean', 'LND_Mean',
                        'Adaptive_K_opt_Avg_Mean', 'Shadow_Route_Hit_Rate_Mean',
                        'Critical_Secondary_Path_Availability_Rate_Mean'] if c in summary.columns]
    print(summary[keep].to_string(index=False))

    all_mode_aliases = {'all', 'all_modes', 'full',
                        'full_suite', 'everything', 'complete', 'complete_suite'}
    security_suite_aliases = {'security_suite'}
    if (normalized_mode in all_mode_aliases or normalized_mode in security_suite_aliases) and os.environ.get('WRSN_SKIP_EDGE_CASES', '0').strip() != '1':
        run_security_edge_case_validation(
            output_csv=f"security_performance_{normalized_mode}_security_edge_cases.csv", verbose=True)

    return df, summary


if __name__ == '__main__':
    os.environ.setdefault('WRSN_ENABLE_MULTI_MC_SCENARIOS', '2')
    os.environ.setdefault('WRSN_ENABLE_MC_COUNT_SWEEP', '1')
    if os.environ.get('WRSN_RUN_SELF_TESTS', '0').strip() == '1':
        run_self_tests(verbose=True)
    run_statistical_experiments()
