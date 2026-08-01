import math
import json
import re
import numpy as np
import pandas as pd


class MetricsCollector:
    def __init__(self, total_nodes, total_rounds_or_time):
        self.total_nodes = total_nodes
        self.total_rounds_or_time = total_rounds_or_time
        self.current_sim_time = 0.0
        self._unique_generated_data_keys = set()
        self._unique_delivered_data_keys = set()

        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.detection_delays = []

        self.generated_data_packets = 0
        self.received_data_packets = 0
        self.generated_critical_packets = 0
        self.received_critical_packets = 0
        self.total_bits_received = 0
        self.end_to_end_delays = []

        self.setup_control_packets = 0
        self.operational_control_packets = 0
        self.control_packets = 0
        self.alive_nodes = total_nodes
        self.node_death_times = {}
        self._currently_dead_nodes = set()
        self.network_residual_energy_history = []
        self.mc_travel_distance = 0.0
        self.mc_travel_energy = 0.0
        self.mc_rf_transmit_energy = 0.0
        self.mc_contact_charging_energy = 0.0
        self.rf_total_delivered_energy = 0.0

        self.fuzzy_objective_values = []
        self.fuzzy_silhouette_scores = []
        self.fuzzy_partition_coefficients = []
        self.fuzzy_partition_entropies = []
        self.xie_beni_indices = []

        self.edge_processing_times = []
        self.bs_processing_times = []

        self.mc_charge_events = 0
        self.mc_total_injected_energy = 0.0
        self.mc_total_charging_time = 0.0
        self.mc_charging_response_times = []
        self.mc_validated_charging_requests = 0
        self.mc_rejected_charging_requests = 0
        self.mc_missed_charging_deadlines = 0
        self.mc_avoided_node_deaths = 0
        self.energy_critical_node_events = 0
        self.energy_critical_node_deaths = 0
        self._critical_nodes_seen = set()
        self._charging_request_first_seen_time = {}
        self._active_charging_requests = set()
        self._rejected_charging_request_keys = set()

        self.mc_fake_reports_attempted = 0
        self.mc_fake_reports_accepted = 0
        self.mc_location_verification_checks = 0
        self.mc_location_verification_passes = 0
        self.mc_location_verification_failures = 0
        self.mc_location_spoof_attempts = 0
        self.mc_location_spoof_accepted = 0
        self.mc_location_spoof_detections = 0
        self.mc_raw_sos_dispatches = 0
        self.mc_rejected_unauth_reports = 0
        self.mc_tpm_compromise_events = 0
        self.mc_tpm_zeroized_events = 0

        self.source_sequence_replay_events = 0
        self.packet_hmac_failures = 0

        # Implicit hardware MAC-ACK accounting. In the default non-regression
        # mode the cost is measured but not double-deducted from the legacy
        # aggregate radio-energy envelope.
        self.implicit_mac_ack_transactions = 0
        self.implicit_mac_ack_energy = 0.0
        self.implicit_mac_ack_physically_deducted_energy = 0.0

        # Independent charging-audit evidence.
        self.node_charge_receipts_received = 0
        self.node_charge_receipts_consistent = 0
        self.node_charge_receipts_inconsistent = 0
        self.node_charge_receipt_auth_failures = 0
        self.mc_energy_claim_mismatches = 0
        self.mc_location_report_rejections = 0
        self.energy_trust_recoveries = 0

        # INNOVATION AUDIT METRICS
        self.charging_decision_tp = 0
        self.charging_decision_fp = 0
        self.charging_decision_tn = 0
        self.charging_decision_fn = 0
        self._charging_decision_keys = set()
        self.silent_node_sos_count = 0
        self.silent_node_rescue_count = 0
        self.false_sos_count = 0
        self.time_to_critical_before_mc_arrival = []

        self.delay_attribution_trials = 0
        self.delay_attribution_correct = 0
        self.false_delay_accusations = 0
        self.tomography_windows = 0
        self.tomography_observable_windows = 0
        self.tomography_rank_deficient_windows = 0
        self.tomography_residual_errors = []

        self.shadow_precompute_count = 0
        self.shadow_route_hit_count = 0
        self.ordinary_route_repair_times_ms = []
        self.shadow_route_repair_times_ms = []
        self.shadow_extra_compute_times_ms = []

        self.critical_secondary_path_opportunities = 0
        self.critical_secondary_path_available = 0
        self.critical_redundant_copy_accepted = 0
        self.critical_redundant_copy_rejected = 0
        self.sparse_topology_no_backup_count = 0

        self.mc_replay_rejections = 0
        self.charge_hoarder_attempts = 0
        self.charge_hoarder_detections = 0
        self.fake_charger_attempts = 0
        self.fake_charger_detections = 0
        self.raw_sos_validation_attempts = 0
        self.raw_sos_blocked = 0

        # ADAPTIVE QIFL-DQN CHARGING METRICS
        self.adaptive_charge_thresholds = []
        self.qifl_training_losses = []
        self.sos_threshold_energies = []
        self.charging_kmeans_cluster_counts = []
        self.charging_cluster_assigned_node_counts = []
        self.dqn_routing_decisions = 0
        self.dqn_competitive_decisions = 0
        self.dqn_single_candidate_decisions = 0
        self.dqn_effective_exploration_count = 0
        self.dqn_safety_override_count = 0
        self.dqn_training_losses = []
        self.dqn_rewards = []
        self.dqn_reward_energy_components = []
        self.dqn_reward_deadline_components = []
        self.dqn_reward_travel_components = []
        self.dqn_reward_revisit_components = []
        self.dqn_reward_death_components = []
        self.adaptive_hyperparameters = {}
        self.dynamic_trust_weights_history = []
        self.mc_service_durations = []
        self.mc_fail_closed_rejections = 0
        self.experiment_parameters = {}
        self.adaptive_compute_device = 'numpy'
        self.dqn_torch_enabled = False
        self.qifl_torch_enabled = False
        self.cuda_available = False
        self.mc_useless_charge_events = 0
        self.mc_repeated_service_events = 0
        self.mc_total_revisit_events = 0
        self.mc_revisit_intervals = []
        self.adaptive_minimum_charge_ratios = []
        self.mc_applied_minimum_charge_ratios = []
        self.ewma_energy_abs_errors = []
        self.qifl_admission_count = 0

        # THREE-VERSION ENERGY-STRESS ABLATION METRICS
        self.qifl_min_gap_conflict_rejections = 0
        self.qifl_min_gap_conflict_preserved = 0
        self.verified_emergency_security_override_count = 0

        # Multi-Node RF Charging & Dynamic Partial Allocation Metrics
        self.rf_cluster_charge_events = 0
        self.rf_total_nodes_charged_in_clusters = 0
        self.dynamic_partial_charge_events = 0
        self.fixed_charger_charge_events = 0
        self.fixed_charger_total_injected_energy = 0.0
        self.fixed_charger_active_nodes = 0
        self.mc_infeasible_mission_rejections = 0
        self.mc_stranded_events = 0
        self.mc_async_missions_started = 0
        self.mc_async_missions_completed = 0
        self.mc_async_missions_incomplete = 0
        self.mc_network_events_while_busy = 0
        self.charging_service_pool_admissions = 0
        self.charging_plus_pool_admissions = 0
        self.charging_opportunistic_plus_charges = 0

        # [INNOVATION] Hybrid Hierarchical Routing & Fuzzy Clustering Metrics
        self.adaptive_k_opts = []

        # DATA-only routing overlay diagnostics.  These metrics are separated
        # from all mobile-charging metrics.
        self.data_routing_diagnostics_history = []

    def record_data_routing_diagnostics(self, diagnostics):
        if not diagnostics:
            return
        clean = {}
        for key, value in dict(diagnostics).items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                clean[str(key)] = number
        if clean:
            self.data_routing_diagnostics_history.append(clean)

    def update_packet_transmission(self, is_critical, is_control, size_bytes, is_delivered=False, delay=None, overhead_phase='operational', count_generated=True, size_bits=None):
        if is_control:
            if overhead_phase == 'setup':
                self.setup_control_packets += 1
            else:
                self.operational_control_packets += 1
            self.control_packets = self.setup_control_packets + \
                self.operational_control_packets
        else:
            if count_generated:
                self.generated_data_packets += 1
                if is_critical:
                    self.generated_critical_packets += 1
            if is_delivered:
                self.received_data_packets += 1
                bit_count = float(size_bytes) * \
                    8.0 if size_bits is None else float(size_bits)
                self.total_bits_received += bit_count
                if is_critical:
                    self.received_critical_packets += 1
                if delay is not None:
                    self.end_to_end_delays.append(delay)

    def update_ids_matrix(self, tp_increment, tn_increment, fp_increment, fn_increment):
        self.tp += tp_increment
        self.tn += tn_increment
        self.fp += fp_increment
        self.fn += fn_increment

    def record_node_death(self, node_id, current_time):
        first_death = node_id not in self.node_death_times
        was_currently_dead = node_id in self._currently_dead_nodes
        if first_death:
            self.node_death_times[node_id] = float(current_time)
        if not was_currently_dead:
            self._currently_dead_nodes.add(node_id)
            self.alive_nodes = max(0, self.alive_nodes - 1)
        if first_death and node_id in self._critical_nodes_seen:
            self.energy_critical_node_deaths += 1
            self.mc_missed_charging_deadlines += 1
            self._critical_nodes_seen.discard(node_id)

    def record_node_revival(self, node_id, current_time=None):
        if node_id in self._currently_dead_nodes:
            self._currently_dead_nodes.discard(node_id)
            self.alive_nodes = min(self.total_nodes, self.alive_nodes + 1)

    def get_charging_request_wait_time(self, node_id, current_time):
        first_seen = self._charging_request_first_seen_time.get(node_id)
        if first_seen is None:
            return 0.0
        return max(0.0, float(current_time) - float(first_seen))

    def record_fixed_charge(self, node_id, injected_energy):
        injected_energy = max(0.0, float(injected_energy))
        if injected_energy <= 0.0:
            return
        self.fixed_charger_charge_events += 1
        self.fixed_charger_total_injected_energy += injected_energy

    def record_energy_state(self, current_avg_energy):
        self.network_residual_energy_history.append(
            max(0.0, float(current_avg_energy))
        )

    def add_mc_travel_distance(self, distance, energy_consumed=0.0):
        self.mc_travel_distance += max(0.0, float(distance))
        self.mc_travel_energy += max(0.0, float(energy_consumed))

    def record_mc_charging_energy(self, energy_spent, *, rf_mode=False,
                                  delivered_energy=0.0):
        energy_spent = max(0.0, float(energy_spent))
        delivered_energy = max(0.0, float(delivered_energy))
        if rf_mode:
            self.mc_rf_transmit_energy += energy_spent
            self.rf_total_delivered_energy += delivered_energy
        else:
            self.mc_contact_charging_energy += energy_spent

    def record_implicit_mac_ack(self, energy_j, physically_deducted=False):
        energy_j = max(0.0, float(energy_j))
        self.implicit_mac_ack_transactions += 1
        self.implicit_mac_ack_energy += energy_j
        if physically_deducted:
            self.implicit_mac_ack_physically_deducted_energy += energy_j

    def record_execution_time(self, component, exec_time_ms):
        if component == 'EDGE':
            self.edge_processing_times.append(exec_time_ms)
        elif component == 'CORE':
            self.bs_processing_times.append(exec_time_ms)

    def record_charging_request(self, node_id, request_time, validated=True):
        node_id = int(node_id)
        request_time = float(request_time)
        if validated:
            if node_id not in self._active_charging_requests:
                self.mc_validated_charging_requests += 1
                self._active_charging_requests.add(node_id)
                self._charging_request_first_seen_time.setdefault(
                    node_id, request_time)
            return

        epoch_key = (node_id, int(request_time // 10.0))
        if epoch_key not in self._rejected_charging_request_keys:
            self._rejected_charging_request_keys.add(epoch_key)
            self.mc_rejected_charging_requests += 1

    def record_energy_critical(self, node_id, current_time):
        if node_id not in self._critical_nodes_seen:
            self._critical_nodes_seen.add(node_id)
            self.energy_critical_node_events += 1
            self._charging_request_first_seen_time.setdefault(
                node_id, float(current_time))

    def record_mc_charge(self, node_id, injected_energy, charging_time, current_time):
        injected_energy = float(max(0.0, injected_energy))
        charging_time = float(max(0.0, charging_time))
        if injected_energy <= 0.0:
            return
        self.mc_charge_events += 1
        self.mc_total_injected_energy += injected_energy
        self.mc_total_charging_time += charging_time
        start_time = self._charging_request_first_seen_time.pop(node_id, None)
        self._active_charging_requests.discard(int(node_id))
        if start_time is not None:
            self.mc_charging_response_times.append(
                max(0.0, float(current_time) - float(start_time)))
        if node_id in self._critical_nodes_seen:
            self.mc_avoided_node_deaths += 1
            self._critical_nodes_seen.discard(node_id)

    # INNOVATION AUDIT RECORDING METHODS
    def record_charging_decision(self, needed, admitted, decision_key=None):
        if decision_key is not None:
            key = tuple(decision_key) if isinstance(
                decision_key, (list, tuple)) else decision_key
            if key in self._charging_decision_keys:
                return
            self._charging_decision_keys.add(key)
        needed = bool(needed)
        admitted = bool(admitted)
        if needed and admitted:
            self.charging_decision_tp += 1
        elif (not needed) and admitted:
            self.charging_decision_fp += 1
        elif (not needed) and (not admitted):
            self.charging_decision_tn += 1
        else:
            self.charging_decision_fn += 1

    def record_silent_sos_result(self, rescued=False, false_sos=False, response_time=None):
        self.silent_node_sos_count += 1
        if rescued:
            self.silent_node_rescue_count += 1
        if false_sos:
            self.false_sos_count += 1
        if response_time is not None:
            self.time_to_critical_before_mc_arrival.append(
                max(0.0, float(response_time)))

    def record_delay_tomography_result(self, observable, correct=False, false_accusation=False, residual_error=None):
        self.tomography_windows += 1
        self.delay_attribution_trials += 1
        if observable:
            self.tomography_observable_windows += 1
        else:
            self.tomography_rank_deficient_windows += 1
        if correct:
            self.delay_attribution_correct += 1
        if false_accusation:
            self.false_delay_accusations += 1
        if residual_error is not None:
            self.tomography_residual_errors.append(float(residual_error))

    def record_shadow_route_result(self, precomputed=False, hit=False, ordinary_ms=None, shadow_ms=None, extra_compute_ms=None):
        if precomputed:
            self.shadow_precompute_count += 1
        if hit:
            self.shadow_route_hit_count += 1
        if ordinary_ms is not None:
            self.ordinary_route_repair_times_ms.append(float(ordinary_ms))
        if shadow_ms is not None:
            self.shadow_route_repair_times_ms.append(float(shadow_ms))
        if extra_compute_ms is not None:
            self.shadow_extra_compute_times_ms.append(float(extra_compute_ms))

    def record_critical_multipath_opportunity(self, secondary_available=False, accepted=0, rejected=0):
        self.critical_secondary_path_opportunities += 1
        if secondary_available:
            self.critical_secondary_path_available += 1
        else:
            self.sparse_topology_no_backup_count += 1
        self.critical_redundant_copy_accepted += int(max(0, accepted))
        self.critical_redundant_copy_rejected += int(max(0, rejected))

    @staticmethod
    def _safe_rate(num, den, scale=1.0, undefined=0.0):
        """Return a rate with an explicit zero-denominator policy.

        Most operational counters retain the historical ``0.0`` fallback.
        Scientific ratios whose denominator represents an observed event use
        ``undefined=np.nan`` so that "not applicable" is not silently reported
        as a measured zero.
        """
        try:
            den = float(den)
            num = float(num)
        except (TypeError, ValueError):
            return undefined
        return (num / den) * scale if den > 0.0 else undefined

    def calculate_ids_metrics(self):
        tp, tn, fp, fn = self.tp, self.tn, self.fp, self.fn
        total_pos = tp + fn
        total_neg = tn + fp
        total = total_pos + total_neg
        tpr = tp / total_pos if total_pos else 0.0
        fnr = fn / total_pos if total_pos else 0.0
        fpr = fp / total_neg if total_neg else 0.0
        tnr = tn / total_neg if total_neg else 0.0
        accuracy = (tp + tn) / total if total else 0.0
        balanced_accuracy = 0.5 * (tpr + tnr)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        npv = tn / (tn + fn) if (tn + fn) else 0.0
        f1 = (2 * precision * tpr / (precision + tpr)
              ) if (precision + tpr) else 0.0
        denom = math.sqrt(max((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn), 1e-12))
        mcc = ((tp * tn) - (fp * fn)) / denom if total else 0.0
        return {
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'Confusion_Matrix': f"[[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]",
            'TPR': tpr, 'Recall': tpr, 'FPR': fpr, 'TNR': tnr, 'Specificity': tnr,
            'FNR': fnr, 'Accuracy': accuracy, 'Balanced_Accuracy': balanced_accuracy,
            'Precision': precision, 'NPV': npv, 'F1_Score': f1, 'MCC': mcc
        }

    def calculate_qos_metrics(self, total_simulation_time):
        overall_pdr = (self.received_data_packets / self.generated_data_packets) * \
            100 if self.generated_data_packets > 0 else 0.0
        critical_pdr = (self.received_critical_packets / self.generated_critical_packets) * \
            100 if self.generated_critical_packets > 0 else 0.0
        throughput = (self.total_bits_received /
                      total_simulation_time) if total_simulation_time > 0 else 0.0
        avg_delay = np.mean(self.end_to_end_delays) if len(
            self.end_to_end_delays) > 0 else 0.0
        return {"Overall_PDR": overall_pdr, "Critical_PDR": critical_pdr, "Throughput_bps": throughput, "Avg_Delay": avg_delay}

    def calculate_network_lifetime(self):
        death_times = sorted(self.node_death_times.values())
        fnd = death_times[0] if death_times else -1

        half_dead_count = max(1, int(math.ceil(self.total_nodes / 2.0)))
        hnd = (death_times[half_dead_count - 1]
               if len(death_times) >= half_dead_count else -1)
        lnd = (death_times[self.total_nodes - 1]
               if self.total_nodes > 0 and len(death_times) >= self.total_nodes
               else -1)
        return {"FND": fnd, "HND": hnd, "LND": lnd}

    def calculate_overhead_ratio(self, phase='total'):
        if phase == 'setup':
            control_packets = self.setup_control_packets
        elif phase == 'operational':
            control_packets = self.operational_control_packets
        else:
            control_packets = self.control_packets
        total_packets = self.generated_data_packets + self.control_packets
        overhead_ratio = (control_packets / total_packets) * \
            100 if total_packets > 0 else 0.0
        return overhead_ratio

    def calculate_computational_overhead(self):
        avg_edge = np.mean(
            self.edge_processing_times) if self.edge_processing_times else 0.0
        avg_core = np.mean(
            self.bs_processing_times) if self.bs_processing_times else 0.0
        return {"Avg_Edge_Time_ms": avg_edge, "Avg_Core_Time_ms": avg_core}

    def export_summary(self, seed_num, ablation_scenario_name, sim_time):
        ids = self.calculate_ids_metrics()
        qos = self.calculate_qos_metrics(sim_time)
        lifetime = self.calculate_network_lifetime()
        computational = self.calculate_computational_overhead()

        summary = {
            'Seed': seed_num,
            'Ablation_Scenario': ablation_scenario_name,
            **ids,
            'Generated_Data_Packets': self.generated_data_packets,
            'Received_Data_Packets': self.received_data_packets,
            'Generated_Critical_Packets': self.generated_critical_packets,
            'Received_Critical_Packets': self.received_critical_packets,
            'Operational_Control_Packets': self.operational_control_packets,
            'Setup_Control_Packets': self.setup_control_packets,
            'Total_Control_Packets': self.control_packets,
            'Overall_PDR': qos['Overall_PDR'],
            'Critical_PDR': qos['Critical_PDR'],
            'Throughput': qos['Throughput_bps'],
            'Avg_Delay': qos['Avg_Delay'],
            'FND': lifetime['FND'], 'HND': lifetime['HND'], 'LND': lifetime['LND'],
            'Communication_Overhead_%': self.calculate_overhead_ratio('total'),
            'Setup_Overhead_%': self.calculate_overhead_ratio('setup'),
            'Operational_Control_Overhead_%': self.calculate_overhead_ratio('operational'),
            'Total_Protocol_Overhead_%': self.calculate_overhead_ratio('total'),
            'Mobile_Charger_Count': int(getattr(self, 'mobile_charger_count', 1)),
            'Mobile_Charger_Capacity_Mode': str(getattr(self, 'mobile_charger_capacity_mode', 'absolute')),
            'Sensor_Initial_Energy': float(getattr(self, 'sensor_initial_energy', 0.0)),
            'Mobile_Charger_Total_Capacity': float(getattr(self, 'mobile_charger_total_capacity', 0.0)),
            'Mobile_Charger_Min_Capacity': float(getattr(self, 'mobile_charger_min_capacity', 0.0)),
            'Mobile_Charger_Max_Capacity': float(getattr(self, 'mobile_charger_max_capacity', 0.0)),
            'Mobile_Charger_Total_Sensor_Equivalent_Capacity': float(getattr(self, 'mobile_charger_total_sensor_equivalent_capacity', 0.0)),
            'Mobile_Charger_Min_Sensor_Equivalent_Capacity': float(getattr(self, 'mobile_charger_min_sensor_equivalent_capacity', 0.0)),
            'Mobile_Charger_Max_Sensor_Equivalent_Capacity': float(getattr(self, 'mobile_charger_max_sensor_equivalent_capacity', 0.0)),
            'Mobile_Charger_Avg_Travel_Energy_Cost_Per_M': float(getattr(self, 'mobile_charger_avg_travel_energy_cost_per_m', 0.0)),
            'MC_Travel_Distance': self.mc_travel_distance,
            'MC_Travel_Energy': self.mc_travel_energy,
            'MC_RF_Transmit_Energy': self.mc_rf_transmit_energy,
            'MC_Contact_Charging_Energy': self.mc_contact_charging_energy,
            'MC_Total_Mission_Energy': (self.mc_travel_energy + self.mc_rf_transmit_energy + self.mc_contact_charging_energy),
            'RF_End_To_End_Efficiency': self._safe_rate(self.rf_total_delivered_energy, self.mc_rf_transmit_energy),
            'MC_End_To_End_Useful_Efficiency': self._safe_rate(self.mc_total_injected_energy, self.mc_travel_energy + self.mc_rf_transmit_energy + self.mc_contact_charging_energy),
            'MC_Charge_Events': self.mc_charge_events,
            'MC_Total_Injected_Energy': self.mc_total_injected_energy,
            'MC_Total_Charging_Time': self.mc_total_charging_time,
            'Avg_Residual_Energy_End': self.network_residual_energy_history[-1] if self.network_residual_energy_history else 0.0,
            'Fuzzy_Silhouette_Avg': np.mean(self.fuzzy_silhouette_scores) if self.fuzzy_silhouette_scores else 0.0,
            'Fuzzy_Partition_Coefficient_Avg': np.mean(self.fuzzy_partition_coefficients) if self.fuzzy_partition_coefficients else 0.0,
            'Fuzzy_Partition_Entropy_Avg': np.mean(self.fuzzy_partition_entropies) if self.fuzzy_partition_entropies else 0.0,
            'Xie_Beni_Avg': np.mean(self.xie_beni_indices) if self.xie_beni_indices else 0.0,
            'BS_Execution_Time_ms': computational['Avg_Core_Time_ms'],
            'Sensor_Execution_Time_ms': computational['Avg_Edge_Time_ms'],
            'MC_Fake_Reports_Attempted': self.mc_fake_reports_attempted,
            'MC_Fake_Reports_Accepted': self.mc_fake_reports_accepted,
            'MC_Rejected_Unauth_Reports': self.mc_rejected_unauth_reports,
            'MC_Location_Spoof_Attempts': self.mc_location_spoof_attempts,
            'MC_Location_Spoof_Accepted': self.mc_location_spoof_accepted,
            'MC_Location_Verification_Checks': self.mc_location_verification_checks,
            'MC_Location_Verification_Passes': self.mc_location_verification_passes,
            'MC_Location_Verification_Failures': self.mc_location_verification_failures,
            'MC_Actual_Location_Spoof_Detections': self.mc_location_spoof_detections,
            'MC_Raw_SOS_Dispatches': self.mc_raw_sos_dispatches,
            'MC_TPM_Compromise_Events': self.mc_tpm_compromise_events,
            'MC_TPM_Zeroized_Events': self.mc_tpm_zeroized_events,
            'Source_Sequence_Replay_Events': self.source_sequence_replay_events,
            'Packet_HMAC_Failures': self.packet_hmac_failures,
            'Implicit_MAC_ACK_Transactions': self.implicit_mac_ack_transactions,
            'Implicit_MAC_ACK_Energy_J': self.implicit_mac_ack_energy,
            'Implicit_MAC_ACK_Physically_Deducted_Energy_J': self.implicit_mac_ack_physically_deducted_energy,
            'Node_Charge_Receipts_Received': self.node_charge_receipts_received,
            'Node_Charge_Receipts_Consistent': self.node_charge_receipts_consistent,
            'Node_Charge_Receipts_Inconsistent': self.node_charge_receipts_inconsistent,
            'Node_Charge_Receipt_Auth_Failures': self.node_charge_receipt_auth_failures,
            'MC_Energy_Claim_Mismatches': self.mc_energy_claim_mismatches,
            'MC_Location_Report_Rejections': self.mc_location_report_rejections,
            'Energy_Trust_Recoveries': self.energy_trust_recoveries,
            'Adaptive_K_opt_Avg': float(np.mean(self.adaptive_k_opts)) if self.adaptive_k_opts else 0.0,
        }

        if self.data_routing_diagnostics_history:
            diagnostic_keys = sorted({
                key for row in self.data_routing_diagnostics_history
                for key in row
            })
            export_names = {
                'primary_replaced_count': 'Data_Routing_Primary_Replaced_Avg',
                'cluster_mst_primary_count': 'Data_Routing_Cluster_MST_Primary_Avg',
                'backbone_biased_primary_count': 'Data_Routing_Backbone_Biased_Primary_Avg',
                'energy_shortest_primary_count': 'Data_Routing_Energy_Shortest_Primary_Avg',
                'projected_primary_energy_ratio': 'Data_Routing_Projected_Energy_Ratio_Avg',
                'secondary_replaced_count': 'Data_Routing_Secondary_Replaced_Avg',
                'secondary_node_disjoint_count': 'Data_Routing_Secondary_Node_Disjoint_Avg',
                'secondary_edge_disjoint_count': 'Data_Routing_Secondary_Edge_Disjoint_Avg',
                'secondary_average_node_overlap': 'Data_Routing_Secondary_Node_Overlap_Avg',
                'secondary_standby_count': 'Data_Routing_Secondary_Standby_Avg',
            }
            for key in diagnostic_keys:
                values = [row[key]
                          for row in self.data_routing_diagnostics_history if key in row]
                if values:
                    name = export_names.get(key, f'Data_Routing_{key}_Avg')
                    summary[name] = float(np.mean(values))
        else:
            summary.update({
                'Data_Routing_Primary_Replaced_Avg': 0.0,
                'Data_Routing_Cluster_MST_Primary_Avg': 0.0,
                'Data_Routing_Projected_Energy_Ratio_Avg': 1.0,
                'Data_Routing_Secondary_Replaced_Avg': 0.0,
                'Data_Routing_Secondary_Node_Disjoint_Avg': 0.0,
                'Data_Routing_Secondary_Node_Overlap_Avg': 0.0,
            })

        response_times = list(self.mc_charging_response_times)
        summary['MC_Charging_Response_Time_Avg'] = float(
            np.mean(response_times)) if response_times else 0.0
        summary['MC_Charging_Response_Time_Max'] = float(
            np.max(response_times)) if response_times else 0.0
        summary['MC_Validated_Charging_Requests'] = self.mc_validated_charging_requests
        summary['MC_Rejected_Charging_Requests'] = self.mc_rejected_charging_requests
        summary['MC_Missed_Charging_Deadlines'] = self.mc_missed_charging_deadlines
        summary['MC_Avoided_Node_Deaths'] = self.mc_avoided_node_deaths
        summary['Energy_Critical_Node_Events'] = self.energy_critical_node_events
        summary['Energy_Critical_Node_Deaths'] = self.energy_critical_node_deaths

        # ADAPTIVE QIFL-DQN CHARGING EXPORTS
        summary['Adaptive_Charge_Threshold_Avg'] = float(np.mean(
            self.adaptive_charge_thresholds)) if self.adaptive_charge_thresholds else 0.0
        summary['Adaptive_Charge_Threshold_Max'] = float(np.max(
            self.adaptive_charge_thresholds)) if self.adaptive_charge_thresholds else 0.0
        summary['QIFL_Training_Loss_Avg'] = float(
            np.mean(self.qifl_training_losses)) if self.qifl_training_losses else 0.0
        summary['SOS_Threshold_Energy_Avg'] = float(
            np.mean(self.sos_threshold_energies)) if self.sos_threshold_energies else 0.0
        summary['Charging_KMeans_Clusters_Avg'] = float(np.mean(
            self.charging_kmeans_cluster_counts)) if self.charging_kmeans_cluster_counts else 0.0
        summary['Charging_Cluster_Assigned_Nodes_Avg'] = float(np.mean(
            self.charging_cluster_assigned_node_counts)) if self.charging_cluster_assigned_node_counts else 0.0
        summary['DQN_Routing_Decisions'] = int(self.dqn_routing_decisions)
        summary['DQN_Competitive_Decisions'] = int(
            self.dqn_competitive_decisions)
        summary['DQN_Single_Candidate_Decisions'] = int(
            self.dqn_single_candidate_decisions)
        summary['DQN_Effective_Exploration_Count'] = int(
            self.dqn_effective_exploration_count)
        summary['DQN_Safety_Override_Count'] = int(
            self.dqn_safety_override_count)
        for key, value in self.adaptive_hyperparameters.items():
            summary[f'AdaptiveHP_{key}'] = value
        summary['DQN_Training_Loss_Avg'] = float(
            np.mean(self.dqn_training_losses)) if self.dqn_training_losses else 0.0
        summary['DQN_Reward_Avg'] = float(
            np.mean(self.dqn_rewards)) if self.dqn_rewards else 0.0
        summary['DQN_Reward_Energy_Component_Avg'] = float(
            np.mean(self.dqn_reward_energy_components)) if self.dqn_reward_energy_components else 0.0
        summary['DQN_Reward_Deadline_Component_Avg'] = float(
            np.mean(self.dqn_reward_deadline_components)) if self.dqn_reward_deadline_components else 0.0
        summary['DQN_Reward_Travel_Component_Avg'] = float(
            np.mean(self.dqn_reward_travel_components)) if self.dqn_reward_travel_components else 0.0
        summary['DQN_Reward_Revisit_Component_Avg'] = float(
            np.mean(self.dqn_reward_revisit_components)) if self.dqn_reward_revisit_components else 0.0
        summary['DQN_Reward_Death_Component_Avg'] = float(
            np.mean(self.dqn_reward_death_components)) if self.dqn_reward_death_components else 0.0
        summary['MC_Service_Duration_Avg'] = float(
            np.mean(self.mc_service_durations)) if self.mc_service_durations else 0.0
        summary['MC_Fail_Closed_Rejections'] = int(
            self.mc_fail_closed_rejections)
        if self.dynamic_trust_weights_history:
            weight_matrix = np.asarray(
                self.dynamic_trust_weights_history, dtype=float)
            for index in range(min(5, weight_matrix.shape[1])):
                summary[f'Dynamic_Trust_Weight_T{index + 1}_Avg'] = float(
                    np.mean(weight_matrix[:, index]))
        else:
            for index in range(5):
                summary[f'Dynamic_Trust_Weight_T{index + 1}_Avg'] = 0.0
        summary['Adaptive_Compute_Device'] = str(self.adaptive_compute_device)
        summary['CUDA_Available_Flag'] = 1.0 if self.cuda_available else 0.0
        summary['DQN_Torch_Enabled_Flag'] = 1.0 if self.dqn_torch_enabled else 0.0
        summary['QIFL_Torch_Enabled_Flag'] = 1.0 if self.qifl_torch_enabled else 0.0
        summary['MC_Injected_Energy_Per_Charge_Event'] = self._safe_rate(
            self.mc_total_injected_energy, self.mc_charge_events)
        summary['MC_Travel_Per_Charged_Joule'] = self._safe_rate(
            self.mc_travel_distance, self.mc_total_injected_energy)
        summary['MC_Charge_Events_Per_Critical_Node'] = self._safe_rate(
            self.mc_charge_events, self.energy_critical_node_events,
            undefined=np.nan)
        resolved_deadlines = (
            self.mc_avoided_node_deaths + self.mc_missed_charging_deadlines
        )
        summary['MC_Resolved_Deadline_Events'] = int(resolved_deadlines)
        summary['MC_Critical_Event_Rate_Defined_Flag'] = int(
            self.energy_critical_node_events > 0)
        summary['MC_Deadline_Rate_Defined_Flag'] = int(
            resolved_deadlines > 0)
        summary['MC_Deadline_Failure_Rate'] = self._safe_rate(
            self.mc_missed_charging_deadlines, resolved_deadlines,
            undefined=np.nan)
        summary['Energy_Critical_Death_Rate'] = self._safe_rate(
            self.energy_critical_node_deaths, self.energy_critical_node_events,
            undefined=np.nan)
        summary['MC_Useless_Charge_Rate_Heuristic'] = self._safe_rate(
            self.mc_useless_charge_events, self.mc_charge_events)
        summary['MC_Repeated_Service_Events'] = int(
            self.mc_repeated_service_events)
        summary['MC_Total_Revisit_Events'] = int(self.mc_total_revisit_events)
        summary['MC_Revisit_Interval_Avg'] = float(
            np.mean(self.mc_revisit_intervals)) if self.mc_revisit_intervals else 0.0
        summary['EWMA_Energy_Prediction_MAE'] = float(
            np.mean(self.ewma_energy_abs_errors)) if self.ewma_energy_abs_errors else 0.0
        applied_min_ratios = (
            self.mc_applied_minimum_charge_ratios
            if self.mc_applied_minimum_charge_ratios
            else self.adaptive_minimum_charge_ratios
        )
        summary['MC_Useless_Charge_Min_Useful_Ratio'] = float(
            np.mean(applied_min_ratios)) if applied_min_ratios else 0.0
        summary['Adaptive_Minimum_Charge_Ratio_Avg'] = float(
            np.mean(self.adaptive_minimum_charge_ratios)
        ) if self.adaptive_minimum_charge_ratios else 0.0
        summary['QIFL_Admission_Count'] = int(self.qifl_admission_count)
        summary['QIFL_MinGap_Conflict_Rejections'] = int(
            self.qifl_min_gap_conflict_rejections
        )
        summary['QIFL_MinGap_Conflict_Preserved'] = int(
            self.qifl_min_gap_conflict_preserved
        )
        summary['Verified_Emergency_Security_Override_Count'] = int(
            self.verified_emergency_security_override_count
        )
        summary['QIFL_Learning_Active_Flag'] = 1.0 if summary['QIFL_Training_Loss_Avg'] > 0 else 0.0
        summary['DQN_Learning_Active_Flag'] = 1.0 if (
            summary['DQN_Routing_Decisions'] > 0 and summary['DQN_Training_Loss_Avg'] > 0) else 0.0

        # INNOVATION AUDIT EXPORTS
        charge_tp = self.charging_decision_tp
        charge_fp = self.charging_decision_fp
        charge_tn = self.charging_decision_tn
        charge_fn = self.charging_decision_fn
        charge_total = charge_tp + charge_fp + charge_tn + charge_fn
        summary['Charging_Decision_Accuracy'] = self._safe_rate(
            charge_tp + charge_tn, charge_total)
        summary['False_Charging_Admissions'] = charge_fp
        summary['Charging_Request_Precision'] = self._safe_rate(
            charge_tp, charge_tp + charge_fp)
        summary['Charging_Request_Recall'] = self._safe_rate(
            charge_tp, charge_tp + charge_fn)
        summary['Silent_Node_SOS_Count'] = self.silent_node_sos_count
        summary['Silent_Node_Rescue_Rate'] = self._safe_rate(
            self.silent_node_rescue_count, self.silent_node_sos_count)
        summary['False_SOS_Rate'] = self._safe_rate(
            self.false_sos_count, self.silent_node_sos_count)
        summary['Avg_Time_To_Critical_Before_MC_Arrival'] = float(np.mean(
            self.time_to_critical_before_mc_arrival)) if self.time_to_critical_before_mc_arrival else 0.0
        summary['MC_Deadline_Success_Rate'] = self._safe_rate(
            self.mc_avoided_node_deaths, resolved_deadlines,
            undefined=np.nan)

        summary['Forged_MC_Report_Rejection_Rate'] = self._safe_rate(
            self.mc_rejected_unauth_reports, self.mc_fake_reports_attempted)
        summary['MC_Replay_Rejection_Rate'] = self._safe_rate(
            self.mc_replay_rejections, self.mc_replay_rejections + max(0, self.mc_fake_reports_accepted))
        summary['Fake_Charger_Detection_Rate'] = self._safe_rate(
            self.fake_charger_detections, self.fake_charger_attempts)
        summary['Charge_Hoarder_Detection_Rate'] = self._safe_rate(
            self.charge_hoarder_detections, self.charge_hoarder_attempts)
        summary['Location_Spoof_Rejection_Rate'] = self._safe_rate(
            self.mc_location_spoof_detections, self.mc_location_spoof_attempts)
        summary['Raw_SOS_Blocking_Rate'] = self._safe_rate(
            self.raw_sos_blocked, self.raw_sos_validation_attempts)

        summary['Delay_Attribution_Accuracy'] = self._safe_rate(
            self.delay_attribution_correct, self.delay_attribution_trials)
        summary['False_Delay_Accusation_Rate'] = self._safe_rate(
            self.false_delay_accusations, self.delay_attribution_trials)
        summary['Tomography_Observability_Ratio'] = self._safe_rate(
            self.tomography_observable_windows, self.tomography_windows)
        summary['Tomography_Rank_Deficient_Windows'] = self.tomography_rank_deficient_windows
        summary['Tomography_Mean_Residual_Error'] = float(np.mean(
            self.tomography_residual_errors)) if self.tomography_residual_errors else 0.0

        summary['Shadow_Precompute_Count'] = self.shadow_precompute_count
        summary['Shadow_Route_Hit_Count'] = self.shadow_route_hit_count
        summary['Shadow_Route_Hit_Rate'] = self._safe_rate(
            self.shadow_route_hit_count, self.shadow_precompute_count)
        summary['Ordinary_Route_Repair_Time_ms'] = float(np.mean(
            self.ordinary_route_repair_times_ms)) if self.ordinary_route_repair_times_ms else 0.0
        summary['Shadow_Route_Repair_Time_ms'] = float(np.mean(
            self.shadow_route_repair_times_ms)) if self.shadow_route_repair_times_ms else 0.0
        ordinary_mean = summary['Ordinary_Route_Repair_Time_ms']
        shadow_mean = summary['Shadow_Route_Repair_Time_ms']
        summary['Route_Repair_Speedup_Ratio'] = (
            ordinary_mean / shadow_mean) if shadow_mean > 0 else 0.0
        summary['Extra_BS_Shadow_Compute_ms'] = float(np.mean(
            self.shadow_extra_compute_times_ms)) if self.shadow_extra_compute_times_ms else 0.0

        summary['Critical_Secondary_Path_Availability_Rate'] = self._safe_rate(
            self.critical_secondary_path_available, self.critical_secondary_path_opportunities)
        summary['Critical_Redundant_Copy_Accepted_Count'] = self.critical_redundant_copy_accepted
        summary['Critical_Redundant_Copy_Rejected_Count'] = self.critical_redundant_copy_rejected
        summary['Sparse_Topology_NoBackup_Count'] = self.sparse_topology_no_backup_count

        summary['RF_Cluster_Charge_Events'] = int(
            getattr(self, 'rf_cluster_charge_events', 0))
        summary['RF_Total_Nodes_Charged_In_Clusters'] = int(
            getattr(self, 'rf_total_nodes_charged_in_clusters', 0))
        summary['RF_Avg_Nodes_Per_Cluster'] = self._safe_rate(
            getattr(self, 'rf_total_nodes_charged_in_clusters', 0),
            getattr(self, 'rf_cluster_charge_events', 0)
        )
        summary['Dynamic_Partial_Charge_Events'] = int(
            getattr(self, 'dynamic_partial_charge_events', 0))
        summary['Fixed_Charger_Charge_Events'] = int(
            getattr(self, 'fixed_charger_charge_events', 0))
        summary['Fixed_Charger_Total_Injected_Energy'] = float(
            getattr(self, 'fixed_charger_total_injected_energy', 0.0))
        summary['Fixed_Charger_Active_Nodes'] = int(
            getattr(self, 'fixed_charger_active_nodes', 0))
        summary['MC_Infeasible_Mission_Rejections'] = int(
            getattr(self, 'mc_infeasible_mission_rejections', 0))
        summary['MC_Stranded_Events'] = int(
            getattr(self, 'mc_stranded_events', 0))
        summary['DQN_Travel_Penalty_Magnitude_Avg'] = summary.get(
            'DQN_Reward_Travel_Component_Avg', 0.0)
        summary['DQN_Revisit_Penalty_Magnitude_Avg'] = summary.get(
            'DQN_Reward_Revisit_Component_Avg', 0.0)
        summary['MC_Async_Missions_Started'] = int(
            getattr(self, 'mc_async_missions_started', 0))
        summary['MC_Async_Missions_Completed'] = int(
            getattr(self, 'mc_async_missions_completed', 0))
        summary['MC_Async_Missions_Incomplete'] = int(
            getattr(self, 'mc_async_missions_incomplete', 0))
        summary['MC_Network_Events_While_Busy'] = int(
            getattr(self, 'mc_network_events_while_busy', 0))
        summary['Charging_Service_Pool_Admissions'] = int(getattr(
            self, 'charging_service_pool_admissions', 0))
        summary['Charging_Plus_Pool_Admissions'] = int(getattr(
            self, 'charging_plus_pool_admissions', 0))
        summary['Charging_Opportunistic_Plus_Charges'] = int(getattr(
            self, 'charging_opportunistic_plus_charges', 0))

        for key, value in sorted(self.experiment_parameters.items()):
            safe_key = re.sub(r'[^0-9A-Za-z_]+', '_', str(key)).strip('_')
            column = f'Config_{safe_key}'
            if isinstance(value, (bool, np.bool_)):
                summary[column] = int(value)
            elif isinstance(value, (int, float, np.integer, np.floating)):
                summary[column] = float(value)
            elif value is None:
                summary[column] = 'None'
            elif isinstance(value, (list, tuple, dict, set)):
                summary[column] = json.dumps(
                    value, sort_keys=True, default=str)
            else:
                summary[column] = str(value)

        return summary

    def save_history_to_csv(self, filename="simulation_history.csv"):
        df = pd.DataFrame({
            "Time_Step": range(len(self.network_residual_energy_history)),
            "Avg_Residual_Energy": self.network_residual_energy_history
        })
        df.to_csv(filename, index=False)
        print(f"[METRICS] Time-series history saved to {filename}")
