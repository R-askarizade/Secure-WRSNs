import math
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
        self.network_residual_energy_history = []
        self.mc_travel_distance = 0.0

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

        self.mc_fake_reports_attempted = 0
        self.mc_fake_reports_accepted = 0
        self.mc_location_spoof_attempts = 0
        self.mc_location_spoof_accepted = 0
        self.mc_raw_sos_dispatches = 0
        self.mc_rejected_unauth_reports = 0
        self.mc_tpm_compromise_events = 0
        self.mc_tpm_zeroized_events = 0

        self.source_sequence_replay_events = 0
        self.packet_hmac_failures = 0

        # INNOVATION AUDIT METRICS  
        self.charging_decision_tp = 0
        self.charging_decision_fp = 0
        self.charging_decision_tn = 0
        self.charging_decision_fn = 0
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

    def update_packet_transmission(self, is_critical, is_control, size_bytes, is_delivered=False, delay=None, overhead_phase='operational', count_generated=True):
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
                self.total_bits_received += (size_bytes * 8)
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
        already_dead = node_id in self.node_death_times
        if not already_dead:
            self.node_death_times[node_id] = current_time
            self.alive_nodes -= 1
        if not already_dead and node_id in self._critical_nodes_seen:
            self.energy_critical_node_deaths += 1
            self.mc_missed_charging_deadlines += 1
            self._critical_nodes_seen.discard(node_id)

    def record_energy_state(self, current_avg_energy):
        self.network_residual_energy_history.append(current_avg_energy)

    def add_mc_travel_distance(self, distance):
        self.mc_travel_distance += distance

    def record_execution_time(self, component, exec_time_ms):
        if component == 'EDGE':
            self.edge_processing_times.append(exec_time_ms)
        elif component == 'CORE':
            self.bs_processing_times.append(exec_time_ms)

    def record_charging_request(self, node_id, request_time, validated=True):
        if validated:
            self.mc_validated_charging_requests += 1
            self._charging_request_first_seen_time.setdefault(
                node_id, float(request_time))
        else:
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
        if start_time is not None:
            self.mc_charging_response_times.append(
                max(0.0, float(current_time) - float(start_time)))
        if node_id in self._critical_nodes_seen:
            self.mc_avoided_node_deaths += 1
            self._critical_nodes_seen.discard(node_id)

    # INNOVATION AUDIT RECORDING METHODS
    def record_charging_decision(self, needed, admitted):
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
    def _safe_rate(num, den, scale=1.0):
        den = float(den)
        return (float(num) / den) * scale if den > 0 else 0.0

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
        death_times = list(self.node_death_times.values())
        death_times.sort()
        fnd = death_times[0] if len(death_times) > 0 else -1
        hnd = death_times[len(
            death_times)//2] if len(death_times) >= (self.total_nodes // 2) else -1
        lnd = death_times[-1] if len(death_times) == self.total_nodes else -1
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
            'MC_Travel_Distance': self.mc_travel_distance,
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
            'MC_Raw_SOS_Dispatches': self.mc_raw_sos_dispatches,
            'MC_TPM_Compromise_Events': self.mc_tpm_compromise_events,
            'MC_TPM_Zeroized_Events': self.mc_tpm_zeroized_events,
            'Source_Sequence_Replay_Events': self.source_sequence_replay_events,
            'Packet_HMAC_Failures': self.packet_hmac_failures,
        }

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
            self.mc_avoided_node_deaths, self.mc_avoided_node_deaths + self.mc_missed_charging_deadlines)

        summary['Forged_MC_Report_Rejection_Rate'] = self._safe_rate(
            self.mc_rejected_unauth_reports, self.mc_fake_reports_attempted)
        summary['MC_Replay_Rejection_Rate'] = self._safe_rate(
            self.mc_replay_rejections, self.mc_replay_rejections + max(0, self.mc_fake_reports_accepted))
        summary['Fake_Charger_Detection_Rate'] = self._safe_rate(
            self.fake_charger_detections, self.fake_charger_attempts)
        summary['Charge_Hoarder_Detection_Rate'] = self._safe_rate(
            self.charge_hoarder_detections, self.charge_hoarder_attempts)
        summary['Location_Spoof_Rejection_Rate'] = self._safe_rate(
            self.mc_location_spoof_attempts - self.mc_location_spoof_accepted, self.mc_location_spoof_attempts)
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

        return summary

    def save_history_to_csv(self, filename="simulation_history.csv"):
        df = pd.DataFrame({
            "Time_Step": range(len(self.network_residual_energy_history)),
            "Avg_Residual_Energy": self.network_residual_energy_history
        })
        df.to_csv(filename, index=False)
        print(f"[METRICS] Time-series history saved to {filename}")
