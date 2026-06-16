import math
import numpy as np
import pandas as pd

class MetricsCollector:
    def __init__(self, total_nodes, total_rounds_or_time):
        self.total_nodes = total_nodes
        self.total_rounds_or_time = total_rounds_or_time

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

        return summary

    def save_history_to_csv(self, filename="simulation_history.csv"):
        df = pd.DataFrame({
            "Time_Step": range(len(self.network_residual_energy_history)),
            "Avg_Residual_Energy": self.network_residual_energy_history
        })
        df.to_csv(filename, index=False)
        print(f"[METRICS] Time-series history saved to {filename}")
