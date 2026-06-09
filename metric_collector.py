import numpy as np
import pandas as pd
import math
import os


class MetricsCollector:
    """
    Global Metrics Collector for WSN-WRSN Simulation.
    This class observes and records all 15 critical evaluation parameters 
    across Security (IDS), Energy/WRSN, QoS, Fuzzy Logic, and Computational Overhead.
    It is designed to be fully decoupled to support Ablation Studies and 30-Seed statistical testing.
    """

    def __init__(self, total_nodes, total_rounds_or_time):
        # 1. Network Configuration Properties
        self.total_nodes = total_nodes
        self.total_rounds_or_time = total_rounds_or_time

        # 2. IDS / Security Metrics (Confusion Matrix Tracking)
        self.tp = 0  # True Positive
        self.tn = 0  # True Negative
        self.fp = 0  # False Positive (Critical to minimize)
        self.fn = 0  # False Negative
        self.detection_delays = []  # Time taken to detect malicious nodes

        # 3. QoS & Routing Metrics
        self.generated_data_packets = 0
        self.received_data_packets = 0
        self.generated_critical_packets = 0
        self.received_critical_packets = 0
        self.total_bits_received = 0
        self.end_to_end_delays = []  # List of delays (T_arrive - T_generate)

        # 4. Energy & WRSN Metrics
        # SOS, initial setup, keep-alive (Piggybacking reduces this)
        self.control_packets = 0
        self.alive_nodes = total_nodes
        self.node_death_times = {}  # Tracks round/time of death for FND, HND, LND
        self.network_residual_energy_history = []  # Average energy per round
        self.mc_travel_distance = 0.0  # Mobile Charger total distance

        # 5. Fuzzy Clustering (FCM) Metrics
        self.fuzzy_objective_values = []  # Track convergence per iteration
        self.fuzzy_silhouette_scores = []  # Track clustering separation quality

        # 6. Computational Overhead
        # Sensor-level processing (e.g., threshold check)
        self.edge_processing_times = []
        # Base Station processing (Fuzzy, Routing)
        self.bs_processing_times = []

    # =====================================================================
    # UPDATE METHODS (Called by Simulation Environment during execution)
    # =====================================================================

    def update_packet_transmission(self, is_critical, is_control, size_bytes, is_delivered=False, delay=None):
        """
        Records packet generation and delivery for PDR, Throughput, and Overhead.
        """
        if is_control:
            self.control_packets += 1
        else:
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
        """
        Updates the confusion matrix for security evaluation.
        """
        self.tp += tp_increment
        self.tn += tn_increment
        self.fp += fp_increment
        self.fn += fn_increment

    def record_node_death(self, node_id, current_time):
        """
        Records when a node's energy drops to 0 to calculate FND, HND, LND.
        """
        if node_id not in self.node_death_times:
            self.node_death_times[node_id] = current_time
            self.alive_nodes -= 1

    def record_energy_state(self, current_avg_energy):
        """
        Records the average residual energy of the network at a specific time step.
        """
        self.network_residual_energy_history.append(current_avg_energy)

    def record_fuzzy_results(self, objective_history, silhouette_score):
        """
        Stores the performance of the Fuzzy C-Means algorithm in the Base Station.
        """
        self.fuzzy_objective_values.append(objective_history)
        self.fuzzy_silhouette_scores.append(silhouette_score)

    def add_mc_travel_distance(self, distance):
        """
        Accumulates the travel distance of the Mobile Charger.
        """
        self.mc_travel_distance += distance

    def record_execution_time(self, component, exec_time_ms):
        """
        component: 'EDGE' (Sensor) or 'CORE' (Base Station)
        """
        if component == 'EDGE':
            self.edge_processing_times.append(exec_time_ms)
        elif component == 'CORE':
            self.bs_processing_times.append(exec_time_ms)

    # =====================================================================
    # CALCULATION METHODS (Formulas implemented for Final Extraction)
    # =====================================================================

    def calculate_ids_metrics(self):
        """
        Returns TPR, FPR, Accuracy, and F1-Score. Handles ZeroDivisionError safely.
        """
        total_positives = self.tp + self.fn
        total_negatives = self.fp + self.tn

        tpr = (self.tp / total_positives) if total_positives > 0 else 0.0
        fpr = (self.fp / total_negatives) if total_negatives > 0 else 0.0

        total_predictions = total_positives + total_negatives
        accuracy = ((self.tp + self.tn) /
                    total_predictions) if total_predictions > 0 else 0.0

        precision = (self.tp / (self.tp + self.fp)
                     ) if (self.tp + self.fp) > 0 else 0.0
        f1_score = (2 * precision * tpr / (precision + tpr)
                    ) if (precision + tpr) > 0 else 0.0

        return {"TPR": tpr, "FPR": fpr, "Accuracy": accuracy, "F1_Score": f1_score}

    def calculate_qos_metrics(self, total_simulation_time):
        """
        Returns overall PDR, Critical PDR, Throughput (bps), and Avg Delay.
        """
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
        """
        Extracts FND (First Node Dies), HND (Half Nodes Die), LND (Last Node Dies).
        """
        death_times = list(self.node_death_times.values())
        death_times.sort()

        fnd = death_times[0] if len(death_times) > 0 else -1
        hnd = death_times[len(
            death_times)//2] if len(death_times) >= (self.total_nodes // 2) else -1
        lnd = death_times[-1] if len(death_times) == self.total_nodes else -1

        return {"FND": fnd, "HND": hnd, "LND": lnd}

    def calculate_overhead_ratio(self):
        """
        Calculates Communication Overhead. Lower is better.
        Piggybacking makes this near 0 in normal operation.
        """
        total_packets = self.generated_data_packets + self.control_packets
        overhead_ratio = (self.control_packets / total_packets) * \
            100 if total_packets > 0 else 0.0
        return overhead_ratio

    def calculate_computational_overhead(self):
        """
        Returns average execution times for Edge (Sensor) and Core (BS).
        """
        avg_edge = np.mean(
            self.edge_processing_times) if self.edge_processing_times else 0.0
        avg_core = np.mean(
            self.bs_processing_times) if self.bs_processing_times else 0.0
        return {"Avg_Edge_Time_ms": avg_edge, "Avg_Core_Time_ms": avg_core}

    # =====================================================================
    # EXPORT RESULTS (For Ablation & Statistical 30-Seed Tests)
    # =====================================================================

    def export_summary(self, seed_num, ablation_scenario_name, sim_time):
        """
        Compiles all metrics into a single dictionary for the current simulation run.
        Useful for appending to a Pandas DataFrame for statistical testing.
        """
        ids = self.calculate_ids_metrics()
        qos = self.calculate_qos_metrics(sim_time)
        lifetime = self.calculate_network_lifetime()
        computational = self.calculate_computational_overhead()

        summary = {
            "Seed": seed_num,
            "Ablation_Scenario": ablation_scenario_name,

            # Security
            "TPR": ids["TPR"],
            "FPR": ids["FPR"],
            "F1_Score": ids["F1_Score"],

            # QoS
            "Overall_PDR": qos["Overall_PDR"],
            "Critical_PDR": qos["Critical_PDR"],
            "Throughput": qos["Throughput_bps"],
            "Avg_Delay": qos["Avg_Delay"],

            # WRSN / Energy
            "FND": lifetime["FND"],
            "Communication_Overhead_%": self.calculate_overhead_ratio(),
            "MC_Travel_Distance": self.mc_travel_distance,
            "Avg_Residual_Energy_End": self.network_residual_energy_history[-1] if self.network_residual_energy_history else 0.0,

            # Computational & Fuzzy
            "Fuzzy_Silhouette_Avg": np.mean(self.fuzzy_silhouette_scores) if self.fuzzy_silhouette_scores else 0.0,
            "BS_Execution_Time_ms": computational["Avg_Core_Time_ms"],
            "Sensor_Execution_Time_ms": computational["Avg_Edge_Time_ms"]
        }

        return summary

    def save_history_to_csv(self, filename="simulation_history.csv"):
        """
        Saves time-series data (like residual energy over time) for plotting continuous graphs.
        """
        df = pd.DataFrame({
            "Time_Step": range(len(self.network_residual_energy_history)),
            "Avg_Residual_Energy": self.network_residual_energy_history
        })
        df.to_csv(filename, index=False)
        print(f"[METRICS] Time-series history saved to {filename}")
