import random
import time
import pandas as pd
import numpy as np
import scipy.stats as stats
from collections import defaultdict


from metric_collector import MetricsCollector
from sensor_node import SensorNode
from base_station import BaseStation
from mobile_charger import MobileCharger
from routing import BaselineRouting


class SimulatorManager:
    """
    Main Execution Manager for the WSN-WRSN SDN-based Architecture.
    Handles physical topology deployment, event generation, packet tracing,
    attack injection (Stealthy, Spoofing), and the 30-seed statistical loop.
    """

    def __init__(self, num_nodes, sim_time, ablation_config, seed, network_size=(100, 100)):
        self.num_nodes = num_nodes
        self.sim_time = sim_time  # Total simulation time in seconds
        self.ablation = ablation_config
        self.seed = seed
        self.network_size = network_size  # 100m x 100m area
        self.BS_ID = 0  # Base Station is ID 0

        # Set Random Seed for Reproducibility (Crucial for 30-seed statistical tests)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Instantiate Global Metrics Collector
        self.metrics = MetricsCollector(
            total_nodes=num_nodes, total_rounds_or_time=sim_time)

        # Instantiate Base Station
        self.bs = BaseStation(metrics_collector=self.metrics,
                              ablation_config=self.ablation)

        # Instantiate Mobile Charger (Depot at BS location)
        bs_x, bs_y = network_size[0] / 2, network_size[1] / 2  # BS at center
        self.mc = MobileCharger(mc_id=999, depot_x=bs_x, depot_y=bs_y, capacity=50000.0,
                                speed=5.0, base_station=self.bs, metrics_collector=self.metrics)

        # Instantiate Sensor Nodes
        self.nodes = {}
        self._deploy_network(bs_x, bs_y)

        # Routing Module
        self.routing_engine = BaselineRouting(
            bs_id=self.BS_ID, transmission_range=30.0)

        # Attack configuration
        self.malicious_nodes = []  # List of IDs that are attackers

        # Simulation Clock
        self.current_time = 0.0

    # =====================================================================
    # 1. NETWORK DEPLOYMENT & BOOTSTRAPPING
    # =====================================================================

    def _deploy_network(self, bs_x, bs_y):
        """Randomly deploys sensor nodes and places the Base Station."""
        # Add Base Station as a special node (Infinite energy)
        self.nodes[self.BS_ID] = SensorNode(
            self.BS_ID, bs_x, bs_y, float('inf'), self.metrics)

        # Add Sensors
        initial_energy = 0.5  # 0.5 Joules
        for i in range(1, self.num_nodes + 1):
            x = random.uniform(0, self.network_size[0])
            y = random.uniform(0, self.network_size[1])
            self.nodes[i] = SensorNode(i, x, y, initial_energy, self.metrics)

    def bootstrap_network(self):
        """Executes Phase 1: Gradient Discovery & Route Provisioning"""
        print(f"[Seed {self.seed}] Bootstrapping network...")

        # 1. BS discovers nodes and registers them
        node_list = list(self.nodes.values())
        self.bs.phase_1_gradient_discovery_and_registration(node_list)

        # 2. Compute Initial Routes
        initial_tables = self.routing_engine.compute_routing_tables(self.nodes)
        self.bs.routing_tables = initial_tables

        # 3. Push policies and routes to sensors (Downstream Policy Provisioning)
        adaptive_threshold = 75.0  # Example temperature threshold
        for n_id, table in initial_tables.items():
            self.nodes[n_id].update_policies_from_bs(
                primary_hop=table['primary'],
                secondary_hop=table['secondary'],
                new_threshold=adaptive_threshold
            )

    def inject_attacks(self, attack_percentage=0.1):
        """
        Injects specific WRSN cyber-physical attacks based on ablation config.
        Selects a random subset of nodes to act maliciously.
        """
        num_attackers = int(self.num_nodes * attack_percentage)
        available_nodes = [n for n in self.nodes.keys() if n != self.BS_ID]
        self.malicious_nodes = random.sample(available_nodes, num_attackers)

        # Flag the nodes internally for simulation behavior
        for n_id in self.malicious_nodes:
            self.nodes[n_id].is_malicious = True

        print(f"[Seed {self.seed}] Injected {num_attackers} malicious nodes.")

    # =====================================================================
    # 2. SIMULATION TIME LOOP (The Engine)
    # =====================================================================

    def run_simulation(self):
        """Main time-stepped simulation loop."""
        self.bootstrap_network()
        self.inject_attacks(attack_percentage=0.15)  # 15% compromised nodes

        packet_seq_num = 0

        while self.current_time < self.sim_time:
            # Check if network is dead (LND)
            if self.metrics.alive_nodes <= 0:
                print(
                    f"[Seed {self.seed}] Network died at t={self.current_time}")
                break

            # 1. ENVIRONMENT EVENT GENERATION (Sensing)
            # Randomly pick a node to generate data
            active_node_id = random.choice(
                [n for n, obj in self.nodes.items() if obj.status == "ALIVE" and n != self.BS_ID])
            active_node = self.nodes[active_node_id]

            # Generate random data (10% chance of critical data spike)
            is_critical_event = random.random() < 0.10
            # 80 triggers threshold (75)
            raw_data = 80.0 if is_critical_event else 25.0

            packet_seq_num += 1

            # Node processes edge logic and transmits
            active_node.sense_environment(raw_data, self.current_time)

            # 2. PACKET ROUTING & TRANSMISSION SIMULATION
            # In a real network, this happens concurrently. Here we trace the packet hop-by-hop.
            self._trace_packet_path(
                active_node_id, packet_seq_num, is_critical_event)

            # 3. WRSN KEEP-ALIVE & SOS CHARGE CHECK
            for n_id, node in self.nodes.items():
                if n_id != self.BS_ID:
                    node.check_inactivity(self.current_time)
                    # If SOS sent, we would dispatch MC here (simplified for space)

            # 4. BASE STATION CONTROL PLANE (Asynchronous Epoch Processing)
            # Runs every T_epoch (e.g., 10 seconds)
            if self.current_time > 0 and self.current_time % self.bs.T_epoch == 0:
                self.bs.control_plane_epoch()
                # If BS calculated new routes, push them
                self._push_updated_routes()

            # Advance time
            self.current_time += 1.0  # 1 second steps

            # Record network energy state
            avg_energy = sum([n.E_current for n in self.nodes.values(
            ) if n.node_id != self.BS_ID]) / self.num_nodes
            self.metrics.record_energy_state(avg_energy)

    # =====================================================================
    # 3. PACKET TRACING & ATTACK BEHAVIOR
    # =====================================================================

    def _trace_packet_path(self, source_id, seq_num, is_critical):
        """
        Traces the physical path of the packet. Applies Malicious node behaviors.
        """
        paths_to_trace = []

        # Strategy B (Multipath for Critical)
        if is_critical and self.ablation["use_critical_multipath_routing"]:
            paths_to_trace.append(
                {'hop': self.nodes[source_id].primary_next_hop, 'type': 'PRIMARY', 'path': [source_id]})
            if self.nodes[source_id].secondary_next_hop:
                paths_to_trace.append(
                    {'hop': self.nodes[source_id].secondary_next_hop, 'type': 'SECONDARY', 'path': [source_id]})
        # Strategy A (Single path for Normal)
        else:
            paths_to_trace.append(
                {'hop': self.nodes[source_id].primary_next_hop, 'type': 'PRIMARY', 'path': [source_id]})

        for p_info in paths_to_trace:
            current_hop_id = p_info['hop']
            path_log = p_info['path']
            packet_dropped = False

            while current_hop_id is not None and current_hop_id != self.BS_ID:
                hop_node = self.nodes[current_hop_id]
                path_log.append(current_hop_id)

                # --- ATTACKER BEHAVIOR: STEALTHY DROPPER ---
                if hasattr(hop_node, 'is_malicious') and hop_node.is_malicious:
                    # Attacker drops the packet
                    packet_dropped = True
                    # Stealthy Alibi: Lie to BS about having no battery to justify the drop
                    hop_node.E_current += 0  # Doesn't consume tx energy
                    # Piggyback fake battery (e.g., claiming 2 Joules left) to fool BS
                    fake_packet_for_telemetry = {"piggybacked_energy": 2.0}
                    break

                # Healthy node forwards the packet
                success = hop_node.forward_packet(
                    {"size_bytes": 128}, self.current_time)
                if not success:
                    packet_dropped = True
                    break

                current_hop_id = hop_node.primary_next_hop

            # Reached Base Station
            if not packet_dropped and current_hop_id == self.BS_ID:
                path_log.append(self.BS_ID)
                # Delivery delay calculation (Simulated hop delay)
                simulated_delay = len(path_log) * 0.100  # 100ms per hop

                # Tell Metrics packet arrived
                self.metrics.update_packet_transmission(
                    is_critical=is_critical, is_control=False,
                    size_bytes=128, is_delivered=True, delay=simulated_delay
                )

                # Tell BS Data Plane (Logging for IDS)
                packet_data = {"source_id": source_id, "seq": seq_num, "timestamp": self.current_time,
                               "piggybacked_energy": self.nodes[source_id].E_current}
                self.bs.data_plane_receive(
                    packet_data, path_log, receive_time=self.current_time + simulated_delay)

    def _push_updated_routes(self):
        """Pushes any new routes calculated by BS during its Epoch."""
        for n_id, table in self.bs.routing_tables.items():
            if n_id in self.nodes:
                self.nodes[n_id].primary_next_hop = table.get('primary')
                self.nodes[n_id].secondary_next_hop = table.get('secondary')

# =====================================================================
# 4. STATISTICAL RUNNER (30 Seeds & Ablation Testing)
# =====================================================================


def run_statistical_experiments():
    """
    Runs the simulation over 30 different random seeds to ensure statistical significance.
    This is the main entry point required for Q1 journal submissions.
    """
    NUM_SEEDS = 30
    NUM_NODES = 100
    SIM_TIME = 3600  # 1 hour simulation

    # Example: Running the PROPOSED architecture (All features ON)
    proposed_config = {
        "use_gradient_symmetric_discovery": True,
        "use_adaptive_edge_threshold": True,
        "use_critical_multipath_routing": True,  # Crucial for QoS
        "use_implicit_piggyback_telemetry": True,  # Crucial for Overhead
        "use_network_tomography_for_delay": True,
        "use_intersection_logic_for_droppers": True,
        "use_asynchronous_processing_pipeline": True,
        "use_shannon_entropy_weights": True,
        "use_security_veto_aggregation": True,
        "use_stealthy_attacker_detection": True,  # WRSN Security
        "use_zero_overhead_spoof_detection": True
    }

    all_results = []

    for seed in range(1, NUM_SEEDS + 1):
        print(f"\n--- Starting Seed {seed}/{NUM_SEEDS} ---")

        sim = SimulatorManager(num_nodes=NUM_NODES, sim_time=SIM_TIME,
                               ablation_config=proposed_config, seed=seed)
        sim.run_simulation()

        # Extract results using the decoupled MetricsCollector
        summary = sim.metrics.export_summary(
            seed_num=seed, ablation_scenario_name="Proposed_WRSN", sim_time=SIM_TIME)
        all_results.append(summary)

    # Export to Excel/CSV for plotting Error Bars and T-tests in MATLAB/Python
    df_results = pd.DataFrame(all_results)
    df_results.to_csv("proposed_model_results_30seeds.csv", index=False)
    print("\n[SUCCESS] Statistical experiments completed. Results saved to CSV.")
    print("Average Overall PDR:", df_results["Overall_PDR"].mean())
    print("Average Critical PDR:", df_results["Critical_PDR"].mean())
    print("Average FPR (False Positive Rate):", df_results["FPR"].mean())

    metrics_to_analyze = ["Overall_PDR", "Critical_PDR",
                          "TPR", "FPR", "F1_Score", "Throughput"]

    for metric in metrics_to_analyze:
        data = df_results[metric]
        mean_val = np.mean(data)
        std_err = stats.sem(data)  # Standard Error of the Mean

        # Calculate the 95% Confidence Interval
        # This means we are 95% confident that the model's performance in the real world is within this interval
        ci = stats.t.interval(0.95, len(data)-1, loc=mean_val, scale=std_err)

        print(f"{metric}:")
        print(f"   Mean: {mean_val:.4f}")
        print(f"   95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        print("-" * 30)

        # Example for P-Value (T-test):
        # t_stat, p_value = stats.ttest_ind(df_results["Overall_PDR"], baseline_results["Overall_PDR"])
        # print(f"P-Value (Proposed vs Baseline): {p_value}")
        # If p_value < 0.05, your method is statistically better by a certain margin!


if __name__ == "__main__":
    run_statistical_experiments()
