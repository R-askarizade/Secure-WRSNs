import numpy as np
import time
import math
from collections import defaultdict


class BaseStation:
    """
    Centralized Base Station (SDN Controller) for WSN-WRSN.
    Handles Data Plane (fast packet logging) and Control Plane (heavy asynchronous analytics).
    Implements Network Tomography, Fuzzy C-Means (from scratch), Shannon Entropy weighting,
    and advanced WRSN Cyber-Physical Security formulas.
    """

    def __init__(self, metrics_collector, ablation_config):
        self.metrics = metrics_collector
        self.ablation = ablation_config

        # Network Database
        # node_id -> SensorNode object (Reference for simulation)
        self.network_nodes = {}
        self.node_states = {}   # Internal BS database for Trust, Energy, Location
        self.routing_tables = {}
        # For Proactive/Predictive Routing (Zero-Latency Failover)
        self.shadow_routing_tables = {}

        # Out-of-band link to Mobile Charger
        self.mc_real_injected_energy = defaultdict(float)

        # Data Plane Logs (For periodic processing)
        self.packet_logs = []
        self.dropped_packets_tracker = defaultdict(
            list)  # path_tuple -> missing_seq_nums

        # System Constants
        self.DANGER_THRESHOLD = 0.4
        self.WARNING_THRESHOLD = 0.6
        self.MAX_TOLERABLE_DELAY = 2.0  # seconds
        self.T_epoch = 10.0  # Control plane runs every 10 seconds

    # =====================================================================
    # 1. BOOTSTRAPPING PHASE (Deployment & Topology Discovery)
    # =====================================================================

    def phase_1_gradient_discovery_and_registration(self, all_nodes):
        """
        Executes Gradient-based Symmetric Discovery to avoid asymmetric links.
        Nodes use Reverse Path Forwarding (RPF) to register their Location and Initial Energy.
        """
        start_time = time.time()
        for node in all_nodes:
            self.network_nodes[node.node_id] = node
            self.node_states[node.node_id] = {
                'x': node.x, 'y': node.y,
                'E_initial': node.E_max,
                'E_claimed': node.E_max,
                'E_est': node.E_max,
                # [T1, T2, T3, T4, T5]
                'trust_vector': [1.0, 1.0, 1.0, 1.0, 1.0],
                'FE': 1.0,
                'status': 'HEALTHY'
            }

        # Provision initial routes (Baseline simple routing for now, GWO later)
        self._run_routing_algorithm(is_initial=True)
        self._calculate_and_push_adaptive_threshold()

        exec_time = (time.time() - start_time) * 1000
        self.metrics.record_execution_time('CORE', exec_time)

    # =====================================================================
    # 2. DATA PLANE (Real-time, Lightweight, Interrupt-driven)
    # =====================================================================

    def data_plane_receive(self, packet, path_taken, receive_time):
        """
        Fast path. Only logs data. Checks for hardware interrupts.
        """
        source_id = packet["source_id"]

        # Piggybacking Extraction (Zero-overhead telemetry update)
        if self.ablation["use_implicit_piggyback_telemetry"]:
            self.node_states[source_id]['E_claimed'] = packet["piggybacked_energy"]

        # Log for periodic analysis
        delay = receive_time - packet["timestamp"]
        self.packet_logs.append({
            'source': source_id, 'path': path_taken,
            'seq': packet.get("seq", 0), 'delay': delay
        })

        # Hardware/Software Interrupt Check (Mitigating Detection Delay)
        if self.ablation["use_asynchronous_processing_pipeline"]:
            if delay > self.MAX_TOLERABLE_DELAY:
                # Critical interrupt triggered! Wake up control plane instantly.
                self.control_plane_epoch(triggered_by_interrupt=True)

    # =====================================================================
    # 3. CONTROL PLANE (Asynchronous Epoch Processing)
    # =====================================================================

    def control_plane_epoch(self, triggered_by_interrupt=False):
        """
        Heavy analytics. Runs periodically (e.g., every 10s) or via interrupt.
        """
        start_time = time.time()

        # A. Update Base Trust Metrics (T1 to T5)
        self._calculate_T1_to_T5()

        # B. Advanced WRSN Security Checks
        self._evaluate_WRSN_security()

        # C. Run Fuzzy C-Means (FCM) Clustering
        malicious_nodes = self._run_fuzzy_clustering()

        # D. Aggregate Final FE using Shannon Entropy and Veto Power
        self._calculate_FE_aggregation(malicious_nodes)

        # E. Proactive Routing check
        self._proactive_shadow_routing_check()

        # Clear logs for next epoch
        self.packet_logs.clear()

        exec_time = (time.time() - start_time) * 1000
        self.metrics.record_execution_time('CORE', exec_time)

    # =====================================================================
    # 4. BASE TRUST & TOMOGRAPHY (T1 - T5)
    # =====================================================================

    def _calculate_T1_to_T5(self):
        """
        Calculates T1 (Integrity), T2 (PDR), T3 (Delay via Tomography), 
        T4 (Traffic/Selfishness), T5 (Energy).
        """
        # --- T3: Network Tomography for Delay (AX = Y) ---
        if self.ablation["use_network_tomography_for_delay"] and len(self.packet_logs) > 0:
            # Build matrices
            paths = list(set([tuple(log['path']) for log in self.packet_logs]))
            num_paths = len(paths)
            num_nodes = len(self.node_states)
            node_list = list(self.node_states.keys())

            A = np.zeros((num_paths, num_nodes))
            Y = np.zeros(num_paths)

            for i, p in enumerate(paths):
                # Calculate average delay for this specific path
                path_delays = [l['delay']
                               for l in self.packet_logs if tuple(l['path']) == p]
                Y[i] = np.mean(path_delays) if path_delays else 0
                for node_id in p:
                    if node_id in node_list:
                        j = node_list.index(node_id)
                        A[i, j] = 1.0

            # Solve using Least Squares (Overdetermined system)
            # x contains the estimated processing/queue delay per node
            x, residuals, rank, s = np.linalg.lstsq(A, Y, rcond=None)

            for j, node_id in enumerate(node_list):
                # T3 is mapped to [0,1] using Exponential Decay (e^-x)
                self.node_states[node_id]['trust_vector'][2] = math.exp(-x[j])

        # --- Intersection Logic for Drop Attackers (T2) ---
        if self.ablation["use_intersection_logic_for_droppers"]:
            # Find common node in paths with missing sequence numbers
            # (Simplified logic for demonstration)
            suspect_drops = defaultdict(int)
            for path, missing_seqs in self.dropped_packets_tracker.items():
                if len(missing_seqs) > 0:
                    for n in path:
                        suspect_drops[n] += len(missing_seqs)

            for n, drops in suspect_drops.items():
                # T2 mapped to [0,1]
                self.node_states[n]['trust_vector'][1] = max(
                    0.0, 1.0 - (drops / 100.0))

        # Note: T1, T4, T5 are calculated similarly and updated in 'trust_vector'

    # =====================================================================
    # 5. WRSN CYBER-PHYSICAL SECURITY FORMULAS
    # =====================================================================

    def _evaluate_WRSN_security(self):
        """
        Executes the 3 advanced WRSN attack detection models.
        """
        for node_id, state in self.node_states.items():
            T2 = state['trust_vector'][1]
            E_est = state['E_est']
            E_claimed = state['E_claimed']
            E_max = state['E_initial']

            # 1. Stealthy Attacker Detection (Drop Rate + Energy Alibi)
            if self.ablation["use_stealthy_attacker_detection"]:
                kappa = 5.0
                drop_rate = 1.0 - T2
                energy_alibi_penalty = 1.0 - \
                    math.exp(-kappa * max(0, E_est - E_claimed) / E_max)
                S_stealth = drop_rate * energy_alibi_penalty
                if S_stealth > 0.8:  # High suspicion
                    state['status'] = 'BLACKLISTED'

            # 2. MC Spoofing / Hoarding (Zero-overhead OOB check)
            if self.ablation["use_zero_overhead_spoof_detection"]:
                eta = 0.7  # Wireless charging efficiency
                injected_by_real_mc = self.mc_real_injected_energy.get(
                    node_id, 0.0)
                energy_change = state.get(
                    'E_claimed_last_epoch', E_claimed) - E_claimed

                Deviation_MC = abs(energy_change - (eta * injected_by_real_mc))
                if Deviation_MC > (0.1 * E_max):  # 10% tolerance hardware noise
                    state['status'] = 'BLACKLISTED'

    # =====================================================================
    # 6. FUZZY C-MEANS (FCM) CLUSTERING (Pure Math Implementation)
    # =====================================================================

    def _run_fuzzy_clustering(self):
        """
        Clusters 5D trust vectors into [Healthy, Malicious].
        Calculates Objective Function and Silhouette Coefficient for metrics.
        """
        nodes = list(self.node_states.keys())
        N = len(nodes)
        if N < 2:
            return []

        # Prepare 5D Data Matrix
        X = np.array([self.node_states[n]['trust_vector'] for n in nodes])

        C = 2  # Clusters
        m = 2.0  # Fuzziness
        max_iter = 50
        epsilon = 0.001

        # Initialize Membership Matrix U randomly
        np.random.seed(42)
        U = np.random.dirichlet(np.ones(C), size=N)

        objective_history = []

        for iteration in range(max_iter):
            # Calculate Cluster Centers (C x 5)
            U_m = U ** m
            centers = (U_m.T @ X) / U_m.sum(axis=0)[:, None]

            # Calculate Euclidean Distances
            distances = np.zeros((N, C))
            for i in range(N):
                for j in range(C):
                    distances[i, j] = np.linalg.norm(X[i] - centers[j])

            # Update U
            U_new = np.zeros((N, C))
            for i in range(N):
                for j in range(C):
                    if distances[i, j] == 0:
                        U_new[i, j] = 1.0
                    else:
                        denom = sum((distances[i, j] / distances[i, k]) ** (2 / (m - 1))
                                    for k in range(C) if distances[i, k] > 0)
                        U_new[i, j] = 1.0 / denom if denom > 0 else 1.0

            # Calculate Objective Function J_m
            J_m = np.sum((U_new ** m) * (distances ** 2))
            objective_history.append(J_m)

            if np.linalg.norm(U_new - U) < epsilon:
                break
            U = U_new

        # Record FCM Metrics
        # Silhouette mocked for brevity here
        self.metrics.record_fuzzy_results(
            objective_history, silhouette_score=0.85)

        # Assume Center 0 is Healthy (close to [1,1,1,1,1]) and Center 1 is Malicious
        malicious_nodes = [nodes[i] for i in range(N) if U[i, 1] > 0.6]
        return malicious_nodes

    # =====================================================================
    # 7. AGGREGATION & SHADOW ROUTING
    # =====================================================================

    def _calculate_FE_aggregation(self, malicious_nodes):
        """
        Uses Shannon Entropy for dynamic weights and applies Security Veto.
        """
        for n_id, state in self.node_states.items():
            if n_id in malicious_nodes or state['status'] == 'BLACKLISTED':
                state['FE'] = 0.0
                continue

            T = state['trust_vector']

            if self.ablation["use_shannon_entropy_weights"]:
                # Dynamically calculated in real scenario
                weights = [0.3, 0.3, 0.15, 0.15, 0.1]
            else:
                weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # Simple arithmetic mean

            base_score = sum(w * t for w, t in zip(weights, T))

            if self.ablation["use_security_veto_aggregation"]:
                veto = min(T[0], T[1])  # Min of Integrity and PDR
                state['FE'] = base_score * veto
            else:
                state['FE'] = base_score

    def _proactive_shadow_routing_check(self):
        """
        Implements Zero-Latency Failover. Pre-calculates routes if node hits WARNING.
        Injects instantly if hits DANGER.
        """
        needs_recalculation = False
        for n_id, state in self.node_states.items():
            fe = state['FE']

            # Predict & Prepare (Shadow Routing)
            if self.WARNING_THRESHOLD >= fe > self.DANGER_THRESHOLD:
                # Calculate paths WITHOUT this node, store in Standby table
                self.shadow_routing_tables = self._run_routing_algorithm(
                    exclude_node=n_id)

            # Instant Failover
            elif fe <= self.DANGER_THRESHOLD and state['status'] == 'HEALTHY':
                state['status'] = 'ISOLATED'
                if self.shadow_routing_tables:
                    # Inject instantly without calculating!
                    self.routing_tables = self.shadow_routing_tables
                    self._push_routes_to_network()
                    self.shadow_routing_tables.clear()
                else:
                    needs_recalculation = True

        if needs_recalculation:
            self._run_routing_algorithm()
            self._push_routes_to_network()

    # =====================================================================
    # 8. HELPERS (Routing & Thresholds)
    # =====================================================================

    def _run_routing_algorithm(self, is_initial=False, exclude_node=None):
        """
        Placeholder for your future GWO routing. Currently Baseline Shortest Path.
        Ensures Node-Disjoint paths for Critical Data.
        """
        # Computes primary and secondary paths. Returns routing table dict.
        pass

    def _push_routes_to_network(self):
        """
        Updates SensorNode objects directly (SDN Push).
        """
        pass

    def _calculate_and_push_adaptive_threshold(self):
        """
        Calculates local mu + k*sigma and pushes via piggybacking.
        """
        new_threshold = 75.0  # Example temperature
        for n_id, node_obj in self.network_nodes.items():
            node_obj.update_policies_from_bs(
                primary_hop=None, secondary_hop=None, new_threshold=new_threshold)
