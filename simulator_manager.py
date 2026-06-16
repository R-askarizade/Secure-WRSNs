import random
import math
from collections import defaultdict
import numpy as np
import networkx as nx

from config import compute_packet_hash, DATA_PACKET_BYTES
from sensor_node import SensorNode
from base_station import BaseStation
from mobile_charger import MobileCharger
from metric_collector import MetricsCollector
from routing import BaselineRouting


class SimulatorManager:
    def __init__(self, num_nodes, sim_time, ablation_config, seed, network_size=(100, 100)):
        self.num_nodes = num_nodes
        self.sim_time = sim_time
        self.ablation = ablation_config
        self.seed = seed
        self.network_size = network_size
        self.BS_ID = 0
        self.transmission_range = float(
            self.ablation.get('transmission_range', 40.0))
        self.source_seq_counters = defaultdict(int)

        random.seed(self.seed)
        np.random.seed(self.seed)

        self.metrics = MetricsCollector(
            total_nodes=num_nodes, total_rounds_or_time=sim_time)
        self.bs = BaseStation(metrics_collector=self.metrics,
                              ablation_config=self.ablation, bs_id=self.BS_ID)

        bs_x, bs_y = network_size[0] / 2, network_size[1] / 2
        self.mc = MobileCharger(mc_id=999, depot_x=bs_x, depot_y=bs_y, capacity=50000.0,
                                speed=5.0, base_station=self.bs, metrics_collector=self.metrics)

        self.nodes = {}

        self.routing_engine = BaselineRouting(
            bs_id=self.BS_ID, transmission_range=self.transmission_range,
            max_critical_paths=4 if self.ablation.get('use_risk_adaptive_coded_multipath', False) else 2)
        self.bs.routing_engine = self.routing_engine

        self._deploy_network(bs_x, bs_y)

        self.malicious_nodes = []
        self.current_time = 0.0
        self.experiment_family = self.ablation.get(
            'experiment_family', 'communication_security')
        self.energy_experiment_active = self.bs.energy_experiment_active
        self.physical_compromise_experiment_active = (
            self.experiment_family == 'physical_compromise'
            or self.ablation.get('enable_physical_compromise_experiment', False)
        )

    def _deploy_network(self, bs_x, bs_y):
        self.nodes[self.BS_ID] = SensorNode(
            self.BS_ID, bs_x, bs_y, float('inf'), self.metrics)
        initial_energy = float(self.ablation.get(
            'initial_energy_override', 0.1))
        for i in range(1, self.num_nodes + 1):
            x = random.uniform(0, self.network_size[0])
            y = random.uniform(0, self.network_size[1])
            self.nodes[i] = SensorNode(i, x, y, initial_energy, self.metrics)

        self.bs.network_nodes = self.nodes
        self.bs.mc_reference = self.mc
        self.routing_engine.quarantined_zones = self.bs.quarantined_zones
        for _node in self.nodes.values():
            _node.network_nodes = self.nodes

    def bootstrap_network(self):
        print(
            f"[Seed {self.seed}] Executing Rigorous Bootstrapping Phase (Gradient & Source Routing)...")

        BS_ADV_SIZE = 16
        REG_SIZE = 48
        POLICY_SIZE = 64

        def get_physical_neighbors(node_id):
            n1 = self.nodes[node_id]
            neighbors = []
            for n2_id, n2 in self.nodes.items():
                if node_id != n2_id:
                    dist = math.hypot(n1.x - n2.x, n1.y - n2.y)
                    if dist <= self.transmission_range:
                        neighbors.append(n2_id)
            return neighbors

        seq_num = 1
        wavefront = [self.BS_ID]
        current_hop = 1

        self.nodes[self.BS_ID]._consume_energy(
            tx_bytes=BS_ADV_SIZE, distance=self.transmission_range)
        self.nodes[self.BS_ID].seen_adv_seq = seq_num
        self.nodes[self.BS_ID].hop_count_to_bs = 0

        while wavefront:
            next_wave = []
            for sender_id in wavefront:
                neighbors = get_physical_neighbors(sender_id)
                for neighbor_id in neighbors:
                    neighbor_node = self.nodes[neighbor_id]
                    if neighbor_id == self.BS_ID:
                        continue
                    if neighbor_node.seen_adv_seq < seq_num:
                        neighbor_node.seen_adv_seq = seq_num
                        neighbor_node.hop_count_to_bs = current_hop
                        neighbor_node.temp_parent = sender_id
                        neighbor_node._consume_energy(rx_bytes=BS_ADV_SIZE)
                        neighbor_node._consume_energy(
                            tx_bytes=BS_ADV_SIZE, distance=self.transmission_range)
                        self.metrics.update_packet_transmission(
                            is_critical=False, is_control=True, size_bytes=BS_ADV_SIZE, overhead_phase='setup')
                        next_wave.append(neighbor_id)
            wavefront = next_wave
            current_hop += 1

        self.current_time += 1.5

        for n_id, node in self.nodes.items():
            if n_id == self.BS_ID:
                continue

            node.crypto_keys = f"AES-128-Key-Node{node.node_id}-{random.randint(1000, 9999)}"
            current_n_id = n_id
            packet_dropped = False

            while current_n_id != self.BS_ID:
                current_node = self.nodes[current_n_id]
                parent_id = current_node.temp_parent
                if parent_id is None:
                    packet_dropped = True
                    break
                parent_node = self.nodes[parent_id]
                dist = math.hypot(current_node.x - parent_node.x,
                                  current_node.y - parent_node.y)
                current_node._consume_energy(tx_bytes=REG_SIZE, distance=dist)
                if parent_id != self.BS_ID:
                    parent_node._consume_energy(rx_bytes=REG_SIZE)
                current_n_id = parent_id

            if not packet_dropped:
                self.metrics.update_packet_transmission(
                    is_critical=False, is_control=True, size_bytes=REG_SIZE, overhead_phase='setup')

                # Standardized Node State Dictionary initialization
                self.bs.node_states[node.node_id] = {
                    'x': node.x, 'y': node.y,
                    'E_initial': node.E_max,
                    'E_claimed': node.E_max,
                    'E_claimed_prev': node.E_max,
                    'E_claimed_prev_SED': node.E_max,
                    'E_est': node.E_current,
                    'E_consumption_est_cumulative': 0.0,
                    'total_injected_energy': 0.0,
                    'trust_vector': [1.0, 1.0, 1.0, 1.0, 1.0],
                    'T1_history': 1.0, 'T2_history': 1.0, 'T3_history': 1.0, 'T4_history': 1.0, 'T5_history': 1.0,
                    'FE': 1.0,
                    'status': 'WHITELISTED',
                    'crypto_keys': node.crypto_keys,
                    'hard_veto': False,
                    'spoof_trust': 1.0,
                    'spoof_rescue_triggered': False,
                    'Deviation_MC_norm': 0.0,
                    'SED_index': 0.0,
                    'S_stealth': 0.0,
                    'risk_mean': 0.0,
                    'risk_instant': 0.0,
                    'risk_lcb': 0.0,
                    'risk_uncertainty': 0.0,
                    'fcm_risk': 0.0,
                    'prev_epoch_seq': 0,
                    'lambda_gen': 0.0,
                    'blacklist_counter': 0,
                    'idle_sensing_energy_epoch': 0.0
                }

        self.current_time += 2.0

        active_nodes_for_bs = {nid: self.nodes[nid]
                               for nid in self.bs.node_states.keys()}
        active_nodes_for_bs[self.BS_ID] = self.nodes[self.BS_ID]
        initial_tables = self.routing_engine.compute_routing_tables(
            active_nodes_for_bs)
        self.bs.routing_tables = initial_tables
        bs_internal_graph = self.routing_engine.build_network_graph(
            active_nodes_for_bs)
        adaptive_threshold = 75.0

        for n_id, table in initial_tables.items():
            if n_id == self.BS_ID:
                continue
            try:
                source_route = nx.shortest_path(
                    bs_internal_graph, source=self.BS_ID, target=n_id, weight='weight')
            except nx.NetworkXNoPath:
                continue

            for i in range(len(source_route) - 1):
                sender_id = source_route[i]
                receiver_id = source_route[i+1]
                sender = self.nodes[sender_id]
                receiver = self.nodes[receiver_id]
                dist = math.hypot(sender.x - receiver.x, sender.y - receiver.y)
                if sender_id != self.BS_ID:
                    sender._consume_energy(tx_bytes=POLICY_SIZE, distance=dist)
                if receiver_id != n_id:
                    receiver._consume_energy(rx_bytes=POLICY_SIZE)

            target_node = self.nodes[n_id]
            target_node._consume_energy(rx_bytes=POLICY_SIZE)
            target_node.update_policies_from_bs(
                table['primary'], table['secondary'], adaptive_threshold)
            target_node.temp_parent = None
            target_node.hop_count_to_bs = float('inf')
            self.metrics.update_packet_transmission(
                is_critical=False, is_control=True, size_bytes=POLICY_SIZE, overhead_phase='setup')

        self.current_time += 1.5

        self.mc.crypto_keys = f"MC-SECURE-KEY-{random.randint(1000, 9999)}"
        self.bs.mc_shared_key = self.mc.crypto_keys
        self.metrics.update_packet_transmission(
            is_critical=False, is_control=True, size_bytes=32, overhead_phase='setup')
        self.current_time += 0.5

        print(
            f"[Seed {self.seed}] Bootstrapping Complete. Temp trees cleared. Network Setup Time: {self.current_time}s")

    def service_mobile_charger(self):
        if self.energy_experiment_active:
            for nid, node in self.nodes.items():
                if nid == self.BS_ID or node.status != 'ALIVE':
                    continue
                if node.E_current <= node.E_critical:
                    self.metrics.record_energy_critical(nid, self.current_time)

        if not self.bs.energy_experiment_active or self.mc.status != "IDLE":
            return

        target_node = None
        secure_queue = self.bs.secure_charging_queue
        secure_queue[:] = [
            nid for nid in secure_queue if self.bs.validate_charging_request(nid, emergency=True)]

        if secure_queue:
            target_id = min(secure_queue, key=lambda nid: math.hypot(
                self.mc.x - self.nodes[nid].x, self.mc.y - self.nodes[nid].y))
            secure_queue.remove(target_id)
            target_node = self.nodes.get(target_id)
        elif len(self.bs.pending_sos_requests) > 0:
            valid_sos = [nid for nid in self.bs.pending_sos_requests if nid in self.nodes and self.bs.validate_charging_request(
                nid, emergency=True)]
            if valid_sos:
                target_id = min(valid_sos, key=lambda nid: math.hypot(
                    self.mc.x - self.nodes[nid].x, self.mc.y - self.nodes[nid].y))
                self.bs.pending_sos_requests.discard(target_id)
                target_node = self.nodes.get(target_id)
        else:
            target_node = self._select_predictive_charging_target()

        if target_node and target_node.status == "ALIVE" and self.bs.validate_charging_request(target_node.node_id):
            travel_time = self.mc.move_to_node(target_node)
            charge_time = self.mc.charge_node(target_node)
            self.current_time += (travel_time + charge_time)
            if self.mc.E_current < 0.1 * self.mc.E_capacity:
                self.mc.return_to_depot()

    def _select_predictive_charging_target(self):
        best_node = None
        best_slack = float('inf')
        for nid, state in self.bs.node_states.items():
            if nid == self.BS_ID or not self.bs.validate_charging_request(nid):
                continue
            node = self.nodes.get(nid)
            if node is None or node.status != "ALIVE":
                continue
            E_initial = state['E_initial']
            E_est = state.get('E_est', E_initial)
            consumed_energy = max(0.0, E_initial - E_est)
            elapsed_time = max(1.0, self.current_time)
            consumption_rate = consumed_energy / elapsed_time
            if consumption_rate <= 0:
                continue
            time_to_death = E_est / consumption_rate
            travel_distance = math.hypot(
                self.mc.x - node.x, self.mc.y - node.y)
            travel_time = travel_distance / max(self.mc.speed, 1e-6)
            slack = time_to_death - travel_time
            if slack < best_slack and time_to_death <= 7200.0:
                best_slack = slack
                best_node = node
        return best_node

    def lab_reprovision_mobile_charger(self):
        lab_time_delay = 7200.0
        self.current_time += lab_time_delay
        new_secure_key = f"MC-SECURE-KEY-NEW-LAB-{random.randint(1000, 9999)}"
        self.mc.crypto_keys = new_secure_key
        self.bs.mc_shared_key = new_secure_key
        self.mc.zeroized = False
        self.mc.requires_lab_reprovisioning = False
        self.mc.status = "IDLE"
        self.mc.x, self.mc.y = self.mc.depot_x, self.mc.depot_y

    def trigger_physical_compromise_event(self, simulate_lab_reprovisioning=False):
        self.mc.tpms_zeroization()
        self.bs.mc_shared_key = None
        if simulate_lab_reprovisioning:
            self.lab_reprovision_mobile_charger()

    def inject_attacks(self, attack_percentage):
        num_attackers = int(self.num_nodes * attack_percentage)
        alive_node_ids = [
            nid for nid in self.nodes.keys() if nid != self.BS_ID]
        self.malicious_nodes = random.sample(alive_node_ids, num_attackers)
        for nid in self.malicious_nodes:
            self.nodes[nid].is_malicious = True
        print(
            f"[ATTACK INJECTION] {num_attackers} malicious nodes injected into the network.")

    def run_simulation(self):
        def _trace_control_packet(source_id, packet):
            table = self.bs.routing_tables.get(source_id, {})
            routes = []

            primary_path = table.get('primary_path') or []
            if primary_path and primary_path[0] == source_id and primary_path[-1] == self.BS_ID:
                routes.append(primary_path)
            elif self.nodes[source_id].primary_next_hop is not None:
                routes.append(
                    [source_id, self.nodes[source_id].primary_next_hop, self.BS_ID])

            secondary_path = table.get('secondary_path') or []
            if (packet.get('priority') == 'HIGH' and
                    self.ablation.get('use_critical_multipath_routing', False) and
                    secondary_path and secondary_path[0] == source_id and
                    secondary_path[-1] == self.BS_ID):
                routes.append(secondary_path)

            for route in routes:
                pkt = dict(packet)
                pkt['energy_trace'] = list(packet.get('energy_trace', []))
                pkt['path_header'] = [source_id]
                pkt['hash'] = compute_packet_hash(
                    pkt, self.nodes[source_id].crypto_keys)

                packet_dropped = False
                path_log = [source_id]

                for idx in range(1, len(route)):
                    current_hop_id = route[idx]
                    if current_hop_id == self.BS_ID:
                        path_log.append(self.BS_ID)
                        break

                    hop_node = self.nodes[current_hop_id]
                    path_log.append(current_hop_id)
                    next_hop_id = route[idx + 1] if idx + \
                        1 < len(route) else self.BS_ID

                    if next_hop_id != self.BS_ID:
                        next_node = self.nodes[next_hop_id]
                        distance = math.hypot(
                            hop_node.x - next_node.x, hop_node.y - next_node.y)
                    else:
                        bs_node = self.nodes[self.BS_ID]
                        distance = math.hypot(
                            hop_node.x - bs_node.x, hop_node.y - bs_node.y)

                    success = hop_node.forward_packet(
                        pkt, self.current_time, distance)
                    if not success:
                        packet_dropped = True
                        break

                if not packet_dropped and path_log[-1] == self.BS_ID:
                    pkt['path_header'] = path_log
                    delay = len(path_log) * 0.100
                    self.bs.data_plane_receive(
                        pkt, path_log, self.current_time + delay)

        if self.ablation.get('enable_periodic_hello', False):
            hello_period = float(self.ablation.get('hello_period', 60.0))
            hello_count = int(max(0.0, self.sim_time) //
                              max(hello_period, 1e-9)) * max(self.num_nodes, 0)
            for _ in range(hello_count):
                self.metrics.update_packet_transmission(
                    is_critical=False, is_control=True, size_bytes=16, overhead_phase='operational')

        self.bootstrap_network()
        self.inject_attacks(attack_percentage=float(
            self.ablation.get('attack_percentage', 0.15)))

        packet_seq_num = 0
        next_event_time = random.expovariate(
            float(self.ablation.get('event_rate_lambda', 0.5)))
        next_epoch_time = self.bs.T_epoch
        self.current_time = 0.0
        self.metrics.current_sim_time = self.current_time

        last_energy_record_time = 0.0
        last_energy_drain_time = 0.0
        tpm_attack_time = random.uniform(
            self.sim_time * 0.3, self.sim_time * 0.7)
        tpm_attack_triggered = False

        while self.current_time < self.sim_time:
            if (self.physical_compromise_experiment_active and not tpm_attack_triggered and self.current_time >= tpm_attack_time):
                self.trigger_physical_compromise_event()
                tpm_attack_triggered = True

            if next_event_time <= next_epoch_time:
                self.current_time = next_event_time
                self.metrics.current_sim_time = self.current_time
                alive_nodes = [n for n, obj in self.nodes.items(
                ) if obj.status == "ALIVE" and n != self.BS_ID]
                if not alive_nodes:
                    print(
                        f"[Seed {self.seed}] Network died at t={self.current_time}")
                    break

                active_node_id = random.choice(alive_nodes)
                active_node = self.nodes[active_node_id]

                is_critical_event = random.random() < float(
                    self.ablation.get('critical_event_probability', 0.10))
                raw_data = 80.0 if is_critical_event else 25.0
                packet_seq_num += 1

                method_is_critical = raw_data > active_node.adaptive_threshold
                active_node.sense_environment(raw_data, self.current_time)
                self._trace_packet_path(
                    active_node_id, packet_seq_num, method_is_critical, raw_data)

                inter_arrival = random.expovariate(
                    float(self.ablation.get('event_rate_lambda', 0.5)))
                next_event_time = self.current_time + inter_arrival
            else:
                self.current_time = next_epoch_time
                self.metrics.current_sim_time = self.current_time
                self.bs.control_plane_epoch()
                self._push_updated_routes()
                next_epoch_time = self.current_time + self.bs.T_epoch

            if self.energy_experiment_active:
                delta_t = max(0.0, self.current_time - last_energy_drain_time)
                if delta_t > 0:
                    idle_power = float(self.ablation.get(
                        'idle_sensing_power_j_per_s', 0.0))
                    if idle_power > 0:
                        for n_id, node in self.nodes.items():
                            if n_id == self.BS_ID or node.status != "ALIVE":
                                continue
                            idle_energy = idle_power * delta_t
                            node.E_current = max(
                                0.0, node.E_current - idle_energy)
                            if n_id in self.bs.node_states:
                                self.bs.node_states[n_id]['idle_sensing_energy_epoch'] += idle_energy
                            if node.E_current <= 0 and node.status == "ALIVE":
                                node.status = "DEAD"
                                self.metrics.record_node_death(
                                    n_id, self.current_time)
                last_energy_drain_time = self.current_time

            if self.current_time - last_energy_record_time >= 10.0:
                avg_energy = sum([n.E_current for n in self.nodes.values(
                ) if n.node_id != self.BS_ID]) / self.num_nodes
                self.metrics.record_energy_state(avg_energy)
                last_energy_record_time = self.current_time

            for n_id, node in self.nodes.items():
                if n_id != self.BS_ID:
                    topology_packet = node.check_route_health(
                        self.nodes, self.transmission_range, self.current_time)
                    if topology_packet is not None:
                        _trace_control_packet(n_id, topology_packet)
                    control_packet = node.check_inactivity(self.current_time)
                    if control_packet is not None:
                        _trace_control_packet(n_id, control_packet)

            if self.energy_experiment_active:
                self.service_mobile_charger()

        detected = self.bs.get_blacklisted_nodes()
        actual = self.malicious_nodes
        detected = [d for d in detected if d != self.bs.BS_ID]
        actual = [a for a in actual if a != self.bs.BS_ID]
        benign_count = self.num_nodes - len(actual)
        tp = len(set(detected) & set(actual))
        fn = len(set(actual) - set(detected))
        fp = len(set(detected) - set(actual))
        tn = benign_count - fp
        self.metrics.update_ids_matrix(tp, tn, fp, fn)

    def _trace_packet_path(self, source_id, seq_num, is_critical, payload_value):
        if not self.ablation.get('use_implicit_piggyback_telemetry', True):
            self.metrics.update_packet_transmission(
                is_critical=False, is_control=True, size_bytes=16, overhead_phase='operational')
        if self.ablation.get('enable_network_layer_ack', False):
            self.metrics.update_packet_transmission(
                is_critical=False, is_control=True, size_bytes=16, overhead_phase='operational')

        self.source_seq_counters[source_id] += 1
        seq_num = self.source_seq_counters[source_id]
        key = (int(source_id), int(seq_num))

        if key not in self.metrics._unique_generated_data_keys:
            self.metrics._unique_generated_data_keys.add(key)
            self.metrics.update_packet_transmission(
                is_critical=bool(is_critical), is_control=False, size_bytes=DATA_PACKET_BYTES,
                is_delivered=False, count_generated=True
            )

        before_received = self.metrics.received_data_packets
        before_received_critical = self.metrics.received_critical_packets
        before_bits = self.metrics.total_bits_received
        before_delay_len = len(self.metrics.end_to_end_delays)

        table = self.bs.routing_tables.get(source_id, {})
        if is_critical and self.ablation.get("use_critical_multipath_routing", False):
            candidate_paths = table.get('critical_paths') or []
            if not candidate_paths:
                candidate_paths = [p for p in [
                    table.get('primary_path'), table.get('secondary_path')] if p]
            self.metrics.record_critical_multipath_opportunity(
                secondary_available=(len(candidate_paths) >= 2))
            source_risk = self.bs.node_states.get(
                source_id, {}).get('risk_mean', 0.0)
            max_paths = 2
            if self.ablation.get('use_risk_adaptive_coded_multipath', False):
                if source_risk >= 0.35:
                    max_paths = 3
                if source_risk >= 0.65:
                    max_paths = 4
            paths_to_trace = [{'type': f'CRITICAL_PATH_{idx+1}', 'path': p}
                              for idx, p in enumerate(candidate_paths[:max_paths])]
            required_fragments = 1
        else:
            primary_path = table.get('primary_path') or []
            if primary_path:
                paths_to_trace = [{'type': 'PRIMARY', 'path': primary_path}]
            elif self.nodes[source_id].primary_next_hop is not None:
                paths_to_trace = [{'type': 'PRIMARY', 'path': [
                    source_id, self.nodes[source_id].primary_next_hop, self.BS_ID]}]
            else:
                paths_to_trace = []
            required_fragments = 1

        delivered_fragments = 0
        first_success_delay = None

        for path_index, p_info in enumerate(paths_to_trace):
            route = list(p_info['path'])
            if not route or route[0] != source_id or route[-1] != self.BS_ID:
                continue
            if is_critical and path_index >= 2:
                self.nodes[source_id]._consume_energy(
                    tx_bytes=DATA_PACKET_BYTES, distance=30.0)

            packet = {
                "source_id": source_id,
                "seq": seq_num,
                "timestamp": self.current_time,
                "size_bytes": DATA_PACKET_BYTES,
                "path_header": [source_id],
                "payload": payload_value,
                "priority": "HIGH" if is_critical else "NORMAL",
                "fragment_id": path_index,
                "required_fragments": required_fragments,
                "energy_trace": [(source_id, self.nodes[source_id].E_current)]
            }
            packet["hash"] = compute_packet_hash(
                packet, self.nodes[source_id].crypto_keys)
            packet_dropped = False
            path_log = [source_id]

            for idx in range(1, len(route)):
                current_hop_id = route[idx]
                if current_hop_id == self.BS_ID:
                    path_log.append(self.BS_ID)
                    break
                hop_node = self.nodes[current_hop_id]
                path_log.append(current_hop_id)
                if hop_node.is_malicious:
                    packet_dropped = True
                    break
                next_hop_id = route[idx + 1] if idx + \
                    1 < len(route) else self.BS_ID
                if next_hop_id != self.BS_ID:
                    next_node = self.nodes[next_hop_id]
                    distance = math.hypot(
                        hop_node.x - next_node.x, hop_node.y - next_node.y)
                else:
                    bs_node = self.nodes[self.BS_ID]
                    distance = math.hypot(
                        hop_node.x - bs_node.x, hop_node.y - bs_node.y)
                success = hop_node.forward_packet(
                    packet, self.current_time, distance)
                if not success:
                    packet_dropped = True
                    break

            if not packet_dropped and path_log[-1] == self.BS_ID:
                simulated_delay = len(path_log) * 0.100
                delivered_fragments += 1
                if first_success_delay is None or simulated_delay < first_success_delay:
                    first_success_delay = simulated_delay
                packet_data = dict(packet)
                packet_data["path_header"] = path_log
                self.bs.data_plane_receive(
                    packet_data, path_log, receive_time=self.current_time + simulated_delay)

        if is_critical and self.ablation.get("use_critical_multipath_routing", False):
            attempted_redundant = max(0, len(paths_to_trace) - 1)
            accepted_redundant = max(0, delivered_fragments - 1)
            rejected_redundant = max(
                0, attempted_redundant - accepted_redundant)
            self.metrics.critical_redundant_copy_accepted += accepted_redundant
            self.metrics.critical_redundant_copy_rejected += rejected_redundant

        if delivered_fragments >= required_fragments:
            self.metrics.update_packet_transmission(
                is_critical=is_critical, is_control=False, size_bytes=DATA_PACKET_BYTES,
                is_delivered=True, delay=first_success_delay if first_success_delay is not None else 0.0,
                count_generated=False
            )

        after_received = self.metrics.received_data_packets
        if after_received > before_received:
            if key in self.metrics._unique_delivered_data_keys:
                delta_recv = after_received - before_received
                self.metrics.received_data_packets -= delta_recv
                if is_critical:
                    self.metrics.received_critical_packets = max(
                        before_received_critical, self.metrics.received_critical_packets - delta_recv)
                self.metrics.total_bits_received = before_bits
                del self.metrics.end_to_end_delays[before_delay_len:]
            else:
                self.metrics._unique_delivered_data_keys.add(key)

    def _push_updated_routes(self):
        for n_id, table in self.bs.routing_tables.items():
            if n_id in self.nodes:
                new_adaptive_th = self.bs.get_adaptive_threshold_for_node(n_id)
                self.nodes[n_id].update_policies_from_bs(
                    primary_hop=table.get('primary'),
                    secondary_hop=table.get('secondary'),
                    new_threshold=new_adaptive_th
                )
