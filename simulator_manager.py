import random
import math
import json
from collections import defaultdict
import numpy as np
import networkx as nx

from config import compute_packet_hash, DATA_PACKET_BYTES
from sensor_node import SensorNode
from base_station import BaseStation
from mobile_charger import MobileCharger
from metric_collector import MetricsCollector
from routing import BaselineRouting
from adaptive_charging import ChargerRoutingDQN


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

        self.sensor_initial_energy = float(self.ablation.get(
            'initial_energy_override', 0.1))

        bs_x, bs_y = network_size[0] / 2, network_size[1] / 2
        self.mobile_chargers = self._make_mobile_charger_fleet(bs_x, bs_y)

        self.mc = self.mobile_chargers[0]
        self.use_multi_mobile_chargers = len(self.mobile_chargers) > 1
        self.metrics.mobile_charger_count = len(self.mobile_chargers)
        self.metrics.mobile_charger_capacity_mode = str(self.ablation.get(
            'mobile_charger_capacity_mode', 'absolute'))
        self.metrics.sensor_initial_energy = float(self.sensor_initial_energy)
        self.metrics.mobile_charger_total_capacity = float(
            sum(mc.E_capacity for mc in self.mobile_chargers))
        self.metrics.mobile_charger_min_capacity = float(
            min(mc.E_capacity for mc in self.mobile_chargers))
        self.metrics.mobile_charger_max_capacity = float(
            max(mc.E_capacity for mc in self.mobile_chargers))
        self.metrics.mobile_charger_avg_travel_energy_cost_per_m = float(
            np.mean([float(getattr(mc, 'travel_energy_cost_per_m', 0.0))
                     for mc in self.mobile_chargers]))
        if self.sensor_initial_energy > 0:
            self.metrics.mobile_charger_total_sensor_equivalent_capacity = (
                self.metrics.mobile_charger_total_capacity / self.sensor_initial_energy)
            self.metrics.mobile_charger_min_sensor_equivalent_capacity = (
                self.metrics.mobile_charger_min_capacity / self.sensor_initial_energy)
            self.metrics.mobile_charger_max_sensor_equivalent_capacity = (
                self.metrics.mobile_charger_max_capacity / self.sensor_initial_energy)
        else:
            self.metrics.mobile_charger_total_sensor_equivalent_capacity = 0.0
            self.metrics.mobile_charger_min_sensor_equivalent_capacity = 0.0
            self.metrics.mobile_charger_max_sensor_equivalent_capacity = 0.0
        self.use_dqn_charger_routing = bool(
            self.ablation.get('use_dqn_charger_routing', False))
        self.charger_dqn = ChargerRoutingDQN(
            lr=float(self.ablation.get('dqn_learning_rate', 1.0e-3)),
            gamma=float(self.ablation.get('dqn_gamma', 0.80)),
            epsilon=float(self.ablation.get('dqn_epsilon', 0.05)),
            seed=int(self.ablation.get('dqn_seed', seed + 1000)))

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

    @staticmethod
    def _as_list(value, default=None):
        if value is None:
            value = [] if default is None else default
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return [] if default is None else list(default if isinstance(default, (list, tuple)) else [default])
            try:
                parsed = json.loads(text)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
                return [parsed]
            except Exception:
                return [v.strip() for v in text.split(',') if v.strip()]
        return [value]

    @staticmethod
    def _extend_to_count(values, count, cast=float):
        values = list(values or [])
        if not values:
            values = [cast(0)]
        out = []
        for idx in range(int(max(1, count))):
            v = values[idx] if idx < len(values) else values[-1]
            out.append(cast(v))
        return out

    def _resolve_mobile_charger_count(self, capacity_values):
        mode = str(self.ablation.get('mobile_charger_selection_mode',
                                     self.ablation.get('mc_selection_mode', 'hardcoded'))).strip().lower()
        explicit = self.ablation.get('num_mobile_chargers',
                                     self.ablation.get('mobile_charger_count', None))
        if mode in ('dynamic', 'auto', 'adaptive'):
            nodes_per_charger = int(
                max(1, int(self.ablation.get('mobile_charger_nodes_per_charger', 40))))
            count = int(math.ceil(max(1, self.num_nodes) /
                        float(nodes_per_charger)))
            count = max(int(self.ablation.get(
                'min_mobile_chargers', 1)), count)
            count = min(int(self.ablation.get(
                'max_mobile_chargers', 8)), count)
            return max(1, count)
        if explicit is not None:
            return max(1, int(explicit))
        # A capacity list by itself is treated as a hard-coded fleet definition.
        if len(capacity_values) > 1:
            return len(capacity_values)
        return 1

    def _resolve_mobile_charger_capacities(self, count, capacity_values):
        mode = str(self.ablation.get(
            'mobile_charger_capacity_mode', 'absolute')).strip().lower()
        relative_modes = {
            'relative', 'sensor_relative', 'sensor-equivalent',
            'sensor_equivalent', 'node_relative', 'scaled'
        }
        if mode in relative_modes:
            equiv_default = self.ablation.get(
                'mobile_charger_sensor_equivalent_capacity', 20.0)
            equiv_values = self._as_list(
                self.ablation.get(
                    'mobile_charger_sensor_equivalent_capacities', None),
                default=[equiv_default])
            equivalents = self._extend_to_count(
                equiv_values, count, cast=float)
            sensor_energy = max(float(self.sensor_initial_energy), 0.0)
            return [max(0.0, eq) * sensor_energy for eq in equivalents]

        return self._extend_to_count(capacity_values, count, cast=float)

    def _resolve_mobile_charger_travel_costs(self, count):
        mode = str(self.ablation.get(
            'mobile_charger_travel_cost_mode', 'absolute')).strip().lower()
        relative_modes = {'relative', 'sensor_relative',
                          'sensor_equivalent', 'scaled'}
        if mode in relative_modes:
            default_equiv = self.ablation.get(
                'mobile_charger_travel_sensor_equivalent_cost_per_m', 0.02)
            equiv_values = self._as_list(
                self.ablation.get(
                    'mobile_charger_travel_sensor_equivalent_costs_per_m', None),
                default=[default_equiv])
            equivalents = self._extend_to_count(
                equiv_values, count, cast=float)
            sensor_energy = max(float(self.sensor_initial_energy), 0.0)
            return [max(0.0, eq) * sensor_energy for eq in equivalents]

        default_cost = float(self.ablation.get(
            'mobile_charger_travel_energy_cost_per_m', 0.1))
        cost_values = self._as_list(
            self.ablation.get(
                'mobile_charger_travel_energy_costs_per_m', None),
            default=[default_cost])
        return self._extend_to_count(cost_values, count, cast=float)

    def _make_mobile_charger_fleet(self, bs_x, bs_y):
        default_capacity = float(self.ablation.get(
            'mobile_charger_capacity', 50000.0))
        capacity_values = self._as_list(
            self.ablation.get('mobile_charger_capacities', None),
            default=[default_capacity])
        count = self._resolve_mobile_charger_count(capacity_values)
        capacities = self._resolve_mobile_charger_capacities(
            count, capacity_values)
        travel_costs = self._resolve_mobile_charger_travel_costs(count)
        speed_values = self._as_list(
            self.ablation.get('mobile_charger_speeds', None),
            default=[float(self.ablation.get('mobile_charger_speed', 5.0))])
        speeds = self._extend_to_count(speed_values, count, cast=float)
        id_start = int(self.ablation.get('mobile_charger_id_start', 999))

        chargers = []
        for idx in range(count):
            charger = MobileCharger(
                mc_id=id_start + idx, depot_x=bs_x, depot_y=bs_y,
                capacity=capacities[idx], speed=speeds[idx],
                base_station=self.bs, metrics_collector=self.metrics)
            charger.travel_energy_cost_per_m = float(travel_costs[idx])
            chargers.append(charger)
        return chargers

    def _deploy_network(self, bs_x, bs_y):
        self.nodes[self.BS_ID] = SensorNode(
            self.BS_ID, bs_x, bs_y, float('inf'), self.metrics)
        initial_energy = float(self.sensor_initial_energy)
        for i in range(1, self.num_nodes + 1):
            x = random.uniform(0, self.network_size[0])
            y = random.uniform(0, self.network_size[1])
            self.nodes[i] = SensorNode(i, x, y, initial_energy, self.metrics)

        self.bs.network_nodes = self.nodes
        self.bs.mc_reference = self.mc
        self.bs.mc_references = self.mobile_chargers
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
                    'idle_sensing_energy_epoch': 0.0,
                    'energy_q_ewma': 0.0,
                    'energy_last_report_time': 0.0,
                    'energy_last_reported': node.E_current,
                    'adaptive_charge_threshold': 0.0,
                    'charge_satisfied': False,
                    'hop_count_to_bs_norm': 0.0
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

        for _nid, _table in initial_tables.items():
            if _nid in self.bs.node_states:
                _path = _table.get('primary_path') or []
                self.bs.node_states[_nid]['hop_count_to_bs_norm'] = min(
                    1.0, max(0, len(_path) - 1) / 10.0)

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

        self.bs.mc_shared_keys = {}
        for idx, charger in enumerate(self.mobile_chargers):
            charger.crypto_keys = f"MC-SECURE-KEY-{charger.mc_id}-{random.randint(1000, 9999)}"
            self.bs.mc_shared_keys[str(charger.mc_id)] = charger.crypto_keys
            if idx == 0:
                self.bs.mc_shared_key = charger.crypto_keys
            self.metrics.update_packet_transmission(
                is_critical=False, is_control=True, size_bytes=32, overhead_phase='setup')
        self.current_time += 0.5

        print(
            f"[Seed {self.seed}] Bootstrapping Complete. Temp trees cleared. Network Setup Time: {self.current_time}s")

    def _record_energy_sentinel_requests(self):
        """Add deadline-safe SOS requests for nodes that reached critical energy."""
        if not self.energy_experiment_active:
            return
        for nid, node in self.nodes.items():
            if nid == self.BS_ID or node.status != 'ALIVE':
                continue
            if node.E_current <= node.E_critical:
                self.metrics.record_energy_critical(nid, self.current_time)
                if self.ablation.get('use_deadline_safe_energy_sentinel', True):
                    st = self.bs.node_states.get(nid)
                    if st is not None:
                        actual_e = float(node.E_current)
                        st['E_est'] = min(
                            float(st.get('E_est', actual_e)), actual_e)
                        st['E_claimed'] = min(
                            float(st.get('E_claimed', actual_e)), actual_e)
                        st['energy_last_reported'] = min(
                            float(st.get('energy_last_reported', actual_e)), actual_e)
                        st['energy_last_report_time'] = float(
                            self.current_time)
                    self.bs.pending_sos_requests.add(nid)

    def service_mobile_charger(self):
        self._record_energy_sentinel_requests()
        if len(self.mobile_chargers) <= 1:
            return self._service_single_mobile_charger_legacy()
        return self._service_multi_mobile_chargers()

    def _service_single_mobile_charger_legacy(self):
        """Original single-MC service loop kept byte-for-byte in behavior."""
        if not self.bs.energy_experiment_active or self.mc.status != "IDLE":
            return

        target_node = None
        adaptive_mode = bool(self.ablation.get('use_adaptive_quantum_charging', False) or
                             self.ablation.get('use_adaptive_charge_guards', False) or
                             self.ablation.get('use_dqn_charger_routing', False))

        if adaptive_mode:
            if not self.bs.secure_charging_queue:
                self.bs.build_adaptive_charging_queue()
            valid_ids = [nid for nid in list(self.bs.secure_charging_queue)
                         if nid in self.nodes and self.nodes[nid].status == 'ALIVE'
                         and self.bs.validate_charging_request(nid, emergency=(nid in self.bs.pending_sos_requests))]
            self.bs.secure_charging_queue[:] = valid_ids
            candidates = [self.nodes[nid] for nid in valid_ids]
            if candidates:
                if self.use_dqn_charger_routing:
                    network_diag = (
                        self.network_size[0] ** 2 + self.network_size[1] ** 2) ** 0.5
                    target_node = self.charger_dqn.select(
                        self.mc.x, self.mc.y, candidates, self.bs.node_states,
                        speed=self.mc.speed, network_diag=network_diag)
                    self.metrics.dqn_routing_decisions += 1
                else:
                    def score(node):
                        return self.bs.charging_dispatch_priority(node.node_id, current_time=self.current_time, mc_x=self.mc.x, mc_y=self.mc.y)
                    target_node = min(candidates, key=score)
                if target_node is not None and target_node.node_id in self.bs.secure_charging_queue:
                    self.bs.secure_charging_queue.remove(target_node.node_id)
                if target_node is not None:
                    self.bs.pending_sos_requests.discard(target_node.node_id)
        else:
            secure_queue = self.bs.secure_charging_queue
            secure_queue[:] = [
                nid for nid in secure_queue if self.bs.validate_charging_request(nid, emergency=True)]

            if secure_queue:
                target_id = min(secure_queue, key=lambda nid: math.hypot(
                    self.mc.x - self.nodes[nid].x, self.mc.y - self.nodes[nid].y))
                secure_queue.remove(target_id)
                self.bs.pending_sos_requests.discard(target_id)
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

        if target_node:
            selected_emergency = (target_node.node_id in self.bs.pending_sos_requests or
                                  self.bs.is_deadline_rescue_node(target_node.node_id, self.current_time))
        else:
            selected_emergency = False

        duration = self._execute_charger_mission(
            self.mc, target_node, selected_emergency, self.current_time)
        if duration > 0:
            self.current_time += duration
            self.metrics.current_sim_time = self.current_time

    def _cluster_center(self, node_ids):
        coords = [(float(self.nodes[nid].x), float(self.nodes[nid].y))
                  for nid in node_ids if nid in self.nodes]
        if not coords:
            return (self.network_size[0] / 2.0, self.network_size[1] / 2.0)
        return (sum(x for x, _ in coords) / len(coords),
                sum(y for _, y in coords) / len(coords))

    def _assign_idle_chargers_to_clusters(self, idle_chargers, clusters):
        """Assign one idle charger to each spatial cluster to avoid same-area collisions."""
        assignments = []
        remaining = sorted(list(idle_chargers), key=lambda mc: int(mc.mc_id))
        cluster_items = [(cid, list(members)) for cid, members in sorted(
            (clusters or {}).items()) if members]
        for cid, members in cluster_items:
            if not remaining:
                break
            cx, cy = self._cluster_center(members)
            chosen = min(remaining, key=lambda mc: (math.hypot(
                float(mc.x) - cx, float(mc.y) - cy), int(mc.mc_id)))
            assignments.append((chosen, cid, members))
            remaining.remove(chosen)
        return assignments

    def _relative_mc_feasibility_guard_enabled(self):
        return (str(self.ablation.get('mobile_charger_capacity_mode', 'absolute')).strip().lower()
                in {'relative', 'sensor_relative', 'sensor-equivalent', 'sensor_equivalent', 'node_relative', 'scaled'}
                and bool(self.ablation.get('use_relative_mc_feasibility_guard', False)))

    def _charger_required_energy_for_mission(self, charger, target_node, include_min_charge=True):
        if target_node is None:
            return float('inf')
        distance = math.hypot(float(charger.x) - float(target_node.x),
                              float(charger.y) - float(target_node.y))
        travel_energy = distance * \
            float(getattr(charger, 'travel_energy_cost_per_m', 0.0))
        if not include_min_charge:
            return travel_energy
        min_ratio = float(self.ablation.get(
            'minimum_charge_energy_ratio', 0.08))
        useful_charge = min_ratio * \
            max(float(getattr(target_node, 'E_max', 0.0)), 0.0)

        useful_charge = min(useful_charge, max(0.0, float(
            target_node.E_max) - float(target_node.E_current)))
        return travel_energy + useful_charge

    def _charger_can_usefully_serve(self, charger, target_node):
        if not self._relative_mc_feasibility_guard_enabled():
            return True
        required = self._charger_required_energy_for_mission(
            charger, target_node, include_min_charge=True)
        reserve_ratio = float(self.ablation.get(
            'relative_mc_min_reserve_ratio', 0.02))
        reserve = reserve_ratio * \
            max(float(getattr(charger, 'E_capacity', 0.0)), 0.0)
        return float(getattr(charger, 'E_current', 0.0)) >= required + reserve

    def _return_to_depot_if_relative_mc_infeasible(self, charger, target_node):
        if not self._relative_mc_feasibility_guard_enabled() or target_node is None:
            return 0.0
        if self._charger_can_usefully_serve(charger, target_node):
            return 0.0

        return float(charger.return_to_depot())

    def _select_target_for_charger(self, charger, candidate_ids, assigned_nodes=None):
        assigned_nodes = set() if assigned_nodes is None else set(assigned_nodes)
        candidate_nodes = [self.nodes[nid] for nid in candidate_ids
                           if nid not in assigned_nodes and nid in self.nodes and self.nodes[nid].status == 'ALIVE']
        if not candidate_nodes:
            return None

        feasible_nodes = [node for node in candidate_nodes
                          if self._charger_can_usefully_serve(charger, node)]
        if feasible_nodes:
            candidate_nodes = feasible_nodes
        elif self._relative_mc_feasibility_guard_enabled():

            charger.return_to_depot()
            candidate_nodes = [node for node in candidate_nodes
                               if self._charger_can_usefully_serve(charger, node)]
            if not candidate_nodes:
                return None

        if self.use_dqn_charger_routing:
            network_diag = (
                self.network_size[0] ** 2 + self.network_size[1] ** 2) ** 0.5
            target_node = self.charger_dqn.select(
                charger.x, charger.y, candidate_nodes, self.bs.node_states,
                speed=charger.speed, network_diag=network_diag)
            self.metrics.dqn_routing_decisions += 1
            return target_node

        def score(node):
            return self.bs.charging_dispatch_priority(
                node.node_id, current_time=self.current_time,
                mc_x=charger.x, mc_y=charger.y, mc_speed=charger.speed)
        return min(candidate_nodes, key=score)

    def _execute_charger_mission(self, charger, target_node, selected_emergency=False, mission_start_time=None):
        if target_node is None:
            return 0.0
        mission_start_time = self.current_time if mission_start_time is None else float(
            mission_start_time)
        self.metrics.current_sim_time = mission_start_time
        if not (target_node.status == "ALIVE" and self.bs.validate_charging_request(target_node.node_id, emergency=selected_emergency)):
            return 0.0

        state_before = dict(self.bs.node_states.get(target_node.node_id, {}))
        network_diag = (self.network_size[0] **
                        2 + self.network_size[1] ** 2) ** 0.5
        feature_before = self.charger_dqn.feature_for(
            charger.x, charger.y, target_node, state_before, network_diag)
        before_x, before_y = charger.x, charger.y
        before_energy = float(target_node.E_current)
        depot_refill_time = self._return_to_depot_if_relative_mc_infeasible(
            charger, target_node)
        if self._relative_mc_feasibility_guard_enabled() and not self._charger_can_usefully_serve(charger, target_node):
            return depot_refill_time
        if depot_refill_time > 0:
            mission_start_time += depot_refill_time
            self.metrics.current_sim_time = mission_start_time
            before_x, before_y = charger.x, charger.y
        travel_time = charger.move_to_node(target_node)
        arrival_time = mission_start_time + travel_time
        self.metrics.current_sim_time = arrival_time
        charge_time = charger.charge_node(target_node)
        finish_time = arrival_time + charge_time
        after_energy = float(target_node.E_current)
        injected = max(0.0, after_energy - before_energy)
        e0 = max(float(target_node.E_max), 1e-12)
        if injected / e0 < float(self.ablation.get('minimum_charge_energy_ratio', 0.08)):
            self.metrics.mc_useless_charge_events += 1
        if self.use_dqn_charger_routing:
            dist = math.hypot(charger.x - before_x, charger.y - before_y)
            post_ratio = after_energy / e0
            # Reward: useful delivered energy and survival margin minus travel cost
            # and repeated tiny-service penalty.  This produces non-zero Bellman
            # loss and trains toward shorter, more useful missions.
            reward = (8.0 * injected / e0) + \
                (2.0 * post_ratio) - (0.025 * dist)
            if injected / e0 < float(self.ablation.get('minimum_charge_energy_ratio', 0.08)):
                reward -= 3.0
            next_candidates = [self.nodes[nid] for nid in self.bs.secure_charging_queue
                               if nid in self.nodes and self.nodes[nid].status == 'ALIVE']
            next_features = [self.charger_dqn.feature_for(
                charger.x, charger.y, n, self.bs.node_states.get(n.node_id, {}), network_diag) for n in next_candidates]
            loss = self.charger_dqn.train_on_transition(
                feature_before, reward, next_features)
            self.metrics.dqn_training_losses.append(float(loss))
        if charger.E_current < 0.1 * charger.E_capacity:
            charger.return_to_depot()
        return max(0.0, finish_time - mission_start_time)

    def _service_multi_mobile_chargers(self):
        idle_chargers = [mc for mc in self.mobile_chargers
                         if getattr(mc, 'status', 'IDLE') == 'IDLE'
                         and not getattr(mc, 'zeroized', False)
                         and not getattr(mc, 'requires_lab_reprovisioning', False)]
        if not self.bs.energy_experiment_active or not idle_chargers:
            return

        adaptive_mode = bool(self.ablation.get('use_adaptive_quantum_charging', False) or
                             self.ablation.get('use_adaptive_charge_guards', False) or
                             self.ablation.get('use_dqn_charger_routing', False))
        assignments = []
        assigned_nodes = set()

        if adaptive_mode:
            if not self.bs.secure_charging_queue:
                self.bs.build_adaptive_charging_queue(
                    idle_charger_count=len(idle_chargers))
            valid_ids = [nid for nid in list(self.bs.secure_charging_queue)
                         if nid in self.nodes and self.nodes[nid].status == 'ALIVE'
                         and self.bs.validate_charging_request(nid, emergency=(nid in self.bs.pending_sos_requests))]
            self.bs.secure_charging_queue[:] = valid_ids
            if not valid_ids:
                return
            clusters = getattr(self.bs, 'current_charging_clusters', {}) or {
                0: list(valid_ids)}
            clusters = {cid: [nid for nid in members if nid in valid_ids]
                        for cid, members in clusters.items()}
            cluster_assignments = self._assign_idle_chargers_to_clusters(
                idle_chargers, clusters)
            for charger, _cid, members in cluster_assignments:
                target_node = self._select_target_for_charger(
                    charger, members, assigned_nodes)
                if target_node is None:
                    continue
                assignments.append((charger, target_node))
                assigned_nodes.add(target_node.node_id)
                if target_node.node_id in self.bs.secure_charging_queue:
                    self.bs.secure_charging_queue.remove(target_node.node_id)
        else:
            secure_queue = self.bs.secure_charging_queue
            secure_queue[:] = [
                nid for nid in secure_queue if self.bs.validate_charging_request(nid, emergency=True)]
            candidate_ids = list(secure_queue)
            if not candidate_ids and len(self.bs.pending_sos_requests) > 0:
                candidate_ids = [nid for nid in self.bs.pending_sos_requests
                                 if nid in self.nodes and self.bs.validate_charging_request(nid, emergency=True)]
            if not candidate_ids:
                fallback = self._select_predictive_charging_target()
                candidate_ids = [] if fallback is None else [fallback.node_id]
            for charger in idle_chargers:
                target_node = self._select_target_for_charger(
                    charger, candidate_ids, assigned_nodes)
                if target_node is None:
                    continue
                assignments.append((charger, target_node))
                assigned_nodes.add(target_node.node_id)
                if target_node.node_id in secure_queue:
                    secure_queue.remove(target_node.node_id)

        if not assignments:
            return

        mission_start = float(self.current_time)
        mission_durations = []
        assigned_chargers = [charger for charger, _target in assignments]
        for charger in assigned_chargers:
            charger.status = "PARALLEL_ASSIGNED"
        for charger, target_node in assignments:
            selected_emergency = (target_node.node_id in self.bs.pending_sos_requests or
                                  self.bs.is_deadline_rescue_node(target_node.node_id, mission_start))
            self.bs.pending_sos_requests.discard(target_node.node_id)
            duration = self._execute_charger_mission(
                charger, target_node, selected_emergency, mission_start)

            charger.status = "PARALLEL_ASSIGNED"
            if duration > 0:
                mission_durations.append(duration)

        if mission_durations:
            self.current_time = mission_start + max(mission_durations)
            self.metrics.current_sim_time = self.current_time
        for charger in assigned_chargers:
            if getattr(charger, 'status', '') == "PARALLEL_ASSIGNED":
                charger.status = "IDLE"

        self.bs.mc_reference = self.mc
        self.bs.mc_references = self.mobile_chargers

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
                before_service_time = self.current_time
                self.service_mobile_charger()

                if self.current_time > before_service_time:
                    if next_event_time < self.current_time:
                        inter_arrival = random.expovariate(
                            float(self.ablation.get('event_rate_lambda', 0.5)))
                        next_event_time = self.current_time + inter_arrival
                    if next_epoch_time < self.current_time:
                        next_epoch_time = self.current_time + self.bs.T_epoch
                    self.metrics.current_sim_time = self.current_time

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
