import time
import math
import hmac
import secrets
from collections import defaultdict, deque
import numpy as np
import networkx as nx
from sklearn.metrics import silhouette_score

from config import compute_packet_hash, make_mc_mac, make_node_charge_receipt_mac, _source_energy, DATA_PACKET_BITS
from routing import BaselineRouting
from adaptive_charging import QuantumFuzzyChargingOracle, DynamicChargingDurationOracle


class ChargingCluster:
    """Spatial RF charging stop containing one or more sensor nodes."""

    def __init__(self, cluster_id, nodes, cx, cy, radius=0.0):
        self.is_cluster = True
        self.cluster_id = cluster_id
        self.nodes = list(nodes)
        self.centroid_x = float(cx)
        self.centroid_y = float(cy)
        self.radius = max(0.0, float(radius))
        member_ids = tuple(sorted(int(node.node_id) for node in self.nodes))
        self.member_ids = member_ids
        self.node_id = "Cluster_" + "_".join(map(str, member_ids))

    def recompute_geometry(self):
        if not self.nodes:
            self.centroid_x = 0.0
            self.centroid_y = 0.0
            self.radius = 0.0
            self.member_ids = tuple()
            self.node_id = f"Cluster_empty_{self.cluster_id}"
            return self
        self.centroid_x = float(np.mean([node.x for node in self.nodes]))
        self.centroid_y = float(np.mean([node.y for node in self.nodes]))
        self.radius = max(math.hypot(
            node.x - self.centroid_x, node.y - self.centroid_y)
            for node in self.nodes)
        self.member_ids = tuple(sorted(int(node.node_id)
                                for node in self.nodes))
        self.node_id = "Cluster_" + "_".join(map(str, self.member_ids))
        return self

    @classmethod
    def from_nodes(cls, cluster_id, nodes):
        cluster = cls(cluster_id, list(nodes), 0.0, 0.0, 0.0)
        return cluster.recompute_geometry()


class BaseStation:
    def __init__(self, metrics_collector, ablation_config, bs_id=0, routing_engine=None):
        self.BS_ID = bs_id
        self.metrics = metrics_collector
        self.ablation = ablation_config

        self.last_seq_per_source = {}
        self.sliding_window_size = 50
        self.packet_logs_per_source = defaultdict(list)

        self.network_nodes = {}
        self.node_states = {}
        self.routing_tables = {}
        # Enhanced DATA-plane routes are deliberately separate from the
        # original control/charging-plane tables.  Charging logic continues to
        # read self.routing_tables exactly as before.
        self.data_routing_tables = {}
        self.data_standby_routing_tables = None
        self.data_routing_engine = None
        self._last_data_route_signature = None

        # [INNOVATION] Initialize cluster_heads mapping
        self.cluster_heads = {}

        self.packet_logs = []
        self.dropped_packets_tracker = defaultdict(list)
        self.pending_sos_requests = set()

        self.secure_charging_queue = []
        self.secure_charging_node_ids = set()
        self.service_charging_node_ids = set()
        self.plus_charging_node_ids = set()

        self.experiment_family = self.ablation.get(
            'experiment_family', 'communication_security')
        self.energy_experiment_active = (
            self.experiment_family == 'energy_charging_stress'
            or self.ablation.get('enable_energy_experiment', False)
        )

        self.FCM_BLACKLIST_THRESHOLD = 0.8
        self.FCM_GREYLIST_THRESHOLD = 0.2
        self.T_epoch = 10.0

        self.eta_charging = float(self.ablation.get(
            'charging_energy_efficiency',
            self.ablation.get('rf_energy_conversion_efficiency', 0.70)))
        self.mc_recent_injected_energy = defaultdict(float)
        self.mc_real_injected_energy = defaultdict(float)
        self.quarantined_zones = []
        self.mc_reference = None
        self.mc_references = []
        self.mc_shared_key = None
        self.mc_shared_keys = {}

        self.env_window_size = 50
        self.threshold_k = 2.0
        self.environmental_history = defaultdict(
            lambda: deque(maxlen=self.env_window_size))
        self.default_threshold = 75.0
        self.zone_wakeup_tokens = {}

        self.T_CRITICAL_FE = 0.1
        self.T_WARNING_FE = 0.6
        self.standby_routing_tables = None
        self.current_dynamic_weights = np.array([0.25, 0.25, 0.15, 0.15, 0.20])

        self.risk_history = defaultdict(lambda: deque(maxlen=20))
        self.path_history = deque(maxlen=500)
        self.control_epoch_index = 0
        self.RISK_WATCHLIST = 0.35
        self.RISK_QUARANTINE = 0.65
        self.RISK_BLACKLIST_LCB = 0.85
        self.RISK_HYSTERESIS_EPOCHS = 3
        self.RISK_WARMUP_EPOCHS = 3

        self.RISK_WEIGHTS = {
            'fcm': 0.45, 'auth': 0.90, 'drop': 0.80, 'delay': 0.35,
            'traffic': 0.35, 'energy': 0.55, 'stealth': 0.95, 'sed': 0.80,
            'mc': 0.95, 'counterfactual': 0.85
        }

        self.tamper_history = defaultdict(lambda: deque(maxlen=5))
        self.TAMPER_PERSISTENCE_EPOCHS = 3
        self.TAMPER_MIN_PATH_EVIDENCE = 1

        self.CHARGE_MAX_DEVIATION_MC_NORM = 1.0
        self.CHARGE_MAX_SED = 3.0
        self.CHARGE_MIN_SPOOF_TRUST = 0.5
        self.CHARGE_NEED_RATIO = float(
            self.ablation.get('charge_need_ratio', 0.25))
        self.CHARGE_MAX_CLAIM_EST_DEVIATION = float(
            self.ablation.get('charge_claim_est_max_deviation', 0.30))

        period = self.ablation.get('route_recompute_period', None)
        self.ROUTE_RECOMPUTE_PERIOD = None if period in (
            None, False, 0, 'None', 'none') else int(period)
        self.routes_updated = False
        self._last_route_signature = None
        self._last_shadow_signature = None

        self.mc_last_counter = defaultdict(lambda: -1)
        self.mc_trust = defaultdict(lambda: 1.0)
        self.mc_violation_count = defaultdict(int)
        self.mc_consistent_report_streak = defaultdict(int)
        self.node_receipt_last_counter = defaultdict(lambda: -1)
        self.node_receipt_seen_nonces = defaultdict(lambda: deque(maxlen=2048))
        self.mc_seen_nonces = defaultdict(lambda: deque(
            maxlen=int(self.ablation.get('mc_nonce_cache_size', 2048))))
        self.sequence_replay_events = defaultdict(int)
        self.packet_auth_failures = defaultdict(int)
        self.mc_replay_rejections = 0
        self.mc_fail_closed_rejections = 0

        self.require_source_packet_hmac = bool(
            self.ablation.get('require_source_packet_hmac', True))
        self.received_seq_paths_by_type = defaultdict(
            lambda: defaultdict(lambda: defaultdict(set)))
        self.last_seq_per_source_by_type = defaultdict(dict)

        self.use_adaptive_quantum_charging = bool(
            self.ablation.get('use_adaptive_quantum_charging', False))

        self.use_hybrid_fixed_chargers = bool(
            self.ablation.get('use_hybrid_fixed_chargers', False))
        self.fixed_charger_nodes = set()

        self.use_rf_multi_node_charging = bool(
            self.ablation.get('use_rf_multi_node_charging', False))
        self.rf_charging_radius = float(
            self.ablation.get('rf_charging_radius', 25.0))

        self.dynamic_charging_oracle = DynamicChargingDurationOracle(
            w_wt=float(self.ablation.get('dynamic_charging_weight_wt', 1.0)),
            w_eps=float(self.ablation.get('dynamic_charging_weight_eps', 1.0)),
            w_now=float(self.ablation.get('dynamic_charging_weight_now', 1.0)))
        self._last_fixed_charge_time = None

        self.min_charge_energy_ratio = float(
            self.ablation.get('minimum_charge_energy_ratio', 0.08))
        self.post_charge_release_ratio = float(
            self.ablation.get('post_charge_release_ratio', 0.88))
        self.charging_reentry_cooldown_s = float(
            self.ablation.get('charging_reentry_cooldown_s', 180.0))
        self.qifl_threshold_lower_bound = float(
            self.ablation.get('qifl_threshold_lower_bound', 0.18))
        self.qifl_threshold_upper_bound = float(
            self.ablation.get('qifl_threshold_upper_bound', 0.52))
        adaptive_device = self.ablation.get('adaptive_compute_device')
        if adaptive_device is None:
            adaptive_device = (
                'auto' if self.ablation.get(
                    'use_gpu_for_adaptive_charging', True) else 'cpu')
        self.qifl_oracle = QuantumFuzzyChargingOracle(
            threshold_centers=self.ablation.get(
                'qifl_threshold_centers', (0.18, 0.26, 0.34, 0.42, 0.50)),
            lr=float(self.ablation.get('qifl_learning_rate', 2.5e-3)),
            device=str(adaptive_device),
            seed=int(self.ablation.get('qifl_seed', 42)))
        self.last_charge_time_by_node = defaultdict(lambda: -float('inf'))
        self.last_selected_charge_time_by_node = defaultdict(
            lambda: -float('inf'))
        self.last_charged_energy_ratio_by_node = defaultdict(float)
        self.last_service_start_by_node = defaultdict(lambda: -float('inf'))
        self.current_charging_clusters = {}

        self.use_travel_aware_admission = bool(
            self.ablation.get('use_travel_aware_admission', True))
        self.travel_admission_lambda = float(
            self.ablation.get('travel_admission_lambda', 2.0))
        self.travel_admission_gate = float(
            self.ablation.get('travel_admission_gate', 0.10))
        self.hard_deadline_margin_s = float(
            self.ablation.get('hard_deadline_margin_s', 90.0))

        if routing_engine is None:
            self.routing_engine = BaselineRouting(
                bs_id=self.BS_ID, transmission_range=40.0)
        else:
            self.routing_engine = routing_engine

    def _update_fixed_charger_assignments(self):
        for nid, node in self.network_nodes.items():
            if nid != self.BS_ID:
                node.has_fixed_charger = False
                if nid in self.node_states:
                    self.node_states[nid]['has_fixed_charger'] = False

        self.fixed_charger_nodes.clear()
        if not self.use_hybrid_fixed_chargers or not self.node_states:
            self.metrics.fixed_charger_active_nodes = 0
            return

        relay_load = defaultdict(int)
        for route in self.routing_tables.values():
            path = route.get('primary_path') or []
            for relay_id in path[1:-1]:
                relay_load[relay_id] += 1

        candidates = [
            nid for nid, state in self.node_states.items()
            if nid != self.BS_ID
            and nid in self.network_nodes
            and state.get('status', 'WHITELISTED') not in ('BLACKLISTED', 'QUARANTINE')
        ]
        if not candidates:
            self.metrics.fixed_charger_active_nodes = 0
            return

        eps_by_node = {
            nid: self._effective_depletion_rate(nid) for nid in candidates
        }
        relay_values = [float(relay_load.get(nid, 0)) for nid in candidates]
        eps_max = max(eps_by_node.values(), default=0.0)
        relay_max = max(relay_values, default=0.0)

        def bottleneck_score(nid):
            eps = eps_by_node[nid]
            eps_norm = eps / eps_max if eps_max > 0.0 else 0.0
            relay_norm = relay_load.get(
                nid, 0) / relay_max if relay_max > 0.0 else 0.0
            return (0.5 * eps_norm + 0.5 * relay_norm, relay_norm, eps_norm)

        requested_count = int(self.ablation.get(
            'fixed_charger_count', self.ablation.get('num_mobile_chargers', 1)))
        selected = sorted(candidates, key=bottleneck_score, reverse=True)[
            :max(0, requested_count)]
        self.fixed_charger_nodes.update(selected)
        for nid in selected:
            self.network_nodes[nid].has_fixed_charger = True
            self.node_states[nid]['has_fixed_charger'] = True
            self.node_states[nid]['fixed_charger_bottleneck_score'] = float(
                bottleneck_score(nid)[0])
        self.metrics.fixed_charger_active_nodes = len(selected)

    def apply_fixed_charging(self, current_time):
        now = float(current_time)
        if self._last_fixed_charge_time is None:
            self._last_fixed_charge_time = now
            return 0.0
        delta_t = max(0.0, now - self._last_fixed_charge_time)
        self._last_fixed_charge_time = now
        if not self.use_hybrid_fixed_chargers or delta_t <= 0.0:
            return 0.0

        trigger_ratio = float(self.ablation.get(
            'fixed_charger_trigger_ratio', self.ablation.get(
                'critical_energy_ratio', 0.15)))
        target_ratio = float(self.ablation.get(
            'fixed_charger_target_ratio', self.post_charge_release_ratio))
        charging_power = float(self.ablation.get('charging_rate', 5.0))
        total_injected = 0.0

        for nid in sorted(self.fixed_charger_nodes):
            node = self.network_nodes.get(nid)
            state = self.node_states.get(nid)
            if node is None or state is None:
                continue
            e_max = max(float(node.E_max), 1e-12)
            if float(node.E_current) / e_max > trigger_ratio:
                continue
            desired = max(0.0, target_ratio * e_max - float(node.E_current))
            injected = min(desired, max(0.0, charging_power * delta_t))
            if injected <= 0.0:
                continue
            actual = float(node.receive_fixed_charge(injected))
            if actual <= 0.0:
                continue
            total_injected += actual
            self.metrics.record_fixed_charge(nid, actual)
            state['E_est'] = min(e_max, float(
                state.get('E_est', 0.0)) + actual)
            state['total_injected_energy'] = float(
                state.get('total_injected_energy', 0.0)) + actual
            state['energy_injected_since_physical_sample'] = float(
                state.get('energy_injected_since_physical_sample', 0.0)) + actual
            state['E_claimed'] = max(
                float(state.get('E_claimed', 0.0)), state['E_est'])
            state['energy_last_reported'] = state['E_est']
            state['energy_last_report_time'] = now
            state['charge_satisfied'] = state['E_est'] / e_max >= target_ratio
            self.pending_sos_requests.discard(nid)
        return total_injected

    def record_environmental_observation(self, node_id, value):
        if value is not None:
            self.environmental_history[node_id].append(float(value))

    def _normalize_path_for_bs(self, path_taken):
        path = list(path_taken or [])
        if path and path[-1] != self.BS_ID:
            path.append(self.BS_ID)
        return path

    def _approved_critical_paths(self, source_id):
        routes = (self.data_routing_tables.get(source_id, {})
                  if self.data_routing_tables else
                  self.routing_tables.get(source_id, {})) or {}
        candidates = []
        for key in ('critical_paths', 'primary_path', 'secondary_path'):
            val = routes.get(key)
            if not val:
                continue
            if key == 'critical_paths':
                candidates.extend(val)
            else:
                candidates.append(val)
        approved = set()
        for p in candidates:
            p = tuple(self._normalize_path_for_bs(p))
            if len(p) >= 2 and p[0] == source_id and p[-1] == self.BS_ID:
                approved.add(p)
        return approved

    def _is_legitimate_critical_multipath_copy_v3(self, source_id, seq, packet_type, path_taken, packet):
        if packet.get('priority') != 'HIGH':
            return False
        if not self.ablation.get('use_critical_multipath_routing', True):
            return False
        path_tuple = tuple(self._normalize_path_for_bs(path_taken))
        if path_tuple not in self._approved_critical_paths(source_id):
            return False
        seen = self.received_seq_paths_by_type[source_id][packet_type][int(
            seq)]
        return path_tuple not in seen

    def _verify_source_hmac(self, packet, source_id):
        key = self.node_states.get(source_id, {}).get('crypto_keys')
        expected_hash = compute_packet_hash(packet, key)
        supplied_hash = packet.get('hash')
        valid = (expected_hash is not None and supplied_hash is not None and
                 hmac.compare_digest(str(supplied_hash), str(expected_hash)))
        packet['hash_valid'] = bool(valid)
        return bool(valid)

    def _log_auth_failure(self, source_id, packet, path_taken, receive_time):
        self.packet_auth_failures[source_id] += 1
        self.metrics.packet_hmac_failures += 1
        if source_id in self.node_states:
            st = self.node_states[source_id]
            st['packet_auth_failures'] = st.get('packet_auth_failures', 0) + 1
        delay = receive_time - packet.get('timestamp', receive_time)
        seq = packet.get('seq', -1)
        entry = {
            'source': source_id,
            'path': path_taken,
            'seq': seq,
            'delay': delay,
            'hash_valid': False,
            'priority': packet.get('priority', 'NORMAL')
        }
        self.packet_logs.append(entry)
        self.path_history.append({'path': tuple(path_taken), 'success': False,
                                  'priority': entry['priority'], 'auth_failure': True})

    def data_plane_receive(self, packet, path_taken, receive_time):
        source_id = packet['source_id']
        path_taken = self._normalize_path_for_bs(
            packet.get('path_header', path_taken))
        path_tuple = tuple(path_taken)
        packet_type = packet.get('type', 'DATA')
        priority = packet.get('priority', 'NORMAL')
        interrupt_triggered = False

        if self.ablation.get('require_source_packet_hmac', True):
            if not self._verify_source_hmac(packet, source_id):
                self._log_auth_failure(
                    source_id, packet, path_taken, receive_time)
                if self.ablation.get('use_asynchronous_processing_pipeline', True):
                    self.control_plane_epoch()
                return
        else:
            packet['hash_valid'] = True

        seq = packet.get('seq')
        if seq is None:
            self.sequence_replay_events[source_id] += 1
            self.metrics.source_sequence_replay_events += 1
            return
        seq = int(seq)

        seq_type = packet_type
        last_by_type = self.last_seq_per_source_by_type[source_id].get(
            seq_type)
        accepted_redundant_multipath_copy = False

        if last_by_type is not None:
            expected = last_by_type + 1
            if seq <= last_by_type:
                if seq == last_by_type and self._is_legitimate_critical_multipath_copy_v3(
                        source_id, seq, seq_type, path_taken, packet):
                    accepted_redundant_multipath_copy = True
                else:
                    self.sequence_replay_events[source_id] += 1
                    self.metrics.source_sequence_replay_events += 1
                    if source_id in self.node_states:
                        st = self.node_states[source_id]
                        st['sequence_replay_count'] = st.get(
                            'sequence_replay_count', 0) + 1
                        st['T1_history'] = max(0.0, float(
                            st.get('T1_history', 1.0)) * 0.90)
                        st['trust_vector'][0] = st['T1_history']
                    if self.ablation.get('use_asynchronous_processing_pipeline', True):
                        self.control_plane_epoch()
                    return
            elif seq > expected and packet_type == 'DATA':
                missing_seqs = list(range(expected, seq))
                self.dropped_packets_tracker[path_tuple].extend(missing_seqs)
                self.path_history.append({
                    'path': path_tuple,
                    'success': False,
                    'priority': priority,
                    'missing_count': len(missing_seqs),
                    'source': source_id
                })
                if seq > expected + int(self.ablation.get('sequence_gap_interrupt_threshold', 50)):
                    interrupt_triggered = True

        if not accepted_redundant_multipath_copy:
            self.last_seq_per_source_by_type[source_id][seq_type] = seq
            if packet_type == 'DATA':
                self.last_seq_per_source[source_id] = seq
        self.received_seq_paths_by_type[source_id][seq_type][seq].add(
            path_tuple)

        delay = receive_time - packet['timestamp']
        if delay > float(self.ablation.get('max_delay_tolerance', 5.0)):
            interrupt_triggered = True

        log_entry = {
            'source': source_id,
            'path': path_taken,
            'seq': seq,
            'delay': delay,
            'hash_valid': True,
            'priority': priority,
            'packet_type': packet_type,
            'redundant_multipath_copy': accepted_redundant_multipath_copy,
        }
        self.packet_logs_per_source[source_id].append(log_entry)
        if len(self.packet_logs_per_source[source_id]) > self.sliding_window_size:
            self.packet_logs_per_source[source_id].pop(0)
        self.packet_logs.append(log_entry)
        self.path_history.append(
            {'path': path_tuple, 'success': True, 'priority': priority})

        src_energy = _source_energy(packet)
        if src_energy is not None and source_id in self.node_states:
            self.node_states[source_id]['E_claimed'] = src_energy
            self._observe_energy_report(source_id, src_energy, receive_time)

        if packet_type == 'SOS_CHARGE':
            self.pending_sos_requests.add(source_id)
        elif packet_type == 'DATA':
            payload = packet.get('payload')
            if payload is not None:
                self.record_environmental_observation(source_id, payload)

        if interrupt_triggered and self.ablation.get('use_asynchronous_processing_pipeline', True):
            self.control_plane_epoch()

    def _now(self):
        return float(getattr(self.metrics, 'current_sim_time', 0.0))

    def _node_energy_or_topology_impaired(self, node_id):
        if node_id == self.BS_ID:
            return False
        node = self.network_nodes.get(node_id)
        if node is None:
            return True
        if getattr(node, 'status', 'ALIVE') != 'ALIVE':
            return True
        critical = float(getattr(node, 'E_critical', 0.0))
        return float(getattr(node, 'E_current', 0.0)) <= critical

    def _path_has_energy_or_topology_failure(self, path):
        path = list(path or [])
        if len(path) < 2:
            return True
        tx_range = float(getattr(self.routing_engine, 'transmission_range',
                                 self.ablation.get('transmission_range', 40.0)))
        for nid in path[:-1]:
            if self._node_energy_or_topology_impaired(nid):
                return True
        for a, b in zip(path[:-1], path[1:]):
            na = self.network_nodes.get(a)
            nb = self.network_nodes.get(b)
            if na is None or nb is None:
                return True
            if math.hypot(float(na.x) - float(nb.x),
                          float(na.y) - float(nb.y)) > tx_range + 1e-9:
                return True
        return False

    def _update_physical_depletion_estimates(self):
        now = self._now()
        beta = float(np.clip(self.ablation.get(
            'physical_eps_ewma_beta', 0.70), 0.0, 0.9999))
        for nid, state in self.node_states.items():
            if nid == self.BS_ID:
                continue
            node = self.network_nodes.get(nid)
            if node is None:
                continue
            current = max(0.0, float(getattr(node, 'E_current', 0.0)))
            last_t = state.get('physical_energy_sample_time')
            last_e = state.get('physical_energy_sample')
            injected = max(0.0, float(
                state.get('energy_injected_since_physical_sample', 0.0)))
            if last_t is not None and last_e is not None:
                dt = max(1e-9, now - float(last_t))
                consumed = max(0.0, float(last_e) + injected - current)
                q_inst = consumed / dt
                previous = float(state.get('physical_eps_ewma', q_inst))
                state['physical_eps_ewma'] = (
                    beta * previous + (1.0 - beta) * q_inst)
                state['physical_eps_last'] = float(q_inst)
            else:
                state.setdefault('physical_eps_ewma', 0.0)
            state['physical_energy_sample'] = current
            state['physical_energy_sample_time'] = now
            state['energy_injected_since_physical_sample'] = 0.0

    def _effective_depletion_rate(self, node_id, current_time=None):
        state = self.node_states.get(node_id, {})
        node = self.network_nodes.get(node_id)
        now = self._now() if current_time is None else float(current_time)
        idle = max(0.0, float(self.ablation.get(
            'idle_sensing_power_j_per_s', 0.0)))
        q_report = max(0.0, float(state.get('energy_q_ewma', 0.0)))
        q_physical = max(0.0, float(state.get('physical_eps_ewma', 0.0)))
        elapsed = max(now, float(self.T_epoch), 1.0)
        q_cumulative = max(0.0, float(state.get(
            'E_consumption_est_cumulative', 0.0))) / elapsed

        base = max(idle, q_report, q_physical, q_cumulative)
        hop_pressure = min(1.0, max(0.0, float(
            state.get('hop_count_to_bs_norm', 0.0))))
        disagreement = abs(q_physical - q_report) / max(base, 1e-12)
        uncertainty = min(0.35, 0.10 * disagreement)
        factor = max(1.0, float(self.ablation.get(
            'depletion_rate_safety_factor', 1.20)))
        factor *= 1.0 + 0.15 * hop_pressure + uncertainty
        effective = base * factor

        e_max = max(float(getattr(node, 'E_max', state.get(
            'E_initial', 1.0))), 1e-12)
        upper = e_max / max(float(self.ablation.get(
            'minimum_depletion_lifetime_s', 30.0)), 1.0)
        effective = float(np.clip(effective, idle, upper))
        state['effective_depletion_rate'] = effective
        return effective

    def _calculate_hybrid_weights(self, valid_nodes):
        if not self.ablation.get('use_shannon_entropy_weights', True):
            return np.array([0.25, 0.25, 0.15, 0.15, 0.20])

        W_AHP = np.array([0.25, 0.25, 0.15, 0.15, 0.20])
        N = len(valid_nodes)
        if N <= 1:
            return W_AHP

        T_matrix = np.array([self.node_states[nid]['trust_vector']
                            for nid in valid_nodes])
        col_sums = T_matrix.sum(axis=0)
        col_sums_safe = np.where(col_sums == 0, 1e-10, col_sums)
        P_matrix = T_matrix / col_sums_safe
        P_safe_for_log = np.where(P_matrix == 0, 1e-10, P_matrix)
        E = (-1.0 / math.log(N)) * \
            np.sum(P_matrix * np.log(P_safe_for_log), axis=0)
        D = 1.0 - E
        D_sum = np.sum(D)

        if D_sum == 0:
            W_Shannon = np.ones(5) / 5.0
        else:
            W_Shannon = D / D_sum

        W_Hybrid_raw = W_AHP * W_Shannon
        W_Hybrid = W_Hybrid_raw / np.sum(W_Hybrid_raw)
        return W_Hybrid

    def _compute_fitness_value(self):
        valid_nodes = [n for n in self.node_states if n != self.BS_ID]
        dynamic_weights = self._calculate_hybrid_weights(valid_nodes)
        self.current_dynamic_weights = dynamic_weights
        try:
            self.metrics.dynamic_trust_weights_history.append(
                np.asarray(dynamic_weights, dtype=float).copy())
        except Exception:
            pass

        use_security_veto = self.ablation.get(
            'use_security_veto_aggregation', True)
        for nid in valid_nodes:
            state = self.node_states[nid]
            if state.get('status') == 'BLACKLISTED':
                state['FE'] = 0.0
                continue
            trust_vector = state['trust_vector']
            weighted_sum = float(np.dot(dynamic_weights, trust_vector))
            veto_multiplier = min(
                trust_vector[0], trust_vector[1]) if use_security_veto else 1.0
            state['FE'] = weighted_sum * veto_multiplier

    def control_plane_epoch(self):
        old_h = self.RISK_HYSTERESIS_EPOCHS
        old_w = self.RISK_WARMUP_EPOCHS
        if not self.ablation.get('use_risk_history_hysteresis', True):
            self.RISK_HYSTERESIS_EPOCHS = 1
            self.RISK_WARMUP_EPOCHS = 0

        try:
            start_time = time.time()
            self.control_epoch_index += 1

            self._update_physical_depletion_estimates()
            self._update_fixed_charger_assignments()
            self._verify_wrsn_cyber_physical_security()
            self._two_tier_trust_matrix()
            self._verify_stealthy_attackers()
            self._verify_trajectory_manipulators()

            self._compute_fitness_value()

            U_malicious_dict = self._fcm_classification()
            self._calibrated_risk_fusion(U_malicious_dict)

            # [INNOVATION] Network clustering mapping for routing and chargers
            self._joint_optimization(U_malicious_dict)

            self.packet_logs.clear()
            self.dropped_packets_tracker.clear()

            exec_time = (time.time() - start_time) * 1000
            self.metrics.record_execution_time('CORE', exec_time)

            if self.ablation.get('use_fuzzy_validity_metrics', True):
                try:
                    node_ids = [
                        nid for nid in self.node_states.keys() if nid != self.BS_ID]
                    if len(node_ids) >= 2:
                        X = np.array([self.node_states[nid].get('trust_vector', [
                                     1, 1, 1, 1, 1]) for nid in node_ids], dtype=float)
                        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

                        def _membership(state):
                            base_risk = state.get(
                                'risk_mean', state.get('fcm_risk', 0.0))
                            try:
                                u_mal = float(base_risk)
                            except Exception:
                                u_mal = 0.0
                            status = state.get('status', 'WHITELISTED')
                            if state.get('hard_veto', False) or status == 'BLACKLISTED':
                                u_mal = max(u_mal, 0.999)
                            elif status == 'QUARANTINE':
                                u_mal = max(u_mal, 0.75)
                            elif status in ('WATCHLIST', 'GREYLISTED'):
                                u_mal = max(u_mal, 0.50)
                            return min(1.0 - 1e-9, max(1e-9, u_mal))

                        u_mal = np.array(
                            [_membership(self.node_states[nid]) for nid in node_ids], dtype=float)
                        U = np.column_stack((1.0 - u_mal, u_mal))

                        N = X.shape[0]
                        C = 2
                        m = 2.0
                        U_m = U ** m
                        denom = np.maximum(U_m.sum(axis=0)[:, None], 1e-12)
                        centers = (U_m.T @ X) / denom

                        diff = X[:, None, :] - centers[None, :, :]
                        dist_sq = np.sum(diff * diff, axis=2)
                        objective = float(np.sum(U_m * dist_sq))

                        fpc = float(np.sum(U ** 2) / N)
                        fpe = float(-np.sum(U * np.log(U + 1e-12)) / N)

                        center_dist_sq = float(
                            np.sum((centers[0] - centers[1]) ** 2))
                        xb = float(
                            objective / (N * max(center_dist_sq, 1e-12)))

                        labels = np.argmax(U, axis=1)
                        if len(set(labels.tolist())) > 1 and N > 2:
                            try:
                                if N > 250:
                                    sil = float(silhouette_score(
                                        X, labels, sample_size=250, random_state=42))
                                else:
                                    sil = float(silhouette_score(X, labels))
                            except Exception:
                                sil = 0.0
                        else:
                            sil = 0.0

                        if hasattr(self.metrics, 'fuzzy_objective_values'):
                            if self.metrics.fuzzy_objective_values:
                                self.metrics.fuzzy_objective_values[-1] = [
                                    float(objective)]
                            else:
                                self.metrics.fuzzy_objective_values.append(
                                    [float(objective)])
                            if self.metrics.fuzzy_silhouette_scores:
                                self.metrics.fuzzy_silhouette_scores[-1] = float(
                                    sil)
                            else:
                                self.metrics.fuzzy_silhouette_scores.append(
                                    float(sil))
                            if self.metrics.fuzzy_partition_coefficients:
                                self.metrics.fuzzy_partition_coefficients[-1] = float(
                                    fpc)
                            else:
                                self.metrics.fuzzy_partition_coefficients.append(
                                    float(fpc))
                            if self.metrics.fuzzy_partition_entropies:
                                self.metrics.fuzzy_partition_entropies[-1] = float(
                                    fpe)
                            else:
                                self.metrics.fuzzy_partition_entropies.append(
                                    float(fpe))
                            if self.metrics.xie_beni_indices:
                                self.metrics.xie_beni_indices[-1] = float(xb)
                            else:
                                self.metrics.xie_beni_indices.append(float(xb))

                except Exception as exc:
                    if self.ablation.get('debug_fuzzy_metrics', False):
                        print(
                            f'[FUZZY-METRICS] Phase-1 validity calculation skipped: {exc}')

        finally:
            self.RISK_HYSTERESIS_EPOCHS = old_h
            self.RISK_WARMUP_EPOCHS = old_w

    def _two_tier_trust_matrix(self):
        nodes = list(self.node_states.keys())
        if not nodes:
            return

        E_elec = float(self.ablation.get('radio_e_elec_j_per_bit', 50e-9))
        eps_fs = float(self.ablation.get('radio_eps_fs_j_per_bit_m2', 10e-12))
        eps_mp = max(float(self.ablation.get(
            'radio_eps_mp_j_per_bit_m4', 0.0013e-12)), 1e-30)
        d0 = math.sqrt(eps_fs / eps_mp)
        packet_bits = float(self.ablation.get(
            'data_packet_bits', DATA_PACKET_BITS))
        ack_bits = float(self.ablation.get('ack_packet_bits', 16 * 8))

        tx_energy_consumed = defaultdict(float)
        rx_energy_consumed = defaultdict(float)
        ack_energy_consumed = defaultdict(float)

        def _tx_energy(bits, distance):
            if distance <= d0:
                amp_cost = eps_fs * (distance ** 2)
            else:
                amp_cost = eps_mp * (distance ** 4)
            return bits * (E_elec + amp_cost)

        for log in self.packet_logs:
            path = log['path']
            for i in range(len(path) - 1):
                sender = path[i]
                receiver = path[i+1]
                if receiver != self.BS_ID:
                    rx_energy_consumed[receiver] += packet_bits * E_elec
                    n_receiver = self.network_nodes[receiver]
                    n_sender = self.network_nodes[sender]
                    ack_dist = math.hypot(
                        n_receiver.x - n_sender.x, n_receiver.y - n_sender.y)
                    ack_energy_consumed[receiver] += _tx_energy(
                        ack_bits, ack_dist)
                if sender != self.BS_ID:
                    n_sender = self.network_nodes[sender]
                    n_receiver = self.network_nodes[receiver]
                    dist = math.hypot(n_sender.x - n_receiver.x,
                                      n_sender.y - n_receiver.y)
                    tx_energy_consumed[sender] += _tx_energy(packet_bits, dist)

        security_packet_logs = [
            log for log in self.packet_logs
            if not self._path_has_energy_or_topology_failure(log.get('path', []))
        ]

        path_success_counts = defaultdict(int)
        path_tamper_counts = defaultdict(int)

        for log in security_packet_logs:
            path_tuple = tuple(log['path'])
            if log['hash_valid']:
                path_success_counts[path_tuple] += 1

        for log in self.packet_logs:
            if not log.get('hash_valid', True):
                path_tamper_counts[tuple(log.get('path', []))] += 1

        path_drop_counts = {
            tuple(path): len(drops)
            for path, drops in self.dropped_packets_tracker.items()
            if len(drops) > 0
            and not self._path_has_energy_or_topology_failure(path)
        }
        node_drop_participation = defaultdict(int)
        for p, d_count in path_drop_counts.items():
            for nid in p[1:-1]:
                node_drop_participation[nid] += d_count

        culprit_drop_per_path = {}
        for p in path_drop_counts.keys():
            internal = p[1:-1]
            culprit_drop_per_path[p] = (
                max(internal, key=node_drop_participation.__getitem__)
                if internal else None)

        for p, d_count in path_drop_counts.items():
            culprit = culprit_drop_per_path.get(p)
            for idx in range(1, len(p) - 1):
                nid = p[idx]
                prev_id = p[idx - 1]
                if nid == self.BS_ID:
                    continue
                rx_energy_consumed[nid] += d_count * packet_bits * E_elec
                if prev_id in self.network_nodes and nid in self.network_nodes:
                    prev_node = self.network_nodes[prev_id]
                    curr_node = self.network_nodes[nid]
                    ack_dist = math.hypot(
                        curr_node.x - prev_node.x, curr_node.y - prev_node.y)
                    ack_energy_consumed[nid] += d_count * \
                        _tx_energy(ack_bits, ack_dist)
                if nid == culprit:
                    break
                next_id = p[idx + 1]
                if next_id in self.network_nodes and nid in self.network_nodes:
                    curr_node = self.network_nodes[nid]
                    next_node = self.network_nodes[next_id]
                    tx_dist = math.hypot(
                        curr_node.x - next_node.x, curr_node.y - next_node.y)
                    tx_energy_consumed[nid] += d_count * \
                        _tx_energy(packet_bits, tx_dist)

        node_tamper_participation = defaultdict(int)
        for p, t_count in path_tamper_counts.items():
            for nid in p[1:-1]:
                node_tamper_participation[nid] += t_count

        culprit_tamper_per_path = {}
        for p in path_tamper_counts.keys():
            internal = p[1:-1]
            culprit_tamper_per_path[p] = (
                max(internal, key=node_tamper_participation.__getitem__)
                if internal else None)

        identified_tamperers = set(culprit_tamper_per_path.values())
        identified_tamperers.discard(None)

        for nid in nodes:
            if nid == self.BS_ID:
                continue
            has_epoch_tamper_evidence = (
                nid in identified_tamperers
                and node_tamper_participation.get(nid, 0) >= self.TAMPER_MIN_PATH_EVIDENCE
            )
            hist = self.tamper_history[nid]
            hist.append(1 if has_epoch_tamper_evidence else 0)
            if sum(hist) >= self.TAMPER_PERSISTENCE_EPOCHS:
                state = self.node_states[nid]
                print(
                    f"[WRSN SECURITY] Persistent cryptographic tamper evidence at Node {nid}. Hard veto triggered.")
                state['hard_veto'] = True
                state['status'] = 'BLACKLISTED'
                state['FE'] = 0.0

        N_recv = {nid: 0 for nid in nodes}
        N_fwd = {nid: 0 for nid in nodes}

        for p, count in path_success_counts.items():
            for nid in p[1:-1]:
                N_recv[nid] += count
                N_fwd[nid] += count

        for p, count in path_tamper_counts.items():
            for nid in p[1:-1]:
                N_recv[nid] += count
                N_fwd[nid] += count

        for p, d_count in path_drop_counts.items():
            culprit = culprit_drop_per_path.get(p)
            for nid in p[1:-1]:
                N_recv[nid] += d_count
                if nid != culprit:
                    N_fwd[nid] += d_count
                else:
                    break

        alpha_reward = 0.9
        alpha_penalty = 0.2
        beta = 0.8

        old_t2 = {nid: (self.node_states[nid].get('T2_history', 1.0), self.node_states[nid].get(
            'trust_vector', [1, 1, 1, 1, 1])[1]) for nid in nodes if nid != self.BS_ID}
        old_t3 = {nid: (self.node_states[nid].get('T3_history', 1.0), self.node_states[nid].get(
            'trust_vector', [1, 1, 1, 1, 1])[2]) for nid in nodes if nid != self.BS_ID}

        for nid in nodes:
            if nid == self.BS_ID:
                continue
            state = self.node_states[nid]
            t1_prev = state.get('T1_history', 1.0)
            t2_prev = state.get('T2_history', 1.0)

            participated = N_recv[nid] > 0
            if nid in identified_tamperers:
                t1_new = alpha_penalty * t1_prev
            else:
                if participated:
                    t1_new = alpha_reward * t1_prev + \
                        (1.0 - alpha_reward) * 1.0
                else:
                    t1_new = t1_prev

            if self._node_energy_or_topology_impaired(nid):
                t2_new = t2_prev
            elif N_recv[nid] > 0:
                pdr = N_fwd[nid] / N_recv[nid]
                t2_new = beta * t2_prev + (1.0 - beta) * pdr
            else:
                t2_new = t2_prev

            state['T1_history'] = t1_new
            state['T2_history'] = t2_new
            state['trust_vector'][0] = t1_new
            state['trust_vector'][1] = t2_new

        delay_logs = security_packet_logs
        P = len(delay_logs)
        N = len(nodes)

        if P > 0 and N > 0:
            node_to_idx = {nid: idx for idx, nid in enumerate(nodes)}
            A = np.zeros((P, N))
            Y = np.zeros(P)
            for p_idx, log in enumerate(delay_logs):
                path = log['path']
                Y[p_idx] = log['delay']
                for nid in path[1:-1]:
                    if nid in node_to_idx:
                        A[p_idx, node_to_idx[nid]] = 1.0

            try:
                X_d_avg, residuals, rank, s = np.linalg.lstsq(A, Y, rcond=None)
            except np.linalg.LinAlgError:
                X_d_avg = np.zeros(N)

            gamma = 0.8
            lambda_penalty = 5.0
            d_prop = 0.01
            d_proc = 0.05
            queue_load = defaultdict(int)
            for log in delay_logs:
                for nid in log['path'][1:-1]:
                    queue_load[nid] += 1

            for nid in nodes:
                if nid == self.BS_ID:
                    continue
                if self._node_energy_or_topology_impaired(nid):
                    continue
                idx = node_to_idx[nid]
                d_avg_i = X_d_avg[idx]
                hop_approx = math.ceil(math.hypot(self.node_states[nid]['x'] - self.network_nodes[self.BS_ID].x,
                                                  self.node_states[nid]['y'] - self.network_nodes[self.BS_ID].y) / 30.0)
                D_expected_i = (hop_approx * d_prop) + \
                    (queue_load[nid] * d_proc)
                excess_delay = max(0.0, d_avg_i - D_expected_i)
                t3_current = math.exp(-lambda_penalty * excess_delay)
                t3_prev = self.node_states[nid].get('T3_history', 1.0)
                t3_new = gamma * t3_prev + (1.0 - gamma) * t3_current
                self.node_states[nid]['T3_history'] = t3_new
                self.node_states[nid]['trust_vector'][2] = t3_new

        if not self.ablation.get('use_intersection_logic_for_droppers', True):
            for nid, vals in old_t2.items():
                if nid in self.node_states:
                    self.node_states[nid]['T2_history'] = vals[0]
                    self.node_states[nid]['trust_vector'][1] = vals[1]
        if not self.ablation.get('use_network_tomography_for_delay', True):
            for nid, vals in old_t3.items():
                if nid in self.node_states:
                    self.node_states[nid]['T3_history'] = vals[0]
                    self.node_states[nid]['trust_vector'][2] = vals[1]

        U_actual = defaultdict(int)
        U_expected = defaultdict(float)
        for log in security_packet_logs:
            path = log['path']
            for nid in path:
                if nid != self.BS_ID:
                    U_actual[nid] += 1

        G_primary = nx.DiGraph()
        for nid, routes in self.routing_tables.items():
            primary_hop = routes.get('primary')
            if primary_hop is not None:
                G_primary.add_edge(nid, primary_hop)

        def get_expected_full_path(start_node, route_type):
            expected_path = [start_node]
            curr = start_node
            visited = set([start_node])
            while True:
                r = self.routing_tables.get(curr, {})
                nxt = r.get(route_type) if curr == start_node else r.get(
                    'primary')
                if nxt is None or nxt == self.BS_ID or nxt in visited:
                    break
                expected_path.append(nxt)
                visited.add(nxt)
                curr = nxt
            return expected_path

        for nid in nodes:
            if nid == self.BS_ID:
                continue
            state = self.node_states[nid]
            curr_seq = self.last_seq_per_source.get(nid, 0)
            prev_seq = state.get('prev_epoch_seq', 0)
            delta_seq = max(0, curr_seq - prev_seq)
            state['prev_epoch_seq'] = curr_seq

            crit_seq_set = {log['seq'] for log in security_packet_logs if log['source'] == nid and log.get(
                'priority') == 'HIGH'}
            crit_count = len(crit_seq_set)
            normal_gen_count = max(0, delta_seq - crit_count)

            prev_lambda = state.get('lambda_gen', 0.0)
            omega_lambda = 0.8
            current_lambda = normal_gen_count / self.T_epoch
            new_lambda = omega_lambda * prev_lambda + \
                (1.0 - omega_lambda) * current_lambda
            state['lambda_gen'] = new_lambda
            expected_normal_gen = new_lambda * self.T_epoch

            try:
                path_to_bs = list(nx.descendants(G_primary, nid)) + [nid]
                for hop in path_to_bs:
                    U_expected[hop] += expected_normal_gen
            except nx.NetworkXError:
                pass

            if crit_count > 0:
                p1_path = get_expected_full_path(nid, 'primary')
                routes = self.routing_tables.get(nid, {})
                p2_hop = routes.get('secondary')
                p2_path = get_expected_full_path(
                    nid, 'secondary') if p2_hop and p2_hop != routes.get('primary') else []
                for hop in set(p1_path + p2_path):
                    U_expected[hop] += crit_count

        theta_t4 = 0.7
        kappa = 2.0
        epsilon = 1.0
        for nid in nodes:
            if nid == self.BS_ID:
                continue
            if self._node_energy_or_topology_impaired(nid):
                continue
            u_act = U_actual.get(nid, 0)
            u_exp = U_expected.get(nid, 0.0)
            deviation = abs(u_act - u_exp) / max(u_exp, epsilon)
            t4_current = math.exp(-kappa * deviation)
            t4_prev = self.node_states[nid].get('T4_history', 1.0)
            t4_new = theta_t4 * t4_prev + (1.0 - theta_t4) * t4_current
            self.node_states[nid]['T4_history'] = t4_new
            self.node_states[nid]['trust_vector'][3] = t4_new

        rho_t5 = 0.8
        nu_t5 = 5.0
        for nid in nodes:
            if nid == self.BS_ID:
                continue
            state = self.node_states[nid]
            E_max = state['E_initial']
            E_claimed = state.get('E_claimed', E_max)

            epoch_consumption = (
                tx_energy_consumed.get(nid, 0.0)
                + rx_energy_consumed.get(nid, 0.0)
                + ack_energy_consumed.get(nid, 0.0)
                + state.get('idle_sensing_energy_epoch', 0.0)
            )
            cumulative_consumption = state.get(
                'E_consumption_est_cumulative', 0.0) + epoch_consumption
            state['E_consumption_est_cumulative'] = cumulative_consumption
            state['idle_sensing_energy_epoch'] = 0.0

            total_injected = state.get('total_injected_energy', 0.0)
            E_est = E_max - cumulative_consumption + total_injected
            E_est = max(0.0, min(E_max, E_est))
            state['E_est'] = E_est

            deviation = abs(E_claimed - E_est) / max(E_max, 1e-12)
            penalty_factor = math.exp(-nu_t5 * deviation)
            spoof_trust = state.get('spoof_trust', 1.0)
            t5_current = penalty_factor * spoof_trust

            t5_prev = state.get('T5_history', 1.0)
            t5_new = rho_t5 * t5_prev + (1.0 - rho_t5) * t5_current
            state['T5_history'] = t5_new
            state['trust_vector'][4] = t5_new

    def _fcm_classification(self):
        if not self.ablation.get('use_fcm_classification', True):
            out = {}
            for nid, state in self.node_states.items():
                if nid == self.BS_ID:
                    continue
                tv = np.array(state.get('trust_vector', [
                              1, 1, 1, 1, 1]), dtype=float)
                risk = float(np.clip(1.0 - np.mean(tv), 0.0, 1.0))
                state['fcm_risk'] = risk
                out[nid] = risk
            return out

        all_nodes = list(self.node_states.keys())
        nodes = [n for n in all_nodes if n != self.BS_ID and self.node_states[n].get(
            'status') != 'BLACKLISTED']
        U_malicious_dict = {nid: 1.0 if self.node_states[nid].get(
            'status') == 'BLACKLISTED' else 0.0 for nid in all_nodes}

        N = len(nodes)
        if N < 2:
            for nid in nodes:
                self.node_states[nid]['fcm_risk'] = 0.0
            return U_malicious_dict

        X = np.array([self.node_states[n]['trust_vector']
                     for n in nodes], dtype=float)
        C = 2
        m = 2.0
        max_iter = 50

        np.random.seed(42)
        U = np.random.dirichlet(np.ones(C), size=N)
        centers = np.zeros((C, X.shape[1]))

        for _ in range(max_iter):
            U_m = U ** m
            centers = (U_m.T @ X) / np.maximum(U_m.sum(axis=0)[:, None], 1e-12)
            distances = np.zeros((N, C))
            for i in range(N):
                for j in range(C):
                    distances[i, j] = np.linalg.norm(X[i] - centers[j]) + 1e-10

            U_new = np.zeros((N, C))
            for i in range(N):
                for j in range(C):
                    denom = sum(
                        (distances[i, j] / distances[i, k]) ** (2 / (m - 1)) for k in range(C))
                    U_new[i, j] = 1.0 / max(denom, 1e-12)
            if np.linalg.norm(U_new - U) < 0.001:
                U = U_new
                break
            U = U_new

        malicious_cluster_idx = 0 if np.linalg.norm(
            centers[0]) < np.linalg.norm(centers[1]) else 1

        for i, nid in enumerate(nodes):
            u_mal = float(U[i, malicious_cluster_idx])
            U_malicious_dict[nid] = u_mal
            self.node_states[nid]['fcm_risk'] = u_mal

        return U_malicious_dict

    def _compute_counterfactual_path_attribution(self):
        if not self.ablation.get('use_counterfactual_path_attribution', True):
            return {nid: 0.0 for nid in self.node_states.keys()}

        attribution = {nid: 0.0 for nid in self.node_states.keys()}
        records = [
            r for r in self.path_history
            if not self._path_has_energy_or_topology_failure(r.get('path', ()))
        ]
        if len(records) < 5:
            return attribution

        for nid in self.node_states.keys():
            if nid == self.BS_ID:
                continue
            contains = [r for r in records if nid in r.get(
                'path', ()) and nid != self.BS_ID]
            excludes = [r for r in records if nid not in r.get('path', ())]
            if len(contains) < 2 or len(excludes) < 2:
                continue
            fail_contains = sum(1 for r in contains if not r.get(
                'success', True)) / len(contains)
            fail_excludes = sum(1 for r in excludes if not r.get(
                'success', True)) / len(excludes)
            attribution[nid] = max(0.0, fail_contains - fail_excludes)
        return attribution

    def _has_explicit_security_evidence(self, node_id, state=None):
        state = self.node_states.get(node_id, {}) if state is None else state
        return bool(
            state.get('hard_veto', False)
            or int(state.get('packet_auth_failures', 0)) > 0
            or int(state.get('sequence_replay_count', 0)) > 0
            or bool(state.get('confirmed_sed_attack', False))
            or bool(state.get('confirmed_stealth_attack', False))
            or float(state.get('Deviation_MC_norm', 0.0)) > self.CHARGE_MAX_DEVIATION_MC_NORM
            or float(state.get('spoof_trust', 1.0)) < self.CHARGE_MIN_SPOOF_TRUST
        )

    def _calibrated_risk_fusion(self, U_malicious_dict):
        if not self.ablation.get('use_uncertainty_calibrated_risk', True):
            for nid, state in self.node_states.items():
                if nid == self.BS_ID:
                    continue
                if state.get('hard_veto', False):
                    state['status'] = 'BLACKLISTED'
                    state['FE'] = 0.0
                    continue
                risk = float(U_malicious_dict.get(
                    nid, state.get('fcm_risk', 0.0)))
                state['risk_instant'] = risk
                state['risk_mean'] = risk
                state['risk_lcb'] = risk
                state['risk_uncertainty'] = 0.0
                if risk >= 0.85:
                    state['status'] = 'BLACKLISTED'
                    state['FE'] = 0.0
                elif risk >= 0.65:
                    state['status'] = 'QUARANTINE'
                elif risk >= 0.35:
                    state['status'] = 'WATCHLIST'
                else:
                    state['status'] = 'WHITELISTED'
            return

        cf_attr = self._compute_counterfactual_path_attribution()

        for nid, state in self.node_states.items():
            if nid == self.BS_ID:
                continue

            if state.get('hard_veto', False):
                state['status'] = 'BLACKLISTED'
                state['FE'] = 0.0
                continue

            T1, T2, T3, T4, T5 = state.get('trust_vector', [1, 1, 1, 1, 1])
            risks = {
                'fcm': float(U_malicious_dict.get(nid, state.get('fcm_risk', 0.0))),
                'auth': max(0.0, 1.0 - float(T1)),
                'drop': max(0.0, 1.0 - float(T2)),
                'delay': max(0.0, 1.0 - float(T3)),
                'traffic': max(0.0, 1.0 - float(T4)),
                'energy': max(0.0, 1.0 - float(T5)),
                'stealth': min(1.0, max(0.0, float(state.get('S_stealth', 0.0)))),
                'sed': min(1.0, max(0.0, float(state.get('SED_index', 0.0)) / 3.0)),
                'mc': min(1.0, max(0.0, float(state.get('Deviation_MC_norm', 0.0)))),
                'counterfactual': min(1.0, max(0.0, float(cf_attr.get(nid, 0.0))))
            }

            impaired = self._node_energy_or_topology_impaired(nid)
            explicit_evidence = self._has_explicit_security_evidence(
                nid, state)
            if impaired:
                for key in ('drop', 'delay', 'traffic', 'counterfactual', 'energy'):
                    risks[key] = 0.0
                explicit_risk = max(
                    risks['auth'], risks['stealth'], risks['sed'], risks['mc'])
                risks['fcm'] = min(risks['fcm'], explicit_risk)

            survival = 1.0
            for key, risk_value in risks.items():
                survival *= (1.0 -
                             min(1.0, self.RISK_WEIGHTS[key] * risk_value))
            instantaneous_risk = 1.0 - survival

            hist = self.risk_history[nid]
            hist.append(float(instantaneous_risk))
            arr = np.array(hist, dtype=float)
            risk_mean = float(np.mean(arr))
            risk_std = float(np.std(arr)) if len(arr) > 1 else 0.25
            risk_lcb = max(0.0, risk_mean - 1.96 * risk_std /
                           math.sqrt(max(len(arr), 1)))
            risk_ucb = min(1.0, risk_mean + 1.96 * risk_std /
                           math.sqrt(max(len(arr), 1)))
            uncertainty = max(0.0, risk_ucb - risk_lcb)

            state['risk_components'] = risks
            state['risk_instant'] = float(instantaneous_risk)
            state['risk_mean'] = risk_mean
            state['risk_lcb'] = risk_lcb
            state['risk_uncertainty'] = uncertainty

            if self.energy_experiment_active and impaired and not explicit_evidence:
                state['blacklist_counter'] = 0
                state['status'] = (
                    'WATCHLIST' if risk_mean >= self.RISK_WATCHLIST
                    else 'WHITELISTED')
                continue

            if (self.control_epoch_index > self.RISK_WARMUP_EPOCHS and
                    risk_lcb >= self.RISK_BLACKLIST_LCB):
                state['blacklist_counter'] = state.get(
                    'blacklist_counter', 0) + 1
            else:
                state['blacklist_counter'] = max(
                    0, state.get('blacklist_counter', 0) - 1)

            if state.get('blacklist_counter', 0) >= self.RISK_HYSTERESIS_EPOCHS:
                state['status'] = 'BLACKLISTED'
                state['FE'] = 0.0
            elif risk_mean >= self.RISK_QUARANTINE:
                state['status'] = 'QUARANTINE'
            elif risk_mean >= self.RISK_WATCHLIST:
                state['status'] = 'WATCHLIST'
            else:
                state['status'] = 'WHITELISTED'

    def _network_clustering(self):
        """
        [INNOVATION] K-Means removed! Clustering is now handled natively by the 
        Hybrid Routing Engine (IFBE+FCBE) inside routing.py.
        This method now extracts the active Cluster Heads and Clusters directly 
        from the computed routing tables to feed the Mobile Charger urgency logic.
        """
        clusters = defaultdict(list)
        self.cluster_heads = {}

        if not self.routing_tables:
            return clusters

        for nid, table in self.routing_tables.items():
            if nid == self.BS_ID:
                continue

            primary_hop = table.get('primary')
            if primary_hop is not None and primary_hop != self.BS_ID:
                clusters[primary_hop].append(nid)
                self.cluster_heads[primary_hop] = primary_hop

        for ch in self.cluster_heads:
            if ch not in clusters[ch]:
                clusters[ch].append(ch)

        try:
            k_opt_val = len(self.cluster_heads)
            if hasattr(self.metrics, 'adaptive_k_opts'):
                self.metrics.adaptive_k_opts.append(k_opt_val)
            else:
                self.metrics.adaptive_k_opts = [k_opt_val]
        except Exception:
            pass

        return clusters

    def _joint_optimization(self, U_malicious_dict):
        """Update reference and enhanced DATA routes on separate planes.

        ``self.routing_tables`` remains the original control/charging routing
        substrate.  ``self.data_routing_tables`` is the only table modified by
        the clustered/MST overlay.  Consequently QIFL, DQN, charging admission,
        CH urgency, SOS thresholds, fixed-charger placement, and charger travel
        decisions retain their original inputs and parameters.
        """
        node_penalties = {}
        active_nodes = {}
        shadow_penalties = {}
        shadow_active_nodes = {}
        needs_shadow_calc = False

        if self.BS_ID in self.network_nodes:
            active_nodes[self.BS_ID] = self.network_nodes[self.BS_ID]
            shadow_active_nodes[self.BS_ID] = self.network_nodes[self.BS_ID]
            node_penalties[self.BS_ID] = 1.0
            shadow_penalties[self.BS_ID] = 1.0

        critical_nodes = []
        for nid, state in self.node_states.items():
            if nid == self.BS_ID:
                continue
            FE_score = max(state.get('FE', 1.0), 1e-6)
            risk = float(state.get('risk_mean', 0.0))
            uncertainty = float(state.get('risk_uncertainty', 0.0))
            status = state.get('status', 'WHITELISTED')
            base_penalty = (1.0 + 3.0 * risk + 1.0 * uncertainty) / FE_score
            node_obj = self.network_nodes.get(nid)
            if node_obj is None:
                continue
            shadow_active_nodes[nid] = node_obj

            if status == 'BLACKLISTED' or FE_score <= self.T_CRITICAL_FE:
                node_penalties[nid] = float('inf')
                shadow_penalties[nid] = float('inf')
                critical_nodes.append(nid)
                continue

            active_nodes[nid] = node_obj
            if status == 'QUARANTINE':
                node_penalties[nid] = base_penalty * 6.0
            elif status in ('WATCHLIST', 'GREYLISTED'):
                node_penalties[nid] = base_penalty * 1.75
            else:
                node_penalties[nid] = base_penalty

            if FE_score <= self.T_WARNING_FE or status in ('WATCHLIST', 'QUARANTINE'):
                shadow_penalties[nid] = float('inf')
                needs_shadow_calc = True
            else:
                shadow_penalties[nid] = node_penalties[nid]

        route_signature = tuple(sorted(
            nid for nid in self.node_states
            if nid != self.BS_ID
            and self.node_states[nid].get('status') == 'BLACKLISTED'))
        warning_set = tuple(sorted(
            nid for nid in self.node_states
            if nid != self.BS_ID and (
                self.node_states[nid].get('FE', 1.0) <= self.T_WARNING_FE
                or self.node_states[nid].get('status') in ('WATCHLIST', 'QUARANTINE'))))
        shadow_signature = warning_set
        force_periodic_refresh = (
            self.ROUTE_RECOMPUTE_PERIOD is not None
            and self.control_epoch_index % self.ROUTE_RECOMPUTE_PERIOD == 0)
        active_changed = route_signature != self._last_route_signature
        shadow_changed = shadow_signature != self._last_shadow_signature
        self.routes_updated = False

        # DATA routing has its own refresh cadence.  This does not alter the
        # original reference route-recompute period used by charging/control.
        data_period = self.ablation.get('data_route_recompute_period', 5)
        try:
            data_period = max(1, int(data_period))
        except (TypeError, ValueError):
            data_period = 5
        force_data_refresh = self.control_epoch_index % data_period == 0

        if critical_nodes and self.standby_routing_tables is not None:
            _shadow_activation_t0 = time.time()
            self.routing_tables = self.standby_routing_tables
            self.standby_routing_tables = None
            if self.data_standby_routing_tables is not None:
                self.data_routing_tables = self.data_standby_routing_tables
                self.data_standby_routing_tables = None
            self._last_route_signature = route_signature
            self.routes_updated = True
            self.metrics.record_shadow_route_result(
                hit=True,
                shadow_ms=(time.time() - _shadow_activation_t0) * 1000.0)
        elif active_changed or force_periodic_refresh or not self.routing_tables:
            _ordinary_route_t0 = time.time()
            self.routing_tables = self.routing_engine.compute_routing_tables(
                active_nodes, node_penalties)
            self._last_route_signature = route_signature
            self.routes_updated = True
            self.metrics.record_shadow_route_result(
                ordinary_ms=(time.time() - _ordinary_route_t0) * 1000.0)

        # Recompute enhanced DATA routes without changing any charging table.
        if self.data_routing_engine is not None and (
                self.routes_updated or force_data_refresh
                or not self.data_routing_tables):
            self.data_routing_tables = self.data_routing_engine.compute_routing_tables(
                active_nodes, node_penalties,
                reference_tables=self.routing_tables)
            self._last_data_route_signature = (
                route_signature, self.control_epoch_index // data_period)
            try:
                diag = dict(getattr(
                    self.data_routing_engine, 'last_diagnostics', {}) or {})
                self.metrics.record_data_routing_diagnostics(diag)
            except Exception:
                pass

        if needs_shadow_calc:
            if shadow_changed or force_periodic_refresh or self.standby_routing_tables is None:
                _shadow_pre_t0 = time.time()
                self.standby_routing_tables = self.routing_engine.compute_routing_tables(
                    shadow_active_nodes, shadow_penalties)
                if self.data_routing_engine is not None:
                    self.data_standby_routing_tables = (
                        self.data_routing_engine.compute_routing_tables(
                            shadow_active_nodes, shadow_penalties,
                            reference_tables=self.standby_routing_tables))
                self._last_shadow_signature = shadow_signature
                self.metrics.record_shadow_route_result(
                    precomputed=True,
                    extra_compute_ms=(time.time() - _shadow_pre_t0) * 1000.0)
        else:
            self.standby_routing_tables = None
            self.data_standby_routing_tables = None
            self._last_shadow_signature = None

        # Important: cluster_heads used by charging are still extracted from
        # self.routing_tables (the original reference plane), not DATA routes.
        self._network_clustering()

    def _verify_wrsn_cyber_physical_security(self):
        if not self.ablation.get('use_zero_overhead_spoof_detection', True):
            for nid in list(self.mc_recent_injected_energy.keys()):
                self.mc_recent_injected_energy[nid] = 0.0
            return

        for nid, state in self.node_states.items():
            if nid == self.BS_ID or state['status'] == 'BLACKLISTED':
                continue

            E_max = state['E_initial']
            E_claimed_t = state.get('E_claimed', E_max)
            E_claimed_t_minus_1 = state.get('E_claimed_prev', E_max)
            E_injected = self.mc_recent_injected_energy.get(nid, 0.0)

            delta_claimed = E_claimed_t - E_claimed_t_minus_1
            deviation_mc = abs(delta_claimed - E_injected)

            tolerance = 0.05 * E_max
            state['Deviation_MC'] = deviation_mc
            state['Deviation_MC_norm'] = min(
                1.0, deviation_mc / max(tolerance, 1e-12))

            legitimate_credit = bool(
                state.get('legitimate_mc_credit_pending', False))
            if deviation_mc > (2.0 * tolerance):
                if E_injected > 0 and delta_claimed <= tolerance and not legitimate_credit:
                    self.metrics.charge_hoarder_attempts += 1
                    self.metrics.charge_hoarder_detections += 1
                    print(
                        f"[WRSN SECURITY] Charge Hoarding detected at Node {nid}! Hard veto triggered.")
                    state['hard_veto'] = True
                    state['hard_veto_reason'] = 'energy_report_mismatch'
                    state['hard_veto_recoverable'] = True
                    state['energy_probation'] = True
                    state['energy_recovery_streak'] = 0
                    state['energy_recovery_reference_energy'] = float(state.get('E_est', E_claimed_t))
                    state['energy_recovery_reference_time'] = float(self._now())
                    state['energy_recovery_last_epoch'] = None
                    state['last_node_charge_receipt_consistent'] = False
                    state['status'] = 'BLACKLISTED'
                    state['FE'] = 0.0
                elif (E_injected == 0 and delta_claimed > tolerance and
                      not legitimate_credit and nid not in self.fixed_charger_nodes):
                    rise_ratio = delta_claimed / max(E_max, 1e-12)
                    count = int(
                        state.get('unexplained_energy_rise_count', 0)) + 1
                    state['unexplained_energy_rise_count'] = count
                    confirmed = rise_ratio >= 0.30 or count >= 3
                    if confirmed:
                        self.metrics.fake_charger_attempts += 1
                        self.metrics.fake_charger_detections += 1
                        print(
                            f"[WRSN SECURITY] Confirmed fake-charger energy rise at Node {nid}.")
                        state['hard_veto'] = True
                        state['hard_veto_reason'] = 'energy_report_mismatch'
                        state['hard_veto_recoverable'] = True
                        state['energy_probation'] = True
                        state['energy_recovery_streak'] = 0
                        state['energy_recovery_reference_energy'] = float(state.get('E_est', E_claimed_t))
                        state['energy_recovery_reference_time'] = float(self._now())
                        state['energy_recovery_last_epoch'] = None
                        state['last_node_charge_receipt_consistent'] = False
                        state['status'] = 'BLACKLISTED'
                        state['FE'] = 0.0
                    else:
                        state['status'] = 'WATCHLIST'
                else:
                    state['unexplained_energy_rise_count'] = max(
                        0, int(state.get('unexplained_energy_rise_count', 0)) - 1)
            else:
                state['unexplained_energy_rise_count'] = max(
                    0, int(state.get('unexplained_energy_rise_count', 0)) - 1)

            t2_current = state.get('T2_history', 1.0)
            already_rescued = state.get('spoof_rescue_triggered', False)

            if t2_current > 0.8:
                state['spoof_rescue_triggered'] = False

            if (t2_current < 0.5 and delta_claimed <= 0 and
                    E_injected == 0 and not already_rescued and
                    not self._node_energy_or_topology_impaired(nid)):
                print(
                    f"[WRSN SECURITY] Spoofing Attack detected affecting Node {nid}! Triggering Incident Response.")
                self.trigger_incident_response(
                    victim_id=nid, spoof_x=state['x'], spoof_y=state['y'])
                state['spoof_rescue_triggered'] = True

            state['E_claimed_prev'] = E_claimed_t
            state['legitimate_mc_credit_pending'] = False
            self.mc_recent_injected_energy[nid] = 0.0

    def _verify_stealthy_attackers(self):
        if not self.ablation.get('use_stealthy_attacker_detection', True):
            return
        kappa = 5.0
        veto_threshold = 0.4
        for nid, state in self.node_states.items():
            if nid == self.BS_ID or state.get('status') == 'BLACKLISTED':
                continue
            if self._node_energy_or_topology_impaired(nid):
                state['S_stealth'] = 0.0
                state['confirmed_stealth_attack'] = False
                continue
            t2 = float(state.get('T2_history', 1.0))
            e_est = float(state.get('E_est', state['E_initial']))
            e_claimed = float(state.get('E_claimed', state['E_initial']))
            e_max = max(float(state['E_initial']), 1e-12)
            drop_rate = 1.0 - t2
            energy_diff = max(0.0, e_est - e_claimed)
            alibi_penalty = 1.0 - math.exp(-kappa * (energy_diff / e_max))
            score = drop_rate * alibi_penalty
            state['S_stealth'] = score

            corroborated = bool(
                int(state.get('packet_auth_failures', 0)) > 0
                or int(state.get('sequence_replay_count', 0)) > 0
                or float(state.get('Deviation_MC_norm', 0.0)) >
                self.CHARGE_MAX_DEVIATION_MC_NORM
                or float(state.get('spoof_trust', 1.0)) <
                self.CHARGE_MIN_SPOOF_TRUST)
            state['confirmed_stealth_attack'] = bool(
                score >= veto_threshold and corroborated)
            if state['confirmed_stealth_attack']:
                print(
                    f"[WRSN SECURITY] Corroborated stealth attacker at Node {nid}; S={score:.3f}.")
                state['hard_veto'] = True
                state['status'] = 'BLACKLISTED'
                state['FE'] = 0.0
            elif score >= veto_threshold:
                state['status'] = 'WATCHLIST'

    def _verify_trajectory_manipulators(self):
        if not self.ablation.get(
                'use_sed_filter',
                self.ablation.get('use_zero_overhead_spoof_detection', True)):
            return
        bs_x = self.network_nodes[self.BS_ID].x
        bs_y = self.network_nodes[self.BS_ID].y
        non_bs_states = [state for nid, state in self.node_states.items()
                         if nid != self.BS_ID]
        if not non_bs_states:
            return
        d_max = max(math.hypot(state['x'] - bs_x, state['y'] - bs_y)
                    for state in non_bs_states)
        d_max = max(d_max, 1e-6)
        epsilon = 1e-6

        for nid, state in self.node_states.items():
            if nid == self.BS_ID:
                continue
            e_prev = state.get('E_claimed_prev_SED', state['E_initial'])
            e_curr = state.get('E_claimed', state['E_initial'])
            state['delta_E'] = max(0.0, e_prev - e_curr)
            state['E_claimed_prev_SED'] = e_curr
            state['confirmed_sed_attack'] = False

        for nid in list(self.pending_sos_requests):
            state = self.node_states.get(nid)
            if not state or state.get('status') == 'BLACKLISTED':
                self.pending_sos_requests.discard(nid)
                continue
            if self._node_energy_or_topology_impaired(nid):
                state['SED_index'] = 0.0
                continue

            delta_i = float(state.get('delta_E', 0.0))
            neighbor_deltas = []
            for other_nid, other_state in self.node_states.items():
                if other_nid in (nid, self.BS_ID):
                    continue
                dist = math.hypot(
                    state['x'] - other_state['x'],
                    state['y'] - other_state['y'])
                if dist <= 30.0 and not self._node_energy_or_topology_impaired(other_nid):
                    neighbor_deltas.append(float(
                        other_state.get('delta_E', 0.0)))

            mu = float(np.mean(neighbor_deltas)) if neighbor_deltas else 0.0
            sigma = float(np.std(neighbor_deltas)) if neighbor_deltas else 0.0
            d_bs = math.hypot(state['x'] - bs_x, state['y'] - bs_y)
            z_score = abs(delta_i - mu) / (sigma + epsilon)
            sed_raw = z_score * (1.0 + d_bs / d_max)

            corroborated = bool(
                int(state.get('packet_auth_failures', 0)) > 0
                or int(state.get('sequence_replay_count', 0)) > 0
                or float(state.get('Deviation_MC_norm', 0.0)) >
                self.CHARGE_MAX_DEVIATION_MC_NORM
                or float(state.get('spoof_trust', 1.0)) <
                self.CHARGE_MIN_SPOOF_TRUST)
            confirmed = sed_raw >= 3.0 and corroborated
            state['confirmed_sed_attack'] = bool(confirmed)
            state['SED_index'] = float(
                sed_raw if confirmed else min(sed_raw, 1.5))

            if confirmed:
                print(
                    f"[WRSN SECURITY] Corroborated trajectory manipulation at Node {nid}; SED={sed_raw:.2f}.")
                state['hard_veto'] = True
                state['status'] = 'BLACKLISTED'
                state['FE'] = 0.0
                self.pending_sos_requests.discard(nid)
            elif sed_raw >= 3.0:
                state['status'] = 'WATCHLIST'

    def get_blacklisted_nodes(self):
        return [nid for nid, state in self.node_states.items() if state.get('status') == 'BLACKLISTED']

    def get_adaptive_threshold_for_node(self, node_id):
        if not self.ablation.get('use_adaptive_edge_threshold', True):
            return self.default_threshold
        history = self.environmental_history.get(node_id, [])
        if len(history) < 5:
            return self.default_threshold
        mu = float(np.mean(history))
        sigma = float(np.std(history))
        return mu + self.threshold_k * sigma

    def estimate_service_latency_s(self, node_id, current_time=None, charger=None):
        now = self._now() if current_time is None else float(current_time)
        travel_time, _ = self._charger_travel_time_distance(
            node_id, charger=charger)
        fleet_size = max(
            1, len(self._active_mobile_chargers(require_idle=False)))

        recent_durations = list(
            getattr(self.metrics, 'mc_service_durations', [])[-50:])
        default_mission = float(self.ablation.get(
            'adaptive_expected_mission_time_s', 30.0))
        avg_mission = (float(np.mean(recent_durations))
                       if recent_durations else default_mission)
        avg_mission = max(1.0, avg_mission)

        if getattr(self.metrics, 'rf_cluster_charge_events', 0) > 0:
            avg_cluster_size = self.metrics._safe_rate(
                self.metrics.rf_total_nodes_charged_in_clusters,
                self.metrics.rf_cluster_charge_events)
        else:
            avg_cluster_size = float(self.ablation.get(
                'expected_nodes_per_charge_stop', 2.0))
        avg_cluster_size = max(1.0, avg_cluster_size)

        service_ids = set(getattr(self, 'service_charging_node_ids', set()))
        sos_ids = set(getattr(self, 'pending_sos_requests', set()))
        queue_nodes = max(len(service_ids | sos_ids), 1)
        max_batches = max(1, int(self.ablation.get(
            'max_service_latency_batches', 6)))
        batches_ahead = max(0, int(math.ceil(
            max(0, queue_nodes - 1) / (fleet_size * avg_cluster_size))))
        batches_ahead = min(batches_ahead, max_batches)
        queue_time = batches_ahead * avg_mission

        state = self.node_states.get(node_id, {})
        node = self.network_nodes.get(node_id)
        e_max = max(
            float(getattr(node, 'E_max', state.get('E_initial', 1.0))), 1e-12)
        e_now = max(0.0, float(
            getattr(node, 'E_current', state.get('E_est', e_max))))
        target_ratio = float(self.ablation.get(
            'post_charge_release_ratio', 0.88))
        gap = max(0.0, target_ratio * e_max - e_now)
        charge_power = max(
            float(self.ablation.get('charging_rate', 5.0)), 1e-12)
        if self.use_rf_multi_node_charging:
            radius = max(float(self.ablation.get(
                'rf_charging_radius', 25.0)), 0.0)
            d0 = max(float(self.ablation.get(
                'rf_reference_distance_m', 1.0)), 1e-6)
            alpha = max(float(self.ablation.get(
                'rf_path_loss_exponent', 2.0)), 0.1)
            eta = float(np.clip(self.ablation.get(
                'rf_energy_conversion_efficiency', 0.70), 0.0, 1.0))
            charge_power = max(
                charge_power * eta / (1.0 + (radius / d0) ** alpha),
                float(self.ablation.get('rf_min_effective_receive_power_w', 1e-4)))
        charge_time = gap / max(charge_power, 1e-12)
        margin = max(0.0, float(self.ablation.get(
            'service_latency_safety_margin_s',
            self.ablation.get('hard_deadline_margin_s', 90.0))))
        latency = queue_time + max(0.0, travel_time) + charge_time + margin
        state['estimated_service_latency_s'] = float(latency)
        state['estimated_queue_wait_s'] = float(queue_time)
        state['estimated_service_at_time'] = float(now)
        return float(latency)

    def _observe_energy_report(self, node_id, reported_energy, receive_time):
        state = self.node_states.get(node_id)
        if state is None:
            return
        now = float(receive_time)
        reported_energy = float(max(0.0, reported_energy))
        last_t = state.get('energy_last_report_time')
        last_e = state.get('energy_last_reported')
        if last_t is not None and last_e is not None:
            dt = max(1e-9, now - float(last_t))
            q_inst = max(0.0, (float(last_e) - reported_energy) / dt)
            beta = float(self.ablation.get('energy_ewma_beta',
                         self.ablation.get('ewma_beta', 0.70)))
            q_prev = float(state.get('energy_q_ewma', q_inst))
            state['energy_q_ewma'] = beta * q_inst + (1.0 - beta) * q_prev
            pred_before_report = state.get('E_est', reported_energy)
            try:
                self.metrics.ewma_energy_abs_errors.append(
                    abs(float(pred_before_report) - reported_energy))
            except Exception:
                pass
        else:
            state.setdefault('energy_q_ewma', 0.0)
        # Preserve the independent prediction before the claim overwrites E_est.
        # Recovery must be based on this reference, not merely on accidental
        # proximity between a forged constant claim and the physical battery.
        prediction_before_update = float(state.get('E_est', reported_energy))
        state['energy_last_report_time'] = now
        state['energy_last_reported'] = reported_energy
        state['E_est'] = max(
            0.0, min(float(state.get('E_initial', reported_energy)), reported_energy))

        if (self.ablation.get('enable_energy_trust_recovery', True)
                and (state.get('energy_probation', False)
                     or state.get('hard_veto_recoverable', False))):
            E0 = max(float(state.get('E_initial', 1.0)), 1e-12)
            tolerance = float(self.ablation.get(
                'energy_trust_recovery_tolerance_ratio', 0.01)) * E0
            min_t1 = float(self.ablation.get(
                'energy_trust_recovery_min_auth_trust', 0.95))
            min_t2 = float(self.ablation.get(
                'energy_trust_recovery_min_forwarding_trust', 0.90))
            t1 = float(state.get('T1_history', state.get('trust_vector', [1.0])[0]))
            t2 = float(state.get('T2_history', state.get('trust_vector', [1.0, 1.0])[1]))

            reference_energy = float(state.get(
                'energy_recovery_reference_energy', prediction_before_update))
            reference_time = float(state.get(
                'energy_recovery_reference_time', now))
            q_ref = max(0.0, float(state.get(
                'physical_eps_ewma', state.get('energy_q_ewma', 0.0))))
            reference_now = max(
                0.0, reference_energy - q_ref * max(0.0, now - reference_time))

            # A recovery is accepted only once per control epoch, requires
            # trustworthy authentication/forwarding behaviour, agreement with
            # an independent BS prediction, and at least one consistent signed
            # charging receipt after the energy-mismatch incident. This blocks
            # malicious constant low-energy claims from accidental recovery.
            current_epoch = int(getattr(self, 'control_epoch_index', 0))
            new_epoch = state.get('energy_recovery_last_epoch') != current_epoch
            receipt_ok = state.get('last_node_charge_receipt_consistent') is True
            prediction_ok = abs(reported_energy - reference_now) <= tolerance
            eligible = bool(new_epoch and receipt_ok and prediction_ok
                            and t1 >= min_t1 and t2 >= min_t2)
            if eligible:
                state['energy_recovery_last_epoch'] = current_epoch
                state['energy_recovery_reference_energy'] = reported_energy
                state['energy_recovery_reference_time'] = now
                self._reward_truthful_energy_report(node_id)
            elif new_epoch:
                state['energy_recovery_last_epoch'] = current_epoch
                state['energy_recovery_streak'] = 0

    def _estimate_energy_now(self, node_id, current_time=None):
        state = self.node_states.get(node_id)
        if state is None:
            return 0.0
        E0 = float(state.get('E_initial', 1.0))
        now = self._now() if current_time is None else float(current_time)
        last_e = float(
            state.get('energy_last_reported', state.get('E_est', E0)))
        last_t = float(state.get('energy_last_report_time', now))
        q = self._effective_depletion_rate(node_id, now)
        predicted = max(0.0, min(E0, last_e - q * max(0.0, now - last_t)))
        service_est = float(state.get('E_est', predicted))
        node = self.network_nodes.get(node_id)
        physical_est = float(getattr(node, 'E_current', predicted)
                             ) if node is not None else predicted
        predicted = min(predicted, min(E0, service_est), min(E0, physical_est))
        state['E_est'] = predicted
        return predicted

    def _node_center_distance_norm(self, state):
        if self.BS_ID in self.network_nodes:
            bx = float(self.network_nodes[self.BS_ID].x)
            by = float(self.network_nodes[self.BS_ID].y)
        else:
            xs = [float(st.get('x', 0.0)) for st in self.node_states.values()]
            ys = [float(st.get('y', 0.0)) for st in self.node_states.values()]
            bx = float(np.mean(xs)) if xs else 0.0
            by = float(np.mean(ys)) if ys else 0.0
        return min(1.0, math.hypot(float(state.get('x', bx)) - bx, float(state.get('y', by)) - by) /
                   max(float(self.ablation.get('network_diagonal', 141.421356)), 1e-6))

    def _ideal_charge_threshold(self, node_id, e_ratio, ed_norm, sd_norm):
        state = self.node_states.get(node_id, {})
        q = self._effective_depletion_rate(node_id)
        E0 = float(state.get('E_initial', 1.0))
        ttd = (e_ratio * E0) / max(q, 1e-12) if q > 0 else 1e6
        risk = float(state.get('risk_mean', state.get('fcm_risk', 0.0)))
        path_len = len((self.routing_tables.get(node_id, {})
                       or {}).get('primary_path', []) or [])
        route_pressure = min(1.0, max(0, path_len - 2) / 8.0)
        time_pressure = 1.0 - \
            min(1.0, ttd / float(self.ablation.get('qifl_safe_horizon_s', 1800.0)))
        target = 0.20 + 0.16 * time_pressure + 0.07 * \
            sd_norm + 0.05 * risk + 0.04 * route_pressure
        return float(np.clip(target, self.qifl_threshold_lower_bound, self.qifl_threshold_upper_bound))

    def get_adaptive_charge_threshold_for_node(self, node_id):
        state = self.node_states.get(node_id)
        if state is None:
            return float(self.ablation.get('charge_need_ratio', self.CHARGE_NEED_RATIO))
        E0 = float(state.get('E_initial', 1.0))
        E_est = self._estimate_energy_now(node_id)
        q = self._effective_depletion_rate(node_id)
        survival_time = E_est / \
            max(q, 1e-12) if q > 0 else float(self.ablation.get('qifl_safe_horizon_s', 1800.0))
        ed_norm = min(1.0, survival_time /
                      max(float(self.ablation.get('qifl_safe_horizon_s', 1800.0)), 1e-6))
        centrality = min(1.0, float(state.get('hop_count_to_bs_norm', 0.0)))
        sd_norm = min(
            1.0, 0.65 * self._node_center_distance_norm(state) + 0.35 * centrality)
        if self.use_adaptive_quantum_charging:
            threshold, _ = self.qifl_oracle.infer(ed_norm, sd_norm)
            ideal = self._ideal_charge_threshold(
                node_id, E_est / max(E0, 1e-12), ed_norm, sd_norm)
            loss = self.qifl_oracle.train_step(ed_norm, sd_norm, ideal)
            try:
                self.metrics.qifl_training_losses.append(float(loss))
            except Exception:
                pass
        else:
            threshold = float(self.ablation.get(
                'charge_need_ratio', self.CHARGE_NEED_RATIO))

        q_effective = max(q, float(self.ablation.get(
            'idle_sensing_power_j_per_s', 0.0)))
        service_latency = self.estimate_service_latency_s(node_id)
        sos_energy = self._compute_sos_threshold_energy(node_id)
        rt_min_ratio = (q_effective * service_latency +
                        sos_energy) / max(E0, 1e-12)

        hard_cap = float(np.clip(self.ablation.get(
            'adaptive_request_threshold_hard_cap', 0.68),
            self.qifl_threshold_lower_bound, 1.0))
        bounded_rt_min = min(float(rt_min_ratio), hard_cap)
        threshold = float(np.clip(
            max(float(threshold), bounded_rt_min),
            self.qifl_threshold_lower_bound,
            hard_cap))
        state['adaptive_charge_threshold'] = threshold
        state['adaptive_rt_min_ratio'] = float(rt_min_ratio)
        state['adaptive_rt_min_overload'] = max(
            0.0, float(rt_min_ratio) - hard_cap)
        try:
            self.metrics.adaptive_charge_thresholds.append(threshold)
        except Exception:
            pass
        return threshold

    def _compute_sos_threshold_energy(self, node_id):
        state = self.node_states.get(node_id, {})
        routes = self.routing_tables.get(node_id, {}) or {}
        primary_path = routes.get('primary_path') or []
        h_i = max(1, len(primary_path) - 1) if primary_path else 1
        E_tx = float(self.ablation.get('sos_control_tx_energy_j', 1.0e-6))
        prr = float(state.get('PRR_to_next',
                    self.ablation.get('default_link_prr', 0.92)))
        retries = int(self.ablation.get('mac_retry_budget', 3))
        success_factor = 1.0 / max(1e-6, 1.0 - (1.0 - prr) ** max(1, retries))
        margin = float(self.ablation.get('sos_energy_margin_j', 2.5e-4))
        thresh = h_i * E_tx * success_factor + margin
        state['last_sos_threshold_energy'] = float(thresh)
        try:
            self.metrics.sos_threshold_energies.append(float(thresh))
        except Exception:
            pass
        return float(thresh)

    def _base_charging_security_ok(self, node_id):
        state = self.node_states.get(node_id)
        if state is None or node_id == self.BS_ID:
            return False
        if state.get('status', 'WHITELISTED') in ('BLACKLISTED', 'QUARANTINE') or state.get('hard_veto', False):
            return False
        if float(state.get('Deviation_MC_norm', 0.0)) > self.CHARGE_MAX_DEVIATION_MC_NORM:
            return False
        if float(state.get('SED_index', 0.0)) >= self.CHARGE_MAX_SED:
            return False
        if float(state.get('spoof_trust', 1.0)) < self.CHARGE_MIN_SPOOF_TRUST:
            return False
        return True

    def _active_mobile_chargers(self, require_idle=False):
        chargers = list(getattr(self, 'mc_references', []) or [])
        if not chargers and self.mc_reference is not None:
            chargers = [self.mc_reference]
        usable = []
        for charger in chargers:
            if charger is None:
                continue
            if getattr(charger, 'zeroized', False) or getattr(charger, 'requires_lab_reprovisioning', False):
                continue
            if require_idle and getattr(charger, 'status', 'IDLE') != 'IDLE':
                continue
            usable.append(charger)
        return usable

    def _charger_travel_time_distance(self, node_id, charger=None):
        node_obj = self.network_nodes.get(node_id)
        if node_obj is None:
            return 0.0, 0.0
        candidates = [charger] if charger is not None else self._active_mobile_chargers(
            require_idle=True)
        fleet = list(getattr(self, 'mc_references', []) or [])
        if not candidates and self.mc_reference is not None and len(fleet) <= 1:
            candidates = [self.mc_reference]
        best = None
        for mc in candidates:
            if mc is None:
                continue
            dist = math.hypot(float(mc.x) - float(node_obj.x),
                              float(mc.y) - float(node_obj.y))
            travel = dist / max(float(getattr(mc, 'speed', 1.0)), 1e-6)
            item = (travel, dist)
            if best is None or item < best:
                best = item
        return best if best is not None else (0.0, 0.0)

    def charging_dispatch_priority(self, node_id, current_time=None, mc_x=None, mc_y=None, mc_speed=None):
        st = self.node_states.get(node_id, {})
        now = self._now() if current_time is None else float(current_time)
        e = self._estimate_energy_now(node_id, now)
        e0 = float(st.get('E_initial', 1.0))
        q = self._effective_depletion_rate(node_id, now)

        # ---------------------------------------------------------
        # [INNOVATION] CH-Aware Charging Urgency Boost
        # If the node is a Cluster Head (or intermediate backbone node),
        # its death would sever the connection for its entire cluster.
        # We artificially inflate its depletion rate (q) so the Mobile Charger
        # views it as a highly critical rescue mission.
        is_ch = node_id in getattr(self, 'cluster_heads', {})
        if is_ch:
            q *= 1.5  # 50% urgency boost for Cluster Heads
        # ---------------------------------------------------------

        ttd = e / max(q, 1e-12) if q > 0 else 1e6
        node = self.network_nodes.get(node_id)
        if node is not None and mc_x is not None and mc_y is not None:
            dist = math.hypot(float(mc_x) - float(node.x),
                              float(mc_y) - float(node.y))
            speed = float(mc_speed) if mc_speed is not None else (float(getattr(
                self.mc_reference, 'speed', 1.0)) if self.mc_reference is not None else 1.0)
            travel = dist / max(speed, 1e-6)
        else:
            travel, dist = self._charger_travel_time_distance(node_id)

        slack = ttd - travel
        e_ratio = e / max(e0, 1e-12)
        hard = slack <= self.hard_deadline_margin_s or e_ratio <= float(
            self.ablation.get('critical_energy_ratio', 0.15))

        return (0 if hard else 1, slack, dist, e_ratio)

    def _charging_pool_class(self, node_id, emergency=False):
        if node_id in self.fixed_charger_nodes:
            return None

        state = self.node_states.get(node_id)
        if state is None or not self._base_charging_security_ok(node_id):
            return None
        now = self._now()
        e0 = max(float(state.get('E_initial', 1.0)), 1e-12)
        e_est = self._estimate_energy_now(node_id, now)
        e_ratio = e_est / e0
        e_claimed = float(state.get('E_claimed', e_est))
        max_claim_dev = float(getattr(
            self, 'CHARGE_MAX_CLAIM_EST_DEVIATION',
            self.ablation.get('charge_claim_est_max_deviation', 0.30)))
        if abs(e_claimed - e_est) / e0 > max_claim_dev:
            return None

        hysteresis = bool(self.ablation.get('use_charging_hysteresis', True))
        if hysteresis and e_ratio >= self.post_charge_release_ratio:
            state['charge_satisfied'] = True
            return None
        cooldown = self.charging_reentry_cooldown_s if hysteresis else 0.0
        if cooldown > 0.0 and not emergency:
            if now - self.last_charge_time_by_node[node_id] < cooldown:
                return None
            selection_gap = max(5.0, 0.25 * cooldown)
            if now - self.last_selected_charge_time_by_node[node_id] < selection_gap:
                return None

        threshold = self.get_adaptive_charge_threshold_for_node(node_id)
        plus_fraction = float(np.clip(self.ablation.get(
            'request_plus_fraction', 0.50), 0.0, 1.0))
        plus_threshold = threshold + plus_fraction * (1.0 - threshold)
        sos_energy = self._compute_sos_threshold_energy(node_id)
        q = self._effective_depletion_rate(node_id, now)
        ttd = e_est / max(q, 1e-12)
        travel, distance = self._charger_travel_time_distance(node_id)
        service_latency = self.estimate_service_latency_s(node_id)
        critical_ratio = float(self.ablation.get(
            'critical_energy_ratio', 0.15))
        emergency_ratio = float(self.ablation.get('emergency_ratio', 0.12))
        sos_admit = e_est <= max(sos_energy, emergency_ratio * e0)
        hard_rescue = ttd <= service_latency or e_ratio <= critical_ratio
        useful_gap_ratio = max(0.0, 1.0 - e_ratio)

        if emergency or sos_admit or hard_rescue:
            pool = 'service'
        elif e_ratio <= threshold:
            pool = 'service'
        elif e_ratio <= plus_threshold:
            pool = 'plus'
        else:
            return None

        if (not emergency and pool == 'service' and
                useful_gap_ratio < self.min_charge_energy_ratio):
            return None

        if self.use_travel_aware_admission and not (
                emergency or sos_admit or hard_rescue):
            diag = float(self.ablation.get('network_diagonal', 141.421356))
            normalized_distance = min(
                1.0, max(0.0, distance) / max(diag, 1e-6))
            slack_to_service = ttd - service_latency
            horizon = max(service_latency, float(self.ablation.get(
                'charge_deadline_horizon_s', 900.0)))
            threshold_pressure = max(
                0.0, (threshold - e_ratio) / max(threshold, 1e-12))
            deadline_pressure = 1.0 - min(
                1.0, max(0.0, slack_to_service) / max(horizon, 1e-6))
            urgency = max(
                threshold_pressure,
                0.65 * deadline_pressure + 0.35 * useful_gap_ratio)
            effective_urgency = urgency / (
                1.0 + self.travel_admission_lambda * normalized_distance)
            state['travel_effective_urgency'] = float(effective_urgency)
            state['travel_normalized_distance'] = float(normalized_distance)
            if pool == 'service' and effective_urgency < self.travel_admission_gate:
                pool = 'plus'

        state['charging_pool_class'] = pool
        state['charging_plus_threshold'] = float(plus_threshold)
        return pool

    def _admit_charging_node(self, node_id, emergency=False):
        return self._charging_pool_class(node_id, emergency=emergency) is not None

    def _ground_truth_charging_need(self, node_id, emergency=False):
        node = self.network_nodes.get(node_id)
        state = self.node_states.get(node_id)
        if node is None or state is None or node_id == self.BS_ID:
            return False
        if node_id in self.fixed_charger_nodes or not self._base_charging_security_ok(node_id):
            return False
        e_max = max(
            float(getattr(node, 'E_max', state.get('E_initial', 1.0))), 1e-12)
        e_now = max(0.0, float(
            getattr(node, 'E_current', state.get('E_est', e_max))))
        ratio = e_now / e_max
        q = self._effective_depletion_rate(node_id)
        ttd = e_now / max(q, 1e-12) if q > 0.0 else 1e6
        travel, _ = self._charger_travel_time_distance(node_id)
        service_latency = self.estimate_service_latency_s(node_id)
        critical_ratio = float(self.ablation.get(
            'critical_energy_ratio', 0.15))
        audit_margin = max(0.0, float(self.ablation.get(
            'charging_need_audit_margin_s', 45.0)))
        return bool(
            emergency
            or ratio <= critical_ratio
            or ttd <= service_latency + audit_margin
        )

    def build_adaptive_charging_queue(self, idle_charger_count=None):
        if not self.energy_experiment_active:
            self.secure_charging_queue = []
            self.secure_charging_node_ids = set()
            self.service_charging_node_ids = set()
            self.plus_charging_node_ids = set()
            self.current_charging_clusters = {}
            return []

        service_ids = []
        plus_ids = []
        for nid in list(self.node_states.keys()):
            if nid == self.BS_ID:
                continue
            emergency = nid in self.pending_sos_requests
            pool = self._charging_pool_class(nid, emergency=emergency)
            needed = self._ground_truth_charging_need(
                nid, emergency=emergency)
            admitted_service = pool == 'service'

            state = self.node_states.get(nid, {})
            cycle = int(state.get('charge_audit_cycle', 0))
            audit_key = (int(nid), cycle, bool(needed), bool(admitted_service))
            self.metrics.record_charging_decision(
                needed=needed,
                admitted=admitted_service,
                decision_key=audit_key)

            if pool == 'service':
                service_ids.append(nid)
            elif pool == 'plus':
                plus_ids.append(nid)

        if not service_ids:
            self.secure_charging_queue = []
            self.secure_charging_node_ids = set()
            self.service_charging_node_ids = set()
            self.plus_charging_node_ids = set()
            self.current_charging_clusters = {}
            try:
                self.metrics.charging_kmeans_cluster_counts.append(0)
                self.metrics.charging_cluster_assigned_node_counts.append(0)
            except Exception:
                pass
            return []

        def priority(nid):
            return self.charging_dispatch_priority(nid)

        max_queue = max(1, int(self.ablation.get(
            'max_adaptive_charge_queue', 8)))
        service_ids = sorted(set(service_ids), key=priority)[:max_queue]
        plus_ids = sorted(set(plus_ids), key=priority)
        self.service_charging_node_ids = set(service_ids)

        for nid in service_ids:
            state = self.node_states.get(nid, {})
            cycle = int(state.get('charge_audit_cycle', 0))
            if state.get('service_pool_counted_cycle') != cycle:
                state['service_pool_counted_cycle'] = cycle
                self.metrics.charging_service_pool_admissions += 1
                if self.use_adaptive_quantum_charging:
                    self.metrics.qifl_admission_count += 1

        if self.use_rf_multi_node_charging:
            clusters_obj = []
            remaining_service = list(service_ids)
            remaining_plus = list(plus_ids)
            cluster_index = 0
            max_members = max(1, int(self.ablation.get(
                'max_rf_cluster_members', 5)))
            max_plus = max(0, int(self.ablation.get(
                'plus_pool_max_members_per_cluster', 2)))

            def radius_feasible(nodes):
                if not nodes:
                    return False
                cx = float(np.mean([node.x for node in nodes]))
                cy = float(np.mean([node.y for node in nodes]))
                radius = max(math.hypot(node.x - cx, node.y - cy)
                             for node in nodes)
                return radius <= self.rf_charging_radius + 1e-9

            while remaining_service:
                seed_id = remaining_service.pop(0)
                members = [self.network_nodes[seed_id]]

                while remaining_service and len(members) < max_members:
                    cx = float(np.mean([node.x for node in members]))
                    cy = float(np.mean([node.y for node in members]))
                    ordered = sorted(
                        remaining_service,
                        key=lambda nid: math.hypot(
                            self.network_nodes[nid].x - cx,
                            self.network_nodes[nid].y - cy))
                    selected = next((nid for nid in ordered if radius_feasible(
                        members + [self.network_nodes[nid]])), None)
                    if selected is None:
                        break
                    members.append(self.network_nodes[selected])
                    remaining_service.remove(selected)

                plus_added = 0
                while (remaining_plus and len(members) < max_members and
                       plus_added < max_plus):
                    cx = float(np.mean([node.x for node in members]))
                    cy = float(np.mean([node.y for node in members]))
                    ordered = sorted(
                        remaining_plus,
                        key=lambda nid: math.hypot(
                            self.network_nodes[nid].x - cx,
                            self.network_nodes[nid].y - cy))
                    selected = next((nid for nid in ordered if radius_feasible(
                        members + [self.network_nodes[nid]])), None)
                    if selected is None:
                        break
                    members.append(self.network_nodes[selected])
                    remaining_plus.remove(selected)
                    plus_added += 1

                cluster = ChargingCluster.from_nodes(cluster_index, members)
                clusters_obj.append(cluster)
                cluster_index += 1

            selected_ids = {
                int(node.node_id)
                for cluster in clusters_obj for node in cluster.nodes
            }
            selected_plus = selected_ids - set(service_ids)
            self.plus_charging_node_ids = set(selected_plus)
            self.secure_charging_node_ids = set(selected_ids)
            self.secure_charging_queue = clusters_obj
            self.current_charging_clusters = {
                cluster.node_id: list(cluster.member_ids)
                for cluster in clusters_obj}
            try:
                self.metrics.charging_kmeans_cluster_counts.append(
                    float(len(clusters_obj)))
                self.metrics.charging_cluster_assigned_node_counts.append(
                    float(len(selected_ids)))
                for nid in selected_plus:
                    state = self.node_states.get(nid, {})
                    cycle = int(state.get('charge_audit_cycle', 0))
                    if state.get('plus_pool_counted_cycle') != cycle:
                        state['plus_pool_counted_cycle'] = cycle
                        self.metrics.charging_plus_pool_admissions += 1
            except Exception:
                pass
            return self.secure_charging_queue

        self.plus_charging_node_ids = set()
        self.secure_charging_node_ids = set(service_ids)
        self.secure_charging_queue = list(service_ids)
        self.current_charging_clusters = {0: list(service_ids)}
        try:
            self.metrics.charging_kmeans_cluster_counts.append(1.0)
            self.metrics.charging_cluster_assigned_node_counts.append(
                float(len(service_ids)))
            pass
        except Exception:
            pass
        return self.secure_charging_queue

    def validate_charging_request(self, node_id, emergency=False):
        if not self.ablation.get('use_secure_charging_request_validation', True):
            self.metrics.mc_raw_sos_dispatches += 1
            ok = node_id in self.node_states and node_id != self.BS_ID
        elif self.energy_experiment_active and (
                self.use_adaptive_quantum_charging or
                self.ablation.get('use_adaptive_charge_guards', False)):
            if node_id in self.secure_charging_node_ids:
                ok = self._base_charging_security_ok(node_id)
            else:
                ok = self._charging_pool_class(
                    node_id, emergency=emergency) == 'service'
        else:
            ok = False
            state = self.node_states.get(node_id)
            if state is not None and node_id != self.BS_ID:
                if state.get('status', 'WHITELISTED') not in ('BLACKLISTED', 'QUARANTINE') and not state.get('hard_veto', False):
                    if float(state.get('Deviation_MC_norm', 0.0)) <= self.CHARGE_MAX_DEVIATION_MC_NORM and float(state.get('SED_index', 0.0)) < self.CHARGE_MAX_SED and float(state.get('spoof_trust', 1.0)) >= self.CHARGE_MIN_SPOOF_TRUST:
                        E_initial = float(state.get('E_initial', 1.0))
                        E_est = float(state.get('E_est', E_initial))
                        E_claimed = float(state.get('E_claimed', E_est))
                        max_claim_dev = float(getattr(self, 'CHARGE_MAX_CLAIM_EST_DEVIATION', self.ablation.get(
                            'charge_claim_est_max_deviation', 0.30)))
                        if abs(E_claimed - E_est) / max(E_initial, 1e-12) <= max_claim_dev:
                            threshold = float(self.CHARGE_NEED_RATIO if not emergency else max(
                                self.CHARGE_NEED_RATIO, 0.35))
                            ok = (E_est / max(E_initial, 1e-12)) <= threshold

            if not ok and self.energy_experiment_active and self.ablation.get('allow_energy_stress_predictive_charging', False):
                state = self.node_states.get(node_id)
                if state is not None and node_id != self.BS_ID:
                    if state.get('status') not in ('BLACKLISTED', 'QUARANTINE') and not state.get('hard_veto', False):
                        if float(state.get('Deviation_MC_norm', 0.0)) <= self.CHARGE_MAX_DEVIATION_MC_NORM and float(state.get('SED_index', 0.0)) < self.CHARGE_MAX_SED and float(state.get('spoof_trust', 1.0)) >= self.CHARGE_MIN_SPOOF_TRUST:
                            E_initial = float(state.get('E_initial', 1.0))
                            E_est = float(state.get('E_est', E_initial))
                            ok = (E_est / max(E_initial, 1e-12)) <= float(
                                self.ablation.get('charge_need_ratio', self.CHARGE_NEED_RATIO))

        if emergency:
            self.metrics.raw_sos_validation_attempts += 1
            if not ok:
                self.metrics.raw_sos_blocked += 1

        if self.energy_experiment_active:
            if node_id in self.pending_sos_requests or node_id in self.secure_charging_node_ids:
                current_t = float(
                    getattr(self.metrics, 'current_sim_time', 0.0))
                self.metrics.record_charging_request(
                    node_id, current_t, validated=ok)
        return bool(ok)

    def _penalize_mc_trust(self, mc_id, severity=0.25):
        key = str(mc_id)
        severity = float(np.clip(severity, 0.0, 1.0))
        self.mc_violation_count[key] += 1
        self.mc_consistent_report_streak[key] = 0
        self.mc_trust[key] = max(0.0, float(self.mc_trust[key]) * (1.0 - severity))
        return self.mc_trust[key]

    def _reward_mc_trust(self, mc_id):
        key = str(mc_id)
        self.mc_consistent_report_streak[key] += 1
        alpha = float(np.clip(self.ablation.get('mc_trust_recovery_alpha', 0.08), 0.0, 1.0))
        current = float(self.mc_trust[key])
        self.mc_trust[key] = min(1.0, current + alpha * (1.0 - current))
        return self.mc_trust[key]

    def _penalize_energy_report(self, node_id, reason, severity=0.5, hard=False):
        state = self.node_states.get(node_id)
        if state is None:
            return
        severity = float(np.clip(severity, 0.0, 1.0))
        state['energy_violation_count'] = int(state.get('energy_violation_count', 0)) + 1
        state['energy_recovery_streak'] = 0
        state['energy_probation'] = True
        state['last_energy_violation_reason'] = str(reason)
        state['energy_recovery_reference_energy'] = float(state.get('E_est', 0.0))
        state['energy_recovery_reference_time'] = float(self._now())
        state['energy_recovery_last_epoch'] = None
        state['last_node_charge_receipt_consistent'] = False
        t5 = float(state.get('T5_history', 1.0))
        t5 = max(0.0, t5 * (1.0 - severity))
        state['T5_history'] = t5
        state['trust_vector'][4] = t5
        if hard:
            state['hard_veto'] = True
            state['hard_veto_reason'] = 'energy_report_mismatch'
            state['hard_veto_recoverable'] = True
            state['status'] = 'BLACKLISTED'
            state['FE'] = 0.0
        elif state.get('status') not in ('BLACKLISTED', 'QUARANTINE'):
            state['status'] = 'WATCHLIST'

    def _reward_truthful_energy_report(self, node_id):
        state = self.node_states.get(node_id)
        if state is None:
            return
        state['energy_recovery_streak'] = int(state.get('energy_recovery_streak', 0)) + 1
        required = max(1, int(self.ablation.get('energy_trust_recovery_epochs', 8)))
        alpha = float(np.clip(self.ablation.get('energy_trust_recovery_alpha', 0.12), 0.0, 1.0))
        t5 = float(state.get('T5_history', 1.0))
        t5 = min(1.0, t5 + alpha * (1.0 - t5))
        state['T5_history'] = t5
        state['trust_vector'][4] = t5
        if (state.get('hard_veto_recoverable', False)
                and state.get('hard_veto_reason') == 'energy_report_mismatch'
                and state['energy_recovery_streak'] >= required):
            state['hard_veto'] = False
            state['hard_veto_recoverable'] = False
            state['hard_veto_reason'] = None
            state['energy_probation'] = False
            state['energy_violation_count'] = 0
            state['Deviation_MC'] = 0.0
            state['Deviation_MC_norm'] = 0.0
            state['status'] = 'WATCHLIST'
            state['blacklist_counter'] = 0
            self.metrics.energy_trust_recoveries += 1
        elif (state.get('energy_probation', False)
              and state['energy_recovery_streak'] >= required):
            state['energy_probation'] = False
            state['energy_violation_count'] = 0
            if not state.get('hard_veto', False):
                state['status'] = 'WHITELISTED'

    def verify_charging_location(self, node_id, mc_x, mc_y, allowed_radius=None, charging_mode='contact', is_spoof_attempt=False, mc_id=None):
        self.metrics.mc_location_verification_checks += 1
        if is_spoof_attempt:
            self.metrics.mc_location_spoof_attempts += 1
        state = self.node_states.get(node_id)
        if state is None:
            self.metrics.mc_location_verification_failures += 1
            if is_spoof_attempt:
                self.metrics.mc_location_spoof_detections += 1
            return False
        if not self.ablation.get('use_mc_location_verification', True):
            state['spoof_trust'] = 1.0
            self.metrics.mc_location_verification_passes += 1
            if is_spoof_attempt:
                self.metrics.mc_location_spoof_accepted += 1
            return True

        if allowed_radius is None:
            allowed_radius = float(self.ablation.get(
                'contact_location_tolerance_m', 5.0))
        allowed_radius = max(0.0, float(allowed_radius))
        distance = math.hypot(
            float(mc_x) - float(state['x']),
            float(mc_y) - float(state['y']))
        valid = distance <= allowed_radius + 1e-9
        state['spoof_trust'] = 1.0 if valid else 0.0
        state['last_charging_mode'] = str(charging_mode)
        state['last_charging_location_distance'] = float(distance)
        state['last_charging_location_radius'] = float(allowed_radius)
        if valid:
            self.metrics.mc_location_verification_passes += 1
            if is_spoof_attempt:
                self.metrics.mc_location_spoof_accepted += 1
            if mc_id is not None:
                self._reward_mc_trust(mc_id)
        else:
            self.metrics.mc_location_verification_failures += 1
            self.metrics.mc_location_report_rejections += 1
            if is_spoof_attempt:
                self.metrics.mc_location_spoof_detections += 1
            if mc_id is not None:
                self._penalize_mc_trust(mc_id, severity=0.40)
        return valid

    def _credit_accepted_mc_injection(self, node_id, injected_energy, timestamp=None):
        state = self.node_states.get(node_id)
        if state is None:
            return
        injected_energy = float(max(0.0, injected_energy))
        now = self._now() if timestamp is None else float(timestamp)
        current_injected = state.get('total_injected_energy', 0.0)
        state['total_injected_energy'] = current_injected + injected_energy
        state['energy_injected_since_physical_sample'] = float(
            state.get('energy_injected_since_physical_sample', 0.0)) + injected_energy
        self.mc_recent_injected_energy[node_id] += injected_energy
        E0 = float(state.get('E_initial', 1.0))
        new_est = min(E0, float(state.get('E_est', 0.0)) + injected_energy)
        state['E_est'] = new_est
        # Preserve the legacy operational estimator for direct comparability
        # with previous experiments. Independent MC/node evidence is retained
        # in separate audit fields and can be enforced with strict audit mode.
        if self.ablation.get('preserve_legacy_mc_claim_credit', True):
            state['E_claimed'] = max(
                float(state.get('E_claimed', 0.0)), new_est)
            state['legitimate_mc_credit_pending'] = True
            state['energy_last_reported'] = max(
                float(state.get('energy_last_reported', 0.0)), new_est)
            state['energy_last_report_time'] = now
        state['pending_mc_audit_energy'] = float(
            state.get('pending_mc_audit_energy', 0.0)) + injected_energy
        state['pending_mc_expected_post_energy'] = new_est
        state['pending_mc_audit_time'] = now
        state['charge_satisfied'] = (
            new_est / max(E0, 1e-12)) >= self.post_charge_release_ratio
        previous = self.last_charge_time_by_node[node_id]
        if math.isfinite(previous) and now >= previous:
            try:
                interval = max(0.0, now - previous)
                self.metrics.mc_total_revisit_events += 1
                self.metrics.mc_revisit_intervals.append(interval)
                if interval < max(0.0, float(self.charging_reentry_cooldown_s)):
                    self.metrics.mc_repeated_service_events += 1
            except Exception:
                pass
        self.last_charge_time_by_node[node_id] = now
        state['charge_audit_cycle'] = int(
            state.get('charge_audit_cycle', 0)) + 1
        self.last_charged_energy_ratio_by_node[node_id] = new_est / max(
            E0, 1e-12)

    def report_node_charge_receipt(self, receipt, expected_injected_energy=None, mc_report_accepted=None):
        """Verify and reconcile an independently signed sensor charging receipt."""
        if not isinstance(receipt, dict):
            self.metrics.node_charge_receipt_auth_failures += 1
            return False
        try:
            node_id = int(receipt.get('node_id'))
            before = float(receipt.get('energy_before'))
            after = float(receipt.get('energy_after'))
            claimed_received = float(receipt.get('claimed_received'))
            mc_id = receipt.get('mc_id')
            counter = int(receipt.get('counter'))
            nonce = str(receipt.get('nonce'))
            timestamp = float(receipt.get('timestamp'))
            supplied = receipt.get('mac_tag')
        except (TypeError, ValueError):
            self.metrics.node_charge_receipt_auth_failures += 1
            return False
        state = self.node_states.get(node_id)
        if state is None:
            self.metrics.node_charge_receipt_auth_failures += 1
            return False
        key = state.get('crypto_keys')
        expected = make_node_charge_receipt_mac(
            key, node_id, before, after, claimed_received, mc_id=mc_id,
            counter=counter, nonce=nonce, timestamp=timestamp)
        replay = (counter <= self.node_receipt_last_counter[node_id]
                  or nonce in self.node_receipt_seen_nonces[node_id])
        if replay or expected is None or supplied is None or not hmac.compare_digest(str(supplied), str(expected)):
            self.metrics.node_charge_receipt_auth_failures += 1
            return False

        self.node_receipt_last_counter[node_id] = counter
        self.node_receipt_seen_nonces[node_id].append(nonce)
        self.metrics.node_charge_receipts_received += 1

        E0 = max(float(state.get('E_initial', 1.0)), 1e-12)
        credited_mc_energy = max(0.0, float(state.get('pending_mc_audit_energy', 0.0)))
        expected_energy = (
            credited_mc_energy if expected_injected_energy is None
            else max(0.0, float(expected_injected_energy)))
        expected_post = float(state.get('pending_mc_expected_post_energy', state.get('E_est', after)))
        physical_node = self.network_nodes.get(node_id)
        trusted_post = (
            float(getattr(physical_node, 'E_current', expected_post))
            if physical_node is not None else expected_post)
        observed_delta = max(0.0, after - before)
        tolerance = max(
            float(self.ablation.get('charge_receipt_absolute_tolerance_j', 1e-9)),
            float(self.ablation.get('charge_receipt_tolerance_ratio', 0.05)) * E0)
        consistent = bool(
            -tolerance <= before <= E0 + tolerance
            and -tolerance <= after <= E0 + tolerance
            and abs(claimed_received - expected_energy) <= tolerance
            and after <= before + claimed_received + tolerance
            and abs(after - trusted_post) <= tolerance)

        state['last_node_charge_receipt_consistent'] = consistent
        state['last_node_charge_receipt_mc_id'] = mc_id
        state['last_node_charge_receipt_time'] = timestamp
        state['last_node_charge_receipt_delta'] = observed_delta
        state['last_node_charge_receipt_claimed'] = claimed_received
        state['pending_mc_audit_energy'] = 0.0
        state['pending_mc_expected_post_energy'] = after if consistent else expected_post

        enforcement = str(self.ablation.get(
            'charging_audit_enforcement_mode', 'shadow')).strip().lower()
        strict = enforcement in {'strict', 'enforce', 'operational'}
        audit_trust = float(state.get('energy_audit_trust', 1.0))

        if consistent:
            self.metrics.node_charge_receipts_consistent += 1
            state['energy_audit_recovery_streak'] = int(
                state.get('energy_audit_recovery_streak', 0)) + 1
            audit_alpha = float(np.clip(self.ablation.get(
                'energy_audit_trust_recovery_alpha', 0.15), 0.0, 1.0))
            state['energy_audit_trust'] = min(
                1.0, audit_trust + audit_alpha * (1.0 - audit_trust))
            if strict:
                was_rehabilitating = bool(
                    state.get('energy_probation', False)
                    or state.get('hard_veto_recoverable', False))
                state['E_claimed'] = max(0.0, min(E0, after))
                self._observe_energy_report(node_id, state['E_claimed'], timestamp)
                state['legitimate_mc_credit_pending'] = bool(
                    mc_report_accepted is not False)
                if not was_rehabilitating:
                    self._reward_truthful_energy_report(node_id)
            if mc_id is not None:
                self._reward_mc_trust(mc_id)
        else:
            self.metrics.node_charge_receipts_inconsistent += 1
            state['energy_audit_recovery_streak'] = 0
            severe = (abs(claimed_received - expected_energy) > 2.0 * tolerance
                      or abs(after - trusted_post) > 2.0 * tolerance)
            state['energy_audit_trust'] = max(
                0.0, audit_trust * (0.45 if severe else 0.75))
            state['energy_audit_probation'] = True
            if strict:
                state['E_claimed'] = max(0.0, min(E0, after))
                self._observe_energy_report(node_id, state['E_claimed'], timestamp)
                self._penalize_energy_report(
                    node_id, 'node_charge_receipt_mismatch',
                    severity=0.55 if severe else 0.25, hard=severe)
        return consistent

    def is_deadline_rescue_node(self, node_id, current_time=None):
        """
        Checks if a node is critically low on energy and requires immediate emergency rescue
        before its hard deadline.
        """
        state = self.node_states.get(node_id)
        if not state:
            return False

        now = self._now() if current_time is None else float(current_time)
        e_est = self._estimate_energy_now(node_id, now)
        e_initial = max(float(state.get('E_initial', 1.0)), 1e-12)
        q = self._effective_depletion_rate(node_id, now)

        # Time To Death (TTD)
        ttd = e_est / max(q, 1e-12) if q > 0 else float('inf')

        hard_margin = float(self.ablation.get('hard_deadline_margin_s', 90.0))
        critical_ratio = float(self.ablation.get(
            'critical_energy_ratio', 0.15))

        # A node needs rescue if its remaining time is less than the safety margin
        # OR its energy ratio has dropped below the critical threshold.
        return bool(ttd <= hard_margin or (e_est / e_initial) <= critical_ratio)

    def report_mc_injection_oob(self, node_id, injected_energy, mc_x=None, mc_y=None, mac_tag=None, mc_id=None, counter=None, nonce=None, timestamp=None, *, charging_mode='contact', allowed_radius=None, expected_injected_energy=None, is_spoof_attempt=False):
        """Authenticate, localize, and reconcile an MC charging report.

        Physical charging has already occurred. Rejection prevents the claim
        from being credited in BS accounting and reduces MC trust; it does not
        attempt to remove energy already delivered to the sensor.
        """
        auth_enabled = self.ablation.get('use_mc_hmac_authentication', True)
        if node_id not in self.node_states:
            return False

        fake_or_missing = (mac_tag is None or mc_x is None or mc_y is None)
        shared_key = getattr(self, 'mc_shared_key', None)
        if mc_id is not None:
            shared_key = getattr(self, 'mc_shared_keys', {}).get(str(mc_id), shared_key)

        if auth_enabled:
            required_missing = (
                shared_key is None or mac_tag is None or mc_x is None or mc_y is None or
                mc_id is None or counter is None or nonce is None or timestamp is None)
            if required_missing:
                self.metrics.mc_fake_reports_attempted += 1
                self.metrics.mc_rejected_unauth_reports += 1
                self.mc_fail_closed_rejections += 1
                if mc_id is not None:
                    self._penalize_mc_trust(mc_id, severity=0.50)
                return False

            counter = int(counter)
            nonce = str(nonce)
            mc_key = str(mc_id)
            if counter <= self.mc_last_counter[mc_key] or nonce in self.mc_seen_nonces[mc_key]:
                self.metrics.mc_fake_reports_attempted += 1
                self.metrics.mc_rejected_unauth_reports += 1
                self.metrics.mc_replay_rejections += 1
                self.mc_replay_rejections += 1
                self._penalize_mc_trust(mc_id, severity=0.35)
                return False

            expected_tag = make_mc_mac(
                shared_key, node_id, injected_energy, mc_x, mc_y,
                mc_id=mc_id, counter=counter, nonce=nonce, timestamp=timestamp)
            if expected_tag is None or not hmac.compare_digest(str(mac_tag), str(expected_tag)):
                self.metrics.mc_fake_reports_attempted += 1
                self.metrics.mc_rejected_unauth_reports += 1
                self._penalize_mc_trust(mc_id, severity=0.50)
                return False

            # An authenticated but physically invalid report is still consumed
            # by the anti-replay state before its location/energy claim is rejected.
            self.mc_last_counter[mc_key] = counter
            self.mc_seen_nonces[mc_key].append(nonce)
        elif fake_or_missing:
            self.metrics.mc_fake_reports_attempted += 1

        if mc_x is None or mc_y is None:
            self.metrics.mc_location_verification_checks += 1
            self.metrics.mc_location_verification_failures += 1
            self.metrics.mc_location_report_rejections += 1
            if mc_id is not None:
                self._penalize_mc_trust(mc_id, severity=0.40)
            return False

        if not self.verify_charging_location(
                node_id, mc_x, mc_y, allowed_radius=allowed_radius,
                charging_mode=charging_mode, is_spoof_attempt=is_spoof_attempt,
                mc_id=mc_id):
            return False

        E0 = max(float(self.node_states[node_id].get('E_initial', 1.0)), 1e-12)
        if expected_injected_energy is not None:
            deviation = abs(float(injected_energy) - float(expected_injected_energy))
            tolerance = max(
                float(self.ablation.get('mc_energy_claim_absolute_tolerance_j', 1e-9)),
                float(self.ablation.get('mc_energy_claim_tolerance_ratio', 0.05)) * E0)
            if deviation > tolerance:
                self.metrics.mc_energy_claim_mismatches += 1
                if mc_id is not None:
                    self._penalize_mc_trust(mc_id, severity=0.45)
                return False

        self._credit_accepted_mc_injection(node_id, injected_energy, timestamp=timestamp)
        if mc_id is not None:
            self._reward_mc_trust(mc_id)
        if not auth_enabled and fake_or_missing:
            self.metrics.mc_fake_reports_accepted += 1
        return True

    def trigger_incident_response(self, victim_id, spoof_x, spoof_y):
        if not self.energy_experiment_active:
            self.pending_sos_requests.add(victim_id)
            return
        token = secrets.token_hex(8)
        self.zone_wakeup_tokens[(spoof_x, spoof_y)] = token

        available_chargers = self._active_mobile_chargers(require_idle=True)
        if available_chargers:
            victim_node = self.network_nodes[victim_id]
            rescue_mc = min(
                available_chargers,
                key=lambda mc: math.hypot(
                    float(mc.x) - float(victim_node.x), float(mc.y) - float(victim_node.y))
            )
            rescue_mc.interrupt_and_rescue(victim_node)
        else:
            self.pending_sos_requests.add(victim_id)

        quarantine_radius = 40.0
        zone = {'x': spoof_x, 'y': spoof_y, 'radius': quarantine_radius}
        if zone not in self.quarantined_zones:
            self.quarantined_zones.append(zone)

        for nid, state in self.node_states.items():
            if nid == self.BS_ID:
                continue
            dist = math.hypot(state['x'] - spoof_x, state['y'] - spoof_y)
            if dist <= quarantine_radius:
                state['requires_otp_for_charge'] = True
                state['wake_up_token'] = token
                self.metrics.update_packet_transmission(
                    is_critical=True, is_control=True, size_bytes=16)
