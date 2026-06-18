import time
import math
import hmac
import secrets
from collections import defaultdict, deque
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config import compute_packet_hash, make_mc_mac, _source_energy, DATA_PACKET_BITS
from routing import BaselineRouting


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

        self.packet_logs = []
        self.dropped_packets_tracker = defaultdict(list)
        self.pending_sos_requests = set()
        self.secure_charging_queue = []

        self.experiment_family = self.ablation.get(
            'experiment_family', 'communication_security')
        self.energy_experiment_active = (
            self.experiment_family == 'energy_charging_stress'
            or self.ablation.get('enable_energy_experiment', False)
        )

        self.FCM_BLACKLIST_THRESHOLD = 0.8
        self.FCM_GREYLIST_THRESHOLD = 0.2
        self.T_epoch = 10.0

        self.eta_charging = 0.70
        self.mc_recent_injected_energy = defaultdict(float)
        self.mc_real_injected_energy = defaultdict(float)
        self.quarantined_zones = []
        self.mc_reference = None
        self.mc_shared_key = None

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

        # Extracted Magic Numbers for Risk Weighting
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

        if routing_engine is None:
            self.routing_engine = BaselineRouting(
                bs_id=self.BS_ID, transmission_range=40.0)
        else:
            self.routing_engine = routing_engine

    def record_environmental_observation(self, node_id, value):
        if value is not None:
            self.environmental_history[node_id].append(float(value))

    def _normalize_path_for_bs(self, path_taken):
        path = list(path_taken or [])
        if path and path[-1] != self.BS_ID:
            path.append(self.BS_ID)
        return path

    def _approved_critical_paths(self, source_id):
        routes = self.routing_tables.get(source_id, {}) or {}
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

        if packet_type == 'SOS_CHARGE':
            self.pending_sos_requests.add(source_id)
        elif packet_type == 'DATA':
            payload = packet.get('payload')
            if payload is not None:
                self.record_environmental_observation(source_id, payload)

        if interrupt_triggered and self.ablation.get('use_asynchronous_processing_pipeline', True):
            self.control_plane_epoch()

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
        nodes = list(self.node_states.keys())
        valid_nodes = [n for n in nodes if n != self.BS_ID]
        dynamic_weights = self._calculate_hybrid_weights(valid_nodes)
        self.current_dynamic_weights = dynamic_weights

        for nid in valid_nodes:
            if self.node_states[nid].get('status') == 'BLACKLISTED':
                self.node_states[nid]['FE'] = 0.0
                continue
            T_vec = self.node_states[nid]['trust_vector']
            T1, T2, T3, T4, T5 = T_vec
            weighted_sum = float(np.dot(dynamic_weights, T_vec))
            veto_multiplier = min(T1, T2) if self.ablation.get(
                'use_security_veto_aggregation', True) else 1.0
            self.node_states[nid]['FE'] = weighted_sum * veto_multiplier

    def control_plane_epoch(self):
        old_h = self.RISK_HYSTERESIS_EPOCHS
        old_w = self.RISK_WARMUP_EPOCHS
        if not self.ablation.get('use_risk_history_hysteresis', True):
            self.RISK_HYSTERESIS_EPOCHS = 1
            self.RISK_WARMUP_EPOCHS = 0

        try:
            start_time = time.time()
            self.control_epoch_index += 1

            self._verify_wrsn_cyber_physical_security()
            self._two_tier_trust_matrix()
            self._verify_stealthy_attackers()
            self._verify_trajectory_manipulators()

            self._compute_fitness_value()

            U_malicious_dict = self._fcm_classification()
            self._calibrated_risk_fusion(U_malicious_dict)

            network_clusters = self._network_clustering()
            self._joint_optimization(
                U_malicious_dict, network_clusters)

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

        E_elec = 50e-9
        eps_fs = 10e-12
        eps_mp = 0.0013e-12
        d0 = math.sqrt(eps_fs / eps_mp)
        packet_bits = DATA_PACKET_BITS
        ack_bits = 16 * 8

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

        path_success_counts = defaultdict(int)
        path_tamper_counts = defaultdict(int)

        for log in self.packet_logs:
            path_tuple = tuple(log['path'])
            if log['hash_valid']:
                path_success_counts[path_tuple] += 1
            else:
                path_tamper_counts[path_tuple] += 1

        path_drop_counts = {tuple(path): len(
            drops) for path, drops in self.dropped_packets_tracker.items() if len(drops) > 0}
        node_drop_participation = defaultdict(int)
        for p, d_count in path_drop_counts.items():
            for nid in p[1:-1]:
                node_drop_participation[nid] += d_count

        culprit_drop_per_path = {}
        for p in path_drop_counts.keys():
            best_culprit = None
            max_part = -1
            for nid in p[1:-1]:
                if node_drop_participation[nid] > max_part:
                    max_part = node_drop_participation[nid]
                    best_culprit = nid
            culprit_drop_per_path[p] = best_culprit

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
            best_culprit = None
            max_part = -1
            for nid in p[1:-1]:
                if node_tamper_participation[nid] > max_part:
                    max_part = node_tamper_participation[nid]
                    best_culprit = nid
            culprit_tamper_per_path[p] = best_culprit

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

            if N_recv[nid] > 0:
                pdr = N_fwd[nid] / N_recv[nid]
                t2_new = beta * t2_prev + (1.0 - beta) * pdr
            else:
                t2_new = t2_prev

            state['T1_history'] = t1_new
            state['T2_history'] = t2_new
            state['trust_vector'][0] = t1_new
            state['trust_vector'][1] = t2_new

        P = len(self.packet_logs)
        N = len(nodes)

        if P > 0 and N > 0:
            node_to_idx = {nid: idx for idx, nid in enumerate(nodes)}
            A = np.zeros((P, N))
            Y = np.zeros(P)
            for p_idx, log in enumerate(self.packet_logs):
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
            for log in self.packet_logs:
                for nid in log['path'][1:-1]:
                    queue_load[nid] += 1

            for nid in nodes:
                if nid == self.BS_ID:
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
        for log in self.packet_logs:
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

            crit_seq_set = {log['seq'] for log in self.packet_logs if log['source'] == nid and log.get(
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
            E_est = E_max - cumulative_consumption + \
                (self.eta_charging * total_injected)
            E_est = max(0.0, min(E_max, E_est))
            state['E_est'] = E_est

            ratio_est = E_est / max(E_max, 1e-12)
            deviation = abs(E_claimed - E_est) / max(E_max, 1e-12)
            penalty_factor = math.exp(-nu_t5 * deviation)
            spoof_trust = state.get('spoof_trust', 1.0)
            t5_current = ratio_est * penalty_factor * spoof_trust

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
        records = list(self.path_history)
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

            if (self.control_epoch_index > self.RISK_WARMUP_EPOCHS and risk_lcb >= self.RISK_BLACKLIST_LCB):
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
        valid_nodes = [nid for nid, state in self.node_states.items(
        ) if state['status'] != 'BLACKLISTED']
        if not valid_nodes:
            return {}

        X_coords = np.array(
            [[self.node_states[nid]['x'], self.node_states[nid]['y']] for nid in valid_nodes])
        K = max(1, int(math.sqrt(len(valid_nodes) / 2)))
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_coords)

        clusters = defaultdict(list)
        for idx, nid in enumerate(valid_nodes):
            clusters[labels[idx]].append(nid)

        self.cluster_heads = {}
        bs_node = self.network_nodes.get(self.BS_ID)

        for cluster_id, members in clusters.items():
            head_candidates = [
                m for m in members
                if self.node_states[m].get('status') not in ('BLACKLISTED', 'QUARANTINE')
            ] or members

            def _cluster_head_score(member_id):
                state = self.node_states[member_id]
                node = self.network_nodes[member_id]
                fe_score = float(state.get('FE', 0.0))
                residual_ratio = float(state.get('E_est', state.get(
                    'E_initial', 1.0))) / max(float(state.get('E_initial', 1.0)), 1e-12)

                local_degree = 0
                for other_id in members:
                    if other_id == member_id:
                        continue
                    other = self.network_nodes[other_id]
                    if math.hypot(node.x - other.x, node.y - other.y) <= 30.0:
                        local_degree += 1
                connectivity_score = local_degree / max(len(members) - 1, 1)

                if bs_node is not None:
                    d_bs = math.hypot(node.x - bs_node.x, node.y - bs_node.y)
                    topology_score = 0.5 * connectivity_score + \
                        0.5 * (1.0 / (1.0 + d_bs))
                else:
                    topology_score = connectivity_score

                return (0.60 * fe_score) + (0.25 * residual_ratio) + (0.15 * topology_score)

            self.cluster_heads[cluster_id] = max(
                head_candidates, key=_cluster_head_score)

        return clusters

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
            deviation_mc = abs(
                delta_claimed - (self.eta_charging * E_injected))

            tolerance = 0.05 * E_max
            state['Deviation_MC'] = deviation_mc
            state['Deviation_MC_norm'] = min(
                1.0, deviation_mc / max(tolerance, 1e-12))

            if deviation_mc > (2.0 * tolerance):
                if E_injected > 0 and delta_claimed <= tolerance:
                    self.metrics.charge_hoarder_attempts += 1
                    self.metrics.charge_hoarder_detections += 1
                    print(
                        f"[WRSN SECURITY] Charge Hoarding detected at Node {nid}! Hard veto triggered.")
                    state['hard_veto'] = True
                    state['status'] = 'BLACKLISTED'
                    state['FE'] = 0.0
                elif E_injected == 0 and delta_claimed > tolerance:
                    self.metrics.fake_charger_attempts += 1
                    self.metrics.fake_charger_detections += 1
                    print(
                        f"[WRSN SECURITY] Vampire Attack detected at Node {nid}! Hard veto triggered.")
                    state['hard_veto'] = True
                    state['status'] = 'BLACKLISTED'
                    state['FE'] = 0.0

            t2_current = state.get('T2_history', 1.0)
            already_rescued = state.get('spoof_rescue_triggered', False)

            if t2_current > 0.8:
                state['spoof_rescue_triggered'] = False

            if t2_current < 0.5 and delta_claimed <= 0 and E_injected == 0 and not already_rescued:
                print(
                    f"[WRSN SECURITY] Spoofing Attack detected affecting Node {nid}! Triggering Incident Response.")
                self.trigger_incident_response(
                    victim_id=nid, spoof_x=state['x'], spoof_y=state['y'])
                state['spoof_rescue_triggered'] = True

            state['E_claimed_prev'] = E_claimed_t
            self.mc_recent_injected_energy[nid] = 0.0

    def validate_charging_request(self, node_id, emergency=False):
        ok = False
        if not self.ablation.get('use_secure_charging_request_validation', True):
            self.metrics.mc_raw_sos_dispatches += 1
            ok = node_id in self.node_states and node_id != self.BS_ID
        else:
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
            if node_id in self.pending_sos_requests or node_id in self.secure_charging_queue:
                current_t = float(
                    getattr(self.metrics, 'current_sim_time', 0.0))
                self.metrics.record_charging_request(
                    node_id, current_t, validated=ok)
        return ok

    def _joint_optimization(self, U_malicious_dict, network_clusters):
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

        route_signature = tuple(sorted(nid for nid in self.node_states if nid !=
                                self.BS_ID and self.node_states[nid].get('status') == 'BLACKLISTED'))
        warning_set = tuple(sorted(nid for nid in self.node_states if nid != self.BS_ID and (self.node_states[nid].get(
            'FE', 1.0) <= self.T_WARNING_FE or self.node_states[nid].get('status') in ('WATCHLIST', 'QUARANTINE'))))
        shadow_signature = warning_set
        force_periodic_refresh = (
            self.ROUTE_RECOMPUTE_PERIOD is not None and self.control_epoch_index % self.ROUTE_RECOMPUTE_PERIOD == 0)
        active_changed = route_signature != self._last_route_signature
        shadow_changed = shadow_signature != self._last_shadow_signature
        self.routes_updated = False

        if critical_nodes and self.standby_routing_tables is not None:
            _shadow_activation_t0 = time.time()
            self.routing_tables = self.standby_routing_tables
            self.standby_routing_tables = None
            self._last_route_signature = route_signature
            self.routes_updated = True
            self.metrics.record_shadow_route_result(
                hit=True, shadow_ms=(time.time() - _shadow_activation_t0) * 1000.0)
        elif active_changed or force_periodic_refresh or not self.routing_tables:
            _ordinary_route_t0 = time.time()
            self.routing_tables = self.routing_engine.compute_routing_tables(
                active_nodes, node_penalties)
            self._last_route_signature = route_signature
            self.routes_updated = True
            self.metrics.record_shadow_route_result(
                ordinary_ms=(time.time() - _ordinary_route_t0) * 1000.0)

        if needs_shadow_calc:
            if shadow_changed or force_periodic_refresh or self.standby_routing_tables is None:
                _shadow_pre_t0 = time.time()
                self.standby_routing_tables = self.routing_engine.compute_routing_tables(
                    shadow_active_nodes, shadow_penalties)
                self._last_shadow_signature = shadow_signature
                self.metrics.record_shadow_route_result(
                    precomputed=True, extra_compute_ms=(time.time() - _shadow_pre_t0) * 1000.0)
        else:
            self.standby_routing_tables = None
            self._last_shadow_signature = None

        secure_sos_queue = [
            nid for nid in self.pending_sos_requests if self.validate_charging_request(nid, emergency=True)]
        self.secure_charging_queue = secure_sos_queue if self.energy_experiment_active else []
        self.pending_sos_requests.clear()

    def verify_charging_location(self, node_id, mc_x, mc_y):
        self.metrics.mc_location_spoof_attempts += 1
        if not self.ablation.get('use_mc_location_verification', True):
            if node_id in self.node_states:
                self.node_states[node_id]['spoof_trust'] = 1.0
            self.metrics.mc_location_spoof_accepted += 1
            return
        if node_id in self.node_states:
            claimed_x = self.node_states[node_id]['x']
            claimed_y = self.node_states[node_id]['y']
            distance = math.hypot(mc_x - claimed_x, mc_y - claimed_y)
            if distance > 5.0:
                self.node_states[node_id]['spoof_trust'] = 0.0
            else:
                self.node_states[node_id]['spoof_trust'] = 1.0

    def report_mc_injection_oob(self, node_id, injected_energy, mc_x=None, mc_y=None, mac_tag=None, mc_id=None, counter=None, nonce=None, timestamp=None):
        auth_enabled = self.ablation.get('use_mc_hmac_authentication', True)
        if node_id not in self.node_states:
            return False

        if not auth_enabled:
            fake_or_missing = (mac_tag is None or mc_x is None or mc_y is None)
            if fake_or_missing:
                self.metrics.mc_fake_reports_attempted += 1
            self.mc_recent_injected_energy[node_id] += float(injected_energy)
            current_injected = self.node_states[node_id].get(
                'total_injected_energy', 0.0)
            self.node_states[node_id]['total_injected_energy'] = current_injected + \
                float(injected_energy)
            if fake_or_missing:
                self.metrics.mc_fake_reports_accepted += 1
            return True

        shared_key = getattr(self, 'mc_shared_key', None)
        required_missing = (
            shared_key is None or mac_tag is None or mc_x is None or mc_y is None or
            mc_id is None or counter is None or nonce is None or timestamp is None
        )
        if required_missing:
            self.metrics.mc_fake_reports_attempted += 1
            self.metrics.mc_rejected_unauth_reports += 1
            self.mc_fail_closed_rejections += 1
            return False

        counter = int(counter)
        nonce = str(nonce)
        mc_key = str(mc_id)
        if counter <= self.mc_last_counter[mc_key] or nonce in self.mc_seen_nonces[mc_key]:
            self.metrics.mc_fake_reports_attempted += 1
            self.metrics.mc_rejected_unauth_reports += 1
            self.metrics.mc_replay_rejections += 1
            self.mc_replay_rejections += 1
            return False

        expected_tag = make_mc_mac(shared_key, node_id, injected_energy, mc_x,
                                   mc_y, mc_id=mc_id, counter=counter, nonce=nonce, timestamp=timestamp)
        if expected_tag is None or not hmac.compare_digest(str(mac_tag), str(expected_tag)):
            self.metrics.mc_fake_reports_attempted += 1
            self.metrics.mc_rejected_unauth_reports += 1
            return False

        self.mc_last_counter[mc_key] = counter
        self.mc_seen_nonces[mc_key].append(nonce)
        self.mc_recent_injected_energy[node_id] += float(injected_energy)
        current_injected = self.node_states[node_id].get(
            'total_injected_energy', 0.0)
        self.node_states[node_id]['total_injected_energy'] = current_injected + \
            float(injected_energy)
        return True

    def trigger_incident_response(self, victim_id, spoof_x, spoof_y):
        if not self.energy_experiment_active:
            self.pending_sos_requests.add(victim_id)
            return
        token = secrets.token_hex(8)
        self.zone_wakeup_tokens[(spoof_x, spoof_y)] = token

        if self.mc_reference is not None:
            victim_node = self.network_nodes[victim_id]
            self.mc_reference.interrupt_and_rescue(victim_node)
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

    def _verify_stealthy_attackers(self):
        if not self.ablation.get('use_stealthy_attacker_detection', True):
            return
        kappa = 5.0
        VETO_THRESHOLD = 0.4
        for nid, state in self.node_states.items():
            if nid == self.BS_ID or state.get('status') == 'BLACKLISTED':
                continue
            T2 = state.get('T2_history', 1.0)
            E_est = state.get('E_est', state['E_initial'])
            E_claimed = state.get('E_claimed', state['E_initial'])
            E_max = state['E_initial']
            drop_rate = 1.0 - T2
            energy_diff = max(0.0, E_est - E_claimed)
            alibi_penalty = 1.0 - math.exp(-kappa * (energy_diff / E_max))
            S_stealth = drop_rate * alibi_penalty
            state['S_stealth'] = S_stealth
            if S_stealth >= VETO_THRESHOLD:
                print(
                    f"[WRSN SECURITY] Stealthy Attacker CAUGHT! Node {nid} used Battery Alibi. S_stealth={S_stealth:.3f}")
                state['hard_veto'] = True
                state['status'] = 'BLACKLISTED'
                state['FE'] = 0.0

    def _verify_trajectory_manipulators(self):
        if not self.ablation.get('use_sed_filter', self.ablation.get('use_zero_overhead_spoof_detection', True)):
            return
        bs_x, bs_y = self.network_nodes[self.BS_ID].x, self.network_nodes[self.BS_ID].y
        non_bs_states = [state for nid,
                         state in self.node_states.items() if nid != self.BS_ID]
        if not non_bs_states:
            return
        D_max = max(math.hypot(state['x'] - bs_x, state['y'] - bs_y)
                    for state in non_bs_states)
        D_max = max(D_max, 1e-6)
        epsilon = 1e-6

        for nid, state in self.node_states.items():
            if nid == self.BS_ID:
                continue
            e_prev = state.get('E_claimed_prev_SED', state['E_initial'])
            e_curr = state.get('E_claimed', state['E_initial'])
            state['delta_E'] = max(0.0, e_prev - e_curr)
            state['E_claimed_prev_SED'] = e_curr

        for nid in list(self.pending_sos_requests):
            state = self.node_states.get(nid)
            if not state or state.get('status') == 'BLACKLISTED':
                self.pending_sos_requests.discard(nid)
                continue

            delta_E_i = state.get('delta_E', 0.0)
            neighbors_delta_e = []
            for other_nid, other_state in self.node_states.items():
                if other_nid != nid and other_nid != self.BS_ID:
                    dist = math.hypot(
                        state['x'] - other_state['x'], state['y'] - other_state['y'])
                    if dist <= 30.0:
                        neighbors_delta_e.append(
                            other_state.get('delta_E', 0.0))

            if len(neighbors_delta_e) > 0:
                mu_delta_e = float(np.mean(neighbors_delta_e))
                sigma_delta_e = float(np.std(neighbors_delta_e))
            else:
                mu_delta_e = 0.0
                sigma_delta_e = 0.0

            D_BS_i = math.hypot(state['x'] - bs_x, state['y'] - bs_y)
            z_score = abs(delta_E_i - mu_delta_e) / (sigma_delta_e + epsilon)
            distance_penalty = 1.0 + (D_BS_i / D_max)
            SED_index = z_score * distance_penalty
            state['SED_index'] = SED_index

            if SED_index >= 3.0:
                print(
                    f"[WRSN SECURITY] Trajectory Manipulator CAUGHT! Node {nid} SED={SED_index:.2f}. Fake SOS Dropped.")
                state['hard_veto'] = True
                state['status'] = 'BLACKLISTED'
                state['FE'] = 0.0
                self.pending_sos_requests.discard(nid)

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
