from __future__ import annotations

import math
from heapq import heappop, heappush
from itertools import count
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from config import DATA_PACKET_BITS, DATA_PACKET_BYTES


def _dijkstra_targets_exact(graph, source, targets, weight='weight', *, paths=False):
    """NetworkX-equivalent Dijkstra, stopped after requested targets settle.

    The relaxation order and floating-point operations mirror NetworkX's
    single-source Dijkstra implementation. Only unnecessary exploration after
    all requested targets are final is omitted.
    """
    requested = set(targets)
    if not requested:
        return ({}, {}) if paths else {}
    if source not in graph:
        raise nx.NodeNotFound(f"Node {source} not found in graph")

    if callable(weight):
        weight_fn = weight
    elif graph.is_multigraph():
        weight_fn = lambda u, v, data: min(
            attr.get(weight, 1) for attr in data.values())
    else:
        weight_fn = lambda u, v, data: data.get(weight, 1)

    adjacency = graph._adj
    final = {}
    seen = {source: 0}
    predecessor = {} if paths else None
    counter = count()
    fringe = [(0, next(counter), source)]
    remaining = set(requested)

    while fringe:
        dist_v, _, node = heappop(fringe)
        if node in final:
            continue
        final[node] = dist_v
        remaining.discard(node)
        if not remaining:
            break

        for neighbor, edge_data in adjacency[node].items():
            cost = weight_fn(node, neighbor, edge_data)
            if cost is None:
                continue
            candidate = dist_v + cost
            if neighbor in final:
                if candidate < final[neighbor]:
                    raise ValueError(
                        'Contradictory paths found:', 'negative weights?')
            elif neighbor not in seen or candidate < seen[neighbor]:
                seen[neighbor] = candidate
                heappush(fringe, (candidate, next(counter), neighbor))
                if predecessor is not None:
                    predecessor[neighbor] = node

    target_distances = {
        target: final[target] for target in requested if target in final
    }
    if not paths:
        return target_distances

    target_paths = {}
    for target in requested:
        if target not in final:
            continue
        path = [target]
        current = target
        while current != source:
            current = predecessor[current]
            path.append(current)
        path.reverse()
        target_paths[target] = path
    return target_distances, target_paths


class BaselineRouting:
    """Reference routing used by the original control/charging plane.

    This class intentionally preserves the public API and the original routing
    methodology.  The enhanced clustered routing is implemented separately in
    :class:`DataPlaneHybridRouting`, so mobile-charger admission, QIFL, DQN,
    charging deadlines, fixed chargers, and control-plane routing continue to
    use the same reference tables as the original method.
    """

    def __init__(self, bs_id, transmission_range=50.0, max_critical_paths=2,
                 data_packet_bits=DATA_PACKET_BITS, config=None):
        self.bs_id = bs_id
        self.transmission_range = transmission_range
        self.max_critical_paths = max_critical_paths
        self.quarantined_zones = []
        self.data_packet_bits = data_packet_bits
        self.config = dict(config or {})

    def _in_quarantine(self, node_id, active_nodes_dict):
        node = active_nodes_dict.get(node_id)
        if node and self.quarantined_zones:
            for zone in self.quarantined_zones:
                if math.hypot(node.x - zone['x'], node.y - zone['y']) <= zone['radius']:
                    return True
        return False

    def _calculate_adaptive_params(self, active_nodes_dict, node_penalties):
        valid_nodes = [
            n for nid, n in active_nodes_dict.items()
            if nid != self.bs_id and n.status == "ALIVE"
        ]
        n_alive = len(valid_nodes)
        if n_alive == 0:
            return 0.33, 0.33, 0.34, 1

        energies = np.array([max(0.0, float(n.E_current))
                            for n in valid_nodes])
        max_energies = np.array([max(1e-9, float(n.E_max))
                                for n in valid_nodes])
        energy_ratios = energies / max_energies
        avg_e_ratio = float(np.mean(energy_ratios))
        std_e = float(np.std(energies))
        mean_e = float(np.mean(energies))
        cv_e = std_e / mean_e if mean_e > 1e-9 else 0.0

        u_e = 1.0 - avg_e_ratio
        penalized_count = sum(
            1 for nid in active_nodes_dict
            if node_penalties.get(nid, 1.0) > 1.5
            or self._in_quarantine(nid, active_nodes_dict)
        )
        u_p = penalized_count / float(n_alive)
        u_d = 0.5
        tau = 0.5
        urgencies = np.array([u_d, u_e, u_p])
        exp_u = np.exp(urgencies / tau)
        weights = exp_u / np.sum(exp_u)
        w_d, w_e, w_p = float(weights[0]), float(weights[1]), float(weights[2])

        k_min = 2
        base_k = math.sqrt(n_alive / (2 * math.pi))
        k_opt = max(k_min, int(round(base_k * avg_e_ratio * (1 + cv_e))))
        return w_d, w_e, w_p, k_opt

    def build_network_graph(self, active_nodes_dict, node_penalties, w_d, w_e, w_p):
        graph = nx.DiGraph()
        for n_id, node in active_nodes_dict.items():
            graph.add_node(n_id, pos=(node.x, node.y))

        node_ids = list(active_nodes_dict.keys())
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                u, v = node_ids[i], node_ids[j]
                n1, n2 = active_nodes_dict[u], active_nodes_dict[v]
                if n1.status != "ALIVE" or n2.status != "ALIVE":
                    continue
                dist = math.hypot(n1.x - n2.x, n1.y - n2.y)
                if dist > self.transmission_range:
                    continue

                def calc_cost(dst_node, dst_id):
                    norm_dist = dist / max(self.transmission_range, 1e-9)
                    # Preserve the original formula for ordinary nodes while
                    # preventing the BS inf/inf NaN that made Dijkstra unstable.
                    if dst_id == self.bs_id:
                        inv_energy = 0.0
                    else:
                        inv_energy = float(dst_node.E_max) / \
                            max(float(dst_node.E_current), 1e-9)
                    penalty = (
                        float('inf') if self._in_quarantine(dst_id, active_nodes_dict)
                        else node_penalties.get(dst_id, 1.0)
                    )
                    return (w_d * norm_dist) + (w_e * inv_energy) + (w_p * penalty)

                c_uv = calc_cost(n2, v)
                c_vu = calc_cost(n1, u)
                if math.isfinite(c_uv):
                    graph.add_edge(u, v, weight=float(
                        c_uv), distance=float(dist))
                if math.isfinite(c_vu):
                    graph.add_edge(v, u, weight=float(
                        c_vu), distance=float(dist))
        return graph

    def _fuzzy_ch_selection(self, active_nodes, node_penalties, k_opt):
        candidates = []
        for nid, node in active_nodes.items():
            if nid == self.bs_id or node.status != "ALIVE":
                continue
            penalty = node_penalties.get(nid, 1.0)
            if penalty == float('inf') or self._in_quarantine(nid, active_nodes):
                continue
            e_ratio = float(node.E_current) / max(float(node.E_max), 1e-9)
            dist_to_bs = math.hypot(
                node.x - active_nodes[self.bs_id].x,
                node.y - active_nodes[self.bs_id].y,
            )
            score = (
                e_ratio * 0.5
                + (1.0 / max(penalty, 1.0)) * 0.3
                + (1.0 / max(dist_to_bs, 1.0)) * 0.2
            )
            candidates.append((score, nid, node))
        candidates.sort(key=lambda x: x[0], reverse=True)

        heads = []
        min_dist = self.transmission_range * 0.65
        for _, nid, node in candidates:
            if len(heads) >= k_opt:
                break
            if all(
                math.hypot(
                    node.x - active_nodes[h].x, node.y - active_nodes[h].y) >= min_dist
                for h in heads
            ):
                heads.append(nid)
        if len(heads) < k_opt:
            for _, nid, _ in candidates:
                if nid not in heads:
                    heads.append(nid)
                if len(heads) >= k_opt:
                    break
        return heads

    def _fcbe_cluster_formation(self, active_nodes, heads, node_penalties):
        clusters = {head: [] for head in heads}
        members = [
            nid for nid in active_nodes
            if nid != self.bs_id and nid not in heads
            and active_nodes[nid].status == "ALIVE"
        ]
        for nid in members:
            node = active_nodes[nid]
            if node_penalties.get(nid, 1.0) == float('inf') or self._in_quarantine(nid, active_nodes):
                continue
            best_head, best_score = None, -float('inf')
            for head in heads:
                hnode = active_nodes[head]
                dist = math.hypot(node.x - hnode.x, node.y - hnode.y)
                if dist > self.transmission_range * 1.5:
                    continue
                score = 1.0 / max(dist, 1e-3) - 0.05 * len(clusters[head])
                if score > best_score:
                    best_head, best_score = head, score
            if best_head is not None:
                clusters[best_head].append(nid)
        return clusters

    @staticmethod
    def _empty_table():
        return {
            'primary': None, 'secondary': None,
            'primary_path': [], 'secondary_path': [], 'critical_paths': [],
        }

    def compute_routing_tables(self, active_nodes_dict, node_penalties=None):
        node_penalties = dict(
            node_penalties or {nid: 1.0 for nid in active_nodes_dict})
        w_d, w_e, w_p, k_opt = self._calculate_adaptive_params(
            active_nodes_dict, node_penalties
        )
        graph = self.build_network_graph(
            active_nodes_dict, node_penalties, w_d, w_e, w_p
        )
        tables = {
            nid: self._empty_table()
            for nid in active_nodes_dict if nid != self.bs_id
        }
        valid_nodes = [
            nid for nid in active_nodes_dict
            if nid != self.bs_id and active_nodes_dict[nid].status == "ALIVE"
        ]
        if not valid_nodes:
            return tables

        heads = self._fuzzy_ch_selection(
            active_nodes_dict, node_penalties, k_opt)
        clusters = self._fcbe_cluster_formation(
            active_nodes_dict, heads, node_penalties)

        macro = nx.DiGraph()
        macro.add_node(self.bs_id)
        for head in heads:
            macro.add_node(head)
            for other in [self.bs_id] + heads:
                if head == other:
                    continue
                try:
                    path = nx.shortest_path(
                        graph, head, other, weight='weight')
                    macro.add_edge(head, other, weight=nx.path_weight(
                        graph, path, 'weight'))
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass

        macro_u = nx.Graph()
        for u, v, data in macro.edges(data=True):
            reverse = macro[v][u]['weight'] if macro.has_edge(
                v, u) else data['weight']
            macro_u.add_edge(u, v, weight=min(data['weight'], reverse))
        try:
            mst = nx.minimum_spanning_tree(macro_u, weight='weight')
            mst_dir = nx.DiGraph()
            if self.bs_id in mst:
                for u, v in nx.bfs_edges(mst, source=self.bs_id):
                    mst_dir.add_edge(v, u)
        except Exception:
            mst_dir = nx.DiGraph()

        for head in heads:
            try:
                macro_path = nx.shortest_path(mst_dir, head, self.bs_id)
                full = [head]
                for u, v in zip(macro_path[:-1], macro_path[1:]):
                    segment = nx.shortest_path(graph, u, v, weight='weight')
                    full.extend(segment[1:])
                tables[head]['primary_path'] = full
                tables[head]['primary'] = full[1] if len(full) > 1 else None
            except Exception:
                try:
                    path = nx.shortest_path(
                        graph, head, self.bs_id, weight='weight')
                    tables[head]['primary_path'] = path
                    tables[head]['primary'] = path[1] if len(
                        path) > 1 else None
                except Exception:
                    pass

        for head, members in clusters.items():
            head_path = tables.get(head, {}).get('primary_path') or []
            for member in members:
                try:
                    to_head = nx.shortest_path(
                        graph, member, head, weight='weight')
                    full = to_head[:-1] + head_path if head_path else to_head
                    if full and full[-1] == self.bs_id and len(set(full)) == len(full):
                        tables[member]['primary_path'] = full
                        tables[member]['primary'] = full[1] if len(
                            full) > 1 else None
                except Exception:
                    pass

        # Reference fallback for any unassigned node.
        for nid in valid_nodes:
            if tables[nid]['primary_path']:
                continue
            try:
                path = nx.shortest_path(
                    graph, nid, self.bs_id, weight='weight')
                tables[nid]['primary_path'] = path
                tables[nid]['primary'] = path[1] if len(path) > 1 else None
            except Exception:
                pass

        # Preserve the original secondary construction for the reference plane.
        for nid in valid_nodes:
            primary = tables[nid]['primary_path']
            if primary:
                tables[nid]['critical_paths'].append(primary)
            try:
                if len(primary) > 2:
                    forbidden = set(primary[1:-1])

                    def node_disjoint_weight(u, v, data, blocked=forbidden):
                        if u in blocked or v in blocked:
                            return None
                        return data.get('weight', 1.0)

                    secondary = nx.shortest_path(
                        graph, nid, self.bs_id, weight=node_disjoint_weight)
                else:
                    blocked_edge = (nid, self.bs_id)

                    def secondary_weight(u, v, data, blocked=blocked_edge):
                        if (u, v) == blocked:
                            return None
                        return data.get('weight', 1.0)

                    secondary = nx.shortest_path(
                        graph, nid, self.bs_id, weight=secondary_weight)
                if len(secondary) > 1 and secondary != primary:
                    tables[nid]['secondary_path'] = secondary
                    tables[nid]['secondary'] = secondary[1]
                    tables[nid]['critical_paths'].append(secondary)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
        return tables


class DataPlaneHybridRouting:
    """Cluster/MST routing overlay restricted to DATA packets only.

    The algorithm starts from the original reference route assignment and only
    replaces a primary or secondary DATA route when a dominance guard confirms
    that the replacement does not increase the projected radio-energy stress of
    the network.  Charging and control routing tables are never modified.
    """

    def __init__(self, bs_id, transmission_range=50.0, max_critical_paths=2,
                 data_packet_bits=DATA_PACKET_BITS, config=None):
        self.bs_id = int(bs_id)
        self.transmission_range = float(transmission_range)
        self.max_critical_paths = max(1, int(max_critical_paths))
        self.data_packet_bits = int(data_packet_bits)
        self.data_packet_bytes = max(
            1, int(math.ceil(self.data_packet_bits / 8.0)))
        self.config = dict(config or {})
        self.quarantined_zones = []
        self.last_cluster_heads: List[int] = []
        self.last_clusters: Dict[int, List[int]] = {}
        self.last_diagnostics: Dict[str, float] = {}
        self.last_graph: Optional[nx.DiGraph] = None

        # Per-computation memoization. These caches are cleared whenever a new
        # routing graph is built, so they cannot leak values across epochs.
        self._path_vector_cache = {}
        self._path_distance_cache = {}
        self._path_risk_cache = {}
        self._stress_energy_cache = {}

    def _cfg(self, key, default):
        return self.config.get(key, default)

    @staticmethod
    def _finite(value, default=0.0):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return float(default)
        return value if math.isfinite(value) else float(default)

    def _alive(self, nid, node):
        return nid == self.bs_id or str(getattr(node, 'status', 'ALIVE')).upper() == 'ALIVE'

    def _in_quarantine(self, nid, nodes):
        if nid == self.bs_id:
            return False
        node = nodes.get(nid)
        if node is None:
            return True
        for zone in self.quarantined_zones or []:
            if math.hypot(float(node.x) - float(zone['x']), float(node.y) - float(zone['y'])) <= float(zone['radius']):
                return True
        return False

    def _blocked(self, nid, nodes, penalties):
        if nid == self.bs_id:
            return False
        if self._in_quarantine(nid, nodes):
            return True
        try:
            return not math.isfinite(float(penalties.get(nid, 1.0)))
        except (TypeError, ValueError):
            return True

    @staticmethod
    def _radio_tx_energy(node, packet_bytes, distance):
        bits = float(getattr(node, 'data_packet_bits',
                     float(packet_bytes) * 8.0))
        e_elec = float(getattr(node, 'E_elec', 50e-9))
        eps_fs = float(getattr(node, 'eps_fs', 10e-12))
        eps_mp = float(getattr(node, 'eps_mp', 0.0013e-12))
        d0 = float(getattr(node, 'd0', math.sqrt(eps_fs / max(eps_mp, 1e-30))))
        if distance <= d0:
            return bits * (e_elec + eps_fs * distance ** 2)
        return bits * (e_elec + eps_mp * distance ** 4)

    @staticmethod
    def _radio_rx_energy(node, packet_bytes):
        bits = float(getattr(node, 'data_packet_bits',
                     float(packet_bytes) * 8.0))
        return bits * float(getattr(node, 'E_elec', 50e-9))

    def _adaptive_weights(self, nodes, penalties):
        valid = [
            node for nid, node in nodes.items()
            if nid != self.bs_id and self._alive(nid, node)
            and not self._blocked(nid, nodes, penalties)
        ]
        if not valid:
            return 0.60, 0.25, 0.15, 1
        ratios = np.asarray([
            np.clip(
                self._finite(node.E_current) /
                max(self._finite(node.E_max, 1.0), 1e-12),
                0.0, 1.0,
            ) for node in valid
        ])
        avg = float(np.mean(ratios))
        cv = float(np.std(ratios) / max(avg, 1e-9))
        penalized = sum(
            1 for nid, node in nodes.items()
            if nid != self.bs_id and self._alive(nid, node)
            and self._finite(penalties.get(nid, 1.0), 1.0) > 1.5
        )
        if bool(self._cfg('use_adaptive_greedy_weights', True)):
            urgency = np.asarray(
                [0.5, 1.0 - avg, penalized / max(len(valid), 1)], dtype=float)
            tau = max(
                float(self._cfg('data_routing_weight_temperature', 1.5599965406582)), 1e-3)
            logits = urgency / tau
            logits -= logits.max()
            soft = np.exp(np.clip(logits, -60.0, 0.0))
            soft /= max(float(soft.sum()), 1e-12)
            floor = float(
                np.clip(self._cfg('data_routing_weight_floor', 0.0259280212224), 0.0, 0.25))
            soft = floor + (1.0 - 3.0 * floor) * soft
            soft /= max(float(soft.sum()), 1e-12)
        else:
            soft = np.asarray(self._cfg('data_static_weights',
                              (0.55, 0.30, 0.15)), dtype=float)
            if soft.shape != (3,) or not np.all(np.isfinite(soft)):
                soft = np.asarray((0.55, 0.30, 0.15), dtype=float)
            soft = np.maximum(soft, 0.0)
            soft /= max(float(soft.sum()), 1e-12)
        k_base = math.sqrt(len(valid) / (2.0 * math.pi))
        k_opt = max(2, int(round(k_base * avg * (1.0 + min(cv, 1.25)))))
        k_opt = min(k_opt, max(
            2, int(math.ceil(math.sqrt(len(valid))))), len(valid))
        return float(soft[0]), float(soft[1]), float(soft[2]), int(k_opt)

    def _build_graph(self, nodes, penalties):
        w_d, w_e, w_p, k_opt = self._adaptive_weights(nodes, penalties)
        graph = nx.DiGraph()
        usable = [
            nid for nid, node in nodes.items()
            if self._alive(nid, node) and not self._blocked(nid, nodes, penalties)
        ]
        if self.bs_id in nodes and self.bs_id not in usable:
            usable.append(self.bs_id)
        for nid in usable:
            node = nodes[nid]
            graph.add_node(nid, pos=(float(node.x), float(node.y)))

        max_hop_energy = 1e-12
        for nid in usable:
            if nid == self.bs_id:
                continue
            node = nodes[nid]
            max_hop_energy = max(
                max_hop_energy,
                self._radio_tx_energy(
                    node, self.data_packet_bytes, self.transmission_range)
                + self._radio_rx_energy(node, self.data_packet_bytes),
            )

        for i, u in enumerate(usable):
            nu = nodes[u]
            for v in usable[i + 1:]:
                nv = nodes[v]
                distance = math.hypot(
                    float(nu.x) - float(nv.x), float(nu.y) - float(nv.y))
                if distance > self.transmission_range + 1e-9:
                    continue
                for src, dst, nsrc, ndst in ((u, v, nu, nv), (v, u, nv, nu)):
                    tx = 0.0 if src == self.bs_id else self._radio_tx_energy(
                        nsrc, self.data_packet_bytes, distance)
                    rx = 0.0 if dst == self.bs_id else self._radio_rx_energy(
                        ndst, self.data_packet_bytes)
                    radio = tx + rx
                    radio_norm = radio / max_hop_energy
                    if dst == self.bs_id:
                        energy_stress = 0.0
                        security_stress = 0.0
                    else:
                        e_ratio = np.clip(
                            self._finite(ndst.E_current) /
                            max(self._finite(ndst.E_max, 1.0), 1e-12),
                            0.0, 1.0,
                        )
                        energy_stress = float((1.0 - e_ratio) ** 1.5)
                        penalty = max(1.0, self._finite(
                            penalties.get(dst, 1.0), 1.0))
                        cap = max(
                            float(self._cfg('data_routing_penalty_cap', 10.0)), 1.01)
                        security_stress = float(
                            np.clip(math.log1p(penalty - 1.0) / math.log(cap), 0.0, 1.0))
                    distance_norm = (
                        distance / max(self.transmission_range, 1e-9)) ** 2
                    weight = (
                        0.45 * radio_norm
                        + 0.20 * w_d * distance_norm
                        + 0.20 * w_e * energy_stress
                        + 0.15 * w_p * security_stress
                        + 1e-6
                    )
                    graph.add_edge(
                        src, dst, weight=float(weight), distance=float(distance),
                        tx_energy=float(tx), rx_energy=float(rx),
                        radio_energy=float(radio), energy_stress=float(energy_stress),
                        security_stress=float(security_stress),
                    )
        self.last_graph = graph
        return graph, k_opt

    @staticmethod
    def _valid_path(graph, path, source, bs_id):
        path = list(path or [])
        return (
            len(path) >= 2 and path[0] == source and path[-1] == bs_id
            and len(set(path)) == len(path)
            and all(graph.has_edge(u, v) for u, v in zip(path[:-1], path[1:]))
        )

    def _reference_path(self, graph, source, reference_tables):
        ref = list((reference_tables.get(source, {})
                   or {}).get('primary_path') or [])
        if self._valid_path(graph, ref, source, self.bs_id):
            return ref
        try:
            return nx.shortest_path(graph, source, self.bs_id, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def _path_vector(self, graph, path):
        path_seq = list(path or [])
        key = tuple(path_seq)
        cached = self._path_vector_cache.get(key)
        if cached is not None:
            return cached

        vector = defaultdict(float)
        if len(path_seq) < 2:
            self._path_vector_cache[key] = vector
            return vector
        source = path_seq[0]
        first = path_seq[1]
        vector[source] += float(graph[source][first].get('tx_energy', 0.0))
        for idx in range(1, len(path_seq) - 1):
            relay = path_seq[idx]
            nxt = path_seq[idx + 1]
            vector[relay] += float(graph[relay][nxt].get('rx_energy', 0.0))
            vector[relay] += float(graph[relay][nxt].get('tx_energy', 0.0))
        self._path_vector_cache[key] = vector
        return vector

    @staticmethod
    def _vector_add(base, delta, factor=1.0):
        out = defaultdict(float, base)
        for key, value in delta.items():
            out[key] += factor * float(value)
        return out

    @staticmethod
    def _vector_total(vector):
        return float(sum(max(0.0, value) for value in vector.values()))

    def _stress_stats(self, burden, nodes):
        stresses = []
        energy_cache = self._stress_energy_cache
        for nid, value in burden.items():
            if nid == self.bs_id or nid not in nodes:
                continue
            energy = energy_cache.get(nid)
            if energy is None:
                energy = max(self._finite(
                    getattr(nodes[nid], 'E_current', 0.0)), 1e-12)
                energy_cache[nid] = energy
            stresses.append(max(0.0, float(value)) / energy)
        if not stresses:
            return 0.0, 0.0
        arr = np.asarray(stresses, dtype=float)
        return float(np.max(arr)), float(np.quantile(arr, 0.95))

    @staticmethod
    def _path_total(vector):
        return float(sum(vector.values()))

    @staticmethod
    def _path_hops(path):
        return max(0, len(list(path or [])) - 1)

    def _path_distance(self, graph, path):
        path_seq = list(path or [])
        key = tuple(path_seq)
        cached = self._path_distance_cache.get(key)
        if cached is not None:
            return cached
        total = 0.0
        for u, v in zip(path_seq[:-1], path_seq[1:]):
            if not graph.has_edge(u, v):
                self._path_distance_cache[key] = math.inf
                return math.inf
            total += float(graph[u][v].get('distance', 0.0))
        self._path_distance_cache[key] = total
        return total

    def _path_risk(self, graph, path):
        path_seq = list(path or [])
        key = tuple(path_seq)
        cached = self._path_risk_cache.get(key)
        if cached is not None:
            return cached
        value = max((float(graph[u][v].get('security_stress', 0.0))
                     for u, v in zip(path_seq[:-1], path_seq[1:])),
                    default=0.0)
        self._path_risk_cache[key] = value
        return value

    def _select_heads(self, graph, nodes, penalties, k_opt):
        if self.bs_id not in nodes:
            return []
        bs = nodes[self.bs_id]
        candidates = []
        max_degree = max((graph.degree(nid)
                         for nid in graph if nid != self.bs_id), default=1)
        max_dist = max((math.hypot(float(nodes[nid].x) - float(bs.x), float(
            nodes[nid].y) - float(bs.y)) for nid in graph if nid != self.bs_id), default=1.0)
        for nid in graph:
            if nid == self.bs_id:
                continue
            node = nodes[nid]
            e_ratio = np.clip(self._finite(node.E_current) /
                              max(self._finite(node.E_max, 1.0), 1e-12), 0.0, 1.0)
            security = 1.0 / \
                max(1.0, self._finite(penalties.get(nid, 1.0), 1.0))
            centrality = 1.0 - \
                math.hypot(float(node.x) - float(bs.x),
                           float(node.y) - float(bs.y)) / max(max_dist, 1e-12)
            degree = graph.degree(nid) / max(max_degree, 1)
            if bool(self._cfg('use_ifbe_fuzzy_ch_selection', True)):
                score = 0.45 * e_ratio + 0.20 * security + 0.15 * centrality + 0.20 * degree
            else:
                score = 0.75 * e_ratio + 0.25 * degree
            candidates.append((score, nid))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        min_spacing = float(
            self._cfg('data_ch_min_spacing_ratio', 0.55)) * self.transmission_range
        heads = []
        for _, nid in candidates:
            node = nodes[nid]
            if all(math.hypot(float(node.x) - float(nodes[h].x), float(node.y) - float(nodes[h].y)) >= min_spacing for h in heads):
                heads.append(nid)
            if len(heads) >= k_opt:
                break
        if len(heads) < k_opt:
            for _, nid in candidates:
                if nid not in heads:
                    heads.append(nid)
                if len(heads) >= k_opt:
                    break
        return heads

    def _assign_clusters(self, graph, nodes, heads):
        clusters = {head: [] for head in heads}
        member_to_head = {}
        if not heads:
            return clusters, member_to_head
        target = max(1.0, (graph.number_of_nodes() - 1 -
                     len(heads)) / max(len(heads), 1))
        balance_lambda = max(
            float(self._cfg('data_fcbe_balance_lambda', 0.30)), 0.0)
        max_cost_ratio = max(
            float(self._cfg('data_member_to_ch_cost_ratio', 1.35)), 1.0)
        for source in sorted(nid for nid in graph if nid != self.bs_id and nid not in heads):
            try:
                # Preserve NetworkX's Dijkstra ordering, but stop as soon as
                # the BS and every candidate head have final distances.
                lengths = _dijkstra_targets_exact(
                    graph, source, set(heads) | {self.bs_id}, weight='weight')
                direct_cost = lengths[self.bs_id]
            except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError):
                continue
            choices = []
            for head in heads:
                cost = lengths.get(head)
                if cost is None:
                    continue
                if cost > max_cost_ratio * max(direct_cost, 1e-12):
                    continue
                load_ratio = (len(clusters[head]) + 1.0) / target
                head_node = nodes[head]
                head_energy = np.clip(self._finite(
                    head_node.E_current) / max(self._finite(head_node.E_max, 1.0), 1e-12), 0.0, 1.0)
                score = cost / max(direct_cost, 1e-12) + \
                    balance_lambda * load_ratio ** 2 - 0.10 * head_energy
                choices.append((score, head))
            if choices:
                if bool(self._cfg('use_fcbe_soft_balancing', True)):
                    _, head = min(choices)
                else:
                    # ``choices`` already contains the exact shortest-path
                    # length, so avoid recomputing it.
                    _, head = min(choices, key=lambda item: lengths[item[1]])
                clusters[head].append(source)
                member_to_head[source] = head
        return clusters, member_to_head

    @staticmethod
    def _undirected(graph):
        out = nx.Graph()
        out.add_nodes_from(graph.nodes(data=True))
        for u, v in graph.edges():
            if out.has_edge(u, v):
                continue
            vals = []
            if graph.has_edge(u, v):
                vals.append(float(graph[u][v]['weight']))
            if graph.has_edge(v, u):
                vals.append(float(graph[v][u]['weight']))
            if vals:
                out.add_edge(u, v, weight=float(np.mean(vals)))
        return out

    def _backbone(self, graph, heads):
        undirected = self._undirected(graph)
        terminals = [self.bs_id] + \
            [head for head in heads if head in undirected]
        metric = nx.Graph()
        physical = {}
        for i, u in enumerate(terminals):
            later_terminals = terminals[i + 1:]
            try:
                lengths, paths = _dijkstra_targets_exact(
                    undirected, u, later_terminals, weight='weight', paths=True)
            except nx.NetworkXError:
                continue
            for v in later_terminals:
                if v in lengths:
                    metric.add_edge(u, v, weight=float(lengths[v]))
                    physical[(u, v)] = list(paths[v])
                    physical[(v, u)] = list(reversed(paths[v]))
        tree = nx.Graph()
        if self.bs_id not in metric:
            return tree
        component = metric.subgraph(
            nx.node_connected_component(metric, self.bs_id)).copy()
        terminal_mst = nx.minimum_spanning_tree(component, weight='weight')
        union = nx.Graph()
        for u, v in terminal_mst.edges():
            path = physical.get((u, v), [])
            for a, b in zip(path[:-1], path[1:]):
                union.add_edge(a, b, weight=float(undirected[a][b]['weight']))
        if self.bs_id not in union:
            return tree
        component = union.subgraph(
            nx.node_connected_component(union, self.bs_id)).copy()
        return nx.minimum_spanning_tree(component, weight='weight')

    def _candidate_paths(self, graph, source, ref_path, head, head_paths, backbone):
        candidates = []
        if head is not None and head in head_paths:
            try:
                to_head = nx.shortest_path(
                    graph, source, head, weight='weight')
                merged = to_head[:-1] + head_paths[head]
                if self._valid_path(graph, merged, source, self.bs_id):
                    candidates.append(('cluster_mst', merged))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

        backbone_edges = frozenset(
            (u, v) for u, v in backbone.edges()
        ) | frozenset((v, u) for u, v in backbone.edges())
        discount = float(
            np.clip(self._cfg('data_backbone_edge_discount', 0.90), 0.50, 1.0))

        def biased_weight(u, v, data):
            return float(data.get('weight', 1.0)) * (discount if (u, v) in backbone_edges else 1.0)
        try:
            path = nx.shortest_path(
                graph, source, self.bs_id, weight=biased_weight)
            if self._valid_path(graph, path, source, self.bs_id):
                candidates.append(('backbone_biased', list(path)))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        try:
            energy_path = nx.shortest_path(
                graph, source, self.bs_id,
                weight=lambda u, v, d: float(
                    d.get('radio_energy', d.get('weight', 1.0)))
            )
            if self._valid_path(graph, energy_path, source, self.bs_id):
                candidates.append(('energy_shortest', list(energy_path)))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

        unique = []
        seen = {tuple(ref_path)}
        for mode, path in candidates:
            key = tuple(path)
            if key not in seen:
                seen.add(key)
                unique.append((mode, path))
        return unique

    def _global_primary_overlay(self, graph, nodes, ref_paths, candidate_map):
        selected = {source: list(path) for source, path in ref_paths.items()}
        ref_vectors = {source: self._path_vector(
            graph, path) for source, path in ref_paths.items()}
        burden = defaultdict(float)
        for vector in ref_vectors.values():
            burden = self._vector_add(burden, vector)
        baseline_total = self._vector_total(burden)
        baseline_max, baseline_p95 = self._stress_stats(burden, nodes)
        accepted_modes = Counter()
        rejected = Counter()

        records = []
        for source, candidates in candidate_map.items():
            ref = ref_paths.get(source, [])
            ref_vec = ref_vectors.get(source, defaultdict(float))
            ref_total = self._path_total(ref_vec)
            for mode, candidate in candidates:
                cand_vec = self._path_vector(graph, candidate)
                cand_total = self._path_total(cand_vec)
                energy_gain = (ref_total - cand_total) / max(ref_total, 1e-12)
                hop_gain = self._path_hops(ref) - self._path_hops(candidate)
                risk_gain = self._path_risk(
                    graph, ref) - self._path_risk(graph, candidate)
                distance_gain = self._path_distance(
                    graph, ref) - self._path_distance(graph, candidate)
                score = 4.0 * energy_gain + 0.25 * hop_gain + \
                    0.5 * risk_gain + 0.002 * distance_gain
                records.append((-score, source, mode, candidate, cand_vec))
        records.sort(key=lambda item: (item[0], item[1], item[2]))

        tolerance = max(
            float(self._cfg('data_route_energy_tolerance', 0.0149334243033)), 0.0)
        stress_tolerance = max(
            float(self._cfg('data_route_stress_tolerance', 0.0434899006505)), 0.0)
        distance_stretch = max(
            float(self._cfg('data_route_distance_stretch', 1.2468301242217)), 1.0)
        hop_slack = max(int(self._cfg('data_route_hop_slack', 0)), 0)
        min_improvement = max(
            float(self._cfg('data_route_min_improvement', 0.0010234782253)), 0.0)

        already_considered = set()
        for _, source, mode, candidate, cand_vec in records:
            if source in already_considered:
                continue
            ref = ref_paths[source]
            ref_vec = ref_vectors[source]
            ref_total = self._path_total(ref_vec)
            cand_total = self._path_total(cand_vec)
            if cand_total > ref_total * (1.0 + tolerance) + 1e-15:
                rejected['path_energy'] += 1
                continue
            if self._path_hops(candidate) > self._path_hops(ref) + hop_slack:
                rejected['hop'] += 1
                continue
            if self._path_distance(graph, candidate) > self._path_distance(graph, ref) * distance_stretch + 1e-12:
                rejected['distance'] += 1
                continue
            if self._path_risk(graph, candidate) > self._path_risk(graph, ref) + 1e-12:
                rejected['risk'] += 1
                continue
            source_ref = float(ref_vec.get(source, 0.0))
            source_cand = float(cand_vec.get(source, 0.0))
            if source_cand > source_ref * (1.0 + tolerance) + 1e-15:
                rejected['source_energy'] += 1
                continue

            projected = self._vector_add(burden, ref_vec, factor=-1.0)
            projected = self._vector_add(projected, cand_vec, factor=1.0)
            total = self._vector_total(projected)
            max_stress, p95 = self._stress_stats(projected, nodes)
            if total > baseline_total * (1.0 + tolerance) + 1e-15:
                rejected['network_energy'] += 1
                continue
            if max_stress > baseline_max * (1.0 + stress_tolerance) + 1e-15:
                rejected['max_stress'] += 1
                continue
            if p95 > baseline_p95 * (1.0 + stress_tolerance) + 1e-15:
                rejected['p95_stress'] += 1
                continue

            energy_gain = (ref_total - cand_total) / max(ref_total, 1e-12)
            max_gain = (baseline_max - max_stress) / max(baseline_max, 1e-12)
            meaningful = (
                energy_gain >= min_improvement
                or max_gain >= min_improvement
                or self._path_hops(candidate) < self._path_hops(ref)
                or self._path_risk(graph, candidate) < self._path_risk(graph, ref) - 1e-12
            )
            if not meaningful:
                rejected['no_improvement'] += 1
                continue

            selected[source] = list(candidate)
            burden = projected
            accepted_modes[mode] += 1
            already_considered.add(source)

        diagnostics = {
            'primary_replaced_count': float(sum(accepted_modes.values())),
            'cluster_mst_primary_count': float(accepted_modes['cluster_mst']),
            'backbone_biased_primary_count': float(accepted_modes['backbone_biased']),
            'energy_shortest_primary_count': float(accepted_modes['energy_shortest']),
            'primary_energy_guard_rejections': float(rejected['path_energy'] + rejected['source_energy'] + rejected['network_energy']),
            'primary_stress_guard_rejections': float(rejected['max_stress'] + rejected['p95_stress']),
            'primary_other_guard_rejections': float(sum(rejected.values()) - rejected['path_energy'] - rejected['source_energy'] - rejected['network_energy'] - rejected['max_stress'] - rejected['p95_stress']),
            'projected_primary_energy_ratio': float(self._vector_total(burden) / max(baseline_total, 1e-12)),
        }
        return selected, diagnostics

    @staticmethod
    def _internal_nodes(path):
        return set(list(path or [])[1:-1])

    @staticmethod
    def _edges(path):
        return set(zip(list(path or [])[:-1], list(path or [])[1:]))

    def _secondary_quality(self, graph, primary, secondary):
        p_nodes = self._internal_nodes(primary)
        s_nodes = self._internal_nodes(secondary)
        p_edges = self._edges(primary)
        s_edges = self._edges(secondary)
        node_overlap = len(p_nodes & s_nodes)
        edge_overlap = len(p_edges & s_edges)
        node_ratio = node_overlap / max(1, len(s_nodes))
        edge_ratio = edge_overlap / max(1, len(s_edges))
        cls = 0 if node_overlap == 0 else (1 if edge_overlap == 0 else 2)
        vector = self._path_vector(graph, secondary)
        return {
            'class': cls,
            'node_overlap': node_overlap,
            'edge_overlap': edge_overlap,
            'node_overlap_ratio': node_ratio,
            'edge_overlap_ratio': edge_ratio,
            'energy': self._path_total(vector),
            'risk': self._path_risk(graph, secondary),
            'hops': self._path_hops(secondary),
            'distance': self._path_distance(graph, secondary),
            'vector': vector,
        }

    def _least_overlap_safe_path(self, graph, source, primary):
        """Find a globally least-overlap alternate with security priority.

        The scalarized edge cost is lexicographic in practice: internal-node
        overlap, shared-edge overlap, accumulated security stress, then the
        original energy-aware route weight.  All components are non-negative,
        so Dijkstra remains valid.
        """
        primary = list(primary or [])
        if len(primary) < 2:
            return []
        primary_internal = frozenset(primary[1:-1])
        primary_edges = frozenset(zip(primary[:-1], primary[1:]))
        n_bound = max(2, graph.number_of_nodes())
        edge_weights = [
            max(0.0, float(data.get('weight', 1.0)))
            for _, _, data in graph.edges(data=True)
            if math.isfinite(float(data.get('weight', 1.0)))
        ]
        max_edge_weight = max(edge_weights, default=1.0)

        # Security stress is in [0, 1] per hop.  Make one unit of accumulated
        # security risk dominate every possible difference in base route cost.
        base_path_bound = n_bound * max(max_edge_weight, 1.0) + 1.0
        security_unit = base_path_bound
        safe_path_bound = n_bound * (security_unit + max_edge_weight) + 1.0
        edge_overlap_unit = safe_path_bound
        node_overlap_unit = (n_bound + 1.0) * edge_overlap_unit

        def route_weight(u, v, data):
            node_overlap = 1.0 if v in primary_internal else 0.0
            edge_overlap = 1.0 if (u, v) in primary_edges else 0.0
            security = float(np.clip(
                data.get('security_stress', 0.0), 0.0, 1.0))
            base = max(0.0, float(data.get('weight', 1.0)))
            return (
                node_overlap * node_overlap_unit
                + edge_overlap * edge_overlap_unit
                + security * security_unit
                + base
            )

        try:
            path = nx.shortest_path(
                graph, source, self.bs_id, weight=route_weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
        if path == primary or not self._valid_path(
                graph, path, source, self.bs_id):
            return []
        return list(path)

    def _secondary_candidates(self, graph, source, primary):
        """Generate a compact, diverse alternate-path pool.

        Unlike repeated k-shortest-simple-path enumeration, this uses a fixed
        number of Dijkstra runs per source and remains practical for long WRSN
        simulations.  The candidates cover node-disjoint, edge-disjoint,
        overlap-penalised, and alternate-first-hop solutions.
        """
        candidates = []
        blocked_nodes = frozenset(primary[1:-1])

        def node_disjoint_weight(u, v, data):
            if u in blocked_nodes or v in blocked_nodes:
                return None
            return data.get('weight', 1.0)

        try:
            candidates.append(nx.shortest_path(
                graph, source, self.bs_id, weight=node_disjoint_weight))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

        blocked_edges = frozenset(zip(primary[:-1], primary[1:]))

        def edge_disjoint_weight(u, v, data):
            if (u, v) in blocked_edges:
                return None
            return data.get('weight', 1.0)

        try:
            candidates.append(nx.shortest_path(
                graph, source, self.bs_id, weight=edge_disjoint_weight))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

        primary_internal = frozenset(primary[1:-1])
        primary_edges = frozenset(zip(primary[:-1], primary[1:]))
        node_penalty = max(float(self._cfg(
            'data_secondary_node_overlap_penalty', 4.0)), 0.0)
        edge_penalty = max(float(self._cfg(
            'data_secondary_edge_overlap_penalty', 6.0)), 0.0)

        def overlap_weight(u, v, data):
            factor = 1.0
            if v in primary_internal:
                factor += node_penalty
            if (u, v) in primary_edges:
                factor += edge_penalty
            return float(data.get('weight', 1.0)) * factor

        # Explore a small number of alternate first hops.  This often finds a
        # different physical corridor when strict node-disjoint routing does
        # not exist, without exponential path enumeration.
        first_hop = primary[1] if len(primary) > 1 else None
        neighbor_budget = max(1, int(self._cfg(
            'data_secondary_alternate_first_hops', 3)))
        neighbors = []
        for neighbor in graph.successors(source):
            if neighbor == first_hop:
                continue
            edge_weight = float(graph[source][neighbor].get('weight', 1.0))
            neighbors.append((edge_weight, neighbor))
        neighbors.sort()
        selected_neighbors = neighbors[:neighbor_budget]

        # The original full reverse traversal is equivalent for the requested
        # source/neighbor suffixes. Stop after those paths are settled.
        overlap_paths_to_bs = {}
        reverse_graph = graph.reverse(copy=False)

        def reverse_overlap_weight(u, v, data):
            return overlap_weight(v, u, data)

        try:
            targets = {source}
            targets.update(neighbor for _, neighbor in selected_neighbors)
            _, reverse_paths = _dijkstra_targets_exact(
                reverse_graph, self.bs_id, targets,
                weight=reverse_overlap_weight, paths=True)
            overlap_paths_to_bs = {
                node: list(reversed(path))
                for node, path in reverse_paths.items()
            }
            source_path = overlap_paths_to_bs.get(source)
            if source_path:
                candidates.append(source_path)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            overlap_paths_to_bs = {}

        for _, neighbor in selected_neighbors:
            suffix = overlap_paths_to_bs.get(neighbor)
            if suffix:
                path = [source] + list(suffix)
                candidates.append(path)

        unique, seen = [], {tuple(primary)}
        for path in candidates:
            key = tuple(path)
            if (key in seen
                    or not self._valid_path(
                        graph, path, source, self.bs_id)):
                continue
            seen.add(key)
            unique.append(list(path))
        return unique

    def _choose_secondary(self, graph, nodes, source, primary, ref_primary, ref_secondary):
        ref_secondary = list(ref_secondary or [])
        ref_valid = self._valid_path(graph, ref_secondary, source, self.bs_id)
        ref_quality = self._secondary_quality(
            graph, primary, ref_secondary) if ref_valid else None
        primary_vec = self._path_vector(graph, primary)

        ref_pair_total = None
        ref_pair_max = None
        ref_pair_p95 = None
        if ref_valid:
            ref_pair = self._vector_add(
                self._path_vector(graph, ref_primary),
                self._path_vector(graph, ref_secondary),
            )
            ref_pair_total = self._vector_total(ref_pair)
            ref_pair_max, ref_pair_p95 = self._stress_stats(ref_pair, nodes)

        candidates = self._secondary_candidates(graph, source, primary)
        # Preserve the exact reference/replacement candidate set whenever a
        # valid secondary already exists.  The global least-overlap search is
        # activated only to fill a genuine missing second route, preventing
        # avoidable changes to established experiment trajectories.
        require_two_routes = bool(self._cfg(
            'critical_require_two_routes', True))
        secondary_selection_mode = str(self._cfg(
            'critical_secondary_selection_mode', 'least_overlap_safest'
        )).strip().lower()
        if (not ref_valid and require_two_routes and not candidates and
                secondary_selection_mode in {
                    'least_overlap_safest', 'least_overlap_safe',
                    'minimum_overlap_safe'
                }):
            least_overlap_safe = self._least_overlap_safe_path(
                graph, source, primary)
            if (least_overlap_safe and
                    tuple(least_overlap_safe) not in {tuple(p) for p in candidates}):
                candidates.append(least_overlap_safe)
        scored = []
        candidate_qualities = []
        max_cost_stretch = max(
            float(self._cfg('data_secondary_max_energy_stretch', 1.0)), 1.0)
        for path in candidates:
            quality = self._secondary_quality(graph, primary, path)
            candidate_qualities.append((path, quality))
            if ref_quality is not None and quality['energy'] > ref_quality['energy'] * max_cost_stretch + 1e-15:
                continue
            # Combined primary+secondary energy may not exceed the original
            # critical-pair energy.  This protects FND and charging demand.
            if ref_valid:
                cand_pair = self._vector_add(primary_vec, quality['vector'])
                if self._vector_total(cand_pair) > ref_pair_total + 1e-15:
                    continue
                cand_max, cand_p95 = self._stress_stats(cand_pair, nodes)
                if cand_max > ref_pair_max + 1e-15 or cand_p95 > ref_pair_p95 + 1e-15:
                    continue
            score = (
                quality['class'], quality['node_overlap_ratio'],
                quality['edge_overlap_ratio'], quality['risk'],
                quality['energy'], quality['hops'], quality['distance'],
            )
            scored.append((score, path, quality))
        if not scored:
            return (ref_secondary if ref_valid else [],
                    'reference_kept' if ref_valid else 'none', [])
        scored.sort(key=lambda item: item[0])
        best_score, best_path, best_quality = scored[0]
        if ref_quality is None:
            if not require_two_routes:
                standby_path = min(
                    candidate_qualities,
                    key=lambda item: (
                        item[1]['class'],
                        item[1]['node_overlap_ratio'],
                        item[1]['energy'],
                    ),
                )[0] if candidate_qualities else []
                return [], 'standby_only', list(standby_path)
            # A distinct second route is now part of the critical route table.
            # To preserve non-regression, synthesized routes default to
            # failover activation and therefore add no radio cost while the
            # primary succeeds.
            return list(best_path), 'synthesized', []
        ref_score = (
            ref_quality['class'], ref_quality['node_overlap_ratio'],
            ref_quality['edge_overlap_ratio'], ref_quality['risk'],
            ref_quality['energy'], ref_quality['hops'], ref_quality['distance'],
        )
        if best_score < ref_score:
            return list(best_path), 'replaced', []
        return ref_secondary, 'reference_kept', []

    @staticmethod
    def _empty_table():
        return {
            'primary': None, 'secondary': None,
            'primary_path': [], 'secondary_path': [], 'critical_paths': [],
            'standby_secondary_path': [],
            'reference_primary_path': [], 'reference_secondary_path': [],
            'secondary_mode': 'none', 'secondary_origin': None,
            'secondary_node_overlap': 0, 'secondary_edge_overlap': 0,
            'cluster_head': None, 'data_route_mode': 'unreachable',
        }

    def compute_routing_tables(self, active_nodes_dict, node_penalties=None,
                               reference_tables=None):
        penalties = dict(node_penalties or {})
        reference_tables = dict(reference_tables or {})

        if not bool(self._cfg('use_data_plane_hybrid_routing', True)):
            mirrored = {}
            for nid in active_nodes_dict:
                if nid == self.bs_id:
                    continue
                ref = reference_tables.get(nid, {}) or {}
                table = self._empty_table()
                table['primary'] = ref.get('primary')
                table['secondary'] = ref.get('secondary')
                table['primary_path'] = list(ref.get('primary_path') or [])
                table['secondary_path'] = list(ref.get('secondary_path') or [])
                table['critical_paths'] = [
                    list(p) for p in (ref.get('critical_paths') or [])]
                table['reference_primary_path'] = list(table['primary_path'])
                table['reference_secondary_path'] = list(
                    table['secondary_path'])
                table['secondary_mode'] = str(
                    ref.get('secondary_mode',
                            'redundant' if table['secondary_path'] else 'none'))
                table['secondary_origin'] = ref.get('secondary_origin')
                table['secondary_node_overlap'] = int(
                    ref.get('secondary_node_overlap', 0) or 0)
                table['secondary_edge_overlap'] = int(
                    ref.get('secondary_edge_overlap', 0) or 0)
                table['data_route_mode'] = 'reference_only'
                mirrored[nid] = table
            self.last_cluster_heads = []
            self.last_clusters = {}
            self.last_diagnostics = {
                'reachable_count': float(sum(1 for t in mirrored.values() if t['primary_path'])),
                'primary_replaced_count': 0.0,
                'secondary_replaced_count': 0.0,
                'projected_primary_energy_ratio': 1.0,
            }
            return mirrored
        self._path_vector_cache.clear()
        self._path_distance_cache.clear()
        self._path_risk_cache.clear()
        self._stress_energy_cache.clear()
        graph, k_opt = self._build_graph(active_nodes_dict, penalties)
        tables = {
            nid: self._empty_table()
            for nid in active_nodes_dict if nid != self.bs_id
        }
        ref_paths = {}
        for source in graph:
            if source == self.bs_id:
                continue
            ref = self._reference_path(graph, source, reference_tables)
            if ref:
                ref_paths[source] = ref
        if not ref_paths:
            self.last_cluster_heads = []
            self.last_clusters = {}
            self.last_diagnostics = {'reachable_count': 0.0}
            return tables

        heads = self._select_heads(graph, active_nodes_dict, penalties, k_opt)
        clusters, member_to_head = self._assign_clusters(
            graph, active_nodes_dict, heads)
        if bool(self._cfg('use_mst_backbone', True)):
            backbone = self._backbone(graph, heads)
        else:
            backbone = nx.Graph()
            backbone.add_nodes_from([self.bs_id] + list(heads))
            for head in heads:
                try:
                    path = nx.shortest_path(
                        graph, head, self.bs_id, weight='weight')
                    for u, v in zip(path[:-1], path[1:]):
                        backbone.add_edge(
                            u, v, weight=float(graph[u][v]['weight']))
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
        self.last_cluster_heads = list(heads)
        self.last_clusters = {head: list(members)
                              for head, members in clusters.items()}

        head_paths = {}
        for head in heads:
            if head in backbone and self.bs_id in backbone:
                try:
                    path = nx.shortest_path(backbone, head, self.bs_id)
                    if self._valid_path(graph, path, head, self.bs_id):
                        head_paths[head] = list(path)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass

        candidate_map = {}
        for source, ref in ref_paths.items():
            head = source if source in heads else member_to_head.get(source)
            candidate_map[source] = self._candidate_paths(
                graph, source, ref, head, head_paths, backbone
            )
        selected, diagnostics = self._global_primary_overlay(
            graph, active_nodes_dict, ref_paths, candidate_map
        )

        secondary_replaced = 0
        secondary_reference_kept = 0
        secondary_synthesized = 0
        secondary_failover = 0
        secondary_node_disjoint = 0
        secondary_edge_disjoint = 0
        standby_count = 0
        overlap_values = []
        unavoidable_no_backup = 0

        for source, ref_primary in ref_paths.items():
            primary = selected.get(source, ref_primary)
            ref_table = reference_tables.get(source, {}) or {}
            ref_secondary = list(ref_table.get('secondary_path') or [])
            secondary, status, standby = self._choose_secondary(
                graph, active_nodes_dict, source, primary,
                ref_primary, ref_secondary,
            )
            if status == 'standby_only' and standby:
                standby_count += 1
            if status == 'replaced':
                secondary_replaced += 1
            elif status == 'reference_kept':
                secondary_reference_kept += 1
            elif status == 'synthesized':
                secondary_synthesized += 1

            table = tables[source]
            table['primary_path'] = list(primary)
            table['primary'] = primary[1]
            table['reference_primary_path'] = list(ref_primary)
            table['reference_secondary_path'] = list(ref_secondary)
            table['cluster_head'] = source if source in heads else member_to_head.get(
                source)
            table['data_route_mode'] = 'enhanced' if primary != ref_primary else 'reference_guarded'
            table['critical_paths'] = [list(primary)]
            if secondary:
                table['secondary_path'] = list(secondary)
                table['secondary'] = secondary[1]
                table['critical_paths'].append(list(secondary))
                quality = self._secondary_quality(graph, primary, secondary)
                overlap_values.append(quality['node_overlap_ratio'])
                table['secondary_node_overlap'] = int(
                    quality['node_overlap'])
                table['secondary_edge_overlap'] = int(
                    quality['edge_overlap'])
                if status == 'synthesized':
                    mode = str(self._cfg(
                        'critical_synthesized_secondary_mode', 'failover'
                    )).strip().lower()
                    origin = 'least_overlap_safe'
                else:
                    # Preserve the activation policy of the reference route.
                    # A reference path synthesized for sparse topologies stays
                    # failover-only even if the DATA overlay improves it.
                    mode = str(ref_table.get(
                        'secondary_mode', 'redundant')).strip().lower()
                    origin = (
                        'enhanced_replacement' if status == 'replaced'
                        else ref_table.get(
                            'secondary_origin', 'reference_secondary')
                    )
                if mode not in {'failover', 'redundant'}:
                    mode = 'failover' if status == 'synthesized' else 'redundant'
                table['secondary_mode'] = mode
                table['secondary_origin'] = origin
                if mode == 'failover':
                    secondary_failover += 1
                if quality['class'] == 0:
                    secondary_node_disjoint += 1
                elif quality['class'] == 1:
                    secondary_edge_disjoint += 1
            else:
                unavoidable_no_backup += 1
            table['standby_secondary_path'] = list(standby)

        diagnostics.update({
            'reachable_count': float(len(ref_paths)),
            'cluster_count': float(len(heads)),
            'secondary_replaced_count': float(secondary_replaced),
            'secondary_reference_kept_count': float(secondary_reference_kept),
            'secondary_synthesized_count': float(secondary_synthesized),
            'secondary_failover_count': float(secondary_failover),
            'secondary_node_disjoint_count': float(secondary_node_disjoint),
            'secondary_edge_disjoint_count': float(secondary_edge_disjoint),
            'secondary_standby_count': float(standby_count),
            'critical_two_route_count': float(
                len(ref_paths) - unavoidable_no_backup),
            'critical_unavoidable_no_backup_count': float(
                unavoidable_no_backup),
            'secondary_average_node_overlap': float(np.mean(overlap_values)) if overlap_values else 0.0,
        })
        self.last_diagnostics = diagnostics
        return tables
