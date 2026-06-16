import math
import networkx as nx
from config import DATA_PACKET_BITS

class BaselineRouting:
    def __init__(self, bs_id, transmission_range=50.0, max_critical_paths=2):
        self.bs_id = bs_id
        self.transmission_range = transmission_range
        self.max_critical_paths = max_critical_paths

    def build_network_graph(self, active_nodes_dict):
        G = nx.Graph()
        node_ids = list(active_nodes_dict.keys())
        for n_id in node_ids:
            G.add_node(n_id, pos=(
                active_nodes_dict[n_id].x, active_nodes_dict[n_id].y))

        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                n1 = active_nodes_dict[node_ids[i]]
                n2 = active_nodes_dict[node_ids[j]]
                if n1.status == "ALIVE" and n2.status == "ALIVE":
                    dist = math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)
                    if dist <= self.transmission_range:
                        G.add_edge(n1.node_id, n2.node_id, weight=dist)
        return G

    def compute_routing_tables(self, active_nodes_dict, node_penalties=None):
        def _link_energy_cost(distance, packet_bits=DATA_PACKET_BITS):
            E_elec = 50e-9
            eps_fs = 10e-12
            eps_mp = 0.0013e-12
            d0 = math.sqrt(eps_fs / eps_mp)
            if distance <= d0:
                return packet_bits * (E_elec + eps_fs * distance ** 2)
            return packet_bits * (E_elec + eps_mp * distance ** 4)

        def _path_weight(graph, path):
            total = 0.0
            for a, b in zip(path[:-1], path[1:]):
                try:
                    w = graph[a][b].get('weight', float('inf'))
                except Exception:
                    return float('inf')
                if not math.isfinite(w):
                    return float('inf')
                total += w
            return total

        if node_penalties is None:
            node_penalties = {nid: 1.0 for nid in active_nodes_dict.keys()}

        G = self.build_network_graph(active_nodes_dict)
        G_dir = nx.DiGraph()
        G_dir.add_nodes_from(G.nodes())
        quarantined_zones = getattr(self, 'quarantined_zones', [])

        def _in_quarantine(node_id):
            node = active_nodes_dict.get(node_id)
            if node and quarantined_zones:
                for zone in quarantined_zones:
                    if math.hypot(node.x - zone['x'], node.y - zone['y']) <= zone['radius']:
                        return True
            return False

        for u, v, data in G.edges(data=True):
            base_weight = _link_energy_cost(data['weight'])
            penalty_v = float('inf') if _in_quarantine(
                v) else node_penalties.get(v, 1.0)
            penalty_u = float('inf') if _in_quarantine(
                u) else node_penalties.get(u, 1.0)
            G_dir.add_edge(u, v, weight=base_weight * penalty_v)
            G_dir.add_edge(v, u, weight=base_weight * penalty_u)

        routing_tables = {}
        for n_id in G_dir.nodes():
            if n_id == self.bs_id:
                continue
            primary_path = []
            secondary_path = []
            primary_next_hop = None
            secondary_next_hop = None
            critical_paths = []

            try:
                p = nx.shortest_path(G_dir, source=n_id,
                                     target=self.bs_id, weight='weight')
                if len(p) > 1 and math.isfinite(_path_weight(G_dir, p)):
                    primary_path = p
                    primary_next_hop = p[1]
                    critical_paths.append(primary_path)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                primary_path = []

            if primary_path:
                try:
                    if len(primary_path) > 2:
                        forbidden_nodes = set(primary_path[1:-1])
                        G_pruned = nx.subgraph_view(
                            G_dir,
                            filter_node=lambda node, forbidden=forbidden_nodes: node not in forbidden
                        )
                        sp = nx.shortest_path(
                            G_pruned, source=n_id, target=self.bs_id, weight='weight')
                        sp_graph = G_pruned
                    else:
                        G_backup = G_dir.copy()
                        if G_backup.has_edge(n_id, self.bs_id):
                            G_backup.remove_edge(n_id, self.bs_id)
                        if G_backup.has_edge(self.bs_id, n_id):
                            G_backup.remove_edge(self.bs_id, n_id)
                        sp = nx.shortest_path(
                            G_backup, source=n_id, target=self.bs_id, weight='weight')
                        sp_graph = G_backup

                    if (len(sp) > 1 and sp != primary_path and
                            math.isfinite(_path_weight(sp_graph, sp)) and
                            set(primary_path[1:-1]).isdisjoint(set(sp[1:-1]))):
                        secondary_path = sp
                        secondary_next_hop = sp[1]
                        critical_paths.append(secondary_path)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    secondary_path = []

            routing_tables[n_id] = {
                'primary': primary_next_hop,
                'secondary': secondary_next_hop,
                'primary_path': primary_path,
                'secondary_path': secondary_path if secondary_next_hop is not None else [],
                'critical_paths': critical_paths[:2]
            }
        return routing_tables
