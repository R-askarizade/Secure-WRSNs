import networkx as nx
import math


class BaselineRouting:
    """
    Centralized Baseline Routing Algorithm (To be replaced by GWO later).
    Calculates Primary (Energy-efficient) and Secondary (Node-disjoint) paths 
    for every node to support the Edge Processing Quality of Service (QoS).
    """

    def __init__(self, bs_id, transmission_range=50.0):
        self.bs_id = bs_id
        self.transmission_range = transmission_range

    def build_network_graph(self, active_nodes_dict):
        """
        Builds a NetworkX graph based on physical distance and alive status.
        Edges represent valid wireless communication links.
        """
        G = nx.Graph()
        node_ids = list(active_nodes_dict.keys())

        # Add nodes
        for n_id in node_ids:
            G.add_node(n_id, pos=(
                active_nodes_dict[n_id].x, active_nodes_dict[n_id].y))

        # Add edges based on transmission range
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                n1 = active_nodes_dict[node_ids[i]]
                n2 = active_nodes_dict[node_ids[j]]

                # Only connect if both are ALIVE (and not ISOLATED by Base Station)
                if n1.status == "ALIVE" and n2.status == "ALIVE":
                    dist = math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)
                    if dist <= self.transmission_range:
                        G.add_edge(n1.node_id, n2.node_id, weight=dist)
        return G

    def compute_routing_tables(self, active_nodes_dict, exclude_node_id=None):
        """
        Computes routing tables for all nodes.
        If exclude_node_id is provided, this acts as the "Proactive Shadow Routing" 
        calculating failover paths without the suspicious node.
        """
        G = self.build_network_graph(active_nodes_dict)

        if exclude_node_id is not None and G.has_node(exclude_node_id):
            G.remove_node(exclude_node_id)  # Prune suspicious node from graph

        routing_tables = {}

        for n_id in G.nodes():
            if n_id == self.bs_id:
                continue

            primary_next_hop = None
            secondary_next_hop = None

            # 1. Calculate Primary Path (Shortest path using Dijkstra)
            try:
                primary_path = nx.shortest_path(
                    G, source=n_id, target=self.bs_id, weight='weight')
                if len(primary_path) > 1:
                    primary_next_hop = primary_path[1]
            except nx.NetworkXNoPath:
                primary_path = []

            # 2. Calculate Secondary Node-Disjoint Path (For Critical Data)
            if len(primary_path) > 2:
                # Create a pruned graph by removing all intermediate nodes of the primary path
                G_pruned = G.copy()
                # Keep source and destination(BS)
                nodes_to_remove = primary_path[1:-1]
                G_pruned.remove_nodes_from(nodes_to_remove)

                try:
                    secondary_path = nx.shortest_path(
                        G_pruned, source=n_id, target=self.bs_id, weight='weight')
                    if len(secondary_path) > 1:
                        secondary_next_hop = secondary_path[1]
                except nx.NetworkXNoPath:
                    secondary_next_hop = None  # No disjoint path available

            routing_tables[n_id] = {
                'primary': primary_next_hop,
                'secondary': secondary_next_hop
            }

        return routing_tables
