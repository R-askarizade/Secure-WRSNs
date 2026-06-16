import math
import time
import random
from config import compute_packet_hash, CONTROL_PACKET_BYTES_SMALL, DATA_PACKET_BYTES


class SensorNode:
    def __init__(self, node_id, x, y, initial_energy, metrics_collector):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.crypto_keys = None
        self.temp_parent = None
        self.hop_count_to_bs = float('inf')
        self.seen_adv_seq = -1

        self.E_max = initial_energy
        self.E_current = initial_energy
        self.E_critical = initial_energy * 0.15
        self.status = "ALIVE"

        self.primary_next_hop = None
        self.secondary_next_hop = None
        self.adaptive_threshold = float('inf')

        self.inactivity_timer = 0
        self.T_max_inactivity = 3600
        self.last_activity_time = 0.0

        self.metrics = metrics_collector
        self.network_nodes = {}
        self.is_malicious = False
        self.sos_sent = False
        self.control_seq_counter = 0

        self.E_elec = 50e-9
        self.eps_fs = 10e-12
        self.eps_mp = 0.0013e-12
        self.E_DA = 5e-9
        self.d0 = math.sqrt(self.eps_fs / self.eps_mp)

        self.last_topology_alert_time = -float('inf')
        self.topology_alert_cooldown = 300.0
        self.last_bad_next_hop = None

    def check_route_health(self, nodes, tx_range, current_time):
        if self.primary_next_hop is None:
            return None

        bad_link = False
        if self.primary_next_hop not in nodes:
            bad_link = True
        else:
            next_node = nodes[self.primary_next_hop]
            distance = math.hypot(self.x - next_node.x, self.y - next_node.y)
            bad_link = next_node.status == "DEAD" or distance > tx_range

        if not bad_link:
            self.last_bad_next_hop = None
            return None

        if (self.last_bad_next_hop == self.primary_next_hop and
                current_time - self.last_topology_alert_time < self.topology_alert_cooldown):
            return None

        self.last_bad_next_hop = self.primary_next_hop
        self.last_topology_alert_time = current_time
        return self._send_topology_change(current_time)

    def _next_control_seq(self):
        self.control_seq_counter += 1
        return self.control_seq_counter

    def _finalize_source_control_packet(self, packet):
        packet.setdefault('seq', self._next_control_seq())
        if self.crypto_keys is not None:
            packet['hash'] = compute_packet_hash(packet, self.crypto_keys)
        return packet

    def _send_topology_change(self, current_time):
        packet = {
            'source_id': self.node_id,
            'type': 'TOPOLOGY_CHANGE',
            'priority': 'HIGH',
            'timestamp': current_time,
            'size_bytes': CONTROL_PACKET_BYTES_SMALL,
            'energy_trace': [(self.node_id, self.E_current)]
        }
        self._finalize_source_control_packet(packet)
        self._transmit_packet(packet, is_critical=True)
        self.metrics.update_packet_transmission(
            is_critical=True, is_control=True, size_bytes=CONTROL_PACKET_BYTES_SMALL)
        self._reset_inactivity_timer(current_time)
        return packet

    def sense_environment(self, raw_data_value, current_time):
        if self.status == "DEAD":
            return None

        start_time = time.time()
        critical_flag = 1 if raw_data_value > self.adaptive_threshold else 0
        is_critical = (critical_flag == 1)

        reported_energy = self.E_current
        if self.is_malicious:
            reported_energy = self.E_max * 0.05

        packet = {
            "source_id": self.node_id,
            "type": "DATA",
            "priority": "HIGH" if is_critical else "NORMAL",
            "payload": raw_data_value,
            "timestamp": current_time,
            "size_bytes": DATA_PACKET_BYTES,
            "energy_trace": [(self.node_id, reported_energy)]
        }

        self._transmit_packet(packet, is_critical)
        exec_time_ms = (time.time() - start_time) * 1000
        self.metrics.record_execution_time('EDGE', exec_time_ms)
        self._reset_inactivity_timer(current_time)

    def _distance_to_next_hop(self, next_hop_id, fallback=40.0):
        if self.network_nodes and next_hop_id in self.network_nodes:
            other = self.network_nodes[next_hop_id]
            return math.hypot(self.x - other.x, self.y - other.y)
        return fallback

    def _transmit_packet(self, packet, is_critical):
        if self.status == "DEAD":
            return
        if not is_critical:
            if self.primary_next_hop is not None:
                self._consume_energy(
                    tx_bytes=packet["size_bytes"], distance=self._distance_to_next_hop(self.primary_next_hop))
        else:
            if self.primary_next_hop is not None:
                self._consume_energy(
                    tx_bytes=packet["size_bytes"], distance=self._distance_to_next_hop(self.primary_next_hop))
            if self.secondary_next_hop is not None and self.secondary_next_hop != self.primary_next_hop:
                self._consume_energy(tx_bytes=packet["size_bytes"], distance=self._distance_to_next_hop(
                    self.secondary_next_hop))

    def forward_packet(self, packet, current_time, distance_to_next):
        if self.status == "DEAD":
            return False

        if "path_header" not in packet:
            packet["path_header"] = []

        packet["path_header"].append(self.node_id)
        self._consume_energy(rx_bytes=packet["size_bytes"])

        reported_energy = self.E_current
        if self.is_malicious:
            reported_energy = self.E_max * 0.05

        if "energy_trace" not in packet:
            packet["energy_trace"] = []
        packet["energy_trace"].append((self.node_id, reported_energy))

        if self.is_malicious:
            if random.random() < 0.50:
                return False

        self._consume_energy(
            tx_bytes=packet["size_bytes"], distance=distance_to_next)
        self._reset_inactivity_timer(current_time)
        return True

    def check_inactivity(self, current_time):
        if self.status == "DEAD":
            return None

        silence_duration = current_time - self.last_activity_time
        battery_critical = self.E_current <= self.E_critical
        long_silence = silence_duration >= self.T_max_inactivity

        if self.is_malicious:
            if random.random() < 0.05 and not self.sos_sent:
                temp_battery = self.E_current
                self.E_current = 0.01
                packet = self._send_sos_charge_request(current_time)
                self.E_current = temp_battery
                self.sos_sent = True
                return packet

        if battery_critical and long_silence and not self.sos_sent:
            self.sos_sent = True
            return self._send_sos_charge_request(current_time)

        if long_silence and not battery_critical:
            return self._send_keep_alive(current_time)
        return None

    def _send_sos_charge_request(self, current_time):
        packet = {
            'source_id': self.node_id,
            'type': 'SOS_CHARGE',
            'priority': 'HIGH',
            'timestamp': current_time,
            'size_bytes': CONTROL_PACKET_BYTES_SMALL,
            'energy_trace': [(self.node_id, self.E_current)]
        }
        self._finalize_source_control_packet(packet)
        self._transmit_packet(packet, is_critical=True)
        self.metrics.update_packet_transmission(
            is_critical=True, is_control=True, size_bytes=CONTROL_PACKET_BYTES_SMALL)
        self._reset_inactivity_timer(current_time)
        return packet

    def _send_keep_alive(self, current_time):
        packet = {
            'source_id': self.node_id,
            'type': 'KEEP_ALIVE',
            'priority': 'NORMAL',
            'timestamp': current_time,
            'size_bytes': CONTROL_PACKET_BYTES_SMALL,
            'energy_trace': [(self.node_id, self.E_current)]
        }
        self._finalize_source_control_packet(packet)
        self._transmit_packet(packet, is_critical=False)
        self.metrics.update_packet_transmission(
            is_critical=False, is_control=True, size_bytes=CONTROL_PACKET_BYTES_SMALL)
        self._reset_inactivity_timer(current_time)
        return packet

    def _consume_energy(self, tx_bytes=0, rx_bytes=0, distance=0.0, aggregation=False):
        bits = (tx_bytes + rx_bytes) * 8
        energy = 0.0

        if rx_bytes > 0:
            energy += rx_bytes * 8 * self.E_elec

        if tx_bytes > 0:
            if distance <= self.d0:
                energy += tx_bytes * 8 * \
                    (self.E_elec + self.eps_fs * (distance ** 2))
            else:
                energy += tx_bytes * 8 * \
                    (self.E_elec + self.eps_mp * (distance ** 4))

        if aggregation:
            energy += bits * self.E_DA

        self.E_current -= energy
        if self.E_current <= 0 and self.status == "ALIVE":
            self.E_current = 0
            self.status = "DEAD"
            self.metrics.record_node_death(
                self.node_id, self.metrics.current_sim_time)

    def receive_wireless_charge(self, injected_energy):
        if self.status == "DEAD":
            self.status = "ALIVE"
        self.E_current += injected_energy
        if self.E_current > self.E_max:
            self.E_current = self.E_max
        self.sos_sent = False

    def _reset_inactivity_timer(self, current_time):
        self.last_activity_time = current_time

    def update_policies_from_bs(self, primary_hop, secondary_hop, new_threshold):
        self.primary_next_hop = primary_hop
        self.secondary_next_hop = secondary_hop
        self.adaptive_threshold = new_threshold
