import math
import time
import random
from config import compute_packet_hash, make_node_charge_receipt_mac, CONTROL_PACKET_BYTES_SMALL, DATA_PACKET_BYTES


class SensorNode:
    def __init__(self, node_id, x, y, initial_energy, metrics_collector, data_packet_bytes=DATA_PACKET_BYTES, rng_seed=None,
                 data_packet_bits=None, control_packet_bits=None,
                 radio_e_elec_j_per_bit=50e-9,
                 radio_eps_fs_j_per_bit_m2=10e-12,
                 radio_eps_mp_j_per_bit_m4=0.0013e-12,
                 radio_e_da_j_per_bit=5e-9):
        self.node_id = node_id
        self.rng = random.Random(
            int(node_id if rng_seed is None else rng_seed))
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
        # DATA-plane routes are separate from the original control/charging
        # routes.  This prevents clustered routing from changing SOS/control
        # energy or any mobile-charger decision input.
        self.data_primary_next_hop = None
        self.data_secondary_next_hop = None
        self.adaptive_threshold = float('inf')

        self.inactivity_timer = 0
        self.T_max_inactivity = 3600
        self.last_activity_time = 0.0

        self.metrics = metrics_collector
        self.data_packet_bytes = max(1, int(data_packet_bytes))
        self.data_packet_bits = max(1.0, float(
            self.data_packet_bytes * 8 if data_packet_bits is None else data_packet_bits))
        self.control_packet_bits = max(1.0, float(
            CONTROL_PACKET_BYTES_SMALL * 8 if control_packet_bits is None else control_packet_bits))
        self.control_packet_bytes = max(
            1, int(math.ceil(self.control_packet_bits / 8.0)))
        self.network_nodes = {}
        self.is_malicious = False
        self.sos_sent = False
        self.control_seq_counter = 0
        self.charge_receipt_counter = 0

        self.E_elec = max(0.0, float(radio_e_elec_j_per_bit))
        self.eps_fs = max(0.0, float(radio_eps_fs_j_per_bit_m2))
        self.eps_mp = max(1e-30, float(radio_eps_mp_j_per_bit_m4))
        self.E_DA = max(0.0, float(radio_e_da_j_per_bit))
        self.d0 = math.sqrt(self.eps_fs / self.eps_mp)

        self.last_topology_alert_time = -float('inf')
        self.topology_alert_cooldown = 300.0
        self.last_bad_next_hop = None

        # ---------------------------------------------------------
        # Hybrid Fixed Charger Properties
        # ---------------------------------------------------------
        self.has_fixed_charger = False
        self.last_trickle_time = None
        self.cached_tx_range = 40.0

    def check_route_health(self, nodes, tx_range, current_time):
        # Cache the tx_range for fixed charger self-discovery
        self.cached_tx_range = float(tx_range)

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
            'size_bytes': self.control_packet_bytes,
            'size_bits': self.control_packet_bits,
            'energy_trace': [(self.node_id, self.E_current)]
        }
        self._finalize_source_control_packet(packet)
        self._transmit_packet(packet, is_critical=True)
        self.metrics.update_packet_transmission(
            is_critical=True, is_control=True, size_bytes=self.control_packet_bytes,
            size_bits=self.control_packet_bits)
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
            "size_bytes": self.data_packet_bytes,
            "size_bits": self.data_packet_bits,
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

        is_data = str(packet.get("type", "DATA")).upper() == "DATA"
        primary_hop = (
            self.data_primary_next_hop
            if is_data and self.data_primary_next_hop is not None
            else self.primary_next_hop
        )
        secondary_hop = (
            self.data_secondary_next_hop
            if is_data and self.data_secondary_next_hop is not None
            else self.secondary_next_hop
        )

        if primary_hop is not None:
            self._consume_energy(
                tx_bytes=packet["size_bytes"],
                tx_bits=packet.get("size_bits"),
                distance=self._distance_to_next_hop(primary_hop))
        if (is_critical and secondary_hop is not None
                and secondary_hop != primary_hop):
            self._consume_energy(
                tx_bytes=packet["size_bytes"],
                tx_bits=packet.get("size_bits"),
                distance=self._distance_to_next_hop(secondary_hop))

    def forward_packet(self, packet, current_time, distance_to_next):
        if self.status == "DEAD":
            return False

        if "path_header" not in packet:
            packet["path_header"] = []

        packet["path_header"].append(self.node_id)
        self._consume_energy(rx_bytes=packet["size_bytes"],
                             rx_bits=packet.get("size_bits"))

        reported_energy = self.E_current
        if self.is_malicious:
            reported_energy = self.E_max * 0.05

        if "energy_trace" not in packet:
            packet["energy_trace"] = []
        packet["energy_trace"].append((self.node_id, reported_energy))

        if self.is_malicious:
            if self.rng.random() < 0.50:
                return False

        self._consume_energy(
            tx_bytes=packet["size_bytes"], tx_bits=packet.get("size_bits"),
            distance=distance_to_next)
        self._reset_inactivity_timer(current_time)
        return True

    def check_inactivity(self, current_time):
        # Fixed-charger energy is applied centrally by BaseStation so that the
        # selected bottlenecks, injected energy, and infrastructure contribution
        # are measurable.  Nodes do not self-assign chargers or create energy.
        if self.status == "DEAD":
            return None

        silence_duration = current_time - self.last_activity_time
        battery_critical = self.E_current <= self.E_critical
        long_silence = silence_duration >= self.T_max_inactivity

        if self.is_malicious:
            if self.rng.random() < 0.05 and not self.sos_sent:
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
            'size_bytes': self.control_packet_bytes,
            'size_bits': self.control_packet_bits,
            'energy_trace': [(self.node_id, self.E_current)]
        }
        self._finalize_source_control_packet(packet)
        self._transmit_packet(packet, is_critical=True)
        self.metrics.update_packet_transmission(
            is_critical=True, is_control=True, size_bytes=self.control_packet_bytes,
            size_bits=self.control_packet_bits)
        self._reset_inactivity_timer(current_time)
        return packet

    def _send_keep_alive(self, current_time):
        packet = {
            'source_id': self.node_id,
            'type': 'KEEP_ALIVE',
            'priority': 'NORMAL',
            'timestamp': current_time,
            'size_bytes': self.control_packet_bytes,
            'size_bits': self.control_packet_bits,
            'energy_trace': [(self.node_id, self.E_current)]
        }
        self._finalize_source_control_packet(packet)
        self._transmit_packet(packet, is_critical=False)
        self.metrics.update_packet_transmission(
            is_critical=False, is_control=True, size_bytes=self.control_packet_bytes,
            size_bits=self.control_packet_bits)
        self._reset_inactivity_timer(current_time)
        return packet

    def _consume_energy(self, tx_bytes=0, rx_bytes=0, distance=0.0, aggregation=False,
                        tx_bits=None, rx_bits=None):
        tx_bit_count = max(0.0, float(tx_bytes) *
                           8.0 if tx_bits is None else float(tx_bits))
        rx_bit_count = max(0.0, float(rx_bytes) *
                           8.0 if rx_bits is None else float(rx_bits))
        bits = tx_bit_count + rx_bit_count
        energy = 0.0

        if rx_bit_count > 0.0:
            energy += rx_bit_count * self.E_elec

        if tx_bit_count > 0.0:
            if distance <= self.d0:
                energy += tx_bit_count * \
                    (self.E_elec + self.eps_fs * (distance ** 2))
            else:
                energy += tx_bit_count * \
                    (self.E_elec + self.eps_mp * (distance ** 4))

        if aggregation:
            energy += bits * self.E_DA

        if self.status == "DEAD":
            return

        previous_energy = max(0.0, float(self.E_current))
        self.E_current = max(0.0, previous_energy - max(0.0, float(energy)))
        if self.E_current <= 0.0 and self.status == "ALIVE":
            self.status = "DEAD"
            self.metrics.record_node_death(
                self.node_id, self.metrics.current_sim_time)

    def receive_wireless_charge(self, injected_energy):
        injected_energy = max(0.0, float(injected_energy))
        if injected_energy <= 0.0:
            return 0.0
        before = max(0.0, float(self.E_current))
        self.E_current = min(float(self.E_max), before + injected_energy)
        actual = max(0.0, float(self.E_current) - before)
        if actual > 0.0 and self.status == "DEAD":
            self.status = "ALIVE"
            if hasattr(self.metrics, 'record_node_revival'):
                self.metrics.record_node_revival(
                    self.node_id, self.metrics.current_sim_time)
        self.sos_sent = False
        return actual

    def create_charge_receipt(self, energy_before, energy_after, received_energy, mc_id, timestamp):
        """Create an independently authenticated sensor-side charging receipt.

        The receipt is carried by the existing charging handshake and therefore
        does not create a separate network-layer packet or alter legacy energy/
        overhead results. A malicious sensor retains its configured false-energy
        behavior, allowing the BS to distinguish MC and sensor claims.
        """
        self.charge_receipt_counter += 1
        counter = int(self.charge_receipt_counter)
        nonce = f"node-{int(self.node_id)}-{counter}-{float(timestamp):.6f}"
        before = max(0.0, float(energy_before))
        physical_after = max(0.0, float(energy_after))
        reported_after = physical_after
        if self.is_malicious:
            reported_after = max(0.0, float(self.E_max) * 0.05)
        claimed_received = max(0.0, float(received_energy))
        if self.is_malicious:
            claimed_received = max(0.0, reported_after - before)
        tag = make_node_charge_receipt_mac(
            self.crypto_keys, self.node_id, before, reported_after,
            claimed_received, mc_id=mc_id, counter=counter,
            nonce=nonce, timestamp=timestamp)
        return {
            'node_id': int(self.node_id),
            'energy_before': before,
            'energy_after': reported_after,
            'claimed_received': claimed_received,
            'mc_id': mc_id,
            'counter': counter,
            'nonce': nonce,
            'timestamp': float(timestamp),
            'mac_tag': tag,
        }

    def receive_fixed_charge(self, injected_energy):
        return self.receive_wireless_charge(injected_energy)

    def _reset_inactivity_timer(self, current_time):
        self.last_activity_time = current_time

    def update_policies_from_bs(self, primary_hop, secondary_hop, new_threshold):
        self.primary_next_hop = primary_hop
        self.secondary_next_hop = secondary_hop
        self.adaptive_threshold = new_threshold

    def update_data_routes(self, primary_hop, secondary_hop):
        """Update DATA-only next hops without touching control/charging state."""
        self.data_primary_next_hop = primary_hop
        self.data_secondary_next_hop = secondary_hop
