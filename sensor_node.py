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
                 radio_e_da_j_per_bit=5e-9,
                 attack_config=None):
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
        self.enable_dead_node_revival = bool(
            (attack_config or {}).get('enable_dead_node_revival', True))
        self.dead_node_revival_min_energy_ratio = max(0.0, min(1.0, float(
            (attack_config or {}).get('dead_node_revival_min_energy_ratio', 0.0))))

        self.primary_next_hop = None
        self.secondary_next_hop = None
        self.secondary_route_mode = 'redundant'
        # DATA-plane routes are separate from the original control/charging
        # routes.  This prevents clustered routing from changing SOS/control
        # energy or any mobile-charger decision input.
        self.data_primary_next_hop = None
        self.data_secondary_next_hop = None
        self.data_secondary_route_mode = 'redundant'
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
        self.behavior_state = "HONEST"
        # Audit timestamps for the local remediation state machine.  They are
        # metadata only and do not alter energy, routing, traffic, or trust.
        self.remediation_time = None
        self.rehabilitation_time = None
        self.sos_sent = False
        self.control_seq_counter = 0
        self.charge_receipt_counter = 0

        # Attack behavior is explicit and configurable.  The default
        # ``legacy_compatible`` profile reproduces the effective workload of
        # the previous thesis baseline: DATA relays act as blackholes, control
        # relays use 50% selective forwarding, source DATA energy is truthful,
        # and malicious charge receipts retain the old false-energy claim.
        # Corrected end-to-end attack scenarios can enable each behavior
        # independently without silently changing the baseline experiment.
        self.attack_config = dict(attack_config or {})
        self.attack_model_mode = str(
            self.attack_config.get('attack_model_mode', 'legacy_compatible')
        ).strip().lower()
        self.attack_false_energy_ratio = max(0.0, min(1.0, float(
            self.attack_config.get('attack_false_energy_ratio', 0.05))))
        self.attack_false_energy_data_enabled = bool(
            self.attack_config.get('attack_false_energy_data_enabled', False))
        self.attack_false_energy_control_enabled = bool(
            self.attack_config.get('attack_false_energy_control_enabled', True))
        self.attack_false_energy_receipt_enabled = bool(
            self.attack_config.get('attack_false_energy_receipt_enabled', True))
        self.attack_data_drop_probability = max(0.0, min(1.0, float(
            self.attack_config.get('attack_data_drop_probability', 1.0))))
        self.attack_control_drop_probability = max(0.0, min(1.0, float(
            self.attack_config.get('attack_control_drop_probability', 0.50))))
        self.attack_false_sos_mode = str(
            self.attack_config.get('attack_false_sos_mode', 'legacy_compatible')
        ).strip().lower()
        self.attack_false_sos_probability_per_epoch = max(0.0, min(1.0, float(
            self.attack_config.get('attack_false_sos_probability_per_epoch', 0.05))))
        self.attack_false_sos_energy_ratio = max(0.0, min(1.0, float(
            self.attack_config.get('attack_false_sos_energy_ratio', 0.05))))
        self.attack_control_epoch_s = max(1e-9, float(
            self.attack_config.get('attack_control_epoch_s', 10.0)))
        self.attack_tamper_enabled = bool(
            self.attack_config.get('attack_tamper_enabled', False))
        self.attack_tamper_probability = max(0.0, min(1.0, float(
            self.attack_config.get('attack_tamper_probability', 0.0))))
        self.attack_delay_enabled = bool(
            self.attack_config.get('attack_delay_enabled', False))
        self.attack_delay_probability = max(0.0, min(1.0, float(
            self.attack_config.get('attack_delay_probability', 0.0))))
        self.attack_delay_min_s = max(0.0, float(
            self.attack_config.get('attack_delay_min_s', 0.25)))
        self.attack_delay_max_s = max(self.attack_delay_min_s, float(
            self.attack_config.get('attack_delay_max_s', 1.00)))
        self.attack_replay_enabled = bool(
            self.attack_config.get('attack_replay_enabled', False))
        self.attack_replay_probability = max(0.0, min(1.0, float(
            self.attack_config.get('attack_replay_probability', 0.0))))
        self.attack_on_off_enabled = bool(
            self.attack_config.get('attack_on_off_enabled', False))
        self.attack_on_off_period_s = max(1e-9, float(
            self.attack_config.get('attack_on_off_period_s', 120.0)))
        self.attack_on_fraction = max(0.0, min(1.0, float(
            self.attack_config.get('attack_on_fraction', 0.50))))
        self.attack_on_off_phase_s = float(
            self.attack_config.get('attack_on_off_phase_s', 0.0))
        self._last_false_sos_epoch = None

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

    def set_malicious(self, value=True):
        self.is_malicious = bool(value)
        self.behavior_state = "MALICIOUS" if self.is_malicious else "HONEST"
        if self.is_malicious:
            self.remediation_time = None
            self.rehabilitation_time = None

    def remediate(self, current_time=None):
        """Move a malicious node into a truthful probation state.

        The baseline scenario does not schedule remediation.  In an explicit
        remediation scenario, SimulatorManager calls this method at
        ``attack_remediation_time_s``.  Attack behavior stops immediately, but
        the node remains labelled PROBATION until the configured probation
        interval has elapsed.
        """
        self.is_malicious = False
        self.behavior_state = "PROBATION"
        self.remediation_time = (
            None if current_time is None else float(current_time))
        self.rehabilitation_time = None

    def complete_rehabilitation(self, current_time=None):
        """Complete the local PROBATION -> HONEST state transition.

        This changes only the node's lifecycle label.  Attack behavior is
        already disabled by :meth:`remediate`, so this transition does not
        change packet generation, forwarding, charging receipts, or energy.
        """
        self.is_malicious = False
        self.behavior_state = "HONEST"
        self.rehabilitation_time = (
            None if current_time is None else float(current_time))

    def is_attack_active(self, current_time):
        if not self.is_malicious or self.behavior_state != "MALICIOUS":
            return False
        if not self.attack_on_off_enabled:
            return True
        phase = (float(current_time) + self.attack_on_off_phase_s) % self.attack_on_off_period_s
        return phase < self.attack_on_fraction * self.attack_on_off_period_s

    def _reported_energy(self, current_time, context='data'):
        if not self.is_attack_active(current_time):
            return float(self.E_current)
        enabled = {
            'data': self.attack_false_energy_data_enabled,
            'control': self.attack_false_energy_control_enabled,
            'receipt': self.attack_false_energy_receipt_enabled,
        }.get(str(context).lower(), False)
        if enabled:
            return max(0.0, float(self.E_max) * self.attack_false_energy_ratio)
        return float(self.E_current)

    def attack_delay_seconds(self, current_time):
        if not self.is_attack_active(current_time) or not self.attack_delay_enabled:
            return 0.0
        if self.rng.random() >= self.attack_delay_probability:
            return 0.0
        return self.rng.uniform(self.attack_delay_min_s, self.attack_delay_max_s)

    def should_replay(self, current_time):
        if not self.is_attack_active(current_time) or not self.attack_replay_enabled:
            return False
        return self.rng.random() < self.attack_replay_probability

    def _maybe_tamper_packet(self, packet, current_time):
        if not self.is_attack_active(current_time) or not self.attack_tamper_enabled:
            return False
        if self.rng.random() >= self.attack_tamper_probability:
            return False
        payload = packet.get('payload')
        if isinstance(payload, (int, float)):
            packet['payload'] = float(payload) + 1.0
        else:
            packet['payload'] = f"{payload}|tampered"
        return True

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

        packet = {
            "source_id": self.node_id,
            "type": "DATA",
            "priority": "HIGH" if is_critical else "NORMAL",
            "payload": raw_data_value,
            "timestamp": current_time,
            "size_bytes": self.data_packet_bytes,
            "size_bits": self.data_packet_bits,
            "energy_trace": []
        }

        self._transmit_packet(packet, is_critical)
        # Preserve the original baseline timing: DATA energy telemetry is the
        # residual value after the source radio transmission has been charged.
        # Corrected false-energy scenarios replace this post-TX value with the
        # configured malicious claim.
        packet["energy_trace"] = [(
            self.node_id,
            self._reported_energy(current_time, context='data')
        )]
        exec_time_ms = (time.time() - start_time) * 1000
        self.metrics.record_execution_time('EDGE', exec_time_ms)
        self._reset_inactivity_timer(current_time)
        return packet

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
        secondary_mode = (
            self.data_secondary_route_mode
            if is_data else self.secondary_route_mode
        )

        if primary_hop is not None:
            self._consume_energy(
                tx_bytes=packet["size_bytes"],
                tx_bits=packet.get("size_bits"),
                distance=self._distance_to_next_hop(primary_hop))
        if (is_critical and secondary_hop is not None
                and secondary_hop != primary_hop
                and str(secondary_mode).strip().lower() == 'redundant'):
            self._consume_energy(
                tx_bytes=packet["size_bytes"],
                tx_bits=packet.get("size_bits"),
                distance=self._distance_to_next_hop(secondary_hop))

    def forward_packet(self, packet, current_time, distance_to_next):
        if self.status == "DEAD":
            return False

        packet_type = str(packet.get('type', 'DATA')).upper()
        attack_active = self.is_attack_active(current_time)

        # Preserve the effective legacy DATA workload exactly: a malicious
        # relay is a blackhole and drops before RX/trace accounting.  The
        # corrected selective-forwarding scenarios use the normal path below.
        if (attack_active and packet_type == 'DATA'
                and self.attack_model_mode == 'legacy_compatible'
                and self.attack_data_drop_probability >= 1.0 - 1e-15):
            return False

        if "path_header" not in packet:
            packet["path_header"] = []

        packet["path_header"].append(self.node_id)
        self._consume_energy(rx_bytes=packet["size_bytes"],
                             rx_bits=packet.get("size_bits"))

        context = 'data' if packet_type == 'DATA' else 'control'
        reported_energy = self._reported_energy(current_time, context=context)

        if "energy_trace" not in packet:
            packet["energy_trace"] = []
        packet["energy_trace"].append((self.node_id, reported_energy))

        if attack_active:
            drop_probability = (
                self.attack_data_drop_probability
                if packet_type == 'DATA'
                else self.attack_control_drop_probability
            )
            if self.rng.random() < drop_probability:
                return False

            # Tampering is performed after source authentication was created.
            # The BS therefore detects it as a source-HMAC failure.
            self._maybe_tamper_packet(packet, current_time)

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

        if self.is_attack_active(current_time):
            if self.attack_false_sos_mode == 'legacy_compatible':
                # Exact compatibility path retained only for the thesis
                # baseline; corrected scenarios use a per-epoch probability
                # and never mutate physical battery state.
                # The random draw intentionally occurs even when ``sos_sent``
                # is already true because that is the exact evaluation order
                # of the previous implementation and therefore preserves the
                # node RNG stream used later by control forwarding.
                if self.rng.random() < 0.05 and not self.sos_sent:
                    temp_battery = self.E_current
                    self.E_current = 0.01
                    packet = self._send_sos_charge_request(current_time)
                    self.E_current = temp_battery
                    self.sos_sent = True
                    return packet
            elif (not self.sos_sent and
                  self.attack_false_sos_mode not in {'disabled', 'off', 'none'}):
                epoch = int(math.floor(float(current_time) / self.attack_control_epoch_s))
                if epoch != self._last_false_sos_epoch:
                    self._last_false_sos_epoch = epoch
                    if self.rng.random() < self.attack_false_sos_probability_per_epoch:
                        false_claim = max(
                            0.0, float(self.E_max) * self.attack_false_sos_energy_ratio)
                        self.sos_sent = True
                        return self._send_sos_charge_request(
                            current_time, reported_energy=false_claim)

        if battery_critical and long_silence and not self.sos_sent:
            self.sos_sent = True
            return self._send_sos_charge_request(current_time)

        if long_silence and not battery_critical:
            return self._send_keep_alive(current_time)
        return None

    def _send_sos_charge_request(self, current_time, reported_energy=None):
        energy_claim = (
            float(self.E_current)
            if reported_energy is None
            else max(0.0, float(reported_energy))
        )
        packet = {
            'source_id': self.node_id,
            'type': 'SOS_CHARGE',
            'priority': 'HIGH',
            'timestamp': current_time,
            'size_bytes': self.control_packet_bytes,
            'size_bits': self.control_packet_bits,
            'energy_trace': [(self.node_id, energy_claim)]
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
        revival_threshold = self.dead_node_revival_min_energy_ratio * max(
            float(self.E_max), 0.0)
        if (self.status == "DEAD" and self.enable_dead_node_revival
                and self.E_current + 1e-15 >= revival_threshold
                and actual > 0.0):
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
        if self.is_attack_active(timestamp) and self.attack_false_energy_receipt_enabled:
            reported_after = max(
                0.0, float(self.E_max) * self.attack_false_energy_ratio)
        claimed_received = max(0.0, float(received_energy))
        if self.is_attack_active(timestamp) and self.attack_false_energy_receipt_enabled:
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

    @staticmethod
    def _normalize_secondary_mode(mode, secondary_hop):
        if secondary_hop is None:
            return 'none'
        normalized = str(mode or 'redundant').strip().lower()
        return normalized if normalized in {'redundant', 'failover'} else 'redundant'

    def update_policies_from_bs(self, primary_hop, secondary_hop, new_threshold,
                                secondary_mode='redundant'):
        self.primary_next_hop = primary_hop
        self.secondary_next_hop = secondary_hop
        self.secondary_route_mode = self._normalize_secondary_mode(
            secondary_mode, secondary_hop)
        self.adaptive_threshold = new_threshold

    def update_data_routes(self, primary_hop, secondary_hop,
                           secondary_mode='redundant'):
        """Update DATA-only next hops without touching control/charging state."""
        self.data_primary_next_hop = primary_hop
        self.data_secondary_next_hop = secondary_hop
        self.data_secondary_route_mode = self._normalize_secondary_mode(
            secondary_mode, secondary_hop)
