import time
import math


class SensorNode:
    """
    Sensor Node Class for WSN-WRSN.
    Implements Edge Processing, Piggybacking, Multipath Routing for Critical Data,
    and Conditional Keep-Alive logic to achieve near-zero control overhead.
    """

    def __init__(self, node_id, x, y, initial_energy, metrics_collector):
        # 1. Identification & Location
        self.node_id = node_id
        self.x = x
        self.y = y

        # 2. Energy Model (Hardware Constraints)
        self.E_max = initial_energy
        self.E_current = initial_energy
        self.E_critical = initial_energy * 0.15  # 15% threshold for SOS charge request
        self.status = "ALIVE"  # Can be "ALIVE" or "DEAD"

        # 3. Routing Table (Provisioned/Pushed by Base Station)
        # Node does NOT calculate routes. It blindly follows BS instructions.
        self.primary_next_hop = None
        self.secondary_next_hop = None  # Node-disjoint path for critical data

        # 4. Edge Processing Parameters (Dynamic threshold from BS)
        self.adaptive_threshold = float('inf')  # Default, updated by BS

        # 5. Timer for Conditional Keep-Alive (Overhead Reduction)
        self.inactivity_timer = 0
        # Assume 3600 seconds (1 hour) of absolute silence
        self.T_max_inactivity = 3600
        self.last_activity_time = 0.0

        # 6. Global Observer Reference
        self.metrics = metrics_collector

    # =====================================================================
    # 1. EDGE PROCESSING & DATA SENSING
    # =====================================================================

    def sense_environment(self, raw_data_value, current_time):
        """
        Simulates sensing data. Performs Edge Evaluation against adaptive threshold.
        Creates packet and uses Piggybacking for battery telemetry.
        """
        if self.status == "DEAD":
            return None

        start_time = time.time()  # For Computational Overhead tracking

        # A. Edge Processing: Is the data Critical or Normal?
        is_critical = False
        if raw_data_value >= self.adaptive_threshold:
            is_critical = True

        # B. Piggybacking (Zero-overhead telemetry)
        # We attach current energy state directly to the data header
        packet = {
            "source_id": self.node_id,
            "type": "DATA",
            "priority": "HIGH" if is_critical else "NORMAL",
            "payload": raw_data_value,
            "piggybacked_energy": self.E_current,  # BS reads this to update E_claimed
            "timestamp": current_time,
            "size_bytes": 128  # Assume standard data packet size
        }

        # C. Select Routing Strategy
        self._transmit_packet(packet, is_critical)

        # Record Execution Time for Sensor (Should be ~0 ms in reality)
        exec_time_ms = (time.time() - start_time) * 1000
        self.metrics.record_execution_time('EDGE', exec_time_ms)

        # Reset Inactivity Timer since we just did something
        self._reset_inactivity_timer(current_time)

    # =====================================================================
    # 2. TRANSMISSION & MULTIPATH ROUTING
    # =====================================================================

    def _transmit_packet(self, packet, is_critical):
        """
        Handles actual transmission logic. Single path for normal, Multipath for critical.
        """
        if self.status == "DEAD":
            return

        # Strategy A: Normal Data -> Primary Path Only
        if not is_critical and self.primary_next_hop is not None:
            self._consume_energy(tx_bytes=packet["size_bytes"])
            # Record generation in Global Metrics
            self.metrics.update_packet_transmission(
                is_critical=False, is_control=False, size_bytes=packet["size_bytes"])
            # Pass to physical layer/next hop (Implementation depends on simulator engine)
            # self.primary_next_hop.receive_forward(packet)

        # Strategy B: Critical Data -> Multipath (Node-Disjoint)
        elif is_critical:
            # Send copy 1 via Primary
            if self.primary_next_hop is not None:
                self._consume_energy(tx_bytes=packet["size_bytes"])

            # Send copy 2 via Secondary
            if self.secondary_next_hop is not None:
                self._consume_energy(tx_bytes=packet["size_bytes"])

            # Record critical generation in Global Metrics (Counted as 1 logical critical event)
            self.metrics.update_packet_transmission(
                is_critical=True, is_control=False, size_bytes=packet["size_bytes"])

        # Note on MAC Auto-ACK:
        # In a real engine (like NS-3), if MAC ACK fails, retransmission occurs.
        # Here, we assume physical layer handles MAC ACK automatically without network layer overhead.

    # =====================================================================
    # 3. RELAY / FORWARDING FUNCTION
    # =====================================================================

    def forward_packet(self, packet, current_time):
        """
        Called when a neighboring node asks this node to relay a packet.
        Source Routing: Node just looks at the packet header or its own table.
        """
        if self.status == "DEAD":
            return False  # Packet dropped (Natural drop, not malicious)

        # Consume Receiver Energy
        self._consume_energy(rx_bytes=packet["size_bytes"])

        # Check if node is malicious (Stealthy Attacker / Blackhole simulation)
        # If this node was configured as a dropper, it returns False here.
        # if self.is_malicious_dropper: return False

        # Consume Transmitter Energy and forward to next hop
        self._consume_energy(tx_bytes=packet["size_bytes"])

        self._reset_inactivity_timer(current_time)
        return True  # MAC ACK Success

    # =====================================================================
    # 4. CONDITIONAL KEEP-ALIVE & SOS (GREEN PROTOCOL)
    # =====================================================================

    def check_inactivity(self, current_time):
        """
        Called periodically by simulator engine.
        Implements Conditional Keep-Alive to request MC only in absolute emergency.
        """
        if self.status == "DEAD":
            return

        silence_duration = current_time - self.last_activity_time

        # Check condition: Absolute Silence AND Critical Battery
        if silence_duration >= self.T_max_inactivity and self.E_current <= self.E_critical:
            self._send_sos_charge_request(current_time)

    def _send_sos_charge_request(self, current_time):
        """
        Zero-data, lightweight priority packet just to say 'I am dying and forgotten'.
        """
        sos_packet = {
            "source_id": self.node_id,
            "type": "SOS_CHARGE",
            "priority": "HIGH",  # Routed immediately via Multipath
            "piggybacked_energy": self.E_current,
            "timestamp": current_time,
            "size_bytes": 16  # Tiny control packet
        }

        self._transmit_packet(sos_packet, is_critical=True)
        self.metrics.update_packet_transmission(
            is_critical=False, is_control=True, size_bytes=16)
        self._reset_inactivity_timer(current_time)

    # =====================================================================
    # 5. ENERGY CONSUMPTION MODEL & MC INTERACTION
    # =====================================================================

    def _consume_energy(self, tx_bytes=0, rx_bytes=0):
        """
        Standard First Order Radio Model.
        """
        E_elec = 50e-9  # 50 nJ/bit
        E_amp = 100e-12  # 100 pJ/bit/m^2 (simplified)

        total_bits_tx = tx_bytes * 8
        total_bits_rx = rx_bytes * 8

        # Calculate TX & RX Energy (Simplified assuming average distance d)
        energy_tx = total_bits_tx * (E_elec + E_amp)
        energy_rx = total_bits_rx * E_elec

        self.E_current -= (energy_tx + energy_rx)

        # Check for death
        if self.E_current <= 0 and self.status == "ALIVE":
            self.E_current = 0
            self.status = "DEAD"
            self.metrics.record_node_death(
                self.node_id, time.time())  # Notify global observer

    def receive_wireless_charge(self, injected_energy):
        """
        Called by the Mobile Charger (MC) physically approaching the node.
        """
        if self.status == "DEAD":
            self.status = "ALIVE"  # Node revived!

        self.E_current += injected_energy
        if self.E_current > self.E_max:
            self.E_current = self.E_max

    def _reset_inactivity_timer(self, current_time):
        self.last_activity_time = current_time

    # =====================================================================
    # BS PROVISIONING (Called by Base Station)
    # =====================================================================

    def update_policies_from_bs(self, primary_hop, secondary_hop, new_threshold):
        """
        Called during Deployment Phase or Routing Update Phase by BS.
        """
        self.primary_next_hop = primary_hop
        self.secondary_next_hop = secondary_hop
        self.adaptive_threshold = new_threshold
