import math
import time


class MobileCharger:
    """
    Mobile Charger (MC) Class for Centralized WSN.
    Currently implemented as a simple baseline for movement and charging,
    but fully compatible with the Base Station's security formulas (OOB reporting).
    """

    def __init__(self, mc_id, depot_x, depot_y, capacity, speed, base_station, metrics_collector):
        # 1. Identification & Physical Attributes
        self.mc_id = mc_id
        self.depot_x = depot_x
        self.depot_y = depot_y
        self.x = depot_x
        self.y = depot_y
        self.speed = speed  # meters per second

        # 2. Energy Constraints
        self.E_capacity = capacity
        self.E_current = capacity
        self.travel_energy_cost_per_m = 0.1  # Joules consumed per meter of travel
        self.charging_rate = 5.0  # Joules transferred per second

        # 3. Status
        self.status = "IDLE"  # States: IDLE, MOVING, CHARGING, RETURNING

        # 4. System References (Crucial for WSN architecture)
        self.bs = base_station  # Out-of-band communication link
        self.metrics = metrics_collector

    # =====================================================================
    # 1. MOVEMENT & KINEMATICS
    # =====================================================================

    def _calculate_distance(self, x1, y1, x2, y2):
        """Standard Euclidean distance."""
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def move_to_node(self, target_node):
        """
        Moves the MC to the target node's location.
        Returns the time taken for the travel (used by simulator clock).
        """
        self.status = "MOVING"

        distance = self._calculate_distance(
            self.x, self.y, target_node.x, target_node.y)
        travel_time = distance / self.speed

        # Consume MC battery for moving
        energy_consumed = distance * self.travel_energy_cost_per_m
        self.E_current -= energy_consumed

        # Update Position
        self.x = target_node.x
        self.y = target_node.y

        # Record distance in Global Metrics (For WRSN Evaluation)
        self.metrics.add_mc_travel_distance(distance)

        return travel_time

    def return_to_depot(self):
        """
        Moves the MC back to the depot to recharge itself.
        """
        self.status = "RETURNING"
        distance = self._calculate_distance(
            self.x, self.y, self.depot_x, self.depot_y)
        travel_time = distance / self.speed

        self.E_current -= (distance * self.travel_energy_cost_per_m)
        self.x = self.depot_x
        self.y = self.depot_y

        self.metrics.add_mc_travel_distance(distance)

        # Recharge MC fully at depot
        self.E_current = self.E_capacity
        self.status = "IDLE"

        return travel_time

    # =====================================================================
    # 2. CHARGING PROCESS & SECURITY HANDSHAKE
    # =====================================================================

    def charge_node(self, target_node):
        """
        Transfers energy to the node and securely reports the transaction 
        to the Base Station (Out-of-band link) for anti-spoofing logic.
        """
        if self.E_current <= 0:
            return 0  # MC is out of battery

        self.status = "CHARGING"

        # Calculate how much energy the node actually needs
        needed_energy = target_node.E_max - target_node.E_current

        # Ensure MC has enough energy to give
        injected_energy = min(needed_energy, self.E_current)

        # 1. Physical Layer: Transfer energy to the sensor
        target_node.receive_wireless_charge(injected_energy)
        self.E_current -= injected_energy

        # Calculate time spent charging
        charging_time = injected_energy / self.charging_rate

        # 2. Cyber-Security Layer: Out-of-band reporting to BS
        # This is EXACTLY how we defeat Mobile Charger Spoofing & Hoarders!
        self._report_to_base_station(target_node.node_id, injected_energy)

        self.status = "IDLE"
        return charging_time

    def _report_to_base_station(self, node_id, injected_energy):
        """
        Secure, Zero-Overhead communication to Base Station.
        Informs BS exactly how much energy was given to a specific node.
        """
        # In a real network, this uses AES-256 over 4G/LoRa.
        # Here, we directly update the BS's secure dictionary.
        self.bs.mc_real_injected_energy[node_id] += injected_energy
