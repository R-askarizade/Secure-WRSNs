import math
import time
import secrets
from config import make_mc_mac


class MobileCharger:
    def __init__(self, mc_id, depot_x, depot_y, capacity, speed, base_station, metrics_collector):
        self.mc_id = mc_id
        self.depot_x = depot_x
        self.depot_y = depot_y
        self.x = depot_x
        self.y = depot_y
        self.speed = speed

        self.E_capacity = capacity
        self.E_current = capacity
        self.travel_energy_cost_per_m = 0.1
        self.charging_rate = 5.0

        self.status = "IDLE"
        self.crypto_keys = None
        self.zeroized = False
        self.requires_lab_reprovisioning = False

        self.bs = base_station
        self.metrics = metrics_collector
        self.report_counter = 0

    def _calculate_distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def move_to_node(self, target_node):
        self.status = "MOVING"
        distance = self._calculate_distance(
            self.x, self.y, target_node.x, target_node.y)
        travel_time = distance / self.speed
        energy_consumed = distance * self.travel_energy_cost_per_m
        self.E_current = max(0.0, self.E_current - energy_consumed)
        self.x = target_node.x
        self.y = target_node.y
        self.metrics.add_mc_travel_distance(distance)
        return travel_time

    def interrupt_and_rescue(self, target_node):
        self.status = "RESCUE_MISSION"
        original_speed = self.speed
        self.speed = self.speed * 1.5
        self.move_to_node(target_node)
        self.charge_node(target_node)
        self.speed = original_speed
        self.status = "IDLE"

    def tpms_zeroization(self):
        self.crypto_keys = None
        self.zeroized = True
        self.requires_lab_reprovisioning = True
        self.status = "BRICKED"
        print("[TPM] Physical tampering detected. Keys zeroized; MC requires offline lab reprovisioning.")

    def return_to_depot(self):
        self.status = "RETURNING"
        distance = self._calculate_distance(
            self.x, self.y, self.depot_x, self.depot_y)
        travel_time = distance / self.speed
        self.E_current = max(0.0, self.E_current - (distance * self.travel_energy_cost_per_m))
        self.x = self.depot_x
        self.y = self.depot_y
        self.metrics.add_mc_travel_distance(distance)
        self.E_current = self.E_capacity
        self.status = "IDLE"
        return travel_time

    def charge_node(self, target_node):
        if self.zeroized or self.crypto_keys is None:
            return 0
        if self.E_current <= 0:
            return 0

        before_e = float(target_node.E_current)
        now = float(self.metrics.current_sim_time)

        self.status = "CHARGING"
        needed_energy = target_node.E_max - target_node.E_current
        injected_energy = min(needed_energy, self.E_current)
        if injected_energy <= 0:
            self.status = "IDLE"
            return 0

        target_node.receive_wireless_charge(injected_energy)
        self.E_current = max(0.0, self.E_current - injected_energy)
        charging_time = injected_energy / self.charging_rate

        self.bs.verify_charging_location(target_node.node_id, self.x, self.y)
        self.report_counter += 1
        nonce = secrets.token_hex(12)
        report_time = float(self.metrics.current_sim_time)

        mac_tag = make_mc_mac(self.crypto_keys, target_node.node_id, injected_energy, self.x, self.y,
                              mc_id=self.mc_id, counter=self.report_counter, nonce=nonce, timestamp=report_time)
        accepted = self.bs.report_mc_injection_oob(
            target_node.node_id, injected_energy, self.x, self.y, mac_tag,
            mc_id=self.mc_id, counter=self.report_counter, nonce=nonce, timestamp=report_time)

        if not accepted:
            print(
                f"[MC AUTH] Charging report from MC {self.mc_id} was rejected by BS.")

        self.status = "IDLE"
        after_e = float(target_node.E_current)
        actual_injected = max(0.0, after_e - before_e)
        self.metrics.record_mc_charge(
            target_node.node_id, actual_injected, charging_time, now)
        return charging_time

    def _report_to_base_station(self, node_id, injected_energy):
        self.bs.mc_real_injected_energy[node_id] += injected_energy
