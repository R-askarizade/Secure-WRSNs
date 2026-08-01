import math
import random
import secrets
from typing import Dict, Iterable, Optional

from config import make_mc_mac


class MobileCharger:
    """Energy-conserving mobile charger with contact and RF-cluster modes.

    Physical constants are preserved from the original implementation:
      * travel energy cost: 0.1 J/m
      * charging/transmit power: 5.0 J/s (W)
      * speed: scenario value (original default 5.0 m/s)

    The class supports both immediate charging and asynchronous charging
    sessions.  The simulator uses asynchronous sessions so sensor/control
    events continue while the charger is moving or dwelling at a cluster.
    """

    def __init__(self, mc_id, depot_x, depot_y, capacity, speed, base_station,
                 metrics_collector, attack_config=None, rng_seed=None):
        self.mc_id = mc_id
        self.depot_x = float(depot_x)
        self.depot_y = float(depot_y)
        self.x = float(depot_x)
        self.y = float(depot_y)
        self.speed = max(float(speed), 1e-9)

        self.E_capacity = max(float(capacity), 0.0)
        self.E_current = self.E_capacity
        self.travel_energy_cost_per_m = 0.1
        self.charging_rate = 5.0

        self.status = "IDLE"
        self.crypto_keys = None
        self.zeroized = False
        self.requires_lab_reprovisioning = False

        self.bs = base_station
        self.metrics = metrics_collector
        self.report_counter = 0
        self.last_move_succeeded = True
        self.last_charge_details = {}
        self.attack_config = dict(attack_config or {})
        if rng_seed is None:
            seed_text = str(self.mc_id)
            rng_seed = sum((idx + 1) * ord(ch)
                           for idx, ch in enumerate(seed_text))
        self.attack_rng = random.Random(int(rng_seed))
        self.attack_energy_claim_enabled = bool(
            self.attack_config.get('mc_attack_energy_claim_enabled', False))
        self.attack_energy_claim_factor = max(0.0, float(
            self.attack_config.get('mc_attack_energy_claim_factor', 1.50)))
        self.attack_location_spoof_enabled = bool(
            self.attack_config.get('mc_attack_location_spoof_enabled', False))
        self.attack_location_spoof_offset_m = float(
            self.attack_config.get('mc_attack_location_spoof_offset_m', 100.0))
        self.attack_forged_hmac_enabled = bool(
            self.attack_config.get('mc_attack_forged_hmac_enabled', False))
        self.attack_replay_enabled = bool(
            self.attack_config.get('mc_attack_replay_enabled', False))
        self.attack_probability = max(0.0, min(1.0, float(
            self.attack_config.get('mc_attack_probability', 1.0))))
        self.attack_on_off_enabled = bool(
            self.attack_config.get('mc_attack_on_off_enabled', False))
        self.attack_on_off_period_s = max(1e-9, float(
            self.attack_config.get('mc_attack_on_off_period_s', 180.0)))
        self.attack_on_fraction = max(0.0, min(1.0, float(
            self.attack_config.get('mc_attack_on_fraction', 0.50))))

    def _attack_active(self, current_time):
        if not any((self.attack_energy_claim_enabled,
                    self.attack_location_spoof_enabled,
                    self.attack_forged_hmac_enabled,
                    self.attack_replay_enabled)):
            return False
        if self.attack_on_off_enabled:
            phase = float(current_time) % self.attack_on_off_period_s
            if phase >= self.attack_on_fraction * self.attack_on_off_period_s:
                return False
        return self.attack_rng.random() < self.attack_probability

    @staticmethod
    def _calculate_distance(x1, y1, x2, y2):
        return math.hypot(float(x2) - float(x1), float(y2) - float(y1))

    @staticmethod
    def _nodes_for_target(target_node) -> list:
        return list(getattr(target_node, "nodes", [target_node]))

    @staticmethod
    def _is_cluster(target_node) -> bool:
        return bool(getattr(target_node, "is_cluster", False))

    def _get_target_coords(self, target_node):
        if self._is_cluster(target_node):
            return float(target_node.centroid_x), float(target_node.centroid_y)
        return float(target_node.x), float(target_node.y)

    def distance_to_target(self, target_node) -> float:
        tx, ty = self._get_target_coords(target_node)
        return self._calculate_distance(self.x, self.y, tx, ty)

    def distance_target_to_depot(self, target_node) -> float:
        tx, ty = self._get_target_coords(target_node)
        return self._calculate_distance(tx, ty, self.depot_x, self.depot_y)

    def movement_energy_for_distance(self, distance: float) -> float:
        return max(0.0, float(distance)) * max(0.0, float(self.travel_energy_cost_per_m))

    def can_move_to(self, target_node, reserve_energy: float = 0.0) -> bool:
        required = self.movement_energy_for_distance(
            self.distance_to_target(target_node))
        return self.E_current + 1e-12 >= required + max(0.0, float(reserve_energy))

    def move_to_node(self, target_node, reserve_energy: float = 0.0):
        """Complete one movement leg without teleporting on failure.

        In asynchronous simulation this method is called at the scheduled
        arrival time.  It commits the already elapsed movement leg, deducts
        energy, updates position, and records distance/energy exactly once.
        """
        if target_node is None or self.zeroized:
            self.last_move_succeeded = False
            return None

        tx, ty = self._get_target_coords(target_node)
        distance = self._calculate_distance(self.x, self.y, tx, ty)
        energy_consumed = self.movement_energy_for_distance(distance)
        reserve_energy = max(0.0, float(reserve_energy))
        if self.E_current + 1e-12 < energy_consumed + reserve_energy:
            self.last_move_succeeded = False
            self.status = "IDLE"
            return None

        travel_time = distance / self.speed
        self.E_current -= energy_consumed
        self.x, self.y = tx, ty
        self.metrics.add_mc_travel_distance(distance, energy_consumed)
        self.last_move_succeeded = True
        self.status = "IDLE"
        return float(travel_time)

    def interrupt_and_rescue(self, target_node):
        self.status = "RESCUE_MISSION"
        original_speed = self.speed
        self.speed = original_speed * 1.5
        travel_time = self.move_to_node(target_node)
        charge_time = 0.0
        if travel_time is not None:
            charge_time = self.charge_node(target_node)
        self.speed = original_speed
        self.status = "IDLE"
        return (0.0 if travel_time is None else travel_time) + float(charge_time)

    def tpms_zeroization(self):
        self.crypto_keys = None
        self.zeroized = True
        self.requires_lab_reprovisioning = True
        self.status = "BRICKED"
        print("[TPM] Physical tampering detected. Keys zeroized; MC requires offline lab reprovisioning.")

    def return_to_depot(self):
        """Complete the return leg and replace the battery at the depot."""
        if self.zeroized:
            return None
        distance = self._calculate_distance(
            self.x, self.y, self.depot_x, self.depot_y)
        energy_required = self.movement_energy_for_distance(distance)
        if self.E_current + 1e-12 < energy_required:
            self.status = "STRANDED"
            return None

        travel_time = distance / self.speed
        self.E_current -= energy_required
        self.x, self.y = self.depot_x, self.depot_y
        self.metrics.add_mc_travel_distance(distance, energy_required)

        # Original system model: battery is replaced upon reaching the BS.
        self.E_current = self.E_capacity
        self.status = "IDLE"
        return float(travel_time)

    def _rf_receive_powers(self, target_node, nodes: Iterable) -> Dict[int, float]:
        """Return distance-aware, aggregate-energy-conserving RF powers."""
        nodes = list(nodes)
        radius = max(float(getattr(self.bs, "rf_charging_radius", 0.0)), 0.0)
        alpha = max(float(self.bs.ablation.get(
            "rf_path_loss_exponent", 2.0)), 1.0)
        d_ref = max(float(self.bs.ablation.get(
            "rf_reference_distance_m", 1.0)), 1e-6)
        eta = float(self.bs.ablation.get(
            "rf_energy_conversion_efficiency", getattr(self.bs, "eta_charging", 0.70)))
        eta = min(1.0, max(0.0, eta))
        tx, ty = self._get_target_coords(target_node)

        raw_gains = {}
        for node in nodes:
            distance = self._calculate_distance(tx, ty, node.x, node.y)
            if radius > 0.0 and distance > radius + 1e-9:
                gain = 0.0
            else:
                gain = min(1.0, (d_ref / max(distance, d_ref)) ** alpha)
            raw_gains[node.node_id] = gain

        gain_sum = sum(raw_gains.values())
        if gain_sum <= 1e-15:
            return {node.node_id: 0.0 for node in nodes}

        conservation_scale = min(1.0, 1.0 / gain_sum)
        available_rx_power = max(0.0, self.charging_rate) * eta
        return {
            node_id: available_rx_power * gain * conservation_scale
            for node_id, gain in raw_gains.items()
        }

    def estimate_charging_plan(self, target_node, energy_allocations: Optional[Dict[int, float]] = None):
        nodes = [n for n in self._nodes_for_target(target_node)
                 if getattr(n, "status", "ALIVE") in {"ALIVE", "DEAD"}]
        allocations = {}
        for node in nodes:
            missing = max(0.0, float(node.E_max) - float(node.E_current))
            requested = missing if energy_allocations is None else float(
                energy_allocations.get(node.node_id, 0.0))
            allocations[node.node_id] = min(missing, max(0.0, requested))

        is_cluster = self._is_cluster(target_node)
        if is_cluster:
            powers = self._rf_receive_powers(target_node, nodes)
        else:
            eta_contact = float(self.bs.ablation.get(
                'contact_charging_efficiency',
                self.bs.ablation.get('charging_energy_efficiency', 1.0)))
            eta_contact = min(1.0, max(0.0, eta_contact))
            powers = {node.node_id: max(float(self.charging_rate), 0.0) * eta_contact
                      for node in nodes}

        durations = {}
        for node in nodes:
            power = powers.get(node.node_id, 0.0)
            energy = allocations.get(node.node_id, 0.0)
            durations[node.node_id] = energy / \
                power if power > 1e-15 else math.inf

        finite_durations = [d for d in durations.values() if math.isfinite(d)]
        stay_time = max(finite_durations, default=0.0)
        impossible = any(
            allocations.get(node.node_id, 0.0) > 0.0 and
            not math.isfinite(durations.get(node.node_id, math.inf))
            for node in nodes
        )
        if impossible:
            stay_time = math.inf

        # In both modes the MC draws the original 5 W while charging.  For a
        # single contact node this equals delivered energy when no clipping is
        # needed; for RF it correctly accounts for propagation/conversion loss.
        mc_energy_cost = max(0.0, float(self.charging_rate)) * stay_time

        return {
            "is_cluster": is_cluster,
            "nodes": nodes,
            "allocations": allocations,
            "receive_powers": powers,
            "node_durations": durations,
            "stay_time": float(stay_time),
            "mc_energy_cost": float(mc_energy_cost),
        }

    def estimate_charging_energy(self, target_node, energy_allocations=None) -> float:
        return float(self.estimate_charging_plan(
            target_node, energy_allocations)["mc_energy_cost"])

    def begin_charging_session(self, target_node, energy_allocations=None,
                               start_time=None):
        """Create a nonblocking charging session for the event scheduler."""
        self.last_charge_details = {}
        if self.zeroized or self.crypto_keys is None or self.E_current <= 0.0:
            return None

        plan = self.estimate_charging_plan(target_node, energy_allocations)
        if not plan["nodes"] or not math.isfinite(plan["stay_time"]):
            return None

        power_draw = max(float(self.charging_rate), 1e-12)
        affordable_time = self.E_current / power_draw
        duration = min(float(plan["stay_time"]), affordable_time)
        if duration <= 1e-15:
            return None

        start = float(
            self.metrics.current_sim_time if start_time is None else start_time)
        session = {
            "target": target_node,
            "is_cluster": bool(plan["is_cluster"]),
            "nodes": list(plan["nodes"]),
            "allocations": dict(plan["allocations"]),
            "receive_powers": dict(plan["receive_powers"]),
            "duration": float(duration),
            "elapsed": 0.0,
            "start_time": start,
            "delivered": {node.node_id: 0.0 for node in plan["nodes"]},
            "missing_at_start": {
                node.node_id: max(0.0, float(node.E_max) -
                                  float(node.E_current))
                for node in plan["nodes"]
            },
            "energy_at_start": {
                node.node_id: max(0.0, float(node.E_current))
                for node in plan["nodes"]
            },
            "mc_energy_spent": 0.0,
            "finalized": False,
        }
        self.status = "CHARGING"
        return session

    def advance_charging_session(self, session, delta_t):
        """Advance RF/contact charging continuously over ``delta_t`` seconds."""
        if not session or session.get("finalized", False):
            return 0.0
        remaining_time = max(0.0, float(
            session["duration"]) - float(session["elapsed"]))
        dt = min(max(0.0, float(delta_t)), remaining_time)
        power_draw = max(float(self.charging_rate), 1e-12)
        dt = min(dt, self.E_current / power_draw)
        if dt <= 1e-15:
            return 0.0

        for node in session["nodes"]:
            node_id = node.node_id
            requested_remaining = max(
                0.0,
                float(session["allocations"].get(node_id, 0.0))
                - float(session["delivered"].get(node_id, 0.0)),
            )
            power = max(0.0, float(
                session["receive_powers"].get(node_id, 0.0)))
            missing_now = max(0.0, float(node.E_max) - float(node.E_current))
            offered = min(requested_remaining, power * dt, missing_now)
            if offered <= 1e-15:
                continue
            actual = float(node.receive_wireless_charge(offered))
            session["delivered"][node_id] += max(0.0, actual)

        emitted = min(self.E_current, power_draw * dt)
        self.E_current = max(0.0, self.E_current - emitted)
        session["mc_energy_spent"] += emitted
        session["elapsed"] += dt
        return float(dt)

    def finish_charging_session(self, session, completion_time=None):
        """Finalize reports, security checks, and metrics for one session."""
        if not session or session.get("finalized", False):
            return self.last_charge_details

        now = float(
            self.metrics.current_sim_time if completion_time is None else completion_time)
        self.metrics.current_sim_time = now
        is_cluster = bool(session["is_cluster"])
        charged_node_count = 0
        total_delivered = 0.0
        was_partial = False
        per_node_details = {}

        for node in session["nodes"]:
            node_id = node.node_id
            actual_injected = max(0.0, float(
                session["delivered"].get(node_id, 0.0)))
            if actual_injected <= 1e-15:
                continue
            charged_node_count += 1
            total_delivered += actual_injected
            initial_missing = max(0.0, float(
                session["missing_at_start"].get(node_id, 0.0)))
            was_partial = was_partial or (
                actual_injected + 1e-12 < initial_missing)
            power = max(0.0, float(
                session["receive_powers"].get(node_id, 0.0)))

            allowed_radius = (
                float(getattr(self.bs, "rf_charging_radius", 0.0))
                if is_cluster else
                float(self.bs.ablation.get("contact_location_tolerance_m", 5.0))
            )

            self.report_counter += 1
            nonce = secrets.token_hex(12)
            attack_active = self._attack_active(now)
            reported_injected = float(actual_injected)
            reported_x = float(self.x)
            reported_y = float(self.y)
            if attack_active and self.attack_energy_claim_enabled:
                reported_injected *= self.attack_energy_claim_factor
            if attack_active and self.attack_location_spoof_enabled:
                reported_x += self.attack_location_spoof_offset_m
                reported_y += self.attack_location_spoof_offset_m
            report_key = self.crypto_keys
            if attack_active and self.attack_forged_hmac_enabled:
                report_key = f"FORGED-{self.mc_id}"
            mac_tag = make_mc_mac(
                report_key, node_id, reported_injected, reported_x, reported_y,
                mc_id=self.mc_id, counter=self.report_counter,
                nonce=nonce, timestamp=now)
            expected_injected = min(
                float(session["allocations"].get(node_id, 0.0)),
                power * float(session["elapsed"]),
                initial_missing)
            accepted = self.bs.report_mc_injection_oob(
                node_id, reported_injected, reported_x, reported_y, mac_tag,
                mc_id=self.mc_id, counter=self.report_counter,
                nonce=nonce, timestamp=now,
                charging_mode=("rf_cluster" if is_cluster else "contact"),
                allowed_radius=allowed_radius,
                expected_injected_energy=expected_injected)
            replay_accepted = None
            if attack_active and self.attack_replay_enabled:
                # The compromised charger retransmits the exact same signed
                # report.  Counter/nonce freshness must reject this second copy.
                replay_accepted = self.bs.report_mc_injection_oob(
                    node_id, reported_injected, reported_x, reported_y, mac_tag,
                    mc_id=self.mc_id, counter=self.report_counter,
                    nonce=nonce, timestamp=now,
                    charging_mode=("rf_cluster" if is_cluster else "contact"),
                    allowed_radius=allowed_radius,
                    expected_injected_energy=expected_injected)
            if not accepted:
                print(
                    f"[MC AUTH] Charging report from MC {self.mc_id} "
                    f"for Node {node_id} was rejected by BS.")

            receipt = node.create_charge_receipt(
                energy_before=float(session["energy_at_start"].get(
                    node_id, max(0.0, float(node.E_current) - actual_injected))),
                energy_after=float(node.E_current),
                received_energy=actual_injected,
                mc_id=self.mc_id, timestamp=now)
            receipt_consistent = self.bs.report_node_charge_receipt(
                receipt, expected_injected_energy=expected_injected,
                mc_report_accepted=accepted)

            recorded_time = float(
                session["elapsed"]) if charged_node_count == 1 else 0.0
            self.metrics.record_mc_charge(
                node_id, actual_injected, recorded_time, now)
            per_node_details[node_id] = {
                "requested_energy": float(session["allocations"].get(node_id, 0.0)),
                "injected_energy": actual_injected,
                "receive_power": power,
                "effective_duration": actual_injected / max(power, 1e-12),
                "distance_to_stop": self._calculate_distance(
                    self.x, self.y, node.x, node.y),
                "mc_report_accepted": bool(accepted),
                "mc_replay_accepted": replay_accepted,
                "node_receipt_consistent": bool(receipt_consistent),
            }

        if is_cluster and charged_node_count > 0:
            self.metrics.rf_cluster_charge_events += 1
            self.metrics.rf_total_nodes_charged_in_clusters += charged_node_count
        if was_partial and charged_node_count > 0:
            self.metrics.dynamic_partial_charge_events += 1
        self.metrics.record_mc_charging_energy(
            session["mc_energy_spent"], rf_mode=is_cluster,
            delivered_energy=total_delivered)

        session["finalized"] = True
        self.status = "IDLE"
        self.last_charge_details = {
            "is_cluster": is_cluster,
            "charged_node_count": charged_node_count,
            "stay_time": float(session["elapsed"]),
            "mc_energy_spent": float(session["mc_energy_spent"]),
            "total_delivered_energy": total_delivered,
            "partial": was_partial,
            "per_node": per_node_details,
        }
        return self.last_charge_details

    def charge_node(self, target_node, energy_allocations=None, partial_durations=None):
        """Immediate compatibility wrapper around the asynchronous session API."""
        if energy_allocations is None and partial_durations:
            energy_allocations = {
                int(node_id): max(0.0, float(duration)) * self.charging_rate
                for node_id, duration in partial_durations.items()
            }
        session = self.begin_charging_session(
            target_node, energy_allocations,
            start_time=float(self.metrics.current_sim_time))
        if session is None:
            return 0.0
        self.advance_charging_session(session, session["duration"])
        self.finish_charging_session(
            session, completion_time=float(self.metrics.current_sim_time) + session["elapsed"])
        return float(session["elapsed"])

    def _report_to_base_station(self, node_id, injected_energy):
        """Fallback unauthenticated reporting logic."""
        self.bs.mc_real_injected_energy[node_id] += injected_energy
