from __future__ import annotations

import math
import random
from collections import deque
from typing import Optional, Sequence, Tuple

import numpy as np

# PyTorch is optional. When it is unavailable, ChargerRoutingDQN uses
# a lightweight linear Q-function so the simulator can still run.
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None


def _resolve_torch_device(device: Optional[str]):
    """Resolve a safe PyTorch device with graceful CUDA fallback.

    ``None`` means CUDA when available and CPU otherwise.  An explicitly
    requested CUDA device also falls back to CPU instead of failing at model
    construction when CUDA is unavailable.
    """
    if torch is None or nn is None or optim is None:
        return None
    requested = "auto" if device is None else str(device).strip().lower()
    if requested in {"", "auto", "adaptive", "derived", "none"}:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)


def _is_adaptive_value(value) -> bool:
    """Return True for unset/string values that request analytical tuning."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "auto", "adaptive", "derived", "none"}
    return False


def _next_power_of_two(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def derive_adaptive_charging_hyperparameters(
    *,
    num_nodes: int,
    sim_time: float,
    network_size: Tuple[float, float],
    mobile_charger_count: int,
    mobile_charger_speed: float,
    charging_rate: float,
    sensor_initial_energy: float,
    transmission_range: float,
    control_epoch_s: float = 10.0,
    safe_horizon_s: Optional[float] = None,
    feature_dim: int = 7,
) -> dict:
    """Derive charging/DQN hyperparameters from physical and scale variables.

    The formulas intentionally avoid scenario-specific constants such as a
    fixed candidate count of 12.  They are deterministic functions of network
    size, node density, radio coverage, fleet size, charger speed and the
    simulation/control time scales.  Explicit numeric user settings can still
    override these values in ``SimulatorManager``.
    """
    n = max(1, int(num_nodes))
    m = max(1, int(mobile_charger_count))
    width = max(float(network_size[0]), 1e-6)
    height = max(float(network_size[1]), 1e-6)
    area = width * height
    diagonal = math.hypot(width, height)
    speed = max(float(mobile_charger_speed), 1e-6)
    radio_range = max(float(transmission_range), 1e-6)
    epoch = max(float(control_epoch_s), 1e-6)
    horizon = max(float(safe_horizon_s or sim_time), epoch)

    # Equal-area disk approximation: E[r] = 2R/3 for a uniformly distributed
    # point.  It estimates a single charger's spatial service distance.
    service_radius = math.sqrt(area / (math.pi * m))
    expected_travel_distance = (2.0 / 3.0) * service_radius
    expected_travel_time = expected_travel_distance / speed

    density = n / area
    expected_radio_neighbors = max(1.0, density * math.pi * radio_range**2)
    nodes_per_charger = max(1.0, n / m)

    # Square-root decomposition balances a bounded DQN action pool against the
    # local service population. Radio-neighbour density prevents the pool from
    # exceeding the amount of locally observable network structure.
    local_scale = max(1.0, min(nodes_per_charger, expected_radio_neighbors))
    candidate_budget = max(2, int(math.ceil(math.sqrt(local_scale))))

    # One mission includes travel plus one control epoch. Charging time is tiny
    # for the current sensor scale, but a full-sensor term keeps the formula
    # valid for larger batteries and slower chargers.
    full_charge_time = max(float(sensor_initial_energy), 0.0) / max(
        float(charging_rate), 1e-12
    )
    expected_mission_time = max(
        epoch, expected_travel_time + full_charge_time + epoch
    )
    expected_decisions = max(
        1, int(math.ceil(float(sim_time) / expected_mission_time)))

    # A complete fleet service cycle provides a physically meaningful cooldown
    # and useful-charge protection horizon.
    service_cycle_time = max(
        epoch, int(math.ceil(nodes_per_charger)) * expected_mission_time
    )

    # Network width is tied to input/action representation rather than a magic
    # 64. The next power of two remains GPU-friendly.
    hidden_dim = _next_power_of_two(
        max(feature_dim + 1, feature_dim * candidate_budget))
    parameter_count = (feature_dim * hidden_dim + hidden_dim) + (
        hidden_dim * hidden_dim + hidden_dim
    ) + (hidden_dim + 1)

    replay_capacity = max(candidate_budget**2, expected_decisions)
    batch_size = max(2, min(replay_capacity, int(
        math.ceil(math.sqrt(replay_capacity)))))
    target_sync_interval = max(1, int(math.ceil(math.sqrt(batch_size))))

    # Coupon-coverage-style warm-up: enough competitive observations to expose
    # each candidate position across a logarithmic number of comparisons.
    warmup_competitive = max(1, int(math.ceil(
        candidate_budget * math.log(candidate_budget + 1.0)
    )))

    # Continuous-time discount mapped onto one expected mission.
    gamma = math.exp(-expected_mission_time / horizon)

    # Parameter/sample-normalized Adam step. This replaces a fixed 1e-3 while
    # naturally shrinking for larger networks and wider models.
    dqn_lr = 1.0 / math.sqrt(max(1.0, parameter_count * expected_decisions))
    qifl_params = 9 * 5
    qifl_updates_per_node = max(1.0, float(sim_time) / epoch)
    qifl_lr = 1.0 / math.sqrt(qifl_params * qifl_updates_per_node)

    # A node is hard-deadline critical when it cannot safely wait for one
    # expected mission. This replaces an unexplained fixed number of seconds.
    hard_deadline_margin = expected_mission_time

    return {
        "network_diagonal": float(diagonal),
        "dqn_max_candidates": int(candidate_budget),
        "max_adaptive_charge_queue": int(m * candidate_budget),
        "dqn_hidden_dim": int(hidden_dim),
        "dqn_replay_capacity": int(replay_capacity),
        "dqn_batch_size": int(batch_size),
        "dqn_target_sync_interval": int(target_sync_interval),
        "dqn_warmup_competitive_decisions": int(warmup_competitive),
        "dqn_gamma": float(gamma),
        "dqn_learning_rate": float(dqn_lr),
        "dqn_epsilon": None,  # uncertainty-driven exploration at decision time
        "dqn_exploration_mode": "uncertainty_adaptive",
        "dqn_safety_override_mode": "deadline_feasibility",
        "qifl_learning_rate": float(qifl_lr),
        "hard_deadline_margin_s": float(hard_deadline_margin),
        "dqn_deadline_safety_margin_s": float(hard_deadline_margin),
        "charging_reentry_cooldown_s": float(service_cycle_time),
        "adaptive_service_cycle_s": float(service_cycle_time),
        "adaptive_expected_mission_time_s": float(expected_mission_time),
        "adaptive_expected_travel_distance": float(expected_travel_distance),
        "adaptive_expected_radio_neighbors": float(expected_radio_neighbors),
    }


class QuantumFuzzyChargingOracle:
    """
    Adaptive probabilistic fuzzy oracle for calculating a charging threshold.

    This class is unchanged in its decision logic. It estimates a threshold
    from energy-depletion and spatial-difficulty memberships and can update
    its rule parameters online when PyTorch is enabled.
    """

    def __init__(
        self,
        threshold_centers: Sequence[float] = (0.18, 0.26, 0.34, 0.42, 0.50),
        lr: Optional[float] = 2.5e-3,
        device: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        self.centers = np.asarray(threshold_centers, dtype=float)
        self.last_loss = 0.0

        self.device = _resolve_torch_device(device)
        self.enabled_torch = self.device is not None

        if self.enabled_torch:
            torch.manual_seed(seed)

            logits = self._initial_rule_logits()
            self.theta = torch.nn.Parameter(
                torch.tensor(
                    logits,
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            if _is_adaptive_value(lr):
                lr = 1.0 / math.sqrt(max(1.0, self.theta.numel()))
            self.optimizer = optim.Adam([self.theta], lr=float(lr))
            self.centers_t = torch.tensor(
                self.centers,
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self.device = None
            self.theta_np = self._initial_rule_probs()

    @staticmethod
    def _triangular(x: float, a: float, b: float, c: float) -> float:
        if x <= a or x >= c:
            return 0.0
        if x == b:
            return 1.0
        if x < b:
            return (x - a) / max(b - a, 1e-12)
        return (c - x) / max(c - b, 1e-12)

    def _memberships(self, x: float) -> np.ndarray:
        x = float(np.clip(x, 0.0, 1.0))
        return np.asarray(
            [
                max(0.0, min(1.0, (0.50 - x) / 0.50)),
                self._triangular(x, 0.20, 0.50, 0.80),
                max(0.0, min(1.0, (x - 0.50) / 0.50)),
            ],
            dtype=float,
        )

    def _input_state_np(self, ed_norm: float, sd_norm: float) -> np.ndarray:
        ed_m = self._memberships(ed_norm)
        sd_m = self._memberships(sd_norm)

        state = np.outer(ed_m, sd_m).reshape(-1)
        total = float(np.sum(state))

        if total <= 1e-12:
            state[:] = 1.0 / 9.0
        else:
            state /= total

        return state

    def _initial_rule_probs(self) -> np.ndarray:
        table = np.asarray(
            [
                [0.02, 0.08, 0.20, 0.38, 0.32],  # ED low, SD low
                [0.01, 0.05, 0.18, 0.38, 0.38],
                [0.01, 0.03, 0.14, 0.34, 0.48],
                [0.08, 0.22, 0.38, 0.24, 0.08],  # ED medium
                [0.05, 0.18, 0.36, 0.30, 0.11],
                [0.03, 0.12, 0.33, 0.34, 0.18],
                [0.45, 0.32, 0.16, 0.05, 0.02],  # ED high
                [0.36, 0.34, 0.20, 0.07, 0.03],
                [0.28, 0.35, 0.24, 0.09, 0.04],
            ],
            dtype=float,
        )
        table /= np.maximum(table.sum(axis=1, keepdims=True), 1e-12)
        return table

    def _initial_rule_logits(self) -> np.ndarray:
        probabilities = self._initial_rule_probs()
        return np.log(np.maximum(probabilities, 1e-6))

    def infer(
        self,
        ed_norm: float,
        sd_norm: float,
    ) -> Tuple[float, np.ndarray]:
        psi = self._input_state_np(ed_norm, sd_norm)

        if self.enabled_torch:
            with torch.inference_mode():
                psi_t = torch.tensor(
                    psi,
                    dtype=torch.float32,
                    device=self.device,
                )
                rule_probabilities = torch.softmax(self.theta, dim=1)
                probabilities = torch.matmul(psi_t, rule_probabilities)
                probabilities = probabilities / torch.clamp(
                    probabilities.sum(),
                    min=1e-12,
                )
                threshold = torch.sum(
                    probabilities * self.centers_t
                ).item()

                return (
                    float(threshold),
                    probabilities.detach().cpu().numpy(),
                )

        probabilities = psi @ self.theta_np
        probabilities = probabilities / max(
            float(np.sum(probabilities)),
            1e-12,
        )

        return (
            float(np.sum(probabilities * self.centers)),
            probabilities,
        )

    def train_step(
        self,
        ed_norm: float,
        sd_norm: float,
        target_threshold: float,
    ) -> float:
        target_threshold = float(target_threshold)

        if not self.enabled_torch:
            prediction, _ = self.infer(ed_norm, sd_norm)
            self.last_loss = float(
                (prediction - target_threshold) ** 2
            )
            return self.last_loss

        psi = torch.tensor(
            self._input_state_np(ed_norm, sd_norm),
            dtype=torch.float32,
            device=self.device,
        )
        target = torch.tensor(
            target_threshold,
            dtype=torch.float32,
            device=self.device,
        )

        rule_probabilities = torch.softmax(self.theta, dim=1)
        probabilities = torch.matmul(psi, rule_probabilities)
        probabilities = probabilities / torch.clamp(
            probabilities.sum(),
            min=1e-12,
        )

        prediction = torch.sum(probabilities * self.centers_t)
        entropy = -torch.sum(
            probabilities
            * torch.log(torch.clamp(probabilities, min=1e-8))
        )

        loss = (
            torch.nn.functional.smooth_l1_loss(prediction, target)
            + 1e-4 * entropy
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([self.theta], 1.0)
        self.optimizer.step()

        self.last_loss = float(loss.detach().cpu().item())
        return self.last_loss


if nn is not None:

    class _QNet(nn.Module):
        """Small Q-network that scores one candidate node at a time."""

        def __init__(self, input_dim: int = 7, hidden_dim: int = 64) -> None:
            super().__init__()
            hidden_dim = max(2, int(hidden_dim))
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

else:

    class _QNet:
        """Placeholder used only when PyTorch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PyTorch is required to instantiate _QNet.")


class ChargerRoutingDQN:
    """
    Safety-shielded DQN for selecting the next charging target.

    Decision structure
    ------------------
    1. Security/admission logic outside this class supplies feasible nodes.
    2. Immediate rescue nodes form a safety-restricted candidate subset.
    3. DQN proposes the primary routing decision inside that safe subset.
    4. During genuinely competitive warm-up decisions, the greedy heuristic
       temporarily controls selection.
    5. After warm-up, the heuristic overrides DQN only when the DQN proposal
       is substantially less safe according to the configured margin.
    6. Epsilon exploration is restricted to the safe subset.

    Important
    ---------
    ``combined_score = q_value + heuristic_score`` is retained only as a
    diagnostic value for backward-compatible reporting. It is not the main
    post-warm-up selection rule.
    """

    def __init__(
        self,
        lr: Optional[float] = 1e-3,
        gamma: Optional[float] = 0.80,
        epsilon: Optional[float] = 0.08,
        device: Optional[str] = None,
        seed: int = 123,
        *,
        warmup_updates: int = 50,
        warmup_competitive_decisions: Optional[int] = None,
        critical_energy_ratio: float = 0.15,
        deadline_safety_margin: float = 2.0,
        heuristic_override_margin: Optional[float] = 0.75,
        safety_override_mode: str = "deadline_feasibility",
        exploration_mode: str = "fixed",
        hidden_dim: int = 64,
        replay_capacity: int = 512,
        batch_size: int = 16,
        target_sync_interval: int = 10,
        expected_mission_time: Optional[float] = None,
        safe_horizon: Optional[float] = None,
        explanation_history_size: int = 1000,
    ) -> None:
        self.expected_mission_time = max(
            1e-6, float(expected_mission_time or deadline_safety_margin or 1.0)
        )
        self.safe_horizon = max(
            self.expected_mission_time, float(safe_horizon or 3600.0)
        )
        self.gamma = (
            math.exp(-self.expected_mission_time / self.safe_horizon)
            if _is_adaptive_value(gamma)
            else float(gamma)
        )
        self.fixed_epsilon = None if _is_adaptive_value(
            epsilon) else float(epsilon)
        self.epsilon = 0.0 if self.fixed_epsilon is None else self.fixed_epsilon
        self.exploration_mode = str(
            exploration_mode or "fixed").strip().lower()
        self.safety_override_mode = str(
            safety_override_mode or "deadline_feasibility"
        ).strip().lower()

        self.warmup_updates = max(0, int(warmup_updates))
        if warmup_competitive_decisions is None:
            warmup_competitive_decisions = self.warmup_updates
        self.warmup_competitive_decisions = max(
            0, int(warmup_competitive_decisions)
        )
        self.competitive_decisions = 0
        self.critical_energy_ratio = float(critical_energy_ratio)
        self.deadline_safety_margin = max(0.0, float(deadline_safety_margin))
        self.heuristic_override_margin = (
            None if _is_adaptive_value(heuristic_override_margin)
            else float(heuristic_override_margin)
        )

        self.hidden_dim = max(2, int(hidden_dim))
        self.batch_size = max(1, int(batch_size))
        self.target_sync_interval = max(1, int(target_sync_interval))
        self.decisions = 0
        self.last_loss = 0.0
        self._updates = 0
        self.replay = deque(maxlen=max(self.batch_size, int(replay_capacity)))

        self.last_decision_explanation = None
        self.explanation_history = deque(
            maxlen=max(1, int(explanation_history_size))
        )

        self.device = _resolve_torch_device(device)
        self.enabled_torch = self.device is not None

        random.seed(seed)
        np.random.seed(seed)

        if self.enabled_torch:
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

            self.net = _QNet(hidden_dim=self.hidden_dim).to(self.device)
            self.target = _QNet(hidden_dim=self.hidden_dim).to(self.device)
            self.target.load_state_dict(self.net.state_dict())
            if _is_adaptive_value(lr):
                parameter_count = sum(p.numel() for p in self.net.parameters())
                lr = 1.0 / math.sqrt(
                    max(1.0, parameter_count * max(1, self.replay.maxlen))
                )
            self.lr = float(lr)
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        else:
            self.device = None
            self.weights = np.zeros(7, dtype=np.float32)
            self.bias = 0.0
            self.lr = float(1e-3 if _is_adaptive_value(lr) else lr)

    @staticmethod
    def _candidate_energy_and_deadline(
        mc_x: float,
        mc_y: float,
        node,
        state: dict,
        speed: float,
    ) -> dict:
        """Calculate physical quantities used by the safety shield."""

        initial_energy = max(
            float(state.get("E_initial", getattr(node, "E_max", 1.0))),
            1e-12,
        )
        estimated_energy = float(
            state.get(
                "E_est",
                getattr(node, "E_current", initial_energy),
            )
        )
        # Keep the routing state inside the physical battery interval. This
        # protects DQN and the shield from stale or corrupted BS estimates.
        estimated_energy = float(np.clip(
            estimated_energy, 0.0, initial_energy
        ))
        energy_ratio = estimated_energy / initial_energy

        depletion_rate = float(state.get("energy_q_ewma", 0.0))
        distance = math.hypot(
            float(node.x) - float(mc_x),
            float(node.y) - float(mc_y),
        )
        travel_time = distance / max(float(speed), 1e-6)

        if depletion_rate > 1e-12:
            time_to_depletion = estimated_energy / depletion_rate
        else:
            # No positive depletion estimate means no immediate predicted
            # deadline. A large finite value avoids infinities in reports.
            time_to_depletion = 1e6

        deadline_slack = time_to_depletion - travel_time

        return {
            "initial_energy": float(initial_energy),
            "estimated_energy": float(estimated_energy),
            "energy_ratio": float(energy_ratio),
            "energy_deficit_ratio": float(
                max(0.0, 1.0 - energy_ratio)
            ),
            "estimated_depletion_rate": float(depletion_rate),
            "distance_to_charger": float(distance),
            "estimated_travel_time": float(travel_time),
            "estimated_time_to_depletion": float(
                time_to_depletion
            ),
            "deadline_slack": float(deadline_slack),
            "critical_energy_ratio": float(
                np.clip(
                    float(getattr(node, "E_critical", 0.0)) / initial_energy
                    if float(getattr(node, "E_critical", 0.0)) > 0.0
                    else float(state.get("critical_energy_ratio", 0.0)),
                    0.0,
                    1.0,
                )
            ),
        }

    @staticmethod
    def feature_for(
        mc_x: float,
        mc_y: float,
        node,
        state: dict,
        network_diag: float = 141.421356,
        survival_horizon: float = 3600.0,
    ) -> np.ndarray:
        """Build the seven normalized inputs supplied to the Q-model."""

        initial_energy = max(
            float(state.get("E_initial", getattr(node, "E_max", 1.0))),
            1e-12,
        )
        estimated_energy = float(
            state.get(
                "E_est",
                getattr(node, "E_current", initial_energy),
            )
        )
        estimated_energy = float(np.clip(
            estimated_energy, 0.0, initial_energy
        ))
        energy_ratio = estimated_energy / initial_energy

        depletion_rate = float(state.get("energy_q_ewma", 0.0))
        distance = math.hypot(
            float(node.x) - float(mc_x),
            float(node.y) - float(mc_y),
        )

        if depletion_rate > 0.0:
            time_to_depletion = estimated_energy / max(
                depletion_rate,
                1e-12,
            )
        else:
            time_to_depletion = 1e6

        normalized_survival = min(
            1.0, time_to_depletion / max(float(survival_horizon), 1e-6)
        )

        return np.asarray(
            [
                (float(node.x) - float(mc_x))
                / max(network_diag, 1e-6),
                (float(node.y) - float(mc_y))
                / max(network_diag, 1e-6),
                energy_ratio,
                1.0 - energy_ratio,
                min(
                    1.0,
                    distance / max(network_diag, 1e-6),
                ),
                1.0 - normalized_survival,
                min(
                    1.0,
                    depletion_rate
                    / max(initial_energy, 1e-12)
                    * max(float(survival_horizon), 1e-6),
                ),
            ],
            dtype=np.float32,
        )

    def heuristic_score(
        self,
        mc_x: float,
        mc_y: float,
        node,
        state: dict,
        speed: float = 1.0,
        network_diag: float = 141.421356,
    ) -> float:
        """Dimensionless deadline/energy/proximity safety score.

        Each term is normalized to [0, 1] and receives equal weight by the
        principle of indifference. This removes the unexplained constants
        4.0, 1.5 and 0.015 while preserving the same preference directions.
        """
        initial_energy = max(
            float(state.get("E_initial", getattr(node, "E_max", 1.0))),
            1e-12,
        )
        estimated_energy = float(np.clip(
            float(state.get("E_est", getattr(node, "E_current", initial_energy))),
            0.0, initial_energy,
        ))
        energy_ratio = estimated_energy / initial_energy
        distance = math.hypot(
            float(node.x) - float(mc_x),
            float(node.y) - float(mc_y),
        )
        depletion_rate = max(0.0, float(state.get("energy_q_ewma", 0.0)))
        time_to_depletion = (
            estimated_energy / max(depletion_rate, 1e-12)
            if depletion_rate > 0.0 else 1e6
        )
        travel_time = distance / max(float(speed), 1e-6)
        slack = time_to_depletion - travel_time
        deadline_urgency = (
            1.0 if slack <= 0.0
            else 1.0 / (1.0 + slack / self.expected_mission_time)
        )
        energy_deficit = float(np.clip(1.0 - energy_ratio, 0.0, 1.0))
        proximity = 1.0 - float(np.clip(
            distance / max(float(network_diag), 1e-12), 0.0, 1.0
        ))
        return float((deadline_urgency + energy_deficit + proximity) / 3.0)

    @staticmethod
    def _rank_by_descending_score(values: np.ndarray) -> dict:
        """Return a mapping from candidate index to one-based rank."""

        sorted_indices = np.argsort(-np.asarray(values, dtype=float))
        return {
            int(candidate_index): int(rank)
            for rank, candidate_index in enumerate(
                sorted_indices,
                start=1,
            )
        }

    def _predict_q_values(
        self,
        features: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Evaluate the learned Q-model for all candidate features."""

        if self.enabled_torch:
            with torch.inference_mode():
                feature_tensor = torch.tensor(
                    np.vstack(features),
                    dtype=torch.float32,
                    device=self.device,
                )
                return (
                    self.net(feature_tensor)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(float)
                )

        return np.asarray(
            [
                float(np.dot(self.weights, feature) + self.bias)
                for feature in features
            ],
            dtype=float,
        )

    def _adaptive_exploration_probability(
        self, q_values: np.ndarray, allowed_indices: np.ndarray
    ) -> float:
        """Uncertainty/exposure-driven epsilon without a fixed magic value."""
        if self.fixed_epsilon is not None and self.exploration_mode not in {
            "adaptive", "uncertainty_adaptive", "uncertainty"
        }:
            return float(np.clip(self.fixed_epsilon, 0.0, 1.0))
        values = np.asarray(q_values[allowed_indices], dtype=float)
        k = len(values)
        if k <= 1:
            return 0.0
        scale = float(np.std(values))
        if scale <= 1e-12:
            probabilities = np.full(k, 1.0 / k, dtype=float)
        else:
            logits = (values - float(np.max(values))) / scale
            exp_logits = np.exp(np.clip(logits, -60.0, 0.0))
            probabilities = exp_logits / max(float(exp_logits.sum()), 1e-12)
        entropy = -float(np.sum(
            probabilities * np.log(np.maximum(probabilities, 1e-12))
        )) / math.log(k)
        exposure = math.sqrt(k / max(k + self.competitive_decisions, 1))
        return float(np.clip(entropy * exposure, 0.0, 1.0))

    def _deadline_feasibility_override(
        self,
        dqn_index: int,
        heuristic_index: int,
        physical_values: Sequence[dict],
    ) -> bool:
        """Override only when the heuristic target cannot safely wait one mission."""
        if dqn_index == heuristic_index:
            return False
        heuristic_slack = float(
            physical_values[heuristic_index]["deadline_slack"])
        dqn_travel = float(physical_values[dqn_index]["estimated_travel_time"])
        waiting_time = max(self.expected_mission_time, dqn_travel)
        return heuristic_slack <= waiting_time

    def select(
        self,
        mc_x: float,
        mc_y: float,
        candidates: Sequence,
        state_lookup: dict,
        speed: float = 1.0,
        network_diag: float = 141.421356,
        charger_id=None,
        decision_time: Optional[float] = None,
    ):
        """Select the next charging target and retain an explanation.

        Singleton decisions bypass DQN inference because no ranking is
        possible. Warm-up advances only when at least two candidates remain
        after the safety shield is applied.
        """

        candidate_list = list(candidates)

        if not candidate_list:
            self.last_decision_explanation = {
                "decision_number": int(self.decisions),
                "decision_time": (
                    None if decision_time is None else float(decision_time)
                ),
                "charger_id": charger_id,
                "selection_mode": "no_candidates",
                "selected_node_id": None,
                "candidate_count": 0,
                "effective_candidate_count": 0,
                "competitive_decisions": int(self.competitive_decisions),
                "candidates": [],
            }
            return None

        self.decisions += 1
        candidate_count = len(candidate_list)

        # Preserve one epsilon draw per non-empty routing call. This keeps the
        # global random sequence aligned with the previous implementation even
        # when singleton neural inference is skipped.
        # A random draw is still consumed in fast paths for reproducibility,
        # but competitive exploration is evaluated only after warm-up.
        exploration_draw = False

        if candidate_count == 1:
            selected_node = candidate_list[0]
            state = state_lookup.get(selected_node.node_id, {})
            physical = self._candidate_energy_and_deadline(
                mc_x, mc_y, selected_node, state, speed
            )
            feature = self.feature_for(
                mc_x, mc_y, selected_node, state, network_diag, self.safe_horizon
            )
            is_hard_rescue = bool(
                physical["energy_ratio"] <= max(
                    self.critical_energy_ratio,
                    physical.get("critical_energy_ratio", 0.0),
                )
                or physical["deadline_slack"] <= self.deadline_safety_margin
            )

            decision_explanation = {
                "decision_number": int(self.decisions),
                "decision_time": (
                    None if decision_time is None else float(decision_time)
                ),
                "charger_id": charger_id,
                "charger_position": {"x": float(mc_x), "y": float(mc_y)},
                "charger_speed": float(speed),
                "epsilon": float(self.epsilon),
                "exploration_draw": bool(exploration_draw),
                "selection_mode": "single_candidate",
                "selected_node_id": int(selected_node.node_id),
                "candidate_count": 1,
                "effective_candidate_count": 1,
                "dqn_proposed_node_id": None,
                "heuristic_best_node_id": int(selected_node.node_id),
                "greedy_best_node_id": int(selected_node.node_id),
                "legacy_combined_best_node_id": None,
                "safety_subset_active": bool(is_hard_rescue),
                "allowed_candidate_count": 1,
                "safety_override": False,
                "safety_override_reason": None,
                "training_updates": int(self._updates),
                "competitive_decisions": int(self.competitive_decisions),
                "warmup_updates": int(self.warmup_updates),
                "warmup_competitive_decisions": int(
                    self.warmup_competitive_decisions
                ),
                "critical_energy_ratio": float(self.critical_energy_ratio),
                "deadline_safety_margin": float(
                    self.deadline_safety_margin
                ),
                "heuristic_override_margin": (
                    None if self.heuristic_override_margin is None
                    else float(self.heuristic_override_margin)
                ),
                "candidates": [{
                    "node_id": int(selected_node.node_id),
                    **physical,
                    # No learned or heuristic ranking was needed.
                    "q_value": None,
                    "heuristic_score": None,
                    "combined_score": None,
                    "q_value_rank": None,
                    "heuristic_score_rank": None,
                    "combined_score_rank": None,
                    "is_hard_rescue": bool(is_hard_rescue),
                    "is_allowed_by_safety_shield": True,
                    "dqn_proposed": False,
                    "heuristic_proposed": False,
                    "selected": True,
                    "normalized_features": {
                        "relative_x": float(feature[0]),
                        "relative_y": float(feature[1]),
                        "energy_ratio": float(feature[2]),
                        "energy_deficit": float(feature[3]),
                        "distance": float(feature[4]),
                        "deadline_urgency": float(feature[5]),
                        "depletion_pressure": float(feature[6]),
                    },
                }],
            }
            self.last_decision_explanation = decision_explanation
            self.explanation_history.append(decision_explanation)
            return selected_node

        # Compute physical urgency first; this lets the shield reduce the
        # action set before any DQN inference is launched on CUDA.
        physical_values = [
            self._candidate_energy_and_deadline(
                mc_x,
                mc_y,
                node,
                state_lookup.get(node.node_id, {}),
                speed,
            )
            for node in candidate_list
        ]
        energy_ratios = np.asarray(
            [item["energy_ratio"] for item in physical_values], dtype=float
        )
        deadline_slacks = np.asarray(
            [item["deadline_slack"] for item in physical_values], dtype=float
        )
        node_critical_ratios = np.asarray([
            max(self.critical_energy_ratio, value.get(
                "critical_energy_ratio", 0.0))
            for value in physical_values
        ], dtype=float)
        hard_rescue_mask = (
            energy_ratios <= node_critical_ratios
        ) | (
            deadline_slacks <= self.deadline_safety_margin
        )

        if np.any(hard_rescue_mask):
            allowed_indices = np.flatnonzero(hard_rescue_mask)
            safety_subset_active = True
        else:
            allowed_indices = np.arange(candidate_count, dtype=int)
            safety_subset_active = False

        effective_candidate_count = int(len(allowed_indices))

        if effective_candidate_count == 1:
            selected_index = int(allowed_indices[0])
            selected_node = candidate_list[selected_index]
            allowed_set = {selected_index}
            candidate_explanations = []

            for index, node in enumerate(candidate_list):
                feature = self.feature_for(
                    mc_x,
                    mc_y,
                    node,
                    state_lookup.get(node.node_id, {}),
                    network_diag,
                    self.safe_horizon,
                )
                candidate_explanations.append({
                    "node_id": int(node.node_id),
                    **physical_values[index],
                    "q_value": None,
                    "heuristic_score": None,
                    "combined_score": None,
                    "q_value_rank": None,
                    "heuristic_score_rank": None,
                    "combined_score_rank": None,
                    "is_hard_rescue": bool(hard_rescue_mask[index]),
                    "is_allowed_by_safety_shield": bool(index in allowed_set),
                    "dqn_proposed": False,
                    "heuristic_proposed": False,
                    "selected": bool(index == selected_index),
                    "normalized_features": {
                        "relative_x": float(feature[0]),
                        "relative_y": float(feature[1]),
                        "energy_ratio": float(feature[2]),
                        "energy_deficit": float(feature[3]),
                        "distance": float(feature[4]),
                        "deadline_urgency": float(feature[5]),
                        "depletion_pressure": float(feature[6]),
                    },
                })

            decision_explanation = {
                "decision_number": int(self.decisions),
                "decision_time": (
                    None if decision_time is None else float(decision_time)
                ),
                "charger_id": charger_id,
                "charger_position": {"x": float(mc_x), "y": float(mc_y)},
                "charger_speed": float(speed),
                "epsilon": float(self.epsilon),
                "exploration_draw": bool(exploration_draw),
                "selection_mode": "safety_forced_single_candidate",
                "selected_node_id": int(selected_node.node_id),
                "candidate_count": int(candidate_count),
                "effective_candidate_count": 1,
                "dqn_proposed_node_id": None,
                "heuristic_best_node_id": int(selected_node.node_id),
                "greedy_best_node_id": int(selected_node.node_id),
                "legacy_combined_best_node_id": None,
                "safety_subset_active": bool(safety_subset_active),
                "allowed_candidate_count": 1,
                "safety_override": True,
                "safety_override_reason": "only_one_safe_candidate",
                "training_updates": int(self._updates),
                "competitive_decisions": int(self.competitive_decisions),
                "warmup_updates": int(self.warmup_updates),
                "warmup_competitive_decisions": int(
                    self.warmup_competitive_decisions
                ),
                "critical_energy_ratio": float(self.critical_energy_ratio),
                "deadline_safety_margin": float(
                    self.deadline_safety_margin
                ),
                "heuristic_override_margin": (
                    None if self.heuristic_override_margin is None
                    else float(self.heuristic_override_margin)
                ),
                "candidates": candidate_explanations,
            }
            self.last_decision_explanation = decision_explanation
            self.explanation_history.append(decision_explanation)
            return selected_node

        # At least two safe alternatives exist, so this is a genuine routing
        # decision and advances the competitive warm-up counter.
        self.competitive_decisions += 1

        features = [
            self.feature_for(
                mc_x,
                mc_y,
                node,
                state_lookup.get(node.node_id, {}),
                network_diag,
                self.safe_horizon,
            )
            for node in candidate_list
        ]
        heuristic_values = np.asarray([
            self.heuristic_score(
                mc_x,
                mc_y,
                node,
                state_lookup.get(node.node_id, {}),
                speed,
                network_diag,
            )
            for node in candidate_list
        ], dtype=float)
        q_values = self._predict_q_values(features)
        combined_scores = q_values + heuristic_values

        dqn_selected_index = int(
            allowed_indices[int(np.argmax(q_values[allowed_indices]))]
        )
        heuristic_selected_index = int(
            allowed_indices[
                int(np.argmax(heuristic_values[allowed_indices]))
            ]
        )

        safety_override = False
        safety_override_reason = None

        in_competitive_warmup = (
            self.competitive_decisions <= self.warmup_competitive_decisions
        )
        self.epsilon = self._adaptive_exploration_probability(
            q_values, allowed_indices
        )
        exploration_draw = (
            (not in_competitive_warmup)
            and random.random() < self.epsilon
        )

        if in_competitive_warmup:
            selected_index = heuristic_selected_index
            selection_mode = "heuristic_competitive_warmup"
            if selected_index != dqn_selected_index:
                safety_override = True
                safety_override_reason = "competitive_training_warmup"
        elif exploration_draw:
            selected_index = int(random.choice(allowed_indices.tolist()))
            selection_mode = "safe_epsilon_exploration"
        else:
            use_feasibility_override = self.safety_override_mode in {
                "adaptive", "deadline", "deadline_feasibility", "feasibility"
            }
            if use_feasibility_override:
                should_override = self._deadline_feasibility_override(
                    dqn_selected_index, heuristic_selected_index, physical_values
                )
                override_reason = "deadline_feasibility"
            else:
                heuristic_gap = float(
                    heuristic_values[heuristic_selected_index]
                    - heuristic_values[dqn_selected_index]
                )
                should_override = (
                    self.heuristic_override_margin is not None
                    and heuristic_gap > self.heuristic_override_margin
                )
                override_reason = "heuristic_gap_exceeded"

            if should_override:
                selected_index = heuristic_selected_index
                selection_mode = "heuristic_safety_override"
                safety_override = True
                safety_override_reason = override_reason
            else:
                selected_index = dqn_selected_index
                selection_mode = "dqn_with_safety_shield"

        selected_node = candidate_list[selected_index]
        q_rank = self._rank_by_descending_score(q_values)
        heuristic_rank = self._rank_by_descending_score(heuristic_values)
        combined_rank = self._rank_by_descending_score(combined_scores)
        allowed_set = {int(i) for i in allowed_indices.tolist()}

        candidate_explanations = []
        for index, node in enumerate(candidate_list):
            feature = features[index]
            candidate_explanations.append({
                "node_id": int(node.node_id),
                **physical_values[index],
                "q_value": float(q_values[index]),
                "heuristic_score": float(heuristic_values[index]),
                "combined_score": float(combined_scores[index]),
                "q_value_rank": int(q_rank[index]),
                "heuristic_score_rank": int(heuristic_rank[index]),
                "combined_score_rank": int(combined_rank[index]),
                "is_hard_rescue": bool(hard_rescue_mask[index]),
                "is_allowed_by_safety_shield": bool(index in allowed_set),
                "dqn_proposed": bool(index == dqn_selected_index),
                "heuristic_proposed": bool(index == heuristic_selected_index),
                "selected": bool(index == selected_index),
                "normalized_features": {
                    "relative_x": float(feature[0]),
                    "relative_y": float(feature[1]),
                    "energy_ratio": float(feature[2]),
                    "energy_deficit": float(feature[3]),
                    "distance": float(feature[4]),
                    "deadline_urgency": float(feature[5]),
                    "depletion_pressure": float(feature[6]),
                },
            })

        legacy_combined_best_index = int(np.argmax(combined_scores))
        decision_explanation = {
            "decision_number": int(self.decisions),
            "decision_time": (
                None if decision_time is None else float(decision_time)
            ),
            "charger_id": charger_id,
            "charger_position": {"x": float(mc_x), "y": float(mc_y)},
            "charger_speed": float(speed),
            "epsilon": float(self.epsilon),
            "exploration_draw": bool(exploration_draw),
            "selection_mode": selection_mode,
            "selected_node_id": int(selected_node.node_id),
            "candidate_count": int(candidate_count),
            "effective_candidate_count": int(effective_candidate_count),
            "dqn_proposed_node_id": int(
                candidate_list[dqn_selected_index].node_id
            ),
            "heuristic_best_node_id": int(
                candidate_list[heuristic_selected_index].node_id
            ),
            "greedy_best_node_id": int(
                candidate_list[heuristic_selected_index].node_id
            ),
            "legacy_combined_best_node_id": int(
                candidate_list[legacy_combined_best_index].node_id
            ),
            "safety_subset_active": bool(safety_subset_active),
            "allowed_candidate_count": int(len(allowed_indices)),
            "safety_override": bool(safety_override),
            "safety_override_reason": safety_override_reason,
            "training_updates": int(self._updates),
            "competitive_decisions": int(self.competitive_decisions),
            "warmup_updates": int(self.warmup_updates),
            "warmup_competitive_decisions": int(
                self.warmup_competitive_decisions
            ),
            "critical_energy_ratio": float(self.critical_energy_ratio),
            "deadline_safety_margin": float(self.deadline_safety_margin),
            "heuristic_override_margin": (
                None if self.heuristic_override_margin is None
                else float(self.heuristic_override_margin)
            ),
            "candidates": candidate_explanations,
        }
        self.last_decision_explanation = decision_explanation
        self.explanation_history.append(decision_explanation)
        return selected_node

    def train_on_transition(
        self,
        feature: np.ndarray,
        reward: float,
        next_features: Optional[Sequence[np.ndarray]] = None,
    ) -> float:
        """Train the Q-model from one completed charging mission."""

        feature = np.asarray(feature, dtype=np.float32)
        next_feature_list = (
            []
            if next_features is None
            else [
                np.asarray(item, dtype=np.float32)
                for item in next_features
            ]
        )

        self.replay.append(
            (feature, float(reward), next_feature_list)
        )

        if not self.enabled_torch:
            target_value = float(reward)

            if next_feature_list:
                target_value += self.gamma * max(
                    float(
                        np.dot(self.weights, next_feature)
                        + self.bias
                    )
                    for next_feature in next_feature_list
                )

            prediction = float(
                np.dot(self.weights, feature) + self.bias
            )
            error = target_value - prediction

            self.weights += self.lr * error * feature
            self.bias += self.lr * error
            self._updates += 1

            self.last_loss = float(error * error)
            return self.last_loss

        batch = random.sample(
            list(self.replay),
            k=min(len(self.replay), self.batch_size),
        )

        feature_tensor = torch.tensor(
            np.vstack([item[0] for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )

        target_values = []

        for _, transition_reward, following_features in batch:
            target_value = float(transition_reward)

            if following_features:
                with torch.inference_mode():
                    next_tensor = torch.tensor(
                        np.vstack(following_features),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    target_value += self.gamma * float(
                        torch.max(self.target(next_tensor))
                        .detach()
                        .cpu()
                        .item()
                    )

            target_values.append(target_value)

        target_tensor = torch.tensor(
            target_values,
            dtype=torch.float32,
            device=self.device,
        )

        predictions = self.net(feature_tensor)
        loss = torch.nn.functional.smooth_l1_loss(
            predictions,
            target_tensor,
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.net.parameters(),
            1.0,
        )
        self.optimizer.step()

        self._updates += 1

        if self._updates % self.target_sync_interval == 0:
            self.target.load_state_dict(self.net.state_dict())

        self.last_loss = float(loss.detach().cpu().item())
        return self.last_loss
