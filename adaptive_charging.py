from __future__ import annotations

import math
import os
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
    """Resolve a safe PyTorch device with graceful CUDA fallback."""
    if torch is None or nn is None or optim is None:
        return None
    requested = "auto" if device is None else str(device).strip().lower()
    if requested in {"", "auto", "adaptive", "derived", "none"}:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)


def configure_torch_runtime(
    seed: int,
    *,
    deterministic: bool = True,
    strict_determinism: bool = False,
    allow_tf32: bool = False,
) -> None:
    """Configure a reproducible PyTorch runtime for research experiments.

    Determinism is guaranteed only for the same software stack and compatible
    hardware.  TF32 is disabled by default because it can alter numerical
    results on recent NVIDIA GPUs.
    """
    if torch is None:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        torch.use_deterministic_algorithms(
            bool(deterministic), warn_only=not bool(strict_determinism))
    except TypeError:
        torch.use_deterministic_algorithms(bool(deterministic))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    except Exception:
        pass


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
    """Derive charging/DQN hyperparameters from physical and scale variables."""
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

    service_radius = math.sqrt(area / (math.pi * m))
    expected_travel_distance = (2.0 / 3.0) * service_radius
    expected_travel_time = expected_travel_distance / speed

    density = n / area
    expected_radio_neighbors = max(1.0, density * math.pi * radio_range**2)
    nodes_per_charger = max(1.0, n / m)

    local_scale = max(1.0, min(nodes_per_charger, expected_radio_neighbors))
    candidate_budget = max(2, int(math.ceil(math.sqrt(local_scale))))

    full_charge_time = max(float(sensor_initial_energy), 0.0) / max(
        float(charging_rate), 1e-12
    )
    expected_mission_time = max(
        epoch, expected_travel_time + full_charge_time + epoch
    )
    expected_decisions = max(
        1, int(math.ceil(float(sim_time) / expected_mission_time)))

    service_cycle_time = max(
        epoch, int(math.ceil(nodes_per_charger)) * expected_mission_time
    )

    hidden_dim = _next_power_of_two(
        max(feature_dim + 1, feature_dim * candidate_budget))
    parameter_count = (feature_dim * hidden_dim + hidden_dim) + (
        hidden_dim * hidden_dim + hidden_dim
    ) + (hidden_dim + 1)

    replay_capacity = max(candidate_budget**2, expected_decisions)
    batch_size = max(2, min(replay_capacity, int(
        math.ceil(math.sqrt(replay_capacity)))))
    target_sync_interval = max(1, int(math.ceil(math.sqrt(batch_size))))

    warmup_competitive = max(1, int(math.ceil(
        candidate_budget * math.log(candidate_budget + 1.0)
    )))

    gamma = math.exp(-expected_mission_time / horizon)
    dqn_lr = 1.0 / math.sqrt(max(1.0, parameter_count * expected_decisions))

    qifl_params = 9 * 5
    qifl_updates_per_node = max(1.0, float(sim_time) / epoch)
    qifl_lr = 1.0 / math.sqrt(qifl_params * qifl_updates_per_node)
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
        "dqn_epsilon": None,
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
        # Hyperparams for Dynamic/Partial Charging
        "dynamic_charging_weight_wt": 1.0,
        "dynamic_charging_weight_eps": 1.0,
        "dynamic_charging_weight_now": 1.0,
    }


class QuantumFuzzyChargingOracle:
    """Adaptive probabilistic fuzzy oracle for calculating a charging threshold."""

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
                torch.tensor(logits, dtype=torch.float32, device=self.device)
            )
            if _is_adaptive_value(lr):
                lr = 1.0 / math.sqrt(max(1.0, self.theta.numel()))
            self.optimizer = optim.Adam([self.theta], lr=float(lr))
            self.centers_t = torch.tensor(
                self.centers, dtype=torch.float32, device=self.device
            )
        else:
            self.device = None
            if _is_adaptive_value(lr):
                lr = 1.0 / math.sqrt(max(1.0, 9 * len(self.centers)))
            self.lr = float(lr)
            self.theta_np = self._initial_rule_logits()

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
                [0.02, 0.08, 0.20, 0.38, 0.32],
                [0.01, 0.05, 0.18, 0.38, 0.38],
                [0.01, 0.03, 0.14, 0.34, 0.48],
                [0.08, 0.22, 0.38, 0.24, 0.08],
                [0.05, 0.18, 0.36, 0.30, 0.11],
                [0.03, 0.12, 0.33, 0.34, 0.18],
                [0.45, 0.32, 0.16, 0.05, 0.02],
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

    def infer(self, ed_norm: float, sd_norm: float) -> Tuple[float, np.ndarray]:
        psi = self._input_state_np(ed_norm, sd_norm)
        if self.enabled_torch:
            with torch.inference_mode():
                psi_t = torch.tensor(
                    psi, dtype=torch.float32, device=self.device)
                rule_probabilities = torch.softmax(self.theta, dim=1)
                probabilities = torch.matmul(psi_t, rule_probabilities)
                probabilities = probabilities / \
                    torch.clamp(probabilities.sum(), min=1e-12)
                threshold = torch.sum(probabilities * self.centers_t).item()
                return float(threshold), probabilities.detach().cpu().numpy()

        logits = self.theta_np - np.max(self.theta_np, axis=1, keepdims=True)
        rule_probabilities = np.exp(np.clip(logits, -60.0, 0.0))
        rule_probabilities /= np.maximum(
            rule_probabilities.sum(axis=1, keepdims=True), 1e-12)
        probabilities = psi @ rule_probabilities
        probabilities = probabilities / \
            max(float(np.sum(probabilities)), 1e-12)
        return float(np.sum(probabilities * self.centers)), probabilities

    def infer_batch(self, inputs: Sequence[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized classical batch inference for offline/audit workloads.

        This method does not replace the scalar control-loop inference, so the
        published decision sequence remains unchanged. It exists to measure
        the computational benefit of vectorized fuzzy-rule evaluation without
        describing the implementation as quantum hardware execution.
        """
        pairs = list(inputs)
        if not pairs:
            return (np.empty(0, dtype=float),
                    np.empty((0, len(self.centers)), dtype=float))
        psi = np.vstack([
            self._input_state_np(float(ed), float(sd))
            for ed, sd in pairs
        ])
        if self.enabled_torch:
            with torch.inference_mode():
                psi_t = torch.tensor(
                    psi, dtype=torch.float32, device=self.device)
                rule_probabilities = torch.softmax(self.theta, dim=1)
                probabilities = torch.matmul(psi_t, rule_probabilities)
                probabilities = probabilities / torch.clamp(
                    probabilities.sum(dim=1, keepdim=True), min=1e-12)
                thresholds = torch.matmul(probabilities, self.centers_t)
                return (
                    thresholds.detach().cpu().numpy().astype(float),
                    probabilities.detach().cpu().numpy().astype(float),
                )

        logits = self.theta_np - np.max(self.theta_np, axis=1, keepdims=True)
        rule_probabilities = np.exp(np.clip(logits, -60.0, 0.0))
        rule_probabilities /= np.maximum(
            rule_probabilities.sum(axis=1, keepdims=True), 1e-12)
        probabilities = psi @ rule_probabilities
        probabilities /= np.maximum(
            probabilities.sum(axis=1, keepdims=True), 1e-12)
        thresholds = probabilities @ self.centers
        return thresholds.astype(float), probabilities.astype(float)

    def train_step(self, ed_norm: float, sd_norm: float, target_threshold: float) -> float:
        target_threshold = float(target_threshold)
        if not self.enabled_torch:
            psi = self._input_state_np(ed_norm, sd_norm)
            logits = self.theta_np - \
                np.max(self.theta_np, axis=1, keepdims=True)
            rule_probabilities = np.exp(np.clip(logits, -60.0, 0.0))
            rule_probabilities /= np.maximum(
                rule_probabilities.sum(axis=1, keepdims=True), 1e-12)
            row_predictions = rule_probabilities @ self.centers
            prediction = float(np.dot(psi, row_predictions))
            error = prediction - target_threshold

            # Exact gradient of squared error through each row-wise softmax.
            gradient = np.zeros_like(self.theta_np)
            for row in range(self.theta_np.shape[0]):
                p_row = rule_probabilities[row]
                centered = self.centers - row_predictions[row]
                gradient[row] = 2.0 * error * psi[row] * p_row * centered
            gradient = np.clip(gradient, -1.0, 1.0)
            self.theta_np -= self.lr * gradient

            self.last_loss = float(error * error)
            return self.last_loss

        psi = torch.tensor(self._input_state_np(
            ed_norm, sd_norm), dtype=torch.float32, device=self.device)
        target = torch.tensor(
            target_threshold, dtype=torch.float32, device=self.device)

        rule_probabilities = torch.softmax(self.theta, dim=1)
        probabilities = torch.matmul(psi, rule_probabilities)
        probabilities = probabilities / \
            torch.clamp(probabilities.sum(), min=1e-12)

        prediction = torch.sum(probabilities * self.centers_t)
        entropy = -torch.sum(probabilities *
                             torch.log(torch.clamp(probabilities, min=1e-8)))

        loss = torch.nn.functional.smooth_l1_loss(
            prediction, target) + 1e-4 * entropy

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([self.theta], 1.0)
        self.optimizer.step()

        self.last_loss = float(loss.detach().cpu().item())
        return self.last_loss


class DynamicChargingDurationOracle:
    """Mamdani-style FLS for dynamic partial charging.

    Inputs are normalized waiting time (WT), energy consumption rate (EPS),
    and number of waiting nodes (NOW).  WT and EPS increase the allocation,
    while NOW decreases the per-node fraction so the charger can reach more
    requests.  This replaces the previous linear average with an actual
    3-input/27-rule fuzzy inference system.
    """

    OUTPUT_CENTERS = np.asarray((0.08, 0.29, 0.50, 0.71, 0.92), dtype=float)

    def __init__(self, w_wt: float = 1.0, w_eps: float = 1.0, w_now: float = 1.0):
        self.w_wt = max(float(w_wt), 1e-6)
        self.w_eps = max(float(w_eps), 1e-6)
        self.w_now = max(float(w_now), 1e-6)

    @staticmethod
    def _memberships(value: float) -> np.ndarray:
        x = float(np.clip(value, 0.0, 1.0))
        low = max(0.0, min(1.0, (0.5 - x) / 0.5))
        medium = max(0.0, 1.0 - abs(x - 0.5) / 0.5)
        high = max(0.0, min(1.0, (x - 0.5) / 0.5))
        memberships = np.asarray((low, medium, high), dtype=float)
        total = float(memberships.sum())
        return memberships / max(total, 1e-12)

    def calculate_weight(self, wt_norm: float, eps_norm: float, now_norm: float) -> float:
        wt_m = self._memberships(wt_norm)
        eps_m = self._memberships(eps_norm)
        now_m = self._memberships(now_norm)

        numerator = 0.0
        denominator = 0.0
        total_input_weight = self.w_wt + self.w_eps + self.w_now
        for wt_level in range(3):
            for eps_level in range(3):
                for now_level in range(3):
                    activation = min(
                        wt_m[wt_level], eps_m[eps_level], now_m[now_level])
                    if activation <= 0.0:
                        continue

                    # Rule consequent: high WT/EPS raise the output; high NOW
                    # lowers it.  The weighted score is quantized to five fuzzy
                    # output sets, then center-of-sets defuzzification is used.
                    score = (
                        self.w_wt * wt_level
                        + self.w_eps * eps_level
                        + self.w_now * (2 - now_level)
                    ) / max(2.0 * total_input_weight, 1e-12)
                    output_index = int(np.clip(round(score * 4.0), 0, 4))
                    numerator += activation * self.OUTPUT_CENTERS[output_index]
                    denominator += activation

        if denominator <= 1e-12:
            return 0.50
        return float(np.clip(numerator / denominator,
                             self.OUTPUT_CENTERS[0],
                             self.OUTPUT_CENTERS[-1]))

    def allocation_fraction(self, wt_norm: float, eps_norm: float, now_norm: float) -> float:
        return self.calculate_weight(wt_norm, eps_norm, now_norm)


if nn is not None:
    class _QNet(nn.Module):
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
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PyTorch is required to instantiate _QNet.")


class ChargerRoutingDQN:
    """Safety-shielded DQN for selecting the next charging target (Nodes or Clusters)."""

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
        warmup_readiness_mode: str = "competitive_only",
        warmup_min_replay_samples: Optional[int] = None,
        warmup_td_abs_threshold: Optional[float] = None,
        warmup_td_window: int = 20,
        warmup_max_total_decisions: Optional[int] = None,
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
            1e-6, float(expected_mission_time or deadline_safety_margin or 1.0))
        self.safe_horizon = max(
            self.expected_mission_time, float(safe_horizon or 3600.0))
        self.gamma = math.exp(-self.expected_mission_time /
                              self.safe_horizon) if _is_adaptive_value(gamma) else float(gamma)
        self.fixed_epsilon = None if _is_adaptive_value(
            epsilon) else float(epsilon)
        self.epsilon = 0.0 if self.fixed_epsilon is None else self.fixed_epsilon
        self.exploration_mode = str(
            exploration_mode or "fixed").strip().lower()
        self.safety_override_mode = str(
            safety_override_mode or "deadline_feasibility").strip().lower()

        self.warmup_updates = max(0, int(warmup_updates))
        if warmup_competitive_decisions is None:
            warmup_competitive_decisions = self.warmup_updates
        self.warmup_competitive_decisions = max(
            0, int(warmup_competitive_decisions))
        self.warmup_readiness_mode = str(
            warmup_readiness_mode or "competitive_only").strip().lower()
        self.warmup_min_replay_samples = (
            None if warmup_min_replay_samples is None
            else max(1, int(warmup_min_replay_samples)))
        self.warmup_td_abs_threshold = (
            None if _is_adaptive_value(warmup_td_abs_threshold)
            else max(0.0, float(warmup_td_abs_threshold)))
        self.warmup_td_window = max(1, int(warmup_td_window))
        self.warmup_max_total_decisions = (
            None if warmup_max_total_decisions is None
            else max(1, int(warmup_max_total_decisions)))
        self.competitive_decisions = 0
        self.critical_energy_ratio = float(critical_energy_ratio)
        self.deadline_safety_margin = max(0.0, float(deadline_safety_margin))
        self.heuristic_override_margin = None if _is_adaptive_value(
            heuristic_override_margin) else float(heuristic_override_margin)

        self.hidden_dim = max(2, int(hidden_dim))
        self.batch_size = max(1, int(batch_size))
        if self.warmup_min_replay_samples is None:
            self.warmup_min_replay_samples = self.batch_size
        self.target_sync_interval = max(1, int(target_sync_interval))
        self.decisions = 0
        self.last_loss = 0.0
        self.last_td_abs_mean = 0.0
        self._updates = 0
        self.replay = deque(maxlen=max(self.batch_size, int(replay_capacity)))

        # Observer-only audit state. These counters never participate in
        # selection, reward calculation, exploration, or training updates.
        self.last_decision_explanation = None
        self.explanation_history = deque(
            maxlen=max(1, int(explanation_history_size)))
        self.training_history = deque(
            maxlen=max(1000, int(explanation_history_size)))
        self.decision_mode_counts = {}
        self.safety_override_reason_counts = {}
        self.dqn_raw_proposal_count = 0
        self.dqn_direct_execution_count = 0
        self.heuristic_warmup_execution_count = 0
        self.heuristic_safety_execution_count = 0
        self.safety_forced_single_count = 0
        self.state_coverage_bins = {
            "energy_ratio": set(),
            "distance": set(),
            "deadline_urgency": set(),
            "depletion_pressure": set(),
        }

        self.device = _resolve_torch_device(device)
        self.enabled_torch = self.device is not None

        # Keep the DQN random stream isolated from the simulator workload RNG.
        self._rng = random.Random(int(seed))
        self._np_rng = np.random.default_rng(int(seed))

        if self.enabled_torch:
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

            self.net = _QNet(hidden_dim=self.hidden_dim).to(self.device)
            self.target = _QNet(hidden_dim=self.hidden_dim).to(self.device)
            self.target.load_state_dict(self.net.state_dict())
            if _is_adaptive_value(lr):
                parameter_count = sum(p.numel() for p in self.net.parameters())
                lr = 1.0 / \
                    math.sqrt(max(1.0, parameter_count *
                              max(1, self.replay.maxlen)))
            self.lr = float(lr)
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        else:
            self.device = None
            self.weights = np.zeros(7, dtype=np.float32)
            self.bias = 0.0
            self.lr = float(1e-3 if _is_adaptive_value(lr) else lr)

    @staticmethod
    def _coverage_bin(value: float, bins: int = 10) -> int:
        value = float(np.clip(value, 0.0, 1.0))
        return min(max(int(value * bins), 0), bins - 1)

    def _commit_decision(self, explanation: dict, selected_candidate):
        """Store audit evidence without changing the selected action."""
        self.last_decision_explanation = explanation
        self.explanation_history.append(explanation)
        mode = str(explanation.get("selection_mode", "unknown"))
        self.decision_mode_counts[mode] = self.decision_mode_counts.get(mode, 0) + 1
        reason = explanation.get("safety_override_reason")
        if reason:
            reason = str(reason)
            self.safety_override_reason_counts[reason] = (
                self.safety_override_reason_counts.get(reason, 0) + 1)
        if explanation.get("dqn_proposed_node_id") is not None:
            self.dqn_raw_proposal_count += 1
        if mode == "dqn_with_safety_shield":
            self.dqn_direct_execution_count += 1
        elif mode == "heuristic_competitive_warmup":
            self.heuristic_warmup_execution_count += 1
        elif mode == "heuristic_safety_override":
            self.heuristic_safety_execution_count += 1
        elif mode == "safety_forced_single_candidate":
            self.safety_forced_single_count += 1

        for candidate in explanation.get("candidates", []) or []:
            features = candidate.get("normalized_features", {}) or {}
            for name in self.state_coverage_bins:
                value = features.get(name)
                if value is not None:
                    self.state_coverage_bins[name].add(self._coverage_bin(value))
        return selected_candidate

    def _append_training_audit(self, reward: float, loss: float, td_abs_mean: float) -> None:
        self.last_td_abs_mean = float(max(0.0, td_abs_mean))
        self.training_history.append({
            "update": int(self._updates),
            "replay_size": int(len(self.replay)),
            "reward": float(reward),
            "loss": float(loss),
            "td_abs_mean": float(self.last_td_abs_mean),
        })

    def audit_snapshot(self) -> dict:
        """Return a JSON-serializable observer-only learning audit."""
        competitive = max(0, int(self.competitive_decisions))
        return {
            "decisions": int(self.decisions),
            "competitive_decisions": competitive,
            "warmup_readiness_mode": self.warmup_readiness_mode,
            "warmup_active": bool(self._warmup_active()),
            "warmup_updates_required": int(self.warmup_updates),
            "warmup_min_replay_samples": int(self.warmup_min_replay_samples),
            "warmup_td_abs_threshold": self.warmup_td_abs_threshold,
            "warmup_max_total_decisions": self.warmup_max_total_decisions,
            "training_updates": int(self._updates),
            "replay_size": int(len(self.replay)),
            "replay_capacity": int(self.replay.maxlen),
            "last_loss": float(self.last_loss),
            "last_td_abs_mean": float(self.last_td_abs_mean),
            "dqn_raw_proposal_count": int(self.dqn_raw_proposal_count),
            "dqn_direct_execution_count": int(self.dqn_direct_execution_count),
            "heuristic_warmup_execution_count": int(self.heuristic_warmup_execution_count),
            "heuristic_safety_execution_count": int(self.heuristic_safety_execution_count),
            "safety_forced_single_count": int(self.safety_forced_single_count),
            "direct_execution_rate": (
                float(self.dqn_direct_execution_count) / competitive
                if competitive > 0 else 0.0),
            "decision_mode_counts": dict(self.decision_mode_counts),
            "safety_override_reason_counts": dict(self.safety_override_reason_counts),
            "state_coverage_fraction": {
                name: len(values) / 10.0
                for name, values in self.state_coverage_bins.items()
            },
        }

    def save_checkpoint(self, path: str) -> str:
        """Save a DQN checkpoint for evaluation on unseen seeds/topologies."""
        path = os.path.abspath(os.fspath(path))
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if self.enabled_torch:
            torch.save({
                "backend": "torch",
                "net": self.net.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "updates": int(self._updates),
                "competitive_decisions": int(self.competitive_decisions),
            }, path)
        else:
            with open(path, "wb") as handle:
                np.savez(handle, backend="numpy", weights=self.weights,
                         bias=np.asarray([self.bias], dtype=float),
                         updates=np.asarray([self._updates], dtype=int),
                         competitive_decisions=np.asarray(
                             [self.competitive_decisions], dtype=int))
        return path

    def load_checkpoint(self, path: str, *, load_optimizer: bool = False) -> None:
        """Load a checkpoint without enabling training automatically."""
        path = os.path.abspath(os.fspath(path))
        if self.enabled_torch:
            payload = torch.load(path, map_location=self.device)
            self.net.load_state_dict(payload["net"])
            self.target.load_state_dict(payload.get("target", payload["net"]))
            if load_optimizer and payload.get("optimizer") is not None:
                self.optimizer.load_state_dict(payload["optimizer"])
            self._updates = int(payload.get("updates", 0))
            self.competitive_decisions = int(
                payload.get("competitive_decisions", 0))
        else:
            with np.load(path, allow_pickle=False) as payload:
                self.weights = np.asarray(payload["weights"], dtype=np.float32)
                self.bias = float(np.asarray(payload["bias"]).reshape(-1)[0])
                self._updates = int(np.asarray(
                    payload.get("updates", [0])).reshape(-1)[0])
                self.competitive_decisions = int(np.asarray(
                    payload.get("competitive_decisions", [0])).reshape(-1)[0])

    @staticmethod
    def _extract_physical_attributes(candidate, state_lookup: dict) -> dict:
        """
        Duck-typing extraction method to natively support both single Nodes and Clusters.
        Aggregates parameters if candidate is a Cluster.
        """
        if hasattr(candidate, 'is_cluster') and candidate.is_cluster:
            x = float(candidate.centroid_x)
            y = float(candidate.centroid_y)
            cand_id = getattr(candidate, 'cluster_id', id(candidate))

            member_ratios = []
            depletion_ratios = []
            critical_ratios = []
            representative_energy = 1.0

            for node in getattr(candidate, 'nodes', []):
                node_id = getattr(node, 'node_id', id(node))
                node_state = state_lookup.get(node_id, {})
                e_max = max(float(node_state.get(
                    "E_initial", getattr(node, "E_max", 1.0))), 1e-12)
                e_curr = float(np.clip(float(node_state.get(
                    "E_est", getattr(node, "E_current", e_max))), 0.0, e_max))
                q = max(0.0, float(node_state.get("energy_q_ewma", 0.0)))
                member_ratios.append(e_curr / e_max)
                depletion_ratios.append(q / e_max)
                critical_ratios.append(float(getattr(
                    node, "E_critical", 0.0)) / e_max)
                representative_energy = max(representative_energy, e_max)

            minimum_ratio = min(member_ratios) if member_ratios else 1.0
            maximum_depletion_ratio = max(
                depletion_ratios) if depletion_ratios else 0.0
            initial_energy = representative_energy
            estimated_energy = minimum_ratio * initial_energy
            depletion_rate = maximum_depletion_ratio * initial_energy
            critical_energy_ratio = max(
                critical_ratios) if critical_ratios else 0.0

        else:
            x = float(candidate.x)
            y = float(candidate.y)
            cand_id = getattr(candidate, 'node_id', id(candidate))

            state = state_lookup.get(cand_id, {})
            initial_energy = max(
                float(state.get("E_initial", getattr(candidate, "E_max", 1.0))), 1e-12)
            estimated_energy = float(np.clip(float(state.get("E_est", getattr(
                candidate, "E_current", initial_energy))), 0.0, initial_energy))
            depletion_rate = float(state.get("energy_q_ewma", 0.0))
            critical_energy_ratio = float(
                getattr(candidate, "E_critical", 0.0)) / initial_energy

        return {
            "id": cand_id,
            "x": x,
            "y": y,
            "initial_energy": initial_energy,
            "estimated_energy": estimated_energy,
            "depletion_rate": depletion_rate,
            "critical_energy_ratio": critical_energy_ratio
        }

    @staticmethod
    def _candidate_energy_and_deadline(
        mc_x: float,
        mc_y: float,
        candidate,
        state_lookup: dict,
        speed: float,
    ) -> dict:
        attrs = ChargerRoutingDQN._extract_physical_attributes(
            candidate, state_lookup)

        energy_ratio = attrs["estimated_energy"] / attrs["initial_energy"]
        distance = math.hypot(
            attrs["x"] - float(mc_x), attrs["y"] - float(mc_y))
        travel_time = distance / max(float(speed), 1e-6)

        if attrs["depletion_rate"] > 1e-12:
            time_to_depletion = attrs["estimated_energy"] / \
                attrs["depletion_rate"]
        else:
            time_to_depletion = 1e6

        deadline_slack = time_to_depletion - travel_time

        return {
            "initial_energy": attrs["initial_energy"],
            "estimated_energy": attrs["estimated_energy"],
            "energy_ratio": float(energy_ratio),
            "energy_deficit_ratio": float(max(0.0, 1.0 - energy_ratio)),
            "estimated_depletion_rate": attrs["depletion_rate"],
            "distance_to_charger": float(distance),
            "estimated_travel_time": float(travel_time),
            "estimated_time_to_depletion": float(time_to_depletion),
            "deadline_slack": float(deadline_slack),
            "critical_energy_ratio": float(np.clip(attrs["critical_energy_ratio"], 0.0, 1.0)),
        }

    @staticmethod
    def feature_for(
        mc_x: float,
        mc_y: float,
        candidate,
        state_lookup: dict,
        network_diag: float = 141.421356,
        survival_horizon: float = 3600.0,
    ) -> np.ndarray:
        attrs = ChargerRoutingDQN._extract_physical_attributes(
            candidate, state_lookup)
        energy_ratio = attrs["estimated_energy"] / attrs["initial_energy"]
        distance = math.hypot(
            attrs["x"] - float(mc_x), attrs["y"] - float(mc_y))

        if attrs["depletion_rate"] > 0.0:
            time_to_depletion = attrs["estimated_energy"] / \
                max(attrs["depletion_rate"], 1e-12)
        else:
            time_to_depletion = 1e6

        normalized_survival = min(
            1.0, time_to_depletion / max(float(survival_horizon), 1e-6))

        return np.asarray(
            [
                (attrs["x"] - float(mc_x)) / max(network_diag, 1e-6),
                (attrs["y"] - float(mc_y)) / max(network_diag, 1e-6),
                energy_ratio,
                1.0 - energy_ratio,
                min(1.0, distance / max(network_diag, 1e-6)),
                1.0 - normalized_survival,
                min(1.0, attrs["depletion_rate"] / max(attrs["initial_energy"],
                    1e-12) * max(float(survival_horizon), 1e-6)),
            ],
            dtype=np.float32,
        )

    def heuristic_score(
        self,
        mc_x: float,
        mc_y: float,
        candidate,
        state_lookup: dict,
        speed: float = 1.0,
        network_diag: float = 141.421356,
    ) -> float:
        attrs = ChargerRoutingDQN._extract_physical_attributes(
            candidate, state_lookup)
        energy_ratio = attrs["estimated_energy"] / attrs["initial_energy"]
        distance = math.hypot(
            attrs["x"] - float(mc_x), attrs["y"] - float(mc_y))

        time_to_depletion = (attrs["estimated_energy"] / max(
            attrs["depletion_rate"], 1e-12) if attrs["depletion_rate"] > 0.0 else 1e6)
        travel_time = distance / max(float(speed), 1e-6)
        slack = time_to_depletion - travel_time

        deadline_urgency = 1.0 if slack <= 0.0 else 1.0 / \
            (1.0 + slack / self.expected_mission_time)
        energy_deficit = float(np.clip(1.0 - energy_ratio, 0.0, 1.0))
        proximity = 1.0 - \
            float(np.clip(distance / max(float(network_diag), 1e-12), 0.0, 1.0))

        return float((deadline_urgency + energy_deficit + proximity) / 3.0)

    @staticmethod
    def _rank_by_descending_score(values: np.ndarray) -> dict:
        sorted_indices = np.argsort(-np.asarray(values, dtype=float))
        return {int(candidate_index): int(rank) for rank, candidate_index in enumerate(sorted_indices, start=1)}

    def _predict_q_values(self, features: Sequence[np.ndarray]) -> np.ndarray:
        if self.enabled_torch:
            with torch.inference_mode():
                feature_tensor = torch.tensor(
                    np.vstack(features), dtype=torch.float32, device=self.device)
                return self.net(feature_tensor).detach().cpu().numpy().astype(float)
        return np.asarray([float(np.dot(self.weights, feature) + self.bias) for feature in features], dtype=float)

    def _adaptive_exploration_probability(self, q_values: np.ndarray, allowed_indices: np.ndarray) -> float:
        if self.fixed_epsilon is not None and self.exploration_mode not in {"adaptive", "uncertainty_adaptive", "uncertainty"}:
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
        entropy = -float(np.sum(probabilities *
                         np.log(np.maximum(probabilities, 1e-12)))) / math.log(k)
        exposure = math.sqrt(k / max(k + self.competitive_decisions, 1))
        return float(np.clip(entropy * exposure, 0.0, 1.0))

    def _warmup_active(self) -> bool:
        """Evaluate warm-up without changing the legacy default policy."""
        if self.warmup_readiness_mode in {
                "competitive_only", "legacy", "legacy_competitive"}:
            return self.competitive_decisions <= self.warmup_competitive_decisions

        enough_updates = self._updates >= self.warmup_updates
        enough_replay = len(self.replay) >= self.warmup_min_replay_samples
        recent = list(self.training_history)[-self.warmup_td_window:]
        if self.warmup_td_abs_threshold is None:
            td_stable = True
        else:
            td_stable = (
                len(recent) >= self.warmup_td_window and
                float(np.mean([item["td_abs_mean"] for item in recent]))
                <= self.warmup_td_abs_threshold)
        readiness = enough_updates and enough_replay and td_stable
        timeout = (
            self.warmup_max_total_decisions is not None and
            self.decisions >= self.warmup_max_total_decisions)
        return not (readiness or timeout)

    def _deadline_feasibility_override(self, dqn_index: int, heuristic_index: int, physical_values: Sequence[dict]) -> bool:
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
        candidate_list = list(candidates)

        if not candidate_list:
            explanation = {
                "decision_number": int(self.decisions),
                "decision_time": (None if decision_time is None else float(decision_time)),
                "charger_id": charger_id,
                "selection_mode": "no_candidates",
                "selected_node_id": None,
                "candidate_count": 0,
                "effective_candidate_count": 0,
                "competitive_decisions": int(self.competitive_decisions),
                "candidates": [],
            }
            return self._commit_decision(explanation, None)

        self.decisions += 1
        candidate_count = len(candidate_list)
        exploration_draw = False

        if candidate_count == 1:
            selected_candidate = candidate_list[0]
            cand_id = self._extract_physical_attributes(
                selected_candidate, state_lookup)["id"]
            physical = self._candidate_energy_and_deadline(
                mc_x, mc_y, selected_candidate, state_lookup, speed)
            feature = self.feature_for(
                mc_x, mc_y, selected_candidate, state_lookup, network_diag, self.safe_horizon)
            is_hard_rescue = bool(physical["energy_ratio"] <= max(self.critical_energy_ratio, physical.get(
                "critical_energy_ratio", 0.0)) or physical["deadline_slack"] <= self.deadline_safety_margin)

            decision_explanation = {
                "decision_number": int(self.decisions),
                "decision_time": (None if decision_time is None else float(decision_time)),
                "charger_id": charger_id,
                "charger_position": {"x": float(mc_x), "y": float(mc_y)},
                "charger_speed": float(speed),
                "epsilon": float(self.epsilon),
                "exploration_draw": bool(exploration_draw),
                "selection_mode": "single_candidate",
                "selected_node_id": str(cand_id),
                "candidate_count": 1,
                "effective_candidate_count": 1,
                "dqn_proposed_node_id": None,
                "heuristic_best_node_id": str(cand_id),
                "greedy_best_node_id": str(cand_id),
                "legacy_combined_best_node_id": None,
                "safety_subset_active": bool(is_hard_rescue),
                "allowed_candidate_count": 1,
                "safety_override": False,
                "safety_override_reason": None,
                "training_updates": int(self._updates),
                "competitive_decisions": int(self.competitive_decisions),
                "warmup_updates": int(self.warmup_updates),
                "warmup_competitive_decisions": int(self.warmup_competitive_decisions),
                "warmup_readiness_mode": self.warmup_readiness_mode,
                "warmup_active": bool(self._warmup_active()),
                "warmup_min_replay_samples": int(self.warmup_min_replay_samples),
                "critical_energy_ratio": float(self.critical_energy_ratio),
                "deadline_safety_margin": float(self.deadline_safety_margin),
                "heuristic_override_margin": (None if self.heuristic_override_margin is None else float(self.heuristic_override_margin)),
                "candidates": [{
                    "node_id": str(cand_id),
                    **physical,
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
            return self._commit_decision(
                decision_explanation, selected_candidate)

        physical_values = [self._candidate_energy_and_deadline(
            mc_x, mc_y, c, state_lookup, speed) for c in candidate_list]
        energy_ratios = np.asarray([item["energy_ratio"]
                                   for item in physical_values], dtype=float)
        deadline_slacks = np.asarray(
            [item["deadline_slack"] for item in physical_values], dtype=float)
        node_critical_ratios = np.asarray([max(self.critical_energy_ratio, val.get(
            "critical_energy_ratio", 0.0)) for val in physical_values], dtype=float)

        hard_rescue_mask = (energy_ratios <= node_critical_ratios) | (
            deadline_slacks <= self.deadline_safety_margin)

        if np.any(hard_rescue_mask):
            allowed_indices = np.flatnonzero(hard_rescue_mask)
            safety_subset_active = True
        else:
            allowed_indices = np.arange(candidate_count, dtype=int)
            safety_subset_active = False

        effective_candidate_count = int(len(allowed_indices))

        if effective_candidate_count == 1:
            selected_index = int(allowed_indices[0])
            selected_candidate = candidate_list[selected_index]
            selected_id = self._extract_physical_attributes(
                selected_candidate, state_lookup)["id"]
            allowed_set = {selected_index}
            candidate_explanations = []

            for index, candidate in enumerate(candidate_list):
                cand_id = self._extract_physical_attributes(
                    candidate, state_lookup)["id"]
                feature = self.feature_for(
                    mc_x, mc_y, candidate, state_lookup, network_diag, self.safe_horizon)
                candidate_explanations.append({
                    "node_id": str(cand_id),
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
                "decision_time": (None if decision_time is None else float(decision_time)),
                "charger_id": charger_id,
                "charger_position": {"x": float(mc_x), "y": float(mc_y)},
                "charger_speed": float(speed),
                "epsilon": float(self.epsilon),
                "exploration_draw": bool(exploration_draw),
                "selection_mode": "safety_forced_single_candidate",
                "selected_node_id": str(selected_id),
                "candidate_count": int(candidate_count),
                "effective_candidate_count": 1,
                "dqn_proposed_node_id": None,
                "heuristic_best_node_id": str(selected_id),
                "greedy_best_node_id": str(selected_id),
                "legacy_combined_best_node_id": None,
                "safety_subset_active": bool(safety_subset_active),
                "allowed_candidate_count": 1,
                "safety_override": True,
                "safety_override_reason": "only_one_safe_candidate",
                "training_updates": int(self._updates),
                "competitive_decisions": int(self.competitive_decisions),
                "warmup_updates": int(self.warmup_updates),
                "warmup_competitive_decisions": int(self.warmup_competitive_decisions),
                "warmup_readiness_mode": self.warmup_readiness_mode,
                "warmup_active": bool(self._warmup_active()),
                "warmup_min_replay_samples": int(self.warmup_min_replay_samples),
                "critical_energy_ratio": float(self.critical_energy_ratio),
                "deadline_safety_margin": float(self.deadline_safety_margin),
                "heuristic_override_margin": (None if self.heuristic_override_margin is None else float(self.heuristic_override_margin)),
                "candidates": candidate_explanations,
            }
            return self._commit_decision(
                decision_explanation, selected_candidate)

        self.competitive_decisions += 1
        features = [self.feature_for(
            mc_x, mc_y, c, state_lookup, network_diag, self.safe_horizon) for c in candidate_list]
        heuristic_values = np.asarray([self.heuristic_score(
            mc_x, mc_y, c, state_lookup, speed, network_diag) for c in candidate_list], dtype=float)
        q_values = self._predict_q_values(features)
        combined_scores = q_values + heuristic_values

        dqn_selected_index = int(
            allowed_indices[int(np.argmax(q_values[allowed_indices]))])
        heuristic_selected_index = int(
            allowed_indices[int(np.argmax(heuristic_values[allowed_indices]))])

        safety_override = False
        safety_override_reason = None
        in_competitive_warmup = self._warmup_active()

        self.epsilon = self._adaptive_exploration_probability(
            q_values, allowed_indices)
        exploration_draw = ((not in_competitive_warmup)
                            and self._rng.random() < self.epsilon)

        if in_competitive_warmup:
            selected_index = heuristic_selected_index
            selection_mode = "heuristic_competitive_warmup"
            if selected_index != dqn_selected_index:
                safety_override = True
                safety_override_reason = "competitive_training_warmup"
        elif exploration_draw:
            selected_index = int(self._rng.choice(allowed_indices.tolist()))
            selection_mode = "safe_epsilon_exploration"
        else:
            use_feasibility_override = self.safety_override_mode in {
                "adaptive", "deadline", "deadline_feasibility", "feasibility"}
            if use_feasibility_override:
                should_override = self._deadline_feasibility_override(
                    dqn_selected_index, heuristic_selected_index, physical_values)
                override_reason = "deadline_feasibility"
            else:
                heuristic_gap = float(
                    heuristic_values[heuristic_selected_index] - heuristic_values[dqn_selected_index])
                should_override = (
                    self.heuristic_override_margin is not None and heuristic_gap > self.heuristic_override_margin)
                override_reason = "heuristic_gap_exceeded"

            if should_override:
                selected_index = heuristic_selected_index
                selection_mode = "heuristic_safety_override"
                safety_override = True
                safety_override_reason = override_reason
            else:
                selected_index = dqn_selected_index
                selection_mode = "dqn_with_safety_shield"

        selected_candidate = candidate_list[selected_index]
        selected_id = self._extract_physical_attributes(
            selected_candidate, state_lookup)["id"]

        q_rank = self._rank_by_descending_score(q_values)
        heuristic_rank = self._rank_by_descending_score(heuristic_values)
        combined_rank = self._rank_by_descending_score(combined_scores)
        allowed_set = {int(i) for i in allowed_indices.tolist()}

        candidate_explanations = []
        for index, candidate in enumerate(candidate_list):
            cand_id = self._extract_physical_attributes(
                candidate, state_lookup)["id"]
            feature = features[index]
            candidate_explanations.append({
                "node_id": str(cand_id),
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

        dqn_proposed_id = self._extract_physical_attributes(
            candidate_list[dqn_selected_index], state_lookup)["id"]
        heuristic_best_id = self._extract_physical_attributes(
            candidate_list[heuristic_selected_index], state_lookup)["id"]
        legacy_best_id = self._extract_physical_attributes(
            candidate_list[legacy_combined_best_index], state_lookup)["id"]

        decision_explanation = {
            "decision_number": int(self.decisions),
            "decision_time": (None if decision_time is None else float(decision_time)),
            "charger_id": charger_id,
            "charger_position": {"x": float(mc_x), "y": float(mc_y)},
            "charger_speed": float(speed),
            "epsilon": float(self.epsilon),
            "exploration_draw": bool(exploration_draw),
            "selection_mode": selection_mode,
            "selected_node_id": str(selected_id),
            "candidate_count": int(candidate_count),
            "effective_candidate_count": int(effective_candidate_count),
            "dqn_proposed_node_id": str(dqn_proposed_id),
            "heuristic_best_node_id": str(heuristic_best_id),
            "greedy_best_node_id": str(heuristic_best_id),
            "legacy_combined_best_node_id": str(legacy_best_id),
            "safety_subset_active": bool(safety_subset_active),
            "allowed_candidate_count": int(len(allowed_indices)),
            "safety_override": bool(safety_override),
            "safety_override_reason": safety_override_reason,
            "training_updates": int(self._updates),
            "competitive_decisions": int(self.competitive_decisions),
            "warmup_updates": int(self.warmup_updates),
            "warmup_competitive_decisions": int(self.warmup_competitive_decisions),
            "warmup_readiness_mode": self.warmup_readiness_mode,
            "warmup_active": bool(self._warmup_active()),
            "warmup_min_replay_samples": int(self.warmup_min_replay_samples),
            "critical_energy_ratio": float(self.critical_energy_ratio),
            "deadline_safety_margin": float(self.deadline_safety_margin),
            "heuristic_override_margin": (None if self.heuristic_override_margin is None else float(self.heuristic_override_margin)),
            "candidates": candidate_explanations,
        }
        return self._commit_decision(
            decision_explanation, selected_candidate)

    def train_on_transition(
        self,
        feature: np.ndarray,
        reward: float,
        next_features: Optional[Sequence[np.ndarray]] = None,
    ) -> float:
        feature = np.asarray(feature, dtype=np.float32)
        next_feature_list = ([] if next_features is None else [
                             np.asarray(item, dtype=np.float32) for item in next_features])

        self.replay.append((feature, float(reward), next_feature_list))

        if not self.enabled_torch:
            target_value = float(reward)
            if next_feature_list:
                target_value += self.gamma * \
                    max(float(np.dot(self.weights, next_feature) + self.bias)
                        for next_feature in next_feature_list)
            prediction = float(np.dot(self.weights, feature) + self.bias)
            error = target_value - prediction

            self.weights += self.lr * error * feature
            self.bias += self.lr * error
            self._updates += 1

            self.last_loss = float(error * error)
            self._append_training_audit(
                reward=float(reward), loss=self.last_loss,
                td_abs_mean=abs(float(error)))
            return self.last_loss

        batch = self._rng.sample(list(self.replay), k=min(
            len(self.replay), self.batch_size))
        feature_tensor = torch.tensor(np.vstack(
            [item[0] for item in batch]), dtype=torch.float32, device=self.device)

        target_values = []
        for _, transition_reward, following_features in batch:
            target_value = float(transition_reward)
            if following_features:
                with torch.inference_mode():
                    next_tensor = torch.tensor(
                        np.vstack(following_features), dtype=torch.float32, device=self.device)
                    target_value += self.gamma * \
                        float(torch.max(self.target(next_tensor)
                                        ).detach().cpu().item())
            target_values.append(target_value)

        target_tensor = torch.tensor(
            target_values, dtype=torch.float32, device=self.device)

        predictions = self.net(feature_tensor)
        td_abs_mean = float(torch.mean(torch.abs(
            predictions.detach() - target_tensor)).cpu().item())
        loss = torch.nn.functional.smooth_l1_loss(predictions, target_tensor)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()

        self._updates += 1

        if self._updates % self.target_sync_interval == 0:
            self.target.load_state_dict(self.net.state_dict())

        self.last_loss = float(loss.detach().cpu().item())
        self._append_training_audit(
            reward=float(reward), loss=self.last_loss,
            td_abs_mean=td_abs_mean)
        return self.last_loss
