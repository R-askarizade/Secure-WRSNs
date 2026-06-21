from __future__ import annotations

import math
import random
from collections import deque
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

# PyTorch is optional at import time; the simulator can still run on CPU-only machines.
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None


class QuantumFuzzyChargingOracle:

    def __init__(
        self,
        threshold_centers: Sequence[float] = (0.18, 0.26, 0.34, 0.42, 0.50),
        lr: float = 2.5e-3,
        device: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        self.centers = np.asarray(threshold_centers, dtype=float)
        self.last_loss = 0.0
        self.enabled_torch = (torch is not None and nn is not None and optim is not None and (
            device is not None or torch.cuda.is_available()))
        if self.enabled_torch:
            if device is None:
                device = "cuda"
            self.device = torch.device(device)
            torch.manual_seed(seed)
            logits = self._initial_rule_logits()
            self.theta = torch.nn.Parameter(torch.tensor(
                logits, dtype=torch.float32, device=self.device))
            self.optimizer = optim.Adam([self.theta], lr=float(lr))
            self.centers_t = torch.tensor(
                self.centers, dtype=torch.float32, device=self.device)
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
        return np.asarray([
            max(0.0, min(1.0, (0.50 - x) / 0.50)),
            self._triangular(x, 0.20, 0.50, 0.80),
            max(0.0, min(1.0, (x - 0.50) / 0.50)),
        ], dtype=float)

    def _input_state_np(self, ed_norm: float, sd_norm: float) -> np.ndarray:
        ed_m = self._memberships(ed_norm)
        sd_m = self._memberships(sd_norm)
        state = np.outer(ed_m, sd_m).reshape(-1)
        s = float(np.sum(state))
        if s <= 1e-12:
            state[:] = 1.0 / 9.0
        else:
            state /= s
        return state

    def _initial_rule_probs(self) -> np.ndarray:
        table = np.asarray([
            [0.02, 0.08, 0.20, 0.38, 0.32],  # ED low, SD low
            [0.01, 0.05, 0.18, 0.38, 0.38],
            [0.01, 0.03, 0.14, 0.34, 0.48],
            [0.08, 0.22, 0.38, 0.24, 0.08],  # ED mid
            [0.05, 0.18, 0.36, 0.30, 0.11],
            [0.03, 0.12, 0.33, 0.34, 0.18],
            [0.45, 0.32, 0.16, 0.05, 0.02],  # ED high
            [0.36, 0.34, 0.20, 0.07, 0.03],
            [0.28, 0.35, 0.24, 0.09, 0.04],
        ], dtype=float)
        table /= np.maximum(table.sum(axis=1, keepdims=True), 1e-12)
        return table

    def _initial_rule_logits(self) -> np.ndarray:
        probs = self._initial_rule_probs()
        return np.log(np.maximum(probs, 1e-6))

    def infer(self, ed_norm: float, sd_norm: float) -> Tuple[float, np.ndarray]:
        psi = self._input_state_np(ed_norm, sd_norm)
        if self.enabled_torch:
            with torch.no_grad():
                psi_t = torch.tensor(
                    psi, dtype=torch.float32, device=self.device)
                U = torch.softmax(self.theta, dim=1)
                probs = torch.matmul(psi_t, U)
                probs = probs / torch.clamp(probs.sum(), min=1e-12)
                threshold = torch.sum(probs * self.centers_t).item()
                return float(threshold), probs.detach().cpu().numpy()
        U = self.theta_np
        probs = psi @ U
        probs = probs / max(float(np.sum(probs)), 1e-12)
        return float(np.sum(probs * self.centers)), probs

    def train_step(self, ed_norm: float, sd_norm: float, target_threshold: float) -> float:
        target_threshold = float(target_threshold)
        if not self.enabled_torch:
            pred, _ = self.infer(ed_norm, sd_norm)
            self.last_loss = float((pred - target_threshold) ** 2)
            return self.last_loss
        psi = torch.tensor(self._input_state_np(
            ed_norm, sd_norm), dtype=torch.float32, device=self.device)
        target = torch.tensor(
            target_threshold, dtype=torch.float32, device=self.device)
        U = torch.softmax(self.theta, dim=1)
        probs = torch.matmul(psi, U)
        probs = probs / torch.clamp(probs.sum(), min=1e-12)
        pred = torch.sum(probs * self.centers_t)
        entropy = -torch.sum(probs * torch.log(torch.clamp(probs, min=1e-8)))
        loss = torch.nn.functional.smooth_l1_loss(
            pred, target) + 1e-4 * entropy
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([self.theta], 1.0)
        self.optimizer.step()
        self.last_loss = float(loss.detach().cpu().item())
        return self.last_loss


class _QNet(nn.Module):
    def __init__(self, input_dim: int = 7) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ChargerRoutingDQN:
    """Safety-shaped DQN scorer for choosing the next node in an assigned cluster."""

    def __init__(self, lr: float = 1e-3, gamma: float = 0.80, epsilon: float = 0.08,
                 device: Optional[str] = None, seed: int = 123) -> None:
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.decisions = 0
        self.last_loss = 0.0
        self.replay = deque(maxlen=512)
        self.enabled_torch = (torch is not None and nn is not None and optim is not None and (
            device is not None or torch.cuda.is_available()))
        random.seed(seed)
        np.random.seed(seed)
        if self.enabled_torch:
            if device is None:
                device = "cuda"
            self.device = torch.device(device)
            torch.manual_seed(seed)
            self.net = _QNet().to(self.device)
            self.target = _QNet().to(self.device)
            self.target.load_state_dict(self.net.state_dict())
            self.optimizer = optim.Adam(self.net.parameters(), lr=float(lr))
            self._updates = 0
        else:
            self.device = None
            self.weights = np.zeros(7, dtype=np.float32)
            self.bias = 0.0
            self.lr = float(lr)

    @staticmethod
    def feature_for(mc_x: float, mc_y: float, node, state: dict, network_diag: float = 141.421356) -> np.ndarray:
        e0 = float(state.get('E_initial', getattr(node, 'E_max', 1.0)))
        e = float(state.get('E_est', getattr(node, 'E_current', e0)))
        ratio = e / max(e0, 1e-12)
        qbar = float(state.get('energy_q_ewma', 0.0))
        dist = math.hypot(float(node.x) - float(mc_x),
                          float(node.y) - float(mc_y))
        ttd = e / max(qbar, 1e-12) if qbar > 0 else 1e6
        slack = min(1.0, ttd / 3600.0)
        return np.asarray([
            (float(node.x) - float(mc_x)) / max(network_diag, 1e-6),
            (float(node.y) - float(mc_y)) / max(network_diag, 1e-6),
            ratio,
            1.0 - ratio,
            min(1.0, dist / max(network_diag, 1e-6)),
            1.0 - slack,
            min(1.0, qbar / max(e0, 1e-12) * 3600.0),
        ], dtype=np.float32)

    @staticmethod
    def heuristic_score(mc_x: float, mc_y: float, node, state: dict, speed: float = 1.0) -> float:
        e0 = float(state.get('E_initial', getattr(node, 'E_max', 1.0)))
        e = float(state.get('E_est', getattr(node, 'E_current', e0)))
        ratio = e / max(e0, 1e-12)
        dist = math.hypot(float(node.x) - float(mc_x),
                          float(node.y) - float(mc_y))
        qbar = float(state.get('energy_q_ewma', 0.0))
        ttd = e / max(qbar, 1e-12) if qbar > 0 else 1e6
        travel = dist / max(float(speed), 1e-6)
        urgency = 1.0 / max(ttd - travel + 1.0, 1.0)
        useful = max(0.0, 1.0 - ratio)
        return 4.0 * urgency + 1.5 * useful - 0.015 * dist

    def select(self, mc_x: float, mc_y: float, candidates: Sequence, state_lookup: dict, speed: float = 1.0,
               network_diag: float = 141.421356):
        if not candidates:
            return None
        self.decisions += 1
        if random.random() < self.epsilon:
            return random.choice(list(candidates))
        features = [self.feature_for(mc_x, mc_y, n, state_lookup.get(
            n.node_id, {}), network_diag) for n in candidates]
        heur = np.asarray([self.heuristic_score(mc_x, mc_y, n, state_lookup.get(
            n.node_id, {}), speed) for n in candidates], dtype=float)
        if self.enabled_torch:
            with torch.no_grad():
                x = torch.tensor(np.vstack(features),
                                 dtype=torch.float32, device=self.device)
                q = self.net(x).detach().cpu().numpy()
            # Safety shaping keeps early training from degenerating into random long tours.
            score = q + heur
        else:
            q = np.asarray([float(np.dot(self.weights, f) + self.bias)
                           for f in features], dtype=float)
            score = q + heur
        return list(candidates)[int(np.argmax(score))]

    def train_on_transition(self, feature: np.ndarray, reward: float, next_features: Optional[Sequence[np.ndarray]] = None) -> float:
        feature = np.asarray(feature, dtype=np.float32)
        next_features = [] if next_features is None else [
            np.asarray(f, dtype=np.float32) for f in next_features]
        self.replay.append((feature, float(reward), next_features))
        if not self.enabled_torch:
            target = float(reward)
            if next_features:
                target += self.gamma * \
                    max(float(np.dot(self.weights, nf) + self.bias)
                        for nf in next_features)
            pred = float(np.dot(self.weights, feature) + self.bias)
            err = target - pred
            self.weights += self.lr * err * feature
            self.bias += self.lr * err
            self.last_loss = float(err * err)
            return self.last_loss
        batch = random.sample(list(self.replay), k=min(len(self.replay), 16))
        x = torch.tensor(np.vstack([b[0] for b in batch]),
                         dtype=torch.float32, device=self.device)
        rewards = []
        for _, r, nf in batch:
            target = float(r)
            if nf:
                with torch.no_grad():
                    nx = torch.tensor(
                        np.vstack(nf), dtype=torch.float32, device=self.device)
                    target += self.gamma * \
                        float(torch.max(self.target(nx)).detach().cpu().item())
            rewards.append(target)
        y = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        pred = self.net(x)
        loss = torch.nn.functional.smooth_l1_loss(pred, y)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()
        self._updates += 1
        if self._updates % 10 == 0:
            self.target.load_state_dict(self.net.state_dict())
        self.last_loss = float(loss.detach().cpu().item())
        return self.last_loss
