from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from adaptive_charging import ChargerRoutingDQN, QuantumFuzzyChargingOracle


class DummyNode:
    def __init__(self, node_id, x, y, energy, emax=1.0):
        self.node_id = node_id
        self.x = float(x)
        self.y = float(y)
        self.E_current = float(energy)
        self.E_max = float(emax)
        self.E_critical = 0.15 * float(emax)


def test_qifl_batch_equivalence():
    oracle = QuantumFuzzyChargingOracle(device="cpu", seed=7)
    pairs = [(0.1, 0.2), (0.5, 0.5), (0.9, 0.1), (0.2, 0.8)]
    batch, probs = oracle.infer_batch(pairs)
    scalar = np.asarray([oracle.infer(a, b)[0] for a, b in pairs])
    assert probs.shape == (len(pairs), len(oracle.centers))
    assert np.max(np.abs(batch - scalar)) < 1e-6


def test_default_warmup_is_legacy_compatible():
    dqn = ChargerRoutingDQN(
        device="cpu", seed=11, warmup_updates=2,
        warmup_competitive_decisions=2,
        warmup_readiness_mode="competitive_only")
    assert dqn._warmup_active()
    dqn.competitive_decisions = 2
    assert dqn._warmup_active()
    dqn.competitive_decisions = 3
    assert not dqn._warmup_active()


def test_composite_warmup_has_timeout_and_training_readiness():
    dqn = ChargerRoutingDQN(
        device="cpu", seed=13, batch_size=2,
        warmup_updates=2, warmup_competitive_decisions=999,
        warmup_readiness_mode="composite",
        warmup_min_replay_samples=2,
        warmup_td_abs_threshold=None,
        warmup_max_total_decisions=10)
    assert dqn._warmup_active()
    f = np.zeros(7, dtype=np.float32)
    dqn.train_on_transition(f, 1.0, [f])
    assert dqn._warmup_active()
    dqn.train_on_transition(f, 1.0, [f])
    assert not dqn._warmup_active()

    timeout_dqn = ChargerRoutingDQN(
        device="cpu", seed=17,
        warmup_updates=1000,
        warmup_readiness_mode="composite",
        warmup_max_total_decisions=3)
    timeout_dqn.decisions = 3
    assert not timeout_dqn._warmup_active()


def test_audit_and_checkpoint_do_not_change_selection():
    state = {
        1: {"E_initial": 1.0, "E_est": 0.3, "energy_q_ewma": 0.001},
        2: {"E_initial": 1.0, "E_est": 0.6, "energy_q_ewma": 0.002},
    }
    candidates = [DummyNode(1, 10, 0, 0.3), DummyNode(2, 20, 0, 0.6)]
    dqn = ChargerRoutingDQN(
        device="cpu", seed=19,
        warmup_competitive_decisions=0,
        epsilon=0.0,
        exploration_mode="fixed")
    selected = dqn.select(0, 0, candidates, state, speed=5.0)
    snapshot = dqn.audit_snapshot()
    assert selected in candidates
    assert snapshot["decisions"] == 1
    assert snapshot["competitive_decisions"] == 1
    assert snapshot["dqn_raw_proposal_count"] == 1

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "checkpoint.bin"
        dqn.save_checkpoint(str(path))
        clone = ChargerRoutingDQN(
            device="cpu", seed=999,
            warmup_competitive_decisions=0,
            epsilon=0.0,
            exploration_mode="fixed")
        clone.load_checkpoint(str(path))
        original_q = dqn._predict_q_values([
            dqn.feature_for(0, 0, c, state) for c in candidates])
        clone_q = clone._predict_q_values([
            clone.feature_for(0, 0, c, state) for c in candidates])
        assert np.max(np.abs(original_q - clone_q)) < 1e-7


def run_all():
    test_qifl_batch_equivalence()
    test_default_warmup_is_legacy_compatible()
    test_composite_warmup_has_timeout_and_training_readiness()
    test_audit_and_checkpoint_do_not_change_selection()
    print("DQN_QIFL_AUDIT_TESTS_OK")


if __name__ == "__main__":
    run_all()
