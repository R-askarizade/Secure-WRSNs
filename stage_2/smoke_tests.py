from __future__ import annotations

import numpy as np
import pandas as pd

from .parameter_registry import (
    LOCKED_CONFIGURATION_KEYS,
    apply_method_values,
    decode_unit_row,
    get_specs,
    method_values_to_overrides,
    unit_problem,
)
from .pipeline import aggregate_design_results, nondominated_mask, objective_costs
from sim_phase import _scenarios_for_mode


def test_registry_decoding_and_constraints() -> None:
    specs = get_specs("qifl_dqn")
    values = decode_unit_row(specs, np.linspace(0.0, 1.0, len(specs)))
    overrides = method_values_to_overrides(values)
    assert overrides["qifl_threshold_upper_bound"] > overrides["qifl_threshold_lower_bound"]
    assert overrides["dqn_replay_capacity"] >= overrides["dqn_batch_size"]
    reward_keys = [
        "dqn_reward_energy_weight", "dqn_reward_deadline_weight",
        "dqn_reward_travel_weight", "dqn_reward_revisit_penalty",
        "dqn_reward_death_penalty",
    ]
    assert abs(sum(overrides[k] for k in reward_keys) - 14.25) < 1e-10
    assert not any(name.endswith("_to_deadline_ratio") for name in overrides)
    assert unit_problem(specs)["num_vars"] == len(specs)


def test_locked_network_configuration() -> None:
    _, base = _scenarios_for_mode("energy")[0]
    before = {key: base.get(key) for key in LOCKED_CONFIGURATION_KEYS}
    changed = apply_method_values(base, {
        "qifl_learning_rate": 0.002,
        "minimum_charge_energy_ratio": 0.10,
    })
    after = {key: changed.get(key) for key in LOCKED_CONFIGURATION_KEYS}
    assert before == after


def test_dynamic_weights_identifiability_removed() -> None:
    overrides = method_values_to_overrides({
        "dynamic_charging_wt_to_now_ratio": 2.0,
        "dynamic_charging_eps_to_now_ratio": 0.5,
    })
    keys = (
        "dynamic_charging_weight_wt",
        "dynamic_charging_weight_eps",
        "dynamic_charging_weight_now",
    )
    assert abs(sum(overrides[key] for key in keys) - 3.0) < 1e-12
    assert abs(
        overrides["dynamic_charging_weight_wt"]
        / overrides["dynamic_charging_weight_now"] - 2.0
    ) < 1e-12
    assert abs(
        overrides["dynamic_charging_weight_eps"]
        / overrides["dynamic_charging_weight_now"] - 0.5
    ) < 1e-12
    assert "dynamic_charging_wt_to_now_ratio" not in overrides
    assert "dynamic_charging_eps_to_now_ratio" not in overrides



def test_reward_weight_identifiability_removed() -> None:
    overrides = method_values_to_overrides({
        "dqn_reward_energy_to_deadline_ratio": 0.5,
        "dqn_reward_travel_to_deadline_ratio": 0.25,
        "dqn_reward_revisit_to_deadline_ratio": 0.75,
        "dqn_reward_death_to_deadline_ratio": 2.0,
    })
    deadline = overrides["dqn_reward_deadline_weight"]
    assert abs(overrides["dqn_reward_energy_weight"] / deadline - 0.5) < 1e-12
    assert abs(overrides["dqn_reward_travel_weight"] / deadline - 0.25) < 1e-12
    assert abs(overrides["dqn_reward_revisit_penalty"] / deadline - 0.75) < 1e-12
    assert abs(overrides["dqn_reward_death_penalty"] / deadline - 2.0) < 1e-12
    assert abs(sum(overrides[key] for key in (
        "dqn_reward_energy_weight", "dqn_reward_deadline_weight",
        "dqn_reward_travel_weight", "dqn_reward_revisit_penalty",
        "dqn_reward_death_penalty",
    )) - 14.25) < 1e-12


def test_paired_seed_completeness_guard() -> None:
    design = pd.DataFrame({"design_id": ["morris_000000", "morris_000001"]})
    evaluations = pd.DataFrame([
        {"design_id": "morris_000000", "Seed": 101, "Status": "SUCCESS", "MCC": 0.1},
        {"design_id": "morris_000000", "Seed": 102, "Status": "SUCCESS", "MCC": 0.2},
    ])
    try:
        aggregate_design_results(
            design, evaluations, ["MCC"], expected_seeds=[101, 102]
        )
    except ValueError as exc:
        assert "complete paired-seed matrix" in str(exc)
    else:
        raise AssertionError("Incomplete paired-seed design was not rejected")

def test_pareto_mask() -> None:
    costs = np.asarray([[1, 1], [2, 2], [0.5, 2], [2, 0.5]], dtype=float)
    mask = nondominated_mask(costs)
    assert mask.tolist() == [True, False, True, True]


def test_objective_constraints() -> None:
    baseline = {
        "MC_Total_Mission_Energy": 100.0,
        "Overall_PDR": 95.0,
        "MCC": 0.5,
        "MC_Fail_Closed_Rejections": 0.0,
    }
    tuning = {
        "constraint_penalty": 10.0,
        "constraints": {
            "overall_pdr_drop_percentage_points": 0.5,
            "mcc_drop": 0.02,
            "max_fail_closed_increase": 0.0,
        },
    }
    good = {
        "MC_Deadline_Failure_Rate": 0.1,
        "MC_Total_Mission_Energy": 90.0,
        "Critical_PDR": 96.0,
        "MCC": 0.52,
        "Overall_PDR": 95.0,
        "MC_Fail_Closed_Rejections": 0.0,
    }
    costs, feasible, _ = objective_costs(good, baseline, tuning)
    assert feasible and costs["mission_energy_ratio"] == 0.9
    bad = dict(good, Overall_PDR=80.0)
    costs_bad, feasible_bad, _ = objective_costs(bad, baseline, tuning)
    assert not feasible_bad and min(costs_bad.values()) > 10.0


def main() -> None:
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)} Stage-2 smoke tests")


if __name__ == "__main__":
    main()
