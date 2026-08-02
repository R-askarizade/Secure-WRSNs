from __future__ import annotations

# CuBLAS determinism must be configured before PyTorch is imported transitively.
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .parameter_registry import (
    LOCKED_CONFIGURATION_KEYS,
    LOCKED_METHOD_SWITCHES,
    PARAMETERS,
    ParameterSpec,
    apply_method_values,
    config_hash,
    decode_unit_row,
    get_specs,
    method_values_to_overrides,
    registry_records,
    unit_problem,
)


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DEFAULT_SETTINGS = PACKAGE_DIR / "settings.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "stage2_results"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=_json_default)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=_json_default)
    os.replace(temp, path)


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp, index=False)
    os.replace(temp, path)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def dependency_guard(*packages: str) -> None:
    missing = [name for name in packages if package_version(name) is None]
    if missing:
        raise RuntimeError(
            "Missing Stage-2 dependencies: " + ", ".join(missing)
            + f". Install them with: python -m pip install -r {PROJECT_ROOT / 'requirements_stage2.txt'}"
        )


def settings_hash(settings: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_json(settings).encode("utf-8")).hexdigest()


def make_manifest(settings: Mapping[str, Any], command: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    manifest = {
        "command": command,
        "created_unix_time": time.time(),
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "settings_sha256": settings_hash(settings),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "packages": {
            name: package_version(name)
            for name in (
                "numpy", "pandas", "scipy", "scikit-learn", "networkx",
                "torch", "SALib", "ConfigSpace", "smac"
            )
        },
        "locked_configuration_keys": sorted(LOCKED_CONFIGURATION_KEYS),
        "locked_method_switches": sorted(LOCKED_METHOD_SWITCHES),
        "settings": dict(settings),
    }
    if extra:
        manifest["extra"] = dict(extra)
    return manifest


def load_settings(path: Path) -> dict[str, Any]:
    settings = load_json(path)
    required = {
        "scenario_mode", "num_nodes", "simulation_time_s", "network_size",
        "sensitivity_seeds", "optimization_seeds", "final_evaluation_seeds",
        "analysis_metrics", "selection_metrics", "morris", "sobol", "tuning",
    }
    missing = sorted(required - set(settings))
    if missing:
        raise ValueError(f"Settings file is missing: {missing}")
    seed_groups = (
        set(settings["sensitivity_seeds"]),
        set(settings["optimization_seeds"]),
        set(settings["final_evaluation_seeds"]),
    )
    if seed_groups[0] & seed_groups[1] or seed_groups[0] & seed_groups[2] or seed_groups[1] & seed_groups[2]:
        raise ValueError("Sensitivity, optimization, and final seed sets must be disjoint")

    # Validate the Morris configuration while loading settings so malformed
    # experiments fail before any registry, audit, or simulation work begins.
    morris_cfg = settings["morris"]
    candidate = int(morris_cfg["candidate_trajectories"])
    optimal = int(morris_cfg["optimal_trajectories"])
    levels = int(morris_cfg["num_levels"])
    if candidate < 2:
        raise ValueError("morris.candidate_trajectories must be at least 2")
    if optimal < 1:
        raise ValueError("morris.optimal_trajectories must be at least 1")
    if optimal >= candidate:
        raise ValueError(
            "morris.optimal_trajectories must be strictly less than "
            "morris.candidate_trajectories"
        )
    if levels < 2:
        raise ValueError("morris.num_levels must be at least 2")
    return settings


def _load_wrsn_modules():
    from sim_phase import _add_charging_efficiency_diagnostics, _scenarios_for_mode
    from simulator_manager import SimulatorManager
    return _add_charging_efficiency_diagnostics, _scenarios_for_mode, SimulatorManager


class WRSNEvaluator:
    """Deterministic, checkpoint-friendly adapter around the current WRSN simulator."""

    def __init__(
        self,
        settings: Mapping[str, Any],
        *,
        require_cuda: bool = False,
        strict_determinism: bool = True,
        quiet: bool = True,
    ) -> None:
        self.settings = dict(settings)
        self.require_cuda = bool(require_cuda)
        self.strict_determinism = bool(strict_determinism)
        self.quiet = bool(quiet)
        diagnostics, scenarios_for_mode, manager = _load_wrsn_modules()
        self._diagnostics = diagnostics
        self._manager = manager
        scenarios = scenarios_for_mode(str(self.settings["scenario_mode"]))
        if len(scenarios) != 1:
            raise ValueError(
                "Stage 2 requires one frozen scenario. "
                f"Mode {self.settings['scenario_mode']!r} produced {len(scenarios)} scenarios."
            )
        self.scenario_name, self.base_config = scenarios[0]
        self.base_config = dict(self.base_config)
        self.locked_fingerprint = {
            key: self.base_config.get(key)
            for key in sorted(LOCKED_CONFIGURATION_KEYS | LOCKED_METHOD_SWITCHES)
        }

    def _build_config(self, method_values: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        overrides = method_values_to_overrides(method_values)
        config = apply_method_values(self.base_config, method_values)
        config["adaptive_torch_deterministic"] = True
        config["adaptive_torch_strict_determinism"] = self.strict_determinism
        config["adaptive_torch_allow_tf32"] = False
        config["require_pytorch_for_adaptive_charging"] = True
        config["require_cuda_for_adaptive_charging"] = self.require_cuda
        config["adaptive_compute_device"] = "cuda" if self.require_cuda else config.get(
            "adaptive_compute_device", "auto"
        )
        for key, expected in self.locked_fingerprint.items():
            if config.get(key) != expected:
                raise AssertionError(f"Locked key {key} changed from {expected!r} to {config.get(key)!r}")
        return config, overrides

    def evaluate(
        self,
        method_values: Mapping[str, Any],
        seed: int,
        *,
        tag: str,
        log_path: Path | None = None,
        num_nodes: int | None = None,
        simulation_time_s: int | None = None,
    ) -> dict[str, Any]:
        config, overrides = self._build_config(method_values)
        n_nodes = int(self.settings["num_nodes"] if num_nodes is None else num_nodes)
        sim_time = int(self.settings["simulation_time_s"] if simulation_time_s is None else simulation_time_s)
        network_size = tuple(float(x) for x in self.settings["network_size"])
        started = time.perf_counter()

        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stream_context = log_path.open("w", encoding="utf-8", newline="\n")
        elif self.quiet:
            stream_context = open(os.devnull, "w", encoding="utf-8")
        else:
            stream_context = contextlib.nullcontext(sys.stdout)

        try:
            with stream_context as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                sim = self._manager(
                    num_nodes=n_nodes,
                    sim_time=sim_time,
                    ablation_config=config,
                    seed=int(seed),
                    network_size=network_size,
                )
                sim.run_simulation()
                summary = sim.metrics.export_summary(int(seed), tag, sim_time)
            frame = self._diagnostics(pd.DataFrame([summary]))
            result = frame.iloc[0].to_dict()
            result.update({
                "Status": "SUCCESS",
                "Tag": tag,
                "Config_Hash": config_hash(overrides),
                "Method_Values_JSON": stable_json(dict(method_values)),
                "Config_Overrides_JSON": stable_json(overrides),
                "Wall_Time_s": time.perf_counter() - started,
                "Num_Nodes": n_nodes,
                "Simulation_Time_s": sim_time,
            })
            return result
        finally:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()


def design_dataframe(method: str, profile: str, specs: Sequence[ParameterSpec], X: np.ndarray) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for idx, unit_row in enumerate(np.asarray(X, dtype=float)):
        values = decode_unit_row(specs, unit_row)
        overrides = method_values_to_overrides(values)
        record: dict[str, Any] = {
            "design_id": f"{method}_{idx:06d}",
            "method": method,
            "profile": profile,
            "parameter_count": len(specs),
            "method_values_json": stable_json(values),
            "config_overrides_json": stable_json(overrides),
            "config_hash": config_hash(overrides),
        }
        for spec, unit_value in zip(specs, unit_row):
            record[f"u__{spec.name}"] = float(unit_value)
            record[f"p__{spec.name}"] = values[spec.name]
        records.append(record)
    return pd.DataFrame(records)


def read_selected_names(path: Path) -> list[str]:
    payload = load_json(path)
    names = payload.get("selected_parameters", payload.get("parameters", []))
    if not isinstance(names, list) or not names:
        raise ValueError(f"No selected parameters in {path}")
    return [str(name) for name in names]


def cmd_registry(args: argparse.Namespace, settings: Mapping[str, Any]) -> None:
    specs = get_specs(args.profile)
    output = Path(args.output)
    atomic_write_csv(pd.DataFrame(registry_records(specs)), output)
    save_json(
        make_manifest(settings, "registry", {"profile": args.profile, "parameters": [p.name for p in specs]}),
        output.with_suffix(".manifest.json"),
    )
    print(f"Saved {output} ({len(specs)} method parameters).")


def source_keys_for_spec(spec: ParameterSpec) -> list[str]:
    virtual = {
        "qifl_threshold_span": ["qifl_threshold_upper_bound"],
        "qifl_center_shift": ["qifl_threshold_centers"],
        "qifl_center_scale": ["qifl_threshold_centers"],
        "dqn_replay_to_batch_ratio": ["dqn_replay_capacity", "dqn_batch_size"],
        "dqn_reward_energy_to_deadline_ratio": [
            "dqn_reward_energy_weight", "dqn_reward_deadline_weight"
        ],
        "dqn_reward_travel_to_deadline_ratio": [
            "dqn_reward_travel_weight", "dqn_reward_deadline_weight"
        ],
        "dqn_reward_revisit_to_deadline_ratio": [
            "dqn_reward_revisit_penalty", "dqn_reward_deadline_weight"
        ],
        "dqn_reward_death_to_deadline_ratio": [
            "dqn_reward_death_penalty", "dqn_reward_deadline_weight"
        ],
        "dynamic_charging_wt_to_now_ratio": [
            "dynamic_charging_weight_wt", "dynamic_charging_weight_now"
        ],
        "dynamic_charging_eps_to_now_ratio": [
            "dynamic_charging_weight_eps", "dynamic_charging_weight_now"
        ],
    }
    return virtual.get(spec.name, [spec.name])


def cmd_audit(args: argparse.Namespace, settings: Mapping[str, Any]) -> None:
    specs = get_specs(args.profile)
    evaluator = WRSNEvaluator(settings, require_cuda=args.require_cuda, strict_determinism=True)
    # Audit the actual simulator modules in the project root.  Stage-2 helper
    # files are kept in a subpackage and must not be counted as evidence that a
    # simulator configuration key is active.
    sources = [
        path
        for path in PROJECT_ROOT.glob("*.py")
        if path.name not in {"test_wrsn_regression.py"}
    ]
    if not sources:
        raise RuntimeError(
            f"No simulator Python files were found in project root: {PROJECT_ROOT}"
        )
    source_text = {path.name: path.read_text(encoding="utf-8", errors="replace") for path in sources}
    rows = []
    for spec in specs:
        keys = source_keys_for_spec(spec)
        occurrences = {
            filename: sum(text.count(key) for key in keys)
            for filename, text in source_text.items()
        }
        occurrences = {name: count for name, count in occurrences.items() if count > 0}
        rows.append({
            **asdict(spec),
            "source_keys": stable_json(keys),
            "source_occurrences": stable_json(occurrences),
            "static_reference_count": sum(occurrences.values()),
            "baseline_values": stable_json({key: evaluator.base_config.get(key) for key in keys}),
            "locked": bool(set(keys) & (LOCKED_CONFIGURATION_KEYS | LOCKED_METHOD_SWITCHES)),
        })
    output = Path(args.output)
    audit_df = pd.DataFrame(rows)

    if args.dynamic:
        baseline = evaluator.evaluate(
            {}, args.audit_seed, tag="Stage2AuditBaseline",
            num_nodes=args.audit_nodes, simulation_time_s=args.audit_time,
        )
        dynamic_rows = []
        metrics = list(settings["selection_metrics"])
        for spec in specs:
            for label, unit_value in (("low", 0.0), ("high", 1.0)):
                values = {spec.name: spec.decode(unit_value)}
                try:
                    result = evaluator.evaluate(
                        values, args.audit_seed, tag=f"Stage2Audit_{spec.name}_{label}",
                        num_nodes=args.audit_nodes, simulation_time_s=args.audit_time,
                    )
                    changed = {}
                    for metric in metrics:
                        a = pd.to_numeric(pd.Series([baseline.get(metric)]), errors="coerce").iloc[0]
                        b = pd.to_numeric(pd.Series([result.get(metric)]), errors="coerce").iloc[0]
                        if pd.notna(a) and pd.notna(b):
                            changed[metric] = float(b - a)
                    dynamic_rows.append({
                        "name": spec.name,
                        "endpoint": label,
                        "status": "SUCCESS",
                        "observed_metric_deltas": stable_json(changed),
                        "any_observed_change": any(abs(v) > 1e-12 for v in changed.values()),
                    })
                except Exception as exc:
                    dynamic_rows.append({
                        "name": spec.name,
                        "endpoint": label,
                        "status": "ERROR",
                        "error": repr(exc),
                    })
        dynamic_path = output.with_name(output.stem + "_dynamic.csv")
        atomic_write_csv(pd.DataFrame(dynamic_rows), dynamic_path)
        print(f"Saved dynamic observability audit to {dynamic_path}.")

    atomic_write_csv(audit_df, output)
    save_json(
        make_manifest(settings, "audit", {"profile": args.profile, "dynamic": args.dynamic}),
        output.with_suffix(".manifest.json"),
    )
    print(f"Saved static parameter audit to {output}.")


def cmd_morris_generate(args: argparse.Namespace, settings: Mapping[str, Any]) -> None:
    specs = get_specs(args.profile)
    cfg = settings["morris"]
    candidate = int(args.candidate_trajectories or cfg["candidate_trajectories"])
    optimal = int(args.optimal_trajectories or cfg["optimal_trajectories"])
    levels = int(args.num_levels or cfg["num_levels"])
    # Repeat validation after resolving CLI overrides. SALib requires the
    # number of selected optimal trajectories to be strictly smaller than
    # the number of candidate trajectories.
    if candidate < 2:
        raise ValueError("candidate_trajectories must be at least 2")
    if optimal < 1:
        raise ValueError("optimal_trajectories must be at least 1")
    if optimal >= candidate:
        raise ValueError(
            "optimal_trajectories must be strictly less than "
            "candidate_trajectories"
        )
    if levels < 2:
        raise ValueError("num_levels must be at least 2")

    dependency_guard("SALib")
    from SALib.sample import morris

    X = morris.sample(
        unit_problem(specs),
        N=candidate,
        num_levels=levels,
        optimal_trajectories=optimal,
        local_optimization=True,
        seed=int(args.design_seed),
    )
    output = Path(args.output)
    atomic_write_csv(design_dataframe("morris", args.profile, specs, X), output)
    save_json(
        make_manifest(settings, "morris-generate", {
            "profile": args.profile,
            "candidate_trajectories": candidate,
            "optimal_trajectories": optimal,
            "num_levels": levels,
            "rows": int(len(X)),
        }),
        output.with_suffix(".manifest.json"),
    )
    print(f"Saved Morris design to {output}: {len(X)} configurations x {len(settings['sensitivity_seeds'])} seeds.")


def cmd_sobol_generate(args: argparse.Namespace, settings: Mapping[str, Any]) -> None:
    dependency_guard("SALib")
    from SALib.sample import sobol

    names = read_selected_names(Path(args.selected))
    specs = get_specs(names=names)
    ineligible = [p.name for p in specs if not p.sobol_eligible]
    if ineligible:
        raise ValueError(f"Sobol-ineligible parameters selected: {ineligible}")
    cfg = settings["sobol"]
    N = int(args.base_sample_n or cfg["base_sample_n"])
    if N <= 0 or (N & (N - 1)) != 0:
        raise ValueError("Sobol base_sample_n must be a positive power of two")
    second_order = bool(cfg["calc_second_order"] if args.second_order is None else args.second_order)
    X = sobol.sample(
        unit_problem(specs), N=N, calc_second_order=second_order,
        scramble=True, seed=int(args.design_seed),
    )
    output = Path(args.output)
    atomic_write_csv(design_dataframe("sobol", "selected", specs, X), output)
    save_json(
        make_manifest(settings, "sobol-generate", {
            "parameters": names, "base_sample_n": N,
            "calc_second_order": second_order, "rows": int(len(X)),
        }),
        output.with_suffix(".manifest.json"),
    )
    print(f"Saved Sobol design to {output}: {len(X)} configurations x {len(settings['sensitivity_seeds'])} seeds.")


def _load_existing(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame()


def cmd_evaluate_design(args: argparse.Namespace, settings: Mapping[str, Any]) -> None:
    design_path = Path(args.design)
    design = pd.read_csv(design_path)
    required = {"design_id", "method_values_json", "config_hash"}
    if not required.issubset(design.columns):
        raise ValueError(f"Design lacks columns: {sorted(required - set(design.columns))}")
    output = Path(args.output)
    existing = _load_existing(output)
    completed = set()
    if not existing.empty and {"design_id", "Seed", "Status"}.issubset(existing.columns):
        completed = {
            (str(row.design_id), int(row.Seed))
            for row in existing.itertuples()
            if str(row.Status) == "SUCCESS"
        }
    evaluator = WRSNEvaluator(
        settings, require_cuda=args.require_cuda,
        strict_determinism=not args.warn_only_determinism,
        quiet=not args.verbose_simulator,
    )
    seeds = [int(s) for s in settings[args.seed_group]]
    rows = existing.to_dict("records") if not existing.empty else []
    runs_done = 0
    for design_row in design.to_dict("records"):
        design_id = str(design_row["design_id"])
        method_values = json.loads(design_row["method_values_json"])
        for seed in seeds:
            if (design_id, seed) in completed:
                continue
            if args.max_runs is not None and runs_done >= int(args.max_runs):
                atomic_write_csv(pd.DataFrame(rows), output)
                print(f"Stopped after --max-runs={args.max_runs}; resume with the same command.")
                return
            print(f"[{runs_done + 1}] {design_id}, seed={seed}")
            log_path = None
            if args.keep_logs:
                log_path = output.parent / "logs" / f"{design_id}_seed{seed}.log"
            try:
                result = evaluator.evaluate(method_values, seed, tag=design_id, log_path=log_path)
                result["design_id"] = design_id
                result["design_method"] = design_row.get("method")
                result["design_profile"] = design_row.get("profile")
                rows.append(result)
                completed.add((design_id, seed))
            except Exception as exc:
                rows.append({
                    "design_id": design_id,
                    "Seed": seed,
                    "Status": "ERROR",
                    "Config_Hash": design_row.get("config_hash"),
                    "Method_Values_JSON": design_row.get("method_values_json"),
                    "Error": repr(exc),
                    "Traceback": traceback.format_exc(),
                })
                atomic_write_csv(pd.DataFrame(rows), output)
                if args.fail_fast:
                    raise
            runs_done += 1
            atomic_write_csv(pd.DataFrame(rows), output)
    save_json(
        make_manifest(settings, "evaluate-design", {
            "design": str(design_path), "seed_group": args.seed_group,
            "successful_runs": len(completed),
        }),
        output.with_suffix(".manifest.json"),
    )
    print(f"Saved/resumed evaluations in {output}; successful runs={len(completed)}.")


def aggregate_design_results(
    design: pd.DataFrame,
    evaluations: pd.DataFrame,
    metrics: Sequence[str],
    *,
    expected_seeds: Sequence[int],
) -> pd.DataFrame:
    """Aggregate paired-seed evaluations only after completeness validation.

    Morris and Sobol designs must use the same seed set for every design point.
    Silently averaging partial runs would break the paired design and bias the
    sensitivity indices, so analysis fails closed when any pair is missing.
    """
    required_columns = {"design_id", "Seed", "Status"}
    missing_columns = sorted(required_columns - set(evaluations.columns))
    if missing_columns:
        raise ValueError(f"Evaluation file lacks columns: {missing_columns}")

    success = evaluations[evaluations["Status"].astype(str) == "SUCCESS"].copy()
    if success.empty:
        raise ValueError("No successful evaluations")
    missing_metrics = [metric for metric in metrics if metric not in success.columns]
    if missing_metrics:
        raise ValueError(f"Evaluation file lacks metrics: {missing_metrics}")

    success["design_id"] = success["design_id"].astype(str)
    success["Seed"] = pd.to_numeric(success["Seed"], errors="raise").astype(int)
    # Keep the last successful record if an interrupted run was later repeated.
    success = success.drop_duplicates(["design_id", "Seed"], keep="last")

    design_ids = design["design_id"].astype(str).drop_duplicates().tolist()
    expected = {int(seed) for seed in expected_seeds}
    incomplete: dict[str, dict[str, list[int]]] = {}
    for design_id in design_ids:
        observed = set(
            success.loc[success["design_id"] == design_id, "Seed"].astype(int)
        )
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        if missing or extra:
            incomplete[design_id] = {"missing": missing, "extra": extra}
    if incomplete:
        preview = dict(list(incomplete.items())[:10])
        raise ValueError(
            "Sensitivity analysis requires a complete paired-seed matrix. "
            f"Incomplete design points (first 10): {preview}"
        )

    for metric in metrics:
        success[metric] = pd.to_numeric(success[metric], errors="coerce")
    grouped = success.groupby("design_id", sort=False)[list(metrics)].mean(
        numeric_only=True
    )
    ordered = design[["design_id"]].drop_duplicates().copy()
    ordered["design_id"] = ordered["design_id"].astype(str)
    merged = ordered.set_index("design_id").join(grouped, how="left").reset_index()
    return merged


def _unit_matrix(design: pd.DataFrame, specs: Sequence[ParameterSpec]) -> np.ndarray:
    columns = [f"u__{spec.name}" for spec in specs]
    missing = [col for col in columns if col not in design.columns]
    if missing:
        raise ValueError(f"Design lacks normalized parameter columns: {missing}")
    return design[columns].to_numpy(dtype=float)


def _select_by_scores(
    score_df: pd.DataFrame,
    *,
    importance_col: str,
    interaction_col: str | None,
    importance_threshold: float,
    interaction_threshold: float,
    min_selected: int,
    max_selected: int,
) -> list[str]:
    frame = score_df.copy()
    mask = frame[importance_col] >= float(importance_threshold)
    if interaction_col is not None:
        mask |= frame[interaction_col] >= float(interaction_threshold)
    selected = frame.loc[mask].sort_values(
        [importance_col] + ([interaction_col] if interaction_col else []), ascending=False
    )["name"].tolist()
    ranked = frame.sort_values(importance_col, ascending=False)["name"].tolist()
    for name in ranked:
        if len(selected) >= min_selected:
            break
        if name not in selected:
            selected.append(name)
    rank_position = {name: idx for idx, name in enumerate(ranked)}
    selected = sorted(set(selected), key=lambda name: rank_position.get(name, 10**9))
    return selected[:max_selected]


def cmd_morris_analyze(args: argparse.Namespace, settings: Mapping[str, Any]) -> None:
    dependency_guard("SALib")
    from SALib.analyze import morris

    design = pd.read_csv(args.design)
    evaluations = pd.read_csv(args.evaluations)
    profile = str(design["profile"].iloc[0])
    specs = get_specs(profile)
    metrics = list(settings["analysis_metrics"])
    aggregated = aggregate_design_results(
        design, evaluations, metrics,
        expected_seeds=settings["sensitivity_seeds"],
    )
    X = _unit_matrix(design, specs)
    if len(X) != len(aggregated):
        raise AssertionError("Design/result alignment failed")
    levels = int(settings["morris"]["num_levels"])
    resamples = int(settings["morris"]["bootstrap_resamples"])
    result_rows = []
    skipped = {}
    for metric in metrics:
        Y = aggregated[metric].to_numpy(dtype=float)
        if not np.all(np.isfinite(Y)):
            skipped[metric] = f"non-finite values at {np.where(~np.isfinite(Y))[0].tolist()[:20]}"
            continue
        if float(np.std(Y)) <= 1e-15:
            skipped[metric] = "constant output"
            continue
        indices = morris.analyze(
            unit_problem(specs), X, Y,
            num_resamples=resamples, conf_level=0.95,
            scaled=True, print_to_console=False,
            num_levels=levels, seed=int(args.analysis_seed),
        )
        mu_sum = max(float(np.nansum(indices["mu_star"])), 1e-15)
        sigma_sum = max(float(np.nansum(np.nan_to_num(indices["sigma"], nan=0.0))), 1e-15)
        for idx, spec in enumerate(specs):
            result_rows.append({
                "metric": metric,
                "name": spec.name,
                "group": spec.group,
                "mu": float(indices["mu"][idx]),
                "mu_star": float(indices["mu_star"][idx]),
                "sigma": float(indices["sigma"][idx]),
                "mu_star_conf": float(indices["mu_star_conf"][idx]),
                "mu_star_fraction": float(indices["mu_star"][idx]) / mu_sum,
                "sigma_fraction": float(np.nan_to_num(indices["sigma"][idx], nan=0.0)) / sigma_sum,
            })
    if not result_rows:
        raise ValueError(f"No analyzable Morris metrics. Skipped: {skipped}")
    results = pd.DataFrame(result_rows)
    output = Path(args.output)
    atomic_write_csv(results, output)

    selection_metrics = set(settings["selection_metrics"])
    evidence = results[results["metric"].isin(selection_metrics)]
    scores = evidence.groupby("name").agg(
        max_mu_star_fraction=("mu_star_fraction", "max"),
        mean_mu_star_fraction=("mu_star_fraction", "mean"),
        max_sigma_fraction=("sigma_fraction", "max"),
    ).reset_index()
    cfg = settings["morris"]
    selected = _select_by_scores(
        scores,
        importance_col="max_mu_star_fraction",
        interaction_col="max_sigma_fraction",
        importance_threshold=float(cfg["selection_mu_star_fraction"]),
        interaction_threshold=float(cfg["selection_sigma_fraction"]),
        min_selected=int(cfg["min_selected"]),
        max_selected=int(cfg["max_selected"]),
    )
    selected_path = Path(args.selected_output)
    save_json({
        "method": "Morris",
        "profile": profile,
        "selected_parameters": selected,
        "selection_rule": {
            "mu_star_fraction": cfg["selection_mu_star_fraction"],
            "sigma_fraction": cfg["selection_sigma_fraction"],
            "min_selected": cfg["min_selected"],
            "max_selected": cfg["max_selected"],
        },
        "scores": scores.to_dict("records"),
        "skipped_metrics": skipped,
    }, selected_path)
    atomic_write_csv(scores, output.with_name(output.stem + "_aggregate_scores.csv"))
    save_json(
        make_manifest(settings, "morris-analyze", {
            "profile": profile, "selected": selected, "skipped_metrics": skipped,
        }), output.with_suffix(".manifest.json")
    )
    print(f"Saved Morris indices to {output}; selected {len(selected)} parameters in {selected_path}.")


def cmd_merge_selected(args: argparse.Namespace, settings: Mapping[str, Any]) -> None:
    names: list[str] = []
    sources = []
    evidence_by_name: dict[str, dict[str, float]] = {}
    valid = {p.name for p in PARAMETERS}
    for item in args.inputs:
        path = Path(item)
        payload = load_json(path)
        source_names = read_selected_names(path)
        unknown = sorted(set(source_names) - valid)
        if unknown:
            raise ValueError(f"Unknown parameters in {path}: {unknown}")
        sources.append({"path": str(path), "parameters": source_names})
        for name in source_names:
            if name not in names:
                names.append(name)
        for score in payload.get("scores", []):
            name = str(score.get("name", ""))
            if name not in source_names:
                continue
            importance = float(score.get("max_mu_star_fraction", score.get("max_ST", 0.0)) or 0.0)
            interaction = float(score.get("max_sigma_fraction", score.get("max_interaction_gap", 0.0)) or 0.0)
            current = evidence_by_name.setdefault(name, {"importance": 0.0, "interaction": 0.0})
            current["importance"] = max(current["importance"], importance)
            current["interaction"] = max(current["interaction"], interaction)

    if evidence_by_name:
        names = sorted(
            names,
            key=lambda name: (
                evidence_by_name.get(name, {}).get("importance", 0.0),
                evidence_by_name.get(name, {}).get("interaction", 0.0),
            ),
            reverse=True,
        )
    max_parameters = int(args.max_parameters)
    if max_parameters > 0:
        names = names[:max_parameters]
    save_json({
        "method": "Morris profile merge with predeclared cross-profile cap",
        "selected_parameters": names,
        "max_parameters": max_parameters,
        "combined_evidence": evidence_by_name,
        "sources": sources,
    }, Path(args.output))
    print(f"Merged {len(args.inputs)} files into {args.output}; retained parameters={len(names)}.")


def cmd_sobol_analyze(args: argparse.Namespace, settings: Mapping[str, Any]) -> None:
    dependency_guard("SALib")
    from SALib.analyze import sobol

    design = pd.read_csv(args.design)
    evaluations = pd.read_csv(args.evaluations)
    param_names = [col[3:] for col in design.columns if col.startswith("u__")]
    specs = get_specs(names=param_names)
    metrics = list(settings["analysis_metrics"])
    aggregated = aggregate_design_results(
        design, evaluations, metrics,
        expected_seeds=settings["sensitivity_seeds"],
    )
    second_order = bool(settings["sobol"]["calc_second_order"])
    resamples = int(settings["sobol"]["bootstrap_resamples"])
    rows = []
    skipped = {}
    for metric in metrics:
        Y = aggregated[metric].to_numpy(dtype=float)
        if not np.all(np.isfinite(Y)):
            skipped[metric] = f"non-finite values at {np.where(~np.isfinite(Y))[0].tolist()[:20]}"
            continue
        if float(np.std(Y)) <= 1e-15:
            skipped[metric] = "constant output"
            continue
        indices = sobol.analyze(
            unit_problem(specs), Y,
            calc_second_order=second_order,
            num_resamples=resamples, conf_level=0.95,
            print_to_console=False, seed=int(args.analysis_seed),
        )
        for idx, spec in enumerate(specs):
            rows.append({
                "metric": metric,
                "name": spec.name,
                "group": spec.group,
                "S1": float(indices["S1"][idx]),
                "S1_conf": float(indices["S1_conf"][idx]),
                "ST": float(indices["ST"][idx]),
                "ST_conf": float(indices["ST_conf"][idx]),
                "interaction_gap": float(indices["ST"][idx] - indices["S1"][idx]),
            })
    if not rows:
        raise ValueError(f"No analyzable Sobol metrics. Skipped: {skipped}")
    results = pd.DataFrame(rows)
    output = Path(args.output)
    atomic_write_csv(results, output)

    evidence = results[results["metric"].isin(set(settings["selection_metrics"]))].copy()
    evidence["ST_positive"] = evidence["ST"].clip(lower=0.0)
    scores = evidence.groupby("name").agg(
        max_ST=("ST_positive", "max"),
        mean_ST=("ST_positive", "mean"),
        max_interaction_gap=("interaction_gap", "max"),
    ).reset_index().sort_values("max_ST", ascending=False)
    cfg = settings["sobol"]
    selected = scores.loc[
        scores["max_ST"] >= float(cfg["selection_total_effect"]), "name"
    ].tolist()
    if not selected:
        selected = scores.head(min(4, len(scores)))["name"].tolist()
    selected = selected[:int(cfg["max_selected"])]
    save_json({
        "method": "Sobol total-effect selection",
        "selected_parameters": selected,
        "selection_rule": {
            "total_effect_threshold": cfg["selection_total_effect"],
            "max_selected": cfg["max_selected"],
        },
        "scores": scores.to_dict("records"),
        "skipped_metrics": skipped,
    }, Path(args.selected_output))
    atomic_write_csv(scores, output.with_name(output.stem + "_aggregate_scores.csv"))
    save_json(
        make_manifest(settings, "sobol-analyze", {
            "parameters": param_names, "selected": selected, "skipped_metrics": skipped,
        }), output.with_suffix(".manifest.json")
    )
    print(f"Saved Sobol indices to {output}; selected {len(selected)} parameters in {args.selected_output}.")


def _cached_evaluation(
    evaluator: WRSNEvaluator,
    cache_path: Path,
    method_values: Mapping[str, Any],
    seed: int,
    tag: str,
) -> dict[str, Any]:
    overrides = method_values_to_overrides(method_values)
    key_hash = config_hash(overrides)
    existing = _load_existing(cache_path)
    if not existing.empty and {"Config_Hash", "Seed", "Status"}.issubset(existing.columns):
        match = existing[
            (existing["Config_Hash"].astype(str) == key_hash)
            & (pd.to_numeric(existing["Seed"], errors="coerce") == int(seed))
            & (existing["Status"].astype(str) == "SUCCESS")
        ]
        if not match.empty:
            return match.iloc[-1].to_dict()
    result = evaluator.evaluate(method_values, int(seed), tag=tag)
    rows = existing.to_dict("records") if not existing.empty else []
    rows.append(result)
    atomic_write_csv(pd.DataFrame(rows), cache_path)
    return result


def ensure_baseline(
    evaluator: WRSNEvaluator,
    settings: Mapping[str, Any],
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, float], dict[int, dict[str, float]]]:
    cache_path = output_dir / "baseline_optimization_evaluations.csv"
    rows = []
    for seed in settings["optimization_seeds"]:
        rows.append(_cached_evaluation(evaluator, cache_path, {}, int(seed), "Stage2OptimizationBaseline"))
    frame = pd.DataFrame(rows)
    numeric_metrics = [
        "MC_Deadline_Failure_Rate", "MC_Total_Mission_Energy", "Critical_PDR",
        "MCC", "Overall_PDR", "Avg_Delay", "MC_Fail_Closed_Rejections",
        "BS_Execution_Time_ms",
    ]
    baseline = {}
    for metric in numeric_metrics:
        if metric not in frame:
            raise ValueError(f"Baseline output lacks {metric}")
        values = pd.to_numeric(frame[metric], errors="coerce")
        if not np.any(np.isfinite(values)):
            raise ValueError(f"Baseline metric {metric} is entirely non-finite")
        baseline[metric] = float(np.nanmean(values))
    by_seed: dict[int, dict[str, float]] = {}
    for row in frame.to_dict("records"):
        seed = int(row["Seed"])
        by_seed[seed] = {
            metric: float(pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0])
            for metric in numeric_metrics
        }
    save_json(
        {"means": baseline, "by_seed": by_seed, "seeds": settings["optimization_seeds"]},
        output_dir / "baseline_optimization_metrics.json",
    )
    return frame, baseline, by_seed


def objective_costs(
    metrics: Mapping[str, Any],
    baseline: Mapping[str, float],
    tuning: Mapping[str, Any],
) -> tuple[dict[str, float], bool, dict[str, float]]:
    def finite(name: str, fallback: float) -> float:
        try:
            value = float(metrics[name])
        except Exception:
            return float(fallback)
        return value if math.isfinite(value) else float(fallback)

    deadline = float(np.clip(finite("MC_Deadline_Failure_Rate", 1.0), 0.0, 1.0))
    energy = max(0.0, finite("MC_Total_Mission_Energy", baseline["MC_Total_Mission_Energy"] * 10.0))
    energy_ratio = energy / max(float(baseline["MC_Total_Mission_Energy"]), 1e-12)
    critical_pdr_loss = 1.0 - float(np.clip(finite("Critical_PDR", 0.0), 0.0, 100.0)) / 100.0
    mcc_loss = (1.0 - float(np.clip(finite("MCC", -1.0), -1.0, 1.0))) / 2.0
    costs = {
        "deadline_failure": deadline,
        "mission_energy_ratio": energy_ratio,
        "critical_pdr_loss": critical_pdr_loss,
        "mcc_loss": mcc_loss,
    }

    cfg = tuning["constraints"]
    violations = {
        "overall_pdr": max(
            0.0,
            baseline["Overall_PDR"] - float(cfg["overall_pdr_drop_percentage_points"])
            - finite("Overall_PDR", -1e9),
        ) / 100.0,
        "mcc": max(
            0.0,
            baseline["MCC"] - float(cfg["mcc_drop"]) - finite("MCC", -1.0),
        ),
        "fail_closed": max(
            0.0,
            finite("MC_Fail_Closed_Rejections", 1e9)
            - baseline["MC_Fail_Closed_Rejections"]
            - float(cfg["max_fail_closed_increase"]),
        ),
    }
    feasible = all(value <= 1e-12 for value in violations.values())
    if not feasible:
        penalty = float(tuning["constraint_penalty"]) + sum(violations.values())
        costs = {key: float(value + penalty) for key, value in costs.items()}
    return costs, feasible, violations


def build_configspace(specs: Sequence[ParameterSpec], seed: int):
    dependency_guard("ConfigSpace")
    from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer

    cs = ConfigurationSpace(seed=int(seed))
    hyperparameters = []
    for spec in specs:
        if spec.kind == "choice":
            hp = Categorical(spec.name, list(spec.choices), default=spec.default)
        elif spec.kind == "int":
            hp = Integer(
                spec.name, (int(spec.low), int(spec.high)),
                default=int(spec.default), log=(spec.scale == "log"),
            )
        else:
            hp = Float(
                spec.name, (float(spec.low), float(spec.high)),
                default=float(spec.default), log=(spec.scale == "log"),
            )
        hyperparameters.append(hp)
    cs.add(hyperparameters)
    return cs


def nondominated_mask(costs: np.ndarray) -> np.ndarray:
    n = len(costs)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if np.all(costs[j] <= costs[i]) and np.any(costs[j] < costs[i]):
                keep[i] = False
                break
    return keep


def summarize_tuning_cache(
    cache_path: Path,
    baseline: Mapping[str, float],
    settings: Mapping[str, Any],
) -> pd.DataFrame:
    raw = pd.read_csv(cache_path)
    raw = raw[raw["Status"].astype(str) == "SUCCESS"].copy()
    if raw.empty:
        raise ValueError("No successful tuning evaluations")
    objective_names = list(settings["tuning"]["objectives"])
    records = []
    for cfg_hash, group in raw.groupby("Config_Hash", sort=False):
        metric_means = {}
        for metric in (
            "MC_Deadline_Failure_Rate", "MC_Total_Mission_Energy", "Critical_PDR",
            "MCC", "Overall_PDR", "Avg_Delay", "MC_Fail_Closed_Rejections",
            "BS_Execution_Time_ms",
        ):
            metric_means[metric] = float(np.nanmean(pd.to_numeric(group[metric], errors="coerce")))
        costs, feasible, violations = objective_costs(metric_means, baseline, settings["tuning"])
        record = {
            "config_hash": cfg_hash,
            "n_seed_evaluations": int(group["Seed"].nunique()),
            "method_values_json": group["Method_Values_JSON"].iloc[-1],
            "config_overrides_json": group["Config_Overrides_JSON"].iloc[-1],
            "feasible": bool(feasible),
            **metric_means,
            **{f"cost__{name}": float(costs[name]) for name in objective_names},
            **{f"violation__{name}": float(value) for name, value in violations.items()},
        }
        records.append(record)
    frame = pd.DataFrame(records)
    feasible = frame[frame["feasible"] == True].copy()  # noqa: E712
    candidate = feasible if not feasible.empty else frame
    cost_cols = [f"cost__{name}" for name in objective_names]
    mask = nondominated_mask(candidate[cost_cols].to_numpy(dtype=float))
    pareto_hashes = set(candidate.loc[mask, "config_hash"].astype(str))
    frame["pareto"] = frame["config_hash"].astype(str).isin(pareto_hashes)
    return frame


def select_configuration(summary: pd.DataFrame) -> pd.Series:
    candidate = summary[(summary["feasible"] == True) & (summary["pareto"] == True)].copy()  # noqa: E712
    if candidate.empty:
        candidate = summary[summary["pareto"] == True].copy()  # noqa: E712
    if candidate.empty:
        candidate = summary.copy()
    # Predeclared lexicographic choice: deadline, energy, critical PDR, MCC, BS time.
    candidate = candidate.sort_values(
        [
            "MC_Deadline_Failure_Rate", "MC_Total_Mission_Energy",
            "Critical_PDR", "MCC", "BS_Execution_Time_ms",
        ],
        ascending=[True, True, False, False, True],
    )
    return candidate.iloc[0]


def cmd_tune(args: argparse.Namespace, settings: Mapping[str, Any]) -> None:
    dependency_guard("ConfigSpace", "smac")
    from smac import HyperparameterOptimizationFacade as HPOFacade
    from smac import Scenario
    from smac.multi_objective.parego import ParEGO

    selected_names = read_selected_names(Path(args.selected))
    specs = get_specs(names=selected_names)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = WRSNEvaluator(
        settings, require_cuda=args.require_cuda,
        strict_determinism=not args.warn_only_determinism,
        quiet=not args.verbose_simulator,
    )
    _, baseline, baseline_by_seed = ensure_baseline(evaluator, settings, output_dir)
    cache_path = output_dir / "tuning_evaluations.csv"
    configspace = build_configspace(specs, int(args.optimizer_seed))
    tuning = settings["tuning"]
    objectives = list(tuning["objectives"])
    instances = [str(seed) for seed in settings["optimization_seeds"]]

    def target(config, instance: str, seed: int = 0) -> dict[str, float]:
        del seed  # The fixed instance name is the scientific simulation seed.
        method_values = dict(config)
        simulation_seed = int(instance)
        result = _cached_evaluation(
            evaluator, cache_path, method_values, simulation_seed,
            tag=f"Stage2Tune_{config_hash(method_values_to_overrides(method_values))}",
        )
        costs, feasible, violations = objective_costs(
            result, baseline_by_seed[simulation_seed], tuning
        )
        print(
            f"[SMAC] cfg={result['Config_Hash']} seed={simulation_seed} "
            f"feasible={feasible} costs={costs} violations={violations}"
        )
        return {name: float(costs[name]) for name in objectives}

    scenario = Scenario(
        configspace,
        name="wrsn_parego",
        objectives=objectives,
        instances=instances,
        deterministic=True,
        n_trials=int(args.n_trials or tuning["n_trials"]),
        walltime_limit=float(args.walltime_limit or tuning["walltime_limit_s"]),
        n_workers=1,
        output_directory=output_dir / "smac_output",
        seed=int(args.optimizer_seed),
    )
    initial_design = HPOFacade.get_initial_design(
        scenario, n_configs=int(tuning["initial_design_configs"])
    )
    intensifier = HPOFacade.get_intensifier(
        scenario,
        max_config_calls=min(int(tuning["max_config_calls"]), len(instances)),
    )
    smac = HPOFacade(
        scenario,
        target,
        initial_design=initial_design,
        multi_objective_algorithm=ParEGO(scenario),
        intensifier=intensifier,
        overwrite=not args.resume,
    )
    incumbents = smac.optimize()

    summary = summarize_tuning_cache(cache_path, baseline, settings)
    atomic_write_csv(summary, output_dir / "tuning_configuration_summary.csv")
    pareto = summary[summary["pareto"] == True].copy()  # noqa: E712
    atomic_write_csv(pareto, output_dir / "pareto_front.csv")
    selected = select_configuration(summary)
    selected_payload = {
        "method": "SMAC 2.4 / ParEGO multi-objective configuration",
        "selected_parameters": selected_names,
        "config_hash": selected["config_hash"],
        "method_values": json.loads(selected["method_values_json"]),
        "config_overrides": json.loads(selected["config_overrides_json"]),
        "feasible": bool(selected["feasible"]),
        "selection_rule": [
            "minimum mean MC_Deadline_Failure_Rate",
            "then minimum mean MC_Total_Mission_Energy",
            "then maximum mean Critical_PDR",
            "then maximum mean MCC",
            "then minimum mean BS_Execution_Time_ms",
        ],
        "mean_metrics": {
            key: float(selected[key])
            for key in (
                "MC_Deadline_Failure_Rate", "MC_Total_Mission_Energy", "Critical_PDR",
                "MCC", "Overall_PDR", "Avg_Delay", "MC_Fail_Closed_Rejections",
                "BS_Execution_Time_ms",
            )
        },
        "smac_incumbent_count": len(incumbents) if isinstance(incumbents, list) else 1,
    }
    save_json(selected_payload, output_dir / "selected_configuration.json")
    save_json(
        make_manifest(settings, "tune", {
            "selected_parameters": selected_names,
            "optimizer_seed": args.optimizer_seed,
            "selected_config_hash": selected["config_hash"],
        }), output_dir / "tuning_manifest.json"
    )
    print(f"Saved Pareto front and selected configuration in {output_dir}.")


def cmd_final_evaluate(args: argparse.Namespace, settings: Mapping[str, Any]) -> None:
    selected_payload = load_json(Path(args.selected_configuration))
    method_values = selected_payload.get("method_values")
    if not isinstance(method_values, dict):
        raise ValueError("selected_configuration.json lacks method_values")
    evaluator = WRSNEvaluator(
        settings, require_cuda=args.require_cuda,
        strict_determinism=not args.warn_only_determinism,
        quiet=not args.verbose_simulator,
    )
    output = Path(args.output)
    existing = _load_existing(output)
    completed = set()
    if not existing.empty and {"Seed", "Status"}.issubset(existing.columns):
        completed = {
            int(row.Seed) for row in existing.itertuples()
            if str(row.Status) == "SUCCESS"
        }
    rows = existing.to_dict("records") if not existing.empty else []
    for seed in settings["final_evaluation_seeds"]:
        seed = int(seed)
        if seed in completed:
            continue
        print(f"[FINAL] seed={seed}")
        result = evaluator.evaluate(method_values, seed, tag="Stage2SelectedConfiguration")
        rows.append(result)
        completed.add(seed)
        atomic_write_csv(pd.DataFrame(rows), output)
    frame = pd.DataFrame(rows)
    numeric = frame.select_dtypes(include=[np.number])
    summary_rows = []
    for column in numeric.columns:
        values = pd.to_numeric(numeric[column], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        summary_rows.append({
            "metric": column,
            "n": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        })
    atomic_write_csv(pd.DataFrame(summary_rows), output.with_name(output.stem + "_summary.csv"))
    save_json(
        make_manifest(settings, "final-evaluate", {
            "selected_configuration": str(args.selected_configuration),
            "completed_seeds": sorted(completed),
        }), output.with_suffix(".manifest.json")
    )
    print(f"Saved final independent-seed results to {output}.")


def add_common_runtime_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--require-cuda", action="store_true", help="Fail closed unless QIFL and DQN both run on CUDA.")
    parser.add_argument("--warn-only-determinism", action="store_true", help="Use warn-only rather than strict PyTorch determinism.")
    parser.add_argument("--verbose-simulator", action="store_true", help="Do not suppress simulator stdout/stderr.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage-2 global sensitivity analysis and systematic tuning for the WRSN method."
    )
    parser.add_argument("--settings", default=str(DEFAULT_SETTINGS))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("registry")
    p.add_argument("--profile", default="qifl_dqn", choices=("qifl_dqn", "charging_policy", "routing", "charging_all", "all"))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT / "parameter_registry.csv"))

    p = sub.add_parser("audit")
    p.add_argument("--profile", default="qifl_dqn", choices=("qifl_dqn", "charging_policy", "routing", "charging_all", "all"))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT / "parameter_audit.csv"))
    p.add_argument("--dynamic", action="store_true")
    p.add_argument("--audit-seed", type=int, default=101)
    p.add_argument("--audit-nodes", type=int, default=30)
    p.add_argument("--audit-time", type=int, default=300)
    add_common_runtime_flags(p)

    p = sub.add_parser("morris-generate")
    p.add_argument("--profile", default="qifl_dqn", choices=("qifl_dqn", "charging_policy", "routing", "charging_all", "all"))
    p.add_argument("--candidate-trajectories", type=int)
    p.add_argument("--optimal-trajectories", type=int)
    p.add_argument("--num-levels", type=int)
    p.add_argument("--design-seed", type=int, default=7301)
    p.add_argument("--output", required=True)

    p = sub.add_parser("sobol-generate")
    p.add_argument("--selected", required=True)
    p.add_argument("--base-sample-n", type=int)
    p.add_argument("--second-order", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--design-seed", type=int, default=7302)
    p.add_argument("--output", required=True)

    p = sub.add_parser("evaluate-design")
    p.add_argument("--design", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed-group", default="sensitivity_seeds", choices=("sensitivity_seeds", "optimization_seeds", "final_evaluation_seeds"))
    p.add_argument("--max-runs", type=int)
    p.add_argument("--keep-logs", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    add_common_runtime_flags(p)

    p = sub.add_parser("morris-analyze")
    p.add_argument("--design", required=True)
    p.add_argument("--evaluations", required=True)
    p.add_argument("--analysis-seed", type=int, default=8301)
    p.add_argument("--output", required=True)
    p.add_argument("--selected-output", required=True)

    p = sub.add_parser("merge-selected")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--max-parameters", type=int, default=10)
    p.add_argument("--output", required=True)

    p = sub.add_parser("sobol-analyze")
    p.add_argument("--design", required=True)
    p.add_argument("--evaluations", required=True)
    p.add_argument("--analysis-seed", type=int, default=8302)
    p.add_argument("--output", required=True)
    p.add_argument("--selected-output", required=True)

    p = sub.add_parser("tune")
    p.add_argument("--selected", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--optimizer-seed", type=int, default=9301)
    p.add_argument("--n-trials", type=int)
    p.add_argument("--walltime-limit", type=float)
    p.add_argument("--resume", action="store_true")
    add_common_runtime_flags(p)

    p = sub.add_parser("final-evaluate")
    p.add_argument("--selected-configuration", required=True)
    p.add_argument("--output", required=True)
    add_common_runtime_flags(p)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = load_settings(Path(args.settings))
    commands = {
        "registry": cmd_registry,
        "audit": cmd_audit,
        "morris-generate": cmd_morris_generate,
        "sobol-generate": cmd_sobol_generate,
        "evaluate-design": cmd_evaluate_design,
        "morris-analyze": cmd_morris_analyze,
        "merge-selected": cmd_merge_selected,
        "sobol-analyze": cmd_sobol_analyze,
        "tune": cmd_tune,
        "final-evaluate": cmd_final_evaluate,
    }
    commands[args.command](args, settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
