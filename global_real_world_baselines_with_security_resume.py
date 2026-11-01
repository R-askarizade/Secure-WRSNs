"""Baseline-only Real-World runner with integrated security resume.

Purpose
-------
Continue the existing common-baseline experiment WITHOUT re-running completed
Method×Seed work unnecessarily.

For every baseline Method×Seed:
- Common performance result present, security not applicable -> SKIP.
- Security-aware method:
    performance present + security present -> SKIP
    performance present + security missing -> run once, save SECURITY only
    performance missing + security present -> run once, save PERFORMANCE only
    both missing -> run once, save BOTH

The same simulation instance is used to produce both performance and security
outputs whenever both are needed.

This file does NOT modify sim_phase.py, simulator_manager.py,
global_real_world.py, global_comparison_metrics.py, metric_collector.py, or any
baseline adapter. The proposed full method is explicitly excluded, and methods
commented out in global_real_world.build_method_registry() remain commented out.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Mapping, Set, Tuple

import pandas as pd

import global_real_world as grw
from global_comparison_metrics import (
    COMMON_METRICS_VERSION,
    PRIMARY_COMPARISON_METRICS,
    attach_global_comparison_metrics,
    select_primary_comparison_table,
    summarize_global_comparison_results,
)
from global_security_metrics import (
    COMMON_SECURITY_METRICS_VERSION,
    LEGACY_SECURITY_METRICS_VERSIONS,
    PRIMARY_SECURITY_COMPARISON_METRICS,
    export_common_security_row,
    select_primary_security_table,
    summarize_common_security_results,
    upgrade_legacy_common_security_row,
)

PROPOSED_LABEL = "Proposed_Final_RealWorld_AdaptiveMultiMC"
SECURITY_PROTOCOL_VERSION = "RealWorld_CommonSecurity_v3_Direct_MetricsCollector"


def _atomic_write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + ".tmp")
    df.to_csv(temp, index=False)
    os.replace(temp, path)


def _is_security_aware(spec) -> bool:
    return "security" in str(spec.comparison_domain).strip().lower()


def _load_valid_performance_rows(
    path: Path,
    *,
    registry,
    seeds: Tuple[int, ...],
    scenario_name: str,
    num_nodes: int,
    network_size: Tuple[float, float],
    canonical_digest: str,
    method_digests: Mapping[str, str],
) -> Tuple[List[Dict[str, object]], Set[Tuple[str, int]]]:
    if not path.exists() or path.stat().st_size <= 0:
        return [], set()

    existing = pd.read_csv(path)
    required = {
        "Method",
        "Seed",
        "Global_Protocol_Version",
        "Global_Source_Scenario",
        "Global_Node_Count",
        "Global_Network_Width_m",
        "Global_Network_Height_m",
        "Common_Metrics_Version",
        "Global_Canonical_Config_SHA256",
        "Method_Config_SHA256",
    }
    if not required.issubset(existing.columns):
        raise RuntimeError(
            f"Existing performance file does not match the global protocol: {path}"
        )

    labels = {spec.label for spec in registry}
    rows: List[Dict[str, object]] = []
    completed: Set[Tuple[str, int]] = set()

    for _, record in existing.iterrows():
        try:
            label = str(record["Method"])
            seed = int(record["Seed"])
            if label not in labels or seed not in seeds:
                continue
            if str(record["Global_Protocol_Version"]) != grw.PROTOCOL_VERSION:
                continue
            if str(record["Common_Metrics_Version"]) != COMMON_METRICS_VERSION:
                continue
            if str(record["Global_Source_Scenario"]) != scenario_name:
                continue
            if int(record["Global_Node_Count"]) != num_nodes:
                continue
            if float(record["Global_Network_Width_m"]) != network_size[0]:
                continue
            if float(record["Global_Network_Height_m"]) != network_size[1]:
                continue
            if str(record["Global_Canonical_Config_SHA256"]) != canonical_digest:
                continue
            if str(record["Method_Config_SHA256"]) != method_digests[label]:
                continue
        except (TypeError, ValueError, KeyError):
            continue

        spec = next(item for item in registry if item.label == label)
        row = grw.apply_metric_applicability(record.to_dict(), spec)
        rows.append(row)
        completed.add((label, seed))

    return rows, completed


def _load_valid_security_rows(
    path: Path,
    *,
    security_registry,
    seeds: Tuple[int, ...],
    scenario_name: str,
    num_nodes: int,
    network_size: Tuple[float, float],
    canonical_digest: str,
    method_digests: Mapping[str, str],
) -> Tuple[List[Dict[str, object]], Set[Tuple[str, int]]]:
    if not path.exists() or path.stat().st_size <= 0:
        return [], set()

    existing = pd.read_csv(path)
    required = {
        "Method",
        "Seed",
        "Common_Security_Metrics_Version",
        "Security_Protocol_Version",
        "Global_Source_Scenario",
        "Global_Node_Count",
        "Global_Network_Width_m",
        "Global_Network_Height_m",
        "Global_Canonical_Config_SHA256",
        "Method_Config_SHA256",
    }
    if not required.issubset(existing.columns):
        raise RuntimeError(
            f"Existing security file does not match the security protocol: {path}"
        )

    labels = {spec.label for spec in security_registry}
    rows: List[Dict[str, object]] = []
    completed: Set[Tuple[str, int]] = set()

    for _, record in existing.iterrows():
        try:
            label = str(record["Method"])
            seed = int(record["Seed"])
            if label not in labels or seed not in seeds:
                continue
            security_metric_version = str(record["Common_Security_Metrics_Version"])
            if (
                security_metric_version != COMMON_SECURITY_METRICS_VERSION
                and security_metric_version not in LEGACY_SECURITY_METRICS_VERSIONS
            ):
                continue
            if str(record["Security_Protocol_Version"]) != SECURITY_PROTOCOL_VERSION:
                continue
            if str(record["Global_Source_Scenario"]) != scenario_name:
                continue
            if int(record["Global_Node_Count"]) != num_nodes:
                continue
            if float(record["Global_Network_Width_m"]) != network_size[0]:
                continue
            if float(record["Global_Network_Height_m"]) != network_size[1]:
                continue
            if str(record["Global_Canonical_Config_SHA256"]) != canonical_digest:
                continue
            if str(record["Method_Config_SHA256"]) != method_digests[label]:
                continue
        except (TypeError, ValueError, KeyError):
            continue

        loaded_row = record.to_dict()
        if security_metric_version != COMMON_SECURITY_METRICS_VERSION:
            loaded_row = upgrade_legacy_common_security_row(loaded_row)
            print(
                f"[SECURITY MIGRATE] {label}; seed={seed}; "
                f"{security_metric_version} -> {COMMON_SECURITY_METRICS_VERSION}"
            )
        rows.append(loaded_row)
        completed.add((label, seed))

    return rows, completed


def _checkpoint_security_outputs(
    security_rows: List[Dict[str, object]],
    *,
    results_csv: Path,
    summary_csv: Path,
    primary_csv: Path,
) -> None:
    if not security_rows:
        return
    df = pd.DataFrame(security_rows)
    _atomic_write_dataframe(df, results_csv)
    summary = summarize_common_security_results(df, summary_csv)
    primary = select_primary_security_table(summary)
    _atomic_write_dataframe(primary, primary_csv)


def run_baselines_with_security_resume():
    scenario_name, canonical = grw.load_canonical_real_world()

    # EXACT same locked evaluation conditions as global_real_world.py.
    num_nodes = int(grw.DEFAULT_NUM_NODES)
    network_size = (
        float(grw.DEFAULT_NETWORK_SIZE[0]),
        float(grw.DEFAULT_NETWORK_SIZE[1]),
    )
    seeds = tuple(int(seed) for seed in grw.DEFAULT_SEEDS)
    sim_time = int(float(canonical["simulation_time_override_s"]))

    output_dir = Path(
        os.environ.get(
            "WRSN_BASELINE_OUTPUT_DIR",
            "realworld_common_baseline_results",
        )
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = output_dir / "shared_workload_traces"

    # Existing common-performance files: SAME filenames as the running baseline.
    performance_results_csv = output_dir / "realworld_common_metrics_results.csv"
    performance_summary_csv = output_dir / "realworld_common_metrics_summary_ci.csv"
    performance_primary_csv = output_dir / "realworld_common_primary_comparison.csv"
    performance_manifest_json = output_dir / "realworld_common_manifest.json"

    # Separate security track in the SAME experiment directory.
    security_results_csv = output_dir / "realworld_common_security_metrics_results.csv"
    security_summary_csv = output_dir / "realworld_common_security_metrics_summary_ci.csv"
    security_primary_csv = output_dir / "realworld_common_security_primary_comparison.csv"
    security_manifest_json = output_dir / "realworld_common_security_manifest.json"
    security_per_method_dir = output_dir / "security_per_method"

    # Keep exactly the methods currently active in build_method_registry(),
    # but ALWAYS exclude the full proposed method in this baseline-only runner.
    registry = [
        spec for spec in grw.build_method_registry()
        if spec.label != PROPOSED_LABEL
    ]
    security_registry = [
        spec for spec in registry if _is_security_aware(spec)
    ]

    labels = [spec.label for spec in registry]
    if len(labels) != len(set(labels)):
        raise RuntimeError("Duplicate method labels in baseline registry.")

    # Build and audit exactly as the original global runner does.
    built_configs: Dict[str, Dict[str, object]] = {}
    audit_changes: Dict[str, List[str]] = {}
    for spec in registry:
        cfg = spec.config_builder(canonical)
        observed = grw.audit_method_config(spec, canonical, cfg)
        built_configs[spec.label] = cfg
        audit_changes[spec.label] = sorted(observed)

    canonical_digest = grw._config_digest(canonical)
    method_digests = {
        label: grw._config_digest(config)
        for label, config in built_configs.items()
    }

    performance_rows, performance_completed = _load_valid_performance_rows(
        performance_results_csv,
        registry=registry,
        seeds=seeds,
        scenario_name=scenario_name,
        num_nodes=num_nodes,
        network_size=network_size,
        canonical_digest=canonical_digest,
        method_digests=method_digests,
    )
    security_rows, security_completed = _load_valid_security_rows(
        security_results_csv,
        security_registry=security_registry,
        seeds=seeds,
        scenario_name=scenario_name,
        num_nodes=num_nodes,
        network_size=network_size,
        canonical_digest=canonical_digest,
        method_digests=method_digests,
    )

    print("=" * 100)
    print("BASELINE-ONLY REAL-WORLD RUNNER + INTEGRATED SECURITY RESUME")
    print("=" * 100)
    print("Source scenario :", scenario_name)
    print("Protocol        :", grw.PROTOCOL_VERSION)
    print("Common metrics  :", COMMON_METRICS_VERSION)
    print("Security metrics:", COMMON_SECURITY_METRICS_VERSION)
    print("Nodes           :", num_nodes)
    print("Network         :", network_size)
    print("Seeds           :", list(seeds))
    print("Operational time:", sim_time)
    print("Proposed method : EXCLUDED")
    print("Methods         :", labels)
    print("Security-aware  :", [spec.label for spec in security_registry])
    print("Existing performance Method×Seed:", len(performance_completed))
    print("Existing security Method×Seed   :", len(security_completed))
    print("=" * 100)

    for seed in seeds:
        trace_path = traces_dir / f"realworld_workload_seed_{seed:03d}.json"
        if not trace_path.exists():
            grw.generate_shared_workload_trace(
                trace_path,
                seed=seed,
                num_nodes=num_nodes,
                sim_time=sim_time,
                canonical_config=canonical,
            )

        for spec in registry:
            pair = (spec.label, int(seed))
            security_aware = _is_security_aware(spec)

            need_performance = pair not in performance_completed
            need_security = security_aware and pair not in security_completed

            if not need_performance and not need_security:
                if security_aware:
                    print(
                        f"[RESUME] SKIP {spec.label}; seed={seed} "
                        "(performance=OK, security=OK)"
                    )
                else:
                    print(
                        f"[RESUME] SKIP {spec.label}; seed={seed} "
                        "(performance=OK)"
                    )
                continue

            missing_parts = []
            if need_performance:
                missing_parts.append("PERFORMANCE")
            if need_security:
                missing_parts.append("SECURITY")

            print("\n" + "-" * 100)
            print(
                f"[RUN ONCE] method={spec.label}; seed={seed}; "
                f"missing={'+'.join(missing_parts)}"
            )
            print(
                f"[AUDIT] method-native changed keys: "
                f"{audit_changes[spec.label]}"
            )
            print("-" * 100)

            # Fresh copy; workload trace is common-injected, exactly like original.
            config = dict(built_configs[spec.label])
            config["workload_trace_record_path"] = None
            config["workload_trace_replay_path"] = str(trace_path)

            simulator = spec.manager_class(
                num_nodes=num_nodes,
                sim_time=sim_time,
                ablation_config=config,
                seed=int(seed),
                network_size=network_size,
            )

            common_metrics = None
            if need_performance:
                common_metrics = attach_global_comparison_metrics(
                    simulator,
                    workload_trace_path=trace_path,
                    method_name=spec.label,
                    scenario_name=scenario_name,
                    charging_deadline_s=canonical.get("hard_deadline_margin_s"),
                )

            wall_start = time.perf_counter()
            simulator.run_simulation()
            wall_runtime = time.perf_counter() - wall_start

            # Save only the missing side(s); never duplicate an already-valid row.
            if need_performance:
                row = common_metrics.export_row(
                    seed=int(seed),
                    sim_time=sim_time,
                    wall_clock_runtime_s=wall_runtime,
                    extra_metadata={
                        "Global_Protocol_Version": grw.PROTOCOL_VERSION,
                        "Global_Source_Scenario": scenario_name,
                        "Global_Canonical_Config_SHA256": canonical_digest,
                        "Global_Node_Count": int(num_nodes),
                        "Global_Network_Width_m": float(network_size[0]),
                        "Global_Network_Height_m": float(network_size[1]),
                        "Global_Operational_Time_s": int(sim_time),
                        "Global_Shared_Workload_Trace": str(trace_path),
                        "Global_Deployment_Policy": spec.deployment_policy,
                        "Method_Native_Changed_Keys": "|".join(
                            audit_changes[spec.label]
                        ),
                        "Method_Config_SHA256": method_digests[spec.label],
                    },
                )
                row = grw.apply_metric_applicability(row, spec)
                performance_rows.append(row)
                performance_completed.add(pair)
                _atomic_write_dataframe(
                    pd.DataFrame(performance_rows),
                    performance_results_csv,
                )
                print(
                    f"[PERFORMANCE CHECKPOINT] saved {spec.label}; seed={seed}"
                )

            if need_security:
                security_row = export_common_security_row(
                    simulator,
                    seed=int(seed),
                    method_name=spec.label,
                    scenario_name=scenario_name,
                    extra_metadata={
                        "Security_Protocol_Version": SECURITY_PROTOCOL_VERSION,
                        "Global_Source_Scenario": scenario_name,
                        "Global_Canonical_Config_SHA256": canonical_digest,
                        "Global_Node_Count": int(num_nodes),
                        "Global_Network_Width_m": float(network_size[0]),
                        "Global_Network_Height_m": float(network_size[1]),
                        "Global_Operational_Time_s": int(sim_time),
                        "Global_Shared_Workload_Trace": str(trace_path),
                        "Global_Deployment_Policy": spec.deployment_policy,
                        "Method_Native_Changed_Keys": "|".join(
                            audit_changes[spec.label]
                        ),
                        "Method_Config_SHA256": method_digests[spec.label],
                    },
                )
                security_rows.append(security_row)
                security_completed.add(pair)
                _checkpoint_security_outputs(
                    security_rows,
                    results_csv=security_results_csv,
                    summary_csv=security_summary_csv,
                    primary_csv=security_primary_csv,
                )
                print(
                    f"[SECURITY CHECKPOINT] saved {spec.label}; seed={seed}; "
                    f"TP={security_row['TP']} TN={security_row['TN']} "
                    f"FP={security_row['FP']} FN={security_row['FN']} "
                    f"TPR={security_row['TPR']:.4f} "
                    f"FPR={security_row['FPR']:.4f} "
                    f"F1={security_row['F1_Score']:.4f}"
                )

    # ------------------------------------------------------------------
    # Final completeness checks
    # ------------------------------------------------------------------
    expected_performance = {
        (spec.label, int(seed))
        for spec in registry
        for seed in seeds
    }
    missing_performance = expected_performance - performance_completed
    if missing_performance:
        raise RuntimeError(
            f"Incomplete performance result set; "
            f"missing {len(missing_performance)} Method×Seed rows."
        )

    expected_security = {
        (spec.label, int(seed))
        for spec in security_registry
        for seed in seeds
    }
    missing_security = expected_security - security_completed
    if missing_security:
        raise RuntimeError(
            f"Incomplete security result set; "
            f"missing {len(missing_security)} Method×Seed rows."
        )

    # ------------------------------------------------------------------
    # Final common-performance outputs: identical contract to global runner.
    # ------------------------------------------------------------------
    performance_df = pd.DataFrame(performance_rows)
    if performance_df.duplicated(["Method", "Seed"], keep=False).any():
        raise RuntimeError("Duplicate performance Method×Seed rows detected.")

    _atomic_write_dataframe(performance_df, performance_results_csv)
    performance_summary = summarize_global_comparison_results(
        performance_df,
        performance_summary_csv,
    )
    performance_primary = select_primary_comparison_table(
        performance_summary
    )
    _atomic_write_dataframe(
        performance_primary,
        performance_primary_csv,
    )

    # ------------------------------------------------------------------
    # Final separate security outputs.
    # ------------------------------------------------------------------
    security_df = pd.DataFrame(security_rows)
    if security_df.duplicated(["Method", "Seed"], keep=False).any():
        raise RuntimeError("Duplicate security Method×Seed rows detected.")

    _atomic_write_dataframe(security_df, security_results_csv)
    security_summary = summarize_common_security_results(
        security_df,
        security_summary_csv,
    )
    security_primary = select_primary_security_table(
        security_summary
    )
    _atomic_write_dataframe(
        security_primary,
        security_primary_csv,
    )

    security_per_method_dir.mkdir(parents=True, exist_ok=True)
    for method_name, group in security_df.groupby("Method", sort=False):
        safe = "".join(
            c if c.isalnum() or c in "_-" else "_"
            for c in str(method_name)
        )
        _atomic_write_dataframe(
            group.copy(),
            security_per_method_dir / f"{safe}_security_results.csv",
        )

    # Preserve the FCBE controlled component track from the original runner.
    # Its own implementation already resumes exact Method×Seed rows.
    fcbe2025_component = grw.run_fcbe2025_component_track(
        scenario_name=scenario_name,
        canonical=canonical,
        num_nodes=num_nodes,
        network_size=network_size,
        seeds=seeds,
        sim_time=sim_time,
        output_dir=output_dir,
        traces_dir=traces_dir,
    )

    # Main performance manifest, same scientific meaning as original baseline.
    performance_manifest = {
        "protocol_version": grw.PROTOCOL_VERSION,
        "common_metrics_version": COMMON_METRICS_VERSION,
        "primary_comparison_metrics": list(PRIMARY_COMPARISON_METRICS),
        "canonical_mode": grw.CANONICAL_MODE,
        "canonical_scenario": scenario_name,
        "canonical_config_sha256": canonical_digest,
        "physical_values_source": (
            "sim_phase.py -> Article_Scaled_Absolute_Single"
        ),
        "num_nodes": num_nodes,
        "network_size": list(network_size),
        "seeds": list(seeds),
        "simulation_time_s": sim_time,
        "evaluation_contract_locked_in_global_real_world_py": True,
        "shared_workload_per_scenario_seed": True,
        "component_tracks": [fcbe2025_component],
        "simulator_rng_streams_are_seeded_by_master_seed": True,
        "proposed_method_excluded_in_baseline_only_runner": True,
        "methods": [
            {
                "label": spec.label,
                "deployment_policy": spec.deployment_policy,
                "comparison_domain": spec.comparison_domain,
                "not_applicable_metric_groups": list(
                    spec.not_applicable_metric_groups
                ),
                "allowed_method_native_keys": sorted(spec.native_keys),
                "observed_changed_keys": audit_changes[spec.label],
                "config_sha256": method_digests[spec.label],
                "notes": spec.notes,
            }
            for spec in registry
        ],
        "results_csv": str(performance_results_csv),
        "summary_csv": str(performance_summary_csv),
        "primary_comparison_csv": str(performance_primary_csv),
    }
    temp_manifest = Path(str(performance_manifest_json) + ".tmp")
    with open(temp_manifest, "w", encoding="utf-8") as handle:
        json.dump(
            performance_manifest,
            handle,
            indent=2,
            sort_keys=True,
        )
    os.replace(temp_manifest, performance_manifest_json)

    # Separate security manifest.
    security_manifest = {
        "security_protocol_version": SECURITY_PROTOCOL_VERSION,
        "common_security_metrics_version": COMMON_SECURITY_METRICS_VERSION,
        "metric_calculation_source": (
            "global_security_metrics.py -> final confusion counts -> metric_collector.py::MetricsCollector.calculate_ids_metrics()"
        ),
        "confusion_matrix_definition": (
            "same final malicious_nodes vs bs.get_blacklisted_nodes() "
            "logic as SimulatorManager.run_simulation()"
        ),
        "primary_security_metrics": list(
            PRIMARY_SECURITY_COMPARISON_METRICS
        ),
        "security_output_separate_from_performance": True,
        "canonical_scenario": scenario_name,
        "canonical_config_sha256": canonical_digest,
        "num_nodes": num_nodes,
        "network_size": list(network_size),
        "seeds": list(seeds),
        "simulation_time_s": sim_time,
        "security_aware_methods": [
            spec.label for spec in security_registry
        ],
        "results_csv": str(security_results_csv),
        "summary_csv": str(security_summary_csv),
        "primary_comparison_csv": str(security_primary_csv),
        "per_method_dir": str(security_per_method_dir),
    }
    temp_security_manifest = Path(str(security_manifest_json) + ".tmp")
    with open(
        temp_security_manifest,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            security_manifest,
            handle,
            indent=2,
            sort_keys=True,
        )
    os.replace(temp_security_manifest, security_manifest_json)

    print("\n" + "=" * 100)
    print("BASELINE + SECURITY EVALUATION COMPLETE")
    print("Performance raw    :", performance_results_csv)
    print("Performance summary:", performance_summary_csv)
    print("Performance primary:", performance_primary_csv)
    print("Security raw       :", security_results_csv)
    print("Security summary   :", security_summary_csv)
    print("Security primary   :", security_primary_csv)
    print("=" * 100)

    return (
        performance_df,
        performance_summary,
        security_df,
        security_summary,
    )


def main():
    return run_baselines_with_security_resume()


if __name__ == "__main__":
    main()
