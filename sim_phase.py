import os
import json
import random
import numpy as np
import pandas as pd

from config import compute_packet_hash, make_mc_mac, summarize_results, make_plots,  DATA_PACKET_BITS, CONTROL_PACKET_BYTES_SMALL, DATA_PACKET_BYTES
from metric_collector import MetricsCollector
from base_station import BaseStation
from mobile_charger import MobileCharger
from simulator_manager import SimulatorManager


def _base_config():
    return {
        'experiment_family': 'communication_security',
        'enable_energy_experiment': False,
        'enable_physical_compromise_experiment': False,
        'use_gradient_symmetric_discovery': True,
        'use_adaptive_edge_threshold': True,
        'use_critical_multipath_routing': True,
        'use_implicit_piggyback_telemetry': True,
        'use_network_tomography_for_delay': True,
        'use_intersection_logic_for_droppers': True,
        'use_asynchronous_processing_pipeline': True,
        'use_shannon_entropy_weights': True,
        'use_security_veto_aggregation': True,
        'use_stealthy_attacker_detection': True,
        'use_zero_overhead_spoof_detection': True,
        'use_sed_filter': True,
        'use_uncertainty_calibrated_risk': True,
        'use_counterfactual_path_attribution': True,
        'use_fcm_classification': True,
        'use_risk_history_hysteresis': True,
        'use_risk_adaptive_coded_multipath': False,
        'use_mc_hmac_authentication': True,
        'use_mc_location_verification': True,
        'use_secure_charging_request_validation': True,
        'enable_network_layer_ack': False,
        'enable_periodic_hello': False,
        'route_recompute_period': None,
        'data_packet_bits': DATA_PACKET_BITS,
        'attack_percentage': 0.15,
        'critical_event_probability': 0.10,
        'transmission_range': 40.0,
    }


def _physical_compromise_scenarios():
    c = dict(_base_config())
    c.update({
        'experiment_family': 'physical_compromise',
        'enable_energy_experiment': False,
        'enable_physical_compromise_experiment': True,
        'simulate_lab_reprovisioning': False,
    })
    return [('PhysicalCompromise_TPM_Zeroization', c)]


def _scenarios_for_mode(mode):
    mode = (mode or 'main').strip().lower().replace('-', '_')
    aliases = {
        'physical': 'physical_compromise', 'tpm': 'physical_compromise', 'tpm_compromise': 'physical_compromise',
        'mc': 'energy', 'mobile_charging': 'energy', 'charging': 'energy',
        'all': 'all_modes', 'full': 'all_modes', 'full_suite': 'all_modes', 'everything': 'all_modes',
        'complete': 'all_modes', 'complete_suite': 'all_modes',
        'sweep': 'parameter_sweep', 'stress': 'parameter_sweep', 'parameter': 'parameter_sweep',
    }
    mode = aliases.get(mode, mode)

    base = _base_config()

    def get_energy():
        c = dict(base)
        c.update({
            'experiment_family': 'energy_charging_stress',
            'enable_energy_experiment': True,
            'initial_energy_override': 0.010,
            'event_rate_lambda': 1.8,
            'idle_sensing_power_j_per_s': 1.0e-5,
            'charge_need_ratio': 0.80,
            'charge_claim_est_max_deviation': 1.00,
            'allow_energy_stress_predictive_charging': True,
        })
        return [('EnergyStress_MC_Security', c)]

    def get_mc_ablation():
        out = []

        def add(name, **changes):
            c = dict(base)
            c.update({
                'experiment_family': 'energy_charging_stress',
                'enable_energy_experiment': True,
                'initial_energy_override': 0.018,
                'event_rate_lambda': 1.2,
                'idle_sensing_power_j_per_s': 2.5e-6,
            })
            c.update(changes)
            out.append((name, c))
        add('MC_Full_Security')
        add('MC_No_HMAC', use_mc_hmac_authentication=False)
        add('MC_No_Location_Check', use_mc_location_verification=False)
        add('MC_Raw_SOS_No_Validation',
            use_secure_charging_request_validation=False)
        add('MC_No_SED', use_sed_filter=False)
        add('MC_No_Deviation_Filter', use_zero_overhead_spoof_detection=False)
        return out

    def get_ablation():
        ablations = []

        def add(name, **changes):
            c = dict(base)
            c.update(changes)
            ablations.append((name, c))
        add('Full_Method')
        add('A01_FCM_Only', use_uncertainty_calibrated_risk=False, use_risk_history_hysteresis=False, use_counterfactual_path_attribution=False,
            use_zero_overhead_spoof_detection=False, use_sed_filter=False, use_stealthy_attacker_detection=False)
        add('A02_No_FCM', use_fcm_classification=False)
        add('A03_No_UCR_Risk_Fusion', use_uncertainty_calibrated_risk=False)
        add('A04_No_Risk_History_Hysteresis', use_risk_history_hysteresis=False)
        add('A05_No_WRSN_Hard_Filters', use_zero_overhead_spoof_detection=False,
            use_sed_filter=False, use_stealthy_attacker_detection=False)
        add('A06_No_SED', use_sed_filter=False)
        add('A07_No_Deviation_MC', use_zero_overhead_spoof_detection=False)
        add('A08_No_Stealth_Filter', use_stealthy_attacker_detection=False)
        add('A09_No_Counterfactual_Path_Attribution',
            use_counterfactual_path_attribution=False)
        add('A10_No_Shannon_Entropy_Weights', use_shannon_entropy_weights=False)
        add('A11_No_Security_Veto_FE', use_security_veto_aggregation=False)
        add('A12_No_Critical_Multipath', use_critical_multipath_routing=False)
        add('A13_Periodic_Route_Recompute', route_recompute_period=1)
        add('A14_Periodic_HELLO_Enabled', enable_periodic_hello=True)
        add('A15_Network_ACK_Enabled', enable_network_layer_ack=True)
        add('A16_No_Piggyback_Telemetry', use_implicit_piggyback_telemetry=False)
        add('A17_No_Adaptive_Threshold', use_adaptive_edge_threshold=False)
        return ablations

    def get_parameter_sweep():
        out = []

        def add(name, **changes):
            c = dict(base)
            c.update(changes)
            out.append((name, c))
        for attack_pct in (0.00, 0.05, 0.15, 0.30, 0.45):
            add(f'Sweep_AttackPct_{int(attack_pct*100):02d}',
                attack_percentage=attack_pct)
        for event_rate in (0.10, 0.50, 1.20, 2.00):
            add(f'Sweep_EventRate_{str(event_rate).replace(".", "p")}',
                event_rate_lambda=event_rate)
        for crit_prob in (0.01, 0.05, 0.10, 0.25, 0.40):
            add(f'Sweep_CriticalProb_{int(crit_prob*100):02d}',
                critical_event_probability=crit_prob)
        for tx_range in (25.0, 40.0, 60.0):
            add(f'Sweep_TxRange_{int(tx_range)}m', transmission_range=tx_range)
        for w in (10, 25, 50, 100):
            add(f'Sweep_WindowW_{w}', sliding_window_size=w)
        return out

    if mode == 'physical_compromise':
        return _physical_compromise_scenarios()
    if mode == 'security_suite':
        return [('Phase1_Method', base)] + get_ablation()
    if mode in ('charging_suite', 'energy_suite', 'mc_suite'):
        return get_energy() + get_mc_ablation()
    if mode == 'all_modes':
        return [('Phase1_Method', base)] + get_ablation() + get_energy() + get_mc_ablation() + _physical_compromise_scenarios()
    if mode in ('parameter_sweep', 'stress_sweep', 'innovation_parameter_sweep'):
        return get_parameter_sweep()
    if mode == 'energy':
        return get_energy()
    if mode == 'mc_ablation':
        return get_mc_ablation()
    if mode == 'ablation':
        return get_ablation()

    return [('Phase1_Method', base)]


def _edgecase_base_station(async_pipeline=False):
    cfg = dict(_base_config())
    cfg['use_asynchronous_processing_pipeline'] = bool(async_pipeline)
    cfg['require_source_packet_hmac'] = True
    cfg['use_critical_multipath_routing'] = True
    metrics = MetricsCollector(total_nodes=4, total_rounds_or_time=10)
    bs = BaseStation(metrics, cfg, bs_id=0)

    bs.node_states = {
        1: {
            'x': 0.0, 'y': 0.0, 'E_initial': 1.0, 'E_claimed': 1.0, 'E_est': 1.0,
            'trust_vector': [1.0, 1.0, 1.0, 1.0, 1.0], 'T1_history': 1.0, 'T2_history': 1.0,
            'T3_history': 1.0, 'T4_history': 1.0, 'T5_history': 1.0, 'FE': 1.0,
            'status': 'WHITELISTED', 'crypto_keys': 'node-key-1', 'Deviation_MC_norm': 0.0,
            'SED_index': 0.0, 'spoof_trust': 1.0, 'total_injected_energy': 0.0,
        }
    }
    bs.routing_tables = {1: {'primary': 2, 'secondary': 3, 'primary_path': [
        1, 2, 0], 'secondary_path': [1, 3, 0], 'critical_paths': [[1, 2, 0], [1, 3, 0]]}}
    return bs, metrics


def _edgecase_packet(seq=1, packet_type='DATA', priority='NORMAL', energy=0.9, payload=25.0, timestamp=0.0, key='node-key-1'):
    pkt = {
        'source_id': 1, 'type': packet_type, 'seq': int(seq), 'timestamp': float(timestamp),
        'priority': priority, 'payload': payload,
        'size_bytes': CONTROL_PACKET_BYTES_SMALL if packet_type != 'DATA' else DATA_PACKET_BYTES,
        'energy_trace': [(1, float(energy))],
    }
    if key is not None:
        pkt['hash'] = compute_packet_hash(pkt, key)
    return pkt


def run_security_edge_case_validation(output_csv='security_edge_cases_results.csv', verbose=True):
    rows = []

    def record(case, passed, expected, observed, details=''):
        rows.append({'Case': case, 'Passed': bool(
            passed), 'Expected': expected, 'Observed': observed, 'Details': details})

    bs, metrics = _edgecase_base_station()
    bad_sos = _edgecase_packet(seq=1, packet_type='SOS_CHARGE',
                               priority='HIGH', energy=0.01, timestamp=0.0, key=None)
    bs.data_plane_receive(dict(bad_sos), [1, 2, 0], 0.1)
    record('Forged_SOS_Rejected', (1 not in bs.pending_sos_requests and abs(bs.node_states[1]['E_claimed'] - 1.0) < 1e-12 and metrics.packet_hmac_failures >= 1),
           'No pending SOS, no E_claimed update, HMAC failure counted', f"pending={1 in bs.pending_sos_requests}, E_claimed={bs.node_states[1]['E_claimed']}, hmac_failures={metrics.packet_hmac_failures}")

    bs, metrics = _edgecase_base_station()
    good_sos = _edgecase_packet(
        seq=1, packet_type='SOS_CHARGE', priority='HIGH', energy=0.2, timestamp=0.0)
    bs.data_plane_receive(dict(good_sos), [1, 2, 0], 0.1)
    record('Authenticated_SOS_Accepted_As_Evidence', (1 in bs.pending_sos_requests and abs(
        bs.node_states[1]['E_claimed'] - 0.2) < 1e-12), 'Pending SOS evidence exists and E_claimed is updated', f"pending={1 in bs.pending_sos_requests}, E_claimed={bs.node_states[1]['E_claimed']}")

    bs, metrics = _edgecase_base_station()
    bad_data = _edgecase_packet(
        seq=1, packet_type='DATA', priority='NORMAL', timestamp=0.0)
    bad_data['hash'] = 'not-a-valid-hmac'
    bs.data_plane_receive(dict(bad_data), [1, 2, 0], 0.1)
    record('Bad_Data_HMAC_Rejected', (metrics.packet_hmac_failures >= 1 and bs.last_seq_per_source.get(1) is None),
           'HMAC failure counted and DATA freshness not advanced', f"hmac_failures={metrics.packet_hmac_failures}, last_data_seq={bs.last_seq_per_source.get(1)}")

    bs, metrics = _edgecase_base_station()
    p = _edgecase_packet(seq=5, packet_type='DATA',
                         priority='NORMAL', timestamp=1.0)
    bs.data_plane_receive(dict(p), [1, 2, 0], 1.1)
    bs.data_plane_receive(dict(p), [1, 2, 0], 1.2)
    record('Source_Sequence_Replay_Rejected', metrics.source_sequence_replay_events >= 1,
           'Replay counter increments for repeated normal DATA sequence', f"replay_events={metrics.source_sequence_replay_events}")

    bs, metrics = _edgecase_base_station()
    high = _edgecase_packet(seq=10, packet_type='DATA',
                            priority='HIGH', payload=80.0, timestamp=1.0)
    bs.data_plane_receive(dict(high), [1, 2, 0], 1.1)
    before = metrics.source_sequence_replay_events
    bs.data_plane_receive(dict(high), [1, 3, 0], 1.2)
    record('Approved_Critical_Multipath_Copy_Accepted', metrics.source_sequence_replay_events == before,
           'Approved secondary path does not count as replay', f"replay_events_before={before}, replay_events_after={metrics.source_sequence_replay_events}")

    bs, metrics = _edgecase_base_station()
    high = _edgecase_packet(seq=10, packet_type='DATA',
                            priority='HIGH', payload=80.0, timestamp=1.0)
    bs.data_plane_receive(dict(high), [1, 2, 0], 1.1)
    before = metrics.source_sequence_replay_events
    bs.data_plane_receive(dict(high), [1, 99, 0], 1.2)
    record('Unapproved_Critical_Multipath_Rejected', metrics.source_sequence_replay_events == before + 1,
           'Unapproved repeated HIGH path increments replay/path-injection counter', f"replay_events_before={before}, replay_events_after={metrics.source_sequence_replay_events}")

    bs, metrics = _edgecase_base_station()
    data = _edgecase_packet(seq=1, packet_type='DATA',
                            priority='NORMAL', timestamp=1.0)
    sos = _edgecase_packet(seq=1, packet_type='SOS_CHARGE',
                           priority='HIGH', timestamp=1.1)
    bs.data_plane_receive(dict(data), [1, 2, 0], 1.2)
    before = metrics.source_sequence_replay_events
    bs.data_plane_receive(dict(sos), [1, 2, 0], 1.3)
    record('Control_Sequence_Separate_From_Data_Sequence', (metrics.source_sequence_replay_events == before and 1 in bs.pending_sos_requests),
           'SOS seq=1 accepted after DATA seq=1 because freshness is per packet type', f"replay_events_before={before}, replay_events_after={metrics.source_sequence_replay_events}, pending={1 in bs.pending_sos_requests}")

    bs, metrics = _edgecase_base_station()
    bs.mc_shared_key = 'mc-key'
    ok = bs.report_mc_injection_oob(
        1, 0.2, mc_x=0.0, mc_y=0.0, mac_tag='bad-tag', mc_id='MC-1', counter=1, nonce='n1', timestamp=1.0)
    record('Forged_MC_Report_Rejected', (not ok and metrics.mc_rejected_unauth_reports >= 1 and abs(bs.node_states[1].get(
        'total_injected_energy', 0.0)) < 1e-12), 'Invalid MC HMAC rejected and injected energy not credited', f"ok={ok}, rejected={metrics.mc_rejected_unauth_reports}, injected={bs.node_states[1].get('total_injected_energy', 0.0)}")

    bs, metrics = _edgecase_base_station()
    bs.mc_shared_key = 'mc-key'
    tag = make_mc_mac('mc-key', 1, 0.2, 0.0, 0.0, mc_id='MC-1',
                      counter=1, nonce='n-valid', timestamp=1.0)
    ok1 = bs.report_mc_injection_oob(
        1, 0.2, mc_x=0.0, mc_y=0.0, mac_tag=tag, mc_id='MC-1', counter=1, nonce='n-valid', timestamp=1.0)
    ok2 = bs.report_mc_injection_oob(
        1, 0.2, mc_x=0.0, mc_y=0.0, mac_tag=tag, mc_id='MC-1', counter=1, nonce='n-valid', timestamp=1.0)
    record('MC_Report_Replay_Rejected', (ok1 is True and ok2 is False and metrics.mc_rejected_unauth_reports >= 1 and abs(bs.node_states[1].get(
        'total_injected_energy', 0.0) - 0.2) < 1e-12), 'First valid MC report accepted, replay rejected, injected energy counted once', f"first_ok={ok1}, second_ok={ok2}, rejected={metrics.mc_rejected_unauth_reports}, injected={bs.node_states[1].get('total_injected_energy', 0.0)}")

    bs, metrics = _edgecase_base_station()
    bs.node_states[1]['E_est'] = 0.2
    bs.node_states[1]['E_claimed'] = 0.9
    bs.CHARGE_MAX_CLAIM_EST_DEVIATION = 0.30
    ok = bs.validate_charging_request(1, emergency=True)
    record('Raw_Inconsistent_SOS_Dispatch_Denied', (ok is False and metrics.mc_raw_sos_dispatches == 0),
           'Claim/estimate inconsistency denies dispatch and raw SOS dispatch counter stays zero', f"ok={ok}, raw_sos_dispatches={metrics.mc_raw_sos_dispatches}")

    bs, metrics = _edgecase_base_station()
    mc = MobileCharger(mc_id='MC-1', depot_x=0.0, depot_y=0.0, capacity=1.0,
                       speed=1.0, base_station=bs, metrics_collector=metrics)
    mc.crypto_keys = 'mc-key'
    mc.tpms_zeroization()
    record('TPM_Zeroization_Bricks_MC', (mc.crypto_keys is None and mc.zeroized is True and mc.requires_lab_reprovisioning is True and mc.status == 'BRICKED'),
           'Keys zeroized, MC marked bricked, lab reprovisioning required', f"crypto_keys={mc.crypto_keys}, zeroized={mc.zeroized}, requires_lab={mc.requires_lab_reprovisioning}, status={mc.status}")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    passed = int(df['Passed'].sum())
    total = int(len(df))
    if verbose:
        print(
            f"[EDGE-VALIDATION] Passed {passed}/{total} security edge cases.")
        print(df.to_string(index=False))
        print(f"[EDGE-VALIDATION] Saved {output_csv}")
    if passed != total:
        failed = df.loc[~df['Passed'], 'Case'].tolist()
        raise AssertionError(f"Security edge-case validation failed: {failed}")
    return df


def _innovation_audit_row(case, passed, expected, observed, metrics=None, details=''):
    row = {
        'Case': case,
        'Passed': bool(passed),
        'Expected': expected,
        'Observed': observed,
        'Details': details,
    }
    if metrics is not None:
        row.update(metrics)
    return row


def _audit_tomography_case(observable=True):
    if observable:
        A = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
        ], dtype=float)
        x_true = np.array([0.02, 0.35, 0.01], dtype=float)
        true_attacker_idx = 1
    else:
        A = np.array([
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
        ], dtype=float)
        x_true = np.array([0.20, 0.20, 0.01], dtype=float)
        true_attacker_idx = None

    D = A @ x_true
    x_hat, residuals, rank, _ = np.linalg.lstsq(A, D, rcond=None)
    rank_ok = rank >= A.shape[1]
    residual = float(np.mean((A @ x_hat - D) ** 2))

    if rank_ok:
        predicted_idx = int(np.argmax(x_hat))
        correct = (predicted_idx == true_attacker_idx)
        false_accusation = False
    else:
        predicted_idx = None
        correct = True
        false_accusation = False

    return {
        'rank': int(rank), 'observable': bool(rank_ok), 'predicted_idx': predicted_idx,
        'true_idx': true_attacker_idx, 'correct': bool(correct),
        'false_accusation': bool(false_accusation), 'residual': residual,
    }


def run_innovation_coverage_validation(output_csv='phase1_innovation_coverage_validation.csv', verbose=True):
    rows = []

    bs, metrics = _edgecase_base_station()
    bs.energy_experiment_active = True
    cases = [
        ('need_accept', 0.20, 0.20, 'WHITELISTED', True),
        ('healthy_reject', 0.90, 0.90, 'WHITELISTED', False),
        ('blacklisted_reject', 0.10, 0.10, 'BLACKLISTED', False),
        ('inconsistent_reject', 0.20, 0.95, 'WHITELISTED', False),
    ]
    for idx, (_, e_est, e_claim, status, needed) in enumerate(cases, start=1):
        bs.node_states[1]['E_est'] = e_est
        bs.node_states[1]['E_claimed'] = e_claim
        bs.node_states[1]['status'] = status
        admitted = bs.validate_charging_request(1, emergency=False)
        metrics.record_charging_decision(needed=needed, admitted=admitted)
    s = metrics.export_summary(1, 'InnovationAudit_ChargingDecision', 10)
    rows.append(_innovation_audit_row(
        'Charging_Decision_Accuracy_and_False_Admissions',
        s['Charging_Decision_Accuracy'] >= 0.99 and s['False_Charging_Admissions'] == 0,
        'Needed low-energy node admitted; healthy, blacklisted, and inconsistent requests rejected.',
        f"accuracy={s['Charging_Decision_Accuracy']:.3f}, false_admissions={s['False_Charging_Admissions']}", s))

    metrics = MetricsCollector(total_nodes=4, total_rounds_or_time=10)
    metrics.record_silent_sos_result(
        rescued=True, false_sos=False, response_time=12.0)
    metrics.record_silent_sos_result(
        rescued=False, false_sos=True, response_time=None)
    s = metrics.export_summary(1, 'InnovationAudit_SilentSOS', 10)
    rows.append(_innovation_audit_row(
        'Silent_Node_SOS_Rescue_and_False_SOS_Rate',
        abs(s['Silent_Node_Rescue_Rate'] -
            0.5) < 1e-12 and abs(s['False_SOS_Rate'] - 0.5) < 1e-12,
        'One true silent-node SOS is rescued and one false SOS is counted as false.',
        f"rescue_rate={s['Silent_Node_Rescue_Rate']:.3f}, false_sos_rate={s['False_SOS_Rate']:.3f}", s))

    metrics = MetricsCollector(total_nodes=4, total_rounds_or_time=10)
    obs = _audit_tomography_case(observable=True)
    deg = _audit_tomography_case(observable=False)
    metrics.record_delay_tomography_result(
        obs['observable'], correct=obs['correct'], false_accusation=obs['false_accusation'], residual_error=obs['residual'])
    metrics.record_delay_tomography_result(
        deg['observable'], correct=deg['correct'], false_accusation=deg['false_accusation'], residual_error=deg['residual'])
    s = metrics.export_summary(1, 'InnovationAudit_Tomography', 10)
    rows.append(_innovation_audit_row(
        'Delay_Tomography_Accuracy_Observability_and_Conservative_Rank_Gating',
        s['Delay_Attribution_Accuracy'] >= 0.99 and abs(
            s['Tomography_Observability_Ratio'] - 0.5) < 1e-12 and s['False_Delay_Accusation_Rate'] == 0.0,
        'Observable window identifies the slow node; rank-deficient window does not force a node accusation.',
        f"accuracy={s['Delay_Attribution_Accuracy']:.3f}, observability={s['Tomography_Observability_Ratio']:.3f}, rank_deficient={s['Tomography_Rank_Deficient_Windows']}", s))

    metrics = MetricsCollector(total_nodes=4, total_rounds_or_time=10)
    metrics.record_shadow_route_result(precomputed=True, extra_compute_ms=2.0)
    metrics.record_shadow_route_result(
        hit=True, ordinary_ms=10.0, shadow_ms=1.0)
    s = metrics.export_summary(1, 'InnovationAudit_ShadowRouting', 10)
    rows.append(_innovation_audit_row(
        'Shadow_Route_Precompute_Hit_and_Speedup_Metrics',
        s['Shadow_Route_Hit_Rate'] == 1.0 and s['Route_Repair_Speedup_Ratio'] >= 9.99,
        'One standby route is precomputed and later activated faster than ordinary recomputation.',
        f"hit_rate={s['Shadow_Route_Hit_Rate']:.3f}, speedup={s['Route_Repair_Speedup_Ratio']:.3f}", s))

    bs, metrics = _edgecase_base_station()
    high = _edgecase_packet(seq=10, packet_type='DATA',
                            priority='HIGH', payload=80.0, timestamp=1.0)
    bs.data_plane_receive(dict(high), [1, 2, 0], 1.1)
    before = metrics.source_sequence_replay_events
    bs.data_plane_receive(dict(high), [1, 3, 0], 1.2)
    accepted = int(metrics.source_sequence_replay_events == before)
    before = metrics.source_sequence_replay_events
    bs.data_plane_receive(dict(high), [1, 99, 0], 1.3)
    rejected = int(metrics.source_sequence_replay_events == before + 1)
    metrics.record_critical_multipath_opportunity(
        secondary_available=True, accepted=accepted, rejected=rejected)
    metrics.record_critical_multipath_opportunity(
        secondary_available=False, accepted=0, rejected=0)
    s = metrics.export_summary(1, 'InnovationAudit_CriticalMultipath', 10)
    rows.append(_innovation_audit_row(
        'Critical_Multipath_Availability_Approved_and_Unapproved_Copy_Handling',
        accepted == 1 and rejected == 1 and abs(
            s['Critical_Secondary_Path_Availability_Rate'] - 0.5) < 1e-12,
        'Approved secondary copy is admitted; unapproved duplicate/path-injection is rejected; sparse no-backup case is counted.',
        f"availability={s['Critical_Secondary_Path_Availability_Rate']:.3f}, accepted={s['Critical_Redundant_Copy_Accepted_Count']}, rejected={s['Critical_Redundant_Copy_Rejected_Count']}", s))

    bs, metrics = _edgecase_base_station()
    bs.mc_shared_key = 'mc-key'
    forged = bs.report_mc_injection_oob(
        1, 0.2, mc_x=0.0, mc_y=0.0, mac_tag='bad', mc_id='MC-1', counter=1, nonce='bad-nonce', timestamp=1.0)
    good_tag = make_mc_mac('mc-key', 1, 0.2, 0.0, 0.0,
                           mc_id='MC-1', counter=2, nonce='n-good', timestamp=2.0)
    ok1 = bs.report_mc_injection_oob(
        1, 0.2, mc_x=0.0, mc_y=0.0, mac_tag=good_tag, mc_id='MC-1', counter=2, nonce='n-good', timestamp=2.0)
    replay = bs.report_mc_injection_oob(
        1, 0.2, mc_x=0.0, mc_y=0.0, mac_tag=good_tag, mc_id='MC-1', counter=2, nonce='n-good', timestamp=2.0)
    bs.verify_charging_location(1, mc_x=999.0, mc_y=999.0)
    bs.node_states[1]['E_est'] = 0.2
    bs.node_states[1]['E_claimed'] = 0.9
    raw_ok = bs.validate_charging_request(1, emergency=True)
    s = metrics.export_summary(1, 'InnovationAudit_MCSecurity', 10)
    rows.append(_innovation_audit_row(
        'MC_HMAC_Replay_Location_Spoof_and_Raw_SOS_Blocking',
        (not forged) and ok1 and (not replay) and (
            not raw_ok) and s['Location_Spoof_Rejection_Rate'] == 1.0,
        'Forged MC report rejected, valid report accepted once, replay rejected, off-location report rejected, inconsistent raw SOS blocked.',
        f"forged_rejection={s['Forged_MC_Report_Rejection_Rate']:.3f}, replay_rejection={s['MC_Replay_Rejection_Rate']:.3f}, location_rejection={s['Location_Spoof_Rejection_Rate']:.3f}, raw_sos_block={s['Raw_SOS_Blocking_Rate']:.3f}", s))

    bs, metrics = _edgecase_base_station()
    bs.node_states[1]['E_claimed_prev'] = 0.20
    bs.node_states[1]['E_claimed'] = 0.80
    bs.mc_recent_injected_energy[1] = 0.0
    bs._verify_wrsn_cyber_physical_security()
    fake_detected = bs.node_states[1].get('hard_veto', False)
    bs, metrics2 = _edgecase_base_station()
    bs.node_states[1]['E_claimed_prev'] = 0.50
    bs.node_states[1]['E_claimed'] = 0.50
    bs.mc_recent_injected_energy[1] = 0.50
    bs._verify_wrsn_cyber_physical_security()
    hoarder_detected = bs.node_states[1].get('hard_veto', False)
    metrics.fake_charger_attempts += metrics2.fake_charger_attempts
    metrics.fake_charger_detections += metrics2.fake_charger_detections
    metrics.charge_hoarder_attempts += metrics2.charge_hoarder_attempts
    metrics.charge_hoarder_detections += metrics2.charge_hoarder_detections
    s = metrics.export_summary(1, 'InnovationAudit_EnergyAttackTaxonomy', 10)
    rows.append(_innovation_audit_row(
        'Energy_Attack_Taxonomy_Fake_Charger_and_Charge_Hoarder_Detection',
        fake_detected and hoarder_detected and s['Fake_Charger_Detection_Rate'] == 1.0 and s[
            'Charge_Hoarder_Detection_Rate'] == 1.0,
        'False energy increase without MC injection and missing claim increase after real injection both trigger hard filters.',
        f"fake_detected={fake_detected}, hoarder_detected={hoarder_detected}, fake_rate={s['Fake_Charger_Detection_Rate']:.3f}, hoarder_rate={s['Charge_Hoarder_Detection_Rate']:.3f}", s))

    sweep_manifest = {
        'network_size': '50,100,200,500 via WRSN_NUM_NODES',
        'attack_percentage': '0,5,15,30,45 suggested through scenario configs',
        'traffic_rate': 'event_rate_lambda low/medium/high',
        'critical_probability': 'critical_event_probability suggested; default remains 0.10',
        'thresholds': 'risk, SOS, MC, sequence, tomography, and window-size sensitivity columns exported',
    }
    rows.append(_innovation_audit_row(
        'Parameter_Sweep_Manifest_For_Paper_Level_Evaluation',
        True,
        'The audit records the remaining parameter families that should be swept for final paper tables.',
        json.dumps(sweep_manifest, sort_keys=True), None))

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    passed = int(df['Passed'].sum())
    total = int(len(df))
    if verbose:
        print(
            f"[INNOVATION-AUDIT] Passed {passed}/{total} innovation-coverage checks.")
        print(df[['Case', 'Passed', 'Observed']].to_string(index=False))
        print(f"[INNOVATION-AUDIT] Saved {output_csv}")
    if passed != total:
        failed = df.loc[~df['Passed'], 'Case'].tolist()
        raise AssertionError(
            f"Innovation coverage validation failed: {failed}")
    return df


def run_self_tests(verbose=True):
    ok = True
    cfg = _base_config()
    cfg['use_asynchronous_processing_pipeline'] = False
    metrics = MetricsCollector(total_nodes=4, total_rounds_or_time=10)
    bs = BaseStation(metrics, cfg, bs_id=0)

    # Initialize standard test node
    bs.node_states[1] = {
        'x': 0.0, 'y': 0.0, 'E_initial': 1.0, 'E_claimed': 1.0, 'E_est': 0.2, 'trust_vector': [1, 1, 1, 1, 1],
        'T1_history': 1.0, 'T2_history': 1.0, 'T3_history': 1.0, 'T4_history': 1.0, 'T5_history': 1.0,
        'FE': 1.0, 'status': 'WHITELISTED', 'crypto_keys': 'k1', 'Deviation_MC_norm': 0.0, 'SED_index': 0.0, 'spoof_trust': 1.0
    }
    bs.routing_tables = {1: {'primary_path': [1, 2, 0], 'secondary_path': [
        1, 3, 0], 'critical_paths': [[1, 2, 0], [1, 3, 0]]}}

    bad_sos = {'source_id': 1, 'type': 'SOS_CHARGE', 'seq': 1, 'timestamp': 0.0,
               'priority': 'HIGH', 'size_bytes': 16, 'energy_trace': [(1, 0.01)]}
    bs.data_plane_receive(dict(bad_sos), [1, 2, 0], 0.1)
    assert 1 not in bs.pending_sos_requests, 'unauthenticated SOS created a pending request'
    assert abs(bs.node_states[1]['E_claimed'] -
               1.0) < 1e-12, 'unauthenticated SOS changed E_claimed'
    assert metrics.packet_hmac_failures >= 1, 'SOS HMAC failure was not exported to metrics'

    good_sos = dict(bad_sos)
    good_sos['hash'] = compute_packet_hash(good_sos, 'k1')
    bs.data_plane_receive(dict(good_sos), [1, 2, 0], 0.2)
    assert 1 in bs.pending_sos_requests, 'authenticated SOS was not admitted as evidence'

    p = {'source_id': 1, 'type': 'DATA', 'seq': 10, 'timestamp': 1.0, 'priority': 'HIGH',
         'payload': 80.0, 'size_bytes': DATA_PACKET_BYTES, 'energy_trace': [(1, 0.8)]}
    p['hash'] = compute_packet_hash(p, 'k1')
    bs.data_plane_receive(dict(p), [1, 2, 0], 1.1)
    before = metrics.source_sequence_replay_events
    bs.data_plane_receive(dict(p), [1, 3, 0], 1.2)
    assert metrics.source_sequence_replay_events == before, 'approved secondary copy counted as replay'
    bs.data_plane_receive(dict(p), [1, 99, 0], 1.3)
    assert metrics.source_sequence_replay_events == before + \
        1, 'unapproved critical path was accepted'

    bs.CHARGE_MAX_CLAIM_EST_DEVIATION = 1.00
    bs.node_states[1]['E_est'] = 0.2
    bs.node_states[1]['E_claimed'] = 0.9
    assert bs.validate_charging_request(
        1, emergency=True), 'configured charge_claim_est_max_deviation was ignored'

    row = metrics.export_summary(1, 'unit', 10)
    assert row['Packet_HMAC_Failures'] >= 1, 'Packet_HMAC_Failures not exported'
    assert row['Source_Sequence_Replay_Events'] >= 1, 'Source_Sequence_Replay_Events not exported'
    if verbose:
        print('[SELF-TEST] Phase-1 coherence V3 self-tests passed.')
    return ok


def run_statistical_experiments():
    run_mode = os.environ.get(
        'WRSN_RUN_MODE', 'all_mode').strip().lower().replace('-', '_')
    aliases = {
        'edge_cases': 'security_edge_cases', 'edge_validation': 'security_edge_cases',
        'security_validation': 'security_edge_cases', 'security_edge_validation': 'security_edge_cases',
        'innovation': 'innovation_coverage', 'innovation_audit': 'innovation_coverage',
        'phase1_innovation_audit': 'innovation_coverage', 'innovation_validation': 'innovation_coverage',
    }
    normalized_mode = aliases.get(run_mode, run_mode)

    if normalized_mode == 'security_edge_cases':
        return run_security_edge_case_validation()
    if normalized_mode == 'innovation_coverage':
        return run_innovation_coverage_validation()

    NUM_SEEDS = int(os.environ.get('WRSN_NUM_SEEDS', 1))
    NUM_NODES = int(os.environ.get('WRSN_NUM_NODES', 100))
    SIM_TIME = int(os.environ.get('WRSN_SIM_TIME', 3600))
    scenarios = _scenarios_for_mode(normalized_mode)
    scenario_names = [name for name, _ in scenarios]
    safe_mode = ''.join(c if c.isalnum() or c in ('_', '-')
                        else '_' for c in normalized_mode)

    print(f"[RUNNER] WRSN_RUN_MODE={normalized_mode}")
    print(
        f"[RUNNER] Running {len(scenarios)} scenario(s): {', '.join(scenario_names)}")

    all_results = []
    for scenario_name, config in scenarios:
        for seed in range(1, NUM_SEEDS + 1):
            print(f"\n--- {scenario_name}: Seed {seed}/{NUM_SEEDS} ---")
            sim = SimulatorManager(
                num_nodes=NUM_NODES, sim_time=SIM_TIME, ablation_config=config, seed=seed)
            sim.run_simulation()
            all_results.append(sim.metrics.export_summary(
                seed, scenario_name, SIM_TIME))

    df = pd.DataFrame(all_results)
    prefix = f"security_performance_{safe_mode}"
    df.to_csv(f'{prefix}_results.csv', index=False)
    summary = summarize_results(df, f'{prefix}_summary_ci.csv')
    make_plots(df, f'plots_{prefix}')
    print(
        f"\n[SUCCESS] Saved {prefix}_results.csv, {prefix}_summary_ci.csv, and plots_{prefix}/")

    keep = [c for c in ['Ablation_Scenario', 'N', 'TPR_Mean', 'FPR_Mean', 'Accuracy_Mean', 'F1_Score_Mean',
                        'Overall_PDR_Mean', 'Critical_PDR_Mean', 'Throughput_Mean', 'Avg_Delay_Mean',
                        'Operational_Control_Overhead_%_Mean', 'MC_Travel_Distance_Mean',
                        'Charging_Decision_Accuracy_Mean', 'Silent_Node_Rescue_Rate_Mean',
                        'Delay_Attribution_Accuracy_Mean', 'Tomography_Observability_Ratio_Mean',
                        'Shadow_Route_Hit_Rate_Mean', 'Critical_Secondary_Path_Availability_Rate_Mean',
                        'Forged_MC_Report_Rejection_Rate_Mean', 'Charge_Hoarder_Detection_Rate_Mean',
                        'MC_TPM_Compromise_Events_Mean', 'MC_TPM_Zeroized_Events_Mean'] if c in summary.columns]
    print(summary[keep].to_string(index=False))

    all_mode_aliases = {'all', 'all_modes', 'full',
                        'full_suite', 'everything', 'complete', 'complete_suite'}
    security_suite_aliases = {'security_suite'}
    if (normalized_mode in all_mode_aliases or normalized_mode in security_suite_aliases) and os.environ.get('WRSN_SKIP_EDGE_CASES', '0').strip() != '1':
        run_security_edge_case_validation(
            output_csv=f"security_performance_{normalized_mode}_security_edge_cases.csv", verbose=True)

    return df, summary


if __name__ == '__main__':
    if os.environ.get('WRSN_RUN_SELF_TESTS', '0').strip() == '1':
        run_self_tests(verbose=True)
    run_statistical_experiments()
