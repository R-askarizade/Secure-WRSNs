import os
os.environ["OMP_NUM_THREADS"] = "1"

import json
import hmac
import hashlib
import numpy as np
import pandas as pd
import scipy.stats as stats


DATA_PACKET_BITS = 4000
DATA_PACKET_BYTES = DATA_PACKET_BITS // 8
ACK_PACKET_BITS = 16 * 8
CONTROL_PACKET_BYTES_SMALL = 16


def _canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _source_energy(packet):
    src = packet.get("source_id")
    for item in packet.get("energy_trace", []) or []:
        try:
            nid, energy = item
        except Exception:
            continue
        if nid == src:
            return float(energy)
    return None


def compute_packet_hash(packet, key):
    if key is None:
        return None
    safe_packet = {
        "source_id": packet.get("source_id"),
        "type": packet.get("type"),
        "seq": packet.get("seq"),
        "timestamp": packet.get("timestamp"),
        "priority": packet.get("priority"),
        "payload": packet.get("payload"),
        "size_bytes": packet.get("size_bytes"),
        "source_energy": _source_energy(packet),
    }
    return hmac.new(str(key).encode("utf-8"), _canonical_json(safe_packet), hashlib.sha256).hexdigest()


def make_mc_mac(shared_key, node_id, injected_energy, mc_x, mc_y, mc_id=None, counter=None, nonce=None, timestamp=None):
    if shared_key is None:
        return None
    msg = {
        "domain": "WRSN_MC_OOB_REPORT_V1",
        "mc_id": mc_id,
        "node_id": node_id,
        "injected_energy": round(float(injected_energy), 9),
        "mc_x": round(float(mc_x), 6),
        "mc_y": round(float(mc_y), 6),
        "counter": None if counter is None else int(counter),
        "nonce": nonce,
        "timestamp": None if timestamp is None else round(float(timestamp), 6),
    }
    return hmac.new(str(shared_key).encode("utf-8"), _canonical_json(msg), hashlib.sha256).hexdigest()


def _ci95(series):
    arr = pd.to_numeric(pd.Series(series),
                        errors='coerce').dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return (0.0, 0.0, 0.0)
    mean = float(np.mean(arr))
    if len(arr) < 2:
        return (mean, mean, mean)
    sem = stats.sem(arr)
    if not np.isfinite(sem) or sem == 0:
        return (mean, mean, mean)
    low, high = stats.t.interval(0.95, len(arr)-1, loc=mean, scale=sem)
    return (mean, float(low), float(high))


def _bounded_ci95(series, col_name=''):
    mean, low, high = _ci95(series)
    unit_interval = {'TPR', 'Recall', 'FPR', 'TNR', 'Specificity', 'FNR',
                     'Accuracy', 'Balanced_Accuracy', 'Precision', 'NPV', 'F1_Score'}
    bounded_negative_positive = {'MCC'}
    percent_cols = ['Overall_PDR', 'Critical_PDR', 'Communication_Overhead_%',
                    'Setup_Overhead_%', 'Operational_Control_Overhead_%', 'Total_Protocol_Overhead_%']
    if col_name in unit_interval:
        return (min(1.0, max(0.0, mean)), max(0.0, low), min(1.0, high))
    if col_name in bounded_negative_positive:
        return (min(1.0, max(-1.0, mean)), max(-1.0, low), min(1.0, high))
    if col_name in percent_cols:
        return (min(100.0, max(0.0, mean)), max(0.0, low), min(100.0, high))
    return (mean, low, high)


def summarize_results(df, out_csv):
    rows = []
    for scenario, g in df.groupby('Ablation_Scenario'):
        row = {'Ablation_Scenario': scenario, 'N': len(g)}
        for col in g.columns:
            if col in ('Seed', 'Ablation_Scenario', 'Confusion_Matrix'):
                continue
            if pd.api.types.is_numeric_dtype(g[col]):
                mean, low, high = _bounded_ci95(g[col], col)
                row[f'{col}_Mean'] = mean
                row[f'{col}_CI95_Low'] = low
                row[f'{col}_CI95_High'] = high
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_csv, index=False)
    return summary


def make_plots(df, out_dir):
    if os.environ.get('WRSN_SKIP_PLOTS', '0').strip() == '1':
        print('[PLOTS] Skipped because WRSN_SKIP_PLOTS=1')
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f'[PLOTS] matplotlib not available: {exc}')
        return
    os.makedirs(out_dir, exist_ok=True)
    metrics = ['TPR', 'FPR', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'MCC',
               'Fuzzy_Silhouette_Avg', 'Fuzzy_Partition_Coefficient_Avg', 'Fuzzy_Partition_Entropy_Avg',
               'Xie_Beni_Avg', 'Operational_Control_Overhead_%', 'Avg_Delay']
    if os.environ.get('WRSN_FAST_PLOTS', '0').strip() != '1':
        metrics.extend(['Balanced_Accuracy', 'Overall_PDR', 'Critical_PDR', 'Throughput',
                        'Setup_Overhead_%', 'Total_Protocol_Overhead_%', 'MC_Travel_Distance',
                        'MC_Charge_Events', 'MC_Total_Injected_Energy', 'Avg_Residual_Energy_End',
                        'BS_Execution_Time_ms', 'Sensor_Execution_Time_ms'])

    for m in metrics:
        if m not in df.columns:
            continue
        means = df.groupby('Ablation_Scenario')[m].mean(
            numeric_only=True).sort_values(ascending=False)
        plt.figure(figsize=(max(7, 0.45 * len(means)), 4.5))
        means.plot(kind='bar')
        plt.ylabel(m)
        plt.title(m)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{m}.png'), dpi=160)
        plt.close()

    for scenario, g in df.groupby('Ablation_Scenario'):
        if not {'TP', 'TN', 'FP', 'FN'}.issubset(g.columns):
            continue
        cm = np.array([[g['TN'].sum(), g['FP'].sum()], [
                      g['FN'].sum(), g['TP'].sum()]], dtype=float)
        plt.figure(figsize=(4, 4))
        plt.imshow(cm)
        for i in range(2):
            for j in range(2):
                plt.text(j, i, int(cm[i, j]), ha='center', va='center')
        plt.xticks([0, 1], ['Pred healthy', 'Pred malicious'])
        plt.yticks([0, 1], ['Actual healthy', 'Actual malicious'])
        plt.title(f'Confusion Matrix - {scenario}')
        plt.tight_layout()
        safe = ''.join(c if c.isalnum() or c in (
            '_', '-') else '_' for c in scenario)
        plt.savefig(os.path.join(
            out_dir, f'confusion_matrix_{safe}.png'), dpi=160)
        plt.close()
