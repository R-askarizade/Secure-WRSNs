import os
import random
import math
import json
import inspect
import platform
import sys
from importlib import metadata as importlib_metadata
from collections import defaultdict
import numpy as np
import networkx as nx

from config import compute_packet_hash, DATA_PACKET_BYTES, ACK_PACKET_BITS
from sensor_node import SensorNode
from base_station import BaseStation
from mobile_charger import MobileCharger
from metric_collector import MetricsCollector
from routing import BaselineRouting, DataPlaneHybridRouting
from adaptive_charging import ChargerRoutingDQN, configure_torch_runtime, torch


class SimulatorManager:
    def __init__(self, num_nodes, sim_time, ablation_config, seed, network_size=(100, 100)):
        self.num_nodes = num_nodes
        self.sim_time = sim_time
        self.ablation = dict(ablation_config)
        self.seed = seed
        self.network_size = network_size
        self.BS_ID = 0
        self.transmission_range = float(
            self.ablation.get('transmission_range', 40.0))
        self.data_packet_bits = max(1.0, float(
            self.ablation.get('data_packet_bits', DATA_PACKET_BYTES * 8)))
        self.data_packet_bytes = max(
            1, int(math.ceil(self.data_packet_bits / 8.0)))
        self.control_packet_bits = max(1.0, float(
            self.ablation.get('control_packet_bits', 16 * 8)))
        self.control_packet_bytes = max(
            1, int(math.ceil(self.control_packet_bits / 8.0)))
        self.ack_packet_bits = max(1.0, float(
            self.ablation.get('ack_packet_bits', ACK_PACKET_BITS)))
        self.implicit_mac_ack_mode = str(
            self.ablation.get('implicit_mac_ack_mode', 'accounting')).strip().lower()
        self.source_seq_counters = defaultdict(int)

        self.topology_rng = random.Random(int(self.seed))
        self.event_rng = random.Random(int(self.seed) + 100_003)
        self.attack_rng = random.Random(int(self.seed) + 200_003)
        self.key_rng = random.Random(int(self.seed) + 300_007)
        self.tpm_rng = random.Random(int(self.seed) + 400_009)

        default_device = (
            'auto' if self.ablation.get(
                'use_gpu_for_adaptive_charging', True) else 'cpu')
        self.adaptive_device_request = str(os.environ.get(
            'WRSN_ADAPTIVE_DEVICE',
            self.ablation.get('adaptive_compute_device', default_device),
        )).strip().lower()
        self.ablation['adaptive_compute_device'] = self.adaptive_device_request

        self.torch_deterministic = bool(
            self.ablation.get('adaptive_torch_deterministic', True))
        self.torch_strict_determinism = bool(
            self.ablation.get('adaptive_torch_strict_determinism', False))
        self.torch_allow_tf32 = bool(
            self.ablation.get('adaptive_torch_allow_tf32', False))
        configure_torch_runtime(
            int(self.seed),
            deterministic=self.torch_deterministic,
            strict_determinism=self.torch_strict_determinism,
            allow_tf32=self.torch_allow_tf32,
        )

        self.metrics = MetricsCollector(
            total_nodes=num_nodes, total_rounds_or_time=sim_time)
        self.metrics.experiment_parameters.update({
            'Data_Routing_Isolated_From_Charging_Flag': True,
            'Charging_Configuration_Frozen_Flag': True,
            'Data_Plane_Hybrid_Enabled_Flag': bool(
                self.ablation.get('use_data_plane_hybrid_routing', True)),
            'Data_Secondary_Replacement_Mode': (
                'energy_guarded_existing_plus_least_overlap_failover'),
            'Critical_Require_Two_Routes_Flag': bool(
                self.ablation.get('critical_require_two_routes', True)),
            'Critical_Secondary_Selection_Mode': str(
                self.ablation.get('critical_secondary_selection_mode',
                                  'least_overlap_safest')),
            'Critical_Synthesized_Secondary_Mode': str(
                self.ablation.get('critical_synthesized_secondary_mode',
                                  'failover')),
            'Critical_Failover_Energy_Guard_Flag': bool(
                self.ablation.get('critical_failover_energy_guard', True)),
            'Critical_Failover_Min_Energy_Ratio': float(
                self.ablation.get('critical_failover_min_energy_ratio', 0.65)),
            'Critical_Failover_Reserve_Margin_Ratio': float(
                self.ablation.get('critical_failover_reserve_margin_ratio', 0.05)),
            'Data_Route_Energy_Tolerance': float(
                self.ablation.get('data_route_energy_tolerance', 0.0149334243033)),
            'Data_Route_Stress_Tolerance': float(
                self.ablation.get('data_route_stress_tolerance', 0.0434899006505)),
            'Article_Energy_Profile_Enabled_Flag': bool(
                self.ablation.get('article_energy_profile_enabled', False)),
            'Article_Single_Scenario_Flag': bool(
                self.ablation.get('article_scaled_single_scenario', False)),
            'Article_Physical_Simulation_Time_s': float(
                self.ablation.get('article_physical_simulation_time_s', 0.0)),
            'Article_Effective_Simulation_Time_s': float(
                self.ablation.get('article_effective_simulation_time_s', self.sim_time)),
            'Article_Time_Scale': float(
                self.ablation.get('article_time_scale', 1.0)),
            'Article_Physical_Sensor_Initial_Energy_J': float(
                self.ablation.get('article_physical_sensor_initial_energy_j',
                                  self.ablation.get('initial_energy_override', 0.1))),
            'Article_Effective_Sensor_Initial_Energy_J': float(
                self.ablation.get('initial_energy_override', 0.1)),
            'Article_Physical_MC_Capacity_Per_Charger_J': float(
                self.ablation.get('article_physical_mc_capacity_per_charger_j',
                                  self.ablation.get('mobile_charger_capacity', 50000.0))),
            'Article_Effective_MC_Capacity_Per_Charger_J': float(
                self.ablation.get('mobile_charger_capacity', 50000.0)),
            'Article_MC_Capacity_Mode_Absolute_Flag': (
                str(self.ablation.get('mobile_charger_capacity_mode',
                    'absolute')).strip().lower()
                == 'absolute'),
            'Article_E_Elec_J_Per_Bit': float(
                self.ablation.get('radio_e_elec_j_per_bit', 50e-9)),
            'Article_Eps_FS_J_Per_Bit_M2': float(
                self.ablation.get('radio_eps_fs_j_per_bit_m2', 10e-12)),
            'Article_Eps_MP_J_Per_Bit_M4': float(
                self.ablation.get('radio_eps_mp_j_per_bit_m4', 0.0013e-12)),
            'Article_Charging_Rate_J_Per_S': float(
                self.ablation.get('charging_rate', 5.0)),
            'Article_MC_Travel_Energy_J_Per_M': float(
                self.ablation.get('mobile_charger_travel_energy_cost_per_m', 0.1)),
            'Article_MC_Charging_Radius_M': float(
                self.ablation.get('rf_charging_radius', 25.0)),
            'Article_MC_Speed_M_Per_S': float(
                self.ablation.get('mobile_charger_speed', 5.0)),
            'Article_Data_Packet_Unchanged_Flag': (
                'data_packet_bits' not in self.ablation
                or abs(float(self.ablation.get('data_packet_bits', DATA_PACKET_BYTES * 8))
                       - float(DATA_PACKET_BYTES * 8)) <= 1e-12),
            'Article_Control_Header_Unchanged_Flag': (
                'control_packet_bits' not in self.ablation),
            'Article_Total_L_Override_Flag': False,
            'Article_Charging_Threshold_Inherited_Flag': True,
            'Implicit_MAC_ACK_Always_Accounted_Flag': True,
            'Implicit_MAC_ACK_Mode': self.implicit_mac_ack_mode,
            'Bootstrap_Excluded_From_Operational_Horizon_Flag': True,
            'Bootstrap_Registration_Is_Aggregate_Fixed_Timestep_Flag': True,
            'Control_Epoch_Duration_s': 10.0,
            'Simulation_Time_Model': 'continuous_event_driven_with_fixed_control_epochs',
            'Adaptive_Training_Mode': 'online_warmup_and_safety_shielded',
            'DQN_Audit_Observer_Only_Flag': bool(
                self.ablation.get('enable_dqn_audit_logging', True)),
            'DQN_Training_Enabled_Flag': bool(
                self.ablation.get('dqn_training_enabled', True)),
            'Charger_Target_Policy': str(self.ablation.get(
                'charger_target_policy', 'dqn_safe')),
            'Workload_Replay_Enabled_Flag': bool(
                self.ablation.get('workload_trace_replay_path')),
            'Charging_Audit_Enforcement_Mode': str(self.ablation.get('charging_audit_enforcement_mode', 'shadow')),
            'Legacy_MC_Claim_Credit_Preserved_Flag': bool(self.ablation.get('preserve_legacy_mc_claim_credit', True)),
            'Attack_Model_Mode': str(self.ablation.get('attack_model_mode', 'legacy_compatible')),
            'Attack_Data_Drop_Probability': float(self.ablation.get('attack_data_drop_probability', 1.0)),
            'Attack_Control_Drop_Probability': float(self.ablation.get('attack_control_drop_probability', 0.50)),
            'Attack_False_Energy_Data_Enabled_Flag': bool(self.ablation.get('attack_false_energy_data_enabled', False)),
            'Attack_False_Energy_Receipt_Enabled_Flag': bool(self.ablation.get('attack_false_energy_receipt_enabled', True)),
            'Attack_False_SOS_Mode': str(self.ablation.get('attack_false_sos_mode', 'legacy_compatible')),
            'Attack_Delay_Enabled_Flag': bool(self.ablation.get('attack_delay_enabled', False)),
            'Attack_Tamper_Enabled_Flag': bool(self.ablation.get('attack_tamper_enabled', False)),
            'Attack_Replay_Enabled_Flag': bool(self.ablation.get('attack_replay_enabled', False)),
            'Attack_On_Off_Enabled_Flag': bool(self.ablation.get('attack_on_off_enabled', False)),
            'Attack_Remediation_Time_s': self.ablation.get(
                'attack_remediation_time_s', None),
            'Attack_Rehabilitation_Delay_s': self.ablation.get(
                'attack_rehabilitation_delay_s', 10.0),
        })

        # Reference control/charging routing remains the original method.
        self.routing_engine = BaselineRouting(
            bs_id=self.BS_ID,
            transmission_range=self.transmission_range,
            max_critical_paths=(
                4 if self.ablation.get('use_risk_adaptive_coded_multipath', False)
                else 2),
            data_packet_bits=self.data_packet_bits,
            config=self.ablation,
        )
        # The clustered/MST engine is a DATA-only overlay.  It cannot change
        # charger parameters, charging queues, control routes, or QIFL/DQN
        # features derived from the reference routing table.
        self.data_routing_engine = DataPlaneHybridRouting(
            bs_id=self.BS_ID,
            transmission_range=self.transmission_range,
            max_critical_paths=(
                4 if self.ablation.get('use_risk_adaptive_coded_multipath', False)
                else 2),
            data_packet_bits=self.data_packet_bits,
            config=self.ablation,
        )

        self.bs = BaseStation(
            metrics_collector=self.metrics,
            ablation_config=self.ablation,
            bs_id=self.BS_ID,
            routing_engine=self.routing_engine,
        )
        self.bs.data_routing_engine = self.data_routing_engine

        self.sensor_initial_energy = float(self.ablation.get(
            'initial_energy_override', 0.1))

        bs_x, bs_y = network_size[0] / 2, network_size[1] / 2
        self.mobile_chargers = self._make_mobile_charger_fleet(bs_x, bs_y)

        self.mc = self.mobile_chargers[0]
        self.use_multi_mobile_chargers = len(self.mobile_chargers) > 1
        self.metrics.mobile_charger_count = len(self.mobile_chargers)
        self.metrics.mobile_charger_capacity_mode = str(self.ablation.get(
            'mobile_charger_capacity_mode', 'absolute'))
        self.metrics.sensor_initial_energy = float(self.sensor_initial_energy)
        self.metrics.mobile_charger_total_capacity = float(
            sum(mc.E_capacity for mc in self.mobile_chargers))
        self.metrics.mobile_charger_min_capacity = float(
            min(mc.E_capacity for mc in self.mobile_chargers))
        self.metrics.mobile_charger_max_capacity = float(
            max(mc.E_capacity for mc in self.mobile_chargers))
        self.metrics.mobile_charger_avg_travel_energy_cost_per_m = float(
            np.mean([float(getattr(mc, 'travel_energy_cost_per_m', 0.0))
                     for mc in self.mobile_chargers]))
        if self.sensor_initial_energy > 0:
            self.metrics.mobile_charger_total_sensor_equivalent_capacity = (
                self.metrics.mobile_charger_total_capacity / self.sensor_initial_energy)
            self.metrics.mobile_charger_min_sensor_equivalent_capacity = (
                self.metrics.mobile_charger_min_capacity / self.sensor_initial_energy)
            self.metrics.mobile_charger_max_sensor_equivalent_capacity = (
                self.metrics.mobile_charger_max_capacity / self.sensor_initial_energy)
        else:
            self.metrics.mobile_charger_total_sensor_equivalent_capacity = 0.0
            self.metrics.mobile_charger_min_sensor_equivalent_capacity = 0.0
            self.metrics.mobile_charger_max_sensor_equivalent_capacity = 0.0

        self.use_dqn_charger_routing = bool(
            self.ablation.get('use_dqn_charger_routing', False))
        default_policy = (
            'dqn_safe' if self.use_dqn_charger_routing else 'heuristic_only')
        self.charger_target_policy = str(self.ablation.get(
            'charger_target_policy', default_policy)).strip().lower()
        self.dqn_training_enabled = bool(
            self.ablation.get('dqn_training_enabled', True))
        self.enable_dqn_audit_logging = bool(
            self.ablation.get('enable_dqn_audit_logging', True))
        self.policy_rng = random.Random(int(self.seed) + 500_009)
        self.workload_trace_record_path = self.ablation.get(
            'workload_trace_record_path')
        self.workload_trace_replay_path = self.ablation.get(
            'workload_trace_replay_path')
        self.metrics.experiment_parameters['Charger_Target_Policy'] = (
            self.charger_target_policy)
        adaptive_device = self.adaptive_device_request
        self.charger_dqn = ChargerRoutingDQN(
            lr=self.ablation.get('dqn_learning_rate', 1.0e-3),
            gamma=self.ablation.get('dqn_gamma', 0.80),
            epsilon=self.ablation.get('dqn_epsilon', 0.05),
            device=adaptive_device,
            seed=int(self.ablation.get('dqn_seed', seed + 1000)),
            warmup_updates=int(self.ablation.get('dqn_warmup_updates', 50)),
            warmup_competitive_decisions=int(self.ablation.get(
                'dqn_warmup_competitive_decisions', 50)),
            warmup_readiness_mode=str(self.ablation.get(
                'dqn_warmup_readiness_mode', 'competitive_only')),
            warmup_min_replay_samples=self.ablation.get(
                'dqn_warmup_min_replay_samples', None),
            warmup_td_abs_threshold=self.ablation.get(
                'dqn_warmup_td_abs_threshold', None),
            warmup_td_window=int(self.ablation.get(
                'dqn_warmup_td_window', 20)),
            warmup_max_total_decisions=self.ablation.get(
                'dqn_warmup_max_total_decisions', None),
            critical_energy_ratio=float(self.ablation.get(
                'critical_energy_ratio', 0.15)),
            deadline_safety_margin=float(self.ablation.get(
                'dqn_deadline_safety_margin_s', self.ablation.get(
                    'hard_deadline_margin_s', 90.0))),
            safety_override_mode=str(self.ablation.get(
                'dqn_safety_override_mode', 'deadline_feasibility')),
            exploration_mode=str(self.ablation.get(
                'dqn_exploration_mode', 'fixed')),
            hidden_dim=int(self.ablation.get('dqn_hidden_dim', 64)),
            replay_capacity=int(self.ablation.get('dqn_replay_capacity', 512)),
            batch_size=int(self.ablation.get('dqn_batch_size', 16)),
            target_sync_interval=int(self.ablation.get(
                'dqn_target_sync_interval', 10)),
            expected_mission_time=float(self.ablation.get(
                'adaptive_expected_mission_time_s', 90.0)),
            safe_horizon=float(self.ablation.get('qifl_safe_horizon_s', 3600.0)))
        checkpoint_load_path = self.ablation.get('dqn_checkpoint_load_path')
        if checkpoint_load_path:
            self.charger_dqn.load_checkpoint(
                checkpoint_load_path,
                load_optimizer=bool(self.ablation.get(
                    'dqn_checkpoint_load_optimizer', False)))

        adaptive_qifl_active = bool(
            self.ablation.get('use_adaptive_quantum_charging', False))
        adaptive_dqn_active = bool(
            self.ablation.get('use_dqn_charger_routing', False))
        adaptive_active = adaptive_qifl_active or adaptive_dqn_active
        require_torch = self._env_flag(
            'WRSN_REQUIRE_TORCH',
            self.ablation.get('require_pytorch_for_adaptive_charging', True),
        )
        require_cuda = self._env_flag(
            'WRSN_REQUIRE_CUDA',
            self.ablation.get('require_cuda_for_adaptive_charging', False),
        )
        missing_torch_components = []
        if adaptive_qifl_active and not self.bs.qifl_oracle.enabled_torch:
            missing_torch_components.append('QIFL')
        if adaptive_dqn_active and not self.charger_dqn.enabled_torch:
            missing_torch_components.append('DQN')
        if adaptive_active and require_torch and missing_torch_components:
            raise RuntimeError(
                'PyTorch is required for the configured adaptive method; '
                f"fallback is disabled for: {', '.join(missing_torch_components)}")

        active_devices = []
        if adaptive_qifl_active:
            active_devices.append(('QIFL', getattr(
                self.bs.qifl_oracle, 'device', None)))
        if adaptive_dqn_active:
            active_devices.append(('DQN', getattr(
                self.charger_dqn, 'device', None)))
        non_cuda = [name for name, device in active_devices
                    if getattr(device, 'type', '') != 'cuda']
        if adaptive_active and require_cuda and non_cuda:
            raise RuntimeError(
                'CUDA is required for the configured adaptive method, but '
                f"these components are not on CUDA: {', '.join(non_cuda)}")

        self.metrics.dqn_torch_enabled = bool(self.charger_dqn.enabled_torch)
        self.metrics.qifl_torch_enabled = bool(
            self.bs.qifl_oracle.enabled_torch)
        self.metrics.cuda_available = bool(
            getattr(self.charger_dqn, 'enabled_torch', False) and
            getattr(getattr(self.charger_dqn, 'device', None), 'type', '') == 'cuda')
        self.metrics.adaptive_compute_device = str(
            getattr(self.charger_dqn, 'device', None) or 'numpy')
        self.metrics.experiment_parameters.update(
            self._runtime_environment_metadata(
                require_torch=require_torch,
                require_cuda=require_cuda,
            ))

        self.nodes = {}

        self._deploy_network(bs_x, bs_y)

        self.malicious_nodes = []
        self.current_time = 0.0
        self.experiment_family = self.ablation.get(
            'experiment_family', 'communication_security')
        self.energy_experiment_active = self.bs.energy_experiment_active
        self.physical_compromise_experiment_active = (
            self.experiment_family == 'physical_compromise'
            or self.ablation.get('enable_physical_compromise_experiment', False)
        )

        self.active_charger_missions = {}
        self.reprovision_events = {}
        self._mission_sequence = 0

    @staticmethod
    def _env_flag(name, default=False):
        raw = os.environ.get(name)
        if raw is None:
            return bool(default)
        return str(raw).strip().lower() not in {
            '', '0', 'false', 'no', 'off', 'none'
        }

    def _runtime_environment_metadata(self, *, require_torch, require_cuda):
        def version(distribution):
            try:
                return importlib_metadata.version(distribution)
            except importlib_metadata.PackageNotFoundError:
                return 'not-installed'

        cuda_available = bool(torch is not None and torch.cuda.is_available())
        gpu_names = []
        if cuda_available:
            for index in range(torch.cuda.device_count()):
                gpu_names.append(torch.cuda.get_device_name(index))
        return {
            'Runtime_Python_Version': platform.python_version(),
            'Runtime_Python_Implementation': platform.python_implementation(),
            'Runtime_Platform': platform.platform(),
            'Runtime_NumPy_Version': np.__version__,
            'Runtime_NetworkX_Version': nx.__version__,
            'Runtime_SciPy_Version': version('scipy'),
            'Runtime_Pandas_Version': version('pandas'),
            'Runtime_Scikit_Learn_Version': version('scikit-learn'),
            'Runtime_PyTorch_Version': (
                getattr(torch, '__version__', 'not-installed')
                if torch is not None else 'not-installed'),
            'Runtime_CUDA_Available': cuda_available,
            'Runtime_CUDA_Runtime_Version': (
                getattr(getattr(torch, 'version', None), 'cuda', None)
                if torch is not None else None),
            'Runtime_GPU_Names': gpu_names,
            'Runtime_Adaptive_Device_Request': self.adaptive_device_request,
            'Runtime_Require_PyTorch': bool(require_torch),
            'Runtime_Require_CUDA': bool(require_cuda),
            'Runtime_Torch_Deterministic': self.torch_deterministic,
            'Runtime_Torch_Strict_Determinism': self.torch_strict_determinism,
            'Runtime_Torch_Allow_TF32': self.torch_allow_tf32,
            'Runtime_Master_Seed': int(self.seed),
            'Runtime_Topology_Seed': int(self.seed),
            'Runtime_Event_Seed': int(self.seed) + 100_003,
            'Runtime_Attack_Seed': int(self.seed) + 200_003,
            'Runtime_Key_Seed': int(self.seed) + 300_007,
            'Runtime_TPM_Seed': int(self.seed) + 400_009,
            'Runtime_QIFL_Seed': int(self.ablation.get('qifl_seed', 42)),
            'Runtime_DQN_Seed': int(
                self.ablation.get('dqn_seed', self.seed + 1000)),
            'Runtime_Quantum_Claim_Scope': (
                'quantum-inspired probabilistic fuzzy representation; '
                'classical PyTorch/CUDA implementation'),
        }

    @staticmethod
    def _as_list(value, default=None):
        if value is None:
            value = [] if default is None else default
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return [] if default is None else list(default if isinstance(default, (list, tuple)) else [default])
            try:
                parsed = json.loads(text)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
                return [parsed]
            except Exception:
                return [v.strip() for v in text.split(',') if v.strip()]
        return [value]

    @staticmethod
    def _extend_to_count(values, count, cast=float):
        values = list(values or [])
        if not values:
            values = [cast(0)]
        out = []
        for idx in range(int(max(1, count))):
            v = values[idx] if idx < len(values) else values[-1]
            out.append(cast(v))
        return out

    @staticmethod
    def _nonnegative_float(value, default=0.0):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not math.isfinite(number):
            return float(default)
        return max(0.0, number)

    def _estimate_mobile_charger_count_cost_aware(self):
        """Return the smallest performance-feasible fleet size and diagnostics.

        The count is the maximum of five independent lower bounds: the legacy
        node-to-charger ratio, aggregate energy balance, service-time capacity,
        response-time coverage, and per-mission battery feasibility.  Because every charger has positive total
        cost of ownership, the smallest feasible count is also the least-cost
        feasible homogeneous fleet.  A budget can be reported as infeasible or
        used as an explicit hard cap; the default performance-first policy never
        reduces the legacy fleet size.
        """
        n = max(1, int(self.num_nodes))
        width = max(float(self.network_size[0]), 1e-9)
        height = max(float(self.network_size[1]), 1e-9)
        area = width * height
        diagonal = math.hypot(width, height)
        tx_range = max(float(self.transmission_range), 1e-9)
        speed = max(self._nonnegative_float(
            self.ablation.get('mobile_charger_speed', 5.0), 5.0), 1e-9)
        charging_rate = max(self._nonnegative_float(
            self.ablation.get('charging_rate', 5.0), 5.0), 1e-12)
        eta = float(np.clip(self.ablation.get(
            'charging_energy_efficiency',
            self.ablation.get('rf_energy_conversion_efficiency', 0.70)),
            1e-6, 1.0))
        utilization = float(np.clip(self.ablation.get(
            'mobile_charger_service_utilization', 0.70), 0.05, 1.0))
        safety = max(1.0, self._nonnegative_float(self.ablation.get(
            'mobile_charger_count_safety_factor', 1.10), 1.10))

        nodes_per_charger = max(1, int(self.ablation.get(
            'mobile_charger_nodes_per_charger', 40)))
        legacy_count = int(math.ceil(n / float(nodes_per_charger)))

        explicit_node_power = self.ablation.get(
            'mobile_charger_estimated_node_power_j_per_s', None)
        if explicit_node_power is not None:
            estimated_node_power = self._nonnegative_float(explicit_node_power)
        else:
            idle_power = self._nonnegative_float(
                self.ablation.get('idle_sensing_power_j_per_s', 0.0))
            event_rate = self._nonnegative_float(
                self.ablation.get('event_rate_lambda', 0.0))
            bits = max(1.0, self._nonnegative_float(
                self.ablation.get('data_packet_bits', self.data_packet_bits),
                self.data_packet_bits))
            e_elec = self._nonnegative_float(
                self.ablation.get('radio_e_elec_j_per_bit', 50e-9), 50e-9)
            eps_fs = self._nonnegative_float(
                self.ablation.get('radio_eps_fs_j_per_bit_m2', 10e-12), 10e-12)
            expected_source_distance = 0.40 * diagonal
            expected_hops = max(1.0, expected_source_distance / tx_range)
            expected_hop_distance = min(tx_range, expected_source_distance /
                                        expected_hops)
            tx_energy = bits * (e_elec + eps_fs * expected_hop_distance ** 2)
            rx_energy = bits * e_elec
            path_energy = tx_energy + max(0.0, expected_hops - 1.0) * (
                tx_energy + rx_energy)
            estimated_node_power = idle_power + event_rate * path_energy

        aggregate_demand_power = n * estimated_node_power * safety
        effective_service_power = charging_rate * eta * utilization
        energy_count = max(1, int(math.ceil(
            aggregate_demand_power / max(effective_service_power, 1e-12))))

        initial_energy = max(self._nonnegative_float(
            getattr(self, 'sensor_initial_energy',
                    self.ablation.get('initial_energy_override', 0.1)), 0.1),
            1e-12)
        recharge_fraction = float(np.clip(self.ablation.get(
            'mobile_charger_average_recharge_fraction', 0.50), 0.0, 1.0))
        request_fraction = float(np.clip(self.ablation.get(
            'mobile_charger_expected_request_fraction', 0.25), 0.0, 1.0))
        service_horizon = max(self._nonnegative_float(self.ablation.get(
            'mobile_charger_service_horizon_s',
            self.ablation.get('charge_deadline_horizon_s', 900.0)), 900.0),
            1e-6)
        reference_count = max(1, legacy_count, energy_count)
        service_radius = math.sqrt(area / (math.pi * reference_count))
        expected_travel_distance = (2.0 / 3.0) * service_radius
        expected_travel_time = expected_travel_distance / speed
        expected_charge_time = (initial_energy * recharge_fraction) / max(
            charging_rate * eta, 1e-12)
        dispatch_overhead = self._nonnegative_float(self.ablation.get(
            'mobile_charger_dispatch_overhead_s', 0.0))
        expected_mission_time = (
            expected_travel_time + expected_charge_time + dispatch_overhead)
        missions_per_horizon = n * request_fraction
        time_count = max(1, int(math.ceil(
            missions_per_horizon * expected_mission_time /
            max(service_horizon * utilization, 1e-12))))

        response_deadline = max(self._nonnegative_float(self.ablation.get(
            'mobile_charger_response_deadline_s',
            self.ablation.get('hard_deadline_margin_s', 90.0)), 90.0), 1e-6)
        response_margin = self._nonnegative_float(self.ablation.get(
            'mobile_charger_response_margin_s', 0.0))
        travel_budget = max(0.0, response_deadline - expected_charge_time -
                            dispatch_overhead - response_margin)
        reachable_radius = speed * travel_budget
        packing_efficiency = float(np.clip(self.ablation.get(
            'mobile_charger_spatial_packing_efficiency', 0.65), 0.10, 1.0))
        if reachable_radius <= 1e-12:
            coverage_count = n
        else:
            coverage_count = max(1, int(math.ceil(
                area / max(math.pi * reachable_radius ** 2 *
                           packing_efficiency, 1e-12))))

        min_count = max(1, int(self.ablation.get('min_mobile_chargers', 1)))
        max_count = max(min_count, int(self.ablation.get(
            'max_mobile_chargers', 8)))

        capacity_mode = str(self.ablation.get(
            'mobile_charger_capacity_mode', 'absolute')).strip().lower()
        if capacity_mode in {'relative', 'sensor_relative', 'sensor-equivalent',
                             'sensor_equivalent', 'node_relative', 'scaled'}:
            eq_values = self._as_list(self.ablation.get(
                'mobile_charger_sensor_equivalent_capacities', None),
                default=[self.ablation.get(
                    'mobile_charger_sensor_equivalent_capacity', 20.0)])
            configured_capacity = min(
                self._nonnegative_float(v) * initial_energy for v in eq_values)
        else:
            cap_values = self._as_list(self.ablation.get(
                'mobile_charger_capacities', None),
                default=[self.ablation.get('mobile_charger_capacity', 50000.0)])
            configured_capacity = min(self._nonnegative_float(v)
                                      for v in cap_values)
        travel_cost = self._nonnegative_float(self.ablation.get(
            'mobile_charger_travel_energy_cost_per_m', 0.1), 0.1)
        reserve_ratio = float(np.clip(self.ablation.get(
            'mc_reserve_ratio', 0.05), 0.0, 0.95))
        usable_capacity = configured_capacity * (1.0 - reserve_ratio)
        capacity_count = max(min_count, legacy_count, energy_count,
                             time_count, coverage_count)
        expected_mission_energy = float('inf')
        for candidate_count in range(capacity_count, max_count + 1):
            candidate_radius = math.sqrt(area / (math.pi * candidate_count))
            candidate_distance = (2.0 / 3.0) * candidate_radius
            candidate_mission_energy = (
                2.0 * candidate_distance * travel_cost
                + charging_rate * expected_charge_time)
            if usable_capacity + 1e-12 >= candidate_mission_energy:
                capacity_count = candidate_count
                expected_mission_energy = candidate_mission_energy
                break
        else:
            capacity_count = max_count + 1
            candidate_radius = math.sqrt(area / (math.pi * max_count))
            candidate_distance = (2.0 / 3.0) * candidate_radius
            expected_mission_energy = (
                2.0 * candidate_distance * travel_cost
                + charging_rate * expected_charge_time)

        raw_count = max(legacy_count, energy_count, time_count, coverage_count,
                        capacity_count)
        unconstrained_count = max(min_count, raw_count)
        selected_count = min(max_count, unconstrained_count)
        upper_bound_limited = unconstrained_count > max_count

        purchase = self._nonnegative_float(self.ablation.get(
            'mobile_charger_unit_purchase_cost', 0.0))
        integration = self._nonnegative_float(self.ablation.get(
            'mobile_charger_unit_integration_cost', 0.0))
        annual_maintenance = self._nonnegative_float(self.ablation.get(
            'mobile_charger_annual_maintenance_cost', 0.0))
        life_years = max(1.0, self._nonnegative_float(self.ablation.get(
            'mobile_charger_economic_life_years', 3.0), 3.0))
        unit_tco = purchase + integration + annual_maintenance * life_years
        budget_value = self.ablation.get('mobile_charger_total_budget', None)
        budget = (None if budget_value in (None, '', 'none', 'None')
                  else self._nonnegative_float(budget_value))
        budget_count = None
        budget_feasible = True
        if budget is not None and unit_tco > 0.0:
            budget_count = int(math.floor(budget / unit_tco))
            budget_feasible = budget_count >= selected_count
            policy = str(self.ablation.get(
                'mobile_charger_budget_policy', 'performance_first')).strip().lower()
            if policy in {'hard_cap', 'budget_first', 'strict_budget'}:
                selected_count = max(min_count, min(selected_count, budget_count))

        diagnostics = {
            'formula': 'max(legacy,energy_balance,service_time,response_coverage,mission_capacity)',
            'legacy_count': int(legacy_count),
            'energy_balance_count': int(energy_count),
            'service_time_count': int(time_count),
            'response_coverage_count': int(coverage_count),
            'mission_capacity_count': int(capacity_count),
            'configured_min_capacity_j': float(configured_capacity),
            'usable_capacity_after_reserve_j': float(usable_capacity),
            'expected_mission_energy_j': float(expected_mission_energy),
            'mission_capacity_feasible': bool(capacity_count <= max_count),
            'unconstrained_count': int(unconstrained_count),
            'selected_count': int(max(1, selected_count)),
            'estimated_node_power_j_per_s': float(estimated_node_power),
            'aggregate_demand_power_j_per_s': float(aggregate_demand_power),
            'effective_service_power_j_per_s': float(effective_service_power),
            'expected_mission_time_s': float(expected_mission_time),
            'response_reachable_radius_m': float(reachable_radius),
            'unit_tco': float(unit_tco),
            'fleet_tco': float(unit_tco * max(1, selected_count)),
            'budget': budget,
            'budget_count': budget_count,
            'budget_feasible': bool(budget_feasible),
            'upper_bound_limited': bool(upper_bound_limited),
        }
        return max(1, int(selected_count)), diagnostics

    def _resolve_mobile_charger_count(self, capacity_values):
        mode = str(self.ablation.get('mobile_charger_selection_mode',
                                     self.ablation.get('mc_selection_mode', 'hardcoded'))).strip().lower()
        explicit = self.ablation.get('num_mobile_chargers',
                                     self.ablation.get('mobile_charger_count', None))
        if mode in ('dynamic', 'auto', 'adaptive', 'cost_aware', 'economic'):
            count, diagnostics = self._estimate_mobile_charger_count_cost_aware()
            self.mobile_charger_count_diagnostics = diagnostics
            if hasattr(self, 'metrics'):
                self.metrics.experiment_parameters.update({
                    f'MC_Count_{key}': value for key, value in diagnostics.items()
                })
            return count
        self.mobile_charger_count_diagnostics = {
            'formula': 'hardcoded_or_capacity_list',
            'selected_count': None,
        }
        if explicit is not None:
            count = max(1, int(explicit))
        elif len(capacity_values) > 1:
            count = len(capacity_values)
        else:
            count = 1
        self.mobile_charger_count_diagnostics['selected_count'] = int(count)
        if hasattr(self, 'metrics'):
            self.metrics.experiment_parameters.update({
                'MC_Count_formula': 'hardcoded_or_capacity_list',
                'MC_Count_selected_count': int(count),
            })
        return count

    def _resolve_mobile_charger_capacities(self, count, capacity_values):
        mode = str(self.ablation.get(
            'mobile_charger_capacity_mode', 'absolute')).strip().lower()
        relative_modes = {
            'relative', 'sensor_relative', 'sensor-equivalent',
            'sensor_equivalent', 'node_relative', 'scaled'
        }
        if mode in relative_modes:
            equiv_default = self.ablation.get(
                'mobile_charger_sensor_equivalent_capacity', 20.0)
            equiv_values = self._as_list(
                self.ablation.get(
                    'mobile_charger_sensor_equivalent_capacities', None),
                default=[equiv_default])
            equivalents = self._extend_to_count(
                equiv_values, count, cast=float)
            sensor_energy = max(float(self.sensor_initial_energy), 0.0)
            return [max(0.0, eq) * sensor_energy for eq in equivalents]

        return self._extend_to_count(capacity_values, count, cast=float)

    def _resolve_mobile_charger_travel_costs(self, count):
        mode = str(self.ablation.get(
            'mobile_charger_travel_cost_mode', 'absolute')).strip().lower()
        relative_modes = {'relative', 'sensor_relative',
                          'sensor_equivalent', 'scaled'}
        if mode in relative_modes:
            default_equiv = self.ablation.get(
                'mobile_charger_travel_sensor_equivalent_cost_per_m', 0.02)
            equiv_values = self._as_list(
                self.ablation.get(
                    'mobile_charger_travel_sensor_equivalent_costs_per_m', None),
                default=[default_equiv])
            equivalents = self._extend_to_count(
                equiv_values, count, cast=float)
            sensor_energy = max(float(self.sensor_initial_energy), 0.0)
            return [max(0.0, eq) * sensor_energy for eq in equivalents]

        default_cost = float(self.ablation.get(
            'mobile_charger_travel_energy_cost_per_m', 0.1))
        cost_values = self._as_list(
            self.ablation.get(
                'mobile_charger_travel_energy_costs_per_m', None),
            default=[default_cost])
        return self._extend_to_count(cost_values, count, cast=float)

    def _make_mobile_charger_fleet(self, bs_x, bs_y):
        default_capacity = float(self.ablation.get(
            'mobile_charger_capacity', 50000.0))
        capacity_values = self._as_list(
            self.ablation.get('mobile_charger_capacities', None),
            default=[default_capacity])
        count = self._resolve_mobile_charger_count(capacity_values)
        capacities = self._resolve_mobile_charger_capacities(
            count, capacity_values)
        travel_costs = self._resolve_mobile_charger_travel_costs(count)
        speed_values = self._as_list(
            self.ablation.get('mobile_charger_speeds', None),
            default=[float(self.ablation.get('mobile_charger_speed', 5.0))])
        speeds = self._extend_to_count(speed_values, count, cast=float)
        id_start = int(self.ablation.get('mobile_charger_id_start', 999))

        chargers = []
        for idx in range(count):
            charger = MobileCharger(
                mc_id=id_start + idx, depot_x=bs_x, depot_y=bs_y,
                capacity=capacities[idx], speed=speeds[idx],
                base_station=self.bs, metrics_collector=self.metrics,
                attack_config=self.ablation,
                rng_seed=int(self.seed) * 2_000_003 + idx)
            charger.travel_energy_cost_per_m = float(travel_costs[idx])
            charger.charging_rate = float(
                self.ablation.get('charging_rate', 5.0))
            chargers.append(charger)
        return chargers

    def _deploy_network(self, bs_x, bs_y):
        node_radio_kwargs = {
            'data_packet_bytes': self.data_packet_bytes,
            'data_packet_bits': self.data_packet_bits,
            'control_packet_bits': self.control_packet_bits,
            'radio_e_elec_j_per_bit': float(self.ablation.get(
                'radio_e_elec_j_per_bit', 50e-9)),
            'radio_eps_fs_j_per_bit_m2': float(self.ablation.get(
                'radio_eps_fs_j_per_bit_m2', 10e-12)),
            'radio_eps_mp_j_per_bit_m4': float(self.ablation.get(
                'radio_eps_mp_j_per_bit_m4', 0.0013e-12)),
            'radio_e_da_j_per_bit': float(self.ablation.get(
                'radio_e_da_j_per_bit', 5e-9)),
        }
        self.nodes[self.BS_ID] = SensorNode(
            self.BS_ID, bs_x, bs_y, float('inf'), self.metrics,
            rng_seed=int(self.seed) * 1_000_003,
            attack_config=self.ablation, **node_radio_kwargs)
        initial_energy = float(self.sensor_initial_energy)
        for i in range(1, self.num_nodes + 1):
            x = self.topology_rng.uniform(0, self.network_size[0])
            y = self.topology_rng.uniform(0, self.network_size[1])
            self.nodes[i] = SensorNode(
                i, x, y, initial_energy, self.metrics,
                rng_seed=int(self.seed) * 1_000_003 + i,
                attack_config=self.ablation, **node_radio_kwargs)

        self.bs.network_nodes = self.nodes
        self.bs.mc_reference = self.mc
        self.bs.mc_references = self.mobile_chargers
        self.routing_engine.quarantined_zones = self.bs.quarantined_zones
        self.data_routing_engine.quarantined_zones = self.bs.quarantined_zones
        for _node in self.nodes.values():
            _node.network_nodes = self.nodes

    def bootstrap_network(self):
        print(
            f"[Seed {self.seed}] Executing Rigorous Bootstrapping Phase (Gradient & Source Routing)...")

        BS_ADV_SIZE = 16
        REG_SIZE = 48
        POLICY_SIZE = 64

        def get_physical_neighbors(node_id):
            n1 = self.nodes[node_id]
            neighbors = []
            for n2_id, n2 in self.nodes.items():
                if node_id != n2_id:
                    dist = math.hypot(n1.x - n2.x, n1.y - n2.y)
                    if dist <= self.transmission_range:
                        neighbors.append(n2_id)
            return neighbors

        seq_num = 1
        wavefront = [self.BS_ID]
        current_hop = 1

        self.nodes[self.BS_ID]._consume_energy(
            tx_bytes=BS_ADV_SIZE, distance=self.transmission_range)
        self.nodes[self.BS_ID].seen_adv_seq = seq_num
        self.nodes[self.BS_ID].hop_count_to_bs = 0

        while wavefront:
            next_wave = []
            for sender_id in wavefront:
                neighbors = get_physical_neighbors(sender_id)
                for neighbor_id in neighbors:
                    neighbor_node = self.nodes[neighbor_id]
                    if neighbor_id == self.BS_ID:
                        continue
                    if neighbor_node.seen_adv_seq < seq_num:
                        neighbor_node.seen_adv_seq = seq_num
                        neighbor_node.hop_count_to_bs = current_hop
                        neighbor_node.temp_parent = sender_id
                        neighbor_node._consume_energy(rx_bytes=BS_ADV_SIZE)
                        neighbor_node._consume_energy(
                            tx_bytes=BS_ADV_SIZE, distance=self.transmission_range)
                        self.metrics.update_packet_transmission(
                            is_critical=False, is_control=True, size_bytes=BS_ADV_SIZE, overhead_phase='setup')
                        next_wave.append(neighbor_id)
            wavefront = next_wave
            current_hop += 1

        self.current_time += 1.5

        for n_id, node in self.nodes.items():
            if n_id == self.BS_ID:
                continue

            node.crypto_keys = f"AES-128-Key-Node{node.node_id}-{self.key_rng.randint(1000, 9999)}"
            current_n_id = n_id
            packet_dropped = False

            while current_n_id != self.BS_ID:
                current_node = self.nodes[current_n_id]
                parent_id = current_node.temp_parent
                if parent_id is None:
                    packet_dropped = True
                    break
                parent_node = self.nodes[parent_id]
                dist = math.hypot(current_node.x - parent_node.x,
                                  current_node.y - parent_node.y)
                current_node._consume_energy(tx_bytes=REG_SIZE, distance=dist)
                if parent_id != self.BS_ID:
                    parent_node._consume_energy(rx_bytes=REG_SIZE)
                self._account_implicit_mac_ack(current_n_id, parent_id, dist)
                current_n_id = parent_id

            if not packet_dropped:
                self.metrics.update_packet_transmission(
                    is_critical=False, is_control=True, size_bytes=REG_SIZE, overhead_phase='setup')

                self.bs.node_states[node.node_id] = {
                    'x': node.x, 'y': node.y,
                    'E_initial': node.E_max,
                    'E_claimed': node.E_max,
                    'E_claimed_prev': node.E_max,
                    'E_claimed_prev_SED': node.E_max,
                    'E_est': node.E_current,
                    'E_consumption_est_cumulative': 0.0,
                    'total_injected_energy': 0.0,
                    'trust_vector': [1.0, 1.0, 1.0, 1.0, 1.0],
                    'T1_history': 1.0, 'T2_history': 1.0, 'T3_history': 1.0, 'T4_history': 1.0, 'T5_history': 1.0,
                    'FE': 1.0,
                    'status': 'WHITELISTED',
                    'crypto_keys': node.crypto_keys,
                    'hard_veto': False,
                    'spoof_trust': 1.0,
                    'spoof_rescue_triggered': False,
                    'Deviation_MC_norm': 0.0,
                    'SED_index': 0.0,
                    'S_stealth': 0.0,
                    'risk_mean': 0.0,
                    'risk_instant': 0.0,
                    'risk_lcb': 0.0,
                    'risk_uncertainty': 0.0,
                    'fcm_risk': 0.0,
                    'prev_epoch_seq': 0,
                    'lambda_gen': 0.0,
                    'blacklist_counter': 0,
                    'idle_sensing_energy_epoch': 0.0,
                    'energy_q_ewma': 0.0,
                    'physical_eps_ewma': 0.0,
                    'physical_energy_sample': node.E_current,
                    'physical_energy_sample_time': 0.0,
                    'energy_injected_since_physical_sample': 0.0,
                    'energy_last_report_time': 0.0,
                    'energy_last_reported': node.E_current,
                    'charge_audit_cycle': 0,
                    'charging_pool_class': None,
                    'adaptive_charge_threshold': 0.0,
                    'charge_satisfied': False,
                    'hop_count_to_bs_norm': 0.0,
                    'pending_mc_audit_energy': 0.0,
                    'pending_mc_expected_post_energy': node.E_current,
                    'energy_probation': False,
                    'energy_recovery_streak': 0,
                    'energy_violation_count': 0,
                    'hard_veto_reason': None,
                    'hard_veto_recoverable': False,
                    'energy_audit_trust': 1.0,
                    'energy_audit_recovery_streak': 0,
                    'energy_recovery_reference_energy': node.E_current,
                    'energy_recovery_reference_time': 0.0,
                    'energy_recovery_last_epoch': None,
                    'last_node_charge_receipt_consistent': None,
                }

        self.current_time += 2.0

        active_nodes_for_bs = {nid: self.nodes[nid]
                               for nid in self.bs.node_states.keys()}
        active_nodes_for_bs[self.BS_ID] = self.nodes[self.BS_ID]

        # Build the unchanged reference control/charging routes first.
        initial_tables = self.routing_engine.compute_routing_tables(
            active_nodes_for_bs)
        self.bs.routing_tables = initial_tables

        # Build the DATA-only clustered/MST overlay against the reference.
        initial_data_tables = self.data_routing_engine.compute_routing_tables(
            active_nodes_for_bs, reference_tables=initial_tables)
        self.bs.data_routing_tables = initial_data_tables
        try:
            self.metrics.record_data_routing_diagnostics(
                self.data_routing_engine.last_diagnostics)
        except Exception:
            pass

        # Charging clustering continues to use only initial_tables.
        self.bs._network_clustering()

        bs_internal_graph = self.routing_engine.build_network_graph(
            active_nodes_for_bs, node_penalties={}, w_d=0.33, w_e=0.33, w_p=0.34)

        adaptive_threshold = 75.0

        for _nid, _table in initial_tables.items():
            if _nid in self.bs.node_states:
                _path = _table.get('primary_path') or []
                self.bs.node_states[_nid]['hop_count_to_bs_norm'] = min(
                    1.0, max(0, len(_path) - 1) / 10.0)

        for n_id, table in initial_tables.items():
            if n_id == self.BS_ID:
                continue
            try:
                source_route = nx.shortest_path(
                    bs_internal_graph, source=self.BS_ID, target=n_id, weight='weight')
            except nx.NetworkXNoPath:
                continue

            for i in range(len(source_route) - 1):
                sender_id = source_route[i]
                receiver_id = source_route[i+1]
                sender = self.nodes[sender_id]
                receiver = self.nodes[receiver_id]
                dist = math.hypot(sender.x - receiver.x, sender.y - receiver.y)
                if sender_id != self.BS_ID:
                    sender._consume_energy(tx_bytes=POLICY_SIZE, distance=dist)
                if receiver_id != n_id:
                    receiver._consume_energy(rx_bytes=POLICY_SIZE)
                self._account_implicit_mac_ack(sender_id, receiver_id, dist)

            target_node = self.nodes[n_id]
            target_node._consume_energy(rx_bytes=POLICY_SIZE)
            target_node.update_policies_from_bs(
                table['primary'], table['secondary'], adaptive_threshold,
                secondary_mode=table.get('secondary_mode', 'redundant'))
            data_table = initial_data_tables.get(n_id, {})
            target_node.update_data_routes(
                data_table.get('primary'), data_table.get('secondary'),
                secondary_mode=data_table.get('secondary_mode', 'redundant'))
            target_node.temp_parent = None
            target_node.hop_count_to_bs = float('inf')
            self.metrics.update_packet_transmission(
                is_critical=False, is_control=True, size_bytes=POLICY_SIZE, overhead_phase='setup')

        self.current_time += 1.5

        self.bs.mc_shared_keys = {}
        for idx, charger in enumerate(self.mobile_chargers):
            charger.crypto_keys = f"MC-SECURE-KEY-{charger.mc_id}-{self.key_rng.randint(1000, 9999)}"
            self.bs.mc_shared_keys[str(charger.mc_id)] = charger.crypto_keys
            if idx == 0:
                self.bs.mc_shared_key = charger.crypto_keys
            self.metrics.update_packet_transmission(
                is_critical=False, is_control=True, size_bytes=32, overhead_phase='setup')
        self.current_time += 0.5

        self.metrics.experiment_parameters.update({
            'Bootstrap_Setup_Time_s': float(self.current_time),
            'Bootstrap_Data_Sensing_Enabled_Flag': False,
            'Bootstrap_Setup_Energy_And_Overhead_Accounted_Flag': True,
        })
        print(
            f"[Seed {self.seed}] Bootstrapping Complete. Temp trees cleared. Network Setup Time: {self.current_time}s")

    def _record_energy_sentinel_requests(self):
        if not self.energy_experiment_active:
            return
        revival_enabled = bool(self.ablation.get(
            'enable_dead_node_revival', True))
        for nid, node in self.nodes.items():
            if nid == self.BS_ID:
                continue
            if node.status not in ('ALIVE', 'DEAD'):
                continue
            if node.status == 'DEAD' and not revival_enabled:
                continue
            if node.E_current <= node.E_critical:
                self.metrics.record_energy_critical(nid, self.current_time)
                if self.ablation.get('use_deadline_safe_energy_sentinel', True):
                    st = self.bs.node_states.get(nid)
                    if st is not None:
                        actual_e = float(node.E_current)
                        st['E_est'] = min(
                            float(st.get('E_est', actual_e)), actual_e)
                        st['E_claimed'] = min(
                            float(st.get('E_claimed', actual_e)), actual_e)
                        st['energy_last_reported'] = min(
                            float(st.get('energy_last_reported', actual_e)), actual_e)
                        st['energy_last_report_time'] = float(
                            self.current_time)
                        st['physical_node_status'] = str(node.status)
                        st['revival_requested'] = bool(node.status == 'DEAD')
                    self.bs.pending_sos_requests.add(nid)

    def service_mobile_charger(self):
        self._record_energy_sentinel_requests()
        if len(self.mobile_chargers) <= 1:
            return self._service_single_mobile_charger_legacy()
        return self._service_multi_mobile_chargers()

    def _service_single_mobile_charger_legacy(self):
        if (not self.bs.energy_experiment_active
                or self.mc.status != 'IDLE'
                or self.mc.mc_id in self.active_charger_missions):
            return

        target_node = None
        adaptive_mode = bool(
            self.ablation.get('use_adaptive_quantum_charging', False)
            or self.ablation.get('use_adaptive_charge_guards', False)
            or self.ablation.get('use_dqn_charger_routing', False))
        in_service = self._active_mission_node_ids()

        if adaptive_mode:
            if not self.bs.secure_charging_queue:
                self.bs.build_adaptive_charging_queue()

            valid_candidates = []
            for candidate in list(self.bs.secure_charging_queue):
                if getattr(candidate, 'is_cluster', False):
                    candidate.nodes = [
                        node for node in candidate.nodes
                        if node.node_id in self.nodes
                        and node.node_id not in in_service
                        and node.status in ('ALIVE', 'DEAD')
                        and self.bs.validate_charging_request(
                            node.node_id,
                            emergency=(node.node_id in self.bs.pending_sos_requests))
                    ]
                    if candidate.nodes:
                        candidate.recompute_geometry()
                        valid_candidates.append(candidate)
                elif candidate in self.nodes and candidate not in in_service:
                    node = self.nodes[candidate]
                    if node.status in ('ALIVE', 'DEAD') and self.bs.validate_charging_request(
                            candidate, emergency=(candidate in self.bs.pending_sos_requests)):
                        valid_candidates.append(candidate)

            self.bs.secure_charging_queue[:] = valid_candidates
            if valid_candidates:
                target_node = self._select_target_for_charger(
                    self.mc, valid_candidates)
        else:
            secure_queue = self.bs.secure_charging_queue
            secure_queue[:] = [
                nid for nid in secure_queue
                if nid not in in_service
                and self.bs.validate_charging_request(nid, emergency=True)]
            candidate_ids = list(secure_queue)
            if not candidate_ids:
                candidate_ids = [
                    nid for nid in self.bs.pending_sos_requests
                    if nid in self.nodes and nid not in in_service
                    and self.bs.validate_charging_request(nid, emergency=True)]
            if candidate_ids:
                target_node = self._select_target_for_charger(
                    self.mc, candidate_ids)
            else:
                target_node = self._select_predictive_charging_target()

        if target_node is None:
            return

        target_ids = [node.node_id for node in self._target_nodes(target_node)]
        selected_emergency = any(
            nid in self.bs.pending_sos_requests
            or self.bs.is_deadline_rescue_node(nid, self.current_time)
            for nid in target_ids)
        if not self._dispatch_charger_mission(
                self.mc, target_node, selected_emergency):
            return

        if target_node in self.bs.secure_charging_queue:
            self.bs.secure_charging_queue.remove(target_node)
        elif not getattr(target_node, 'is_cluster', False):
            target_id = target_node.node_id
            if target_id in self.bs.secure_charging_queue:
                self.bs.secure_charging_queue.remove(target_id)
        for nid in target_ids:
            self.bs.pending_sos_requests.discard(nid)

    def _cluster_center(self, node_ids):
        coords = [(float(self.nodes[nid].x), float(self.nodes[nid].y))
                  for nid in node_ids if nid in self.nodes]
        if not coords:
            return (self.network_size[0] / 2.0, self.network_size[1] / 2.0)
        return (sum(x for x, _ in coords) / len(coords),
                sum(y for _, y in coords) / len(coords))

    def _assign_idle_chargers_to_clusters(self, idle_chargers, clusters):
        assignments = []
        remaining = sorted(list(idle_chargers), key=lambda mc: int(mc.mc_id))
        cluster_items = [(cid, list(members)) for cid, members in sorted(
            (clusters or {}).items()) if members]
        for cid, members in cluster_items:
            if not remaining:
                break
            cx, cy = self._cluster_center(members)
            chosen = min(remaining, key=lambda mc: (math.hypot(
                float(mc.x) - cx, float(mc.y) - cy), int(mc.mc_id)))
            assignments.append((chosen, cid, members))
            remaining.remove(chosen)
        return assignments

    @staticmethod
    def _target_nodes(target_node):
        return list(getattr(target_node, 'nodes', [target_node]))

    @staticmethod
    def _target_coords(target_node):
        if getattr(target_node, 'is_cluster', False):
            return float(target_node.centroid_x), float(target_node.centroid_y)
        return float(target_node.x), float(target_node.y)

    def _candidate_target_object(self, candidate):
        if isinstance(candidate, (int, np.integer)):
            return self.nodes.get(int(candidate))
        return candidate

    def _target_is_emergency(self, candidate):
        target = self._candidate_target_object(candidate)
        if target is None:
            return False
        return any(
            node.node_id in self.bs.pending_sos_requests
            or self.bs.is_deadline_rescue_node(node.node_id, self.current_time)
            or node.status == 'DEAD'
            for node in self._target_nodes(target)
        )

    def _stable_region_owner_id(self, candidate):
        target = self._candidate_target_object(candidate)
        chargers = sorted(self.mobile_chargers, key=lambda mc: int(mc.mc_id))
        if target is None or not chargers:
            return None
        tx, ty = self._target_coords(target)
        cx = float(self.nodes[self.BS_ID].x)
        cy = float(self.nodes[self.BS_ID].y)
        if math.hypot(tx - cx, ty - cy) <= 1e-12:
            return int(chargers[0].mc_id)
        angle = (math.atan2(ty - cy, tx - cx) + 2.0 * math.pi) % (
            2.0 * math.pi)
        index = min(len(chargers) - 1, int(
            math.floor(angle / (2.0 * math.pi) * len(chargers))))
        return int(chargers[index].mc_id)

    def _target_interference_radius(self, target):
        target = self._candidate_target_object(target)
        if target is None:
            return 0.0
        if getattr(target, 'is_cluster', False):
            return max(0.0, float(self.ablation.get(
                'rf_charging_radius', 25.0)))
        return 0.5 * max(0.0, float(self.ablation.get(
            'mobile_charger_physical_footprint_m', 0.0)))

    def _targets_spatially_conflict(self, first, second):
        a = self._candidate_target_object(first)
        b = self._candidate_target_object(second)
        if a is None or b is None:
            return False
        ax, ay = self._target_coords(a)
        bx, by = self._target_coords(b)
        safe_gap = max(0.0, float(self.ablation.get(
            'mobile_charger_safe_distance_m', 0.0)))
        required = (self._target_interference_radius(a) +
                    self._target_interference_radius(b) + safe_gap)
        if required <= 0.0:
            return False
        return math.hypot(ax - bx, ay - by) + 1e-9 < required

    def _candidate_has_active_interference_conflict(self, candidate,
                                                     reserved_targets=()):
        if not self.ablation.get(
                'use_interference_safe_multi_mc_dispatch', False):
            return False
        for mission in self.active_charger_missions.values():
            if self._targets_spatially_conflict(candidate, mission.get('target')):
                return True
        return any(self._targets_spatially_conflict(candidate, other)
                   for other in reserved_targets)

    def _relative_mc_feasibility_guard_enabled(self):
        return bool(self.ablation.get('use_universal_mc_feasibility_guard', True))

    def _minimum_energy_allocations(self, target_node):
        min_ratio = float(self.ablation.get(
            'minimum_charge_energy_ratio', 0.08))
        allocations = {}
        for node in self._target_nodes(target_node):
            missing = max(0.0, float(node.E_max) - float(node.E_current))
            allocations[node.node_id] = min(
                missing, min_ratio * max(float(node.E_max), 0.0))
        return allocations

    def _charger_required_energy_for_mission(self, charger, target_node, include_min_charge=True):
        if target_node is None:
            return float('inf')
        tx, ty = self._target_coords(target_node)
        outbound = math.hypot(float(charger.x) - tx, float(charger.y) - ty)
        return_leg = math.hypot(tx - float(charger.depot_x),
                                ty - float(charger.depot_y))
        travel_energy = (outbound + return_leg) * float(
            getattr(charger, 'travel_energy_cost_per_m', 0.0))
        if not include_min_charge:
            return travel_energy
        allocations = self._minimum_energy_allocations(target_node)
        charging_energy = charger.estimate_charging_energy(
            target_node, allocations)
        return travel_energy + charging_energy

    def _charger_can_usefully_serve(self, charger, target_node):
        if not self._relative_mc_feasibility_guard_enabled():
            return True
        required = self._charger_required_energy_for_mission(
            charger, target_node, include_min_charge=True)
        reserve_ratio = float(self.ablation.get('mc_reserve_ratio', 0.05))
        reserve = reserve_ratio * max(float(charger.E_capacity), 0.0)
        return float(charger.E_current) + 1e-12 >= required + reserve

    def _return_to_depot_if_relative_mc_infeasible(self, charger, target_node):
        if not self._relative_mc_feasibility_guard_enabled() or target_node is None:
            return 0.0
        if self._charger_can_usefully_serve(charger, target_node):
            return 0.0
        at_depot = math.hypot(
            float(charger.x) - float(charger.depot_x),
            float(charger.y) - float(charger.depot_y)) <= 1e-9
        if at_depot:
            self.metrics.mc_infeasible_mission_rejections += 1
            return 0.0
        return_time = charger.return_to_depot()
        if return_time is None:
            self.metrics.mc_stranded_events += 1
            return 0.0
        return float(return_time)

    def _safe_policy_candidates(self, charger, candidate_nodes):
        """Apply the same hard-rescue subset used by the DQN safety shield."""
        if len(candidate_nodes) <= 1:
            return list(candidate_nodes)
        physical = [ChargerRoutingDQN._candidate_energy_and_deadline(
            charger.x, charger.y, candidate, self.bs.node_states,
            charger.speed) for candidate in candidate_nodes]
        allowed = []
        for candidate, values in zip(candidate_nodes, physical):
            critical_ratio = max(
                float(self.charger_dqn.critical_energy_ratio),
                float(values.get('critical_energy_ratio', 0.0)))
            if (float(values['energy_ratio']) <= critical_ratio or
                    float(values['deadline_slack']) <=
                    float(self.charger_dqn.deadline_safety_margin)):
                allowed.append(candidate)
        return allowed if allowed else list(candidate_nodes)

    def _select_baseline_policy_target(self, policy, charger, candidate_nodes):
        policy = str(policy).strip().lower()
        if policy in {'random', 'random_safe'}:
            safe = self._safe_policy_candidates(charger, candidate_nodes)
            return self.policy_rng.choice(safe) if safe else None
        if policy in {'nearest', 'nearest_first'}:
            safe = self._safe_policy_candidates(charger, candidate_nodes)
            return min(safe, key=lambda node: charger.distance_to_target(node))
        if policy in {'lowest_energy', 'lowest_energy_first'}:
            safe = self._safe_policy_candidates(charger, candidate_nodes)
            def energy_ratio(candidate):
                attrs = ChargerRoutingDQN._extract_physical_attributes(
                    candidate, self.bs.node_states)
                return float(attrs['estimated_energy']) / max(
                    float(attrs['initial_energy']), 1e-12)
            return min(safe, key=energy_ratio)
        if policy in {'earliest_deadline', 'earliest_deadline_first'}:
            safe = self._safe_policy_candidates(charger, candidate_nodes)
            return min(safe, key=lambda candidate:
                       ChargerRoutingDQN._candidate_energy_and_deadline(
                           charger.x, charger.y, candidate,
                           self.bs.node_states, charger.speed)['deadline_slack'])
        return None

    def _select_target_for_charger(self, charger, candidate_ids, assigned_nodes=None):
        assigned_nodes = set() if assigned_nodes is None else set(assigned_nodes)
        candidate_nodes = []
        for candidate in candidate_ids:
            if getattr(candidate, 'is_cluster', False):
                if candidate.node_id not in assigned_nodes:
                    candidate_nodes.append(candidate)
            elif candidate not in assigned_nodes and candidate in self.nodes:
                node = self.nodes[candidate]
                if node.status == 'ALIVE':
                    candidate_nodes.append(node)
        if not candidate_nodes:
            return None

        feasible = [candidate for candidate in candidate_nodes
                    if self._charger_can_usefully_serve(charger, candidate)]
        if feasible:
            candidate_nodes = feasible
        elif self._relative_mc_feasibility_guard_enabled():
            return None

        policy = self.charger_target_policy
        if self.use_dqn_charger_routing and policy in {
                'dqn', 'dqn_safe', 'dqn_with_safety_shield'}:
            network_diag = math.hypot(*self.network_size)
            selected = self.charger_dqn.select(
                charger.x, charger.y, candidate_nodes, self.bs.node_states,
                speed=charger.speed, network_diag=network_diag,
                decision_time=self.current_time, charger_id=charger.mc_id)
            self.metrics.dqn_routing_decisions += 1
            explanation = self.charger_dqn.last_decision_explanation or {}
            if len(candidate_nodes) > 1:
                self.metrics.dqn_competitive_decisions += 1
            else:
                self.metrics.dqn_single_candidate_decisions += 1
            if explanation.get('exploration_draw'):
                self.metrics.dqn_effective_exploration_count += 1
            if explanation.get('safety_override'):
                self.metrics.dqn_safety_override_count += 1
            if self.enable_dqn_audit_logging:
                self.metrics.record_dqn_decision_audit(explanation)
            return selected

        baseline_selected = self._select_baseline_policy_target(
            policy, charger, candidate_nodes)
        if baseline_selected is not None:
            return baseline_selected

        # Original heuristic-only behavior. This is intentionally retained as
        # the fallback and as the explicit heuristic_only policy.
        def score(node):
            member_ids = [n.node_id for n in self._target_nodes(node)]
            priorities = [self.bs.charging_dispatch_priority(
                nid, current_time=self.current_time, mc_x=charger.x,
                mc_y=charger.y, mc_speed=charger.speed) for nid in member_ids]
            return min(priorities)
        return min(candidate_nodes, key=score)

    def _service_horizon_for_node(self, node_id, charger=None, emergency=False):
        cooldown = max(1.0, float(self.bs.charging_reentry_cooldown_s))
        fallback = float(self.ablation.get(
            'service_cycle_fallback_s', 1.25 * cooldown))

        revisit_samples = list(self.metrics.mc_revisit_intervals[-60:])
        observed_revisit = (
            float(np.percentile(revisit_samples, 60))
            if revisit_samples else fallback)

        request_waits = [
            self.metrics.get_charging_request_wait_time(nid, self.current_time)
            for nid in getattr(self.bs, 'service_charging_node_ids', set())
        ]
        queue_wait_p75 = (
            float(np.percentile(request_waits, 75))
            if request_waits else 0.0)

        recent_service = list(self.metrics.mc_service_durations[-60:])
        average_service = (
            float(np.mean(recent_service))
            if recent_service else float(self.ablation.get(
                'adaptive_expected_mission_time_s', 30.0)))
        latency = self.bs.estimate_service_latency_s(
            node_id, current_time=self.current_time, charger=charger)

        state = self.bs.node_states.get(node_id, {})
        pool = str(state.get('charging_pool_class') or 'service').lower()
        max_queue = max(1, int(self.ablation.get(
            'max_adaptive_charge_queue', 8)))
        active_service = len(getattr(
            self.bs, 'service_charging_node_ids', set()))
        queue_pressure = min(1.0, active_service / float(max_queue))

        if pool == 'plus' and not emergency:
            horizon = min(
                max(latency, 0.60 * cooldown),
                float(self.ablation.get(
                    'opportunistic_plus_horizon_s', 150.0)))
            return float(max(30.0, horizon))

        raw_horizon = max(
            cooldown,
            fallback,
            observed_revisit,
            queue_wait_p75 + average_service,
            latency)
        base_safety = max(1.0, float(self.ablation.get(
            'service_horizon_safety_multiplier', 1.12)))
        dynamic_safety = 1.0 + (base_safety - 1.0) * \
            (1.0 - 0.60 * queue_pressure)
        horizon = raw_horizon * dynamic_safety
        cap_key = ('emergency_service_horizon_cap_s' if emergency
                   else 'balanced_service_horizon_cap_s')
        cap_default = 720.0 if emergency else 420.0
        cap = max(cooldown, float(self.ablation.get(cap_key, cap_default)))
        horizon = min(horizon, cap)
        return float(max(cooldown, horizon))

    def _survival_requirement(self, node, charger=None, emergency=False):
        state = self.bs.node_states.get(node.node_id, {})
        e_max = max(float(node.E_max), 1e-12)
        current = max(0.0, float(node.E_current))
        eps = self.bs._effective_depletion_rate(
            node.node_id, self.current_time)
        horizon = self._service_horizon_for_node(
            node.node_id, charger=charger, emergency=emergency)
        pool = str(state.get('charging_pool_class') or 'service').lower()

        comm_reserve = float(self.ablation.get(
            'communication_energy_reserve_ratio', 0.04)) * e_max
        sos_reserve = float(state.get(
            'last_sos_threshold_energy',
            self.ablation.get('sos_energy_margin_j', 2.5e-4)))
        physical_required_after = eps * horizon + comm_reserve + sos_reserve

        if pool == 'plus' and not emergency:
            plus_target = float(np.clip(self.ablation.get(
                'plus_pool_target_ratio', 0.45), 0.0, 1.0)) * e_max
            required_after = min(
                e_max, max(current, min(plus_target, physical_required_after)))
        elif emergency:
            emergency_target = float(np.clip(self.ablation.get(
                'emergency_post_charge_ratio', 0.65), 0.0, 1.0)) * e_max
            required_after = min(
                e_max, max(emergency_target, physical_required_after))
        else:
            required_after = min(e_max, max(0.0, physical_required_after))

        allocation = max(0.0, required_after - current)
        service_latency = self.bs.estimate_service_latency_s(
            node.node_id, current_time=self.current_time, charger=charger)
        max_queue = max(1, int(self.ablation.get(
            'max_adaptive_charge_queue', 8)))
        queue_pressure = min(1.0, len(getattr(
            self.bs, 'service_charging_node_ids', set())) / float(max_queue))
        state['predicted_service_horizon_s'] = float(horizon)
        state['minimum_survival_allocation_j'] = float(allocation)
        state['charging_queue_pressure'] = float(queue_pressure)
        return {
            'eps': float(eps),
            'horizon': float(horizon),
            'required_after': float(required_after),
            'allocation': float(allocation),
            'service_latency': float(service_latency),
            'pool_class': pool,
            'queue_pressure': float(queue_pressure),
        }

    def _mission_reward(self, mission, valid_nodes, before, after,
                        delivered_by_node, travel_distance):
        requirements = mission.get('survival_requirements', {})
        useful_delivered = 0.0
        required_total = 0.0
        slack_gains = []
        rescued = 0
        deaths = 0

        for node in valid_nodes:
            nid = node.node_id
            req = requirements.get(nid, {})
            required_alloc = max(0.0, float(req.get('allocation', 0.0)))
            delivered = max(0.0, float(delivered_by_node.get(nid, 0.0)))
            required_total += required_alloc
            useful_delivered += min(delivered,
                                    required_alloc) if required_alloc > 0 else delivered

            eps = max(float(req.get('eps', 0.0)), 1e-12)
            horizon = max(float(req.get('horizon', 1.0)), 1.0)
            latency = max(float(req.get('service_latency', 0.0)), 0.0)
            before_slack = float(before.get(nid, 0.0)) / eps - latency
            after_slack = float(after.get(nid, 0.0)) / eps - latency
            slack_gains.append(
                min(1.0, max(0.0, after_slack - before_slack) / horizon))

            crossed_critical = (
                float(before.get(nid, 0.0)) <= float(node.E_critical)
                and float(after.get(nid, 0.0)) > float(node.E_critical)
            )
            restored_deadline = before_slack <= 0.0 < after_slack
            if crossed_critical or restored_deadline:
                rescued += 1
            if float(after.get(nid, 0.0)) <= 1e-15:
                deaths += 1

        total_capacity = max(float(mission.get('total_capacity', 1.0)), 1e-12)
        energy_component = (
            min(1.0, useful_delivered / required_total)
            if required_total > 1e-15
            else min(1.0, sum(delivered_by_node.values()) / total_capacity)
        )
        rescued_fraction = rescued / max(1.0, float(len(valid_nodes)))
        slack_component = float(np.mean(slack_gains)) if slack_gains else 0.0
        deadline_component = 0.5 * rescued_fraction + 0.5 * slack_component
        travel_component = max(0.0, float(travel_distance)) / max(
            float(mission.get('network_diag', 1.0)), 1e-12)
        revisit_component = sum(
            1 for node in valid_nodes
            if math.isfinite(mission.get('previous_charge_times', {}).get(
                node.node_id, -math.inf))
            and float(mission.get('start_time', self.current_time))
            - mission['previous_charge_times'][node.node_id]
            < self.bs.charging_reentry_cooldown_s
        ) / max(1.0, float(len(valid_nodes)))
        death_component = deaths / max(1.0, float(len(valid_nodes)))

        reward = (
            float(self.ablation.get('dqn_reward_energy_weight', 1.5)) *
            energy_component
            + float(self.ablation.get('dqn_reward_deadline_weight', 4.0)
                    ) * deadline_component
            - float(self.ablation.get('dqn_reward_travel_weight', 0.75)
                    ) * travel_component
            - float(self.ablation.get('dqn_reward_revisit_penalty', 2.0)
                    ) * revisit_component
            - float(self.ablation.get('dqn_reward_death_penalty', 6.0)
                    ) * death_component
        )
        return {
            'reward': float(reward),
            'energy': float(energy_component),
            'deadline': float(deadline_component),
            'travel': float(travel_component),
            'revisit': float(revisit_component),
            'death': float(death_component),
        }

    def _build_dynamic_energy_allocations(self, charger, target_node, emergency=False):
        nodes = self._target_nodes(target_node)
        if not self.ablation.get('use_dynamic_partial_charging', False):
            return {node.node_id: max(
                0.0, float(node.E_max) - float(node.E_current))
                for node in nodes}

        service_ids = set(getattr(
            self.bs, 'service_charging_node_ids', set()))
        waiting_count = max(1, len(service_ids))
        queue_capacity = max(1, int(self.ablation.get(
            'max_adaptive_charge_queue', 8)))
        now_norm = min(1.0, waiting_count / float(queue_capacity))
        wt_horizon = max(
            self.bs.charging_reentry_cooldown_s,
            float(self.ablation.get('balanced_service_horizon_cap_s', 420.0)))

        eps_values = [
            self.bs._effective_depletion_rate(nid, self.current_time)
            for nid in (service_ids or {node.node_id for node in nodes})
            if nid in self.bs.node_states
        ]
        eps_max = max(eps_values, default=0.0)

        allocations = {}
        floors = {}
        min_ratio = float(self.ablation.get(
            'minimum_charge_energy_ratio', 0.08))
        normal_target_ratio = float(self.ablation.get(
            'post_charge_release_ratio', 0.88))
        for node in nodes:
            state = self.bs.node_states.get(node.node_id, {})
            pool = str(state.get('charging_pool_class') or 'service').lower()
            node_emergency = bool(
                emergency
                or node.node_id in self.bs.pending_sos_requests
                or self.bs.is_deadline_rescue_node(
                    node.node_id, self.current_time))
            e_max = max(float(node.E_max), 1e-12)
            current = max(0.0, float(node.E_current))
            missing = max(0.0, e_max - current)
            wait_time = self.metrics.get_charging_request_wait_time(
                node.node_id, self.current_time)
            wt_norm = min(1.0, wait_time / max(wt_horizon, 1e-12))
            eps = self.bs._effective_depletion_rate(
                node.node_id, self.current_time)
            eps_norm = eps / eps_max if eps_max > 1e-15 else 0.0
            fraction = self.bs.dynamic_charging_oracle.allocation_fraction(
                wt_norm, eps_norm, now_norm)
            fraction = max(min_ratio, min(1.0, fraction))

            survival = self._survival_requirement(
                node, charger=charger, emergency=node_emergency)
            survival_gap = min(missing, float(survival['allocation']))
            floors[node.node_id] = survival_gap

            if pool == 'plus' and not node_emergency:
                fraction *= float(np.clip(self.ablation.get(
                    'plus_pool_fraction_scale', 0.55), 0.0, 1.0))
                plus_target = float(np.clip(self.ablation.get(
                    'plus_pool_target_ratio', 0.45), 0.0, 1.0))
                preferred_gap = max(0.0, plus_target * e_max - current)
                requested = max(survival_gap, missing * fraction)
                allocations[node.node_id] = min(
                    missing, preferred_gap, requested)
            else:
                target_ratio = (
                    float(self.ablation.get(
                        'emergency_post_charge_ratio', 0.65))
                    if node_emergency else normal_target_ratio)
                preferred_gap = max(0.0, target_ratio * e_max - current)
                requested = max(
                    missing * fraction,
                    survival_gap,
                    min(min_ratio * e_max, missing))
                allowed_gap = missing if survival_gap > preferred_gap else preferred_gap
                allocations[node.node_id] = min(
                    missing, allowed_gap, requested)

            self.metrics.adaptive_minimum_charge_ratios.append(
                min(1.0, survival_gap / e_max))

        max_stay = max(0.0, float(self.ablation.get(
            'max_cluster_stay_time_s', 1.50)))
        if max_stay > 0.0 and len(nodes) > 0:
            plan = charger.estimate_charging_plan(target_node, allocations)
            stay = float(plan.get('stay_time', 0.0))
            if stay > max_stay + 1e-12:
                for node in nodes:
                    state = self.bs.node_states.get(node.node_id, {})
                    if str(state.get('charging_pool_class') or '').lower() == 'plus':
                        allocations[node.node_id] = min(
                            allocations[node.node_id], floors.get(node.node_id, 0.0))
                plan = charger.estimate_charging_plan(target_node, allocations)
                stay = float(plan.get('stay_time', 0.0))

            if stay > max_stay + 1e-12:
                floor_plan = charger.estimate_charging_plan(
                    target_node, floors)
                floor_stay = float(floor_plan.get('stay_time', 0.0))
                if floor_stay < max_stay:
                    discretionary_scale = max(
                        0.0, min(1.0,
                                 (max_stay - floor_stay) /
                                 max(stay - floor_stay, 1e-12)))
                    for nid in allocations:
                        floor = min(allocations[nid], floors.get(nid, 0.0))
                        allocations[nid] = floor + discretionary_scale * max(
                            0.0, allocations[nid] - floor)

        return allocations

    def _scale_allocations_to_energy_budget(self, charger, target_node, allocations):
        tx, ty = self._target_coords(target_node)
        outbound = math.hypot(float(charger.x) - tx, float(charger.y) - ty)
        return_leg = math.hypot(tx - float(charger.depot_x),
                                ty - float(charger.depot_y))
        travel_cost = (outbound + return_leg) * float(
            charger.travel_energy_cost_per_m)
        reserve = float(self.ablation.get('mc_reserve_ratio', 0.05)) * float(
            charger.E_capacity)
        charging_budget = max(0.0, float(
            charger.E_current) - travel_cost - reserve)
        plan = charger.estimate_charging_plan(target_node, allocations)
        required = float(plan['mc_energy_cost'])
        if required <= charging_budget + 1e-12:
            return allocations
        if required <= 1e-15 or charging_budget <= 0.0:
            return {node_id: 0.0 for node_id in allocations}
        scale = charging_budget / required
        return {node_id: max(0.0, energy * scale)
                for node_id, energy in allocations.items()}

    def _active_mission_node_ids(self):
        node_ids = set()
        for mission in self.active_charger_missions.values():
            node_ids.update(mission.get('target_ids', ()))
        return node_ids

    def _next_charger_event_time(self):
        times = [float(mission.get('next_event_time', float('inf')))
                 for mission in self.active_charger_missions.values()]
        times.extend(float(t) for t in self.reprovision_events.values())
        return min(times, default=float('inf'))

    def _dispatch_charger_mission(self, charger, target_node,
                                  selected_emergency=False):
        if target_node is None or charger.status != 'IDLE':
            return False
        if charger.mc_id in self.active_charger_missions:
            return False

        active_ids = self._active_mission_node_ids()
        valid_nodes = []
        for node in self._target_nodes(target_node):
            if node.node_id in active_ids:
                continue
            emergency = bool(selected_emergency or self.bs.is_deadline_rescue_node(
                node.node_id, self.current_time))
            if node.status in ('ALIVE', 'DEAD') and self.bs.validate_charging_request(
                    node.node_id, emergency=emergency):
                valid_nodes.append(node)
        if not valid_nodes:
            return False

        if getattr(target_node, 'is_cluster', False):
            target_node.nodes = valid_nodes
            target_node.recompute_geometry()
        else:
            target_node = valid_nodes[0]

        allocations = self._build_dynamic_energy_allocations(
            charger, target_node, emergency=selected_emergency)
        allocations = self._scale_allocations_to_energy_budget(
            charger, target_node, allocations)
        if sum(allocations.values()) <= 1e-15:
            self.metrics.mc_infeasible_mission_rejections += 1
            return False

        for node in valid_nodes:
            applied_ratio = max(0.0, float(allocations.get(node.node_id, 0.0))) / max(
                float(node.E_max), 1e-12)
            self.metrics.mc_applied_minimum_charge_ratios.append(
                min(1.0, applied_ratio))

        plan = charger.estimate_charging_plan(target_node, allocations)
        if not math.isfinite(float(plan.get('stay_time', math.inf))):
            self.metrics.mc_infeasible_mission_rejections += 1
            return False

        tx, ty = self._target_coords(target_node)
        outbound_distance = math.hypot(float(charger.x) - tx,
                                       float(charger.y) - ty)
        outbound_energy = outbound_distance * \
            float(charger.travel_energy_cost_per_m)
        return_distance = math.hypot(tx - float(charger.depot_x),
                                     ty - float(charger.depot_y))
        return_energy = return_distance * \
            float(charger.travel_energy_cost_per_m)
        reserve = (return_energy + float(plan['mc_energy_cost']) +
                   float(self.ablation.get('mc_reserve_ratio', 0.05)) *
                   float(charger.E_capacity))
        if float(charger.E_current) + 1e-12 < outbound_energy + reserve:
            self.metrics.mc_infeasible_mission_rejections += 1
            return False

        network_diag = math.hypot(*self.network_size)
        self._mission_sequence += 1
        mission = {
            'mission_id': int(self._mission_sequence),
            'charger': charger,
            'target': target_node,
            'target_ids': tuple(node.node_id for node in valid_nodes),
            'phase': 'OUTBOUND',
            'start_time': float(self.current_time),
            'next_event_time': float(self.current_time) + outbound_distance / max(float(charger.speed), 1e-9),
            'allocations': dict(allocations),
            'selected_emergency': bool(selected_emergency),
            'reserve_at_arrival': float(reserve),
            'before_x': float(charger.x),
            'before_y': float(charger.y),
            'before_energies': {node.node_id: float(node.E_current) for node in valid_nodes},
            'previous_charge_times': {
                node.node_id: float(
                    self.bs.last_charge_time_by_node[node.node_id])
                for node in valid_nodes},
            'total_capacity': max(sum(float(node.E_max) for node in valid_nodes), 1e-12),
            'survival_requirements': {
                node.node_id: self._survival_requirement(
                    node, charger=charger, emergency=selected_emergency)
                for node in valid_nodes
            },
            'pool_classes': {
                node.node_id: str(self.bs.node_states.get(
                    node.node_id, {}).get('charging_pool_class') or 'service')
                for node in valid_nodes
            },
            'feature_before': self.charger_dqn.feature_for(
                charger.x, charger.y, target_node, self.bs.node_states, network_diag),
            'network_diag': float(network_diag),
            'session': None,
        }
        for node in valid_nodes:
            self.bs.last_selected_charge_time_by_node[node.node_id] = float(
                self.current_time)
            self.bs.last_service_start_by_node[node.node_id] = float(
                self.current_time)

        charger.status = 'MOVING'
        self.active_charger_missions[charger.mc_id] = mission
        self.metrics.mc_async_missions_started += 1
        return True

    def _advance_continuous_energy(self, new_time):
        new_time = float(new_time)
        dt = max(0.0, new_time - float(self.current_time))
        if dt <= 0.0:
            return
        self.metrics.current_sim_time = new_time

        for mission in list(self.active_charger_missions.values()):
            if mission.get('phase') == 'CHARGING' and mission.get('session') is not None:
                mission['charger'].advance_charging_session(
                    mission['session'], dt)

        if self.energy_experiment_active:
            self.bs.apply_fixed_charging(new_time)
            idle_power = float(self.ablation.get(
                'idle_sensing_power_j_per_s', 0.0))
            if idle_power > 0.0:
                for node_id, node in self.nodes.items():
                    if node_id == self.BS_ID or node.status != 'ALIVE':
                        continue
                    idle_energy = idle_power * dt
                    node.E_current = max(0.0, float(
                        node.E_current) - idle_energy)
                    if node_id in self.bs.node_states:
                        self.bs.node_states[node_id]['idle_sensing_energy_epoch'] += idle_energy
                    if node.E_current <= 0.0 and node.status == 'ALIVE':
                        node.status = 'DEAD'
                        self.metrics.record_node_death(node_id, new_time)

    def _record_async_dqn_transition(self, mission):
        if not self.use_dqn_charger_routing:
            return
        charger = mission['charger']
        valid_nodes = [self.nodes[nid] for nid in mission['target_ids']
                       if nid in self.nodes]
        before = mission['before_energies']
        after = {node.node_id: float(node.E_current) for node in valid_nodes}
        details = getattr(charger, 'last_charge_details', {}) or {}
        per_node = details.get('per_node', {}) or {}
        delivered_by_node = {
            node.node_id: max(0.0, float(
                per_node.get(node.node_id, {}).get('injected_energy', 0.0)))
            for node in valid_nodes
        }
        self.metrics.charging_opportunistic_plus_charges += sum(
            1 for nid, delivered in delivered_by_node.items()
            if delivered > 0.0 and str(mission.get(
                'pool_classes', {}).get(nid, '')).lower() == 'plus')
        travel_distance = math.hypot(
            float(charger.x) - float(mission['before_x']),
            float(charger.y) - float(mission['before_y']))
        components = self._mission_reward(
            mission, valid_nodes, before, after,
            delivered_by_node, travel_distance)
        reward = components['reward']
        self.metrics.dqn_rewards.append(reward)
        self.metrics.dqn_reward_energy_components.append(components['energy'])
        self.metrics.dqn_reward_deadline_components.append(
            components['deadline'])
        self.metrics.dqn_reward_travel_components.append(components['travel'])
        self.metrics.dqn_reward_revisit_components.append(
            components['revisit'])
        self.metrics.dqn_reward_death_components.append(components['death'])

        active_ids = self._active_mission_node_ids()
        next_candidates = []
        for candidate in self.bs.secure_charging_queue:
            if getattr(candidate, 'is_cluster', False):
                if not any(node.node_id in active_ids for node in candidate.nodes):
                    next_candidates.append(candidate)
            elif candidate in self.nodes and candidate not in active_ids:
                eligible_status = ('ALIVE', 'DEAD') if self.ablation.get(
                    'enable_dead_node_revival', True) else ('ALIVE',)
                if self.nodes[candidate].status in eligible_status:
                    next_candidates.append(self.nodes[candidate])
        next_features = [self.charger_dqn.feature_for(
            charger.x, charger.y, candidate, self.bs.node_states,
            mission['network_diag']) for candidate in next_candidates]
        if self.dqn_training_enabled:
            loss = self.charger_dqn.train_on_transition(
                mission['feature_before'], reward, next_features)
            self.metrics.dqn_training_losses.append(float(loss))
            if self.enable_dqn_audit_logging:
                self.metrics.record_dqn_training_audit(self.charger_dqn)

    def _complete_async_mission(self, mission, completed=True):
        charger = mission['charger']
        duration = max(0.0, float(self.current_time) -
                       float(mission['start_time']))
        self.metrics.mc_service_durations.append(duration)
        if completed:
            self.metrics.mc_async_missions_completed += 1
        if charger.status not in ('STRANDED', 'BRICKED'):
            charger.status = 'IDLE'
        self.active_charger_missions.pop(charger.mc_id, None)

    def _process_due_charger_events(self):
        while True:
            due_reprovision = [
                mc_id for mc_id, ready_time in self.reprovision_events.items()
                if float(ready_time) <= float(self.current_time) + 1e-9]
            for mc_id in sorted(due_reprovision):
                charger = next((mc for mc in self.mobile_chargers
                                if mc.mc_id == mc_id), None)
                if charger is not None:
                    new_key = f"MC-SECURE-KEY-NEW-LAB-{self.key_rng.randint(1000, 9999)}"
                    charger.crypto_keys = new_key
                    self.bs.mc_shared_keys[str(charger.mc_id)] = new_key
                    if charger is self.mc:
                        self.bs.mc_shared_key = new_key
                    charger.zeroized = False
                    charger.requires_lab_reprovisioning = False
                    charger.status = 'IDLE'
                    charger.x, charger.y = charger.depot_x, charger.depot_y
                self.reprovision_events.pop(mc_id, None)

            due = [mission for mission in self.active_charger_missions.values()
                   if float(mission.get('next_event_time', math.inf))
                   <= float(self.current_time) + 1e-9]
            if not due:
                if not due_reprovision:
                    return
                continue
            due.sort(key=lambda item: (float(item['next_event_time']),
                                       int(item['charger'].mc_id)))
            for mission in due:
                charger = mission['charger']
                phase = mission['phase']
                if phase == 'OUTBOUND':
                    travel_time = charger.move_to_node(
                        mission['target'], reserve_energy=mission['reserve_at_arrival'])
                    if travel_time is None:
                        self.metrics.mc_infeasible_mission_rejections += 1
                        self._complete_async_mission(mission, completed=False)
                        continue
                    current_allocations = {}
                    for node in self._target_nodes(mission['target']):
                        if node.node_id not in mission['target_ids']:
                            continue
                        missing = max(0.0, float(node.E_max) -
                                      float(node.E_current))
                        current_allocations[node.node_id] = min(
                            missing, max(0.0, float(mission['allocations'].get(node.node_id, 0.0))))
                    session = charger.begin_charging_session(
                        mission['target'], current_allocations,
                        start_time=self.current_time)
                    if session is None or session['duration'] <= 1e-15:
                        self._complete_async_mission(mission, completed=False)
                        continue
                    mission['session'] = session
                    mission['phase'] = 'CHARGING'
                    mission['next_event_time'] = float(
                        self.current_time) + float(session['duration'])

                elif phase == 'CHARGING':
                    charger.finish_charging_session(
                        mission['session'], completion_time=self.current_time)
                    self._record_async_dqn_transition(mission)
                    threshold = float(self.ablation.get(
                        'mc_return_threshold_ratio', 0.10)) * float(charger.E_capacity)
                    distance = math.hypot(
                        float(charger.x) - float(charger.depot_x),
                        float(charger.y) - float(charger.depot_y))
                    if charger.E_current < threshold and distance > 1e-9:
                        required = charger.movement_energy_for_distance(
                            distance)
                        if charger.E_current + 1e-12 < required:
                            charger.status = 'STRANDED'
                            self.metrics.mc_stranded_events += 1
                            self._complete_async_mission(
                                mission, completed=False)
                            continue
                        charger.status = 'RETURNING'
                        mission['phase'] = 'RETURNING'
                        mission['next_event_time'] = float(
                            self.current_time) + distance / max(float(charger.speed), 1e-9)
                    else:
                        self._complete_async_mission(mission, completed=True)

                elif phase == 'RETURNING':
                    if charger.return_to_depot() is None:
                        self.metrics.mc_stranded_events += 1
                        self._complete_async_mission(mission, completed=False)
                    else:
                        self._complete_async_mission(mission, completed=True)

    def _execute_charger_mission(self, charger, target_node, selected_emergency=False, mission_start_time=None):
        if target_node is None:
            return 0.0
        mission_start_time = self.current_time if mission_start_time is None else float(
            mission_start_time)
        self.metrics.current_sim_time = mission_start_time

        valid_nodes = []
        for node in self._target_nodes(target_node):
            emergency = bool(selected_emergency or self.bs.is_deadline_rescue_node(
                node.node_id, mission_start_time))
            eligible_status = ('ALIVE', 'DEAD') if self.ablation.get(
                'enable_dead_node_revival', True) else ('ALIVE',)
            if node.status in eligible_status and self.bs.validate_charging_request(
                    node.node_id, emergency=emergency):
                valid_nodes.append(node)
        if not valid_nodes:
            return 0.0
        if getattr(target_node, 'is_cluster', False):
            target_node.nodes = valid_nodes
            target_node.recompute_geometry()

        for node in valid_nodes:
            self.bs.last_selected_charge_time_by_node[node.node_id] = mission_start_time
            self.bs.last_service_start_by_node[node.node_id] = mission_start_time

        network_diag = math.hypot(*self.network_size)
        feature_before = self.charger_dqn.feature_for(
            charger.x, charger.y, target_node, self.bs.node_states, network_diag)
        before_x, before_y = float(charger.x), float(charger.y)
        before_energies = {node.node_id: float(
            node.E_current) for node in valid_nodes}
        previous_charge_times = {
            node.node_id: float(self.bs.last_charge_time_by_node[node.node_id])
            for node in valid_nodes}
        total_capacity = max(sum(float(node.E_max)
                             for node in valid_nodes), 1e-12)

        allocations = self._build_dynamic_energy_allocations(
            charger, target_node, emergency=selected_emergency)
        allocations = self._scale_allocations_to_energy_budget(
            charger, target_node, allocations)
        if sum(allocations.values()) <= 1e-15:
            self.metrics.mc_infeasible_mission_rejections += 1
            return 0.0

        for node in valid_nodes:
            self.metrics.mc_applied_minimum_charge_ratios.append(
                min(1.0, max(0.0, float(allocations.get(node.node_id, 0.0))) /
                    max(float(node.E_max), 1e-12)))

        reward_mission = {
            'start_time': float(mission_start_time),
            'previous_charge_times': previous_charge_times,
            'total_capacity': float(total_capacity),
            'network_diag': float(network_diag),
            'survival_requirements': {
                node.node_id: self._survival_requirement(
                    node, charger=charger, emergency=selected_emergency)
                for node in valid_nodes
            },
            'pool_classes': {
                node.node_id: str(self.bs.node_states.get(
                    node.node_id, {}).get('charging_pool_class') or 'service')
                for node in valid_nodes
            },
        }

        depot_refill_time = self._return_to_depot_if_relative_mc_infeasible(
            charger, target_node)
        if depot_refill_time > 0.0:
            mission_start_time += depot_refill_time
            self.metrics.current_sim_time = mission_start_time
            before_x, before_y = float(charger.x), float(charger.y)
            allocations = self._scale_allocations_to_energy_budget(
                charger, target_node, allocations)

        plan = charger.estimate_charging_plan(target_node, allocations)
        tx, ty = self._target_coords(target_node)
        return_energy = math.hypot(
            tx - charger.depot_x, ty - charger.depot_y) * charger.travel_energy_cost_per_m
        reserve_energy = (return_energy + float(plan['mc_energy_cost']) +
                          float(self.ablation.get('mc_reserve_ratio', 0.05)) *
                          charger.E_capacity)
        travel_time = charger.move_to_node(
            target_node, reserve_energy=reserve_energy)
        if travel_time is None:
            self.metrics.mc_infeasible_mission_rejections += 1
            return max(0.0, depot_refill_time)

        arrival_time = mission_start_time + float(travel_time)
        self.metrics.current_sim_time = arrival_time
        charge_time = charger.charge_node(
            target_node, energy_allocations=allocations)
        finish_time = arrival_time + float(charge_time)

        after_energies = {node.node_id: float(
            node.E_current) for node in valid_nodes}
        injected = sum(max(0.0, after_energies[nid] - before_energies[nid])
                       for nid in before_energies)
        if injected / total_capacity < float(self.ablation.get(
                'minimum_charge_energy_ratio', 0.08)):
            self.metrics.mc_useless_charge_events += 1

        if self.use_dqn_charger_routing:
            travel_distance = math.hypot(float(charger.x) - before_x,
                                         float(charger.y) - before_y)
            per_node = (getattr(charger, 'last_charge_details', {}) or {}).get(
                'per_node', {}) or {}
            delivered_by_node = {
                node.node_id: max(0.0, float(
                    per_node.get(node.node_id, {}).get('injected_energy', 0.0)))
                for node in valid_nodes
            }
            self.metrics.charging_opportunistic_plus_charges += sum(
                1 for nid, delivered in delivered_by_node.items()
                if delivered > 0.0 and str(reward_mission.get(
                    'pool_classes', {}).get(nid, '')).lower() == 'plus')
            components = self._mission_reward(
                reward_mission, valid_nodes, before_energies, after_energies,
                delivered_by_node, travel_distance)
            reward = components['reward']
            self.metrics.dqn_rewards.append(reward)
            self.metrics.dqn_reward_energy_components.append(
                components['energy'])
            self.metrics.dqn_reward_deadline_components.append(
                components['deadline'])
            self.metrics.dqn_reward_travel_components.append(
                components['travel'])
            self.metrics.dqn_reward_revisit_components.append(
                components['revisit'])
            self.metrics.dqn_reward_death_components.append(
                components['death'])

            next_candidates = []
            for candidate in self.bs.secure_charging_queue:
                if getattr(candidate, 'is_cluster', False):
                    next_candidates.append(candidate)
                elif candidate in self.nodes:
                    eligible_status = ('ALIVE', 'DEAD') if self.ablation.get(
                        'enable_dead_node_revival', True) else ('ALIVE',)
                    if self.nodes[candidate].status in eligible_status:
                        next_candidates.append(self.nodes[candidate])
            next_features = [self.charger_dqn.feature_for(
                charger.x, charger.y, candidate, self.bs.node_states, network_diag)
                for candidate in next_candidates]
            if self.dqn_training_enabled:
                loss = self.charger_dqn.train_on_transition(
                    feature_before, reward, next_features)
                self.metrics.dqn_training_losses.append(float(loss))
                if self.enable_dqn_audit_logging:
                    self.metrics.record_dqn_training_audit(self.charger_dqn)

        if charger.E_current < float(self.ablation.get(
                'mc_return_threshold_ratio', 0.10)) * charger.E_capacity:
            return_time = charger.return_to_depot()
            if return_time is None:
                self.metrics.mc_stranded_events += 1
            else:
                finish_time += float(return_time)

        return max(0.0, finish_time - mission_start_time)

    def _service_multi_mobile_chargers(self):
        idle_chargers = [
            mc for mc in self.mobile_chargers
            if getattr(mc, 'status', 'IDLE') == 'IDLE'
            and mc.mc_id not in self.active_charger_missions
            and not getattr(mc, 'zeroized', False)
            and not getattr(mc, 'requires_lab_reprovisioning', False)]
        if not self.bs.energy_experiment_active or not idle_chargers:
            return

        adaptive_mode = bool(
            self.ablation.get('use_adaptive_quantum_charging', False)
            or self.ablation.get('use_adaptive_charge_guards', False)
            or self.ablation.get('use_dqn_charger_routing', False))

        if adaptive_mode and not self.bs.secure_charging_queue:
            self.bs.build_adaptive_charging_queue(
                idle_charger_count=len(idle_chargers))

        in_service = self._active_mission_node_ids()
        candidates = []
        for candidate in list(self.bs.secure_charging_queue):
            if getattr(candidate, 'is_cluster', False):
                candidate.nodes = [
                    node for node in candidate.nodes
                    if node.node_id not in in_service
                    and node.status in ('ALIVE', 'DEAD')
                    and self.bs.validate_charging_request(
                        node.node_id,
                        emergency=(node.node_id in self.bs.pending_sos_requests))]
                if candidate.nodes:
                    candidate.recompute_geometry()
                    candidates.append(candidate)
            elif candidate in self.nodes and candidate not in in_service:
                node = self.nodes[candidate]
                if node.status in ('ALIVE', 'DEAD') and self.bs.validate_charging_request(
                        candidate, emergency=(candidate in self.bs.pending_sos_requests)):
                    candidates.append(candidate)

        if not candidates and not adaptive_mode:
            candidates = [
                nid for nid in self.bs.pending_sos_requests
                if nid in self.nodes and nid not in in_service
                and self.bs.validate_charging_request(nid, emergency=True)]
        if not candidates:
            fallback = self._select_predictive_charging_target()
            if fallback is not None and fallback.node_id not in in_service:
                candidates = [fallback.node_id]
        if not candidates:
            return

        assignments = []
        remaining_candidates = list(candidates)
        assigned_ids = set()
        reserved_targets = []
        stable_regions = bool(self.ablation.get(
            'use_stable_charger_regions', False))
        interference_guard = bool(self.ablation.get(
            'use_interference_safe_multi_mc_dispatch', False))
        for charger in sorted(idle_chargers, key=lambda mc: int(mc.mc_id)):
            allowed = []
            emergency_conflicts = []
            for candidate in remaining_candidates:
                conflict = self._candidate_has_active_interference_conflict(
                    candidate, reserved_targets)
                if not conflict:
                    allowed.append(candidate)
                elif self._target_is_emergency(candidate):
                    emergency_conflicts.append(candidate)
                else:
                    self.metrics.mc_interference_conflict_deferrals += 1

            candidate_pool = allowed if allowed else emergency_conflicts
            if not candidate_pool:
                continue
            if stable_regions:
                regional = [candidate for candidate in candidate_pool
                            if self._stable_region_owner_id(candidate)
                            == int(charger.mc_id)]
                if regional:
                    candidate_pool = regional

            target = self._select_target_for_charger(
                charger, candidate_pool, assigned_ids)
            if target is None:
                continue
            target_ids = {node.node_id for node in self._target_nodes(target)}
            if target_ids & in_service:
                continue
            if (interference_guard and
                    self._candidate_has_active_interference_conflict(
                        target, reserved_targets)):
                # Only DEAD/SOS/deadline-rescue targets may override the guard.
                if not self._target_is_emergency(target):
                    self.metrics.mc_interference_conflict_deferrals += 1
                    continue
                self.metrics.mc_interference_emergency_overrides += 1
            if stable_regions:
                owner_id = self._stable_region_owner_id(target)
                if owner_id is not None and owner_id != int(charger.mc_id):
                    self.metrics.mc_cross_region_assignments += 1

            assignments.append((charger, target))
            reserved_targets.append(target)
            assigned_ids.add(getattr(target, 'node_id', None))
            in_service.update(target_ids)
            remaining_candidates = [
                candidate for candidate in remaining_candidates
                if candidate is not target
                and candidate != getattr(target, 'node_id', None)]
            if not remaining_candidates:
                break

        for charger, target in assignments:
            target_ids = [node.node_id for node in self._target_nodes(target)]
            selected_emergency = any(
                nid in self.bs.pending_sos_requests
                or self.bs.is_deadline_rescue_node(nid, self.current_time)
                for nid in target_ids)
            if not self._dispatch_charger_mission(
                    charger, target, selected_emergency):
                continue
            for nid in target_ids:
                self.bs.pending_sos_requests.discard(nid)
            if target in self.bs.secure_charging_queue:
                self.bs.secure_charging_queue.remove(target)
            elif (not getattr(target, 'is_cluster', False)
                  and target.node_id in self.bs.secure_charging_queue):
                self.bs.secure_charging_queue.remove(target.node_id)

        self.bs.mc_reference = self.mc
        self.bs.mc_references = self.mobile_chargers

    def _select_predictive_charging_target(self):
        best_node = None
        best_slack = float('inf')
        for nid, state in self.bs.node_states.items():
            if nid == self.BS_ID or not self.bs.validate_charging_request(nid):
                continue
            node = self.nodes.get(nid)
            eligible_status = ('ALIVE', 'DEAD') if self.ablation.get(
                'enable_dead_node_revival', True) else ('ALIVE',)
            if node is None or node.status not in eligible_status:
                continue
            E_initial = state['E_initial']
            E_est = state.get('E_est', E_initial)
            consumed_energy = max(0.0, E_initial - E_est)
            elapsed_time = max(1.0, self.current_time)
            consumption_rate = consumed_energy / elapsed_time
            if consumption_rate <= 0:
                continue
            time_to_death = E_est / consumption_rate
            travel_distance = math.hypot(
                self.mc.x - node.x, self.mc.y - node.y)
            travel_time = travel_distance / max(self.mc.speed, 1e-6)
            slack = time_to_death - travel_time
            if slack < best_slack and time_to_death <= 7200.0:
                best_slack = slack
                best_node = node
        return best_node

    def lab_reprovision_mobile_charger(self):
        lab_time_delay = float(self.ablation.get(
            'mc_lab_reprovisioning_delay_s', 7200.0))
        self.mc.status = 'REPROVISIONING'
        self.reprovision_events[self.mc.mc_id] = (
            float(self.current_time) + max(0.0, lab_time_delay))
        return self.reprovision_events[self.mc.mc_id]

    def trigger_physical_compromise_event(self, simulate_lab_reprovisioning=False):
        active = self.active_charger_missions.pop(self.mc.mc_id, None)
        if active is not None:
            self.metrics.mc_async_missions_incomplete += 1
        self.mc.tpms_zeroization()
        self.bs.mc_shared_key = None
        self.bs.mc_shared_keys.pop(str(self.mc.mc_id), None)
        if simulate_lab_reprovisioning:
            self.lab_reprovision_mobile_charger()

    def inject_attacks(self, attack_percentage):
        num_attackers = int(self.num_nodes * attack_percentage)
        alive_node_ids = [
            nid for nid in self.nodes.keys() if nid != self.BS_ID]
        self.malicious_nodes = self.attack_rng.sample(
            alive_node_ids, num_attackers)
        for nid in self.malicious_nodes:
            self.nodes[nid].set_malicious(True)
        print(
            f"[ATTACK INJECTION] {num_attackers} malicious nodes injected into the network.")

    def _maybe_inject_fake_charger_attack(self):
        """Physically raise one sensor's energy without an authenticated MC report.

        The rise is later reported by the normal SensorNode DATA/control path,
        allowing the BS zero-overhead energy audit to detect it end-to-end.
        This is disabled in the non-regression baseline.
        """
        if not self.ablation.get('fake_charger_attack_enabled', False):
            return None
        probability = max(0.0, min(1.0, float(self.ablation.get(
            'fake_charger_attack_probability_per_epoch', 0.0))))
        if self.attack_rng.random() >= probability:
            return None
        candidates = [
            node for nid, node in self.nodes.items()
            if nid != self.BS_ID and node.status == 'ALIVE'
            and nid not in self.bs.fixed_charger_nodes
        ]
        if not candidates:
            return None
        node = self.attack_rng.choice(candidates)
        ratio = max(0.0, float(self.ablation.get(
            'fake_charger_attack_energy_ratio', 0.35)))
        injected = node.receive_wireless_charge(float(node.E_max) * ratio)
        return (node.node_id, injected)

    def _account_implicit_mac_ack(self, sender_id, receiver_id, distance):
        """Account a hardware MAC ACK without creating a network packet.

        'accounting' is the non-regression default: ACK energy is measured and
        used by BS trust accounting, but is not double-deducted from the legacy
        aggregate radio envelope. 'strict_physical' additionally deducts the
        sender RX and receiver TX costs from physical batteries.
        """
        sender = self.nodes.get(sender_id)
        receiver = self.nodes.get(receiver_id)
        if sender is None or receiver is None:
            return
        bits = self.ack_packet_bits
        sender_rx = 0.0 if sender_id == self.BS_ID else bits * float(sender.E_elec)
        if receiver_id == self.BS_ID:
            receiver_tx = 0.0
        elif distance <= float(receiver.d0):
            receiver_tx = bits * (float(receiver.E_elec) + float(receiver.eps_fs) * distance ** 2)
        else:
            receiver_tx = bits * (float(receiver.E_elec) + float(receiver.eps_mp) * distance ** 4)
        strict = self.implicit_mac_ack_mode in {'strict', 'strict_physical', 'physical'}
        self.metrics.record_implicit_mac_ack(sender_rx + receiver_tx, physically_deducted=strict)
        if strict:
            if sender_id != self.BS_ID:
                sender._consume_energy(rx_bits=bits)
            if receiver_id != self.BS_ID:
                receiver._consume_energy(tx_bits=bits, distance=distance)

    def run_simulation(self):
        def _trace_control_packet(source_id, packet):
            table = self.bs.routing_tables.get(source_id, {})
            routes = []

            primary_path = table.get('primary_path') or []
            if primary_path and primary_path[0] == source_id and primary_path[-1] == self.BS_ID:
                routes.append(primary_path)
            elif self.nodes[source_id].primary_next_hop is not None:
                routes.append(
                    [source_id, self.nodes[source_id].primary_next_hop, self.BS_ID])

            secondary_path = table.get('secondary_path') or []
            if (packet.get('priority') == 'HIGH' and
                    self.ablation.get('use_critical_multipath_routing', False) and
                    secondary_path and secondary_path[0] == source_id and
                    secondary_path[-1] == self.BS_ID):
                routes.append(secondary_path)

            for route in routes:
                pkt = dict(packet)
                pkt['energy_trace'] = list(packet.get('energy_trace', []))
                pkt['path_header'] = [source_id]
                pkt['hash'] = compute_packet_hash(
                    pkt, self.nodes[source_id].crypto_keys)

                packet_dropped = False
                path_log = [source_id]
                malicious_delay_s = 0.0

                for idx in range(1, len(route)):
                    current_hop_id = route[idx]
                    if current_hop_id == self.BS_ID:
                        sender_id = route[idx - 1]
                        bs_distance = math.hypot(
                            self.nodes[sender_id].x - self.nodes[self.BS_ID].x,
                            self.nodes[sender_id].y - self.nodes[self.BS_ID].y)
                        self._account_implicit_mac_ack(
                            sender_id, self.BS_ID, bs_distance)
                        path_log.append(self.BS_ID)
                        break

                    hop_node = self.nodes[current_hop_id]
                    path_log.append(current_hop_id)
                    next_hop_id = route[idx + 1] if idx + \
                        1 < len(route) else self.BS_ID

                    if next_hop_id != self.BS_ID:
                        next_node = self.nodes[next_hop_id]
                        distance = math.hypot(
                            hop_node.x - next_node.x, hop_node.y - next_node.y)
                    else:
                        bs_node = self.nodes[self.BS_ID]
                        distance = math.hypot(
                            hop_node.x - bs_node.x, hop_node.y - bs_node.y)

                    sender_id = route[idx - 1]
                    self._account_implicit_mac_ack(
                        sender_id, current_hop_id,
                        math.hypot(self.nodes[sender_id].x - self.nodes[current_hop_id].x,
                                   self.nodes[sender_id].y - self.nodes[current_hop_id].y))
                    success = hop_node.forward_packet(
                        pkt, self.current_time, distance)
                    if not success:
                        packet_dropped = True
                        break
                    malicious_delay_s += hop_node.attack_delay_seconds(
                        self.current_time)

                if not packet_dropped and path_log[-1] == self.BS_ID:
                    pkt['path_header'] = path_log
                    delay = len(path_log) * 0.100 + malicious_delay_s
                    self.bs.data_plane_receive(
                        pkt, path_log, self.current_time + delay)

        if self.ablation.get('enable_periodic_hello', False):
            hello_period = float(self.ablation.get('hello_period', 60.0))
            hello_count = int(max(0.0, self.sim_time) //
                              max(hello_period, 1e-9)) * max(self.num_nodes, 0)
            for _ in range(hello_count):
                self.metrics.update_packet_transmission(
                    is_critical=False, is_control=True, size_bytes=self.control_packet_bytes, size_bits=self.control_packet_bits, overhead_phase='operational')

        self.bootstrap_network()
        self.bs._update_fixed_charger_assignments()
        self.bs._last_fixed_charge_time = 0.0
        self.inject_attacks(attack_percentage=float(
            self.ablation.get('attack_percentage', 0.15)))

        packet_seq_num = 0
        event_rate = max(float(self.ablation.get(
            'event_rate_lambda', 0.5)), 1e-12)
        workload_events = None
        workload_index = 0
        recorded_workload = []
        if self.workload_trace_replay_path:
            with open(self.workload_trace_replay_path, 'r', encoding='utf-8') as handle:
                workload_events = list(json.load(handle))
            next_event_time = (
                float(workload_events[0]['time']) if workload_events
                else float('inf'))
        else:
            next_event_time = self.event_rng.expovariate(event_rate)
        next_epoch_time = float(self.bs.T_epoch)
        next_energy_record_time = 10.0
        self.current_time = 0.0
        self.metrics.current_sim_time = 0.0

        tpm_attack_time = self.tpm_rng.uniform(
            self.sim_time * 0.3, self.sim_time * 0.7)
        tpm_attack_triggered = False
        remediation_applied = False
        rehabilitation_applied = False
        remediation_time_raw = self.ablation.get(
            'attack_remediation_time_s', None)
        remediation_time = (
            None if remediation_time_raw in (None, '')
            else max(0.0, float(remediation_time_raw))
        )
        rehabilitation_delay_raw = self.ablation.get(
            'attack_rehabilitation_delay_s', self.bs.T_epoch)
        rehabilitation_delay = (
            None if rehabilitation_delay_raw in (None, '')
            else max(0.0, float(rehabilitation_delay_raw))
        )
        rehabilitation_time = (
            None if remediation_time is None or rehabilitation_delay is None
            else remediation_time + rehabilitation_delay
        )

        while self.current_time < self.sim_time - 1e-12:
            next_mc_time = self._next_charger_event_time()
            next_tpm_time = (
                tpm_attack_time
                if self.physical_compromise_experiment_active and not tpm_attack_triggered
                else float('inf'))
            next_remediation_time = (
                float(remediation_time)
                if remediation_time is not None and not remediation_applied
                else float('inf'))
            next_rehabilitation_time = (
                float(rehabilitation_time)
                if rehabilitation_time is not None and not rehabilitation_applied
                else float('inf'))
            next_time = min(
                float(next_event_time), float(next_epoch_time),
                float(next_energy_record_time), float(next_mc_time),
                float(next_tpm_time), float(next_remediation_time),
                float(next_rehabilitation_time), float(self.sim_time))
            if not math.isfinite(next_time):
                break

            self._advance_continuous_energy(next_time)
            self.current_time = float(next_time)
            self.metrics.current_sim_time = float(next_time)

            if next_tpm_time <= self.current_time + 1e-9:
                self.trigger_physical_compromise_event()
                tpm_attack_triggered = True

            self._process_due_charger_events()

            if (not remediation_applied and remediation_time is not None
                    and self.current_time + 1e-9 >= remediation_time):
                for nid in self.malicious_nodes:
                    self.nodes[nid].remediate(self.current_time)
                remediation_applied = True

            if (remediation_applied and not rehabilitation_applied
                    and rehabilitation_time is not None
                    and self.current_time + 1e-9 >= rehabilitation_time):
                for nid in self.malicious_nodes:
                    self.nodes[nid].complete_rehabilitation(self.current_time)
                rehabilitation_applied = True

            processed_network_event = False
            if next_event_time <= self.current_time + 1e-9:
                processed_network_event = True
                if self.active_charger_missions:
                    self.metrics.mc_network_events_while_busy += 1

                if workload_events is not None:
                    event = workload_events[workload_index]
                    active_node_id = int(event['source_id'])
                    raw_data = float(event['raw_data'])
                    workload_index += 1
                    next_event_time = (
                        float(workload_events[workload_index]['time'])
                        if workload_index < len(workload_events)
                        else float('inf'))
                    active_node = self.nodes.get(active_node_id)
                    source_is_alive = (
                        active_node is not None and active_node.status == 'ALIVE')
                else:
                    alive_nodes = [
                        node_id for node_id, node in self.nodes.items()
                        if node.status == 'ALIVE' and node_id != self.BS_ID]
                    source_is_alive = bool(alive_nodes)
                    if source_is_alive:
                        active_node_id = self.event_rng.choice(alive_nodes)
                        active_node = self.nodes[active_node_id]
                        is_critical_event = self.event_rng.random() < float(
                            self.ablation.get('critical_event_probability', 0.10))
                        raw_data = 80.0 if is_critical_event else 25.0
                        recorded_workload.append({
                            'time': float(self.current_time),
                            'source_id': int(active_node_id),
                            'raw_data': float(raw_data),
                        })
                    next_event_time += self.event_rng.expovariate(event_rate)

                if source_is_alive:
                    packet_seq_num += 1
                    method_is_critical = raw_data > active_node.adaptive_threshold
                    source_packet = active_node.sense_environment(
                        raw_data, self.current_time)
                    self._trace_packet_path(
                        active_node_id, packet_seq_num,
                        method_is_critical, raw_data,
                        source_packet=source_packet)

            if next_epoch_time <= self.current_time + 1e-9:
                processed_network_event = True
                if self.active_charger_missions:
                    self.metrics.mc_network_events_while_busy += 1
                self._maybe_inject_fake_charger_attack()
                self.bs.control_plane_epoch()
                self._push_updated_routes()
                next_epoch_time += float(self.bs.T_epoch)

            if next_energy_record_time <= self.current_time + 1e-9:
                avg_energy = sum(
                    node.E_current for node in self.nodes.values()
                    if node.node_id != self.BS_ID) / max(1, self.num_nodes)
                self.metrics.record_energy_state(avg_energy)
                next_energy_record_time += 10.0

            for node_id, node in self.nodes.items():
                if node_id == self.BS_ID:
                    continue
                topology_packet = node.check_route_health(
                    self.nodes, self.transmission_range, self.current_time)
                if topology_packet is not None:
                    _trace_control_packet(node_id, topology_packet)
                control_packet = node.check_inactivity(self.current_time)
                if control_packet is not None:
                    _trace_control_packet(node_id, control_packet)

            if self.energy_experiment_active:
                self.service_mobile_charger()

        self.metrics.mc_async_missions_incomplete += len(
            self.active_charger_missions)
        self.metrics.current_sim_time = min(
            float(self.current_time), float(self.sim_time))

        detected = self.bs.get_blacklisted_nodes()
        actual = self.malicious_nodes
        detected = [d for d in detected if d != self.bs.BS_ID]
        actual = [a for a in actual if a != self.bs.BS_ID]
        benign_count = self.num_nodes - len(actual)
        tp = len(set(detected) & set(actual))
        fn = len(set(actual) - set(detected))
        fp = len(set(detected) - set(actual))
        tn = benign_count - fp
        self.metrics.update_ids_matrix(tp, tn, fp, fn)
        mc_trust_values = [float(self.bs.mc_trust[str(mc.mc_id)])
                           for mc in self.mobile_chargers]
        self.metrics.experiment_parameters.update({
            'MC_Trust_Average_End': float(np.mean(mc_trust_values)) if mc_trust_values else 1.0,
            'MC_Trust_Minimum_End': float(np.min(mc_trust_values)) if mc_trust_values else 1.0,
        })
        if self.enable_dqn_audit_logging:
            self.metrics.record_dqn_training_audit(self.charger_dqn)
        if self.workload_trace_record_path and workload_events is None:
            trace_path = os.path.abspath(os.fspath(
                self.workload_trace_record_path))
            os.makedirs(os.path.dirname(trace_path) or '.', exist_ok=True)
            with open(trace_path, 'w', encoding='utf-8') as handle:
                json.dump(recorded_workload, handle, indent=2)
        checkpoint_save_path = self.ablation.get('dqn_checkpoint_save_path')
        if checkpoint_save_path and self.use_dqn_charger_routing:
            self.charger_dqn.save_checkpoint(checkpoint_save_path)

    def _critical_failover_energy_feasible(self, route):
        """Return whether failover preserves per-node energy reserves."""
        if not self.ablation.get('critical_failover_energy_guard', True):
            return True
        route = list(route or [])
        if len(route) < 2:
            return False
        min_ratio = max(0.0, min(1.0, float(self.ablation.get(
            'critical_failover_min_energy_ratio', 0.15))))
        margin_ratio = max(0.0, float(self.ablation.get(
            'critical_failover_reserve_margin_ratio', 0.02)))
        ack_bits = float(self.ack_packet_bits)

        def tx_cost(node, bits, distance):
            if distance <= float(node.d0):
                return bits * (
                    float(node.E_elec) + float(node.eps_fs) * distance ** 2)
            return bits * (
                float(node.E_elec) + float(node.eps_mp) * distance ** 4)

        projected = defaultdict(float)
        for u, v in zip(route[:-1], route[1:]):
            sender = self.nodes.get(u)
            receiver = self.nodes.get(v)
            if sender is None or receiver is None:
                return False
            distance = math.hypot(
                float(sender.x) - float(receiver.x),
                float(sender.y) - float(receiver.y))
            if u != self.BS_ID:
                projected[u] += tx_cost(
                    sender, self.data_packet_bits, distance)
                projected[u] += ack_bits * float(sender.E_elec)
            if v != self.BS_ID:
                projected[v] += (
                    self.data_packet_bits * float(receiver.E_elec))
                projected[v] += tx_cost(receiver, ack_bits, distance)

        for nid, energy_cost in projected.items():
            node = self.nodes.get(nid)
            if (node is None or
                    str(getattr(node, 'status', 'DEAD')).upper() != 'ALIVE'):
                return False
            e_max = max(float(node.E_max), 1e-12)
            reserve = max(
                float(getattr(node, 'E_critical', 0.0)),
                (min_ratio + margin_ratio) * e_max)
            if float(node.E_current) - energy_cost <= reserve + 1e-15:
                return False
        return True

    def _trace_packet_path(self, source_id, seq_num, is_critical, payload_value,
                           source_packet=None, seq_override=None,
                           count_generated=True, allow_source_replay=True):
        if not self.ablation.get('use_implicit_piggyback_telemetry', True):
            self.metrics.update_packet_transmission(
                is_critical=False, is_control=True, size_bytes=self.control_packet_bytes, size_bits=self.control_packet_bits, overhead_phase='operational')
        if self.ablation.get('enable_network_layer_ack', False):
            self.metrics.update_packet_transmission(
                is_critical=False, is_control=True, size_bytes=self.control_packet_bytes, size_bits=self.control_packet_bits, overhead_phase='operational')

        if seq_override is None:
            self.source_seq_counters[source_id] += 1
            seq_num = self.source_seq_counters[source_id]
        else:
            seq_num = int(seq_override)
        key = (int(source_id), int(seq_num))

        if count_generated and key not in self.metrics._unique_generated_data_keys:
            self.metrics._unique_generated_data_keys.add(key)
            self.metrics.update_packet_transmission(
                is_critical=bool(is_critical), is_control=False, size_bytes=self.data_packet_bytes,
                size_bits=self.data_packet_bits, is_delivered=False, count_generated=True
            )

        before_received = self.metrics.received_data_packets
        before_received_critical = self.metrics.received_critical_packets
        before_bits = self.metrics.total_bits_received
        before_delay_len = len(self.metrics.end_to_end_delays)

        table = (self.bs.data_routing_tables.get(source_id, {})
                 if self.bs.data_routing_tables else
                 self.bs.routing_tables.get(source_id, {}))
        secondary_path_for_mode = list(table.get('secondary_path') or [])
        secondary_mode = str(table.get(
            'secondary_mode', 'redundant')).strip().lower()
        if secondary_mode not in {'redundant', 'failover'}:
            secondary_mode = 'redundant'
        if is_critical and self.ablation.get("use_critical_multipath_routing", False):
            candidate_paths = table.get('critical_paths') or []
            if not candidate_paths:
                candidate_paths = [p for p in [
                    table.get('primary_path'), table.get('secondary_path')] if p]
            self.metrics.record_critical_multipath_opportunity(
                secondary_available=(len(candidate_paths) >= 2))
            source_risk = self.bs.node_states.get(
                source_id, {}).get('risk_mean', 0.0)
            max_paths = 2
            if self.ablation.get('use_risk_adaptive_coded_multipath', False):
                if source_risk >= 0.35:
                    max_paths = 3
                if source_risk >= 0.65:
                    max_paths = 4
            paths_to_trace = [{'type': f'CRITICAL_PATH_{idx+1}', 'path': p}
                              for idx, p in enumerate(candidate_paths[:max_paths])]
            required_fragments = 1
        else:
            primary_path = table.get('primary_path') or []
            if primary_path:
                paths_to_trace = [{'type': 'PRIMARY', 'path': primary_path}]
            elif self.nodes[source_id].primary_next_hop is not None:
                paths_to_trace = [{'type': 'PRIMARY', 'path': [
                    source_id, self.nodes[source_id].primary_next_hop, self.BS_ID]}]
            else:
                paths_to_trace = []
            required_fragments = 1

        delivered_fragments = 0
        first_success_delay = None
        first_delivered_packet = None
        attempted_failover = 0
        successful_failover = 0

        for path_index, p_info in enumerate(paths_to_trace):
            route = list(p_info['path'])
            if not route or route[0] != source_id or route[-1] != self.BS_ID:
                continue
            is_failover_route = (
                is_critical and secondary_mode == 'failover'
                and route == secondary_path_for_mode
            )
            if is_failover_route and delivered_fragments > 0:
                continue
            if is_failover_route:
                if not self._critical_failover_energy_feasible(route):
                    self.metrics.critical_failover_energy_guard_deferrals += 1
                    continue
                attempted_failover += 1
                first_hop = route[1] if len(route) > 1 else None
                if first_hop is not None:
                    self.nodes[source_id]._consume_energy(
                        tx_bytes=self.data_packet_bytes,
                        tx_bits=self.data_packet_bits,
                        distance=self.nodes[source_id]._distance_to_next_hop(
                            first_hop))
            elif is_critical and path_index >= 2:
                self.nodes[source_id]._consume_energy(
                    tx_bytes=self.data_packet_bytes, tx_bits=self.data_packet_bits, distance=30.0)

            if str(self.ablation.get('attack_model_mode', 'legacy_compatible')).strip().lower() == 'legacy_compatible':
                # Byte-for-byte behavioral compatibility with the prior
                # baseline packet construction. Corrected attack scenarios use
                # the SensorNode-produced packet in the branch below.
                packet = {
                    "source_id": source_id,
                    "seq": seq_num,
                    "timestamp": self.current_time,
                    "size_bytes": self.data_packet_bytes,
                    "size_bits": self.data_packet_bits,
                    "path_header": [source_id],
                    "payload": payload_value,
                    "priority": "HIGH" if is_critical else "NORMAL",
                    "fragment_id": path_index,
                    "required_fragments": required_fragments,
                    "energy_trace": [(source_id, self.nodes[source_id].E_current)],
                }
            else:
                packet = dict(source_packet or {})
                packet.update({
                    "source_id": source_id,
                    "type": str(packet.get("type", "DATA")).upper(),
                    "seq": seq_num,
                    "timestamp": float(packet.get("timestamp", self.current_time)),
                    "size_bytes": int(packet.get("size_bytes", self.data_packet_bytes)),
                    "size_bits": float(packet.get("size_bits", self.data_packet_bits)),
                    "path_header": [source_id],
                    "payload": packet.get("payload", payload_value),
                    "priority": "HIGH" if is_critical else "NORMAL",
                    "fragment_id": path_index,
                    "required_fragments": required_fragments,
                    "energy_trace": list(packet.get(
                        "energy_trace", [(source_id, self.nodes[source_id].E_current)])),
                })
            packet["hash"] = compute_packet_hash(
                packet, self.nodes[source_id].crypto_keys)
            packet_dropped = False
            path_log = [source_id]
            malicious_delay_s = 0.0

            for idx in range(1, len(route)):
                current_hop_id = route[idx]
                if current_hop_id == self.BS_ID:
                    sender_id = route[idx - 1]
                    bs_distance = math.hypot(
                        self.nodes[sender_id].x - self.nodes[self.BS_ID].x,
                        self.nodes[sender_id].y - self.nodes[self.BS_ID].y)
                    self._account_implicit_mac_ack(
                        sender_id, self.BS_ID, bs_distance)
                    path_log.append(self.BS_ID)
                    break
                hop_node = self.nodes[current_hop_id]
                path_log.append(current_hop_id)
                sender_id = route[idx - 1]
                incoming_distance = math.hypot(
                    self.nodes[sender_id].x - hop_node.x,
                    self.nodes[sender_id].y - hop_node.y)
                self._account_implicit_mac_ack(
                    sender_id, current_hop_id, incoming_distance)
                next_hop_id = route[idx + 1] if idx + \
                    1 < len(route) else self.BS_ID
                if next_hop_id != self.BS_ID:
                    next_node = self.nodes[next_hop_id]
                    distance = math.hypot(
                        hop_node.x - next_node.x, hop_node.y - next_node.y)
                else:
                    bs_node = self.nodes[self.BS_ID]
                    distance = math.hypot(
                        hop_node.x - bs_node.x, hop_node.y - bs_node.y)
                success = hop_node.forward_packet(
                    packet, self.current_time, distance)
                if not success:
                    packet_dropped = True
                    break
                malicious_delay_s += hop_node.attack_delay_seconds(
                    self.current_time)

            if not packet_dropped and path_log[-1] == self.BS_ID:
                simulated_delay = len(path_log) * 0.100 + malicious_delay_s
                delivered_fragments += 1
                if is_failover_route:
                    successful_failover += 1
                if first_success_delay is None or simulated_delay < first_success_delay:
                    first_success_delay = simulated_delay
                packet_data = dict(packet)
                packet_data["path_header"] = path_log
                self.bs.data_plane_receive(
                    packet_data, path_log, receive_time=self.current_time + simulated_delay)
                if first_delivered_packet is None:
                    first_delivered_packet = dict(packet_data)

        if is_critical and self.ablation.get("use_critical_multipath_routing", False):
            if secondary_mode != 'failover':
                attempted_redundant = max(0, len(paths_to_trace) - 1)
                accepted_redundant = max(0, delivered_fragments - 1)
                rejected_redundant = max(
                    0, attempted_redundant - accepted_redundant)
                self.metrics.critical_redundant_copy_accepted += accepted_redundant
                self.metrics.critical_redundant_copy_rejected += rejected_redundant
            self.metrics.critical_failover_attempts += int(attempted_failover)
            self.metrics.critical_failover_successes += int(successful_failover)

        if delivered_fragments >= required_fragments:
            self.metrics.update_packet_transmission(
                is_critical=is_critical, is_control=False, size_bytes=self.data_packet_bytes,
                size_bits=self.data_packet_bits, is_delivered=True, delay=first_success_delay if first_success_delay is not None else 0.0,
                count_generated=False
            )

        after_received = self.metrics.received_data_packets
        if after_received > before_received:
            if key in self.metrics._unique_delivered_data_keys:
                delta_recv = after_received - before_received
                self.metrics.received_data_packets -= delta_recv
                if is_critical:
                    self.metrics.received_critical_packets = max(
                        before_received_critical, self.metrics.received_critical_packets - delta_recv)
                self.metrics.total_bits_received = before_bits
                del self.metrics.end_to_end_delays[before_delay_len:]
            else:
                self.metrics._unique_delivered_data_keys.add(key)

        # A replay attack is generated by the actual source node and sent
        # through the normal routing path with the same sequence number.  It is
        # disabled in the non-regression baseline and enabled only by explicit
        # end-to-end attack scenarios.
        source_node = self.nodes.get(source_id)
        if (allow_source_replay and first_delivered_packet is not None
                and source_node is not None
                and source_node.should_replay(self.current_time)):
            replay_packet = dict(first_delivered_packet)
            replay_packet['path_header'] = [source_id]
            replay_packet['energy_trace'] = list(
                first_delivered_packet.get('energy_trace', []))
            source_node._transmit_packet(replay_packet, bool(is_critical))
            self._trace_packet_path(
                source_id, seq_num, is_critical, payload_value,
                source_packet=replay_packet, seq_override=seq_num,
                count_generated=False, allow_source_replay=False)

    def _push_updated_routes(self):
        # Reference routes and adaptive thresholds remain unchanged.
        for n_id, table in self.bs.routing_tables.items():
            if n_id in self.nodes:
                new_adaptive_th = self.bs.get_adaptive_threshold_for_node(n_id)
                self.nodes[n_id].update_policies_from_bs(
                    primary_hop=table.get('primary'),
                    secondary_hop=table.get('secondary'),
                    new_threshold=new_adaptive_th,
                    secondary_mode=table.get('secondary_mode', 'redundant')
                )

        # Push DATA-only next hops separately.  No charging/control state is
        # overwritten by the clustered routing overlay.
        data_tables = self.bs.data_routing_tables or {}
        for n_id, table in data_tables.items():
            if n_id in self.nodes:
                self.nodes[n_id].update_data_routes(
                    primary_hop=table.get('primary'),
                    secondary_hop=table.get('secondary'),
                    secondary_mode=table.get('secondary_mode', 'redundant'))
