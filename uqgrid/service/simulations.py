"""Headless power-flow and dynamics orchestration."""

import platform
import warnings
from datetime import datetime, timezone

import numpy as np

import uqgrid
from uqgrid.io.parse import add_dyr, load_matpower, load_psse
from uqgrid.simulation.config import IntegrationConfig
from uqgrid.simulation.dynamics import integrate_system
from uqgrid.simulation.pflow import runpf

from .cases import LocalCaseRepository
from .results import LocalResultRepository
from .schemas import (
    CaseFormat,
    DynamicsJobRequest,
    DynamicsResultSummary,
    PowerFlowJobRequest,
    PowerFlowResultSummary,
    SignalDescriptor,
)


class SimulationService:
    """Run UQGrid from persisted case IDs and create durable result artifacts."""

    def __init__(self, cases: LocalCaseRepository, results: LocalResultRepository):
        self.cases = cases
        self.results = results

    def run_power_flow(self, owner_id: str, request: PowerFlowJobRequest):
        case = self.cases.get(owner_id, request.case_id)
        primary, _ = self.cases.resolve_files(owner_id, request.case_id)
        psys = load_matpower(str(primary)) if case.format == CaseFormat.MATPOWER else load_psse(str(primary))
        psys.createYbusComplex()
        solution = runpf(
            psys,
            verbose=False,
            enforce_q_limits=request.options.enforce_q_limits,
            q_limit_tolerance=request.options.q_limit_tolerance,
            max_q_limit_iterations=request.options.max_q_limit_iterations,
        )
        result_id = self.results.new_result_id()
        signals = []
        arrays = {"time_s": np.array([0.0])}
        for index, bus in enumerate(psys.buses):
            bus_id = str(bus.id)
            for field, unit, value in (
                ("voltage_magnitude", "pu", solution.v_magnitudes[index]),
                ("voltage_angle", "rad", solution.v_angles[index]),
            ):
                name = f"bus.{bus_id}.{field}"
                arrays[name] = np.array([value], dtype=float)
                signals.append(SignalDescriptor(
                    name=name,
                    unit=unit,
                    entity_type="bus",
                    entity_id=bus_id,
                    source_state_indices=[2 * index + (field == "voltage_angle")],
                ))
        summary = PowerFlowResultSummary(
            result_id=result_id,
            converged=True,
            residual_norm=float(solution.residual_norm),
            bus_count=psys.nbuses,
            generator_count=psys.ngens,
            voltage_min_pu=float(np.min(solution.v_magnitudes)),
            voltage_max_pu=float(np.max(solution.v_magnitudes)),
            q_limit_iterations=solution.q_limit_iterations,
            q_limit_event_count=len(solution.q_limit_events),
        )
        arrays.update({
            "bus_voltage_magnitude_pu": solution.v_magnitudes,
            "bus_voltage_angle_rad": solution.v_angles,
            "generator_p_pu": solution.gen_psch,
            "generator_q_pu": solution.gen_qsch,
        })
        return self.results.create(
            owner_id,
            summary,
            arrays,
            signals,
            state_metadata={},
            manifest=self._manifest(case, "power_flow", request.model_dump(mode="json")),
        )

    def run_dynamics(self, owner_id: str, request: DynamicsJobRequest):
        case = self.cases.get(owner_id, request.case_id)
        primary, dynamics = self.cases.resolve_files(owner_id, request.case_id)
        if case.format != CaseFormat.PSSE or dynamics is None:
            raise ValueError("dynamics requires a PSS/E case with a .dyr file")
        psys = load_psse(str(primary))
        add_dyr(psys, str(dynamics))
        event = request.scenario.events[0]
        try:
            internal_bus = psys.ext2int[event.bus_id]
        except KeyError as exc:
            raise ValueError(f"unknown external bus ID: {event.bus_id}") from exc
        psys.add_busfault(internal_bus, event.impedance_pu)
        psys.createYbusComplex()
        config = IntegrationConfig(
            method=request.integration.method,
            dt=request.integration.dt_s,
            tend=request.integration.end_s,
            ton=event.start_s,
            toff=event.clear_s,
            petsc=False,
            enforce_q_limits=request.integration.enforce_q_limits,
            enforce_dynamic_limits=request.integration.enforce_dynamic_limits,
        )
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            simulation = integrate_system(psys, config)
        history = simulation["history"]
        arrays, signals = self._dynamic_signals(psys, simulation["tvec"], history)
        state_metadata = self._state_metadata(psys)
        speeds = history[psys.genspeed_idx_set(), :]
        voltages = np.vstack([arrays[item.name] for item in signals if item.name.endswith(".voltage_magnitude")])
        result_id = self.results.new_result_id()
        summary = DynamicsResultSummary(
            result_id=result_id,
            step_count=len(simulation["tvec"]),
            state_count=history.shape[0],
            end_s=float(simulation["tvec"][-1]),
            minimum_bus_voltage_pu=float(np.min(voltages)),
            maximum_abs_generator_speed_pu=float(np.max(np.abs(speeds))),
            warnings=[str(record.message) for record in records],
        )
        arrays["history"] = history
        return self.results.create(
            owner_id,
            summary,
            arrays,
            signals,
            state_metadata,
            self._manifest(case, "dynamics", request.model_dump(mode="json")),
        )

    @staticmethod
    def _dynamic_signals(psys, time, history):
        arrays = {"time_s": np.asarray(time, dtype=float)}
        signals = []
        voltage_offset = psys.num_dof_dif + psys.num_dof_alg
        for index, bus in enumerate(psys.buses):
            real = history[voltage_offset + 2 * index]
            imaginary = history[voltage_offset + 2 * index + 1]
            name = f"bus.{bus.id}.voltage_magnitude"
            arrays[name] = np.hypot(real, imaginary)
            signals.append(SignalDescriptor(
                name=name,
                unit="pu",
                entity_type="bus",
                entity_id=str(bus.id),
                source_state_indices=[voltage_offset + 2 * index, voltage_offset + 2 * index + 1],
            ))
        inverse_bus = {internal: external for external, internal in psys.ext2int.items()}
        for generator in psys.gendyn:
            bus_id = inverse_bus[generator.bus]
            name = f"generator.{bus_id}.{generator.id_tag}.speed"
            arrays[name] = np.asarray(history[generator.dif_ptr + 4], dtype=float)
            signals.append(SignalDescriptor(
                name=name,
                unit="pu_deviation",
                entity_type="generator",
                entity_id=f"{bus_id}:{generator.id_tag}",
                source_state_indices=[generator.dif_ptr + 4],
            ))
        return arrays, signals

    @staticmethod
    def _state_metadata(psys):
        metadata = {}
        for device in psys.devices:
            for offset in range(device.dif_dim):
                name = device.state_list[offset] if offset < len(device.state_list) else f"state_{offset}"
                metadata[str(device.dif_ptr + offset)] = {
                    "type": "differential",
                    "model": device.__class__.__name__,
                    "device_id": str(device.id_tag),
                    "state_name": name,
                }
            for offset in range(device.alg_dim):
                local = device.dif_dim + offset
                name = device.state_list[local] if local < len(device.state_list) else f"state_{offset}"
                metadata[str(psys.num_dof_dif + device.alg_ptr + offset)] = {
                    "type": "algebraic",
                    "model": device.__class__.__name__,
                    "device_id": str(device.id_tag),
                    "state_name": name,
                }
        voltage_offset = psys.num_dof_dif + psys.num_dof_alg
        for index, bus in enumerate(psys.buses):
            metadata[str(voltage_offset + 2 * index)] = {
                "type": "network_voltage",
                "bus_id": str(bus.id),
                "state_name": "real",
            }
            metadata[str(voltage_offset + 2 * index + 1)] = {
                "type": "network_voltage",
                "bus_id": str(bus.id),
                "state_name": "imaginary",
            }
        return metadata

    @staticmethod
    def _manifest(case, kind, request):
        return {
            "schema_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "case_id": case.case_id,
            "case_sha256": case.sha256,
            "request": request,
            "software": {
                "uqgrid": uqgrid.__version__,
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
        }
