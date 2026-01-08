#!/usr/bin/env python3
"""Compare UQGrid vs ANDES power flow and dynamics trajectories.

Defaults: data/2bus_33.raw + data/GENROU.dyr
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uqgrid.io.parse import load_psse, add_dyr
from uqgrid.simulation.pflow import runpf
from uqgrid.simulation.dynamics import integrate_system, initialize_system
from uqgrid.simulation.jacobian_check import compare_jacobians
from uqgrid.simulation.config import IntegrationConfig


def _prepare_andes_dyr(dyr_path: str, workdir: Path) -> str:
    """ANDES expects space-separated DYR rows; some files are comma-separated."""
    text = Path(dyr_path).read_text()
    if "," not in text:
        return dyr_path
    cleaned = []
    for line in text.splitlines():
        if not line.strip():
            cleaned.append(line)
            continue
        # Replace commas with spaces to match ANDES parser expectations.
        cleaned.append(line.replace(",", " "))
    out_path = workdir / f"{Path(dyr_path).stem}_andes.dyr"
    out_path.write_text("\n".join(cleaned))
    return str(out_path)


def _load_andes(raw_path: str, dyr_path: str, *, fault_bus: int, fault_r: float, fault_on: float, fault_off: float, dt: float, tend: float):
    raw_path = str(Path(raw_path).resolve())
    dyr_path = str(Path(dyr_path).resolve())
    andes_home = Path.cwd() / ".andes"
    andes_home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(andes_home)

    try:
        import andes  # type: ignore
    except Exception as exc:
        raise RuntimeError("ANDES is not installed. Install with `pip install andes`.") from exc

    # Use andes.load to get a mutable System, then run PF/TDS explicitly.
    # Force input_format to bypass format auto-detection for non-standard headers.
    dyr_path = _prepare_andes_dyr(dyr_path, andes_home)
    ss = andes.load(
        raw_path,
        addfile=dyr_path,
        default_config=True,
        setup=False,
        input_format="psse",
        add_format="psse",
    )
    # Add a temporary bus fault disturbance.
    # ANDES uses rf/xf in p.u. to build a shunt impedance to ground.
    # Setting xf=0 makes it purely resistive, matching UQGrid's BusFault model.
    ss.add("Fault", dict(bus=fault_bus, tf=fault_on, tc=fault_off, rf=fault_r, xf=0.0))

    ss.setup()
    ss.PFlow.run()
    # Configure TDS to match UQGrid (backward Euler, fixed step).
    # ANDES defaults to trapezoidal if method is not set.
    if hasattr(ss, "TDS") and hasattr(ss.TDS, "config"):
        ss.TDS.config.method = "backeuler"
        ss.TDS.config.fixt = 1
        ss.TDS.config.tstep = dt
        ss.TDS.config.tf = tend
        try:
            ss.TDS.set_method("backeuler")
        except Exception:
            pass
    # Match UQGrid event handling: ANDES Fault restores algebraic variables
    # at clearance by default; disable restore to avoid smoothing artifacts.
    if hasattr(ss, "Fault") and hasattr(ss.Fault, "config"):
        ss.Fault.config.restore = 0
    return ss


def _andes_bus_ids(ss):
    df = ss.Bus.as_df()
    if "idx" in df.columns:
        bus_ids = df["idx"].to_numpy()
    elif "name" in df.columns:
        bus_ids = df["name"].to_numpy()
    else:
        bus_ids = np.arange(len(df))
    coerced = []
    for val in bus_ids:
        try:
            coerced.append(int(val))
        except Exception:
            coerced.append(val)
    return np.array(coerced)


def _andes_ts_matrix(ss, kind: str):
    ts = ss.dae.ts
    if kind == "x" and hasattr(ts, "x"):
        return ts.x
    if kind == "y" and hasattr(ts, "y"):
        return ts.y
    if hasattr(ts, "xy"):
        nx = getattr(ss.dae, "nx", None)
        if nx is None:
            return ts.xy
        return ts.xy[:, :nx] if kind == "x" else ts.xy[:, nx:]
    raise RuntimeError("ANDES time-series arrays not available for kind='%s'." % kind)


def _andes_series(ss, var, kind: str):
    mat = _andes_ts_matrix(ss, kind)
    return mat[:, var.a]


def _andes_init_values(ss, var, kind: str | None = None):
    if var is None:
        return None
    if kind is None:
        vtype = var.__class__.__name__
        kind = "x" if vtype == "State" else "y"
    mat = _andes_ts_matrix(ss, kind)
    return mat[0, var.a]


def _andes_series_auto(ss, var):
    if var is None:
        return None
    vtype = var.__class__.__name__
    kind = "x" if vtype == "State" else "y"
    return _andes_series(ss, var, kind)


def _bus_maps_uqgrid(psys, v_vector):
    ext2int = psys.ext2int
    int2ext = {v: k for k, v in ext2int.items()}
    bus_ids = []
    vmag = []
    vang = []
    for int_idx in range(psys.nbuses):
        ext_idx = int2ext[int_idx]
        bus_ids.append(int(ext_idx))
        vmag.append(v_vector[2 * int_idx])
        vang.append(v_vector[2 * int_idx + 1])
    return np.array(bus_ids), np.array(vmag), np.array(vang)


def _bus_slack_ext(psys):
    ext2int = psys.ext2int
    int2ext = {v: k for k, v in ext2int.items()}
    for int_idx, bus in enumerate(psys.buses):
        if bus.type == bus.SLACK:
            return int(int2ext[int_idx])
    return None


def _uqgrid_gen_series(psys, history, state_name: str):
    series = []
    bus_ids = []
    ext2int = psys.ext2int
    int2ext = {v: k for k, v in ext2int.items()}
    for gen in psys.gendyn:
        if not hasattr(gen, "state_list"):
            continue
        if state_name not in gen.state_list:
            continue
        state_idx = gen.state_list.index(state_name)
        global_idx = gen.dif_ptr + state_idx
        bus_ids.append(int(int2ext[gen.bus]))
        series.append(history[global_idx, :])
    return np.array(bus_ids), np.array(series)


def _andes_gen_series(ss, state: str):
    if not hasattr(ss, "GENROU"):
        raise RuntimeError("ANDES system has no GENROU model loaded.")
    gen = ss.GENROU
    if not hasattr(gen, state):
        raise RuntimeError(f"ANDES GENROU has no state '{state}'.")
    bus_ids = np.array([int(b) for b in gen.bus.v])
    series = _andes_series(ss, getattr(gen, state), "x")
    return bus_ids, series.T


def _interp_series(t_src, y_src, t_dst):
    if y_src.ndim == 1:
        return np.interp(t_dst, t_src, y_src)
    out = np.zeros((y_src.shape[0], t_dst.shape[0]))
    for i in range(y_src.shape[0]):
        out[i] = np.interp(t_dst, t_src, y_src[i])
    return out


def main():
    parser = argparse.ArgumentParser(description="Compare UQGrid vs ANDES dynamics.")
    parser.add_argument("--raw", default="data/2bus_33.raw", help="PSS/E RAW file")
    parser.add_argument("--dyr", default="data/GENROU.dyr", help="PSS/E DYR file")
    parser.add_argument("--tend", type=float, default=10.0, help="Simulation end time (s)")
    parser.add_argument("--dt", type=float, default=1.0 / 120.0, help="Time step (s)")
    parser.add_argument("--max-buses", type=int, default=3, help="Max buses to plot")
    parser.add_argument("--max-gens", type=int, default=3, help="Max gens to plot")
    parser.add_argument("--plot-dir", default="plots/andes_compare", help="Directory for output plots")
    parser.add_argument("--fault-bus", type=int, default=1, help="External bus number to fault")
    parser.add_argument("--fault-r", type=float, default=1.0, help="Fault resistance (p.u.)")
    parser.add_argument("--fault-on", type=float, default=0.1, help="Fault start time (s)")
    parser.add_argument("--fault-off", type=float, default=0.2, help="Fault clear time (s)")
    parser.add_argument("--power-injection", action="store_true", help="Use polar voltage states in UQGrid")
    parser.add_argument("--diag-bus", type=int, default=2, help="Bus number for angle diagnostics")
    parser.add_argument("--disable-exc-limits", action="store_true", help="Disable ANDES exciter limits/saturation")
    parser.add_argument("--check-jacobian", action="store_true", help="Run FD Jacobian check vs analytical")
    args = parser.parse_args()

    # Load and solve power flow
    psys = load_psse(args.raw)
    add_dyr(psys, args.dyr)
    psys.createYbusComplex()
    pf = runpf(psys, verbose=False)
    z0, theta = initialize_system(psys, pf)

    if args.fault_bus not in psys.ext2int:
        raise ValueError(f"Fault bus {args.fault_bus} not found in system.")
    fault_bus_int = psys.ext2int[args.fault_bus]
    psys.add_busfault(fault_bus_int, args.fault_r)

    ss = _load_andes(
        args.raw,
        args.dyr,
        fault_bus=args.fault_bus,
        fault_r=args.fault_r,
        fault_on=args.fault_on,
        fault_off=args.fault_off,
        dt=args.dt,
        tend=args.tend,
    )
    if args.disable_exc_limits and hasattr(ss, "ESDC1A") and ss.ESDC1A.n > 0:
        # Disable anti-windup limits and saturation by parameter overrides.
        for name, val in [
            ("VRMAX", 999.0),
            ("VRMIN", -999.0),
            ("E1", 0.0),
            ("SE1", 0.0),
            ("E2", 0.0),
            ("SE2", 0.0),
        ]:
            if hasattr(ss.ESDC1A, name):
                getattr(ss.ESDC1A, name).v[:] = val

    # Power flow comparison
    uq_bus_ids, uq_vmag, uq_vang = _bus_maps_uqgrid(psys, pf.v_vector)
    an_bus_ids = _andes_bus_ids(ss)
    an_vmag = np.array(ss.Bus.v.v)
    an_vang = np.array(ss.Bus.a.v)

    shared = [b for b in uq_bus_ids if b in set(an_bus_ids)]
    if not shared:
        raise RuntimeError("No overlapping buses between UQGrid and ANDES.")

    an_bus_to_idx = {b: i for i, b in enumerate(an_bus_ids)}
    uq_bus_to_idx = {b: i for i, b in enumerate(uq_bus_ids)}
    slack = _bus_slack_ext(psys)
    angle_offset = 0.0
    if slack is not None and slack in an_bus_to_idx and slack in uq_bus_to_idx:
        angle_offset = an_vang[an_bus_to_idx[slack]] - uq_vang[uq_bus_to_idx[slack]]

    vmag_diff = []
    vang_diff = []
    for b in shared:
        vmag_diff.append(abs(uq_vmag[uq_bus_to_idx[b]] - an_vmag[an_bus_to_idx[b]]))
        vang_diff.append(abs(uq_vang[uq_bus_to_idx[b]] - (an_vang[an_bus_to_idx[b]] - angle_offset)))
    print(f"PF max |Vm| diff: {max(vmag_diff):.3e}")
    print(f"PF max |Va| diff: {max(vang_diff):.3e} rad")

    # Run dynamics
    config = IntegrationConfig(
        power_injection=args.power_injection,
        tend=args.tend,
        dt=args.dt,
        steps=-1,
        verbose=False,
        fsolve=False,
        petsc=False,
        ton=args.fault_on,
        toff=args.fault_off,
    )
    uq_res = integrate_system(psys, config=config)

    ss.TDS.run()

    t_uq = uq_res["tvec"]
    hist = uq_res["history"]
    t_an = np.array(ss.dae.ts.t)

    # UQGrid bus series:
    # - If power_injection is False (default), UQGrid stores rectangular v=(vr, vi).
    # - If power_injection is True, UQGrid stores polar (vmag, vang).
    busmag_idx = psys.busmag_idx_set()
    busang_idx = psys.busang_idx_set()
    uq_bus_vr_ts = np.array([hist[i, :] for i in busmag_idx])
    uq_bus_vi_ts = np.array([hist[i, :] for i in busang_idx])
    if args.power_injection:
        uq_bus_vmag_ts = uq_bus_vr_ts
        uq_bus_vang_ts = uq_bus_vi_ts
    else:
        uq_bus_vmag_ts = np.sqrt(uq_bus_vr_ts**2 + uq_bus_vi_ts**2)
        uq_bus_vang_ts = np.arctan2(uq_bus_vi_ts, uq_bus_vr_ts)

    # ANDES bus series
    an_bus_vmag_ts = _andes_series(ss, ss.Bus.v, "y").T
    an_bus_vang_ts = _andes_series(ss, ss.Bus.a, "y").T

    # Align ANDES to UQGrid times
    an_bus_vmag_ts_i = _interp_series(t_an, an_bus_vmag_ts, t_uq)
    an_bus_vang_ts_i = _interp_series(t_an, an_bus_vang_ts, t_uq)

    # Apply slack angle alignment at each time step to remove reference offsets.
    if slack is not None and slack in an_bus_to_idx and slack in uq_bus_to_idx:
        uq_slack_idx = uq_bus_to_idx[slack]
        an_slack_idx = an_bus_to_idx[slack]
        uq_bus_vang_ts = uq_bus_vang_ts - uq_bus_vang_ts[uq_slack_idx]
        an_bus_vang_ts_i = an_bus_vang_ts_i - an_bus_vang_ts_i[an_slack_idx]

    # Unwrap angles over time to avoid 2*pi discontinuities.
    uq_bus_vang_ts = np.unwrap(uq_bus_vang_ts, axis=1)
    an_bus_vang_ts_i = np.unwrap(an_bus_vang_ts_i, axis=1)

    # Generator series
    uq_gen_buses, uq_delta = _uqgrid_gen_series(psys, hist, "delta")
    _, uq_omega = _uqgrid_gen_series(psys, hist, "w")
    an_gen_buses, an_delta = _andes_gen_series(ss, "delta")
    _, an_omega = _andes_gen_series(ss, "omega")

    # Match generators by bus
    gen_shared = [b for b in uq_gen_buses if b in set(an_gen_buses)]
    an_gen_to_idx = {b: i for i, b in enumerate(an_gen_buses)}
    uq_gen_to_idx = {b: i for i, b in enumerate(uq_gen_buses)}

    an_delta_i = _interp_series(t_an, an_delta, t_uq)
    an_omega_i = _interp_series(t_an, an_omega, t_uq)

    # Align rotor angles by initial offsets
    for b in gen_shared:
        i_uq = uq_gen_to_idx[b]
        i_an = an_gen_to_idx[b]
        offset = an_delta_i[i_an, 0] - uq_delta[i_uq, 0]
        an_delta_i[i_an, :] -= offset
        omega_offset = an_omega_i[i_an, 0] - uq_omega[i_uq, 0]
        an_omega_i[i_an, :] -= omega_offset

    # Unwrap rotor angles over time to avoid 2*pi discontinuities.
    if uq_delta.size:
        uq_delta = np.unwrap(uq_delta, axis=1)
    if an_delta_i.size:
        an_delta_i = np.unwrap(an_delta_i, axis=1)

    # Initial vector check (bus and generator states)
    init_vmag_diff = []
    init_vang_diff = []
    for b in shared:
        i_uq = uq_bus_to_idx[b]
        i_an = an_bus_to_idx[b]
        init_vmag_diff.append(abs(uq_bus_vmag_ts[i_uq, 0] - an_bus_vmag_ts_i[i_an, 0]))
        init_vang_diff.append(abs(uq_bus_vang_ts[i_uq, 0] - an_bus_vang_ts_i[i_an, 0]))
    print(f"Init |Vm| max diff: {max(init_vmag_diff):.3e}")
    print(f"Init |Va| max diff: {max(init_vang_diff):.3e} rad")

    init_delta_diff = []
    init_omega_diff = []
    for b in gen_shared:
        i_uq = uq_gen_to_idx[b]
        i_an = an_gen_to_idx[b]
        init_delta_diff.append(abs(uq_delta[i_uq, 0] - an_delta_i[i_an, 0]))
        init_omega_diff.append(abs(uq_omega[i_uq, 0] - an_omega_i[i_an, 0]))
    if init_delta_diff:
        print(f"Init |delta| max diff: {max(init_delta_diff):.3e}")
        print(f"Init |omega| max diff: {max(init_omega_diff):.3e}")

    # Trajectory diffs for machine variables (delta/omega)
    if gen_shared:
        for b in gen_shared[: args.max_gens]:
            i_uq = uq_gen_to_idx[b]
            i_an = an_gen_to_idx[b]
            delta_diff = np.abs(uq_delta[i_uq] - an_delta_i[i_an])
            omega_diff = np.abs(uq_omega[i_uq] - an_omega_i[i_an])
            print(
                f"Gen@Bus {b} max |delta diff|: {np.max(delta_diff):.3e}, "
                f"max |omega diff|: {np.max(omega_diff):.3e}"
            )

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Angle diagnostics around fault events for a selected bus
    diag_bus = args.diag_bus
    if diag_bus in uq_bus_to_idx and diag_bus in an_bus_to_idx:
        i_uq = uq_bus_to_idx[diag_bus]
        i_an = an_bus_to_idx[diag_bus]

        def _nearest_index(tvec, tval):
            return int(np.argmin(np.abs(tvec - tval)))

        idx_on = _nearest_index(t_uq, args.fault_on)
        idx_off = _nearest_index(t_uq, args.fault_off)

        def _jump(ts, idx):
            if idx <= 0 or idx >= len(ts) - 1:
                return 0.0, 0.0
            pre = ts[idx] - ts[idx - 1]
            post = ts[idx + 1] - ts[idx]
            return pre, post

        uq_on_pre, uq_on_post = _jump(uq_bus_vang_ts[i_uq], idx_on)
        uq_off_pre, uq_off_post = _jump(uq_bus_vang_ts[i_uq], idx_off)
        an_on_pre, an_on_post = _jump(an_bus_vang_ts_i[i_an], idx_on)
        an_off_pre, an_off_post = _jump(an_bus_vang_ts_i[i_an], idx_off)

        diff_ts = np.abs(uq_bus_vang_ts[i_uq] - an_bus_vang_ts_i[i_an])
        max_diff = float(np.max(diff_ts))
        max_diff_t = float(t_uq[int(np.argmax(diff_ts))])

        print(f"Bus {diag_bus} angle jump @t_on (pre/post): UQGrid {uq_on_pre:.3e}/{uq_on_post:.3e}, ANDES {an_on_pre:.3e}/{an_on_post:.3e}")
        print(f"Bus {diag_bus} angle jump @t_off (pre/post): UQGrid {uq_off_pre:.3e}/{uq_off_post:.3e}, ANDES {an_off_pre:.3e}/{an_off_post:.3e}")
        print(f"Bus {diag_bus} max |angle diff|: {max_diff:.3e} at t={max_diff_t:.4f}s")

        # Zoom plot around the disturbance window
        zoom_mask = (t_uq >= args.fault_on - 0.05) & (t_uq <= args.fault_off + 0.05)
        if np.any(zoom_mask):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(t_uq[zoom_mask], uq_bus_vang_ts[i_uq][zoom_mask], label="UQGrid")
            ax.plot(t_uq[zoom_mask], an_bus_vang_ts_i[i_an][zoom_mask], "--", label="ANDES")
            ax.set_title(f"Bus {diag_bus} voltage angle (zoom)")
            ax.set_xlabel("Time (s)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
        fig.savefig(plot_dir / f"vang_bus{diag_bus}_zoom.png", dpi=150)

    # Controller initialization comparison (IEESGO, ESDC1A) by bus
    # Map ANDES SynGen device names to bus ids
    syn_to_bus = {}
    if hasattr(ss, "GENROU"):
        for idx, bus in zip(ss.GENROU.idx.v, ss.GENROU.bus.v):
            syn_to_bus[idx] = int(bus)
    if hasattr(ss, "GENCLS"):
        for idx, bus in zip(ss.GENCLS.idx.v, ss.GENCLS.bus.v):
            syn_to_bus[idx] = int(bus)

    def _andes_bus_map(model):
        if model is None or getattr(model, "n", 0) == 0:
            return {}
        if hasattr(model, "bus"):
            bus_arr = model.bus.v
            return {int(bus_arr[i]): i for i in range(len(bus_arr))}
        if hasattr(model, "syn"):
            bus_arr = [syn_to_bus.get(s, None) for s in model.syn.v]
            return {int(bus_arr[i]): i for i in range(len(bus_arr)) if bus_arr[i] is not None}
        return {}

    # UQGrid controllers
    dif_size = psys.num_dof_dif
    int2ext_bus = {v: k for k, v in psys.ext2int.items()}
    for gov in getattr(psys, "gov", []):
        if gov.state_list and "PF0" in gov.state_list:
            bus_ext = int(int2ext_bus.get(gov.bus, gov.bus))
            print(f"UQGrid IEESGO init (bus {bus_ext}):")
            print(
                f"  PF0={z0[gov.dif_ptr]:.6g}, PLL={z0[gov.dif_ptr+1]:.6g}, "
                f"TP1={z0[gov.dif_ptr+2]:.6g}, TP2={z0[gov.dif_ptr+3]:.6g}, "
                f"TP3={z0[gov.dif_ptr+4]:.6g}, p_m={z0[dif_size + gov.alg_ptr]:.6g}"
            )

            model = getattr(ss, "IEESGO", None)
            bus_map = _andes_bus_map(model)
            if model is not None and bus_ext in bus_map:
                i = bus_map[bus_ext]
                F1 = _andes_init_values(ss, model.F1_y)[i]
                F2 = _andes_init_values(ss, model.F2_y)[i]
                F3 = _andes_init_values(ss, model.F3_y)[i]
                F4 = _andes_init_values(ss, model.F4_y)[i]
                F5 = _andes_init_values(ss, model.F5_y)[i]
                pout = _andes_init_values(ss, model.pout)[i]
                print(f"ANDES IEESGO init (bus {bus_ext}):")
                print(
                    f"  F1={F1:.6g}, F2={F2:.6g}, F3={F3:.6g}, F4={F4:.6g}, "
                    f"F5={F5:.6g}, pout={pout:.6g}"
                )

    for exc in getattr(psys, "exc", []):
        if exc.state_list and "vr1" in exc.state_list:
            bus_ext = int(int2ext_bus.get(exc.bus, exc.bus))
            print(f"UQGrid ESDC1A init (bus {bus_ext}):")
            print(
                f"  vr1={z0[exc.dif_ptr]:.6g}, vr2={z0[exc.dif_ptr+1]:.6g}, "
                f"e_fd={z0[exc.dif_ptr+2]:.6g}, vref={exc.vref:.6g}"
            )

            model = getattr(ss, "ESDC1A", None)
            bus_map = _andes_bus_map(model)
            if model is not None and bus_ext in bus_map:
                i = bus_map[bus_ext]
                vref = _andes_init_values(ss, model.vref)[i]
                vi = _andes_init_values(ss, model.vi)[i]
                LL = _andes_init_values(ss, model.LL_y)[i]
                LA = _andes_init_values(ss, model.LA_y)[i]
                INT = _andes_init_values(ss, model.INT_y)[i]
                WF = _andes_init_values(ss, model.WF_y)[i]
                vout = _andes_init_values(ss, model.vout)[i]
                vf = _andes_init_values(ss, model.vf)[i]
                print(f"ANDES ESDC1A init (bus {bus_ext}):")
                print(
                    f"  vref={vref:.6g}, vi={vi:.6g}, LL={LL:.6g}, LA={LA:.6g}, "
                    f"INT={INT:.6g}, WF={WF:.6g}, vout={vout:.6g}, vf={vf:.6g}"
                )
        if exc.state_list and "x1" in exc.state_list:
            bus_ext = int(int2ext_bus.get(exc.bus, exc.bus))
            print(f"UQGrid SEXS init (bus {bus_ext}):")
            print(
                f"  x1={z0[exc.dif_ptr]:.6g}, e_fd={z0[exc.dif_ptr+1]:.6g}, "
                f"vref={exc.vref:.6g}"
            )

            model = getattr(ss, "SEXS", None)
            bus_map = _andes_bus_map(model)
            if model is not None and bus_ext in bus_map:
                i = bus_map[bus_ext]
                vref = _andes_init_values(ss, model.vref)[i]
                vi = _andes_init_values(ss, model.vi)[i]
                LL = _andes_init_values(ss, model.LL_y)[i]
                LAW = _andes_init_values(ss, model.LAW_y)[i]
                vout = _andes_init_values(ss, model.vout)[i]
                vf = _andes_init_values(ss, model.vf)[i]
                print(f"ANDES SEXS init (bus {bus_ext}):")
                print(
                    f"  vref={vref:.6g}, vi={vi:.6g}, LL={LL:.6g}, LAW={LAW:.6g}, "
                    f"vout={vout:.6g}, vf={vf:.6g}"
                )

            # Trajectory comparison for SEXS internals (e_fd, x1, lead-lag output)
            if model is not None and bus_ext in bus_map:
                i = bus_map[bus_ext]
                # UQGrid series
                x1_uq = hist[exc.dif_ptr, :]
                efd_uq = hist[exc.dif_ptr + 1, :]
                vm_uq = uq_bus_vmag_ts[uq_bus_to_idx[bus_ext]]
                y1_uq = x1_uq + exc.TA_TB * (exc.vref - vm_uq)

                # ANDES series
                vout_an = _andes_series_auto(ss, model.vout).T[i]
                law_an = _andes_series_auto(ss, model.LAW_y).T[i]
                ll_y_an = _andes_series_auto(ss, model.LL_y).T[i]
                ll_x_an = _andes_series_auto(ss, model.LL_x).T[i] if hasattr(model, "LL_x") else None

                # Interpolate ANDES to UQGrid time grid
                vout_an_i = _interp_series(t_an, vout_an, t_uq)
                law_an_i = _interp_series(t_an, law_an, t_uq)
                ll_y_an_i = _interp_series(t_an, ll_y_an, t_uq)
                ll_x_an_i = _interp_series(t_an, ll_x_an, t_uq) if ll_x_an is not None else None

                print(f"SEXS traj (bus {bus_ext}) max |e_fd - vout|: {np.max(np.abs(efd_uq - vout_an_i)):.3e}")
                print(f"SEXS traj (bus {bus_ext}) max |e_fd - LAW_y|: {np.max(np.abs(efd_uq - law_an_i)):.3e}")
                print(f"SEXS traj (bus {bus_ext}) max |y1 - LL_y|: {np.max(np.abs(y1_uq - ll_y_an_i)):.3e}")
                if ll_x_an_i is not None:
                    print(f"SEXS traj (bus {bus_ext}) max |x1 - LL_x|: {np.max(np.abs(x1_uq - ll_x_an_i)):.3e}")

    if args.check_jacobian:
        print("== Jacobian FD check (top mismatches) ==")
        from uqgrid.simulation.dynamics import preallocate_jacobian
        from uqgrid.simulation.jacobian import residual_jacobian
        J = preallocate_jacobian(psys)
        residual_jacobian(J, z0, theta, psys)
        mismatches = compare_jacobians(psys, z0, theta, J, eps=1e-6, top_k=10, tol=1e-8)
        for m in mismatches:
            print(
                f"{m['row_desc']} <- {m['col_desc']}: "
                f"analytical={m['analytical']:.3e}, fd={m['finite_diff']:.3e}, "
                f"|diff|={m['abs_diff']:.3e}"
            )
    # Plot trajectories
    # Rotor angle plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for b in gen_shared[: args.max_gens]:
        i_uq = uq_gen_to_idx[b]
        i_an = an_gen_to_idx[b]
        ax.plot(t_uq, uq_delta[i_uq], label=f"UQGrid G@{b}")
        ax.plot(t_uq, an_delta_i[i_an], "--", label=f"ANDES G@{b}")
    ax.set_title("Rotor angle (delta)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    delta_path = plot_dir / "delta.png"
    fig.savefig(delta_path, dpi=150)

    # Rotor speed plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for b in gen_shared[: args.max_gens]:
        i_uq = uq_gen_to_idx[b]
        i_an = an_gen_to_idx[b]
        ax.plot(t_uq, uq_omega[i_uq], label=f"UQGrid G@{b}")
        ax.plot(t_uq, an_omega_i[i_an], "--", label=f"ANDES G@{b}")
    ax.set_title("Rotor speed (omega)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    omega_path = plot_dir / "omega.png"
    fig.savefig(omega_path, dpi=150)

    # Bus voltage magnitude/angle plots (one plot per bus per variable)
    for b in shared[: args.max_buses]:
        i_uq = uq_bus_to_idx[b]
        i_an = an_bus_to_idx[b]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t_uq, uq_bus_vmag_ts[i_uq], label="UQGrid")
        ax.plot(t_uq, an_bus_vmag_ts_i[i_an], "--", label="ANDES")
        ax.set_title(f"Bus {b} voltage magnitude")
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"vmag_bus{b}.png", dpi=150)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t_uq, uq_bus_vang_ts[i_uq], label="UQGrid")
        ax.plot(t_uq, an_bus_vang_ts_i[i_an], "--", label="ANDES")
        ax.set_title(f"Bus {b} voltage angle (slack-referenced)")
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"vang_bus{b}.png", dpi=150)

    print(f"Saved plots to {plot_dir}")


if __name__ == "__main__":
    main()
