"""Example: generate one OSL-style case on ACTIVSg200 and plot it.

Injects two signal sources into the ACTIVSg200 dynamics simulation:
  * a ~0.8 Hz sinusoidal forced oscillation on the TGOV1 governor pref
    of generator at PSSE bus 49,
  * colored load noise on every load (low- + high-frequency components).

Then runs the case through the PMU emulator and produces three plots:
  1. raw bus-voltage magnitude trajectories (sim-rate),
  2. PMU bus-voltage magnitude at a handful of observed buses,
  3. PMU branch current magnitude at observed branches.

Reference
---------
Maslennikov, S. and Wang, B. (2022). NREL/CP-6A40-81394.
https://www.nrel.gov/docs/fy22osti/81394.pdf

Usage
-----
    python bin/activsg200_osl.py [--outdir plots/osl] [--tend 20.0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uqgrid.osl import (
    ColoredNoise,
    ForcedOscillation,
    PMUEmulator,
    build_osl_case,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = REPO_ROOT / "data" / "ACTIVSg200.raw"
DEFAULT_DYR = REPO_ROOT / "data" / "ACTIVSg200.dyr"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raw", default=str(DEFAULT_RAW))
    p.add_argument("--dyr", default=str(DEFAULT_DYR))
    p.add_argument("--outdir", default="plots/osl",
                   help="Where to write PNGs + case artifacts.")
    p.add_argument("--tend", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=1.0 / 240.0)
    p.add_argument("--fo-bus", type=int, default=49,
                   help="PSSE generator bus to host the governor FO.")
    p.add_argument("--fo-freq", type=float, default=0.8)
    p.add_argument("--fo-amp", type=float, default=0.20,
                   help="FO amplitude in per-unit of system MVA (TGOV1 pref).")
    p.add_argument("--fo-start", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-noise", action="store_true",
                   help="Disable the colored-noise injector.")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    injectors = [
        ForcedOscillation(
            target=("gov", args.fo_bus),
            freq_hz=args.fo_freq,
            amplitude=args.fo_amp,
            t_start=args.fo_start,
            t_end=args.tend,
            waveform="sine",
        ),
    ]
    noise = None if args.no_noise else ColoredNoise(
        sigma_lf=0.002, sigma_hf=0.001, seed=args.seed,
    )

    case = build_osl_case(
        raw=args.raw,
        dyr=args.dyr,
        forced_oscillations=injectors,
        colored_noise=noise if noise is not None else False,
        tend=args.tend,
        dt=args.dt,
        pmu=PMUEmulator(
            rate_hz=30.0,
            p_class_fraction=0.7,
            observed_buses="50%",
            missing_rate=0.0,
            seed=args.seed,
        ),
        label=f"ACTIVSg200_gov{args.fo_bus}_{args.fo_freq:g}Hz",
        keep_raw=True,
    )

    safe_label = case.metadata["label"].replace(".", "p")
    stem = outdir / safe_label
    npz, js = case.export(stem)
    print(f"wrote {npz}\nwrote {js}")

    _plot_raw_voltages(case, args, stem)
    _plot_pmu_voltages(case, args, stem)
    _plot_pmu_currents(case, stem)


def _plot_raw_voltages(case, args, stem: Path):
    history = case.raw_history
    tvec = case.raw_tvec
    psys_nbuses = (history.shape[0] - 0) // 1  # not needed; recover from metadata
    nbus = case.metadata["system"]["nbuses"]
    v_block = history[-2 * nbus:, :]
    vr = v_block[0::2, :]
    vi = v_block[1::2, :]
    vm = np.sqrt(vr ** 2 + vi ** 2)

    obs_psse = case.pmu["observed_buses_psse"].tolist()
    obs_int = case.pmu["observed_buses_internal"].tolist()
    sel = obs_int[:5]
    sel_labels = obs_psse[:5]

    fig, ax = plt.subplots(figsize=(9, 4))
    for k, b_int in enumerate(sel):
        ax.plot(tvec, vm[b_int, :], lw=0.8, label=f"PSSE bus {sel_labels[k]}")
    ax.axvline(args.fo_start, color="k", lw=0.5, ls="--",
               label=f"FO on at t={args.fo_start:g}s")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("|V| (pu)")
    ax.set_title("Raw bus voltage magnitudes (simulation rate)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".raw_voltages.png"), dpi=130)
    plt.close(fig)


def _plot_pmu_voltages(case, args, stem: Path):
    pmu = case.pmu
    t = pmu["t"]
    V = pmu["V_mag"]
    obs = pmu["observed_buses_psse"].tolist()

    fig, ax = plt.subplots(figsize=(9, 4))
    for k in range(min(5, V.shape[0])):
        ax.plot(t, V[k, :], lw=0.8, label=f"PSSE bus {obs[k]} ({pmu['pmu_class'][k]})")
    ax.axvline(args.fo_start, color="k", lw=0.5, ls="--")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("|V| (pu)")
    ax.set_title(f"PMU voltages — {pmu['rate_hz']:.0f} Hz, "
                 f"FO {args.fo_freq:g} Hz at gen bus {args.fo_bus}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".pmu_voltages.png"), dpi=130)
    plt.close(fig)


def _plot_pmu_currents(case, stem: Path):
    pmu = case.pmu
    if pmu["I_mag"].size == 0:
        return
    t = pmu["t"]
    I = pmu["I_mag"]
    branches = pmu["branches"]

    fig, ax = plt.subplots(figsize=(9, 4))
    for k in range(min(5, I.shape[0])):
        fr, to = branches[k]
        ax.plot(t, I[k, :], lw=0.8, label=f"branch {fr}->{to}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("|I| (pu)")
    ax.set_title("PMU branch currents (sample)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".pmu_currents.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
