#!/usr/bin/env python3
"""Monte Carlo experiments for ARKIMEX slow/fast partitions.

This script randomly samples slow differential subsets for a given power
system case, executes the PETSc-based ARKIMEX integrator, and records the
relationship between slow partition size and overall runtime (from PETSc's
log summary). A scatter plot of runtime versus slow-partition percentage is
written to disk for quick inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib

# Use a non-interactive backend so the script works on headless machines
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uqgrid.io.parse import add_dyr, load_psse
from uqgrid.utils.partition import (
    extract_runtime_from_log,
    format_index_list,
    sample_slow_indices,
)

DEFAULT_RAW = "data/IEEE39_v33.raw"
DEFAULT_DYR = "data/IEEE39.dyr"
DEFAULT_TEND = 1.0
DEFAULT_DT = 1.0 / 120.0
DEFAULT_TON = 0.05
DEFAULT_TOFF = 0.1

@dataclass
class ExperimentResult:
    slow_count: int
    total_diff: int
    runtime_sec: Optional[float]
    log_path: Path
    exit_code: int
    stderr: str
    stdout: str

    @property
    def slow_fraction(self) -> float:
        if self.total_diff == 0:
            return 0.0
        return self.slow_count / self.total_diff

    @property
    def slow_percent(self) -> float:
        return 100.0 * self.slow_fraction


def split_script_and_petsc_args(argv: Sequence[str]) -> Tuple[List[str], List[str]]:
    if "--" in argv:
        idx = argv.index("--")
        return list(argv[1:idx]), list(argv[idx + 1 :])
    return list(argv[1:]), []


def parse_args(argv: Sequence[str]) -> Tuple[argparse.Namespace, List[str]]:
    script_args, petsc_args = split_script_and_petsc_args(list(argv))

    parser = argparse.ArgumentParser(
        description="Monte Carlo experiments for ARKIMEX fast/slow partitions."
    )
    parser.add_argument("--raw", default=DEFAULT_RAW, help="Path to RAW file (default: %(default)s).")
    parser.add_argument("--dyr", default=DEFAULT_DYR, help="Path to DYR file (default: %(default)s).")
    parser.add_argument("--samples", type=int, default=10, help="Number of Monte Carlo runs to perform.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Maximum number of simulations to attempt (defaults to 3x samples).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    parser.add_argument("--tend", type=float, default=DEFAULT_TEND, help="Simulation end time in seconds.")
    parser.add_argument("--dt", type=float, default=DEFAULT_DT, help="Time step for dynamics integration.")
    parser.add_argument("--ton", type=float, default=DEFAULT_TON, help="Fault activation time.")
    parser.add_argument("--toff", type=float, default=DEFAULT_TOFF, help="Fault clearing time.")
    parser.add_argument(
        "--log-dir",
        default="arkimex_logs",
        help="Directory to store PETSc log files (created if missing).",
    )
    parser.add_argument(
        "--output",
        default="arkimex_partition_scatter.png",
        help="Filename for the scatter plot (PNG).",
    )
    parser.add_argument(
        "--dynamics-script",
        default="dynamics_partition_driver.py",
        help="Helper script used to run each ARKIMEX simulation (default: %(default)s).",
    )
    parser.add_argument(
        "--min-slow",
        type=float,
        default=0.0,
        help="Minimum fraction (0-1) of the differential states to treat as slow.",
    )
    parser.add_argument(
        "--max-slow",
        type=float,
        default=1.0,
        help="Maximum fraction (0-1) of the differential states to treat as slow.",
    )
    parser.add_argument(
        "--keep-logs",
        action="store_true",
        help="Retain PETSc log files after parsing. By default they are deleted.",
    )

    args = parser.parse_args(script_args)

    if args.samples <= 0:
        parser.error("--samples must be a positive integer.")
    if args.max_attempts is not None and args.max_attempts <= 0:
        parser.error("--max-attempts must be positive when provided.")
    if not 0.0 <= args.min_slow <= 1.0:
        parser.error("--min-slow must be within [0, 1].")
    if not 0.0 <= args.max_slow <= 1.0:
        parser.error("--max-slow must be within [0, 1].")
    if args.min_slow > args.max_slow:
        parser.error("--min-slow cannot exceed --max-slow.")

    return args, petsc_args


def ensure_case_exists(raw_path: Path, dyr_path: Path) -> None:
    missing = [str(path) for path in (raw_path, dyr_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing input files: " + ", ".join(missing)
        )


def compute_total_differential_states(raw_path: Path, dyr_path: Path) -> int:
    psys = load_psse(raw_filename=str(raw_path))
    add_dyr(psys, str(dyr_path))
    return psys.num_dof_dif


def build_simulation_command(
    python_executable: str,
    dynamics_script: Path,
    raw_path: Path,
    dyr_path: Path,
    tend: float,
    dt: float,
    ton: float,
    toff: float,
    slow_indices: Sequence[int],
    log_path: Path,
    extra_petsc_args: Sequence[str],
) -> List[str]:
    cmd = [
        python_executable,
        str(dynamics_script),
        "--raw",
        str(raw_path),
        "--dyr",
        str(dyr_path),
        "--tend",
        f"{tend}",
        "--dt",
        f"{dt}",
        "--ton",
        f"{ton}",
        "--toff",
        f"{toff}",
        "--petsc",
        "--arkimex",
        "--slow-diff",
        format_index_list(slow_indices),
    ]

    # Forward PETSc options and include our log view directive
    petsc_args = list(extra_petsc_args) if extra_petsc_args else []
    petsc_args += ["-log_view", f":{log_path}"]

    if petsc_args:
        cmd.append("--")
        cmd.extend(petsc_args)

    return cmd


def run_single_simulation(
    python_executable: str,
    dynamics_script: Path,
    raw_path: Path,
    dyr_path: Path,
    tend: float,
    dt: float,
    ton: float,
    toff: float,
    slow_indices: Sequence[int],
    log_path: Path,
    extra_petsc_args: Sequence[str],
) -> ExperimentResult:
    cmd = build_simulation_command(
        python_executable,
        dynamics_script,
        raw_path,
        dyr_path,
        tend,
        dt,
        ton,
        toff,
        slow_indices,
        log_path,
        extra_petsc_args,
    )

    try:
        log_path.unlink(missing_ok=True)  # remove stale log from previous runs
    except OSError:
        pass

    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    runtime = extract_runtime_from_log(log_path)
    return ExperimentResult(
        slow_count=len(slow_indices),
        total_diff=-1,  # temporarily set, caller updates after sampling
        runtime_sec=runtime,
        log_path=log_path,
        exit_code=completed.returncode,
        stderr=completed.stderr,
        stdout=completed.stdout,
    )


def summarize_results(results: Sequence[ExperimentResult]) -> None:
    successful = [res for res in results if res.exit_code == 0 and res.runtime_sec is not None]
    failed = [res for res in results if res not in successful]

    print(f"Successful runs: {len(successful)}/{len(results)}")
    if failed:
        print("Failed runs (showing exit code and log path):")
        for res in failed:
            print(f"  - slow_count={res.slow_count}, exit={res.exit_code}, log={res.log_path}")

    if successful:
        fastest = min(successful, key=lambda r: r.runtime_sec or float("inf"))
        slowest = max(successful, key=lambda r: r.runtime_sec or float("-inf"))
        print(
            f"Fastest run: {fastest.runtime_sec:.3f}s with slow fraction {fastest.slow_percent:.1f}%"
        )
        print(
            f"Slowest run: {slowest.runtime_sec:.3f}s with slow fraction {slowest.slow_percent:.1f}%"
        )


def plot_results(results: Sequence[ExperimentResult], output_path: Path) -> None:
    successful = [res for res in results if res.exit_code == 0 and res.runtime_sec is not None]
    if not successful:
        print("No successful runs to plot.")
        return

    x = [res.slow_percent for res in successful]
    y = [res.runtime_sec for res in successful]

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, c="tab:blue", alpha=0.7, edgecolors="black")
    plt.xlabel("Slow differential partition (%)")
    plt.ylabel("Total runtime (s)")
    plt.title("ARKIMEX runtime vs slow partition size")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Plot saved to {output_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(argv) if argv is not None else list(sys.argv)
    args, petsc_args = parse_args(argv)

    raw_path = Path(args.raw).resolve()
    dyr_path = Path(args.dyr).resolve()
    dynamics_script = Path(__file__).resolve().parent / args.dynamics_script
    log_dir = Path(args.log_dir).resolve()
    output_path = Path(args.output).resolve()

    ensure_case_exists(raw_path, dyr_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    total_diff = compute_total_differential_states(raw_path, dyr_path)
    if total_diff <= 0:
        raise RuntimeError("Power system has zero differential equations; nothing to partition.")

    if args.seed is not None:
        random.seed(args.seed)

    results: List[ExperimentResult] = []
    target_successes = args.samples
    max_attempts = args.max_attempts or (target_successes * 3)
    successes = 0
    attempt = 0

    while attempt < max_attempts and successes < target_successes:
        attempt += 1
        slow_indices = sample_slow_indices(total_diff, args.min_slow, args.max_slow)
        log_path = log_dir / f"petsc_log_attempt_{attempt:03d}.txt"

        # Ensure we always pass a valid string argument for --slow-diff
        sim_result = run_single_simulation(
            python_executable=sys.executable,
            dynamics_script=dynamics_script,
            raw_path=raw_path,
            dyr_path=dyr_path,
            tend=args.tend,
            dt=args.dt,
            ton=args.ton,
            toff=args.toff,
            slow_indices=slow_indices,
            log_path=log_path,
            extra_petsc_args=petsc_args,
        )
        sim_result.total_diff = total_diff
        results.append(sim_result)

        if sim_result.exit_code == 0 and sim_result.runtime_sec is not None:
            successes += 1

        print(
            f"Attempt {attempt}/{max_attempts} | successes {successes}/{target_successes}: "
            f"slow_count={len(slow_indices)} ({len(slow_indices)/total_diff*100:.1f}%), "
            f"exit={sim_result.exit_code}, runtime={sim_result.runtime_sec if sim_result.runtime_sec is not None else 'N/A'}"
        )
        if slow_indices:
            preview = ",".join(map(str, slow_indices[:10]))
            if len(slow_indices) > 10:
                preview += ",..."
            print(f"  slow indices preview: {preview}")
        else:
            print("  slow indices preview: <empty>")
        if sim_result.exit_code != 0:
            snippet = sim_result.stderr.strip() or sim_result.stdout.strip()
            if snippet:
                preview_lines = snippet.splitlines()
                print("  stderr/stdout (trimmed):")
                for line in preview_lines[:20]:
                    print("    " + line)
                if len(preview_lines) > 20:
                    print("    ...")

    summarize_results(results)
    if successes:
        plot_results(results, output_path)
    else:
        print("Skipping plot because no successful simulations were recorded.")

    if not args.keep_logs:
        for res in results:
            try:
                res.log_path.unlink(missing_ok=True)
            except OSError:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
