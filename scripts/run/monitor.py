#!/usr/bin/env python
"""
Real-time Monitoring Script for Power Grid Simulations.

This module provides a terminal-based monitoring dashboard for tracking the progress
of power grid fault simulations. It displays real-time system resource usage,
simulation progress, failure analysis, and data file statistics.

Features
--------
- Real-time CPU and memory monitoring
- Simulation progress tracking with ETA estimation
- Failure analysis grouped by fault location
- Data file size tracking
- Automatic warnings for resource issues or simulation problems

Usage
-----
Run in a separate terminal while the simulation is executing:

    $ python monitor.py --log simulation_log.json --refresh 10

Command-line Arguments
----------------------
--log : str, default='simulation_log.json'
    Path to the simulation log file (JSON format) that tracks scenario outcomes.
--refresh : int, default=10
    Screen refresh rate in seconds.

Requirements
------------
- psutil : For system resource monitoring (CPU, memory, processes)

Example
-------
To monitor a running simulation with a 5-second refresh rate::

    $ python monitor.py --log my_simulation.json --refresh 5

The monitor will display:
    - System resource usage (CPU, memory, active Python processes)
    - Simulation completion percentage and success/failure counts
    - Estimated time to completion based on current processing rate
    - Top bus locations where faults are causing simulation failures
    - Total data files generated and their cumulative size

Notes
-----
- Press Ctrl+C to gracefully stop the monitor at any time.
- The monitor expects the log file to be a JSON dictionary where keys are
  scenario IDs and values contain 'diverged' (bool) and 'fault_location' fields.
- The expected total scenarios is currently hardcoded to 200,000; modify the
  `expected_total` variable in `display_status()` if your simulation differs.

Author
------
Power Grid Simulation Team

See Also
--------
- Main simulation script that generates the log file being monitored
- Data analysis scripts for post-simulation processing
"""

import json
import os
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path
import psutil
import signal


class SimulationMonitor:
    """
    Real-time monitor for power grid simulation progress.

    This class provides a terminal-based dashboard that tracks simulation
    progress, system resources, and identifies potential issues during
    long-running power grid fault simulations.

    Parameters
    ----------
    log_file : str, default='simulation_log.json'
        Path to the JSON log file created by the simulation. The file should
        contain a dictionary mapping scenario IDs to result dictionaries with
        'diverged' and 'fault_location' keys.
    refresh_rate : int, default=10
        How often to refresh the display, in seconds.

    Attributes
    ----------
    log_file : str
        Path to the simulation log file.
    refresh_rate : int
        Display refresh interval in seconds.
    running : bool
        Flag indicating whether the monitor loop is active.
    start_time : float or None
        Timestamp when monitoring began.
    last_update : float or None
        Timestamp of the last log analysis (used for rate calculation).
    last_count : int
        Number of scenarios at last update (used for rate calculation).

    Examples
    --------
    >>> monitor = SimulationMonitor('my_sim.json', refresh_rate=5)
    >>> monitor.run()  # Starts the monitoring loop
    """

    def __init__(self, log_file="simulation_log.json", refresh_rate=10):
        self.log_file = log_file
        self.refresh_rate = refresh_rate
        self.running = True
        self.start_time = None
        self.last_update = None
        self.last_count = 0

        # Register signal handler for graceful shutdown on Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        """
        Handle interrupt signals (Ctrl+C) for graceful shutdown.

        Parameters
        ----------
        sig : int
            The signal number received.
        frame : frame object
            The current stack frame (unused).
        """
        print("\n\nMonitoring stopped.")
        self.running = False
        sys.exit(0)

    def clear_screen(self):
        """Clear the terminal screen in a cross-platform manner."""
        os.system('clear' if os.name == 'posix' else 'cls')

    def get_system_stats(self):
        """
        Collect current system resource usage statistics.

        Gathers CPU usage, memory consumption, and information about
        running Python processes to help identify resource bottlenecks.

        Returns
        -------
        dict
            Dictionary containing:
            - cpu_percent : float
                Overall CPU usage percentage.
            - memory_percent : float
                Percentage of RAM in use.
            - memory_gb : float
                RAM currently used in gigabytes.
            - memory_total_gb : float
                Total system RAM in gigabytes.
            - python_processes : int
                Number of active Python processes.
            - python_cpu : float
                Combined CPU usage of all Python processes.
            - python_memory : float
                Combined memory usage of all Python processes.
        """
        stats = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_gb': psutil.virtual_memory().used / (1024**3),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3)
        }

        # Identify and aggregate stats for all Python processes
        python_procs = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                if 'python' in proc.info['name'].lower():
                    python_procs.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process may have terminated or we lack permissions
                pass

        stats['python_processes'] = len(python_procs)
        stats['python_cpu'] = sum(p.get('cpu_percent', 0) for p in python_procs)
        stats['python_memory'] = sum(p.get('memory_percent', 0) for p in python_procs)

        return stats

    def analyze_log(self):
        """
        Parse and analyze the simulation log file.

        Reads the JSON log file and computes statistics including total
        scenarios run, success/failure counts, processing rate, and
        failure distribution by fault location.

        Returns
        -------
        dict or None
            Dictionary containing analysis results:
            - total : int
                Total number of scenarios processed.
            - succeeded : int
                Number of scenarios that converged successfully.
            - failed : int
                Number of scenarios that diverged.
            - rate : float
                Current processing rate in scenarios per minute.
            - failures_by_location : dict
                Mapping of bus IDs to failure counts.

            Returns None if the log file doesn't exist or can't be parsed.
        """
        if not os.path.exists(self.log_file):
            return None

        try:
            with open(self.log_file, 'r') as f:
                log = json.load(f)
        except (json.JSONDecodeError, IOError):
            # File may be partially written or corrupted
            return None

        # Count successes and failures
        total = len(log)
        succeeded = sum(1 for s in log.values() if not s.get('diverged', True))
        failed = sum(1 for s in log.values() if s.get('diverged', True))

        # Calculate processing rate (scenarios per minute)
        rate = 0
        if self.last_update and total > self.last_count:
            elapsed = time.time() - self.last_update
            rate = (total - self.last_count) / elapsed * 60

        # Update tracking variables for next rate calculation
        self.last_count = total
        self.last_update = time.time()

        # Aggregate failures by fault location for pattern identification
        failures_by_location = {}
        for sid, info in log.items():
            if info.get('diverged', True):
                floc = info.get('fault_location', 'unknown')
                failures_by_location[floc] = failures_by_location.get(floc, 0) + 1

        return {
            'total': total,
            'succeeded': succeeded,
            'failed': failed,
            'rate': rate,
            'failures_by_location': failures_by_location
        }

    def check_data_files(self):
        """
        Check the simulation data output directory.

        Scans the 'simulation_data' directory for generated .npz files
        and calculates their total size.

        Returns
        -------
        tuple of (int, float)
            - Number of scenario data files found
            - Total size of all files in gigabytes
        """
        data_dir = Path("simulation_data")
        if not data_dir.exists():
            return 0, 0

        files = list(data_dir.glob("scenario_*.npz"))
        total_size = sum(f.stat().st_size for f in files) / (1024**3)  # Convert to GB

        return len(files), total_size

    def display_status(self):
        """
        Render the monitoring dashboard to the terminal.

        Clears the screen and displays a formatted status panel with:
        - Current timestamp and refresh rate
        - System resource usage (CPU, memory, Python processes)
        - Simulation progress with ETA
        - Top failure locations
        - Data file statistics
        - Active warnings for potential issues
        """
        self.clear_screen()

        # === Header ===
        print("=" * 60)
        print("POWER GRID SIMULATION MONITOR")
        print("=" * 60)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Refresh rate: {self.refresh_rate}s | Press Ctrl+C to stop")
        print()

        # === System Resources Section ===
        sys_stats = self.get_system_stats()
        print("SYSTEM RESOURCES")
        print("-" * 40)
        print(f"CPU Usage: {sys_stats['cpu_percent']:.1f}%")
        print(f"Memory: {sys_stats['memory_gb']:.1f}/{sys_stats['memory_total_gb']:.1f} GB "
              f"({sys_stats['memory_percent']:.1f}%)")
        print(f"Python Processes: {sys_stats['python_processes']} "
              f"(CPU: {sys_stats['python_cpu']:.1f}%, "
              f"Mem: {sys_stats['python_memory']:.1f}%)")
        print()

        # === Simulation Progress Section ===
        log_stats = self.analyze_log()

        if log_stats:
            print("SIMULATION PROGRESS")
            print("-" * 40)

            # TODO: Make this configurable via command-line argument
            # Expected total: 400 samples * 500 buses * 1 impedance = 200,000
            expected_total = 200000
            progress = (log_stats['total'] / expected_total) * 100

            print(f"Completed: {log_stats['total']:,} / {expected_total:,} "
                  f"({progress:.2f}%)")
            print(f"Succeeded: {log_stats['succeeded']:,} "
                  f"({log_stats['succeeded']/max(1,log_stats['total'])*100:.1f}%)")
            print(f"Failed: {log_stats['failed']:,} "
                  f"({log_stats['failed']/max(1,log_stats['total'])*100:.1f}%)")

            # Show processing rate and ETA if we have rate data
            if log_stats['rate'] > 0:
                print(f"Rate: {log_stats['rate']:.1f} scenarios/min")
                remaining = expected_total - log_stats['total']
                eta = remaining / (log_stats['rate'] / 60)  # Convert rate to per-second
                print(f"ETA: {timedelta(seconds=int(eta))}")

            # Display top 5 bus locations causing failures
            if log_stats['failures_by_location']:
                print("\nTop Failure Locations:")
                sorted_failures = sorted(
                    log_stats['failures_by_location'].items(),
                    key=lambda x: x[1], reverse=True
                )[:5]
                for floc, count in sorted_failures:
                    if floc != 'unknown':
                        print(f"  Bus {floc}: {count} failures")
        else:
            print("SIMULATION PROGRESS")
            print("-" * 40)
            print("No log file found or empty")

        print()

        # === Data Files Section ===
        num_files, total_size = self.check_data_files()
        print("DATA FILES")
        print("-" * 40)
        print(f"Files created: {num_files:,}")
        print(f"Total size: {total_size:.2f} GB")

        # === Warnings Section ===
        print()
        print("WARNINGS")
        print("-" * 40)

        warnings = []

        # Check for high memory usage
        if sys_stats['memory_percent'] > 80:
            warnings.append(f"⚠️  High memory usage: {sys_stats['memory_percent']:.1f}%")

        # Check for potential stall (low CPU but slow progress)
        if sys_stats['cpu_percent'] < 10 and log_stats and log_stats['rate'] < 1:
            warnings.append("⚠️  Low CPU usage - simulation may be stalled")

        # Check for high failure rate (>10% of total)
        if log_stats and log_stats['failed'] > log_stats['succeeded'] * 0.1:
            warnings.append(f"⚠️  High failure rate: "
                            f"{log_stats['failed']/max(1,log_stats['total'])*100:.1f}%")

        if warnings:
            for w in warnings:
                print(w)
        else:
            print("✓ No warnings")

    def run(self):
        """
        Start the main monitoring loop.

        Continuously refreshes the display at the configured interval
        until interrupted by Ctrl+C or an error occurs.
        """
        print("Starting simulation monitor...")
        print(f"Monitoring {self.log_file}")
        print(f"Refresh rate: {self.refresh_rate} seconds")
        print("\nPress Ctrl+C to stop monitoring")
        time.sleep(2)  # Brief pause before starting to allow user to read

        while self.running:
            try:
                self.display_status()
                time.sleep(self.refresh_rate)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(self.refresh_rate)


def main():
    """
    Entry point for the simulation monitor.

    Parses command-line arguments, validates dependencies, and starts
    the monitoring loop.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Monitor power grid simulation progress in real-time"
    )
    parser.add_argument(
        '--log', default='simulation_log.json',
        help='Path to simulation log file (default: simulation_log.json)'
    )
    parser.add_argument(
        '--refresh', type=int, default=10,
        help='Refresh rate in seconds (default: 10)'
    )

    args = parser.parse_args()

    # Verify psutil is available before starting
    try:
        import psutil
    except ImportError:
        print("Error: psutil is required for system monitoring")
        print("Install with: pip install psutil")
        sys.exit(1)

    # Create and run the monitor
    monitor = SimulationMonitor(args.log, args.refresh)
    monitor.run()


if __name__ == "__main__":
    main()
