#!/usr/bin/env python
"""
Real-time monitoring script for power grid simulations.
Run this in a separate terminal to monitor progress.
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
    def __init__(self, log_file="simulation_log.json", refresh_rate=10):
        self.log_file = log_file
        self.refresh_rate = refresh_rate
        self.running = True
        self.start_time = None
        self.last_update = None
        self.last_count = 0
        
        # Handle Ctrl+C gracefully
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        print("\n\nMonitoring stopped.")
        self.running = False
        sys.exit(0)
    
    def clear_screen(self):
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def get_system_stats(self):
        """Get current system resource usage."""
        stats = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_gb': psutil.virtual_memory().used / (1024**3),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3)
        }
        
        # Check for Python processes
        python_procs = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                if 'python' in proc.info['name'].lower():
                    python_procs.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        stats['python_processes'] = len(python_procs)
        stats['python_cpu'] = sum(p.get('cpu_percent', 0) for p in python_procs)
        stats['python_memory'] = sum(p.get('memory_percent', 0) for p in python_procs)
        
        return stats
    
    def analyze_log(self):
        """Analyze the current simulation log."""
        if not os.path.exists(self.log_file):
            return None
        
        try:
            with open(self.log_file, 'r') as f:
                log = json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
        
        total = len(log)
        succeeded = sum(1 for s in log.values() if not s.get('diverged', True))
        failed = sum(1 for s in log.values() if s.get('diverged', True))
        
        # Calculate rate
        rate = 0
        if self.last_update and total > self.last_count:
            elapsed = time.time() - self.last_update
            rate = (total - self.last_count) / elapsed * 60  # per minute
        
        self.last_count = total
        self.last_update = time.time()
        
        # Group failures by fault location
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
        """Check simulation data files."""
        data_dir = Path("simulation_data")
        if not data_dir.exists():
            return 0, 0
        
        files = list(data_dir.glob("scenario_*.npz"))
        total_size = sum(f.stat().st_size for f in files) / (1024**3)  # GB
        
        return len(files), total_size
    
    def display_status(self):
        """Display current status."""
        self.clear_screen()
        
        print("=" * 60)
        print("POWER GRID SIMULATION MONITOR")
        print("=" * 60)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Refresh rate: {self.refresh_rate}s | Press Ctrl+C to stop")
        print()
        
        # System stats
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
        
        # Simulation progress
        log_stats = self.analyze_log()
        
        if log_stats:
            print("SIMULATION PROGRESS")
            print("-" * 40)
            
            # Expected total
            # TODO fix this hardcoded version
            expected_total = 200000  # 400 samples * 500 buses * 1 impedance
            progress = (log_stats['total'] / expected_total) * 100
            
            print(f"Completed: {log_stats['total']:,} / {expected_total:,} "
                  f"({progress:.2f}%)")
            print(f"Succeeded: {log_stats['succeeded']:,} "
                  f"({log_stats['succeeded']/max(1,log_stats['total'])*100:.1f}%)")
            print(f"Failed: {log_stats['failed']:,} "
                  f"({log_stats['failed']/max(1,log_stats['total'])*100:.1f}%)")
            
            if log_stats['rate'] > 0:
                print(f"Rate: {log_stats['rate']:.1f} scenarios/min")
                remaining = expected_total - log_stats['total']
                eta = remaining / (log_stats['rate'] / 60)  # seconds
                print(f"ETA: {timedelta(seconds=int(eta))}")
            
            # Show top failure locations
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
        
        # Data files
        num_files, total_size = self.check_data_files()
        print("DATA FILES")
        print("-" * 40)
        print(f"Files created: {num_files:,}")
        print(f"Total size: {total_size:.2f} GB")
        
        # Warnings
        print()
        print("WARNINGS")
        print("-" * 40)
        
        warnings = []
        if sys_stats['memory_percent'] > 80:
            warnings.append(f"⚠️  High memory usage: {sys_stats['memory_percent']:.1f}%")
        if sys_stats['cpu_percent'] < 10 and log_stats and log_stats['rate'] < 1:
            warnings.append("⚠️  Low CPU usage - simulation may be stalled")
        if log_stats and log_stats['failed'] > log_stats['succeeded'] * 0.1:
            warnings.append(f"⚠️  High failure rate: "
                          f"{log_stats['failed']/max(1,log_stats['total'])*100:.1f}%")
        
        if warnings:
            for w in warnings:
                print(w)
        else:
            print("✓ No warnings")
    
    def run(self):
        """Main monitoring loop."""
        print("Starting simulation monitor...")
        print(f"Monitoring {self.log_file}")
        print(f"Refresh rate: {self.refresh_rate} seconds")
        print("\nPress Ctrl+C to stop monitoring")
        time.sleep(2)
        
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
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Monitor power grid simulation progress"
    )
    parser.add_argument(
        '--log', default='simulation_log.json',
        help='Path to simulation log file'
    )
    parser.add_argument(
        '--refresh', type=int, default=10,
        help='Refresh rate in seconds'
    )
    
    args = parser.parse_args()
    
    # Check if psutil is installed
    try:
        import psutil
    except ImportError:
        print("Error: psutil is required for system monitoring")
        print("Install with: pip install psutil")
        sys.exit(1)
    
    monitor = SimulationMonitor(args.log, args.refresh)
    monitor.run()


if __name__ == "__main__":
    main()
