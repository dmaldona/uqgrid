#!/usr/bin/env python
"""
Recovery script for failed scenarios in generate_scenarios.py
Allows retrying failed scenarios and resuming from crashes.
"""

import json
import sys

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
sys.path.insert(0, '.') 


# Import the fixed version functions
from scripts.run.generate_scenarios import run_simulation_driver_batched

def analyze_simulation_log(log_file="simulation_log.json"):
    """Analyze the simulation log to find failed scenarios."""
    
    if not os.path.exists(log_file):
        print(f"Error: {log_file} not found")
        return None
    
    with open(log_file, 'r') as f:
        log = json.load(f)
    
    total = len(log)
    succeeded = sum(1 for s in log.values() if not s.get('diverged', True))
    failed = sum(1 for s in log.values() if s.get('diverged', True))
    missing_files = sum(1 for s in log.values() if s.get('file') is None)
    
    print(f"\n=== Simulation Status ===")
    print(f"Total scenarios: {total}")
    print(f"Succeeded: {succeeded} ({succeeded/total*100:.1f}%)")
    print(f"Failed: {failed} ({failed/total*100:.1f}%)")
    print(f"Missing files: {missing_files}")
    
    # Find failed scenarios
    failed_scenarios = {
        sid: info for sid, info in log.items() 
        if info.get('diverged', True) or info.get('file') is None
    }
    
    if failed_scenarios:
        print(f"\nFailed scenario IDs: {len(failed_scenarios)}")
        # Group by fault location
        by_fault = {}
        for sid, info in failed_scenarios.items():
            floc = info.get('fault_location', 'unknown')
            if floc not in by_fault:
                by_fault[floc] = []
            by_fault[floc].append(sid)
        
        print("\nFailed scenarios by fault location:")
        for floc in sorted(by_fault.keys()):
            if floc != 'unknown':
                print(f"  Fault location {floc}: {len(by_fault[floc])} failures")
    
    return log, failed_scenarios

def create_retry_script(failed_scenarios, metadata_file="scenario_metadata.json"):
    """Create a script to retry only the failed scenarios."""
    
    if not failed_scenarios:
        print("No failed scenarios to retry")
        return
    
    # Load original metadata
    with open(metadata_file, 'r') as f:
        all_metadata = json.load(f)
    
    # Create metadata for retry
    retry_metadata = {sid: all_metadata[sid] for sid in failed_scenarios.keys()}
    
    # Save retry metadata
    retry_metadata_file = "retry_metadata.json"
    with open(retry_metadata_file, 'w') as f:
        json.dump(retry_metadata, f, indent=4)
    
    print(f"\nCreated {retry_metadata_file} with {len(retry_metadata)} scenarios to retry")
    
    # Create retry script
    retry_script = "#!/usr/bin/env python"

def retry_failed_scenarios():
    # Load configuration based on model
    # TODO fix this to be dynamic
    PowerGridModel = "ACTIVSg500"  
    
    if PowerGridModel == "ACTIVSg500":
        raw = "data/ACTIVSg500.raw"
        dyr = "data/ACTIVSg500.dyr"
    # Add other models as needed
    
    # Load retry metadata
    with open("retry_metadata.json", 'r') as f:
        retry_metadata = json.load(f)
    
    print(f"Retrying {len(retry_metadata)} failed scenarios...")
    
    # Run with more conservative settings
    run_simulation_driver_batched(
        raw, dyr, retry_metadata,
        noise_type="normal",
        noise_var=0.25,
        balance_generation=True,
        n_jobs=1,  # Single job to avoid MPI issues
        batch_size=3,  # Smaller batches
        checkpoint_interval=10
    )

if __name__ == "__main__":
    retry_failed_scenarios()
    
    with open("retry_failed.py", 'w') as f:
        f.write(retry_script)
    
    os.chmod("retry_failed.py", 0o755)
    print(f"Created retry_failed.py - run with: python retry_failed.py")


def check_data_integrity():
    """Check if simulation data files match the log."""
    
    data_dir = Path("simulation_data")
    if not data_dir.exists():
        print("Warning: simulation_data directory not found")
        return
    
    # Get all .npz files
    data_files = list(data_dir.glob("scenario_*.npz"))
    print(f"\nFound {len(data_files)} data files in {data_dir}")
    
    # Check file sizes
    small_files = []
    for f in data_files:
        size = f.stat().st_size
        if size < 1000:  # Less than 1KB is suspicious
            small_files.append(f)
    
    if small_files:
        print(f"Warning: {len(small_files)} files are suspiciously small (<1KB)")


def main():
    print("Power Grid Simulation Recovery Tool")
    print("=" * 40)
    
    # Analyze current status
    log, failed = analyze_simulation_log()
    
    if log is None:
        return
    
    # Check data integrity
    check_data_integrity()
    
    # Offer recovery options
    if failed:
        print("\n=== Recovery Options ===")
        print("1. Create retry script for failed scenarios")
        print("2. Export failed scenario list")
        print("3. Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            create_retry_script(failed)
        elif choice == '2':
            with open("failed_scenarios.json", 'w') as f:
                json.dump(failed, f, indent=4)
            print(f"Exported failed scenarios to failed_scenarios.json")
    else:
        print("\nAll scenarios completed successfully!")


if __name__ == "__main__":
    main()
