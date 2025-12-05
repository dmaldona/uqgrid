#!/usr/bin/env python
"""
Utility Functions for Visualizing TSI (Transient Stability Index) Histograms.

This module provides convenient functions for loading and visualizing TSI data
from power grid simulation datasets. It creates publication-quality histograms
showing the distribution of TSI values across scenarios and fault conditions.

The TSI (Transient Stability Index) measures power system stability following
disturbances:
- TSI > 0 : Stable system response
- TSI < 0 : Unstable system response
- TSI = 100 : Maximum stability (no rotor angle deviation)
- TSI = -100 : Severe instability

Features
--------
- Load TSI datasets from compressed NumPy archives (.npz)
- Plot aggregate histograms across all scenarios and fault conditions
- Plot per-scenario histograms for detailed analysis
- Automatic statistics annotation (mean, std, sample counts)
- Optional figure saving with customizable output paths

Data Format
-----------
Expected input file structure (from export_probml_dataset):
    - Y : ndarray (N, F, Z)
        TSI values at the last time step where:
        - N = number of operating condition samples
        - F = number of fault locations
        - Z = number of fault impedance values

Usage
-----
Command-line execution generates example histograms::

    $ python TSI_histogram_utils.py

Programmatic usage::

    from TSI_histogram_utils import plot_histogram_all_samples, plot_histogram_single_scenario

    # Plot histogram of all TSI values in the dataset
    fig1 = plot_histogram_all_samples(
        "tsi_probml_fullinputs.npz",
        save_path="all_tsi_histogram.png"
    )

    # Plot histogram for a specific operating condition (scenario)
    fig2 = plot_histogram_single_scenario(
        scenario_idx=5,
        filepath="tsi_probml_fullinputs.npz",
        save_path="scenario_5_histogram.png"
    )

Output Files
------------
When save_path is specified:
- histogram_all_tsi.png : Distribution of all TSI values
- histogram_scenario_N.png : Distribution for scenario N

Dependencies
------------
- numpy : Data loading and numerical operations
- matplotlib : Histogram plotting and figure generation

See Also
--------
- TSI_analysis.py : Generates the TSI datasets visualized by this module
- export_probml_dataset() : Creates the .npz files consumed by these functions

Examples
--------
Analyze stability distribution across a simulation campaign::

    >>> fig = plot_histogram_all_samples("my_simulation.npz")
    >>> # Check if most scenarios are stable (TSI > 0)
    >>> plt.show()

Compare stability for different operating conditions::

    >>> for scenario_idx in [0, 10, 20]:
    ...     fig = plot_histogram_single_scenario(scenario_idx, "my_simulation.npz")
    ...     plt.show()

Author
------
Power Grid Simulation Team
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


# =============================================================================
# Data Loading
# =============================================================================

def load_tsi_data(filepath: str = "tsi_probml_fullinputs.npz") -> dict:
    """
    Load TSI dataset from a compressed NumPy archive (.npz) file.

    Parameters
    ----------
    filepath : str, default='tsi_probml_fullinputs.npz'
        Path to the .npz file containing TSI data. Expected to contain
        at minimum a 'Y' array with TSI values.

    Returns
    -------
    dict
        Dictionary containing all arrays from the .npz file. Keys typically
        include:
        - 'Y' : TSI values array (N, F, Z)
        - 'X' or 'X_flat' : Input features (if saved)
        - 'sample_idx' : Sample indices
        - 'fault_locations' : Fault location bus numbers
        - 'fault_impedances' : Fault impedance values
        - 'meta' : Metadata dictionary

    Notes
    -----
    Uses allow_pickle=True to load metadata dictionaries stored as
    object arrays. The returned dictionary provides direct access to
    all arrays without keeping the file handle open.

    Examples
    --------
    >>> data = load_tsi_data("my_dataset.npz")
    >>> Y = data["Y"]
    >>> print(f"Dataset shape: {Y.shape}")
    Dataset shape: (1000, 50, 3)
    """
    # Load with allow_pickle for metadata object arrays
    data = np.load(filepath, allow_pickle=True)

    # Convert to regular dict to close file handle and enable easy access
    return {key: data[key] for key in data.files}


# =============================================================================
# Histogram Plotting Functions
# =============================================================================

def plot_histogram_all_samples(
    filepath: str = "tsi_probml_fullinputs.npz",
    bins: int = 50,
    figsize: tuple = (10, 6),
    title: str = "Histogram of All TSI Values",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a histogram of all TSI values across the entire dataset.

    Creates a histogram showing the aggregate distribution of TSI values
    from all scenarios, fault locations, and fault impedances combined.
    This provides an overview of the overall stability characteristics
    of the simulation campaign.

    Parameters
    ----------
    filepath : str, default='tsi_probml_fullinputs.npz'
        Path to the .npz file containing TSI data.
    bins : int, default=50
        Number of histogram bins. More bins provide finer resolution
        but may be noisy for small datasets.
    figsize : tuple, default=(10, 6)
        Figure size as (width, height) in inches.
    title : str, default='Histogram of All TSI Values'
        Plot title displayed at the top of the figure.
    save_path : str, optional
        If provided, saves the figure to this path. Supports any format
        recognized by matplotlib (png, pdf, svg, etc.).

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object, which can be further customized
        or displayed with plt.show().

    Notes
    -----
    - The histogram is restricted to the range [-100, 100] since TSI
      values outside this range indicate numerical issues.
    - NaN values (from incomplete fault grids) are automatically removed.
    - Statistics box shows total samples, valid samples in range,
      mean, and standard deviation.

    Examples
    --------
    >>> fig = plot_histogram_all_samples("tsi_data.npz")
    >>> plt.show()

    >>> # Save high-resolution figure for publication
    >>> fig = plot_histogram_all_samples(
    ...     "tsi_data.npz",
    ...     bins=100,
    ...     figsize=(12, 8),
    ...     save_path="figures/tsi_distribution.pdf"
    ... )
    """
    # Load dataset
    data = load_tsi_data(filepath)
    Y = data["Y"]  # Shape: (N, F, Z) - samples × fault_locations × fault_impedances

    # Flatten all TSI values into a 1D array for histogram
    all_tsi_values = Y.flatten()

    # Remove NaN values (occur when require_complete_grid=False)
    all_tsi_values = all_tsi_values[~np.isnan(all_tsi_values)]

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram with fixed range for consistency across datasets
    ax.hist(
        all_tsi_values,
        bins=bins,
        range=(-100, 100),  # TSI theoretical range
        edgecolor="black",
        alpha=0.7
    )

    # Configure axes
    ax.set_xlim(-100, 100)
    ax.set_xlabel("TSI Value")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add statistics annotation box
    valid_in_range = all_tsi_values[
        (all_tsi_values >= -100) & (all_tsi_values <= 100)
    ]
    stats_text = (
        f"Total samples: {len(all_tsi_values):,}\n"
        f"In range [-100, 100]: {len(valid_in_range):,}\n"
        f"Mean: {np.mean(all_tsi_values):.2f}\n"
        f"Std: {np.std(all_tsi_values):.2f}"
    )
    ax.text(
        0.02, 0.98,  # Position in axes coordinates (top-left)
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig


def plot_histogram_single_scenario(
    scenario_idx: int,
    filepath: str = "tsi_probml_fullinputs.npz",
    bins: int = 50,
    figsize: tuple = (10, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a histogram of TSI values for a single scenario (operating condition).

    Creates a histogram showing the distribution of TSI values across all
    fault locations and impedances for one specific operating condition
    (sample_idx). This reveals how stability varies with fault parameters
    for a fixed system state.

    Parameters
    ----------
    scenario_idx : int
        Index of the scenario (sample) to plot. Corresponds to the first
        dimension of the Y array with shape (N, F, Z). This typically
        represents a unique combination of generator dispatch and load
        conditions.
    filepath : str, default='tsi_probml_fullinputs.npz'
        Path to the .npz file containing TSI data.
    bins : int, default=50
        Number of histogram bins.
    figsize : tuple, default=(10, 6)
        Figure size as (width, height) in inches.
    title : str, optional
        Plot title. If None, auto-generates a title based on scenario_idx:
        "Histogram of TSI Values for Scenario {scenario_idx}"
    save_path : str, optional
        If provided, saves the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.

    Raises
    ------
    IndexError
        If scenario_idx is out of bounds for the dataset. The error message
        includes the valid range of indices.

    Notes
    -----
    - Each scenario contains F × Z TSI values (one per fault condition).
    - The histogram shows how stability varies across the fault parameter
      space for a single operating point.
    - Useful for identifying operating conditions that are marginally
      stable (TSI values clustered near zero).

    Examples
    --------
    >>> # Plot for the first scenario
    >>> fig = plot_histogram_single_scenario(0, "tsi_data.npz")
    >>> plt.show()

    >>> # Compare two different operating conditions
    >>> fig1 = plot_histogram_single_scenario(
    ...     scenario_idx=10,
    ...     title="Low Load Condition"
    ... )
    >>> fig2 = plot_histogram_single_scenario(
    ...     scenario_idx=50,
    ...     title="High Load Condition"
    ... )

    >>> # Save with custom filename
    >>> fig = plot_histogram_single_scenario(
    ...     scenario_idx=42,
    ...     save_path="scenario_42_analysis.png"
    ... )
    """
    # Load dataset
    data = load_tsi_data(filepath)
    Y = data["Y"]  # Shape: (N, F, Z)

    # Extract dimensions
    N, F, Z = Y.shape

    # Validate scenario index
    if scenario_idx < 0 or scenario_idx >= N:
        raise IndexError(
            f"scenario_idx {scenario_idx} is out of bounds. "
            f"Valid range: [0, {N - 1}]"
        )

    # Extract TSI values for the specified scenario (all fault conditions)
    scenario_tsi_values = Y[scenario_idx, :, :].flatten()

    # Remove NaN values if present
    scenario_tsi_values = scenario_tsi_values[~np.isnan(scenario_tsi_values)]

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Generate default title if not provided
    if title is None:
        title = f"Histogram of TSI Values for Scenario {scenario_idx}"

    # Plot histogram with distinctive color for single-scenario plots
    ax.hist(
        scenario_tsi_values,
        bins=bins,
        range=(-100, 100),
        edgecolor="black",
        alpha=0.7,
        color="steelblue"  # Different color to distinguish from aggregate plots
    )

    # Configure axes
    ax.set_xlim(-100, 100)
    ax.set_xlabel("TSI Value")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add statistics annotation box with scenario-specific info
    valid_in_range = scenario_tsi_values[
        (scenario_tsi_values >= -100) & (scenario_tsi_values <= 100)
    ]
    stats_text = (
        f"Scenario index: {scenario_idx}\n"
        f"Grid size (F×Z): {F}×{Z} = {F * Z}\n"
        f"Valid samples: {len(scenario_tsi_values)}\n"
        f"In range [-100, 100]: {len(valid_in_range)}\n"
        f"Mean: {np.mean(scenario_tsi_values):.2f}\n"
        f"Std: {np.std(scenario_tsi_values):.2f}"
    )
    ax.text(
        0.02, 0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    """
    Example usage demonstrating histogram generation capabilities.

    When run as a script, generates two example histograms:
    1. Aggregate histogram of all TSI values in the dataset
    2. Per-scenario histogram for scenario index 0

    For programmatic usage, import the functions directly::

        from TSI_histogram_utils import (
            load_tsi_data,
            plot_histogram_all_samples,
            plot_histogram_single_scenario
        )

        # Load data for custom analysis
        data = load_tsi_data("tsi_probml_fullinputs.npz")
        Y = data["Y"]

        # Plot histogram of all TSI values
        fig1 = plot_histogram_all_samples("tsi_probml_fullinputs.npz")

        # Plot histogram for a specific scenario
        fig2 = plot_histogram_single_scenario(5, "tsi_probml_fullinputs.npz")
    """
    import sys

    # Default dataset path
    filepath = "tsi_probml_fullinputs.npz"

    # Generate aggregate histogram
    print("Generating histogram of all TSI values...")
    fig1 = plot_histogram_all_samples(
        filepath=filepath,
        save_path="histogram_all_tsi.png"
    )
    plt.show()

    # Generate per-scenario histogram for scenario 0
    print("\nGenerating histogram for scenario 0...")
    fig2 = plot_histogram_single_scenario(
        scenario_idx=0,
        filepath=filepath,
        save_path="histogram_scenario_0.png"
    )
    plt.show()
