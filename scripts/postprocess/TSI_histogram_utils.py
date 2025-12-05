"""
Utility functions for visualizing TSI (Transient Stability Index) data histograms.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def load_tsi_data(filepath: str = "tsi_probml_fullinputs.npz") -> dict:
    """
    Load TSI dataset from npz file.
    
    Parameters
    ----------
    filepath : str
        Path to the npz file containing TSI data.
    
    Returns
    -------
    dict
        Dictionary containing the loaded data arrays.
    """
    data = np.load(filepath, allow_pickle=True)
    return {key: data[key] for key in data.files}


def plot_histogram_all_samples(
    filepath: str = "tsi_probml_fullinputs.npz",
    bins: int = 50,
    figsize: tuple = (10, 6),
    title: str = "Histogram of All TSI Values",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a histogram of all TSI values in the dataset.
    
    Parameters
    ----------
    filepath : str
        Path to the npz file containing TSI data.
    bins : int
        Number of histogram bins.
    figsize : tuple
        Figure size (width, height).
    title : str
        Plot title.
    save_path : str, optional
        If provided, save the figure to this path.
    
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    data = load_tsi_data(filepath)
    Y = data["Y"]  # Shape: (N, F, Z)
    
    # Flatten all TSI values
    all_tsi_values = Y.flatten()
    
    # Remove NaN values if present
    all_tsi_values = all_tsi_values[~np.isnan(all_tsi_values)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(all_tsi_values, bins=bins, range=(-100, 100), edgecolor="black", alpha=0.7)
    ax.set_xlim(-100, 100)
    ax.set_xlabel("TSI Value")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add statistics annotation
    valid_in_range = all_tsi_values[(all_tsi_values >= -100) & (all_tsi_values <= 100)]
    stats_text = (
        f"Total samples: {len(all_tsi_values):,}\n"
        f"In range [-100, 100]: {len(valid_in_range):,}\n"
        f"Mean: {np.mean(all_tsi_values):.2f}\n"
        f"Std: {np.std(all_tsi_values):.2f}"
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )
    
    plt.tight_layout()
    
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
    Plot a histogram of TSI values for a given scenario (first dimension of Y).
    
    Parameters
    ----------
    scenario_idx : int
        Index of the scenario (sample) to plot. Corresponds to the first 
        dimension of Y array with shape (N, F, Z).
    filepath : str
        Path to the npz file containing TSI data.
    bins : int
        Number of histogram bins.
    figsize : tuple
        Figure size (width, height).
    title : str, optional
        Plot title. If None, auto-generates based on scenario index.
    save_path : str, optional
        If provided, save the figure to this path.
    
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    
    Raises
    ------
    IndexError
        If scenario_idx is out of bounds.
    """
    data = load_tsi_data(filepath)
    Y = data["Y"]  # Shape: (N, F, Z)
    
    N, F, Z = Y.shape
    
    if scenario_idx < 0 or scenario_idx >= N:
        raise IndexError(
            f"scenario_idx {scenario_idx} is out of bounds. "
            f"Valid range: [0, {N - 1}]"
        )
    
    # Get TSI values for the specified scenario
    scenario_tsi_values = Y[scenario_idx, :, :].flatten()
    
    # Remove NaN values if present
    scenario_tsi_values = scenario_tsi_values[~np.isnan(scenario_tsi_values)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if title is None:
        title = f"Histogram of TSI Values for Scenario {scenario_idx}"
    
    ax.hist(
        scenario_tsi_values, bins=bins, range=(-100, 100),
        edgecolor="black", alpha=0.7, color="steelblue"
    )
    ax.set_xlim(-100, 100)
    ax.set_xlabel("TSI Value")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add statistics annotation
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
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    
    return fig


if __name__ == "__main__":
    ''' 
    Example usage
    from tsi_histogram_utils import plot_histogram_all_samples, plot_histogram_single_scenario

    # Plot histogram of all TSI values
    fig1 = plot_histogram_all_samples("tsi_probml_fullinputs.npz")

    # Plot histogram for scenario 5
    fig2 = plot_histogram_single_scenario(5, "tsi_probml_fullinputs.npz")
    '''
    import sys
    
    filepath = "tsi_probml_fullinputs.npz"
    
    print("Generating histogram of all TSI values...")
    fig1 = plot_histogram_all_samples(
        filepath=filepath,
        save_path="histogram_all_tsi.png"
    )
    plt.show()
    
    print("\nGenerating histogram for scenario 0...")
    fig2 = plot_histogram_single_scenario(
        scenario_idx=0,
        filepath=filepath,
        save_path="histogram_scenario_0.png"
    )
    plt.show()
