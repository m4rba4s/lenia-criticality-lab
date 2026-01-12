"""
Analysis and Visualization

Publication-quality figures and statistical analysis for criticality detection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter
from typing import Optional, Tuple, List
from pathlib import Path


# Style configuration for publication
def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.figsize': (6, 5),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


# Color schemes
PHASE_COLORS = {
    'dead': '#1a1a2e',
    'explosive': '#e63946',
    'stable': '#2a9d8f',
    'oscillating': '#457b9d',
    'chaotic': '#f4a261',
}

LYAPUNOV_CMAP = plt.cm.RdBu_r  # Red = chaotic, Blue = ordered


def plot_phase_diagram(df: pd.DataFrame,
                       value_col: str = 'classification',
                       ax: Optional[plt.Axes] = None,
                       title: str = "Phase Diagram") -> plt.Axes:
    """
    Plot phase diagram from experiment results.

    Args:
        df: DataFrame with mu, sigma, and classification columns
        value_col: Column to plot
        ax: Matplotlib axes (creates new if None)
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    mu_vals = np.sort(df['mu'].unique())
    sigma_vals = np.sort(df['sigma'].unique())

    # Create grid
    grid = np.full((len(sigma_vals), len(mu_vals)), np.nan)

    for _, row in df.iterrows():
        mi = np.searchsorted(mu_vals, row['mu'])
        si = np.searchsorted(sigma_vals, row['sigma'])
        if mi < len(mu_vals) and si < len(sigma_vals):
            if value_col == 'classification':
                # Map to numeric
                phase_map = {'dead': 0, 'explosive': 1, 'stable': 2, 'oscillating': 3, 'chaotic': 4}
                grid[si, mi] = phase_map.get(row[value_col], np.nan)
            else:
                grid[si, mi] = row[value_col]

    if value_col == 'classification':
        # Discrete colormap
        cmap = mcolors.ListedColormap([PHASE_COLORS[k] for k in ['dead', 'explosive', 'stable', 'oscillating', 'chaotic']])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        im = ax.imshow(grid, origin='lower', aspect='auto',
                       extent=[mu_vals[0], mu_vals[-1], sigma_vals[0], sigma_vals[-1]],
                       cmap=cmap, norm=norm, interpolation='nearest')

        # Legend
        patches = [Patch(facecolor=PHASE_COLORS[k], label=k.capitalize())
                   for k in ['dead', 'explosive', 'stable', 'oscillating', 'chaotic']]
        ax.legend(handles=patches, loc='upper right', fontsize=8)
    else:
        im = ax.imshow(grid, origin='lower', aspect='auto',
                       extent=[mu_vals[0], mu_vals[-1], sigma_vals[0], sigma_vals[-1]],
                       cmap='viridis', interpolation='bilinear')
        plt.colorbar(im, ax=ax, label=value_col)

    ax.set_xlabel('μ (growth center)')
    ax.set_ylabel('σ (growth width)')
    ax.set_title(title)

    return ax


def plot_lyapunov_diagram(df: pd.DataFrame,
                          ax: Optional[plt.Axes] = None,
                          vmin: float = -0.1,
                          vmax: float = 0.1) -> plt.Axes:
    """
    Plot Lyapunov exponent phase diagram.

    Critical line is where λ ≈ 0 (white in RdBu colormap).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    df_valid = df[df['lyapunov'].notna()].copy()

    mu_vals = np.sort(df_valid['mu'].unique())
    sigma_vals = np.sort(df_valid['sigma'].unique())

    grid = np.full((len(sigma_vals), len(mu_vals)), np.nan)

    for _, row in df_valid.iterrows():
        mi = np.searchsorted(mu_vals, row['mu'])
        si = np.searchsorted(sigma_vals, row['sigma'])
        if mi < len(mu_vals) and si < len(sigma_vals):
            grid[si, mi] = row['lyapunov']

    # Smooth for visualization
    grid_smooth = gaussian_filter(np.nan_to_num(grid, nan=0), sigma=0.5)
    grid_smooth[np.isnan(grid)] = np.nan

    im = ax.imshow(grid_smooth, origin='lower', aspect='auto',
                   extent=[mu_vals[0], mu_vals[-1], sigma_vals[0], sigma_vals[-1]],
                   cmap=LYAPUNOV_CMAP, vmin=vmin, vmax=vmax,
                   interpolation='bilinear')

    # Add contour at λ = 0 (critical line)
    if not np.all(np.isnan(grid_smooth)):
        try:
            contour = ax.contour(grid_smooth, levels=[0], colors='black',
                                 linewidths=2, linestyles='--',
                                 extent=[mu_vals[0], mu_vals[-1], sigma_vals[0], sigma_vals[-1]])
            ax.clabel(contour, fmt='λ=0', fontsize=9)
        except Exception:
            pass

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Lyapunov exponent (λ)')

    ax.set_xlabel('μ (growth center)')
    ax.set_ylabel('σ (growth width)')
    ax.set_title('Lyapunov Exponent Phase Diagram\n(λ<0: ordered, λ=0: critical, λ>0: chaotic)')

    return ax


def plot_criticality_signatures(df: pd.DataFrame,
                                figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Multi-panel figure showing criticality signatures.

    Panels:
    1. Phase diagram
    2. Lyapunov exponents
    3. Correlation length
    4. Entropy/Complexity
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Phase diagram
    plot_phase_diagram(df, ax=axes[0, 0], title="(a) Phase Classification")

    # 2. Lyapunov
    plot_lyapunov_diagram(df, ax=axes[0, 1])
    axes[0, 1].set_title("(b) Lyapunov Exponent")

    # 3. Correlation length
    df_valid = df[df['correlation_length'].notna() & np.isfinite(df['correlation_length'])].copy()
    if len(df_valid) > 0:
        mu_vals = np.sort(df_valid['mu'].unique())
        sigma_vals = np.sort(df_valid['sigma'].unique())
        grid = np.full((len(sigma_vals), len(mu_vals)), np.nan)

        for _, row in df_valid.iterrows():
            mi = np.searchsorted(mu_vals, row['mu'])
            si = np.searchsorted(sigma_vals, row['sigma'])
            if mi < len(mu_vals) and si < len(sigma_vals):
                # Clip to reasonable range for visualization
                grid[si, mi] = min(row['correlation_length'], 100)

        im = axes[1, 0].imshow(grid, origin='lower', aspect='auto',
                               extent=[mu_vals[0], mu_vals[-1], sigma_vals[0], sigma_vals[-1]],
                               cmap='YlOrRd', interpolation='bilinear')
        plt.colorbar(im, ax=axes[1, 0], label='ξ')
        axes[1, 0].set_xlabel('μ')
        axes[1, 0].set_ylabel('σ')
    axes[1, 0].set_title("(c) Correlation Length ξ")

    # 4. Entropy
    df_ent = df[df['entropy'].notna()].copy()
    if len(df_ent) > 0:
        mu_vals = np.sort(df_ent['mu'].unique())
        sigma_vals = np.sort(df_ent['sigma'].unique())
        grid = np.full((len(sigma_vals), len(mu_vals)), np.nan)

        for _, row in df_ent.iterrows():
            mi = np.searchsorted(mu_vals, row['mu'])
            si = np.searchsorted(sigma_vals, row['sigma'])
            if mi < len(mu_vals) and si < len(sigma_vals):
                grid[si, mi] = row['entropy']

        im = axes[1, 1].imshow(grid, origin='lower', aspect='auto',
                               extent=[mu_vals[0], mu_vals[-1], sigma_vals[0], sigma_vals[-1]],
                               cmap='plasma', interpolation='bilinear')
        plt.colorbar(im, ax=axes[1, 1], label='H (bits)')
        axes[1, 1].set_xlabel('μ')
        axes[1, 1].set_ylabel('σ')
    axes[1, 1].set_title("(d) Shannon Entropy")

    plt.tight_layout()
    return fig


def plot_critical_line_analysis(df: pd.DataFrame,
                                sigma_col: str = 'sigma',
                                figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Detailed analysis along a critical line (fixed μ, varying σ).

    Shows transition signatures:
    - Lyapunov → 0
    - Correlation length → ∞
    - Susceptibility peak
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    sigma = df[sigma_col].values

    # Lyapunov
    if 'lyapunov' in df.columns:
        lyap = df['lyapunov'].values
        lyap_std = df['lyapunov_std'].values if 'lyapunov_std' in df.columns else np.zeros_like(lyap)

        axes[0, 0].errorbar(sigma, lyap, yerr=lyap_std, fmt='o-', capsize=3, markersize=4)
        axes[0, 0].axhline(0, color='red', linestyle='--', alpha=0.7, label='λ=0 (critical)')
        axes[0, 0].fill_between(sigma, -0.01, 0.01, alpha=0.2, color='green', label='Critical zone')
        axes[0, 0].set_xlabel('σ')
        axes[0, 0].set_ylabel('Lyapunov exponent (λ)')
        axes[0, 0].set_title('(a) Lyapunov Exponent')
        axes[0, 0].legend(fontsize=8)

    # Correlation length
    if 'correlation_length' in df.columns:
        xi = df['correlation_length'].values
        xi_finite = np.clip(xi, 0, 100)  # Clip infinities

        axes[0, 1].semilogy(sigma, xi_finite, 'o-', markersize=4)
        axes[0, 1].set_xlabel('σ')
        axes[0, 1].set_ylabel('Correlation length (ξ)')
        axes[0, 1].set_title('(b) Correlation Length\n(diverges at criticality)')

    # Entropy
    if 'entropy' in df.columns:
        axes[1, 0].plot(sigma, df['entropy'].values, 'o-', markersize=4, label='Entropy')
        if 'mutual_info' in df.columns:
            ax2 = axes[1, 0].twinx()
            ax2.plot(sigma, df['mutual_info'].values, 's-', color='orange', markersize=4, label='MI')
            ax2.set_ylabel('Mutual Information', color='orange')
        axes[1, 0].set_xlabel('σ')
        axes[1, 0].set_ylabel('Entropy (bits)')
        axes[1, 0].set_title('(c) Information Measures')
        axes[1, 0].legend(loc='upper left', fontsize=8)

    # Susceptibility
    if 'susceptibility' in df.columns:
        axes[1, 1].plot(sigma, df['susceptibility'].values, 'o-', markersize=4)
        axes[1, 1].set_xlabel('σ')
        axes[1, 1].set_ylabel('Susceptibility (χ)')
        axes[1, 1].set_title('(d) Susceptibility\n(peaks at criticality)')

    plt.tight_layout()
    return fig


def summarize_results(df: pd.DataFrame) -> dict:
    """Generate summary statistics from experiment results."""
    summary = {
        'total_points': len(df),
        'phase_counts': df['classification'].value_counts().to_dict(),
    }

    if 'lyapunov' in df.columns:
        lyap = df['lyapunov'].dropna()
        summary['lyapunov'] = {
            'mean': lyap.mean(),
            'std': lyap.std(),
            'min': lyap.min(),
            'max': lyap.max(),
            'n_critical': ((lyap > -0.01) & (lyap < 0.01)).sum(),
        }

    if 'correlation_length' in df.columns:
        xi = df['correlation_length'].dropna()
        xi_finite = xi[np.isfinite(xi)]
        summary['correlation'] = {
            'mean_length': xi_finite.mean() if len(xi_finite) > 0 else None,
            'n_power_law': (df['correlation_type'] == 'power_law').sum(),
            'n_exponential': (df['correlation_type'] == 'exponential').sum(),
        }

    if 'entropy' in df.columns:
        summary['entropy'] = {
            'mean': df['entropy'].mean(),
            'std': df['entropy'].std(),
        }

    return summary


def find_critical_points(df: pd.DataFrame, lyapunov_threshold: float = 0.01) -> pd.DataFrame:
    """
    Identify points near criticality (λ ≈ 0).
    """
    if 'lyapunov' not in df.columns:
        return pd.DataFrame()

    critical = df[
        (df['lyapunov'].notna()) &
        (np.abs(df['lyapunov']) < lyapunov_threshold)
    ].copy()

    return critical.sort_values('lyapunov', key=abs)
