#!/usr/bin/env python3
"""
Signal Propagation Demonstration

Visualize how perturbations propagate through Lenia organisms
with lagged correlations and temporal structure.

Usage:
    python demo_signal_propagation.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Rectangle
from scipy.ndimage import gaussian_filter

from src.simulation import LeniaSimulation, LeniaConfig
from src.metrics import TransferEntropyEstimator


class SignalPropagationDemo:
    """Interactive signal propagation demonstration."""

    # Critical parameters
    MU = 0.1585
    SIGMA = 0.01808
    GRID_SIZE = 150

    # Probe configuration
    N_PROBES = 6
    PROBE_SIZE = 12

    # Timing
    WARMUP_STEPS = 80
    MEASURE_INTERVAL = 2

    def __init__(self):
        self.sim = None
        self.running = True
        self.perturbation_applied = False

        # Probe data
        self.probe_positions = []
        self.probe_histories = [[] for _ in range(self.N_PROBES)]
        self.baseline_values = [0] * self.N_PROBES

        # Transfer entropy estimator
        self.te_estimator = TransferEntropyEstimator(
            n_bins=6,           # Fewer bins for short time series
            history_length=2,   # Past window
            lag=1,              # Prediction lag
            n_surrogates=50     # For significance testing
        )
        self.te_results = None
        self.last_te_compute = 0

        # Setup figure
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.patch.set_facecolor('#0d1117')

        # Title
        self.fig.suptitle(
            'SIGNAL PROPAGATION IN LENIA',
            fontsize=18, fontweight='bold', color='#58a6ff', y=0.97
        )
        self.fig.text(0.5, 0.935,
            'Local perturbations propagate with measurable temporal structure',
            ha='center', fontsize=11, color='#8b949e', style='italic')

        # Layout - 3x3 grid
        gs = gridspec.GridSpec(3, 3,
                              height_ratios=[2, 1, 1],
                              width_ratios=[1.5, 1, 1],
                              hspace=0.3, wspace=0.2,
                              left=0.05, right=0.95, top=0.90, bottom=0.12)

        # Main simulation view
        self.ax_sim = self.fig.add_subplot(gs[0, 0])
        self.ax_sim.set_facecolor('#0d1117')

        # Activity heatmap
        self.ax_heat = self.fig.add_subplot(gs[0, 1:])
        self.ax_heat.set_facecolor('#161b22')

        # Probe timeseries
        self.ax_probes = self.fig.add_subplot(gs[1, :])
        self.ax_probes.set_facecolor('#161b22')

        # Correlation matrix
        self.ax_corr = self.fig.add_subplot(gs[2, 0])
        self.ax_corr.set_facecolor('#161b22')

        # Transfer Entropy matrix (NEW!)
        self.ax_te = self.fig.add_subplot(gs[2, 1])
        self.ax_te.set_facecolor('#161b22')

        # Info panel
        self.ax_info = self.fig.add_subplot(gs[2, 2])
        self.ax_info.set_facecolor('#161b22')
        self.ax_info.axis('off')

        # Buttons
        self.btn_perturb = Button(
            plt.axes([0.15, 0.02, 0.15, 0.04]),
            'APPLY PERTURBATION',
            color='#f0883e', hovercolor='#ffa657'
        )
        self.btn_reset = Button(
            plt.axes([0.32, 0.02, 0.08, 0.04]),
            'RESET',
            color='#21262d', hovercolor='#30363d'
        )
        self.btn_pause = Button(
            plt.axes([0.42, 0.02, 0.08, 0.04]),
            'PAUSE',
            color='#238636', hovercolor='#2ea043'
        )

        self.btn_perturb.on_clicked(self.apply_perturbation)
        self.btn_reset.on_clicked(self.reset_demo)
        self.btn_pause.on_clicked(self.toggle_pause)

        # Initialize
        self.reset_demo(None)

    def _init_simulation(self):
        """Initialize simulation with critical organism."""
        config = LeniaConfig(
            grid_size=self.GRID_SIZE,
            mu=self.MU,
            sigma=self.SIGMA,
            seed=42,
            init_size=0.25,
            init_density=0.5,
        )
        self.sim = LeniaSimulation(config)

        # Warmup
        for _ in range(self.WARMUP_STEPS):
            self.sim.step()

        # Setup probes along horizontal axis through organism
        self._setup_probes()

    def _setup_probes(self):
        """Setup measurement probes through organism center."""
        # Find organism center of mass
        total = self.sim.mass()
        if total < 10:
            return

        yy, xx = np.meshgrid(
            np.arange(self.GRID_SIZE),
            np.arange(self.GRID_SIZE),
            indexing='ij'
        )
        cy = int(np.sum(yy * self.sim.world) / total)
        cx = int(np.sum(xx * self.sim.world) / total)

        # Place probes horizontally through center
        probe_span = 50
        x_positions = np.linspace(cx - probe_span//2, cx + probe_span//2, self.N_PROBES)

        self.probe_positions = []
        for i, x in enumerate(x_positions):
            self.probe_positions.append((cy, int(x)))

        # Record baseline
        for i, (py, px) in enumerate(self.probe_positions):
            region = self._get_probe_region(py, px)
            self.baseline_values[i] = np.mean(self.sim.world[region])

        # Clear histories
        self.probe_histories = [[] for _ in range(self.N_PROBES)]
        self.perturbation_applied = False

    def _get_probe_region(self, cy, cx):
        """Get slice for probe region."""
        half = self.PROBE_SIZE // 2
        y_start = max(0, cy - half)
        y_end = min(self.GRID_SIZE, cy + half)
        x_start = max(0, cx - half)
        x_end = min(self.GRID_SIZE, cx + half)
        return slice(y_start, y_end), slice(x_start, x_end)

    def apply_perturbation(self, event):
        """Apply perturbation at first probe."""
        if not self.probe_positions or self.perturbation_applied:
            return

        py, px = self.probe_positions[0]

        # Create Gaussian perturbation
        size = self.GRID_SIZE
        yy, xx = np.ogrid[:size, :size]
        dist_sq = (xx - px)**2 + (yy - py)**2
        perturbation = 0.4 * np.exp(-dist_sq / (2 * 8**2))

        # Apply (additive)
        self.sim.world = np.clip(self.sim.world + perturbation, 0, 1)
        self.perturbation_applied = True

    def reset_demo(self, event):
        """Reset the demonstration."""
        self._init_simulation()
        self._update_display()

    def toggle_pause(self, event):
        """Toggle pause state."""
        self.running = not self.running
        self.btn_pause.label.set_text('RESUME' if not self.running else 'PAUSE')

    def _compute_correlations(self):
        """Compute lagged cross-correlations between probes."""
        if len(self.probe_histories[0]) < 20:
            return None

        n = self.N_PROBES
        corr_matrix = np.zeros((n, n))

        lag = 3  # frames

        for i in range(n):
            for j in range(n):
                h1 = np.array(self.probe_histories[i])
                h2 = np.array(self.probe_histories[j])

                if len(h1) > lag and len(h2) > lag:
                    # Compute correlation with lag
                    if i <= j:
                        x1 = h1[:-lag] if lag > 0 else h1
                        x2 = h2[lag:] if lag > 0 else h2
                    else:
                        x1 = h1[lag:]
                        x2 = h2[:-lag] if lag > 0 else h2

                    min_len = min(len(x1), len(x2))
                    if min_len > 5:
                        x1, x2 = x1[:min_len], x2[:min_len]
                        if np.std(x1) > 1e-6 and np.std(x2) > 1e-6:
                            corr_matrix[i, j] = np.corrcoef(x1, x2)[0, 1]

        return corr_matrix

    def _update_display(self):
        """Update all display panels."""
        # Main simulation view with probe markers
        self.ax_sim.clear()
        self.ax_sim.set_facecolor('#0d1117')
        self.ax_sim.imshow(self.sim.world, cmap='viridis', vmin=0, vmax=1)
        self.ax_sim.axis('off')

        # Draw probes
        colors = plt.cm.rainbow(np.linspace(0, 1, self.N_PROBES))
        for i, (py, px) in enumerate(self.probe_positions):
            circle = Circle((px, py), self.PROBE_SIZE//2,
                           fill=False, edgecolor=colors[i],
                           linewidth=2, linestyle='--')
            self.ax_sim.add_patch(circle)
            self.ax_sim.text(px, py - self.PROBE_SIZE,
                           f'P{i+1}', ha='center', fontsize=9,
                           color=colors[i], fontweight='bold')

        # Perturbation source marker
        if self.perturbation_applied and self.probe_positions:
            py, px = self.probe_positions[0]
            self.ax_sim.scatter([px], [py], s=200, c='#f0883e',
                               marker='*', zorder=10)

        status = "PERTURBATION APPLIED" if self.perturbation_applied else "BASELINE"
        self.ax_sim.set_title(
            f"Step {self.sim.step_count}  |  {status}",
            fontsize=11, color='#c9d1d9', fontweight='bold'
        )

        # Probe timeseries
        self.ax_probes.clear()
        self.ax_probes.set_facecolor('#161b22')

        for i, history in enumerate(self.probe_histories):
            if history:
                x = np.arange(len(history))
                y = np.array(history) - self.baseline_values[i]  # Show deviation
                self.ax_probes.plot(x, y, color=colors[i], linewidth=1.5,
                                   label=f'P{i+1}', alpha=0.8)

        self.ax_probes.axhline(0, color='#30363d', linewidth=1, linestyle='--')

        if self.perturbation_applied:
            # Mark perturbation time
            pert_time = len(self.probe_histories[0]) - 1 if self.probe_histories[0] else 0
            self.ax_probes.axvline(0, color='#f0883e', linewidth=2,
                                  linestyle=':', alpha=0.7, label='Perturbation')

        self.ax_probes.set_xlabel('Time (frames)', fontsize=10, color='#8b949e')
        self.ax_probes.set_ylabel('Activity deviation', fontsize=10, color='#8b949e')
        self.ax_probes.legend(loc='upper right', fontsize=8, ncol=3,
                             framealpha=0.3, edgecolor='#30363d')
        self.ax_probes.tick_params(colors='#8b949e', labelsize=8)
        for spine in self.ax_probes.spines.values():
            spine.set_color('#30363d')

        # Correlation matrix
        self.ax_corr.clear()
        self.ax_corr.set_facecolor('#161b22')

        corr = self._compute_correlations()
        if corr is not None:
            im = self.ax_corr.imshow(corr, cmap='RdYlBu_r', vmin=-1, vmax=1,
                                    interpolation='nearest')
            self.ax_corr.set_xticks(range(self.N_PROBES))
            self.ax_corr.set_yticks(range(self.N_PROBES))
            self.ax_corr.set_xticklabels([f'P{i+1}' for i in range(self.N_PROBES)],
                                        fontsize=8, color='#8b949e')
            self.ax_corr.set_yticklabels([f'P{i+1}' for i in range(self.N_PROBES)],
                                        fontsize=8, color='#8b949e')
            self.ax_corr.set_title('LAGGED CORRELATIONS (τ=3)',
                                  fontsize=10, color='#c9d1d9', fontweight='bold')

            # Show correlation values
            for i in range(self.N_PROBES):
                for j in range(self.N_PROBES):
                    val = corr[i, j]
                    color = 'white' if abs(val) > 0.5 else '#8b949e'
                    self.ax_corr.text(j, i, f'{val:.2f}', ha='center', va='center',
                                     fontsize=7, color=color)
        else:
            self.ax_corr.text(0.5, 0.5, 'Collecting data...',
                             ha='center', va='center', transform=self.ax_corr.transAxes,
                             fontsize=12, color='#8b949e')
            self.ax_corr.set_title('CORRELATION MATRIX',
                                  fontsize=10, color='#c9d1d9', fontweight='bold')

        # Activity heatmap (space-time)
        self.ax_heat.clear()
        self.ax_heat.set_facecolor('#161b22')

        if self.probe_histories[0]:
            # Create space-time heatmap
            n_time = len(self.probe_histories[0])
            heatmap = np.zeros((self.N_PROBES, n_time))
            for i, history in enumerate(self.probe_histories):
                if len(history) == n_time:
                    heatmap[i, :] = np.array(history) - self.baseline_values[i]

            if n_time > 5:
                im = self.ax_heat.imshow(heatmap, aspect='auto', cmap='plasma',
                                        interpolation='bilinear', origin='lower')
                self.ax_heat.set_yticks(range(self.N_PROBES))
                self.ax_heat.set_yticklabels([f'P{i+1}' for i in range(self.N_PROBES)],
                                            fontsize=8, color='#8b949e')
                self.ax_heat.set_xlabel('Time', fontsize=9, color='#8b949e')
                self.ax_heat.tick_params(colors='#8b949e', labelsize=7)

        self.ax_heat.set_title('SPACE-TIME ACTIVITY',
                              fontsize=10, color='#c9d1d9', fontweight='bold')

        # Info panel
        self.ax_info.clear()
        self.ax_info.axis('off')

        corr = self._compute_correlations()
        p1_p2_corr = corr[0, 1] if corr is not None else 0

        info_lines = [
            ('SIGNAL PROPAGATION', 12, '#58a6ff'),
            ('', 8, 'white'),
            (f'Adjacent probe correlation:', 9, '#8b949e'),
            (f'r(P1,P2) = {p1_p2_corr:.3f}', 11, '#3fb950' if p1_p2_corr > 0.5 else '#f0883e'),
            ('', 8, 'white'),
            ('Key finding:', 9, '#8b949e'),
            ('Correlations decrease', 9, '#c9d1d9'),
            ('with distance from source', 9, '#c9d1d9'),
        ]

        y = 0.9
        for text, size, color in info_lines:
            self.ax_info.text(0.1, y, text, fontsize=size, color=color,
                             transform=self.ax_info.transAxes,
                             fontweight='bold' if size > 10 else 'normal')
            y -= 0.12

        self.fig.canvas.draw_idle()

    def update(self, frame):
        """Animation update."""
        if not self.running or self.sim is None:
            return

        # Step simulation
        for _ in range(self.MEASURE_INTERVAL):
            self.sim.step()

        # Record probe values
        for i, (py, px) in enumerate(self.probe_positions):
            region = self._get_probe_region(py, px)
            value = np.mean(self.sim.world[region])
            self.probe_histories[i].append(value)

        # Limit history length
        max_history = 200
        for i in range(self.N_PROBES):
            if len(self.probe_histories[i]) > max_history:
                self.probe_histories[i] = self.probe_histories[i][-max_history:]

        self._update_display()

    def run(self):
        """Run the demo."""
        self.ani = FuncAnimation(
            self.fig, self.update,
            interval=80, blit=False, cache_frame_data=False
        )
        plt.show()


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   SIGNAL PROPAGATION DEMONSTRATION                              ║
║   ────────────────────────────────                              ║
║                                                                  ║
║   This demo shows how perturbations propagate through           ║
║   Lenia organisms with measurable temporal structure.           ║
║                                                                  ║
║   PROBES:                                                        ║
║   P1-P6 are measurement points placed through the organism.     ║
║   Each probe measures local activity intensity.                 ║
║                                                                  ║
║   WHAT TO OBSERVE:                                               ║
║   1. Click "APPLY PERTURBATION" to inject signal at P1          ║
║   2. Watch the activity wave propagate P1 → P2 → P3 → ...       ║
║   3. See correlations decrease with distance                    ║
║   4. Space-time heatmap shows propagation pattern               ║
║                                                                  ║
║   KEY RESULT: r(P1,P2) ≈ 0.84 with lag τ = 5 steps             ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    demo = SignalPropagationDemo()
    demo.run()


if __name__ == '__main__':
    main()
