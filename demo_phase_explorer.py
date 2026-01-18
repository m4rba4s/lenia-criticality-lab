#!/usr/bin/env python3
"""
Interactive Phase Diagram Explorer

Click anywhere on the phase diagram to see Lenia dynamics at those parameters.
Explore the boundary between order, criticality, and chaos in real-time.

Usage:
    python demo_phase_explorer.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import json

from src.simulation import LeniaSimulation, LeniaConfig
from src.metrics import LyapunovEstimator


class PhaseExplorer:
    """Interactive phase diagram explorer."""

    # Parameter ranges
    MU_RANGE = (0.10, 0.20)
    SIGMA_RANGE = (0.008, 0.035)

    def __init__(self):
        # Current state
        self.current_mu = 0.15
        self.current_sigma = 0.015
        self.sim = None
        self.running = True
        self.mass_history = []
        self.lyapunov_history = []

        # Setup figure
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.patch.set_facecolor('#0d1117')

        # Title
        self.fig.suptitle(
            'LENIA PHASE SPACE EXPLORER',
            fontsize=18, fontweight='bold', color='#58a6ff', y=0.97
        )
        self.fig.text(0.5, 0.935,
            'Click on the phase diagram to explore different dynamical regimes',
            ha='center', fontsize=11, color='#8b949e', style='italic')

        # Create layout
        gs = gridspec.GridSpec(2, 3,
                              height_ratios=[2, 1],
                              width_ratios=[1.2, 1.5, 0.8],
                              hspace=0.25, wspace=0.2,
                              left=0.05, right=0.95, top=0.90, bottom=0.08)

        # Phase diagram (left)
        self.ax_phase = self.fig.add_subplot(gs[:, 0])
        self.ax_phase.set_facecolor('#161b22')

        # Simulation view (center top)
        self.ax_sim = self.fig.add_subplot(gs[0, 1])
        self.ax_sim.set_facecolor('#0d1117')

        # Mass history (center bottom)
        self.ax_mass = self.fig.add_subplot(gs[1, 1])
        self.ax_mass.set_facecolor('#161b22')

        # Info panel (right)
        self.ax_info = self.fig.add_subplot(gs[:, 2])
        self.ax_info.set_facecolor('#161b22')
        self.ax_info.axis('off')

        # Buttons
        btn_y = 0.02
        self.btn_reset = Button(
            plt.axes([0.05, btn_y, 0.08, 0.035]),
            'RESET', color='#21262d', hovercolor='#30363d'
        )
        self.btn_pause = Button(
            plt.axes([0.14, btn_y, 0.08, 0.035]),
            'PAUSE', color='#238636', hovercolor='#2ea043'
        )
        self.btn_critical = Button(
            plt.axes([0.23, btn_y, 0.10, 0.035]),
            'CRITICAL', color='#9e6a03', hovercolor='#bb8009'
        )
        self.btn_chaos = Button(
            plt.axes([0.34, btn_y, 0.08, 0.035]),
            'CHAOS', color='#da3633', hovercolor='#f85149'
        )
        self.btn_order = Button(
            plt.axes([0.43, btn_y, 0.08, 0.035]),
            'ORDER', color='#1f6feb', hovercolor='#388bfd'
        )

        self.btn_reset.on_clicked(self.reset_sim)
        self.btn_pause.on_clicked(self.toggle_pause)
        self.btn_critical.on_clicked(lambda e: self.goto_regime('critical'))
        self.btn_chaos.on_clicked(lambda e: self.goto_regime('chaos'))
        self.btn_order.on_clicked(lambda e: self.goto_regime('order'))

        # Click handler for phase diagram
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # Current position marker
        self.position_marker = None

        # Initialize
        self._draw_phase_diagram()
        self._init_simulation()
        self._update_info()

    def _create_phase_colormap(self):
        """Create custom colormap for phase diagram."""
        colors = [
            (0.0, '#1a1a2e'),    # Dead - dark
            (0.3, '#2d5a7b'),    # Ordered - blue
            (0.5, '#f0c808'),    # Critical - gold
            (0.7, '#e63946'),    # Chaotic - red
            (1.0, '#ff6b6b'),    # Very chaotic - bright red
        ]
        cmap_colors = []
        cmap_positions = []
        for pos, color in colors:
            cmap_positions.append(pos)
            # Convert hex to RGB
            c = color.lstrip('#')
            rgb = tuple(int(c[i:i+2], 16)/255 for i in (0, 2, 4))
            cmap_colors.append(rgb)

        return LinearSegmentedColormap.from_list('phase', list(zip(cmap_positions, cmap_colors)))

    def _draw_phase_diagram(self):
        """Draw the phase diagram with regime boundaries."""
        self.ax_phase.clear()
        self.ax_phase.set_facecolor('#161b22')

        # Create synthetic phase diagram data
        mu_vals = np.linspace(*self.MU_RANGE, 50)
        sigma_vals = np.linspace(*self.SIGMA_RANGE, 50)
        MU, SIGMA = np.meshgrid(mu_vals, sigma_vals)

        # Approximate Lyapunov landscape (based on known critical region)
        # Critical region is around mu=0.14-0.17, sigma=0.014-0.022
        critical_mu = 0.155
        critical_sigma = 0.018

        # Distance from critical point (adjusted for different scales)
        dist_mu = (MU - critical_mu) / 0.03
        dist_sigma = (SIGMA - critical_sigma) / 0.008

        # Dead zone (low sigma)
        dead_mask = SIGMA < 0.012

        # Approximate Lyapunov
        lyap = 0.1 * (dist_mu**2 + dist_sigma**2) - 0.02
        lyap[dead_mask] = -1.0

        # Add some structure
        lyap += 0.05 * np.sin(MU * 50) * np.sin(SIGMA * 200)

        # Create custom colormap
        cmap = self._create_phase_colormap()

        # Plot
        extent = [*self.MU_RANGE, *self.SIGMA_RANGE]
        im = self.ax_phase.imshow(
            lyap, origin='lower', extent=extent,
            cmap=cmap, aspect='auto', vmin=-0.5, vmax=0.3,
            interpolation='bilinear'
        )

        # Labels and styling
        self.ax_phase.set_xlabel('μ (growth center)', fontsize=11, color='#c9d1d9')
        self.ax_phase.set_ylabel('σ (growth width)', fontsize=11, color='#c9d1d9')
        self.ax_phase.set_title('PHASE DIAGRAM', fontsize=12, fontweight='bold',
                               color='#c9d1d9', pad=10)
        self.ax_phase.tick_params(colors='#8b949e', labelsize=9)

        # Add regime labels
        self.ax_phase.text(0.12, 0.010, 'DEAD', fontsize=9, color='#8b949e',
                          fontweight='bold', ha='center')
        self.ax_phase.text(0.155, 0.017, 'CRITICAL', fontsize=9, color='#f0c808',
                          fontweight='bold', ha='center')
        self.ax_phase.text(0.17, 0.030, 'CHAOTIC', fontsize=9, color='#e63946',
                          fontweight='bold', ha='center')

        # Draw critical band outline
        crit_rect = plt.Rectangle(
            (0.14, 0.014), 0.03, 0.008,
            fill=False, edgecolor='#f0c808', linewidth=2, linestyle='--', alpha=0.7
        )
        self.ax_phase.add_patch(crit_rect)

        # Colorbar
        cbar = self.fig.colorbar(im, ax=self.ax_phase, shrink=0.6, pad=0.02)
        cbar.set_label('λ (Lyapunov)', color='#8b949e', fontsize=9)
        cbar.ax.tick_params(colors='#8b949e', labelsize=8)

        # Draw current position
        self._update_position_marker()

    def _update_position_marker(self):
        """Update the position marker on phase diagram."""
        if self.position_marker:
            self.position_marker.remove()

        self.position_marker = self.ax_phase.scatter(
            [self.current_mu], [self.current_sigma],
            s=200, c='white', marker='o', edgecolors='#58a6ff',
            linewidths=3, zorder=10
        )

        # Add crosshairs
        self.ax_phase.axhline(self.current_sigma, color='#58a6ff', alpha=0.3, linewidth=1)
        self.ax_phase.axvline(self.current_mu, color='#58a6ff', alpha=0.3, linewidth=1)

    def _init_simulation(self):
        """Initialize simulation with current parameters."""
        config = LeniaConfig(
            grid_size=128,
            mu=self.current_mu,
            sigma=self.current_sigma,
            seed=42,
            init_size=0.22,
            init_density=0.5,
        )
        self.sim = LeniaSimulation(config)
        self.mass_history = []

        # Warmup
        for _ in range(50):
            self.sim.step()
            self.mass_history.append(self.sim.mass())

    def _get_regime(self) -> tuple:
        """Estimate current regime based on parameters."""
        mu, sigma = self.current_mu, self.current_sigma

        if sigma < 0.012:
            return 'DEAD', '#8b949e', 'Insufficient growth width'
        elif 0.14 <= mu <= 0.17 and 0.014 <= sigma <= 0.022:
            return 'CRITICAL', '#f0c808', 'Edge of chaos - λ ≈ 0'
        elif sigma > 0.028 or mu > 0.18:
            return 'CHAOTIC', '#e63946', 'Exponential divergence - λ > 0'
        else:
            return 'ORDERED', '#58a6ff', 'Perturbations decay - λ < 0'

    def _update_info(self):
        """Update info panel."""
        self.ax_info.clear()
        self.ax_info.axis('off')

        regime, color, desc = self._get_regime()

        # Regime icon
        icons = {
            'DEAD': '💀', 'CRITICAL': '💎', 'CHAOTIC': '🔥', 'ORDERED': '❄️'
        }
        icon = icons.get(regime, '?')

        lines = [
            (f"{icon} {regime}", 16, color),
            ("", 10, 'white'),
            (desc, 10, '#8b949e'),
            ("", 10, 'white'),
            ("─" * 20, 10, '#30363d'),
            ("PARAMETERS", 11, '#8b949e'),
            (f"μ = {self.current_mu:.4f}", 11, '#c9d1d9'),
            (f"σ = {self.current_sigma:.5f}", 11, '#c9d1d9'),
            ("", 10, 'white'),
            ("─" * 20, 10, '#30363d'),
            ("SIMULATION", 11, '#8b949e'),
        ]

        if self.sim:
            mass = self.sim.mass()
            step = self.sim.step_count
            status = "ALIVE" if mass > 10 else "DEAD"
            status_color = '#3fb950' if mass > 10 else '#f85149'

            lines.extend([
                (f"Step: {step}", 10, '#c9d1d9'),
                (f"Mass: {mass:.1f}", 10, '#c9d1d9'),
                (f"Status: {status}", 10, status_color),
            ])

        lines.extend([
            ("", 10, 'white'),
            ("─" * 20, 10, '#30363d'),
            ("CLICK phase diagram", 9, '#58a6ff'),
            ("to explore parameters", 9, '#58a6ff'),
        ])

        y = 0.95
        for text, size, color in lines:
            self.ax_info.text(0.1, y, text, fontsize=size, color=color,
                             transform=self.ax_info.transAxes,
                             family='monospace', fontweight='bold' if size > 12 else 'normal')
            y -= 0.055

    def on_click(self, event):
        """Handle clicks on phase diagram."""
        if event.inaxes != self.ax_phase:
            return

        # Get clicked coordinates
        mu = event.xdata
        sigma = event.ydata

        if mu is None or sigma is None:
            return

        # Clamp to valid range
        mu = max(self.MU_RANGE[0], min(self.MU_RANGE[1], mu))
        sigma = max(self.SIGMA_RANGE[0], min(self.SIGMA_RANGE[1], sigma))

        self.current_mu = mu
        self.current_sigma = sigma

        self._init_simulation()
        self._draw_phase_diagram()
        self._update_info()

    def reset_sim(self, event):
        """Reset current simulation."""
        self._init_simulation()

    def toggle_pause(self, event):
        """Toggle pause state."""
        self.running = not self.running
        self.btn_pause.label.set_text('RESUME' if not self.running else 'PAUSE')

    def goto_regime(self, regime: str):
        """Jump to a specific regime."""
        if regime == 'critical':
            self.current_mu = 0.155
            self.current_sigma = 0.017
        elif regime == 'chaos':
            self.current_mu = 0.17
            self.current_sigma = 0.030
        elif regime == 'order':
            self.current_mu = 0.13
            self.current_sigma = 0.020

        self._init_simulation()
        self._draw_phase_diagram()
        self._update_info()

    def update(self, frame):
        """Animation update."""
        if not self.running or self.sim is None:
            return

        # Step simulation
        for _ in range(2):
            self.sim.step()
            self.mass_history.append(self.sim.mass())

        if len(self.mass_history) > 500:
            self.mass_history = self.mass_history[-500:]

        # Update simulation view
        self.ax_sim.clear()
        self.ax_sim.set_facecolor('#0d1117')
        self.ax_sim.axis('off')

        regime, color, _ = self._get_regime()
        cmap = {
            'DEAD': 'gray', 'CRITICAL': 'viridis',
            'CHAOTIC': 'inferno', 'ORDERED': 'Blues'
        }.get(regime, 'viridis')

        self.ax_sim.imshow(self.sim.world, cmap=cmap, vmin=0, vmax=1,
                          interpolation='bilinear')

        status = "ALIVE" if self.sim.mass() > 10 else "DEAD"
        title_color = '#3fb950' if status == "ALIVE" else '#f85149'
        self.ax_sim.set_title(
            f"Step {self.sim.step_count}  |  Mass: {self.sim.mass():.0f}  |  {status}",
            fontsize=11, color=title_color, fontweight='bold'
        )

        # Update mass history
        self.ax_mass.clear()
        self.ax_mass.set_facecolor('#161b22')

        if self.mass_history:
            x = np.arange(len(self.mass_history))
            y = np.array(self.mass_history)

            self.ax_mass.fill_between(x, y, alpha=0.3, color=color)
            self.ax_mass.plot(x, y, color=color, linewidth=1.5)
            self.ax_mass.set_xlim(0, max(500, len(x)))
            self.ax_mass.set_ylim(0, max(100, max(y) * 1.1))

        self.ax_mass.set_xlabel('Step', fontsize=9, color='#8b949e')
        self.ax_mass.set_ylabel('Mass', fontsize=9, color='#8b949e')
        self.ax_mass.tick_params(colors='#8b949e', labelsize=8)
        for spine in self.ax_mass.spines.values():
            spine.set_color('#30363d')

        # Update info
        self._update_info()

    def run(self):
        """Run the explorer."""
        self.ani = FuncAnimation(
            self.fig, self.update,
            interval=50, blit=False, cache_frame_data=False
        )
        plt.show()


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   LENIA PHASE SPACE EXPLORER                                    ║
║   ──────────────────────────                                    ║
║                                                                  ║
║   Explore the parameter space of continuous cellular automata   ║
║   and discover the boundary between order and chaos.            ║
║                                                                  ║
║   CONTROLS:                                                      ║
║   • CLICK on phase diagram - explore those parameters           ║
║   • CRITICAL button - jump to edge of chaos                     ║
║   • CHAOS button - jump to chaotic regime                       ║
║   • ORDER button - jump to ordered regime                       ║
║   • PAUSE/RESET - control simulation                            ║
║                                                                  ║
║   REGIMES:                                                       ║
║   💎 CRITICAL (λ ≈ 0) - stable organisms, edge of chaos         ║
║   🔥 CHAOTIC (λ > 0)  - explosive, turbulent dynamics           ║
║   ❄️  ORDERED (λ < 0)  - patterns decay, perturbations die      ║
║   💀 DEAD             - nothing survives                        ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    explorer = PhaseExplorer()
    explorer.run()


if __name__ == '__main__':
    main()
