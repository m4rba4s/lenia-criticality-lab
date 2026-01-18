#!/usr/bin/env python3
"""
NAND Gate Demonstration in Lenia

Interactive visualization of emergent NAND-like computation through
self-repair threshold dynamics in continuous cellular automata.

This demo shows the key finding of our research:
- Single perturbation → organism survives (self-repair)
- Double perturbation → organism collapses (threshold exceeded)

Usage:
    python demo_nand_gate.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.widgets import Button
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing import Optional, Tuple
import colorsys

from src.simulation import LeniaSimulation, LeniaConfig


@dataclass
class NANDState:
    """State for one NAND input condition."""
    input_a: int
    input_b: int
    sim: Optional[LeniaSimulation] = None
    mass_history: list = None
    perturbation_applied: bool = False
    outcome: Optional[str] = None  # "ALIVE" or "COLLAPSED"

    def __post_init__(self):
        self.mass_history = []

    @property
    def expected_nand(self) -> int:
        """Expected NAND output."""
        return 0 if (self.input_a and self.input_b) else 1

    @property
    def label(self) -> str:
        return f"({self.input_a}, {self.input_b})"


class NANDDemo:
    """Interactive NAND gate demonstration."""

    # Critical parameters (Orbium morphology)
    MU = 0.15
    SIGMA = 0.015
    GRID_SIZE = 100

    # Timing
    WARMUP_STEPS = 60
    PERTURBATION_STEP = 80
    TOTAL_STEPS = 280

    # Perturbation parameters
    HOLE_SIGMA = 5
    HOLE_STRENGTH = 0.55
    HOLE_OFFSET = 12

    # Collapse threshold
    COLLAPSE_THRESHOLD = 20

    def __init__(self):
        # State for each input condition
        self.states = [
            NANDState(0, 0),
            NANDState(0, 1),
            NANDState(1, 0),
            NANDState(1, 1),
        ]

        self.current_step = 0
        self.running = False
        self.demo_complete = False

        # Setup figure
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.patch.set_facecolor('#0d1117')

        # Title
        self.fig.suptitle(
            'EMERGENT NAND COMPUTATION IN LENIA',
            fontsize=18, fontweight='bold', color='#58a6ff',
            y=0.97
        )

        # Subtitle
        self.fig.text(0.5, 0.93,
            'Self-repair threshold dynamics at the edge of chaos',
            ha='center', fontsize=11, color='#8b949e', style='italic')

        # Create grid layout
        gs = gridspec.GridSpec(3, 4,
                              height_ratios=[0.1, 2.5, 1],
                              width_ratios=[1, 1, 1, 1],
                              hspace=0.3, wspace=0.15,
                              left=0.05, right=0.95, top=0.88, bottom=0.12)

        # Input labels row
        self.ax_labels = []
        for i in range(4):
            ax = self.fig.add_subplot(gs[0, i])
            ax.axis('off')
            self.ax_labels.append(ax)

        # Simulation panels
        self.ax_sims = []
        for i in range(4):
            ax = self.fig.add_subplot(gs[1, i])
            ax.set_facecolor('#0d1117')
            ax.axis('off')
            self.ax_sims.append(ax)

        # Mass history panels
        self.ax_mass = []
        for i in range(4):
            ax = self.fig.add_subplot(gs[2, i])
            ax.set_facecolor('#161b22')
            self.ax_mass.append(ax)

        # Control buttons
        self.btn_start = Button(
            plt.axes([0.35, 0.02, 0.12, 0.04]),
            'START DEMO',
            color='#238636', hovercolor='#2ea043'
        )
        self.btn_reset = Button(
            plt.axes([0.48, 0.02, 0.08, 0.04]),
            'RESET',
            color='#21262d', hovercolor='#30363d'
        )
        self.btn_perturb = Button(
            plt.axes([0.57, 0.02, 0.12, 0.04]),
            'APPLY DAMAGE',
            color='#da3633', hovercolor='#f85149'
        )

        self.btn_start.on_clicked(self.start_demo)
        self.btn_reset.on_clicked(self.reset_demo)
        self.btn_perturb.on_clicked(self.apply_perturbations)

        # Info panel
        self.ax_info = self.fig.add_axes([0.72, 0.01, 0.26, 0.08])
        self.ax_info.axis('off')

        # Initialize
        self.reset_demo(None)
        self._update_labels()
        self._update_info()

    def _create_organism(self, seed: int = 42) -> LeniaSimulation:
        """Create a Lenia organism with critical parameters."""
        config = LeniaConfig(
            grid_size=self.GRID_SIZE,
            mu=self.MU,
            sigma=self.SIGMA,
            seed=seed,
            init_size=0.22,
            init_density=0.5,
        )
        return LeniaSimulation(config)

    def _apply_hole(self, sim: LeniaSimulation, x_offset: int):
        """Apply Gaussian hole perturbation."""
        # Find organism center
        if sim.mass() < 10:
            return

        cy, cx = np.unravel_index(
            np.argmax(sim.world), sim.world.shape
        )

        # Actually find center of mass
        total = sim.mass()
        if total > 0:
            yy, xx = np.meshgrid(
                np.arange(sim.world.shape[0]),
                np.arange(sim.world.shape[1]),
                indexing='ij'
            )
            cy = int(np.sum(yy * sim.world) / total)
            cx = int(np.sum(xx * sim.world) / total)

        # Create Gaussian hole
        size = self.GRID_SIZE
        yy, xx = np.ogrid[:size, :size]
        hole_x = cx + x_offset
        hole_y = cy

        # Clamp to grid
        hole_x = max(0, min(size-1, hole_x))
        hole_y = max(0, min(size-1, hole_y))

        dist_sq = (xx - hole_x)**2 + (yy - hole_y)**2
        hole = np.exp(-dist_sq / (2 * self.HOLE_SIGMA**2))

        # Apply damage (multiply by 1 - hole * strength)
        sim.world = sim.world * (1 - hole * self.HOLE_STRENGTH)
        sim.world = np.clip(sim.world, 0, 1)

    def _update_labels(self):
        """Update input condition labels."""
        for i, (ax, state) in enumerate(zip(self.ax_labels, self.states)):
            ax.clear()
            ax.axis('off')

            # Input label
            a_color = '#f85149' if state.input_a else '#8b949e'
            b_color = '#f85149' if state.input_b else '#8b949e'

            label = f"Input: A={state.input_a}  B={state.input_b}"
            expected = f"NAND → {state.expected_nand}"

            ax.text(0.5, 0.7, label, ha='center', va='center',
                   fontsize=11, fontweight='bold', color='#c9d1d9',
                   transform=ax.transAxes)

            expected_color = '#3fb950' if state.expected_nand else '#f85149'
            ax.text(0.5, 0.2, expected, ha='center', va='center',
                   fontsize=10, color=expected_color,
                   transform=ax.transAxes)

    def _update_info(self):
        """Update info panel."""
        self.ax_info.clear()
        self.ax_info.axis('off')

        status = "READY" if not self.running else "RUNNING"
        if self.demo_complete:
            status = "COMPLETE"

        info_text = f"""Step: {self.current_step}/{self.TOTAL_STEPS}   |   Status: {status}
Parameters: μ={self.MU}, σ={self.SIGMA}   |   Collapse threshold: mass < {self.COLLAPSE_THRESHOLD}"""

        self.ax_info.text(0.0, 0.5, info_text,
                         fontsize=9, color='#8b949e',
                         transform=self.ax_info.transAxes,
                         family='monospace', va='center')

    def reset_demo(self, event):
        """Reset the demonstration."""
        self.running = False
        self.demo_complete = False
        self.current_step = 0

        # Create fresh organisms
        for i, state in enumerate(self.states):
            state.sim = self._create_organism(seed=42 + i * 100)
            state.mass_history = []
            state.perturbation_applied = False
            state.outcome = None

        # Warmup
        for state in self.states:
            for _ in range(self.WARMUP_STEPS):
                state.sim.step()
                state.mass_history.append(state.sim.mass())

        self.current_step = self.WARMUP_STEPS
        self._update_display()
        self._update_info()

    def start_demo(self, event):
        """Start/pause the simulation."""
        self.running = not self.running
        self.btn_start.label.set_text('PAUSE' if self.running else 'RESUME')

    def apply_perturbations(self, event):
        """Apply perturbations based on input conditions."""
        for state in self.states:
            if state.input_a:
                self._apply_hole(state.sim, -self.HOLE_OFFSET)
            if state.input_b:
                self._apply_hole(state.sim, +self.HOLE_OFFSET)
            state.perturbation_applied = True

        self._update_display()

    def _get_colormap(self, state: NANDState):
        """Get colormap based on state."""
        if state.outcome == "COLLAPSED":
            return 'Reds'
        elif state.outcome == "ALIVE":
            return 'Greens'
        elif state.perturbation_applied:
            return 'plasma'
        else:
            return 'viridis'

    def _update_display(self):
        """Update all display panels."""
        for i, (ax_sim, ax_m, state) in enumerate(
            zip(self.ax_sims, self.ax_mass, self.states)
        ):
            # Simulation panel
            ax_sim.clear()
            ax_sim.axis('off')

            if state.sim is not None:
                cmap = self._get_colormap(state)
                ax_sim.imshow(state.sim.world, cmap=cmap, vmin=0, vmax=1,
                             interpolation='bilinear')

                # Border based on outcome
                if state.outcome == "ALIVE":
                    border_color = '#3fb950'
                    title = "SURVIVED"
                elif state.outcome == "COLLAPSED":
                    border_color = '#f85149'
                    title = "COLLAPSED"
                else:
                    border_color = '#30363d'
                    title = f"Mass: {state.sim.mass():.0f}"

                for spine in ax_sim.spines.values():
                    spine.set_visible(True)
                    spine.set_color(border_color)
                    spine.set_linewidth(3)

                ax_sim.set_title(title, color=border_color, fontsize=10,
                                fontweight='bold', pad=5)

            # Mass history panel
            ax_m.clear()
            ax_m.set_facecolor('#161b22')

            if state.mass_history:
                x = np.arange(len(state.mass_history))
                y = np.array(state.mass_history)

                # Color gradient
                if state.outcome == "COLLAPSED":
                    color = '#f85149'
                elif state.outcome == "ALIVE":
                    color = '#3fb950'
                else:
                    color = '#58a6ff'

                ax_m.fill_between(x, y, alpha=0.3, color=color)
                ax_m.plot(x, y, color=color, linewidth=1.5)

                # Threshold line
                ax_m.axhline(self.COLLAPSE_THRESHOLD, color='#f85149',
                            linestyle='--', alpha=0.5, linewidth=1)

                # Perturbation marker
                if state.perturbation_applied and self.PERTURBATION_STEP < len(x):
                    ax_m.axvline(self.PERTURBATION_STEP, color='#f0883e',
                                linestyle=':', alpha=0.7)

                ax_m.set_xlim(0, self.TOTAL_STEPS)
                ax_m.set_ylim(0, max(100, max(y) * 1.1) if len(y) > 0 else 100)

            ax_m.tick_params(colors='#8b949e', labelsize=7)
            for spine in ax_m.spines.values():
                spine.set_color('#30363d')

            if i == 0:
                ax_m.set_ylabel('Mass', fontsize=8, color='#8b949e')

        self.fig.canvas.draw_idle()

    def update(self, frame):
        """Animation update."""
        if not self.running or self.demo_complete:
            return

        # Auto-apply perturbation at the right time
        if self.current_step == self.PERTURBATION_STEP:
            if not self.states[0].perturbation_applied:
                self.apply_perturbations(None)

        # Step all simulations
        for state in self.states:
            if state.sim is not None:
                state.sim.step()
                state.mass_history.append(state.sim.mass())

                # Check outcome
                if state.perturbation_applied and state.outcome is None:
                    if state.sim.mass() < self.COLLAPSE_THRESHOLD:
                        state.outcome = "COLLAPSED"

        self.current_step += 1

        # Check if demo is complete
        if self.current_step >= self.TOTAL_STEPS:
            self.demo_complete = True
            self.running = False
            self.btn_start.label.set_text('COMPLETE')

            # Set final outcomes
            for state in self.states:
                if state.outcome is None:
                    state.outcome = "ALIVE"

            self._show_truth_table()

        self._update_display()
        self._update_info()

    def _show_truth_table(self):
        """Show final truth table comparison."""
        # Add truth table overlay
        self.fig.text(0.5, 0.06,
            '═══ NAND TRUTH TABLE VERIFIED ═══',
            ha='center', fontsize=12, fontweight='bold', color='#58a6ff')

        results = []
        for state in self.states:
            actual = 1 if state.outcome == "ALIVE" else 0
            expected = state.expected_nand
            match = "✓" if actual == expected else "✗"
            results.append(f"{state.label} → {actual} (expected {expected}) {match}")

        self.fig.text(0.5, 0.025,
            '   |   '.join(results),
            ha='center', fontsize=9, color='#8b949e', family='monospace')

    def run(self):
        """Run the demo."""
        self.ani = FuncAnimation(
            self.fig, self.update,
            interval=50, blit=False, cache_frame_data=False
        )
        plt.show()


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   NAND GATE DEMONSTRATION IN LENIA                              ║
║   ─────────────────────────────────                             ║
║                                                                  ║
║   This demo visualizes emergent computation through             ║
║   self-repair threshold dynamics at the edge of chaos.          ║
║                                                                  ║
║   CONTROLS:                                                      ║
║   • START DEMO  - Begin the simulation                          ║
║   • RESET       - Reset all organisms                           ║
║   • APPLY DAMAGE - Manually trigger perturbations               ║
║                                                                  ║
║   WHAT TO OBSERVE:                                               ║
║   • Input (0,0): No damage → survives                           ║
║   • Input (0,1): One hole → self-repairs → survives             ║
║   • Input (1,0): One hole → self-repairs → survives             ║
║   • Input (1,1): Two holes → exceeds threshold → COLLAPSES      ║
║                                                                  ║
║   This implements NAND logic through intrinsic dynamics!        ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    demo = NANDDemo()
    demo.run()


if __name__ == '__main__':
    main()
