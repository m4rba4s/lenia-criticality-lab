#!/usr/bin/env python3
"""
CASCADING EXPERIMENT

Can one organism's collapse trigger another's?
This is critical for demonstrating functional universality.

The experiment places two organisms and tests whether:
1. Collapse of organism A creates a shockwave
2. This shockwave can trigger collapse of organism B
3. The effect depends on distance between organisms

This is an honest experiment - we report results whether positive or negative.

Usage:
    python experiment_cascading.py
    python experiment_cascading.py --visualize
    python experiment_cascading.py --distance 30
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json
from datetime import datetime
import argparse

from src.simulation import LeniaSimulation, LeniaConfig


@dataclass
class CascadeTrialResult:
    """Result of a single cascading trial."""
    distance: int
    organism_a_collapsed: bool
    organism_b_collapsed: bool
    time_to_a_collapse: int
    time_to_b_collapse: Optional[int]
    cascade_detected: bool
    max_wave_amplitude: float
    wave_reached_b: bool

    def __repr__(self):
        cascade = "YES" if self.cascade_detected else "NO"
        return f"CascadeResult(d={self.distance}, cascade={cascade})"


class DualOrganismSimulation:
    """
    Simulation with two Lenia organisms.

    Organism A (left) can be killed by perturbation.
    Organism B (right) is monitored for cascade effects.
    """

    MU = 0.15
    SIGMA = 0.015
    GRID_SIZE = 200

    def __init__(self, distance: int = 40, seed: int = 42):
        """
        Args:
            distance: Distance between organism centers
            seed: Random seed
        """
        self.distance = distance
        self.seed = seed

        # Create single large grid
        config = LeniaConfig(
            grid_size=self.GRID_SIZE,
            mu=self.MU,
            sigma=self.SIGMA,
            seed=seed,
            init_type="disk",
            init_size=0.12,  # Smaller to fit two
            init_density=0.5,
        )

        self.sim = LeniaSimulation(config)

        # Position organisms
        self.center_y = self.GRID_SIZE // 2
        self.center_a = self.GRID_SIZE // 2 - distance // 2
        self.center_b = self.GRID_SIZE // 2 + distance // 2

        # Initialize with two blobs
        self._initialize_two_organisms()

        # Tracking
        self.mass_a_history = []
        self.mass_b_history = []
        self.wave_history = []
        self.perturbation_applied = False
        self.perturbation_step = 0

    def _initialize_two_organisms(self):
        """Create two separate organisms on the grid."""
        self.sim.world.fill(0)

        size = self.GRID_SIZE
        yy, xx = np.ogrid[:size, :size]

        # Organism A (left)
        r_a = 12
        dist_a = np.sqrt((xx - self.center_a)**2 + (yy - self.center_y)**2)
        blob_a = np.where(dist_a < r_a, (1 - (dist_a/r_a)**2)**2, 0)
        blob_a *= 0.5

        # Organism B (right)
        dist_b = np.sqrt((xx - self.center_b)**2 + (yy - self.center_y)**2)
        blob_b = np.where(dist_b < r_a, (1 - (dist_b/r_a)**2)**2, 0)
        blob_b *= 0.5

        # Add noise for asymmetry
        rng = np.random.default_rng(self.seed)
        noise = rng.uniform(0.95, 1.05, size=self.sim.world.shape)

        self.sim.world = (blob_a + blob_b) * noise
        self.sim.world = np.clip(self.sim.world, 0, 1)

    def get_mass_a(self) -> float:
        """Get mass of organism A region."""
        half_dist = self.distance // 2
        region = self.sim.world[:, :self.center_a + half_dist//2]
        return float(np.sum(region))

    def get_mass_b(self) -> float:
        """Get mass of organism B region."""
        half_dist = self.distance // 2
        region = self.sim.world[:, self.center_b - half_dist//2:]
        return float(np.sum(region))

    def get_interface_activity(self) -> float:
        """Get activity in the region between organisms."""
        interface_start = self.center_a + 10
        interface_end = self.center_b - 10
        if interface_end <= interface_start:
            return 0.0
        region = self.sim.world[:, interface_start:interface_end]
        return float(np.mean(region))

    def kill_organism_a(self, strength: float = 0.8):
        """Apply lethal perturbation to organism A."""
        size = self.GRID_SIZE
        yy, xx = np.ogrid[:size, :size]

        # Large hole at organism A center
        dist = np.sqrt((xx - self.center_a)**2 + (yy - self.center_y)**2)
        hole = np.exp(-dist**2 / (2 * 10**2))

        self.sim.world = self.sim.world * (1 - hole * strength)
        self.sim.world = np.clip(self.sim.world, 0, 1)

        self.perturbation_applied = True
        self.perturbation_step = self.sim.step_count

    def step(self):
        """Advance simulation one step."""
        self.sim.step()

        # Record histories
        self.mass_a_history.append(self.get_mass_a())
        self.mass_b_history.append(self.get_mass_b())
        self.wave_history.append(self.get_interface_activity())

    def run_warmup(self, steps: int = 80):
        """Run warmup to let organisms stabilize."""
        for _ in range(steps):
            self.step()

    def is_a_collapsed(self, threshold: float = 15) -> bool:
        """Check if organism A has collapsed."""
        if not self.mass_a_history:
            return False
        return self.mass_a_history[-1] < threshold

    def is_b_collapsed(self, threshold: float = 15) -> bool:
        """Check if organism B has collapsed."""
        if not self.mass_b_history:
            return False
        return self.mass_b_history[-1] < threshold


def run_cascade_trial(distance: int, seed: int = 42,
                     warmup: int = 80, observe: int = 200,
                     kill_strength: float = 0.8) -> CascadeTrialResult:
    """
    Run a single cascade trial.

    Args:
        distance: Distance between organisms
        seed: Random seed
        warmup: Warmup steps before killing
        observe: Steps to observe after killing
        kill_strength: How strong the killing perturbation is

    Returns:
        CascadeTrialResult with observations
    """
    sim = DualOrganismSimulation(distance=distance, seed=seed)

    # Warmup
    sim.run_warmup(warmup)

    # Record baseline
    baseline_mass_a = sim.get_mass_a()
    baseline_mass_b = sim.get_mass_b()
    baseline_interface = sim.get_interface_activity()

    # Kill organism A
    sim.kill_organism_a(strength=kill_strength)

    # Observe
    time_to_a_collapse = None
    time_to_b_collapse = None
    max_wave_amplitude = 0

    for t in range(observe):
        sim.step()

        # Check A collapse
        if time_to_a_collapse is None and sim.is_a_collapsed():
            time_to_a_collapse = t

        # Check B collapse
        if time_to_b_collapse is None and sim.is_b_collapsed():
            time_to_b_collapse = t

        # Track wave
        wave = sim.get_interface_activity()
        if wave > max_wave_amplitude:
            max_wave_amplitude = wave

    # Analysis
    a_collapsed = sim.is_a_collapsed()
    b_collapsed = sim.is_b_collapsed()

    # Did wave reach B? (interface activity increased significantly)
    wave_reached_b = max_wave_amplitude > baseline_interface * 1.5

    # Cascade detected if:
    # 1. A collapsed
    # 2. B collapsed AFTER A
    # 3. Wave was detected traveling toward B
    cascade_detected = (
        a_collapsed and
        b_collapsed and
        time_to_b_collapse is not None and
        (time_to_a_collapse is None or time_to_b_collapse > time_to_a_collapse)
    )

    return CascadeTrialResult(
        distance=distance,
        organism_a_collapsed=a_collapsed,
        organism_b_collapsed=b_collapsed,
        time_to_a_collapse=time_to_a_collapse or -1,
        time_to_b_collapse=time_to_b_collapse,
        cascade_detected=cascade_detected,
        max_wave_amplitude=max_wave_amplitude,
        wave_reached_b=wave_reached_b
    )


def run_distance_sweep(distances: List[int] = None,
                       n_trials: int = 10,
                       seed_base: int = 42) -> List[CascadeTrialResult]:
    """
    Sweep over distances to find cascade range.

    Args:
        distances: List of distances to test
        n_trials: Trials per distance
        seed_base: Base random seed

    Returns:
        List of all trial results
    """
    if distances is None:
        distances = [25, 30, 35, 40, 45, 50, 60]

    results = []

    print(f"\n{'='*60}")
    print("CASCADING EXPERIMENT")
    print(f"{'='*60}")
    print(f"Distances: {distances}")
    print(f"Trials per distance: {n_trials}")
    print()

    for d in distances:
        print(f"Distance {d}px: ", end="", flush=True)
        cascade_count = 0
        wave_count = 0

        for trial in range(n_trials):
            seed = seed_base + trial * 100 + d
            result = run_cascade_trial(distance=d, seed=seed)
            results.append(result)

            if result.cascade_detected:
                cascade_count += 1
                print("C", end="", flush=True)
            elif result.wave_reached_b:
                wave_count += 1
                print("w", end="", flush=True)
            else:
                print(".", end="", flush=True)

        print(f" | Cascade: {cascade_count}/{n_trials}, Wave: {wave_count}/{n_trials}")

    return results


def analyze_results(results: List[CascadeTrialResult]) -> dict:
    """Analyze cascade experiment results."""
    # Group by distance
    by_distance = {}
    for r in results:
        if r.distance not in by_distance:
            by_distance[r.distance] = []
        by_distance[r.distance].append(r)

    summary = {
        "total_trials": len(results),
        "distances_tested": sorted(by_distance.keys()),
        "cascade_rate_by_distance": {},
        "wave_rate_by_distance": {},
        "overall_cascade_rate": sum(r.cascade_detected for r in results) / len(results),
        "overall_wave_rate": sum(r.wave_reached_b for r in results) / len(results),
    }

    for d, trials in by_distance.items():
        n = len(trials)
        summary["cascade_rate_by_distance"][d] = sum(r.cascade_detected for r in trials) / n
        summary["wave_rate_by_distance"][d] = sum(r.wave_reached_b for r in trials) / n

    return summary


class CascadeVisualizer:
    """Interactive visualization of cascade experiment."""

    def __init__(self, distance: int = 35):
        self.distance = distance
        self.sim = None
        self.running = True

        # Setup figure
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.patch.set_facecolor('#0d1117')

        self.fig.suptitle(
            'CASCADING EXPERIMENT',
            fontsize=16, fontweight='bold', color='#58a6ff', y=0.97
        )
        self.fig.text(0.5, 0.935,
            f'Distance between organisms: {distance}px',
            ha='center', fontsize=11, color='#8b949e')

        # Layout
        gs = gridspec.GridSpec(2, 2,
                              height_ratios=[2, 1],
                              hspace=0.25, wspace=0.15,
                              left=0.05, right=0.95, top=0.90, bottom=0.1)

        self.ax_sim = self.fig.add_subplot(gs[0, :])
        self.ax_mass = self.fig.add_subplot(gs[1, 0])
        self.ax_wave = self.fig.add_subplot(gs[1, 1])

        # Buttons
        from matplotlib.widgets import Button
        self.btn_kill = Button(
            plt.axes([0.3, 0.02, 0.15, 0.04]),
            'KILL ORGANISM A',
            color='#da3633', hovercolor='#f85149'
        )
        self.btn_reset = Button(
            plt.axes([0.5, 0.02, 0.1, 0.04]),
            'RESET',
            color='#21262d', hovercolor='#30363d'
        )

        self.btn_kill.on_clicked(self.kill_a)
        self.btn_reset.on_clicked(self.reset)

        self.reset(None)

    def reset(self, event):
        """Reset simulation."""
        self.sim = DualOrganismSimulation(distance=self.distance, seed=42)
        self.sim.run_warmup(80)
        self._update_display()

    def kill_a(self, event):
        """Kill organism A."""
        if self.sim and not self.sim.perturbation_applied:
            self.sim.kill_organism_a()

    def _update_display(self):
        """Update all displays."""
        # Main view
        self.ax_sim.clear()
        self.ax_sim.imshow(self.sim.sim.world, cmap='viridis', vmin=0, vmax=1)
        self.ax_sim.axis('off')

        # Mark organisms
        self.ax_sim.axvline(self.sim.center_a, color='#f85149', alpha=0.5, linestyle='--')
        self.ax_sim.axvline(self.sim.center_b, color='#3fb950', alpha=0.5, linestyle='--')
        self.ax_sim.text(self.sim.center_a, 10, 'A', ha='center', fontsize=12,
                        color='#f85149', fontweight='bold')
        self.ax_sim.text(self.sim.center_b, 10, 'B', ha='center', fontsize=12,
                        color='#3fb950', fontweight='bold')

        status_a = "DEAD" if self.sim.is_a_collapsed() else "ALIVE"
        status_b = "DEAD" if self.sim.is_b_collapsed() else "ALIVE"
        color_a = '#f85149' if status_a == "DEAD" else '#3fb950'
        color_b = '#f85149' if status_b == "DEAD" else '#3fb950'

        self.ax_sim.set_title(
            f"Step {self.sim.sim.step_count}  |  A: {status_a}  |  B: {status_b}",
            fontsize=11, color='#c9d1d9'
        )

        # Mass history
        self.ax_mass.clear()
        self.ax_mass.set_facecolor('#161b22')

        if self.sim.mass_a_history:
            x = np.arange(len(self.sim.mass_a_history))
            self.ax_mass.plot(x, self.sim.mass_a_history, color='#f85149',
                             label='Organism A', linewidth=1.5)
            self.ax_mass.plot(x, self.sim.mass_b_history, color='#3fb950',
                             label='Organism B', linewidth=1.5)

            if self.sim.perturbation_applied:
                pert_idx = self.sim.perturbation_step - (self.sim.sim.step_count - len(x))
                if 0 <= pert_idx < len(x):
                    self.ax_mass.axvline(pert_idx, color='#f0883e', linestyle=':', alpha=0.7)

        self.ax_mass.set_xlabel('Step', fontsize=9, color='#8b949e')
        self.ax_mass.set_ylabel('Mass', fontsize=9, color='#8b949e')
        self.ax_mass.legend(fontsize=8, loc='upper right')
        self.ax_mass.tick_params(colors='#8b949e', labelsize=8)

        # Wave/interface activity
        self.ax_wave.clear()
        self.ax_wave.set_facecolor('#161b22')

        if self.sim.wave_history:
            x = np.arange(len(self.sim.wave_history))
            self.ax_wave.fill_between(x, self.sim.wave_history, color='#58a6ff', alpha=0.3)
            self.ax_wave.plot(x, self.sim.wave_history, color='#58a6ff', linewidth=1.5)

        self.ax_wave.set_xlabel('Step', fontsize=9, color='#8b949e')
        self.ax_wave.set_ylabel('Interface Activity', fontsize=9, color='#8b949e')
        self.ax_wave.set_title('Wave Between Organisms', fontsize=10, color='#c9d1d9')
        self.ax_wave.tick_params(colors='#8b949e', labelsize=8)

        self.fig.canvas.draw_idle()

    def update(self, frame):
        """Animation update."""
        if self.running and self.sim:
            self.sim.step()
            self._update_display()

    def run(self):
        """Run visualization."""
        self.ani = FuncAnimation(
            self.fig, self.update,
            interval=50, blit=False, cache_frame_data=False
        )
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Cascading experiment")
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Run interactive visualization')
    parser.add_argument('--distance', '-d', type=int, default=35,
                       help='Distance for visualization (default: 35)')
    parser.add_argument('--sweep', '-s', action='store_true',
                       help='Run full distance sweep experiment')
    parser.add_argument('--trials', '-t', type=int, default=10,
                       help='Trials per distance (default: 10)')
    args = parser.parse_args()

    if args.visualize:
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   CASCADING EXPERIMENT - VISUALIZATION                          ║
║   ────────────────────────────────────                          ║
║                                                                  ║
║   Can organism collapse trigger cascade to neighbor?            ║
║                                                                  ║
║   CLICK "KILL ORGANISM A" to apply lethal perturbation.         ║
║   Watch if the shockwave triggers collapse of organism B.       ║
║                                                                  ║
║   Distance: {args.distance}px                                             ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
        """)
        viz = CascadeVisualizer(distance=args.distance)
        viz.run()

    elif args.sweep:
        results = run_distance_sweep(n_trials=args.trials)
        summary = analyze_results(results)

        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Total trials: {summary['total_trials']}")
        print(f"Overall cascade rate: {summary['overall_cascade_rate']:.1%}")
        print(f"Overall wave detection rate: {summary['overall_wave_rate']:.1%}")
        print()
        print("By distance:")
        for d in summary['distances_tested']:
            cascade = summary['cascade_rate_by_distance'][d]
            wave = summary['wave_rate_by_distance'][d]
            print(f"  {d}px: cascade={cascade:.0%}, wave={wave:.0%}")

        # Save results
        output_path = Path('experiments') / f'cascading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                'summary': summary,
                'trials': [
                    {
                        'distance': r.distance,
                        'cascade_detected': r.cascade_detected,
                        'wave_reached_b': r.wave_reached_b,
                        'a_collapsed': r.organism_a_collapsed,
                        'b_collapsed': r.organism_b_collapsed,
                    }
                    for r in results
                ]
            }, f, indent=2)
        print(f"\nResults saved to: {output_path}")

        # Interpretation for paper
        print(f"\n{'='*60}")
        print("INTERPRETATION FOR PAPER")
        print(f"{'='*60}")
        if summary['overall_cascade_rate'] > 0.3:
            print("POSITIVE RESULT: Cascading observed!")
            print("→ Include in Results section")
        elif summary['overall_wave_rate'] > 0.5:
            print("PARTIAL RESULT: Waves propagate but don't trigger collapse")
            print("→ Discuss as 'signal propagation without cascading'")
        else:
            print("NEGATIVE RESULT: No reliable cascading detected")
            print("→ Report honestly in Limitations section")
            print("→ Suggests organisms are too robust or distance too far")

    else:
        # Default: single trial demo
        print("Running single cascade trial...")
        result = run_cascade_trial(distance=35)
        print(f"\nResult: {result}")
        print(f"  A collapsed: {result.organism_a_collapsed}")
        print(f"  B collapsed: {result.organism_b_collapsed}")
        print(f"  Cascade detected: {result.cascade_detected}")
        print(f"  Wave reached B: {result.wave_reached_b}")

        print("\nUse --visualize for interactive demo")
        print("Use --sweep for full experiment")


if __name__ == '__main__':
    main()
