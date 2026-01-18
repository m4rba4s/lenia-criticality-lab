#!/usr/bin/env python3
"""
NULL MODEL EXPERIMENT

Compare NAND gate behavior with:
1. Standard symmetric holes (our protocol)
2. Random position holes (null model)
3. Single large hole (control)

This tests whether the SPECIFIC positioning matters,
not just the total damage amount.

Usage:
    python experiment_null_model.py
    python experiment_null_model.py --visualize
    python experiment_null_model.py --trials 50
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
from scipy.stats import fisher_exact, chi2_contingency
import json
from datetime import datetime
import argparse

from src.simulation import LeniaSimulation, LeniaConfig


@dataclass
class NullModelResult:
    """Result of a null model trial."""
    condition: str  # "symmetric", "random", "single_large"
    n_holes: int
    total_damage: float  # Total mass removed
    survived: bool
    final_mass: float
    baseline_mass: float


class NullModelExperiment:
    """
    Test whether hole positioning matters for NAND behavior.
    """

    MU = 0.15
    SIGMA = 0.015
    GRID_SIZE = 128
    WARMUP = 100
    OBSERVE = 200
    COLLAPSE_THRESHOLD = 25

    # Standard NAND parameters (corrected - paper says 0.5 but 0.8 works)
    HOLE_SIGMA = 6
    HOLE_STRENGTH = 0.8
    HOLE_OFFSET = 14

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _create_organism(self) -> LeniaSimulation:
        """Create a fresh organism."""
        config = LeniaConfig(
            grid_size=self.GRID_SIZE,
            mu=self.MU,
            sigma=self.SIGMA,
            seed=self.seed,
            init_size=0.20,
            init_density=0.5,
        )
        sim = LeniaSimulation(config)
        sim.run(self.WARMUP)
        return sim

    def _find_center(self, sim: LeniaSimulation) -> Tuple[int, int]:
        """Find organism center of mass."""
        total = sim.mass()
        if total < 10:
            return self.GRID_SIZE // 2, self.GRID_SIZE // 2

        yy, xx = np.meshgrid(
            np.arange(self.GRID_SIZE),
            np.arange(self.GRID_SIZE),
            indexing='ij'
        )
        cy = int(np.sum(yy * sim.world) / total)
        cx = int(np.sum(xx * sim.world) / total)
        return cy, cx

    def _apply_hole(self, sim: LeniaSimulation, cy: int, cx: int,
                   sigma: float = None, strength: float = None) -> float:
        """Apply a single hole and return mass removed."""
        sigma = sigma or self.HOLE_SIGMA
        strength = strength or self.HOLE_STRENGTH

        before_mass = sim.mass()

        size = self.GRID_SIZE
        yy, xx = np.ogrid[:size, :size]
        dist_sq = (xx - cx)**2 + (yy - cy)**2
        hole = np.exp(-dist_sq / (2 * sigma**2))

        sim.world = sim.world * (1 - hole * strength)
        sim.world = np.clip(sim.world, 0, 1)

        return before_mass - sim.mass()

    def run_symmetric_double(self) -> NullModelResult:
        """
        Standard NAND protocol: two symmetric holes.
        This is the [1,1] input case.
        """
        sim = self._create_organism()
        baseline = sim.mass()
        cy, cx = self._find_center(sim)

        # Two symmetric holes
        damage1 = self._apply_hole(sim, cy, cx - self.HOLE_OFFSET)
        damage2 = self._apply_hole(sim, cy, cx + self.HOLE_OFFSET)
        total_damage = damage1 + damage2

        # Observe
        sim.run(self.OBSERVE)

        return NullModelResult(
            condition="symmetric",
            n_holes=2,
            total_damage=total_damage,
            survived=sim.mass() > self.COLLAPSE_THRESHOLD,
            final_mass=sim.mass(),
            baseline_mass=baseline
        )

    def run_random_double(self) -> NullModelResult:
        """
        Null model: two holes at RANDOM positions within organism.
        Same total damage amount, different positioning.
        """
        sim = self._create_organism()
        baseline = sim.mass()
        cy, cx = self._find_center(sim)

        # Random offsets (within organism radius ~20px)
        angle1 = self.rng.uniform(0, 2 * np.pi)
        angle2 = self.rng.uniform(0, 2 * np.pi)
        r1 = self.rng.uniform(5, 18)
        r2 = self.rng.uniform(5, 18)

        x1 = int(cx + r1 * np.cos(angle1))
        y1 = int(cy + r1 * np.sin(angle1))
        x2 = int(cx + r2 * np.cos(angle2))
        y2 = int(cy + r2 * np.sin(angle2))

        damage1 = self._apply_hole(sim, y1, x1)
        damage2 = self._apply_hole(sim, y2, x2)
        total_damage = damage1 + damage2

        # Observe
        sim.run(self.OBSERVE)

        return NullModelResult(
            condition="random",
            n_holes=2,
            total_damage=total_damage,
            survived=sim.mass() > self.COLLAPSE_THRESHOLD,
            final_mass=sim.mass(),
            baseline_mass=baseline
        )

    def run_single_large(self) -> NullModelResult:
        """
        Control: single large hole with same total damage.
        Tests if it's the NUMBER of holes or the POSITIONING.
        """
        sim = self._create_organism()
        baseline = sim.mass()
        cy, cx = self._find_center(sim)

        # Single larger hole at center
        # Use larger sigma to match total damage of two small holes
        large_sigma = self.HOLE_SIGMA * 1.4
        large_strength = self.HOLE_STRENGTH * 1.3

        total_damage = self._apply_hole(sim, cy, cx,
                                       sigma=large_sigma,
                                       strength=large_strength)

        # Observe
        sim.run(self.OBSERVE)

        return NullModelResult(
            condition="single_large",
            n_holes=1,
            total_damage=total_damage,
            survived=sim.mass() > self.COLLAPSE_THRESHOLD,
            final_mass=sim.mass(),
            baseline_mass=baseline
        )

    def run_symmetric_single(self) -> NullModelResult:
        """
        Standard NAND [0,1] or [1,0]: single hole.
        """
        sim = self._create_organism()
        baseline = sim.mass()
        cy, cx = self._find_center(sim)

        # Single hole at standard position
        total_damage = self._apply_hole(sim, cy, cx - self.HOLE_OFFSET)

        sim.run(self.OBSERVE)

        return NullModelResult(
            condition="symmetric_single",
            n_holes=1,
            total_damage=total_damage,
            survived=sim.mass() > self.COLLAPSE_THRESHOLD,
            final_mass=sim.mass(),
            baseline_mass=baseline
        )


def run_null_model_experiment(n_trials: int = 30,
                             seed_base: int = 42) -> dict:
    """
    Run full null model comparison.

    Returns dict with results for each condition.
    """
    results = {
        "symmetric": [],
        "random": [],
        "single_large": [],
        "symmetric_single": [],
    }

    print(f"\n{'='*60}")
    print("NULL MODEL EXPERIMENT")
    print(f"{'='*60}")
    print(f"Trials per condition: {n_trials}")
    print()

    for trial in range(n_trials):
        seed = seed_base + trial * 100

        exp = NullModelExperiment(seed=seed)

        # Run all conditions
        results["symmetric"].append(exp.run_symmetric_double())
        results["symmetric_single"].append(exp.run_symmetric_single())

        # New seed for random conditions (different randomness)
        exp_random = NullModelExperiment(seed=seed + 50000)
        results["random"].append(exp_random.run_random_double())
        results["single_large"].append(exp_random.run_single_large())

        # Progress
        if (trial + 1) % 10 == 0:
            print(f"  Completed {trial + 1}/{n_trials} trials")

    return results


def analyze_null_model(results: dict) -> dict:
    """Analyze null model results with statistical tests."""

    def survival_rate(condition_results):
        survived = sum(r.survived for r in condition_results)
        total = len(condition_results)
        return survived, total

    summary = {}

    for condition, trials in results.items():
        survived, total = survival_rate(trials)
        avg_damage = np.mean([r.total_damage for r in trials])
        avg_final = np.mean([r.final_mass for r in trials])

        summary[condition] = {
            "survived": survived,
            "total": total,
            "survival_rate": survived / total,
            "avg_damage": avg_damage,
            "avg_final_mass": avg_final,
        }

    # Statistical comparison: symmetric vs random
    sym = results["symmetric"]
    rnd = results["random"]

    sym_survived = sum(r.survived for r in sym)
    sym_died = len(sym) - sym_survived
    rnd_survived = sum(r.survived for r in rnd)
    rnd_died = len(rnd) - rnd_survived

    # Fisher's exact test
    table = [[sym_survived, sym_died], [rnd_survived, rnd_died]]
    odds, p_fisher = fisher_exact(table)

    summary["statistical_test"] = {
        "comparison": "symmetric vs random (double holes)",
        "table": table,
        "fisher_p_value": p_fisher,
        "odds_ratio": odds,
        "significant": p_fisher < 0.05
    }

    return summary


def print_results(summary: dict):
    """Print formatted results."""
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}\n")

    print(f"{'Condition':<20} {'Survived':<12} {'Rate':<10} {'Avg Damage':<12}")
    print("-" * 54)

    for cond in ["symmetric_single", "symmetric", "random", "single_large"]:
        s = summary[cond]
        print(f"{cond:<20} {s['survived']}/{s['total']:<10} "
              f"{s['survival_rate']:.0%}       {s['avg_damage']:.1f}")

    print()
    print("STATISTICAL COMPARISON: Symmetric vs Random double holes")
    print("-" * 54)
    st = summary["statistical_test"]
    print(f"  Fisher's exact test p-value: {st['fisher_p_value']:.4f}")
    print(f"  Odds ratio: {st['odds_ratio']:.2f}")
    print(f"  Significant (p < 0.05): {'YES' if st['significant'] else 'NO'}")

    print()
    print("INTERPRETATION:")
    print("-" * 54)

    sym_rate = summary["symmetric"]["survival_rate"]
    rnd_rate = summary["random"]["survival_rate"]

    if st["significant"] and sym_rate < rnd_rate:
        print("  POSITIONING MATTERS!")
        print(f"  Symmetric positioning causes more collapse ({1-sym_rate:.0%}) than")
        print(f"  random positioning ({1-rnd_rate:.0%}).")
        print("  → This supports the NAND gate mechanism being position-specific.")
    elif st["significant"] and sym_rate > rnd_rate:
        print("  UNEXPECTED: Random holes cause MORE collapse!")
        print("  → This would undermine the NAND mechanism hypothesis.")
    else:
        print("  NO SIGNIFICANT DIFFERENCE")
        print("  → Positioning may not matter as much as total damage.")
        print("  → Consider this a null result for the paper.")


class NullModelVisualizer:
    """Visualize null model comparison."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.exp = NullModelExperiment(seed)
        self.current_condition = None
        self.sim = None
        self.history = []
        self.running = True

        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.patch.set_facecolor('#0d1117')

        self.fig.suptitle(
            'NULL MODEL COMPARISON',
            fontsize=16, fontweight='bold', color='#58a6ff', y=0.97
        )

        gs = gridspec.GridSpec(2, 3,
                              height_ratios=[2, 1],
                              hspace=0.25, wspace=0.2,
                              left=0.05, right=0.95, top=0.90, bottom=0.12)

        self.ax_sim = self.fig.add_subplot(gs[0, :])
        self.ax_mass = self.fig.add_subplot(gs[1, 0])
        self.ax_stats = self.fig.add_subplot(gs[1, 1:])
        self.ax_stats.axis('off')

        # Buttons
        from matplotlib.widgets import Button

        self.btn_symmetric = Button(
            plt.axes([0.1, 0.02, 0.15, 0.04]),
            'SYMMETRIC (NAND)',
            color='#238636', hovercolor='#2ea043'
        )
        self.btn_random = Button(
            plt.axes([0.27, 0.02, 0.15, 0.04]),
            'RANDOM HOLES',
            color='#9e6a03', hovercolor='#bb8009'
        )
        self.btn_single = Button(
            plt.axes([0.44, 0.02, 0.15, 0.04]),
            'SINGLE LARGE',
            color='#1f6feb', hovercolor='#388bfd'
        )
        self.btn_reset = Button(
            plt.axes([0.61, 0.02, 0.1, 0.04]),
            'RESET',
            color='#21262d', hovercolor='#30363d'
        )

        self.btn_symmetric.on_clicked(lambda e: self.run_condition("symmetric"))
        self.btn_random.on_clicked(lambda e: self.run_condition("random"))
        self.btn_single.on_clicked(lambda e: self.run_condition("single_large"))
        self.btn_reset.on_clicked(self.reset)

        self.trial_results = {"symmetric": [], "random": [], "single_large": []}
        self.reset(None)

    def reset(self, event):
        """Reset to fresh organism."""
        self.exp = NullModelExperiment(seed=np.random.randint(0, 100000))
        self.sim = self.exp._create_organism()
        self.history = [self.sim.mass()]
        self.current_condition = None
        self._update_display()

    def run_condition(self, condition: str):
        """Apply perturbation for given condition."""
        self.reset(None)
        self.current_condition = condition
        cy, cx = self.exp._find_center(self.sim)

        if condition == "symmetric":
            self.exp._apply_hole(self.sim, cy, cx - self.exp.HOLE_OFFSET)
            self.exp._apply_hole(self.sim, cy, cx + self.exp.HOLE_OFFSET)
        elif condition == "random":
            angle1 = self.exp.rng.uniform(0, 2 * np.pi)
            angle2 = self.exp.rng.uniform(0, 2 * np.pi)
            r1, r2 = self.exp.rng.uniform(5, 18, 2)
            x1 = int(cx + r1 * np.cos(angle1))
            y1 = int(cy + r1 * np.sin(angle1))
            x2 = int(cx + r2 * np.cos(angle2))
            y2 = int(cy + r2 * np.sin(angle2))
            self.exp._apply_hole(self.sim, y1, x1)
            self.exp._apply_hole(self.sim, y2, x2)
        elif condition == "single_large":
            self.exp._apply_hole(self.sim, cy, cx,
                                sigma=self.exp.HOLE_SIGMA * 1.4,
                                strength=self.exp.HOLE_STRENGTH * 1.3)

    def _update_display(self):
        """Update display."""
        self.ax_sim.clear()
        self.ax_sim.imshow(self.sim.world, cmap='viridis', vmin=0, vmax=1)
        self.ax_sim.axis('off')

        status = "ALIVE" if self.sim.mass() > self.exp.COLLAPSE_THRESHOLD else "COLLAPSED"
        color = '#3fb950' if status == "ALIVE" else '#f85149'
        title = f"Step {self.sim.step_count} | Mass: {self.sim.mass():.0f} | {status}"
        if self.current_condition:
            title = f"{self.current_condition.upper()}: " + title
        self.ax_sim.set_title(title, fontsize=11, color=color)

        # Mass history
        self.ax_mass.clear()
        self.ax_mass.set_facecolor('#161b22')
        if self.history:
            self.ax_mass.plot(self.history, color='#58a6ff', linewidth=1.5)
            self.ax_mass.axhline(self.exp.COLLAPSE_THRESHOLD, color='#f85149',
                                linestyle='--', alpha=0.5)
        self.ax_mass.set_xlabel('Step', fontsize=9, color='#8b949e')
        self.ax_mass.set_ylabel('Mass', fontsize=9, color='#8b949e')
        self.ax_mass.tick_params(colors='#8b949e', labelsize=8)

        # Stats
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        self.ax_stats.text(0.1, 0.9, "TRIAL RESULTS", fontsize=12,
                          color='#58a6ff', transform=self.ax_stats.transAxes,
                          fontweight='bold')

        y = 0.7
        for cond, results in self.trial_results.items():
            if results:
                survived = sum(results)
                total = len(results)
                rate = survived / total
                self.ax_stats.text(0.1, y, f"{cond}: {survived}/{total} ({rate:.0%})",
                                  fontsize=10, color='#c9d1d9',
                                  transform=self.ax_stats.transAxes)
                y -= 0.15

        self.fig.canvas.draw_idle()

    def update(self, frame):
        """Animation update."""
        if self.running and self.sim and self.current_condition:
            self.sim.step()
            self.history.append(self.sim.mass())

            # After 200 steps, record result
            if len(self.history) > 200 and len(self.history) == 201:
                survived = self.sim.mass() > self.exp.COLLAPSE_THRESHOLD
                self.trial_results[self.current_condition].append(survived)

            if len(self.history) > 250:
                self.history = self.history[-250:]

            self._update_display()

    def run(self):
        """Run visualizer."""
        self.ani = FuncAnimation(
            self.fig, self.update,
            interval=50, blit=False, cache_frame_data=False
        )
        plt.show()


def run_regime_comparison(n_trials: int = 20, seed_base: int = 42) -> dict:
    """
    Compare NAND behavior across different dynamical regimes.

    Tests whether criticality is necessary for NAND-like behavior.
    """
    regimes = {
        "critical": {"mu": 0.15, "sigma": 0.015},
        "ordered": {"mu": 0.15, "sigma": 0.010},   # Lower sigma
        "chaotic": {"mu": 0.15, "sigma": 0.025},   # Higher sigma
    }

    results = {regime: {"single": [], "double": []} for regime in regimes}

    print(f"\n{'='*60}")
    print("REGIME COMPARISON EXPERIMENT")
    print(f"{'='*60}")
    print("Testing if criticality is necessary for NAND behavior")
    print()

    for regime_name, params in regimes.items():
        print(f"\nTesting {regime_name} regime (mu={params['mu']}, sigma={params['sigma']})...")

        for trial in range(n_trials):
            seed = seed_base + trial * 100

            # Modify experiment parameters
            exp = NullModelExperiment(seed=seed)
            exp.MU = params["mu"]
            exp.SIGMA = params["sigma"]

            # Run single hole (should survive)
            result_single = exp.run_symmetric_single()
            results[regime_name]["single"].append(result_single.survived)

            # Run double hole (should collapse in critical)
            result_double = exp.run_symmetric_double()
            results[regime_name]["double"].append(result_double.survived)

        single_rate = sum(results[regime_name]["single"]) / n_trials
        double_rate = sum(results[regime_name]["double"]) / n_trials
        print(f"  Single hole survival: {single_rate:.0%}")
        print(f"  Double hole survival: {double_rate:.0%}")
        print(f"  Gap (NAND-like): {single_rate - double_rate:.0%}")

    # Statistical analysis
    summary = {"regimes": {}}

    for regime_name in regimes:
        single = results[regime_name]["single"]
        double = results[regime_name]["double"]

        single_survived = sum(single)
        double_survived = sum(double)

        # Fisher's exact for single vs double within regime
        table = [[single_survived, n_trials - single_survived],
                [double_survived, n_trials - double_survived]]
        _, p_value = fisher_exact(table)

        summary["regimes"][regime_name] = {
            "single_survival": single_survived / n_trials,
            "double_survival": double_survived / n_trials,
            "gap": (single_survived - double_survived) / n_trials,
            "p_value": p_value,
            "nand_like": p_value < 0.05 and single_survived > double_survived
        }

    # Compare critical vs non-critical
    crit_double = sum(results["critical"]["double"])
    ord_double = sum(results["ordered"]["double"])

    table_crit_ord = [[crit_double, n_trials - crit_double],
                      [ord_double, n_trials - ord_double]]
    _, p_crit_vs_ord = fisher_exact(table_crit_ord)

    summary["critical_vs_ordered"] = {
        "p_value": p_crit_vs_ord,
        "significant": p_crit_vs_ord < 0.05
    }

    print(f"\n{'='*60}")
    print("REGIME COMPARISON RESULTS")
    print(f"{'='*60}")

    for regime_name, stats in summary["regimes"].items():
        nand_str = "YES" if stats["nand_like"] else "NO"
        print(f"\n{regime_name.upper()}:")
        print(f"  Single survival: {stats['single_survival']:.0%}")
        print(f"  Double survival: {stats['double_survival']:.0%}")
        print(f"  Gap: {stats['gap']:.0%} (p = {stats['p_value']:.4f})")
        print(f"  NAND-like behavior: {nand_str}")

    print(f"\nCritical vs Ordered (double hole): p = {p_crit_vs_ord:.4f}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Null model experiment")
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Interactive visualization')
    parser.add_argument('--trials', '-t', type=int, default=30,
                       help='Trials per condition (default: 30)')
    parser.add_argument('--regimes', '-r', action='store_true',
                       help='Run regime comparison experiment')
    args = parser.parse_args()

    if args.regimes:
        summary = run_regime_comparison(n_trials=args.trials)
        output_path = Path('experiments') / f'regime_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=float)
        print(f"\nResults saved to: {output_path}")
    elif args.visualize:
        print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   NULL MODEL EXPERIMENT - VISUALIZATION                         ║
║   ─────────────────────────────────────                         ║
║                                                                  ║
║   Compare different hole positioning strategies:                ║
║                                                                  ║
║   SYMMETRIC (NAND): Standard protocol - holes at ±14px         ║
║   RANDOM HOLES: Same damage, random positions                   ║
║   SINGLE LARGE: One big hole at center                          ║
║                                                                  ║
║   If positioning matters, symmetric should cause MORE collapse  ║
║   than random (because it hits repair capacity optimally).      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
        """)
        viz = NullModelVisualizer()
        viz.run()
    else:
        results = run_null_model_experiment(n_trials=args.trials)
        summary = analyze_null_model(results)
        print_results(summary)

        # Save
        output_path = Path('experiments') / f'null_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=float)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
