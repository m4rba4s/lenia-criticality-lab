#!/usr/bin/env python3
"""
Multiple Morphologies NAND Test

Test whether NAND-like behavior generalizes across different
Lenia configurations in the critical regime.

This addresses the "single morphology" limitation in the paper.

Usage:
    python experiment_morphologies.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import json
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple
from scipy.stats import fisher_exact
from tqdm import tqdm

from src.simulation import LeniaSimulation, LeniaConfig


# Critical configurations from phase diagram analysis
# Selected to span the critical regime (|λ| < 0.01)
CRITICAL_CONFIGURATIONS = [
    # (mu, sigma, name) - spanning the critical regime
    (0.141, 0.0142, "low_mu_low_sigma"),      # Lower edge
    (0.145, 0.0150, "mid_mu_low_sigma"),      # Standard-ish
    (0.150, 0.0150, "baseline"),              # Paper baseline
    (0.150, 0.0165, "baseline_higher_sigma"),
    (0.154, 0.0173, "mid_mu_mid_sigma"),
    (0.158, 0.0181, "high_mu_mid_sigma"),     # Near chaos boundary
    (0.158, 0.0165, "high_mu_low_sigma"),
    (0.154, 0.0196, "approaching_chaos"),
]

# Non-critical configurations for comparison
ORDERED_CONFIGURATIONS = [
    (0.150, 0.0100, "ordered_1"),
    (0.140, 0.0120, "ordered_2"),
]

CHAOTIC_CONFIGURATIONS = [
    (0.150, 0.0250, "chaotic_1"),
    (0.158, 0.0212, "chaotic_2"),
]


@dataclass
class MorphologyResult:
    """Result from testing NAND on a specific morphology."""
    mu: float
    sigma: float
    name: str
    regime: str  # critical, ordered, chaotic

    # NAND results
    n_trials: int
    survival_00: int  # [0,0] - no damage
    survival_01: int  # [0,1] - single hole
    survival_10: int  # [1,0] - single hole
    survival_11: int  # [1,1] - double hole

    # Statistics
    single_survival_rate: float  # (01 + 10) / (2 * n_trials)
    double_survival_rate: float  # 11 / n_trials
    nand_gap: float  # single - double
    p_value: float  # Fisher exact test
    is_nand_like: bool  # p < 0.05 and gap > 0

    # Organism viability
    viable_organisms: int  # How many seeds produced living organisms


def apply_hole(world: np.ndarray, cy: int, cx: int,
               strength: float = 0.8, sigma: float = 6.0) -> np.ndarray:
    """Apply Gaussian hole perturbation."""
    size = world.shape[0]
    yy, xx = np.ogrid[:size, :size]
    dist_sq = (xx - cx)**2 + (yy - cy)**2
    hole = np.exp(-dist_sq / (2 * sigma**2))
    return world * (1 - hole * strength)


def find_center(world: np.ndarray) -> Tuple[int, int]:
    """Find organism center of mass."""
    total = np.sum(world)
    if total < 10:
        return world.shape[0] // 2, world.shape[1] // 2
    size = world.shape[0]
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    cy = int(np.sum(yy * world) / total)
    cx = int(np.sum(xx * world) / total)
    return cy, cx


def run_nand_trial(mu: float, sigma: float, input_a: bool, input_b: bool,
                   seed: int, grid_size: int = 128, warmup: int = 100,
                   observe: int = 200, collapse_threshold: float = 25.0,
                   hole_offset: int = 14) -> Tuple[bool, bool]:
    """
    Run single NAND trial.

    Returns:
        (survived, viable): survived = mass > threshold at end,
                           viable = organism was alive after warmup
    """
    config = LeniaConfig(
        grid_size=grid_size,
        mu=mu,
        sigma=sigma,
        seed=seed,
        init_size=0.20,
        init_density=0.5,
    )
    sim = LeniaSimulation(config)

    # Warmup
    for _ in range(warmup):
        sim.step()

    baseline_mass = sim.mass()
    if baseline_mass < collapse_threshold:
        # Dead organism - not viable
        return False, False

    # Find center
    cy, cx = find_center(sim.world)

    # Apply perturbations
    if input_a:
        sim.world = apply_hole(sim.world, cy, cx - hole_offset)
    if input_b:
        sim.world = apply_hole(sim.world, cy, cx + hole_offset)

    # Observe
    for _ in range(observe):
        sim.step()

    survived = sim.mass() > collapse_threshold
    return survived, True


def test_morphology(mu: float, sigma: float, name: str, regime: str,
                    n_trials: int = 20, seed_base: int = 42) -> MorphologyResult:
    """Test NAND behavior on a specific morphology."""

    survival_counts = {
        "00": 0, "01": 0, "10": 0, "11": 0
    }
    viable_count = 0

    inputs = [
        ("00", False, False),
        ("01", False, True),
        ("10", True, False),
        ("11", True, True),
    ]

    for trial in range(n_trials):
        seed = seed_base + trial * 1000

        for input_name, input_a, input_b in inputs:
            survived, viable = run_nand_trial(
                mu, sigma, input_a, input_b, seed
            )
            if viable:
                if input_name == "00":
                    viable_count += 1  # Count once per trial
                if survived:
                    survival_counts[input_name] += 1

    # Compute statistics
    # Only use viable organisms
    if viable_count == 0:
        return MorphologyResult(
            mu=mu, sigma=sigma, name=name, regime=regime,
            n_trials=n_trials,
            survival_00=0, survival_01=0, survival_10=0, survival_11=0,
            single_survival_rate=0, double_survival_rate=0,
            nand_gap=0, p_value=1.0, is_nand_like=False,
            viable_organisms=0
        )

    # Normalize to viable organisms
    s00 = survival_counts["00"]
    s01 = survival_counts["01"]
    s10 = survival_counts["10"]
    s11 = survival_counts["11"]

    single_pooled = s01 + s10
    single_total = 2 * viable_count
    double_total = viable_count

    single_rate = single_pooled / single_total if single_total > 0 else 0
    double_rate = s11 / double_total if double_total > 0 else 0
    gap = single_rate - double_rate

    # Fisher exact test: single vs double
    single_survived = single_pooled
    single_died = single_total - single_survived
    double_survived = s11
    double_died = double_total - double_survived

    if single_total > 0 and double_total > 0:
        table = [[single_survived, single_died], [double_survived, double_died]]
        _, p_value = fisher_exact(table)
    else:
        p_value = 1.0

    is_nand = p_value < 0.05 and gap > 0.2  # Require substantial gap

    return MorphologyResult(
        mu=mu, sigma=sigma, name=name, regime=regime,
        n_trials=n_trials,
        survival_00=s00, survival_01=s01, survival_10=s10, survival_11=s11,
        single_survival_rate=single_rate,
        double_survival_rate=double_rate,
        nand_gap=gap, p_value=p_value, is_nand_like=is_nand,
        viable_organisms=viable_count
    )


def run_morphology_experiment(n_trials: int = 20) -> Dict:
    """Run full morphology generalization experiment."""

    print("\n" + "="*70)
    print("MULTIPLE MORPHOLOGIES NAND TEST")
    print("="*70)
    print("\nTesting NAND generalization across different Lenia configurations")
    print(f"Trials per configuration: {n_trials}")

    all_configs = (
        [(mu, sigma, name, "critical") for mu, sigma, name in CRITICAL_CONFIGURATIONS] +
        [(mu, sigma, name, "ordered") for mu, sigma, name in ORDERED_CONFIGURATIONS] +
        [(mu, sigma, name, "chaotic") for mu, sigma, name in CHAOTIC_CONFIGURATIONS]
    )

    results = []

    for mu, sigma, name, regime in tqdm(all_configs, desc="Testing configurations"):
        result = test_morphology(mu, sigma, name, regime, n_trials)
        results.append(result)

    # Analyze results
    critical_results = [r for r in results if r.regime == "critical" and r.viable_organisms > 0]
    ordered_results = [r for r in results if r.regime == "ordered" and r.viable_organisms > 0]
    chaotic_results = [r for r in results if r.regime == "chaotic" and r.viable_organisms > 0]

    # Summary statistics
    n_critical_nand = sum(1 for r in critical_results if r.is_nand_like)
    n_ordered_nand = sum(1 for r in ordered_results if r.is_nand_like)
    n_chaotic_nand = sum(1 for r in chaotic_results if r.is_nand_like)

    print("\n" + "-"*70)
    print("RESULTS BY REGIME")
    print("-"*70)

    print("\nCRITICAL REGIME:")
    print(f"{'Config':<25} {'Viable':<8} {'Single':<10} {'Double':<10} {'Gap':<10} {'NAND?':<8}")
    print("-"*70)
    for r in critical_results:
        nand_str = "YES" if r.is_nand_like else "no"
        print(f"{r.name:<25} {r.viable_organisms:<8} {r.single_survival_rate:.0%}      "
              f"{r.double_survival_rate:.0%}       {r.nand_gap:+.0%}      {nand_str}")

    print(f"\nSummary: {n_critical_nand}/{len(critical_results)} critical configurations show NAND-like behavior")

    if ordered_results:
        print("\nORDERED REGIME:")
        for r in ordered_results:
            nand_str = "YES" if r.is_nand_like else "no"
            print(f"{r.name:<25} {r.viable_organisms:<8} {r.single_survival_rate:.0%}      "
                  f"{r.double_survival_rate:.0%}       {r.nand_gap:+.0%}      {nand_str}")
        print(f"Summary: {n_ordered_nand}/{len(ordered_results)} ordered configurations show NAND-like behavior")

    if chaotic_results:
        print("\nCHAOTIC REGIME:")
        for r in chaotic_results:
            nand_str = "YES" if r.is_nand_like else "no"
            print(f"{r.name:<25} {r.viable_organisms:<8} {r.single_survival_rate:.0%}      "
                  f"{r.double_survival_rate:.0%}       {r.nand_gap:+.0%}      {nand_str}")
        print(f"Summary: {n_chaotic_nand}/{len(chaotic_results)} chaotic configurations show NAND-like behavior")

    # Key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    avg_critical_gap = np.mean([r.nand_gap for r in critical_results]) if critical_results else 0
    avg_ordered_gap = np.mean([r.nand_gap for r in ordered_results]) if ordered_results else 0

    print(f"\n1. GENERALIZATION ACROSS CRITICAL REGIME:")
    print(f"   {n_critical_nand}/{len(critical_results)} configurations show NAND behavior")
    print(f"   Average gap: {avg_critical_gap:.0%}")

    if len(critical_results) > 0 and n_critical_nand / len(critical_results) > 0.5:
        print("   >>> NAND behavior GENERALIZES across critical regime!")
    else:
        print("   >>> NAND behavior may be configuration-specific")

    print(f"\n2. CRITICALITY REQUIREMENT:")
    if n_ordered_nand == 0 and n_critical_nand > 0:
        print("   >>> Ordered regime does NOT show NAND behavior")
        print("   >>> Criticality appears NECESSARY for NAND mechanism")
    elif n_ordered_nand > 0:
        print("   >>> Some ordered configurations also show NAND-like behavior")
        print("   >>> Criticality may not be strictly necessary")

    # Prepare summary for saving
    summary = {
        "n_trials_per_config": n_trials,
        "timestamp": datetime.now().isoformat(),
        "results": {
            "critical": [
                {
                    "mu": r.mu, "sigma": r.sigma, "name": r.name,
                    "viable": r.viable_organisms,
                    "single_rate": r.single_survival_rate,
                    "double_rate": r.double_survival_rate,
                    "gap": r.nand_gap,
                    "p_value": r.p_value,
                    "is_nand_like": bool(r.is_nand_like)
                }
                for r in critical_results
            ],
            "ordered": [
                {
                    "mu": r.mu, "sigma": r.sigma, "name": r.name,
                    "viable": r.viable_organisms,
                    "single_rate": r.single_survival_rate,
                    "double_rate": r.double_survival_rate,
                    "gap": r.nand_gap,
                    "p_value": r.p_value,
                    "is_nand_like": bool(r.is_nand_like)
                }
                for r in ordered_results
            ],
            "chaotic": [
                {
                    "mu": r.mu, "sigma": r.sigma, "name": r.name,
                    "viable": r.viable_organisms,
                    "single_rate": r.single_survival_rate,
                    "double_rate": r.double_survival_rate,
                    "gap": r.nand_gap,
                    "p_value": r.p_value,
                    "is_nand_like": bool(r.is_nand_like)
                }
                for r in chaotic_results
            ]
        },
        "summary": {
            "n_critical_tested": len(critical_results),
            "n_critical_nand": n_critical_nand,
            "n_ordered_tested": len(ordered_results),
            "n_ordered_nand": n_ordered_nand,
            "n_chaotic_tested": len(chaotic_results),
            "n_chaotic_nand": n_chaotic_nand,
            "avg_critical_gap": avg_critical_gap,
            "avg_ordered_gap": avg_ordered_gap,
            "generalizes_to_critical": bool(n_critical_nand / len(critical_results) > 0.5) if critical_results else False,
            "criticality_necessary": bool(n_ordered_nand == 0 and n_critical_nand > 0)
        }
    }

    return summary


def main():
    """Run morphology experiment and save results."""
    import argparse

    parser = argparse.ArgumentParser(description="Multiple morphologies NAND test")
    parser.add_argument('--trials', '-t', type=int, default=20,
                       help='Trials per configuration (default: 20)')
    args = parser.parse_args()

    summary = run_morphology_experiment(n_trials=args.trials)

    # Save results
    output_dir = Path("experiments")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"morphologies_{timestamp}.json"

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Also save as latest
    latest_path = output_dir / "morphologies_latest.json"
    with open(latest_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Latest results: {latest_path}")


if __name__ == '__main__':
    main()
