#!/usr/bin/env python3
"""
Transfer Entropy Analysis for Signal Propagation

Replaces correlation-based analysis with causal information flow measurement.
This addresses the "correlation != causation" limitation in the paper.

Usage:
    python experiment_transfer_entropy.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from tqdm import tqdm

from src.simulation import LeniaSimulation, LeniaConfig
from src.metrics import TransferEntropyEstimator, compute_transfer_entropy_matrix


@dataclass
class TEExperimentConfig:
    """Configuration for transfer entropy experiment."""
    # Critical organism parameters (baseline from paper)
    mu: float = 0.15
    sigma: float = 0.015
    grid_size: int = 128

    # Experiment parameters
    n_trials: int = 20
    warmup_steps: int = 100
    pre_perturbation_steps: int = 40  # Baseline recording
    post_perturbation_steps: int = 200  # After perturbation (longer for TE)
    measure_interval: int = 1  # More frequent sampling

    # Probe configuration
    n_probes: int = 6
    probe_size: int = 8
    probe_span: int = 35  # Horizontal spread (within organism)

    # Perturbation (moderate signal - enough to propagate, not enough to kill)
    perturbation_strength: float = 0.15
    perturbation_sigma: float = 5.0

    # Transfer entropy parameters
    te_bins: int = 8
    te_history: int = 3
    te_surrogates: int = 100


@dataclass
class TEResult:
    """Results from a single trial."""
    trial_id: int
    seed: int
    probe_histories: List[List[float]]
    te_matrix: List[List[float]]
    significance_matrix: List[List[float]]
    correlations: Dict[str, float]
    organism_survived: bool


def run_single_trial(config: TEExperimentConfig, trial_id: int, seed: int) -> TEResult:
    """Run single transfer entropy trial."""
    # Initialize simulation
    sim_config = LeniaConfig(
        grid_size=config.grid_size,
        mu=config.mu,
        sigma=config.sigma,
        seed=seed,
        init_size=0.20,
        init_density=0.5,
    )
    sim = LeniaSimulation(sim_config)

    # Warmup
    for _ in range(config.warmup_steps):
        sim.step()

    # Find organism center
    total = sim.mass()
    if total < 10:
        # Dead organism
        return TEResult(
            trial_id=trial_id,
            seed=seed,
            probe_histories=[[] for _ in range(config.n_probes)],
            te_matrix=[],
            significance_matrix=[],
            correlations={},
            organism_survived=False
        )

    size = config.grid_size
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    cy = int(np.sum(yy * sim.world) / total)
    cx = int(np.sum(xx * sim.world) / total)

    # Setup probes
    x_positions = np.linspace(
        cx - config.probe_span // 2,
        cx + config.probe_span // 2,
        config.n_probes
    ).astype(int)
    probe_positions = [(cy, x) for x in x_positions]

    def get_probe_value(sim, py, px):
        half = config.probe_size // 2
        y_start = max(0, py - half)
        y_end = min(size, py + half)
        x_start = max(0, px - half)
        x_end = min(size, px + half)
        return np.mean(sim.world[y_start:y_end, x_start:x_end])

    # Initialize histories
    probe_histories = [[] for _ in range(config.n_probes)]

    # Record baseline (pre-perturbation)
    for _ in range(config.pre_perturbation_steps // config.measure_interval):
        for _ in range(config.measure_interval):
            sim.step()
        for i, (py, px) in enumerate(probe_positions):
            probe_histories[i].append(get_probe_value(sim, py, px))

    # Apply perturbation at first probe
    py, px = probe_positions[0]
    yy_grid, xx_grid = np.ogrid[:size, :size]
    dist_sq = (xx_grid - px)**2 + (yy_grid - py)**2
    perturbation = config.perturbation_strength * np.exp(
        -dist_sq / (2 * config.perturbation_sigma**2)
    )
    sim.world = np.clip(sim.world + perturbation, 0, 1)

    # Record post-perturbation
    for _ in range(config.post_perturbation_steps // config.measure_interval):
        for _ in range(config.measure_interval):
            sim.step()
        for i, (py, px) in enumerate(probe_positions):
            probe_histories[i].append(get_probe_value(sim, py, px))

    # Check survival
    organism_survived = sim.mass() > 25

    # Compute transfer entropy matrix
    te_estimator = TransferEntropyEstimator(
        n_bins=config.te_bins,
        history_length=config.te_history,
        lag=1,
        n_surrogates=config.te_surrogates
    )

    te_matrix, sig_matrix = compute_transfer_entropy_matrix(
        probe_histories,
        n_bins=config.te_bins,
        history_length=config.te_history
    )

    # Compute lagged correlations for comparison
    correlations = {}
    for i in range(config.n_probes - 1):
        h1 = np.array(probe_histories[i])
        h2 = np.array(probe_histories[i + 1])
        lag = 3
        if len(h1) > lag + 5:
            x1 = h1[:-lag]
            x2 = h2[lag:]
            min_len = min(len(x1), len(x2))
            x1, x2 = x1[:min_len], x2[:min_len]
            if np.std(x1) > 1e-6 and np.std(x2) > 1e-6:
                correlations[f"P{i+1}->P{i+2}"] = float(np.corrcoef(x1, x2)[0, 1])

    return TEResult(
        trial_id=trial_id,
        seed=seed,
        probe_histories=[list(h) for h in probe_histories],
        te_matrix=te_matrix.tolist(),
        significance_matrix=sig_matrix.tolist(),
        correlations=correlations,
        organism_survived=organism_survived
    )


def run_experiment(config: TEExperimentConfig) -> Dict:
    """Run full transfer entropy experiment."""
    print("\n" + "="*60)
    print("TRANSFER ENTROPY ANALYSIS")
    print("="*60)
    print(f"\nParameters:")
    print(f"  Organism: mu={config.mu}, sigma={config.sigma}")
    print(f"  Trials: {config.n_trials}")
    print(f"  Probes: {config.n_probes}")
    print(f"  TE bins: {config.te_bins}, history: {config.te_history}")
    print()

    results = []
    base_seed = 42

    for trial in tqdm(range(config.n_trials), desc="Running trials"):
        seed = base_seed + trial * 1000
        result = run_single_trial(config, trial, seed)
        results.append(result)

    # Aggregate results
    survived_results = [r for r in results if r.organism_survived]
    n_survived = len(survived_results)

    print(f"\n{n_survived}/{config.n_trials} trials with surviving organisms")

    if n_survived == 0:
        return {"error": "No surviving organisms"}

    # Aggregate TE matrices
    te_matrices = np.array([r.te_matrix for r in survived_results])
    sig_matrices = np.array([r.significance_matrix for r in survived_results])

    mean_te = np.mean(te_matrices, axis=0)
    std_te = np.std(te_matrices, axis=0)
    mean_sig = np.mean(sig_matrices, axis=0)

    # Adjacent probe TE (forward direction)
    forward_te = []
    forward_sig = []
    for i in range(config.n_probes - 1):
        forward_te.append(mean_te[i, i+1])
        forward_sig.append(mean_sig[i, i+1])

    # Backward TE (should be lower)
    backward_te = []
    for i in range(1, config.n_probes):
        backward_te.append(mean_te[i, i-1])

    # Aggregate correlations
    all_corrs = {}
    for r in survived_results:
        for k, v in r.correlations.items():
            if k not in all_corrs:
                all_corrs[k] = []
            all_corrs[k].append(v)

    mean_corrs = {k: float(np.mean(v)) for k, v in all_corrs.items()}
    std_corrs = {k: float(np.std(v)) for k, v in all_corrs.items()}

    # Summary
    summary = {
        "config": asdict(config),
        "n_trials": config.n_trials,
        "n_survived": n_survived,
        "mean_te_matrix": mean_te.tolist(),
        "std_te_matrix": std_te.tolist(),
        "mean_significance_matrix": mean_sig.tolist(),
        "forward_te": {
            f"P{i+1}->P{i+2}": {
                "te": forward_te[i],
                "significance": forward_sig[i]
            }
            for i in range(len(forward_te))
        },
        "backward_te": {
            f"P{i+2}->P{i+1}": backward_te[i]
            for i in range(len(backward_te))
        },
        "net_flow": {
            f"P{i+1}-P{i+2}": forward_te[i] - backward_te[i]
            for i in range(len(forward_te))
        },
        "correlations": {
            "mean": mean_corrs,
            "std": std_corrs
        },
        "timestamp": datetime.now().isoformat()
    }

    # Print results
    print("\n" + "-"*60)
    print("TRANSFER ENTROPY RESULTS")
    print("-"*60)

    print("\nForward TE (P_i -> P_{i+1}):")
    for i in range(len(forward_te)):
        sig_str = "*" if forward_sig[i] < 0.05 else ""
        print(f"  P{i+1} -> P{i+2}: TE = {forward_te[i]:.4f} (p = {forward_sig[i]:.3f}){sig_str}")

    print("\nBackward TE (P_{i+1} -> P_i):")
    for i in range(len(backward_te)):
        print(f"  P{i+2} -> P{i+1}: TE = {backward_te[i]:.4f}")

    print("\nNet information flow (Forward - Backward):")
    for i in range(len(forward_te)):
        net = forward_te[i] - backward_te[i]
        direction = ">>>" if net > 0.01 else ("<<<" if net < -0.01 else "---")
        print(f"  P{i+1} <-> P{i+2}: {net:+.4f} {direction}")

    print("\nCorrelations (for comparison):")
    for k, v in mean_corrs.items():
        print(f"  {k}: r = {v:.3f} +/- {std_corrs[k]:.3f}")

    # Key finding
    mean_forward_te = np.mean(forward_te)
    mean_backward_te = np.mean(backward_te)
    mean_net_flow = mean_forward_te - mean_backward_te

    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    print(f"Mean forward TE:  {mean_forward_te:.4f}")
    print(f"Mean backward TE: {mean_backward_te:.4f}")
    print(f"Net flow:         {mean_net_flow:+.4f}")

    if mean_net_flow > 0.01:
        print("\n>>> CAUSAL INFORMATION FLOW CONFIRMED: P1 -> P2 -> ... -> P6")
    elif mean_net_flow < -0.01:
        print("\n<<< REVERSE FLOW DETECTED (unexpected)")
    else:
        print("\n--- No significant directional flow")

    # Significance
    n_significant = sum(1 for s in forward_sig if s < 0.05)
    print(f"\n{n_significant}/{len(forward_sig)} forward TE values significant (p < 0.05)")

    return summary


def main():
    """Run transfer entropy experiment and save results."""
    config = TEExperimentConfig(
        n_trials=20,
        te_surrogates=100
    )

    summary = run_experiment(config)

    # Save results
    output_dir = Path("experiments")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"transfer_entropy_{timestamp}.json"

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Also update latest results
    latest_path = output_dir / "transfer_entropy_latest.json"
    with open(latest_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Latest results: {latest_path}")


if __name__ == '__main__':
    main()
