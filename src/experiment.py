"""
Experiment Framework

Reproducible, structured experiment execution with:
- Parameter sweeps
- Result persistence
- Progress tracking
- Parallel execution support
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings

from .simulation import LeniaSimulation, LeniaConfig
from .metrics import (LyapunovEstimator, SpatialAnalyzer, InformationMetrics,
                      SusceptibilityMeasure, LyapunovResult, CorrelationResult)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    name: str
    description: str = ""

    # Parameter ranges for sweep
    mu_range: tuple = (0.08, 0.25)
    mu_steps: int = 30
    sigma_range: tuple = (0.005, 0.035)
    sigma_steps: int = 30

    # Fixed parameters
    grid_size: int = 128
    kernel_radius: int = 13
    dt: float = 0.1

    # Simulation parameters
    warmup_steps: int = 200
    measure_steps: int = 300

    # Analysis
    compute_lyapunov: bool = True
    lyapunov_trials: int = 3
    compute_correlation: bool = True
    compute_information: bool = True
    compute_susceptibility: bool = False  # Slow

    # Execution
    n_workers: int = 4
    seed: int = 42

    # Output
    output_dir: str = "experiments"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PointResult:
    """Result for a single parameter point."""
    mu: float
    sigma: float

    # Basic outcome
    final_mass: float
    mean_mass: float
    mass_std: float
    mean_activity: float
    classification: str  # "dead", "explosive", "stable", "oscillating", "chaotic"

    # Lyapunov
    lyapunov: Optional[float] = None
    lyapunov_std: Optional[float] = None
    lyapunov_class: Optional[str] = None

    # Correlation
    correlation_type: Optional[str] = None
    correlation_length: Optional[float] = None
    correlation_exponent: Optional[float] = None

    # Information
    entropy: Optional[float] = None
    spatial_entropy: Optional[float] = None
    mutual_info: Optional[float] = None

    # Susceptibility
    susceptibility: Optional[float] = None


def classify_outcome(mass_history: np.ndarray, activity_history: np.ndarray,
                     grid_size: int) -> str:
    """Classify simulation outcome."""
    grid_area = grid_size ** 2

    final_mass = mass_history[-1] if len(mass_history) > 0 else 0
    mean_mass = np.mean(mass_history) if len(mass_history) > 0 else 0
    mass_ratio = mean_mass / grid_area

    if mass_ratio < 0.001:
        return "dead"
    elif mass_ratio > 0.4:
        return "explosive"
    else:
        if len(mass_history) > 10:
            mass_std = np.std(mass_history)
            stability = mass_std / (mean_mass + 1e-8)
            mean_activity = np.mean(activity_history) if len(activity_history) > 0 else 0

            if stability < 0.1:
                if mean_activity > 50:
                    return "oscillating"
                else:
                    return "stable"
            else:
                return "chaotic"
        return "stable"


def run_single_point(args: tuple) -> PointResult:
    """
    Run analysis for a single (mu, sigma) point.
    Designed to be called in parallel.
    """
    mu, sigma, exp_config, point_seed = args

    # Create config for this point
    config = LeniaConfig(
        grid_size=exp_config.grid_size,
        kernel_radius=exp_config.kernel_radius,
        mu=mu,
        sigma=sigma,
        dt=exp_config.dt,
        seed=point_seed,
    )

    # Run simulation and collect basic metrics
    sim = LeniaSimulation(config)

    # Warmup
    sim.run(exp_config.warmup_steps)

    # Measurement
    mass_history = []
    activity_history = []
    prev_world = sim.get_state()

    for _ in range(exp_config.measure_steps):
        sim.step()
        mass_history.append(sim.mass())
        activity_history.append(sim.activity(prev_world))
        prev_world = sim.get_state()

    mass_history = np.array(mass_history)
    activity_history = np.array(activity_history)

    # Basic classification
    classification = classify_outcome(mass_history, activity_history, exp_config.grid_size)

    result = PointResult(
        mu=mu,
        sigma=sigma,
        final_mass=mass_history[-1],
        mean_mass=np.mean(mass_history),
        mass_std=np.std(mass_history),
        mean_activity=np.mean(activity_history),
        classification=classification,
    )

    # Only compute expensive metrics for "alive" states
    if classification not in ["dead", "explosive"]:

        # Lyapunov exponent
        if exp_config.compute_lyapunov:
            try:
                estimator = LyapunovEstimator(
                    warmup_steps=exp_config.warmup_steps,
                    measure_steps=exp_config.measure_steps,
                )
                lyap_result = estimator.estimate(config, n_trials=exp_config.lyapunov_trials)
                result.lyapunov = lyap_result.exponent
                result.lyapunov_std = lyap_result.exponent_std
                result.lyapunov_class = lyap_result.classification
            except Exception as e:
                warnings.warn(f"Lyapunov failed at mu={mu}, sigma={sigma}: {e}")

        # Spatial correlation
        if exp_config.compute_correlation:
            try:
                analyzer = SpatialAnalyzer()
                corr_result = analyzer.analyze(sim.get_state())
                result.correlation_type = corr_result.fit_type
                result.correlation_length = corr_result.correlation_length
                result.correlation_exponent = corr_result.exponent
            except Exception as e:
                warnings.warn(f"Correlation failed at mu={mu}, sigma={sigma}: {e}")

        # Information metrics
        if exp_config.compute_information:
            try:
                info = InformationMetrics()
                metrics = info.complexity_measures(sim.get_state())
                result.entropy = metrics["entropy"]
                result.spatial_entropy = metrics["spatial_entropy_4"]
                result.mutual_info = metrics["mutual_info_10"]
            except Exception as e:
                warnings.warn(f"Information failed at mu={mu}, sigma={sigma}: {e}")

        # Susceptibility (expensive)
        if exp_config.compute_susceptibility:
            try:
                susc = SusceptibilityMeasure()
                result.susceptibility = susc.measure(config)
            except Exception as e:
                warnings.warn(f"Susceptibility failed at mu={mu}, sigma={sigma}: {e}")

    return result


class Experiment:
    """
    Main experiment class for parameter sweeps.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: List[PointResult] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.output_path: Optional[Path] = None

    def _setup_output(self):
        """Create output directory structure."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = Path(self.config.output_dir) / f"{self.config.name}_{timestamp}"
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.output_path / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    def _generate_parameter_grid(self) -> List[tuple]:
        """Generate all parameter combinations."""
        mu_values = np.linspace(*self.config.mu_range, self.config.mu_steps)
        sigma_values = np.linspace(*self.config.sigma_range, self.config.sigma_steps)

        rng = np.random.default_rng(self.config.seed)

        points = []
        for i, mu in enumerate(mu_values):
            for j, sigma in enumerate(sigma_values):
                point_seed = rng.integers(0, 2**31)
                points.append((mu, sigma, self.config, point_seed))

        return points

    def run(self, parallel: bool = True, progress: bool = True) -> pd.DataFrame:
        """
        Run the experiment.

        Args:
            parallel: Use multiprocessing
            progress: Show progress bar

        Returns:
            DataFrame with all results
        """
        self._setup_output()
        self.start_time = datetime.now()

        points = self._generate_parameter_grid()
        total = len(points)

        print(f"Running experiment: {self.config.name}")
        print(f"  Total points: {total}")
        print(f"  Grid: {self.config.mu_steps} x {self.config.sigma_steps}")
        print(f"  Workers: {self.config.n_workers if parallel else 1}")
        print(f"  Output: {self.output_path}")

        if parallel and self.config.n_workers > 1:
            with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
                futures = [executor.submit(run_single_point, p) for p in points]

                iterator = as_completed(futures)
                if progress:
                    iterator = tqdm(iterator, total=total, desc="Computing")

                for future in iterator:
                    try:
                        result = future.result()
                        self.results.append(result)
                    except Exception as e:
                        warnings.warn(f"Point failed: {e}")
        else:
            iterator = points
            if progress:
                iterator = tqdm(iterator, desc="Computing")

            for point in iterator:
                try:
                    result = run_single_point(point)
                    self.results.append(result)
                except Exception as e:
                    warnings.warn(f"Point failed: {e}")

        self.end_time = datetime.now()

        # Convert to DataFrame
        df = pd.DataFrame([asdict(r) for r in self.results])

        # Save results
        df.to_parquet(self.output_path / "results.parquet")
        df.to_csv(self.output_path / "results.csv", index=False)

        # Save summary
        duration = (self.end_time - self.start_time).total_seconds()
        summary = {
            "name": self.config.name,
            "total_points": total,
            "completed_points": len(self.results),
            "duration_seconds": duration,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
        }
        with open(self.output_path / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nExperiment complete!")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Results saved to: {self.output_path}")

        return df


class ExperimentRunner:
    """High-level interface for running experiments."""

    @staticmethod
    def phase_diagram(name: str = "phase_diagram",
                      mu_range: tuple = (0.08, 0.25),
                      sigma_range: tuple = (0.005, 0.035),
                      resolution: int = 30,
                      **kwargs) -> pd.DataFrame:
        """Run standard phase diagram experiment."""
        config = ExperimentConfig(
            name=name,
            description="Phase diagram with criticality metrics",
            mu_range=mu_range,
            sigma_range=sigma_range,
            mu_steps=resolution,
            sigma_steps=resolution,
            **kwargs
        )
        exp = Experiment(config)
        return exp.run()

    @staticmethod
    def lyapunov_scan(name: str = "lyapunov_scan",
                      mu_range: tuple = (0.10, 0.20),
                      sigma_range: tuple = (0.010, 0.025),
                      resolution: int = 20,
                      **kwargs) -> pd.DataFrame:
        """High-resolution Lyapunov exponent scan."""
        config = ExperimentConfig(
            name=name,
            description="High-resolution Lyapunov exponent scan",
            mu_range=mu_range,
            sigma_range=sigma_range,
            mu_steps=resolution,
            sigma_steps=resolution,
            compute_lyapunov=True,
            lyapunov_trials=5,
            compute_correlation=False,
            compute_information=False,
            **kwargs
        )
        exp = Experiment(config)
        return exp.run()

    @staticmethod
    def criticality_line(mu_center: float = 0.15,
                         sigma_range: tuple = (0.008, 0.030),
                         resolution: int = 50,
                         **kwargs) -> pd.DataFrame:
        """
        High-resolution scan along likely critical line.
        Fixed mu, varying sigma.
        """
        config = ExperimentConfig(
            name=f"criticality_mu{mu_center:.2f}",
            description=f"Critical line scan at mu={mu_center}",
            mu_range=(mu_center - 0.001, mu_center + 0.001),
            mu_steps=1,
            sigma_range=sigma_range,
            sigma_steps=resolution,
            compute_lyapunov=True,
            lyapunov_trials=5,
            compute_correlation=True,
            compute_information=True,
            compute_susceptibility=True,
            measure_steps=500,
            **kwargs
        )
        exp = Experiment(config)
        return exp.run()
