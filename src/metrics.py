"""
Criticality Metrics for Lenia

Implements measures to detect edge-of-chaos dynamics:
- Lyapunov exponents (sensitivity to initial conditions)
- Spatial correlation functions (power-law vs exponential decay)
- Mutual information (information-theoretic complexity)
- Susceptibility (response to perturbations)
"""

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.optimize import curve_fit
from scipy.stats import entropy as scipy_entropy
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import warnings

from .simulation import LeniaSimulation, LeniaConfig, create_perturbation_pair


@dataclass
class LyapunovResult:
    """Results from Lyapunov exponent estimation."""
    exponent: float                    # Estimated λ
    exponent_std: float               # Standard error
    convergence_history: np.ndarray   # λ estimates over time
    classification: str               # "ordered", "critical", "chaotic"

    def __repr__(self):
        return f"LyapunovResult(λ={self.exponent:.4f}±{self.exponent_std:.4f}, {self.classification})"


class LyapunovEstimator:
    """
    Estimate largest Lyapunov exponent using Benettin algorithm.

    The Lyapunov exponent measures sensitivity to initial conditions:
    - λ < 0: Ordered (perturbations decay)
    - λ ≈ 0: Critical (marginal stability)
    - λ > 0: Chaotic (perturbations grow)

    Algorithm:
    1. Evolve reference and perturbed trajectories in parallel
    2. Periodically renormalize the perturbation to prevent overflow
    3. Accumulate log of stretching factors
    4. Average to get λ
    """

    def __init__(self,
                 epsilon: float = 1e-8,
                 renorm_interval: int = 10,
                 warmup_steps: int = 100,
                 measure_steps: int = 500,
                 critical_threshold: float = 0.01):
        """
        Args:
            epsilon: Initial perturbation size
            renorm_interval: Steps between renormalizations
            warmup_steps: Transient steps before measuring
            measure_steps: Steps for measurement
            critical_threshold: |λ| below this is considered critical
        """
        self.epsilon = epsilon
        self.renorm_interval = renorm_interval
        self.warmup_steps = warmup_steps
        self.measure_steps = measure_steps
        self.critical_threshold = critical_threshold

    def estimate(self, config: LeniaConfig, n_trials: int = 3) -> LyapunovResult:
        """
        Estimate Lyapunov exponent with multiple trials for statistics.

        Args:
            config: Lenia configuration
            n_trials: Number of independent trials

        Returns:
            LyapunovResult with estimate and uncertainty
        """
        all_exponents = []

        for trial in range(n_trials):
            # Create fresh config with different seed for each trial
            trial_config = LeniaConfig(**config.to_dict())
            trial_config.seed = (config.seed or 0) + trial * 1000

            exponent, history = self._single_trial(trial_config)
            all_exponents.append(exponent)

        mean_exp = np.mean(all_exponents)
        std_exp = np.std(all_exponents) / np.sqrt(n_trials)

        # Classification
        if mean_exp < -self.critical_threshold:
            classification = "ordered"
        elif mean_exp > self.critical_threshold:
            classification = "chaotic"
        else:
            classification = "critical"

        return LyapunovResult(
            exponent=mean_exp,
            exponent_std=std_exp,
            convergence_history=history,
            classification=classification
        )

    def _single_trial(self, config: LeniaConfig) -> Tuple[float, np.ndarray]:
        """Run single Lyapunov estimation trial."""
        # Create reference and perturbed simulations
        ref, pert, delta = create_perturbation_pair(config, self.epsilon)

        # Warmup: let transients die out
        ref.run(self.warmup_steps)
        pert.run(self.warmup_steps)

        # Measurement phase
        log_stretches = []
        n_renorms = self.measure_steps // self.renorm_interval

        for _ in range(n_renorms):
            # Evolve both
            for _ in range(self.renorm_interval):
                ref.step()
                pert.step()

            # Compute separation
            delta = pert.world - ref.world
            delta_norm = np.linalg.norm(delta)

            if delta_norm < 1e-15:
                # Perturbation collapsed - ordered regime
                log_stretches.append(-20)  # Very negative
            else:
                # Log of stretching factor
                log_stretch = np.log(delta_norm / self.epsilon)
                log_stretches.append(log_stretch)

                # Renormalize: rescale perturbation to original size
                delta = delta / delta_norm * self.epsilon
                pert.world = ref.world + delta
                pert.world = np.clip(pert.world, 0, 1)

        # Compute running average for convergence history
        log_stretches = np.array(log_stretches)
        cumsum = np.cumsum(log_stretches)
        n_steps = np.arange(1, len(log_stretches) + 1) * self.renorm_interval
        history = cumsum / n_steps  # Running average λ

        # Final estimate
        exponent = np.mean(log_stretches) / self.renorm_interval

        return exponent, history


@dataclass
class CorrelationResult:
    """Results from spatial correlation analysis."""
    distances: np.ndarray           # r values
    correlations: np.ndarray        # C(r) values
    fit_type: str                   # "power_law" or "exponential"
    fit_params: Dict[str, float]    # Fit parameters
    correlation_length: float       # ξ (finite for exponential, inf for power-law)
    exponent: Optional[float]       # η for power-law, decay rate for exponential


class SpatialAnalyzer:
    """
    Analyze spatial structure and correlations.

    At criticality:
    - Correlation function C(r) ~ r^(-η) (power-law)
    - Correlation length ξ → ∞

    Off-critical:
    - C(r) ~ exp(-r/ξ) (exponential decay)
    - Finite correlation length ξ
    """

    def __init__(self, max_r: Optional[int] = None, n_bins: int = 50):
        """
        Args:
            max_r: Maximum distance to compute (default: grid_size/4)
            n_bins: Number of distance bins
        """
        self.max_r = max_r
        self.n_bins = n_bins

    def correlation_function(self, world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute radial correlation function C(r).

        C(r) = <s(x) * s(x+r)> - <s>²

        Uses FFT for efficient computation.
        """
        # Subtract mean
        mean_s = np.mean(world)
        fluctuation = world - mean_s

        # Autocorrelation via FFT
        fft = np.fft.fft2(fluctuation)
        autocorr = np.real(np.fft.ifft2(fft * np.conj(fft))) / world.size

        # Radial average
        size = world.shape[0]
        max_r = self.max_r or size // 4

        center = size // 2
        y, x = np.ogrid[:size, :size]

        # Shift autocorrelation to center
        autocorr = np.fft.fftshift(autocorr)

        # Compute distances from center
        dist = np.sqrt((x - center)**2 + (y - center)**2)

        # Bin by distance
        bins = np.linspace(0, max_r, self.n_bins + 1)
        distances = (bins[:-1] + bins[1:]) / 2
        correlations = np.zeros(self.n_bins)

        for i in range(self.n_bins):
            mask = (dist >= bins[i]) & (dist < bins[i+1])
            if np.any(mask):
                correlations[i] = np.mean(autocorr[mask])

        return distances, correlations

    def fit_correlation(self, distances: np.ndarray, correlations: np.ndarray
                        ) -> CorrelationResult:
        """
        Fit correlation function to power-law or exponential.

        Compares:
        - Power-law: C(r) = A * r^(-η)
        - Exponential: C(r) = A * exp(-r/ξ)
        """
        # Remove zero distance and negative correlations for fitting
        mask = (distances > 0) & (correlations > 0)
        r = distances[mask]
        c = correlations[mask]

        if len(r) < 5:
            return CorrelationResult(
                distances=distances,
                correlations=correlations,
                fit_type="insufficient_data",
                fit_params={},
                correlation_length=0,
                exponent=None
            )

        # Try power-law fit: log(C) = log(A) - η*log(r)
        try:
            log_r, log_c = np.log(r), np.log(c)
            power_coeffs = np.polyfit(log_r, log_c, 1)
            power_eta = -power_coeffs[0]
            power_A = np.exp(power_coeffs[1])
            power_pred = power_A * r**(-power_eta)
            power_residual = np.sum((c - power_pred)**2)
        except Exception:
            power_residual = np.inf
            power_eta, power_A = 0, 0

        # Try exponential fit: log(C) = log(A) - r/ξ
        try:
            def exp_func(r, A, xi):
                return A * np.exp(-r / xi)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exp_params, _ = curve_fit(exp_func, r, c,
                                          p0=[c[0], r[-1]/2],
                                          bounds=([0, 0.1], [np.inf, r[-1]*10]),
                                          maxfev=5000)
            exp_A, exp_xi = exp_params
            exp_pred = exp_func(r, exp_A, exp_xi)
            exp_residual = np.sum((c - exp_pred)**2)
        except Exception:
            exp_residual = np.inf
            exp_A, exp_xi = 0, 1

        # Choose better fit
        if power_residual < exp_residual and power_eta > 0:
            return CorrelationResult(
                distances=distances,
                correlations=correlations,
                fit_type="power_law",
                fit_params={"A": power_A, "eta": power_eta},
                correlation_length=np.inf,  # Diverging for power-law
                exponent=power_eta
            )
        else:
            return CorrelationResult(
                distances=distances,
                correlations=correlations,
                fit_type="exponential",
                fit_params={"A": exp_A, "xi": exp_xi},
                correlation_length=exp_xi,
                exponent=1/exp_xi if exp_xi > 0 else np.inf
            )

    def analyze(self, world: np.ndarray) -> CorrelationResult:
        """Full correlation analysis."""
        distances, correlations = self.correlation_function(world)
        return self.fit_correlation(distances, correlations)


class InformationMetrics:
    """
    Information-theoretic measures of complexity.

    - Entropy: uncertainty/disorder
    - Mutual Information: statistical dependencies
    - Block Entropy: scaling of entropy with block size
    """

    def __init__(self, n_bins: int = 50):
        """
        Args:
            n_bins: Number of bins for discretizing continuous values
        """
        self.n_bins = n_bins

    def entropy(self, world: np.ndarray) -> float:
        """
        Compute Shannon entropy of the state distribution.

        H = -Σ p(s) log₂ p(s)
        """
        # Discretize
        bins = np.linspace(0, 1, self.n_bins + 1)
        hist, _ = np.histogram(world.flatten(), bins=bins, density=True)
        hist = hist / hist.sum()  # Normalize

        # Filter zeros
        hist = hist[hist > 0]

        return float(scipy_entropy(hist, base=2))

    def spatial_entropy(self, world: np.ndarray, block_size: int = 4) -> float:
        """
        Block entropy: entropy of coarse-grained blocks.

        Higher = more spatial disorder
        """
        # Coarse-grain by averaging blocks
        size = world.shape[0]
        n_blocks = size // block_size

        coarse = np.zeros((n_blocks, n_blocks))
        for i in range(n_blocks):
            for j in range(n_blocks):
                block = world[i*block_size:(i+1)*block_size,
                              j*block_size:(j+1)*block_size]
                coarse[i, j] = np.mean(block)

        return self.entropy(coarse)

    def mutual_information(self, world: np.ndarray, separation: int = 10) -> float:
        """
        Mutual information between spatially separated regions.

        I(X;Y) = H(X) + H(Y) - H(X,Y)

        At criticality, MI decays slowly with distance.
        """
        size = world.shape[0]

        # Take two strips separated by `separation`
        strip_width = size // 8
        x1 = world[:, :strip_width].flatten()
        x2 = world[:, separation:separation+strip_width].flatten()

        # Discretize to 2D histogram
        bins = np.linspace(0, 1, self.n_bins + 1)

        h1, _ = np.histogram(x1, bins=bins, density=True)
        h2, _ = np.histogram(x2, bins=bins, density=True)
        h12, _, _ = np.histogram2d(x1, x2, bins=[bins, bins], density=True)

        # Normalize
        h1 = h1 / h1.sum() + 1e-10
        h2 = h2 / h2.sum() + 1e-10
        h12 = h12 / h12.sum() + 1e-10

        # Entropies
        H1 = float(scipy_entropy(h1, base=2))
        H2 = float(scipy_entropy(h2, base=2))
        H12 = float(scipy_entropy(h12.flatten(), base=2))

        return H1 + H2 - H12

    def complexity_measures(self, world: np.ndarray) -> Dict[str, float]:
        """Compute all information metrics."""
        return {
            "entropy": self.entropy(world),
            "spatial_entropy_4": self.spatial_entropy(world, 4),
            "spatial_entropy_8": self.spatial_entropy(world, 8),
            "mutual_info_10": self.mutual_information(world, 10),
            "mutual_info_20": self.mutual_information(world, 20),
        }


class SusceptibilityMeasure:
    """
    Measure susceptibility: response to external perturbations.

    At criticality, susceptibility diverges.
    χ = d<m>/dh where h is external field strength.
    """

    def __init__(self, field_strengths: List[float] = None):
        self.field_strengths = field_strengths or [0.001, 0.005, 0.01, 0.02]

    def measure(self, config: LeniaConfig, warmup: int = 100, measure: int = 50) -> float:
        """
        Estimate susceptibility by measuring mass response to small fields.
        """
        base_masses = []
        perturbed_masses = []

        for h in self.field_strengths:
            # Baseline
            sim = LeniaSimulation(config)
            sim.run(warmup)

            base_mass = []
            for _ in range(measure):
                sim.step()
                base_mass.append(sim.mass_ratio())

            # With field: add constant to world each step
            sim2 = LeniaSimulation(config)
            sim2.run(warmup)

            pert_mass = []
            for _ in range(measure):
                sim2.world = np.clip(sim2.world + h * 0.1, 0, 1)
                sim2.step()
                pert_mass.append(sim2.mass_ratio())

            base_masses.append(np.mean(base_mass))
            perturbed_masses.append(np.mean(pert_mass))

        # Estimate susceptibility as slope
        base_masses = np.array(base_masses)
        perturbed_masses = np.array(perturbed_masses)
        response = perturbed_masses - base_masses[0]  # Change from baseline
        h_arr = np.array(self.field_strengths)

        # Linear fit: response = χ * h
        if np.std(response) > 1e-10:
            chi = np.polyfit(h_arr, response, 1)[0]
        else:
            chi = 0

        return float(chi)
