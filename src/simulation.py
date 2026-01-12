"""
Core Lenia Simulation Engine

Headless, research-focused implementation optimized for:
- Reproducibility (fixed seeds, serializable configs)
- Performance (optional numba JIT, GPU support)
- Analysis (clean state access, perturbation support)
"""

import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Callable, Dict, Any
import json
import hashlib


@dataclass
class LeniaConfig:
    """
    Complete specification of a Lenia simulation.
    Fully serializable for reproducibility.
    """
    # Grid
    grid_size: int = 128

    # Kernel parameters
    kernel_radius: int = 13
    kernel_type: str = "gaussian_ring"  # gaussian_ring, polynomial, multi_ring
    kernel_peak: float = 0.5            # For ring kernel: peak position in [0,1]
    kernel_width: float = 0.23          # For ring kernel: width of the ring

    # Growth function
    mu: float = 0.15                    # Growth center
    sigma: float = 0.015                # Growth width
    growth_type: str = "gaussian"       # gaussian, polynomial

    # Dynamics
    dt: float = 0.1                     # Time step

    # Initial condition
    seed: Optional[int] = None          # Random seed for reproducibility
    init_type: str = "disk"             # disk, noise, ring, orbium, custom
    init_size: float = 0.20             # Relative size of initial pattern
    init_density: float = 0.5           # Initial density (calibrated for mu=0.15)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LeniaConfig':
        return cls(**d)

    def hash(self) -> str:
        """Unique hash for this configuration."""
        return hashlib.md5(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'LeniaConfig':
        with open(path) as f:
            return cls.from_dict(json.load(f))


class LeniaSimulation:
    """
    Core Lenia simulation.

    Design principles:
    - Stateless kernel/growth (can be JIT compiled)
    - Clean state access for analysis
    - Perturbation support for Lyapunov calculation
    """

    def __init__(self, config: LeniaConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # State
        self.world = np.zeros((config.grid_size, config.grid_size), dtype=np.float64)
        self.t = 0  # Simulation time
        self.step_count = 0

        # Precompute kernel
        self.kernel = self._make_kernel()

        # Growth function
        self.growth_func = self._get_growth_function()

        # Initialize
        self._initialize()

    def _make_kernel(self) -> np.ndarray:
        """Build convolution kernel."""
        r = self.config.kernel_radius
        size = 2 * r + 1
        x = np.linspace(-1, 1, size)
        xx, yy = np.meshgrid(x, x)
        dist = np.sqrt(xx**2 + yy**2)

        if self.config.kernel_type == "gaussian_ring":
            peak = self.config.kernel_peak
            width = self.config.kernel_width
            kernel = np.exp(-0.5 * ((dist - peak) / width)**2)
            kernel[dist > 1] = 0

        elif self.config.kernel_type == "polynomial":
            alpha = 4.0
            kernel = np.where(dist < 1, (1 - dist**alpha)**alpha, 0)
            kernel[r, r] = 0  # No self-influence

        else:
            raise ValueError(f"Unknown kernel type: {self.config.kernel_type}")

        # Normalize
        kernel = kernel / kernel.sum()
        return kernel

    def _get_growth_function(self) -> Callable:
        """Return growth function."""
        if self.config.growth_type == "gaussian":
            def growth(U, mu, sigma):
                return 2 * np.exp(-((U - mu)**2) / (2 * sigma**2)) - 1
            return growth
        elif self.config.growth_type == "polynomial":
            def growth(U, mu, sigma):
                x = (U - mu) / sigma
                return np.where(np.abs(x) < 1, 2 * (1 - x**2)**2 - 1, -1)
            return growth
        else:
            raise ValueError(f"Unknown growth type: {self.config.growth_type}")

    def _initialize(self):
        """Set initial condition."""
        size = self.config.grid_size
        center = size // 2
        init_r = int(size * self.config.init_size / 2)

        yy, xx = np.ogrid[:size, :size]
        dist = np.sqrt((xx - center)**2 + (yy - center)**2)

        # For Lenia, initial patterns need to produce potentials close to mu
        # This requires carefully calibrated densities

        if self.config.init_type == "disk":
            # Smooth disk with gradient edge - Orbium-like
            r_norm = dist / init_r
            # Smooth bump that goes to 0 at edges
            values = np.where(r_norm < 1, (1 - r_norm**2)**2, 0)
            # Add subtle noise for symmetry breaking
            noise = self.rng.uniform(0.9, 1.1, size=self.world.shape)
            self.world = values * self.config.init_density * noise
            self.world = np.clip(self.world, 0, 1)

        elif self.config.init_type == "ring":
            # Ring pattern - good for some Lenia species
            r_norm = dist / init_r
            # Gaussian ring at r=0.7
            ring = np.exp(-((r_norm - 0.7) / 0.15)**2)
            ring[dist > init_r] = 0
            self.world = ring * self.config.init_density

        elif self.config.init_type == "noise":
            # Smooth noise blob
            noise = self.rng.uniform(0, 1, (size, size))
            noise = gaussian_filter(noise, sigma=init_r/4)
            noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
            mask = dist < init_r
            # Taper at edges
            taper = np.where(dist < init_r, 1 - (dist / init_r)**2, 0)
            self.world = noise * taper * self.config.init_density

        elif self.config.init_type == "uniform_disk":
            mask = dist < init_r
            self.world[mask] = self.config.init_density

        elif self.config.init_type == "orbium":
            # Classic Orbium blob with asymmetry for motion
            r_norm = dist / init_r
            bump = np.where(r_norm < 1, np.exp(-4 * r_norm**2) * (1 - r_norm**2), 0)
            # Slight asymmetry
            self.world = bump * 0.6
            self.world = np.roll(self.world, 2, axis=0)
            self.world = np.roll(self.world, 1, axis=1)

    def step(self) -> np.ndarray:
        """
        Advance simulation by one time step.
        Returns the growth field for analysis.
        """
        # Convolution: neighbor potential
        potential = fftconvolve(self.world, self.kernel, mode='same')

        # Growth
        growth = self.growth_func(potential, self.config.mu, self.config.sigma)

        # Update
        self.world = np.clip(self.world + self.config.dt * growth, 0.0, 1.0)

        self.t += self.config.dt
        self.step_count += 1

        return growth

    def run(self, steps: int) -> None:
        """Run simulation for given number of steps."""
        for _ in range(steps):
            self.step()

    def get_state(self) -> np.ndarray:
        """Return copy of current state."""
        return self.world.copy()

    def set_state(self, state: np.ndarray) -> None:
        """Set state (for perturbation experiments)."""
        assert state.shape == self.world.shape
        self.world = state.copy()

    def perturb(self, epsilon: float = 1e-8) -> np.ndarray:
        """
        Add small random perturbation.
        Returns the perturbation vector (for Lyapunov calculation).
        """
        perturbation = self.rng.standard_normal(self.world.shape)
        perturbation = perturbation / np.linalg.norm(perturbation) * epsilon
        self.world = np.clip(self.world + perturbation, 0.0, 1.0)
        return perturbation

    def clone(self) -> 'LeniaSimulation':
        """Create exact copy with same state."""
        new_sim = LeniaSimulation(self.config)
        new_sim.world = self.world.copy()
        new_sim.t = self.t
        new_sim.step_count = self.step_count
        return new_sim

    def mass(self) -> float:
        """Total mass (sum of all cells)."""
        return float(np.sum(self.world))

    def mass_ratio(self) -> float:
        """Mass as fraction of grid."""
        return self.mass() / self.world.size

    def activity(self, prev_world: np.ndarray) -> float:
        """Activity: sum of absolute changes."""
        return float(np.sum(np.abs(self.world - prev_world)))

    def reset(self):
        """Reset to initial condition."""
        self.world.fill(0)
        self.t = 0
        self.step_count = 0
        self.rng = np.random.default_rng(self.config.seed)
        self._initialize()


def create_perturbation_pair(config: LeniaConfig, epsilon: float = 1e-8
                             ) -> Tuple['LeniaSimulation', 'LeniaSimulation', np.ndarray]:
    """
    Create two simulations: reference and perturbed.
    Used for Lyapunov exponent calculation.

    Returns:
        (reference_sim, perturbed_sim, initial_perturbation)
    """
    ref = LeniaSimulation(config)
    pert = ref.clone()

    # Add perturbation
    delta = pert.perturb(epsilon)

    return ref, pert, delta
