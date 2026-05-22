"""
Lenia Reservoir Computing

Use Lenia as a computational substrate for machine learning.
The complex nonlinear dynamics of Lenia transform inputs,
and we train only a simple linear readout.

Note: The current transform resets the base state per sample, operating as a nonlinear spatial feature map over sliding windows rather than a continuous temporal memory reservoir.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score

from .simulation import LeniaConfig, LeniaSimulation


@dataclass
class ReservoirConfig:
    """Configuration for Lenia Reservoir."""
    # Lenia parameters (critical regime for best computation)
    grid_size: int = 64
    mu: float = 0.15
    sigma: float = 0.015

    # Reservoir dynamics
    washout_steps: int = 20      # Initial steps to stabilize
    compute_steps: int = 30      # Steps for computation

    # Input encoding
    input_region_size: int = 10  # Size of input injection region
    input_strength: float = 0.5  # Strength of input perturbation

    # Readout
    readout_samples: int = 256   # Number of samples from reservoir state


class LeniaReservoir:
    """
    Lenia as a Reservoir Computer.

    The reservoir exploits Lenia's edge-of-chaos dynamics
    to perform nonlinear transformations on input data.
    """

    def __init__(self, config: ReservoirConfig = None):
        self.config = config or ReservoirConfig()
        self.readout_model = None
        self._sample_indices = None

    def _create_base_state(self, seed: int = None) -> LeniaSimulation:
        """Create a Lenia simulation in a stable 'ready' state."""
        lenia_config = LeniaConfig(
            grid_size=self.config.grid_size,
            mu=self.config.mu,
            sigma=self.config.sigma,
            seed=seed or 42,
            init_size=0.3,
            init_density=0.5,
        )
        sim = LeniaSimulation(lenia_config)

        # Washout: let it stabilize
        sim.run(self.config.washout_steps)

        return sim

    def _encode_input(self, sim: LeniaSimulation, input_data: np.ndarray) -> None:
        """
        Encode input data into the reservoir by perturbing regions.

        For scalar/vector input: perturb intensity at fixed locations
        For 2D input: directly inject pattern
        """
        input_data = np.atleast_1d(input_data).flatten()
        n_inputs = len(input_data)

        # Divide the grid into input regions
        region_size = self.config.input_region_size
        grid = self.config.grid_size

        # Place inputs in a ring around center
        center = grid // 2
        radius = grid // 4

        for i, val in enumerate(input_data):
            # Position on circle
            angle = 2 * np.pi * i / max(n_inputs, 1)
            x = int(center + radius * np.cos(angle))
            y = int(center + radius * np.sin(angle))

            # Clamp to valid range
            x = max(region_size, min(grid - region_size, x))
            y = max(region_size, min(grid - region_size, y))

            # Inject perturbation proportional to input value
            # Normalize input to [0, 1]
            val_norm = (val + 1) / 2 if val < 0 else val  # Handle [-1,1] or [0,1]
            val_norm = np.clip(val_norm, 0, 1)

            # Create gaussian blob perturbation
            yy, xx = np.ogrid[:region_size*2, :region_size*2]
            dist = np.sqrt((xx - region_size)**2 + (yy - region_size)**2)
            blob = np.exp(-dist**2 / (region_size/2)**2) * val_norm * self.config.input_strength

            # Add to reservoir
            y_start, x_start = y - region_size, x - region_size
            y_end, x_end = y + region_size, x + region_size

            sim.world[y_start:y_end, x_start:x_end] += blob
            sim.world = np.clip(sim.world, 0, 1)

    def _readout(self, sim: LeniaSimulation) -> np.ndarray:
        """
        Extract features from reservoir state.

        Uses random but fixed sampling for consistency.
        """
        world = sim.world.flatten()

        # Initialize sample indices if needed
        if self._sample_indices is None or len(self._sample_indices) != self.config.readout_samples:
            rng = np.random.default_rng(42)
            self._sample_indices = rng.choice(
                len(world),
                size=min(self.config.readout_samples, len(world)),
                replace=False
            )

        # Sample state
        samples = world[self._sample_indices]

        # Add some global statistics for richer representation
        stats = np.array([
            np.mean(sim.world),
            np.std(sim.world),
            np.max(sim.world),
            sim.mass() / sim.world.size,  # Normalized mass
        ])

        return np.concatenate([samples, stats])

    def transform(self, X: np.ndarray, seed: int = None) -> np.ndarray:
        """
        Transform sequence inputs through the reservoir.
        State is maintained between time steps for temporal memory.

        Args:
            X: Input data sequence, shape (n_samples, n_features) or (n_samples,)
            seed: Random seed for reproducibility

        Returns:
            Reservoir states over time, shape (n_samples, readout_dim)
        """
        X = np.atleast_2d(X)
        n_samples = X.shape[0]

        # Get output dimension using a dummy simulation
        test_sim = self._create_base_state(seed)
        self._encode_input(test_sim, X[0])
        test_sim.run(self.config.compute_steps)
        sample_output = self._readout(test_sim)
        outputs = np.zeros((n_samples, len(sample_output)))

        # Process the entire sequence sequentially
        sim = self._create_base_state(seed)
        for t, u_t in enumerate(X):
            self._encode_input(sim, np.atleast_1d(u_t))
            sim.run(self.config.compute_steps)
            outputs[t] = self._readout(sim)

        return outputs

    def fit(self, X: np.ndarray, y: np.ndarray, task: str = 'classification'):
        """
        Fit the readout layer.

        Args:
            X: Input data
            y: Target labels/values
            task: 'classification' or 'regression'
        """
        # Transform through reservoir
        print(f"Transforming {len(X)} samples through Lenia reservoir...")
        reservoir_states = self.transform(X)

        # Fit readout
        print("Fitting readout layer...")
        if task == 'classification':
            self.readout_model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            self.readout_model = Ridge(alpha=1.0)

        self.readout_model.fit(reservoir_states, y)

        # Training score
        train_pred = self.readout_model.predict(reservoir_states)
        if task == 'classification':
            score = accuracy_score(y, train_pred)
            print(f"Training accuracy: {score:.2%}")
        else:
            score = np.mean((y - train_pred)**2)
            print(f"Training MSE: {score:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the trained reservoir."""
        if self.readout_model is None:
            raise ValueError("Reservoir not fitted. Call fit() first.")

        reservoir_states = self.transform(X)
        return self.readout_model.predict(reservoir_states)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy/R² score."""
        predictions = self.predict(X)
        if hasattr(self.readout_model, 'predict_proba'):
            return accuracy_score(y, predictions)
        else:
            return 1 - np.mean((y - predictions)**2) / np.var(y)


def generate_narma10(n_samples: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Generate NARMA-10 dataset."""
    rng = np.random.default_rng(42)
    u = rng.uniform(0, 0.5, n_samples)
    y = np.zeros(n_samples)
    
    for t in range(9, n_samples - 1):
        y[t+1] = 0.3 * y[t] + 0.05 * y[t] * np.sum(y[t-9:t+1]) + 1.5 * u[t-9] * u[t] + 0.1
        
    return u, y

def test_narma10():
    """
    Test reservoir on NARMA-10 benchmark.
    This is a standard temporal memory and nonlinearity test for Reservoir Computing.
    """
    print("="*60)
    print("TEST: NARMA-10 Task (Temporal Memory & Nonlinearity)")
    print("="*60)
    print("NARMA-10 requires both nonlinear transformation and memory of past inputs.")

    # Generate data
    n_samples = 400
    u, y = generate_narma10(n_samples)
    
    # In a true reservoir, we feed the sequence directly
    # and the reservoir state retains the memory.
    X = u.reshape(-1, 1)
    
    # We must skip the first 10 steps for evaluation since NARMA-10 needs 10 steps of history
    # to be fully defined by the sequence.
    valid_idx = 10
    
    # Split train/test
    split = 200
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # For baseline, we'll use a sliding window of 10 for the linear model to be fair
    # (since linear model has no memory of its own)
    X_baseline = np.zeros((n_samples - 10, 10))
    for i in range(len(X_baseline)):
        X_baseline[i] = u[i:i+10]
    
    X_base_train, X_base_test = X_baseline[:split-10], X_baseline[split-10:]
    y_base_train, y_base_test = y[10:split], y[split:]

    # Baseline: Linear regression on recent history
    print("1. Baseline (Linear on 10-step window):")
    baseline = Ridge(alpha=1.0)
    baseline.fit(X_base_train, y_base_train)
    baseline_mse = np.mean((y_base_test - baseline.predict(X_base_test))**2)
    baseline_nmse = baseline_mse / np.var(y_base_test)
    print(f"   Test NMSE: {baseline_nmse:.4f}\n")

    # Reservoir
    print("2. Lenia Reservoir Computing:")
    config = ReservoirConfig(
        grid_size=48,
        compute_steps=15,
        readout_samples=256,
        input_region_size=8,
    )
    reservoir = LeniaReservoir(config)
    
    # We fit on the full sequence, but evaluate only on valid indices
    print(f"Transforming {len(X_train)} samples through Lenia reservoir...")
    res_states_train = reservoir.transform(X_train)
    res_states_test = reservoir.transform(X_test)
    
    print("Fitting readout layer...")
    reservoir.readout_model = Ridge(alpha=1.0)
    # Fit only on states where NARMA-10 is well-defined
    reservoir.readout_model.fit(res_states_train[valid_idx:], y_train[valid_idx:])
    
    predictions = reservoir.readout_model.predict(res_states_test)
    test_mse = np.mean((y_test - predictions)**2)
    test_nmse = test_mse / np.var(y_test)
    print(f"\nTest NMSE: {test_nmse:.4f}")

    if test_nmse < baseline_nmse:
        print("\n🎉 SUCCESS! Lenia outperforms linear baseline on NARMA-10!")
        print("   The reservoir computing paradigm works for temporal tasks!")
    else:
        print("\n🤔 Reservoir didn't beat baseline. Needs tuning.")

    return test_nmse


def test_pattern_recognition():
    """
    Test reservoir on simple pattern recognition.
    Can it distinguish different input patterns?
    """
    print("\n" + "="*60)
    print("TEST: Pattern Recognition")
    print("="*60)

    # Create simple patterns: "left heavy" vs "right heavy"
    n_samples = 20
    X = []
    y = []

    rng = np.random.default_rng(42)

    for _ in range(n_samples // 2):
        # Class 0: more on left
        X.append([0.8 + rng.uniform(-0.1, 0.1), 0.2 + rng.uniform(-0.1, 0.1)])
        y.append(0)

        # Class 1: more on right
        X.append([0.2 + rng.uniform(-0.1, 0.1), 0.8 + rng.uniform(-0.1, 0.1)])
        y.append(1)

    X = np.array(X)
    y = np.array(y)

    # Shuffle
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    # Split
    X_train, X_test = X[:14], X[14:]
    y_train, y_test = y[:14], y[14:]

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Reservoir
    config = ReservoirConfig(grid_size=48, compute_steps=15)
    reservoir = LeniaReservoir(config)
    reservoir.fit(X_train, y_train)

    test_acc = reservoir.score(X_test, y_test)
    print(f"\nTest accuracy: {test_acc:.2%}")

    return test_acc


if __name__ == "__main__":
    print("\n" + "🧠 "*20)
    print("LENIA RESERVOIR COMPUTING")
    print("Using Lenia as a computational substrate")
    print("🧠 "*20 + "\n")

    narma_nmse = test_narma10()
    pattern_acc = test_pattern_recognition()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"NARMA-10 NMSE:    {narma_nmse:.4f}")
    print(f"Pattern accuracy: {pattern_acc:.2%}")

    if narma_nmse < 1.0:
        print("\n✨ Lenia Reservoir Computing WORKS!")
        print("   The complex dynamics provide useful computation.")
    else:
        print("\n🔧 Needs tuning - try different μ, σ, or compute_steps")
