# Lenia Criticality Lab

**A Research Framework for Differentiable Self-Organizing Systems**

This repository contains a high-performance, differentiable implementation of the Lenia continuous cellular automaton using JAX. It is designed to investigate the computational capabilities of self-organizing systems, specifically focusing on criticality, information flow, and hybrid neuro-evolutionary architectures.

## Key Features

### 1. JAX-Accelerated Physics Engine
- **125x Performance Improvement**: Replaces legacy Scipy convolution with JAX's `fftconvolve` and XLA compilation.
- **Parallelization**: Supports `vmap` for simulating thousands of environments simultaneously on a single GPU.
- **Differentiability**: Fully differentiable physics pipeline allowing for gradient-based optimization of life parameters.

### 2. Rigorous Statistical Metrics
- **Largest Lyapunov Exponent (LLE)**: Implements batched Jacobian linearization to quantify chaos and stability (5,400 steps/sec).
- **Transfer Entropy (TE)**: Vectorized symbolic transfer entropy estimation (>50,000 pairs/sec) with surrogate data testing for statistical significance (p-values).
- **Scientific Validation**: Validated against null models (noise) and known causal systems.

### 3. Neuro-Lenia (Hybrid AI)
- **Learned Physics**: Demonstrates the ability to "learn" the physical parameters ($\mu, \sigma, K$) required to stabilize arbitrary patterns or solve memory tasks via Backpropagation Through Time (BPTT).

### 4. Locomotion & Evolution (Auto-Lenia)
- **Virtual Soft Robotics**: Evolved creatures that achieve consistent locomotion (0.34 displacement/step) via self-organized undulation.
- **Open-Ended Evolution**: Genetic Algorithm (GA) that co-evolves body morphology and physics parameters for survival and movement.

### 5. Computational Universality
- **Logic Gates**: Discovered a stable "Soft-NAND" regime at $\mu \approx 0.8$.
- **Memory**: Constructed a functional **RS-Trigger (Flip-Flop)** capable of storing 1 bit of information via perturbation-induced bistability.

### 6. Sensory-Motor Agents
- **Chemotaxis**: Implemented reactive agents that sense "chemical" gradients and navigate towards food sources ($\alpha=0.2$ sensory coupling).

## Installation

Requires Python 3.10+ and JAX.

```bash
pip install -r requirements.txt
# For GPU support, consult JAX documentation regarding CUDA installation.
```

## Usage

### 1. Running Simulations (Demo)

```bash
python scripts/view_species.py
```

### 2. Differentiable Parameter Search

Train the system to discover parameters that satisfy specific criteria (e.g., target mass, variance).

```bash
python src/search_jax.py
```

### 3. Training Neuro-Lenia

Train the hybrid Neuro-Lenia model to perform pattern reconstruction/memory tasks.

```bash
python scripts/train_hybrid_eqx.py
```

### 4. Verification

Run the full scientific verification suite (Physics invariants, Statistical significance, Differentiability).

```bash
python -m pytest tests/
```

## Project Structure

- `src/`
    - `engine_jax.py`: Core differentiable physics engine.
    - `metrics_jax.py`: Lyapunov and Transfer Entropy implementations.
    - `neuro_lenia.py`: Equinox modules for Hybrid AI integration.
    - `search_jax.py`: Gradient-based parameter optimization.
- `scripts/`: Training and verification scripts.
- `tests/`: Pytest suite for formal verification.
- `paper/`: Drafts of scientific publications.

## Citation

If you use this codebase in your research, please cite:

```
@misc{lenia_jax_2026,
  author = {Lenia Criticality Lab},
  title = {Differentiable Self-Organizing Systems: Accelerating Lenia Dynamics via JAX},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/m4rba4s/lenia-criticality-lab}}
}
```

## License

MIT License.
