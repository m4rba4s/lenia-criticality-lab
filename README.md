# Lenia Criticality Lab

JAX/NumPy research lab for Lenia criticality, self-organization, null models, and reservoir-style computation.

The project contains two related simulation paths:

- a NumPy/SciPy reference engine in `src/simulation.py` for reproducible experiments and analysis;
- a JAX engine in `src/engine_jax.py` for batched, differentiable, accelerator-friendly execution.

This is a computational self-organization framework, not a validated molecular biology simulator. The state field is a bounded continuous density evolved by convolutional neighborhood potentials and nonlinear growth rules. It can be useful for studying reaction-diffusion-like dynamics, criticality proxies, soft-body morphology, and emergent computation, but it does not encode biochemical stoichiometry, energy conservation, steric exclusion, or genetic regulatory networks.

## What Is Implemented

- Periodic Lenia dynamics with normalized radial kernels and bounded synchronous updates.
- JAX-accelerated stepping with `jit`, `vmap`, FFT convolution, and differentiable parameters.
- Criticality metrics: Lyapunov estimation, spatial correlations, entropy, mutual information, and transfer entropy.
- Neuro-Lenia modules built with Equinox for gradient-based experiments.
- Locomotion and morphology scripts for soft-robotics-style behavior search.
- Interactive demos for phase exploration, signal propagation, and NAND-like threshold dynamics.
- A pytest suite covering physics invariants, metrics, differentiability, locomotion behavior, and NumPy/JAX boundary assumptions.

## Repository Layout

```text
src/
  simulation.py       NumPy reference Lenia engine
  engine_jax.py       JAX Lenia engine
  metrics.py          NumPy/SciPy analysis metrics
  metrics_jax.py      JAX metrics and batched estimators
  neuro_lenia.py      Equinox differentiable Lenia layer/RNN
  experiment.py       Parameter-sweep runner
  analysis.py         Result summaries and plotting helpers

tests/
  test_simulation_numpy.py  Reference-engine invariants
  test_physics.py           JAX physics invariants
  test_metrics.py           Transfer entropy and Lyapunov smoke tests
  test_neuro.py             Differentiability tests
  test_locomotion.py        Locomotion behavior tests

scripts/              Experiment, verification, training, and visualization helpers
paper/                Draft manuscript text
figures/              Generated figures and demo outputs
```

## Installation

Use a virtual environment. The default dependency set installs CPU JAX.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

For CUDA/TPU support, install the JAX build matching your accelerator from the official JAX instructions, then install the remaining requirements.

## Verification

Run the full test suite:

```bash
.venv/bin/python -m pytest tests/
```

Run the JAX-accelerated null models (Equal Mass, Random Damage, Temporal Shuffling) to verify baseline claims:

```bash
.venv/bin/python scripts/verify_null_models_jax.py
```

Run the Lenia Reservoir Computing benchmark (now upgraded to NARMA-10 for strict temporal memory evaluation):

```bash
.venv/bin/python -m src.reservoir
```

Current expected result in this workspace:

```text
18 passed
```

The tests verify determinism, toroidal translation invariance, kernel sanity, transfer entropy behavior, Lyapunov computation shape/NaN safety, differentiability through the Equinox model, and locomotion smoke behavior.

## Running Experiments

Minimal headless smoke run:

```bash
MPLCONFIGDIR=/tmp/lenia-mplconfig MPLBACKEND=Agg \
  .venv/bin/python scripts/run_experiment.py \
  --experiment phase_diagram \
  --resolution 1 \
  --grid-size 32 \
  --workers 1 \
  --no-lyapunov \
  --serial \
  --output /tmp/lenia_smoke
```

Larger phase diagram scan:

```bash
.venv/bin/python scripts/run_experiment.py \
  --experiment phase_diagram \
  --resolution 30 \
  --grid-size 128 \
  --workers 4 \
  --output experiments
```

Lyapunov-focused scan:

```bash
.venv/bin/python scripts/run_experiment.py \
  --experiment lyapunov \
  --resolution 20 \
  --grid-size 128 \
  --workers 4
```

## Interactive Demos

The demos use Matplotlib interactive windows and require a GUI backend:

```bash
.venv/bin/python demo.py help
.venv/bin/python demo.py phase
.venv/bin/python demo.py signal
.venv/bin/python demo.py nand
```

In a headless shell, use `MPLBACKEND=Agg` only for import/start smoke checks. It will not display the interactive window.

## Model Notes

The reference update is synchronous and double-buffered:

1. compute the toroidal convolution potential from the previous state;
2. apply the nonlinear growth function;
3. add `dt * growth`;
4. clip the density field into `[0, 1]`;
5. swap buffers.

The toroidal boundary condition is intentional: it makes every cell topologically equivalent and matches the JAX engine's translation-invariance tests. This avoids edge-specific artifacts from zero-padded convolution.

## Reproducibility

- `LeniaConfig` is serializable and hashable.
- Seeds are explicit in the simulation config.
- Experiment runs persist `config.json`, `results.csv`, `results.parquet`, and `summary.json`.
- For headless or sandboxed environments, set `MPLCONFIGDIR` to a writable directory to avoid Matplotlib cache warnings.

## Known Limits

- The biological interpretation is qualitative. Treat results as artificial life / dynamical systems experiments unless independently validated against a specific biological mechanism.
- The "NAND-like" behavior is a destructive threshold response: the organism is consumed upon collapse, precluding direct gate reuse or cascading without a restoration mechanism.
- Long parameter sweeps can be expensive because Lyapunov and correlation metrics require many simulation steps.
- GPU acceleration depends on a correctly installed JAX accelerator build.
- Some demo claims are exploratory and should be backed by fresh experiment output before publication.

## Citation

```bibtex
@misc{lenia_criticality_lab_2026,
  author = {Lenia Criticality Lab},
  title = {Lenia Criticality Lab: Differentiable Self-Organizing Systems},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/m4rba4s/lenia-criticality-lab}}
}
```

## License

MIT License.
