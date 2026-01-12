# Emergent NAND Computation in Lenia Near Criticality

**Research project exploring computation at the edge of chaos in continuous cellular automata.**

## Key Results

| Finding | Evidence |
|---------|----------|
| **NAND gate via self-repair** | p = 2.4×10⁻¹⁶, 80 trials |
| **Signal propagation** | r = 0.84 lagged correlation |
| **Critical regime identified** | λ ≈ 0 in (μ, σ) space |
| **1,600 parameter configurations** | Phase diagram mapped |

## Abstract

Lenia is a continuous cellular automaton exhibiting lifelike self-organizing patterns. We demonstrate emergent NAND-like computation arising from self-repair dynamics: organisms survive single perturbations but collapse when two simultaneous perturbations exceed the repair threshold. Since NAND is functionally complete, this suggests a pathway toward richer computation in self-organizing systems.

📄 **Full abstract**: [`paper/abstract_v3.1_final.md`](paper/abstract_v3.1_final.md)

## Quick Start

```bash
# View elite species (interactive)
python view_species.py

# Run phase diagram experiment
python scripts/run_experiment.py --experiment phase_diagram

# View specific species
python scripts/simple_view.py
```

## Project Structure

```
lenia_criticality/
├── src/
│   ├── simulation.py      # Headless Lenia engine
│   ├── metrics.py         # Lyapunov (Benettin), correlations, MI
│   ├── reservoir.py       # Reservoir computing
│   ├── experiment.py      # Parallel experiment runner
│   └── analysis.py        # Publication figures
├── paper/
│   └── abstract_v3.1_final.md  # Current draft
├── experiments/
│   ├── elite_species.json      # 31 discovered species
│   └── results.csv             # Phase diagram data
├── figures/
│   ├── fig1_phase_diagram.png
│   ├── fig2_species.png
│   ├── fig3_signal_propagation.png
│   ├── fig4_correlations.png
│   ├── fig5_lyapunov.png
│   └── fig6_xor_gate.png
├── scripts/
│   ├── run_experiment.py
│   ├── view_elite.py
│   └── simple_view.py
└── view_species.py        # Interactive species viewer
```

## Methods

### Lyapunov Exponent (λ)
- **Algorithm**: Benettin (paired-trajectory divergence)
- **Perturbation**: δ = 10⁻⁸, L2 norm
- **Renormalization**: every 10 steps
- **Measurement**: 500 steps after 100-step warmup

### NAND Gate
- **Mechanism**: Self-repair threshold
- **Single perturbation**: absorbed → survive
- **Double perturbation**: exceeds threshold → collapse
- **Collapse criterion**: mass < 25 at t=200 (baseline ~73)

### Signal Propagation
- **Measurement**: 6 probes along organism axis
- **Correlation**: lag=5 steps, n=40 timepoints
- **Result**: r = 0.84 between adjacent probes

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Grid size | 128×128 | Balance speed/resolution |
| μ (growth center) | 0.15 | Standard Orbium |
| σ (growth width) | 0.015 | Standard Orbium |
| Critical regime | μ ≈ 0.158, σ ≈ 0.019 | λ ≈ 0 |
| Hole strength | 0.5 | NAND working range [0.45, 0.65] |

## Figures

| Figure | Description |
|--------|-------------|
| fig1 | Phase diagram (μ, σ) with λ coloring |
| fig2 | Species evolution snapshots |
| fig3 | Signal propagation heatmap |
| fig4 | Temporal correlation matrix |
| fig5 | Lyapunov distribution |
| fig6 | NAND gate before/after |

## Work in Progress

- [ ] **Transfer entropy** — establish causal information flow (not just correlation)
- [ ] **Multiple morphologies** — validate NAND on 2-3 other species
- [ ] **NARMA / Mackey-Glass** — standard reservoir computing benchmarks
- [ ] **Gate cascading** — demonstrate signal chain between gates

## Requirements

```bash
pip install numpy scipy matplotlib scikit-learn
```

Optional:
- `pygame` — real-time visualization
- `numba` — JIT acceleration

## Citation

```bibtex
@misc{lenia_nand_2025,
  title={Emergent NAND Computation and Signal Propagation in Lenia Near Criticality},
  author={...},
  year={2025},
  note={In preparation}
}
```

## References

1. Chan, B.W.C. (2019). Lenia: Biology of Artificial Life. Complex Systems, 28(3).
2. Langton, C.G. (1990). Computation at the edge of chaos. Physica D, 42(1-3).
3. Benettin, G. et al. (1980). Lyapunov characteristic exponents. Meccanica, 15(1).

## License

MIT
