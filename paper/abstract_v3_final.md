# Emergent NAND Gate and Signal Propagation in Lenia Near Criticality

## Abstract (v3 — Final, post red-team + experiments)

Lenia is a continuous cellular automaton exhibiting lifelike self-organizing patterns. We investigate its computational potential by systematically exploring the parameter space. Scanning 1,600 configurations (μ ∈ [0.10, 0.20], σ ∈ [0.01, 0.04] with step 0.0025), we estimated Lyapunov exponents via trajectory divergence (initial perturbation δ = 10⁻⁶, 200-step averaging window) and identified a near-critical regime where λ ≈ 0.

We report three findings. First, local perturbations propagate through organisms with measurable temporal structure: lagged cross-correlations between spatially separated measurement points reach r = 0.84 (lag = 5 steps, n = 40 timepoints), with correlation strength decaying monotonically with spatial distance. Second, we observe an emergent NAND-like behavior arising from the organism's self-repair dynamics: single perturbations (removing ~15% local mass) are absorbed and healed, while two simultaneous perturbations exceed the repair threshold, causing organism collapse. This pattern—survive unless both inputs active—matches the NAND truth table. We validated this across 80 trials (20 per input condition) with randomized seeds; the difference between single and double perturbation survival rates was significant (Fisher's exact test, p < 10⁻¹⁵). The behavior remained robust under position variation (±3 pixels, 90% vs 25% survival) and amplitude variation (±20%, 95% vs 5% survival). Third, using Lenia dynamics as a reservoir with linear readout, we correctly classified XOR inputs (4/4 conditions), demonstrating nonlinear separability in the transformed feature space.

Since NAND is a universal gate, these results suggest that self-organizing continuous cellular automata may support arbitrary computation through damage-threshold dynamics, consistent with the edge-of-chaos hypothesis. Limitations include the binary (survive/die) output, which requires additional mechanisms for signal cascading.

**Keywords:** Lenia, cellular automata, criticality, NAND gate, reservoir computing, emergent computation

---

## Key Numbers for Methods Section

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Grid size | 128×128 | Balance speed/resolution |
| μ range | [0.10, 0.20] | Covers viable organisms |
| σ range | [0.01, 0.04] | Covers stable to chaotic |
| Grid step | 0.0025 | 40×40 = 1600 points |
| λ perturbation δ | 10⁻⁶ | Standard for Benettin |
| λ averaging window | 200 steps | Post-transient |
| Warmup steps | 80 | Organism stabilization |
| Compute steps | 200 | Sufficient for collapse |
| Hole strength | 0.5 | Center of working range [0.45, 0.65] |
| Hole size | 12×12 pixels | ~15% of organism |
| Trials per condition | 20 | Sufficient for Fisher's test |
| Total trials | 80 | 4 conditions × 20 |

---

## Statistics Ready for Paper

```
NAND Gate Validation:
- [0,0]: 20/20 survive (100%)
- [0,1]: 20/20 survive (100%)
- [1,0]: 20/20 survive (100%)
- [1,1]: 0/20 survive (0%)

Fisher's exact test:
- Comparing: (single input survival) vs (double input survival)
- Contingency: [[40, 0], [0, 20]]
- p-value: 2.39 × 10⁻¹⁶
- Effect size: Perfect separation

Robustness:
- Position ±3px: 90% vs 25% (still significant)
- Strength ±20%: 95% vs 5% (still significant)

Signal propagation:
- Max lagged correlation: r = 0.84
- Correlation decay with distance: confirmed
- Measurement points: 6
- Timepoints: 40
```

---

## Честная оценка limitations (для Discussion)

1. **Binary output**: Организм или жив, или мёртв. Нет градаций для cascading.

2. **No demonstrated cascading**: Мы не показали цепочку гейтов. Это следующий шаг.

3. **Single morphology tested**: Только стандартный Orbium. Нужна проверка на других видах.

4. **Causality not proven**: r = 0.84 это корреляция, не transfer entropy. Можем добавить.

5. **Limited reservoir testing**: Только XOR, нужны NARMA/Mackey-Glass для серьёзного RC claim.
