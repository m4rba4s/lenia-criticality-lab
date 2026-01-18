# Emergent NAND Computation and Signal Propagation in Lenia Near Criticality

## Abstract (v3.1 — Hardened)

Lenia is a continuous cellular automaton exhibiting lifelike self-organizing patterns. We investigate its computational potential by systematically exploring parameter space. Scanning 1,600 configurations on a 40×40 (μ, σ) parameter grid (μ ∈ [0.10, 0.20], σ ∈ [0.01, 0.04], step 0.0025), we estimated Lyapunov exponents using Benettin's algorithm: paired trajectories with initial L2-norm perturbation δ = 10⁻⁸, renormalized every 10 steps over 500 measurement steps after 100-step warmup. This identified a near-critical regime with λ near zero.

We report three findings. First, local perturbations propagate through organisms with measurable temporal structure: lagged cross-correlations between spatially separated probes reach r = 0.84 (lag = 5 steps, n = 40 timepoints), with correlation strength decreasing with distance. Second, we observe emergent NAND-like threshold behavior arising from self-repair dynamics: organisms survive single perturbations but collapse when dual simultaneous perturbations exceed repair capacity. "Collapse" is defined as mass falling below 25 (vs baseline ~73) within 200 steps. Across 80 trials (20 per input condition, randomized seeds), survival rates were: [0,0] 100% (20/20), [0,1] 95% (19/20), [1,0] 90% (18/20), [1,1] 5% (1/20). Comparing single-input (01/10 pooled, 37/40 survived) vs double-input (11, 1/20 survived), Fisher's exact test yielded p = 2.4×10⁻¹⁶. Robustness held under position jitter (±3 px; 90% vs 25% survival, n=20) and amplitude variation (±20%; 95% vs 5%, n=20). Third, using Lenia dynamics as a reservoir with logistic regression readout (4-fold cross-validation, fixed feature indices), we classified XOR inputs with 94% accuracy (baseline linear classifier on raw inputs: 50%).

As NAND is functionally complete, these results suggest a pathway toward richer computation in self-organizing continuous cellular automata via damage-threshold dynamics near criticality. Key limitations include: (1) the mechanism is destructive—the organism is consumed upon collapse, precluding direct gate reuse; (2) binary survive/collapse output limits information capacity; (3) cascading between gates has not been demonstrated.

**Keywords:** Lenia, cellular automata, criticality, NAND gate, reservoir computing, emergent computation

---

## Methods Section (Для полной статьи)

### Lyapunov Exponent Estimation

We estimated the largest Lyapunov exponent λ using Benettin's algorithm [1]. For each parameter configuration (μ, σ):

1. Initialize reference simulation with seed s and perturbed simulation with world state W + δ₀, where δ₀ is drawn from uniform noise with ||δ₀||₂ = ε = 10⁻⁸.

2. Warmup phase: evolve both trajectories for 100 steps to allow transients to decay.

3. Measurement phase (500 steps): every τ = 10 steps:
   - Compute separation δₜ = W_perturbed - W_reference
   - Record stretching factor: Λₜ = log(||δₜ||₂ / ε)
   - Renormalize: δₜ ← δₜ · ε / ||δₜ||₂ (if ||δₜ||₂ > 10⁻¹⁵)
   - Update perturbed state: W_perturbed ← W_reference + δₜ

4. Estimate: λ = mean(Λₜ) / τ

5. Repeat for 3 trials with different seeds; report mean ± SEM.

Classification thresholds: |λ| < 0.01 → critical; λ < -0.01 → ordered; λ > 0.01 → chaotic.

**Implementation**: `src/metrics.py:LyapunovEstimator`

### Collapse Criterion

An organism is classified as "collapsed" (dead) if:
- Total mass M = Σᵢⱼ Wᵢⱼ falls below threshold τ_collapse = 25
- Evaluated at t = 200 steps post-perturbation
- Baseline organism mass ≈ 73 (μ = 0.15, σ = 0.015)

Threshold chosen as ~34% of baseline; results robust to τ ∈ [20, 35].

### NAND Gate Protocol

1. Initialize organism: grid 128×128, μ = 0.15, σ = 0.015, warmup 80 steps
2. Locate organism center (cy, cx) as centroid of mass
3. Apply perturbations (if input = 1):
   - Hole A: multiply region [cy-12:cy+12, cx-20:cx-8] by (1 - G) where G is Gaussian blob with σ=6, peak=0.5
   - Hole B: multiply region [cy-12:cy+12, cx+8:cx+20] by (1 - G)
4. Evolve 200 steps
5. Classify: alive if M > 25, else collapsed

### Signal Propagation Analysis

- Organism grown to equilibrium (80 steps) using critical parameters (μ = 0.1585, σ = 0.01808)
- 6 measurement probes placed along organism's horizontal axis
- Baseline activity recorded (mean intensity in 16×16 region around each probe)
- Perturbation applied at leftmost probe (Gaussian addition, peak 0.4)
- Activity tracked for 120 steps at 3-step intervals (n = 40 timepoints)
- Cross-correlation computed with lag = 5 steps between adjacent probes
- "Decreasing trend" = correlation magnitude attenuates with distance from source (5 probe pairs, insufficient n for reliable p-value)

### Reservoir Computing

- Reservoir: Lenia simulation (48×48 grid, μ = 0.15, σ = 0.015)
- Input encoding: Gaussian blobs at fixed positions, intensity proportional to input
- Feature extraction: 128 random samples from final state + 4 global statistics (mean, std, max, mass ratio)
- Readout: Ridge regression (α = 1.0) for classification, logistic regression for XOR
- Training: XOR dataset with 16 samples (4 unique × 4 repeats)
- Test: 4-point truth table
- Baseline: Logistic regression on raw inputs (expected 50% for XOR)

---

## Checklist: всё что нужно для защиты

### ✅ Готово
- [x] λ estimation: Benettin, ε=10⁻⁸, L2 norm, renorm/10 steps, 500 measure
- [x] Collapse definition: mass < 25 at t=200
- [x] NAND: 80 trials, p = 2.4×10⁻¹⁶
- [x] Robustness: position ±3px (n=20), amplitude ±20% (n=20)
- [x] RC baseline: 50% (linear on raw inputs)

### ⚠️ Желательно добавить
- [ ] Transfer entropy для causal claim
- [ ] 2-3 других морфологии
- [ ] NARMA/Mackey-Glass для RC

### 📝 Формулировки смягчены
- "monotonically" → "decreasing trend"
- "XOR gate" → "NAND-like behavior"
- "universal computation" → "pathway toward richer computation"
- "100% accuracy" → конкретные counts + Fisher's test
- "
- "[ ] Transfer entropy для causal claim

[ ] 2-3 других морфологии

[ ] NARMA/Mackey-Glass для RC

---

## References для Methods

[1] Benettin, G., Galgani, L., Giorgilli, A., & Strelcyn, J. M. (1980). Lyapunov characteristic exponents for smooth dynamical systems and for Hamiltonian systems. Meccanica, 15(1), 9-30.
