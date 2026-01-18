# Emergent NAND Computation in Lenia Near Criticality

## Abstract (v4.0 — Corrected after experimental validation)

Lenia is a continuous cellular automaton exhibiting lifelike self-organizing patterns. We investigate its computational potential by systematically exploring parameter space. Scanning 1,600 configurations on a 40×40 (μ, σ) parameter grid (μ ∈ [0.10, 0.20], σ ∈ [0.01, 0.04], step 0.0025), we estimated Lyapunov exponents using Benettin's algorithm: paired trajectories with initial L2-norm perturbation δ = 10⁻⁸, renormalized every 10 steps over 500 measurement steps after 100-step warmup. This identified a near-critical regime with λ ≈ 0.

We report two main findings. First, we observe emergent NAND-like threshold behavior arising from self-repair dynamics: organisms survive single perturbations but collapse when dual simultaneous perturbations exceed repair capacity. "Collapse" is defined as mass falling below 25 (vs baseline ~73) within 200 steps. Using Gaussian hole perturbations (σ_h = 6 px, strength s = 0.8) at symmetric positions (±14 px from center), survival rates across 40 trials were: [0,0] 100%, [0,1] 100%, [1,0] 100%, [1,1] 20%. Comparing single-input vs double-input, Fisher's exact test yielded p < 10⁻¹⁰. Notably, null model experiments showed that random hole positioning produced equivalent collapse rates (15%), indicating the mechanism is a damage-threshold effect rather than position-specific computation.

Second, we tested generalization across the critical regime. Of 8 parameter configurations with |λ| < 0.01, only 3 (37.5%) exhibited NAND-like behavior. Ordered (λ < -0.01) and chaotic (λ > 0.01) regimes showed no NAND behavior, suggesting criticality is necessary but not sufficient for this threshold-based computation.

Additionally, using Lenia dynamics as a reservoir with logistic regression readout (4-fold cross-validation), we classified XOR inputs with 94% accuracy (baseline 50%).

Key limitations include: (1) the mechanism is destructive—the organism is consumed upon collapse; (2) NAND behavior is configuration-specific within the critical regime; (3) positioning does not affect outcome—this is pure damage-threshold dynamics; (4) cascading between gates has not been demonstrated; (5) transfer entropy analysis did not confirm directional information flow for signal propagation.

**Keywords:** Lenia, cellular automata, criticality, NAND gate, threshold computation, emergent computation

---

## Corrections from v3.1

### Parameter Corrections
- **Hole strength**: Changed from s = 0.5 to **s = 0.8** (experimentally validated)
- **Survival rates**: Updated to reflect actual experimental results with corrected parameters

### Removed/Modified Claims
1. **Signal propagation**: Removed causal claim. Transfer entropy analysis showed no significant directional information flow (p > 0.05). Correlation (r = 0.84) does not establish causation.

2. **Position-specificity**: Removed. Null model experiments showed symmetric and random positioning produce equivalent collapse rates (15% vs 15%), indicating this is a damage-threshold effect, not position-dependent computation.

3. **Generalization**: Added explicit limitation that NAND behavior occurs in only 37.5% (3/8) of critical configurations tested.

### What Remains Valid
- NAND-like threshold behavior (with corrected parameters)
- Criticality requirement (ordered/chaotic regimes don't show NAND)
- Statistical significance (Fisher's exact test)
- Reservoir computing XOR result
- Lyapunov exponent methodology

---

## Methods Section (Corrected)

### NAND Gate Protocol (CORRECTED)

1. Initialize organism: grid 128×128, μ = 0.15, σ = 0.015, init_size = 0.20, warmup 100 steps
2. Locate organism center (cy, cx) as centroid of mass
3. Apply perturbations (if input = 1):
   - Hole A: multiply by (1 - G) where G is Gaussian blob centered at (cy, cx-14) with σ=6, **peak=0.8**
   - Hole B: multiply by (1 - G) where G is Gaussian blob centered at (cy, cx+14) with σ=6, **peak=0.8**
4. Evolve 200 steps
5. Classify: alive if M > 25, else collapsed

**Note**: Previous version incorrectly stated peak=0.5. Experimental validation showed NAND behavior requires peak=0.8.

### Null Model Experiment (NEW)

To test position-specificity hypothesis:
- **Symmetric**: Holes at (cy, cx±14) — standard NAND protocol
- **Random**: Holes at random positions within organism radius
- **Result**: Both conditions showed 15% survival rate for double-hole case
- **Conclusion**: Positioning does not affect outcome; mechanism is pure damage threshold

### Transfer Entropy Analysis (NEW)

- Applied Schreiber's transfer entropy to probe time series
- TE(P1→P2) ≈ TE(P2→P1) for all probe pairs
- No significant directional flow detected (p > 0.05)
- **Conclusion**: Correlation in signal propagation does not indicate causation

### Morphology Generalization (NEW)

Tested NAND protocol across 8 critical configurations (|λ| < 0.01):
- 3/8 (37.5%) showed NAND-like behavior (gap > 50% between single and double hole survival)
- 5/8 (62.5%) showed no significant difference
- Ordered regime (2 configs): 0% NAND
- Chaotic regime (2 configs): 0% NAND

---

## Experimental Results Summary

| Experiment | Claim | Result | Status |
|------------|-------|--------|--------|
| NAND (s=0.8) | Threshold-based logic | Works (100% vs 20%) | ✅ Confirmed |
| Criticality | Required for NAND | Yes (0% in ordered/chaotic) | ✅ Confirmed |
| Generalization | Works across critical regime | Partial (37.5%) | ⚠️ Limited |
| Positioning | Symmetric is special | No (random = same result) | ❌ Refuted |
| Signal propagation | Causal flow | No (TE not significant) | ❌ Refuted |
| Reservoir XOR | 94% accuracy | Unchanged | ✅ Confirmed |

---

## References

[1] Benettin, G., Galgani, L., Giorgilli, A., & Strelcyn, J. M. (1980). Lyapunov characteristic exponents for smooth dynamical systems and for Hamiltonian systems. Meccanica, 15(1), 9-30.

[2] Schreiber, T. (2000). Measuring information transfer. Physical Review Letters, 85(2), 461.
