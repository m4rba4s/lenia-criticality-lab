# Methods (v2.0 — Experimentally Validated)

## 2.1 Lenia Simulation

We implemented Lenia following Chan [4]. The system evolves on a 128×128 continuous grid $A \in [0,1]^{N \times N}$ according to:

$$A^{t+dt} = \text{clip}\left(A^t + dt \cdot G(U^t)\right), \quad U^t = K * A^t$$

where $K$ is a ring-shaped kernel and $G$ is a Gaussian growth function:

$$G(u) = 2 \exp\left(-\frac{(u - \mu)^2}{2\sigma^2}\right) - 1$$

The kernel $K$ has inner radius $r_i = 0.5R$, outer radius $R = 13$ pixels, and is normalized to sum to 1. Convolution uses FFT for efficiency. Time step $dt = 0.1$.

**Implementation:** `src/simulation.py`, NumPy/SciPy, deterministic given seed.

## 2.2 Phase Diagram Construction

We scanned the parameter space systematically:

| Parameter | Range | Steps | Total |
|-----------|-------|-------|-------|
| μ (growth center) | [0.10, 0.20] | 40 | — |
| σ (growth width) | [0.01, 0.04] | 40 | — |
| **Configurations** | — | — | **1,600** |

For each (μ, σ) configuration:
1. Initialize with random blob (init_size = 0.20, density = 0.5)
2. Evolve 300 steps
3. Record: mean mass, final mass
4. Estimate Lyapunov exponent

**Classification criteria:**
- Dead: final mass < 10
- Critical: |λ| < 0.01
- Chaotic: λ > 0.01
- Ordered: λ < -0.01

## 2.3 Lyapunov Exponent Estimation

We estimated the largest Lyapunov exponent λ using Benettin's algorithm [11] with explicit renormalization:

**Algorithm:**
1. Initialize reference trajectory $W_{ref}$ and perturbed trajectory $W_{pert} = W_{ref} + \delta_0$
2. Initial perturbation: $\|\delta_0\|_2 = \varepsilon = 10^{-8}$ (uniform noise, L2 norm)
3. Warmup: evolve both trajectories for $T_{warm} = 100$ steps
4. Measurement phase ($T_{measure} = 500$ steps):
   - Every $\tau = 10$ steps: renormalize and record stretching
5. Estimate: $\lambda = \frac{1}{n} \sum_t \Lambda_t / \tau$

**Averaging:** 3 independent trials per configuration.

**Implementation:** `src/metrics.py:LyapunovEstimator`

## 2.4 NAND Gate Protocol (CORRECTED)

### 2.4.1 Organism Preparation
1. Initialize grid 128×128 with parameters μ = 0.15, σ = 0.015
2. **CORRECTED:** init_size = 0.20 (not 0.22)
3. Evolve **100 steps** to reach stable configuration (not 80)
4. Verify mass > 50 (organism alive)
5. Locate organism centroid $(c_y, c_x)$ as center of mass

### 2.4.2 Perturbation Application (CORRECTED)

**Hole template:** Gaussian blob $H(x,y) = \exp(-(x^2+y^2)/(2\sigma_h^2))$
- $\sigma_h = 6$ pixels
- **CORRECTED:** Peak strength $s = 0.8$ (not 0.5)

**Perturbation positions:**
- Input A: center at $(c_y, c_x - 14)$
- Input B: center at $(c_y, c_x + 14)$

**Application:** If input = 1, multiply local region by $(1 - s \cdot H)$.

### 2.4.3 Collapse Criterion
After perturbation, evolve 200 steps. Classify outcome:

$$\text{Outcome} = \begin{cases} \text{ALIVE} & \text{if } M_{t=200} > 25 \\ \text{COLLAPSED} & \text{if } M_{t=200} \leq 25 \end{cases}$$

### 2.4.4 Statistical Validation
- **Trials:** 40 per input condition
- **Randomization:** Different seed per trial
- **Test:** Fisher's exact test comparing single-input vs double-input survival rates

## 2.5 Null Model Experiments (NEW)

### 2.5.1 Position Invariance Test

To test whether symmetric positioning is essential for NAND behavior:

**Conditions:**
1. **Symmetric:** Holes at $(c_y, c_x \pm 14)$ — standard protocol
2. **Random:** Holes at random positions within 20px of center

**Procedure:**
- 20 trials per condition
- Same hole parameters (s = 0.8, σ_h = 6)
- Compare survival rates between conditions

### 2.5.2 Regime Comparison

Test NAND across dynamical regimes:
- **Critical:** 8 configurations with |λ| < 0.01
- **Ordered:** 2 configurations with λ < -0.01
- **Chaotic:** 2 configurations with λ > 0.01

## 2.6 Transfer Entropy Analysis (NEW)

### 2.6.1 Motivation

To distinguish correlation from causation in signal propagation, we applied Schreiber's transfer entropy [12]:

$$TE(X \rightarrow Y) = H(Y_t | Y_{past}) - H(Y_t | Y_{past}, X_{past})$$

### 2.6.2 Procedure
1. Place 6 probes along organism horizontal axis
2. Record activity time series (200 steps, interval = 1)
3. Apply perturbation at P₁
4. Compute TE matrix for all probe pairs
5. Significance via surrogate shuffling (100 surrogates)

**Parameters:**
- Discretization bins: 8
- History length: 3
- Lag: 1

### 2.6.3 Implementation
`src/metrics.py:TransferEntropyEstimator`

## 2.7 Signal Propagation Analysis

### 2.7.1 Experimental Setup
1. Initialize organism with critical parameters (μ = 0.15, σ = 0.015)
2. Evolve 100 steps to equilibrium
3. Place 6 measurement probes P₁...P₆ along horizontal axis
4. Probe regions: 8×8 pixels centered at each probe location

### 2.7.2 Correlation Analysis
- **Lagged cross-correlation:** Between adjacent probes with lag τ = 3 steps
- **Formula:** $r_{i,i+1}(\tau) = \text{corr}(\Delta_i(t), \Delta_{i+1}(t+\tau))$

**LIMITATION:** Correlation does not establish causation. Transfer entropy analysis (§2.6) is required to confirm directional information flow.

## 2.8 Reservoir Computing

### 2.8.1 Architecture
- **Reservoir:** Lenia simulation (48×48 grid, μ = 0.15, σ = 0.015)
- **Warmup:** 20 steps
- **Compute:** 30 steps after input injection

### 2.8.2 Feature Extraction
From final reservoir state:
- 128 randomly sampled pixel values
- 4 global statistics: mean, std, max, mass ratio

### 2.8.3 Evaluation
- **Model:** Logistic regression with L2 regularization (α = 1.0)
- **Dataset:** XOR with 16 samples
- **Validation:** 4-fold stratified cross-validation
- **Restarts:** 10 random restarts; report mean ± SD
- **Baseline:** Logistic regression on raw 2D inputs (50% for XOR)

## 2.9 Software and Reproducibility

- **Code:** Python 3.10+, NumPy, SciPy, scikit-learn
- **Determinism:** All experiments use fixed seeds; fully reproducible
- **Scripts:**
  - `experiment_transfer_entropy.py` — Transfer entropy analysis
  - `experiment_null_model.py` — Position invariance and regime comparison
  - `experiment_morphologies.py` — Cross-configuration NAND testing

---

## Parameter Summary (CORRECTED)

| Parameter | Original | Corrected | Notes |
|-----------|----------|-----------|-------|
| init_size | 0.22 | **0.20** | Original value caused dead organisms |
| warmup_steps | 80 | **100** | More stable initialization |
| hole_strength (s) | 0.5 | **0.8** | Critical correction - NAND only works at 0.8 |
| working_range | [0.45, 0.65] | **[0.75, 0.85]** | Corrected based on experiments |

---

## References

[11] Benettin, G., Galgani, L., Giorgilli, A., & Strelcyn, J. M. (1980). Lyapunov characteristic exponents for smooth dynamical systems and for Hamiltonian systems. Meccanica, 15(1), 9-30.

[12] Schreiber, T. (2000). Measuring information transfer. Physical Review Letters, 85(2), 461.
