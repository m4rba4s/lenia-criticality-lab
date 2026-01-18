# Methods

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
1. Initialize with random blob (size 0.2, density 0.5, seed=42 for morphology discovery)
2. Evolve 300 steps
3. Record: mean mass, final mass
4. Estimate Lyapunov exponent (separate procedure, see §2.3)

**Note:** Phase diagram morphology uses fixed seed=42 for reproducibility. Lyapunov estimation uses 3 trials with different seeds and averages them.

**Classification criteria:**
- Dead: final mass < 10
- Alive: final mass ≥ 10
- Critical: |λ| < 0.01 (based on averaged λ from 3 trials)

**Scope of criticality claim:** We use λ ≈ 0 as an operational proxy for near-critical dynamics. Full characterization of criticality would require additional signatures (power-law correlations, diverging susceptibility, 1/f spectra) which we do not measure here. Our claim is therefore that stable Lenia organisms operate in a *near-critical regime* where perturbations neither decay rapidly (ordered) nor amplify unboundedly (chaotic).

## 2.3 Lyapunov Exponent Estimation

We estimated the largest Lyapunov exponent λ using Benettin's algorithm [11] with explicit renormalization:

**Algorithm:**
1. Initialize reference trajectory $W_{ref}$ and perturbed trajectory $W_{pert} = W_{ref} + \delta_0$
2. Initial perturbation: $\|\delta_0\|_2 = \varepsilon = 10^{-8}$ (uniform noise, L2 norm)
3. Warmup: evolve both trajectories for $T_{warm} = 100$ steps
4. Measurement phase ($T_{measure} = 500$ steps):
   - Every $\tau = 10$ steps:
     - Compute separation: $\delta_t = W_{pert} - W_{ref}$
     - Record stretching: $\Lambda_t = \log(\|\delta_t\|_2 / \varepsilon)$
     - **Renormalize:** $\delta_t \leftarrow \varepsilon \cdot \delta_t / \|\delta_t\|_2$
     - Update: $W_{pert} \leftarrow W_{ref} + \delta_t$
5. Estimate: $\lambda = \frac{1}{n} \sum_t \Lambda_t / \tau$ where $n = T_{measure}/\tau$

**Averaging:** 3 independent trials per configuration (different seeds). Per-configuration λ is the mean of 3 trials. Results section reports across-configuration statistics: mean ± SD for regime summaries, with SEM for confidence intervals.

**Rationale:** Renormalization prevents numerical overflow/underflow while accumulating the log-stretching factors that define λ.

**Implementation:** `src/metrics.py:LyapunovEstimator`

## 2.4 NAND Gate Protocol

### 2.4.1 Organism Preparation
1. Initialize grid 128×128 with parameters μ = 0.15, σ = 0.015
2. Evolve 80 steps to reach stable configuration
3. Verify mass > 50 (organism alive)
4. Locate organism centroid $(c_y, c_x)$ as center of mass

### 2.4.2 Perturbation Application
For each input condition (A, B) ∈ {(0,0), (0,1), (1,0), (1,1)}:

**Hole template:** Gaussian blob $H(x,y) = \exp(-(x^2+y^2)/(2\sigma_h^2))$ with $\sigma_h = 6$ pixels, peak strength $s = 0.5$.

**Perturbation positions:**
- Input A: center at $(c_y, c_x - 14)$
- Input B: center at $(c_y, c_x + 14)$

**Application:** If input = 1, multiply local region by $(1 - H)$, removing ~15% of local mass.

### 2.4.3 Collapse Criterion
After perturbation, evolve 200 steps. Classify outcome:

$$\text{Outcome} = \begin{cases} \text{ALIVE} & \text{if } M_{t=200} > \tau_{collapse} \\ \text{COLLAPSED} & \text{if } M_{t=200} \leq \tau_{collapse} \end{cases}$$

where $M = \sum_{i,j} A_{i,j}$ is total mass and $\tau_{collapse} = 25$.

**Threshold selection:** Baseline organism mass ≈ 73. Threshold τ = 25 represents ~34% of baseline. Results robust to τ ∈ [20, 35].

### 2.4.4 Statistical Validation
- **Trials:** 20 per input condition, 80 total
- **Randomization:** Different seed per trial (seed = 42 + trial × 100)
- **Test:** Fisher's exact test comparing single-input (01/10 pooled, n=40) vs double-input (11, n=20) survival rates
- **Robustness tests:**
  - Position jitter: offset ±3 pixels (n=20 trials)
  - Amplitude variation: strength ±20% (n=20 trials)

## 2.5 Signal Propagation Analysis

### 2.5.1 Experimental Setup
1. Initialize organism with critical parameters (μ = 0.1585, σ = 0.01808)
2. Evolve 80 steps to equilibrium
3. Place 6 measurement probes P₁...P₆ along horizontal axis of organism
4. Probe regions: 16×16 pixels centered at each probe location

### 2.5.2 Perturbation and Measurement
1. Record baseline activity: $B_i = \text{mean}(A_{probe_i})$ for each probe
2. Apply Gaussian perturbation at P₁ (leftmost): peak amplitude 0.4
3. Track activity for 120 steps at 3-step intervals (n=40 timepoints)
4. Compute activity change: $\Delta_i(t) = \text{mean}(A_{probe_i}(t)) - B_i$

### 2.5.3 Correlation Analysis
- **Lagged cross-correlation:** Between adjacent probes with lag τ = 5 steps
- **Formula:** $r_{i,i+1}(\tau) = \text{corr}(\Delta_i(t), \Delta_{i+1}(t+\tau))$
- **Distance trend:** Visual inspection suggests correlation magnitude decreases with distance from source (n = 5 probe pairs). We note that correlation does not establish causation; transfer entropy analysis would be required to confirm directional information flow, which we leave to future work.

## 2.6 Reservoir Computing

### 2.6.1 Architecture
- **Reservoir:** Lenia simulation (48×48 grid, μ = 0.15, σ = 0.015)
- **Warmup:** 20 steps
- **Compute:** 30 steps after input injection

### 2.6.2 Input Encoding
Inputs encoded as Gaussian blobs at fixed positions around organism center:
- Position determined by input index (circular arrangement)
- Intensity proportional to input value

### 2.6.3 Feature Extraction
From final reservoir state, extract:
- 128 randomly sampled pixel values (indices fixed once per restart, held constant across all samples within that restart — anti-leakage measure)
- 4 global statistics: mean, std, max, mass ratio

Feature vector dimension: 132

### 2.6.4 Readout and Evaluation
- **Model:** Logistic regression with L2 regularization (α = 1.0)
- **Dataset:** XOR with 16 samples (4 unique patterns × 4 repetitions, different seeds)
- **Validation:** 4-fold stratified cross-validation (each fold contains one instance of each pattern)
- **Restarts:** 10 random restarts with different feature index selections; report mean ± SD
- **Baseline:** Logistic regression on raw 2D inputs (expected 50% for XOR, which is linearly inseparable)

## 2.7 Software and Reproducibility

- **Code:** Python 3.10+, NumPy 1.20+, SciPy 1.7+, scikit-learn 1.0+
- **Repository:** https://github.com/m4rba4s/lenia-criticality-lab
- **Determinism:** All experiments use fixed seeds; results reproducible given same seed
- **Compute:** All experiments run on single CPU core (Intel i7), total runtime ~2 hours for phase diagram

---

## References

[11] Benettin, G., Galgani, L., Giorgilli, A., & Strelcyn, J. M. (1980). Lyapunov characteristic exponents for smooth dynamical systems and for Hamiltonian systems. Meccanica, 15(1), 9-30.
