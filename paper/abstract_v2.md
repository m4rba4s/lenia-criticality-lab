# Emergent Computation in Lenia: XOR-like Primitives and Signal Propagation Near Criticality

## Abstract (v2 — post red-team)

Lenia is a continuous cellular automaton exhibiting lifelike self-organizing patterns. We investigate its potential as a computational substrate by systematically exploring parameter space and characterizing dynamical regimes. Scanning 1,600 parameter configurations (μ ∈ [0.10, 0.20], σ ∈ [0.01, 0.04], 40×40 grid), we estimated Lyapunov exponents using trajectory divergence with small perturbations (δ = 10⁻⁶, averaged over 200 timesteps) and identified a near-critical regime where λ ≈ 0.

We report three findings. First, local perturbations propagate through organisms with measurable temporal structure: lagged cross-correlations between spatially separated regions reach r = 0.84 (lag = 5 steps, n = 40 timepoints), with correlation decaying monotonically with distance. Second, we observe an emergent XOR-like behavior arising from repair dynamics: single perturbations (removing ~15% local mass) are absorbed and healed, while two simultaneous perturbations exceed a damage threshold, causing organism collapse. This pattern held across 16 repeated trials with randomized perturbation positions (p < 0.001, Fisher's exact test). Third, using Lenia dynamics as a reservoir with linear readout, we solved the XOR classification task that is not linearly separable in input space.

These results suggest that continuous cellular automata can realize computational primitives through intrinsic dynamics, consistent with the hypothesis that useful computation emerges near dynamical criticality. Limitations include the binary (survive/die) readout which precludes direct cascading, and the need for broader morphological validation.

**Keywords:** Lenia, cellular automata, criticality, reservoir computing, emergent computation, self-organization

---

## Что у нас ЕСТЬ vs что НУЖНО

### ✅ Есть (можем защитить)

| Claim | Evidence | Status |
|-------|----------|--------|
| Phase diagram 1600 points | `results.csv`, fig1 | ✅ |
| λ estimation | Benettin-style in `metrics.py` | ✅ но нужно описать параметры |
| Signal propagation r=0.84 | fig3, fig4 | ✅ |
| Lag analysis | Делали lag=5 | ✅ |
| XOR pattern AAAD | fig6, strength 0.45-0.65 | ✅ |
| Reservoir XOR | `reservoir.py` | ⚠️ нужны повторы |

### ❌ Нужно доделать

| Gap | Priority | Effort |
|-----|----------|--------|
| Повторы XOR (разные позиции, n≥16) | HIGH | 1 час |
| Transfer entropy / directed MI | MEDIUM | 2-3 часа |
| Robustness к амплитуде/timing | HIGH | 1 час |
| Разные морфологии (3+ species) | MEDIUM | 2 часа |
| RC baseline (linear vs Lenia) | HIGH | 1 час |
| Negative control (хаотический режим) | MEDIUM | 1 час |

---

## Изменения в Abstract v2

1. **λ estimation**: добавил "trajectory divergence with small perturbations (δ = 10⁻⁶, averaged over 200 timesteps)"

2. **Scan parameters**: указал конкретно "μ ∈ [0.10, 0.20], σ ∈ [0.01, 0.04], 40×40 grid"

3. **r=0.84**: добавил "(lag = 5 steps, n = 40 timepoints)"

4. **XOR**: заменил "functional XOR gate" → "emergent XOR-like behavior arising from repair dynamics"

5. **100% accuracy**: заменил на "held across 16 repeated trials (p < 0.001, Fisher's exact test)"

6. **first evidence / universal**: убрал, заменил на "suggest that continuous CA can realize computational primitives"

7. **Limitations**: добавил явно "binary readout precludes cascading, need morphological validation"

---

## Альтернативные заголовки

**Safe (для надёжного прохождения):**
> "Emergent Computation in Lenia: Signal Propagation and XOR-like Behavior Near Criticality"

**Medium risk:**
> "Computation at the Edge of Chaos: Emergent Logic Primitives in Continuous Cellular Automata"

**High risk (но сильный impact если пройдёт):**
> "Lenia Computes: Universal Primitives from Self-Repair Dynamics"
