# Conclusion

To our knowledge, we have presented the first systematic investigation of intrinsic computation in Lenia, a continuous cellular automaton exhibiting lifelike self-organizing patterns. Our findings establish three main results:

**1. Critical regime identification.** Scanning 1,600 parameter configurations, we identified a critical regime (~11% of explored space) where Lyapunov exponents approach zero (mean λ = +0.003, SD = 0.015). This regime coincides with stable, motile organisms, supporting the edge-of-chaos hypothesis for computation in self-organizing systems.

**2. Emergent NAND gate.** We discovered that Lenia organisms implement NAND-like computation through self-repair dynamics. Single perturbations (removing ~15% local mass) are healed; double simultaneous perturbations exceed the repair threshold, causing organism collapse. This pattern—survive unless both inputs active—matches the NAND truth table, validated across 80 trials (Fisher's exact test, p = 2.4 × 10⁻¹⁶) with robustness to position jitter (±3 pixels) and amplitude variation (±20%).

**3. Signal propagation.** Local perturbations propagate through organisms with measurable temporal structure. Lagged cross-correlations between adjacent measurement points reach r = 0.84, with correlation strength decreasing with distance from the perturbation source.

The NAND gate mechanism is notable for requiring no training, evolution, or engineering. Computation emerges spontaneously from the organism's homeostatic response to damage—a qualitatively novel form of emergent computation distinct from constructive logic gates (glider collisions) or reservoir computing (trained readouts).

Since NAND is functionally complete, these results open a theoretical pathway toward universal computation in continuous self-organizing systems. Realizing this potential requires demonstrating gate cascading—showing that one organism's collapse can trigger perturbation of another—which remains an open challenge.

Our work connects Lenia to broader research on unconventional computing, edge-of-chaos dynamics, and the computational capabilities of physical substrates. It suggests that self-repair mechanisms, ubiquitous in biological systems, may serve not only homeostatic but also computational functions.

---

## Acknowledgments

[To be added]

## Author Contributions

[To be added]

## Data Availability

All code and data are available at https://github.com/m4rba4s/lenia-criticality-lab

## Competing Interests

The authors declare no competing interests.
