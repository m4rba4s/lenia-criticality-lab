# Introduction

## 1.1 Computation in Self-Organizing Systems

The question of how physical systems can perform computation has fascinated researchers across disciplines, from Turing's foundational work on abstract machines to modern investigations of neuromorphic and unconventional computing substrates [1]. A particularly intriguing class of systems are those that compute not through explicit programming, but through their intrinsic dynamics—where the physics of the system itself implements the computation [2].

Cellular automata (CA) have long served as a bridge between these perspectives. Conway's Game of Life demonstrated that simple local rules can give rise to universal computation through emergent structures like gliders and glider guns [3]. However, the discrete nature of classical CA limits their applicability as models of natural computation, where continuous dynamics and smooth gradients are the norm.

## 1.2 Lenia: Continuous Cellular Automata

Lenia, introduced by Chan [4], generalizes cellular automata to continuous space, time, and state. The system evolves according to:

$$A^{t+dt} = \text{clip}\left(A^t + dt \cdot G(K * A^t)\right)$$

where $A^t$ is the continuous state field, $K$ is a smooth kernel (typically ring-shaped), $G$ is a growth function parameterized by $(\mu, \sigma)$, and $*$ denotes convolution. This formulation produces remarkably lifelike "organisms"—self-organizing patterns that move, grow, and interact in ways reminiscent of biological cells.

Despite extensive cataloging of Lenia's morphological diversity [4,5], its computational capabilities remain largely unexplored. Can these self-organizing patterns process information? Can they implement logical operations without explicit programming?

## 1.3 The Edge of Chaos Hypothesis

A guiding principle from complexity science suggests that computation is optimized at the boundary between order and chaos [6,7]. In ordered regimes, perturbations decay rapidly, preventing information transmission. In chaotic regimes, small differences amplify exponentially, destroying stored information. At the critical boundary—the "edge of chaos"—systems exhibit a delicate balance: sensitive enough to respond to inputs, yet stable enough to maintain coherent outputs.

This principle has been validated in diverse contexts: Boolean networks [8], reservoir computing [9], and neural systems [10]. The largest Lyapunov exponent $\lambda$ serves as the canonical order parameter: $\lambda < 0$ indicates ordered dynamics, $\lambda > 0$ indicates chaos, and $\lambda \approx 0$ marks the critical regime.

For Lenia, this raises a natural question: do the lifelike patterns that Chan and others have documented correspond to critical dynamics? And if so, does criticality enable computation?

## 1.4 Our Contribution

To our knowledge, we present the first systematic investigation of computation in Lenia. Our contributions are:

1. **Phase diagram characterization.** We map 1,600 parameter configurations in the $(\mu, \sigma)$ space, estimating Lyapunov exponents via Benettin's algorithm. We identify a critical regime where $\lambda \approx 0$, corresponding to the parameter region that supports stable, motile organisms.

2. **Signal propagation.** We demonstrate that local perturbations propagate through Lenia organisms with measurable temporal structure. Lagged cross-correlations between spatially separated measurement points reach $r = 0.84$, with correlation strength decreasing with distance.

3. **Emergent NAND gate.** We discover that Lenia organisms implement NAND-like computation through self-repair dynamics. Single perturbations (removing ~15% local mass) are absorbed and healed. Two simultaneous perturbations exceed the repair threshold, causing organism collapse. This survive-unless-both-active pattern matches the NAND truth table—a functionally complete gate from which any Boolean function can be constructed.

4. **Statistical validation.** We validate the NAND behavior across 80 trials with randomized seeds (Fisher's exact test, $p = 2.4 \times 10^{-16}$) and demonstrate robustness to position jitter (±3 pixels) and amplitude variation (±20%).

## 1.5 Significance

Our findings suggest that self-organizing continuous cellular automata can realize computational primitives through intrinsic dynamics, without explicit programming of logical operations. The mechanism—a damage threshold arising from self-repair—is qualitatively different from the constructive logic gates of discrete CA (glider collisions in Life) or the trained readouts of reservoir computing. Instead, computation emerges from the organism's homeostatic response to perturbation.

Since NAND is functionally complete, these results open a pathway toward richer computation in continuous self-organizing systems. Key challenges remain: demonstrating signal cascading between gates, validating across diverse morphologies, and establishing causal (not merely correlational) information flow.

## 1.6 Paper Organization

The remainder of this paper is organized as follows. Section 2 describes our methods: the Lenia simulation framework, Lyapunov estimation via Benettin's algorithm, the NAND gate protocol, and signal propagation analysis. Section 3 presents results: phase diagram structure, NAND gate validation, robustness analysis, and signal propagation measurements. Section 4 discusses implications for unconventional computing, limitations, and future directions. Section 5 concludes.

---

## References

[1] Adamatzky, A. (Ed.). (2017). Advances in Unconventional Computing. Springer.

[2] Jaeger, H. (2021). Towards a generalized theory comprising digital, neuromorphic and unconventional computing. Neuromorphic Computing and Engineering, 1(1).

[3] Berlekamp, E. R., Conway, J. H., & Guy, R. K. (1982). Winning Ways for Your Mathematical Plays, Vol. 2. Academic Press.

[4] Chan, B. W. C. (2019). Lenia: Biology of Artificial Life. Complex Systems, 28(3), 251-286.

[5] Chan, B. W. C. (2020). Lenia and Expanded Universe. ALIFE 2020: The 2020 Conference on Artificial Life, 221-229.

[6] Langton, C. G. (1990). Computation at the edge of chaos: Phase transitions and emergent computation. Physica D, 42(1-3), 12-37.

[7] Packard, N. H. (1988). Adaptation toward the edge of chaos. Dynamic Patterns in Complex Systems, 293-301.

[8] Kauffman, S. A. (1993). The Origins of Order: Self-Organization and Selection in Evolution. Oxford University Press.

[9] Boedecker, J., Obst, O., Lizier, J. T., Mayer, N. M., & Asada, M. (2012). Information processing in echo state networks at the edge of chaos. Theory in Biosciences, 131(3), 205-213.

[10] Beggs, J. M., & Plenz, D. (2003). Neuronal avalanches in neocortical circuits. Journal of Neuroscience, 23(35), 11167-11177.
