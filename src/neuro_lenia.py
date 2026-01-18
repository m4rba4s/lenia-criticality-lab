"""
Neuro-Lenia: Differentiable Lenia Layer using Equinox.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from src.engine_jax import growth_gaussian

class LeniaLayer(eqx.Module):
    conv: eqx.nn.Conv2d
    mu: jnp.ndarray
    sigma: jnp.ndarray
    dt: float
    kernel_size: int

    def __init__(self, key):
        self.kernel_size = 27
        self.dt = 0.1
        
        # Kernel initialization
        k_size = self.kernel_size
        center = k_size // 2
        x = jnp.linspace(-1, 1, k_size)
        xx, yy = jnp.meshgrid(x, x)
        r = jnp.sqrt(xx**2 + yy**2)
        kernel = jnp.exp(-((r - 0.5)**2) / 0.05)
        kernel = kernel / (jnp.sum(kernel) + 1e-10)
        kernel = kernel.reshape(1, 1, k_size, k_size)
        
        # Init Conv2d
        # Equinox Conv2d expects inputs (C, H, W) usually
        self.conv = eqx.nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=k_size,
            padding=0, # Manual padding later
            use_bias=False,
            key=key
        )
        
        # Override weights
        self.conv = eqx.tree_at(lambda c: c.weight, self.conv, kernel)
        
        # Trainable params
        self.mu = jnp.array([0.15])
        self.sigma = jnp.array([0.015])

    def __call__(self, state):
        # state: (1, H, W) - explicit channels first for Equinox
        
        pad = self.kernel_size // 2
        # Pad (C, H, W) -> pad H and W
        state_padded = jnp.pad(state, ((0,0), (pad,pad), (pad,pad)), mode='wrap')
        
        potential = self.conv(state_padded)
        
        # Growth
        growth = growth_gaussian(potential, self.mu, self.sigma)
        
        new_state = jnp.clip(state + self.dt * growth, 0.0, 1.0)
        return new_state

class LeniaRNN(eqx.Module):
    cell: LeniaLayer
    steps: int

    def __init__(self, key, steps=50):
        self.cell = LeniaLayer(key)
        self.steps = steps

    def __call__(self, x):
        # x: (1, H, W)
        
        def scan_fn(carry, _):
            new_state = self.cell(carry)
            return new_state, new_state
            
        final_state, history = jax.lax.scan(scan_fn, x, None, length=self.steps)
        return final_state, history
