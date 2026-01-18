"""
JAX-Accelerated Lenia Engine
Mirroring src/simulation.py but optimized for GPU/TPU and differentiability.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from typing import Tuple, NamedTuple, Callable

# Use NamedTuple for JAX-friendly config (PyTree compatible)
class JAXLeniaParams(NamedTuple):
    # Kernel
    mu: float
    sigma: float
    beta: float  # Growth peak (usually 1.0)
    
    # Grid
    dt: float
    
    # Kernel shape params
    k_width: float  # For ring kernel
    k_peak: float   # For ring kernel

def get_default_params() -> JAXLeniaParams:
    return JAXLeniaParams(
        mu=0.15,
        sigma=0.015,
        beta=1.0,
        dt=0.1,
        k_width=0.23,
        k_peak=0.5
    )

@jit
def sigmoid(x, a):
    return 1.0 / (1.0 + jnp.exp(-a * x))

@jit
def growth_gaussian(U, mu, sigma):
    """Gaussian growth function: 2 * exp(-(U-mu)^2 / 2sigma^2) - 1"""
    return 2.0 * jnp.exp(-((U - mu)**2) / (2.0 * sigma**2)) - 1.0

def get_kernel_fft(size: int, R: int, k_peak: float, k_width: float) -> jnp.ndarray:
    """Precompute kernel FFT for convolution."""
    x = jnp.linspace(-1, 1, 2*R + 1)
    # Use meshgrid with indexing='ij' or 'xy' carefully. 
    # Here we want a centered grid.
    xx, yy = jnp.meshgrid(x, x)
    dist = jnp.sqrt(xx**2 + yy**2)
    
    # Gaussian Ring
    # kernel = exp(-0.5 * ((dist - peak) / width)^2)
    kernel = jnp.exp(-0.5 * ((dist - k_peak) / k_width)**2)
    
    # Mask outside unit disk
    kernel = jnp.where(dist > 1.0, 0.0, kernel)
    
    # Normalize
    kernel = kernel / jnp.sum(kernel)
    
    # Pad to grid size
    # We pad to the right/bottom
    pad_h = size - kernel.shape[0]
    pad_w = size - kernel.shape[1]
    
    # Pad: ((0, pad_h), (0, pad_w)) -> places 27x27 kernel at top-left
    kernel_padded = jnp.pad(kernel, ((0, pad_h), (0, pad_w)))
    
    # Shift peak to (0,0)
    # The peak of the small kernel is at (R, R)
    # So we roll by (-R, -R)
    kernel_shifted = jnp.roll(kernel_padded, (-R, -R), axis=(0, 1))
    
    # FFT
    return jnp.fft.rfft2(kernel_shifted)

@jit
def step(state: jnp.ndarray, kernel_fft: jnp.ndarray, params: JAXLeniaParams) -> jnp.ndarray:
    """
    Single Lenia step.
    
    Args:
        state: Grid [H, W]
        kernel_fft: Precomputed FFT of kernel
        params: Simulation parameters
    
    Returns:
        Next state [H, W]
    """
    # 1. Potential field (Convolution)
    # FFT of state
    state_fft = jnp.fft.rfft2(state)
    
    # Convolution in freq domain
    potential_fft = state_fft * kernel_fft
    
    # Inverse FFT
    potential = jnp.fft.irfft2(potential_fft)
    
    # Fix shape if needed (sometimes irfft2 is slightly off due to odd/even)
    # Assuming square grid size power of 2, usually safe.
    
    # 2. Growth
    growth = growth_gaussian(potential, params.mu, params.sigma)
    
    # 3. Update
    new_state = state + params.dt * growth
    
    # Clip
    return jnp.clip(new_state, 0.0, 1.0)

# Batch step vectorization
step_batch = vmap(step, in_axes=(0, None, None))
step_batch_params = vmap(step, in_axes=(0, None, 0)) # Separate params per instance

class LeniaJAX:
    """
    Stateful wrapper for JAX Lenia.
    Keeps state on GPU/Device.
    """
    def __init__(self, size: int = 128, R: int = 13, seed: int = 42):
        self.size = size
        self.R = R
        self.key = jax.random.PRNGKey(seed)
        self.params = get_default_params()
        
        # Init kernel
        self.kernel_fft = get_kernel_fft(size, R, self.params.k_peak, self.params.k_width)
        
        # Init state (random noise for simple start)
        self.state = self._init_random()
        
    def _init_random(self):
        self.key, subkey = jax.random.split(self.key)
        # Orbium-like noise init
        # Simplified: random noise in center
        noise = jax.random.uniform(subkey, (self.size, self.size))
        
        # Mask
        x = jnp.linspace(-1, 1, self.size)
        xx, yy = jnp.meshgrid(x, x)
        dist = jnp.sqrt(xx**2 + yy**2)
        mask = jnp.where(dist < 0.25, 1.0, 0.0)
        
        return noise * mask

    def step(self):
        self.state = step(self.state, self.kernel_fft, self.params)
        
    def run(self, steps: int):
        # Scan is more efficient than python loop for JIT
        def body_fun(carry, _):
            state = carry
            new_state = step(state, self.kernel_fft, self.params)
            return new_state, None
            
        self.state, _ = jax.lax.scan(body_fun, self.state, jnp.arange(steps))
