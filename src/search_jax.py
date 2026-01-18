"""
Differentiable Parameter Search for Lenia
"""

import jax
import jax.numpy as jnp
from jax import jit, grad, value_and_grad
from typing import Tuple, NamedTuple
import optax  # Use optax for optimization if available, else simple GD

from src.engine_jax import step, JAXLeniaParams, get_kernel_fft

# Define a differentiable loss function
# Goal: Find parameters that lead to "complex" dynamics
# Proxy: Maximize variance of mass over time, while keeping mass within bounds (not 0, not 1)

def simulation_loss(params_array, init_state, kernel_fft_base, steps=200):
    """
    Loss function for Lenia parameters.
    params_array: [mu, sigma]
    
    We want to MAXIMIZE complexity, so we MINIMIZE negative variance.
    """
    mu, sigma = params_array
    
    # Reconstruct params tuple (we optimize mu, sigma only for now)
    # We need to treat params as JAX array for differentiation to work on them
    # But step expects NamedTuple. JIT will handle this if we unpack correctly.
    
    # NOTE: We assume kernel shape (beta, k_width, k_peak) is fixed or pre-calculated
    # If we want to optimize k_width, we'd need to recompute kernel_fft inside the loop
    # which is expensive but possible. For now, let's stick to growth params (mu, sigma).
    
    # We use the kernel_fft passed in.
    
    # Fixed params
    dt = 0.1
    
    # We need a custom params object that works with the step function
    # The step function expects JAXLeniaParams named tuple.
    # We can construct it here.
    
    current_params = JAXLeniaParams(
        mu=mu,
        sigma=sigma,
        beta=1.0,
        dt=dt,
        k_width=0.0, # Unused in step if kernel is precomputed
        k_peak=0.0   # Unused
    )
    
    # Run simulation
    def body_fun(carry, _):
        state = carry
        new_state = step(state, kernel_fft_base, current_params)
        # Return state as carry, and mass as output for analysis
        return new_state, jnp.mean(new_state)
        
    final_state, masses = jax.lax.scan(body_fun, init_state, None, length=steps)
    
    # Loss Formulation
    
    mass_std = jnp.std(masses)
    mean_mass = jnp.mean(masses)
    
    # Loss Formulation for Solitons
    # Solitons are localized: Mass small but non-zero.
    # Solitons are active: Variance > 0.
    
    # 1. Target Mass Loss (Crucial for localization)
    target_mass = 0.15
    loss_mass = 10.0 * (mean_mass - target_mass)**2
    
    # 2. Activity Reward
    # We want variance, but not infinite variance.
    # Just enough to be "alive".
    # Log-variance is often smoother.
    loss_activity = -jnp.log(mass_std + 1e-6)
    
    # Combined Loss
    # If mass is wrong, the mass term dominates.
    # If mass is correct, we optimize for activity.
    total_loss = loss_mass + 0.1 * loss_activity
    
    return total_loss, (mass_std, mean_mass)

@jit
def update_step(params_array, opt_state, init_state, kernel_fft):
    """Perform one gradient update step."""
    # params_array is [mu, sigma]
    
    (loss, aux), grads = value_and_grad(simulation_loss, has_aux=True)(params_array, init_state, kernel_fft)
    
    # Simple Gradient Descent with momentum?
    # Or just use optax if we had it. Let's write simple SGD first.
    lr = 0.001
    
    # Update
    # params = params - lr * grads
    new_params = params_array - lr * grads
    
    return new_params, opt_state, loss, aux, grads

class CriticalitySeeker:
    def __init__(self, size=64):
        self.size = size
        self.key = jax.random.PRNGKey(42)
        
        # Init fixed kernel basis
        # We optimize mu/sigma, so kernel shape is constant
        from src.engine_jax import get_default_params
        defaults = get_default_params()
        self.kernel_fft = get_kernel_fft(size, 13, defaults.k_peak, defaults.k_width)
        
    def find_critical_parameters(self, start_mu=0.10, start_sigma=0.02, steps=100):
        print(f"Searching for criticality starting from mu={start_mu}, sigma={start_sigma}...")
        
        current_params = jnp.array([start_mu, start_sigma])
        
        # Init state (random noise)
        self.key, subkey = jax.random.split(self.key)
        init_state = jax.random.uniform(subkey, (self.size, self.size))
        # Mask to circle
        x = jnp.linspace(-1, 1, self.size)
        xx, yy = jnp.meshgrid(x, x)
        mask = jnp.where(xx**2 + yy**2 < 0.5, 1.0, 0.0)
        init_state = init_state * mask
        
        history = []
        
        for i in range(steps):
            current_params, _, loss, (std, mean), grads = update_step(
                current_params, None, init_state, self.kernel_fft
            )
            
            # Clip params to sane ranges to prevent nan
            # mu > 0, sigma > 0.001
            current_params = jnp.clip(current_params, a_min=jnp.array([0.01, 0.001]), a_max=jnp.array([1.0, 0.2]))
            
            if i % 10 == 0:
                print(f"Step {i}: Loss={loss:.4f} | Mu={current_params[0]:.4f} Sigma={current_params[1]:.4f} | Std={std:.4f} Mean={mean:.2f} | Grads={grads}")
                history.append((float(loss), float(current_params[0]), float(current_params[1])))
                
        return current_params, history

