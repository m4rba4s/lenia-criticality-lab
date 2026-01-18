
import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import jax.numpy as jnp
from src.simulation import LeniaSimulation, LeniaConfig
from src.engine_jax import LeniaJAX, JAXLeniaParams

def compare_engines(steps=100, size=128):
    print(f"Comparing Engines ({steps} steps)...")
    
    # 1. Setup Scipy Sim
    config = LeniaConfig(grid_size=size, seed=42)
    scipy_sim = LeniaSimulation(config)
    
    # Get initial state
    init_state = scipy_sim.get_state()
    
    # 2. Setup JAX Sim
    jax_sim = LeniaJAX(size=size)
    jax_sim.params = JAXLeniaParams(
        mu=config.mu,
        sigma=config.sigma,
        beta=1.0,
        dt=config.dt,
        k_width=config.kernel_width,
        k_peak=config.kernel_peak
    )
    
    # Inject state
    jax_sim.state = jnp.array(init_state)
    
    # 3. Run
    scipy_sim.run(steps)
    scipy_final = scipy_sim.get_state()
    
    jax_sim.run(steps)
    jax_final = np.array(jax_sim.state)
    
    # 4. Compare
    diff = scipy_final - jax_final
    abs_diff = np.abs(diff)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    
    print(f"Max Difference: {max_diff:.6f}")
    print(f"Mean Difference: {mean_diff:.6f}")
    
    # Debug points
    mid = size // 2
    print(f"Center Scipy: {scipy_final[mid, mid]:.6f}")
    print(f"Center JAX:   {jax_final[mid, mid]:.6f}")
    
    if max_diff < 1e-4:
        print("✅ PASS: Engines match!")
    else:
        print("❌ FAIL: Divergence detected.")

if __name__ == "__main__":
    compare_engines()
