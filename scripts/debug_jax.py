
import sys
import os
sys.path.append(os.getcwd())
import jax.numpy as jnp
import numpy as np
from src.engine_jax import get_kernel_fft, get_default_params, step, JAXLeniaParams

def debug_kernel():
    size = 128
    R = 13
    params = get_default_params()
    
    print("Generating Kernel FFT...")
    k_fft = get_kernel_fft(size, R, params.k_peak, params.k_width)
    
    print(f"Kernel FFT Shape: {k_fft.shape}")
    print(f"Kernel FFT DC (0,0): {k_fft[0,0]}")
    
    if jnp.abs(k_fft[0,0] - 1.0) > 1e-3:
        print("❌ WARNING: Kernel sums to non-1!")
        
    # Check spatial kernel
    # Inverse FFT to check spatial layout
    k_spatial = jnp.fft.irfft2(k_fft)
    # Undo shift to see it visually centered (for debug print)
    k_centered = jnp.fft.fftshift(k_spatial)
    
    print(f"Spatial Sum: {jnp.sum(k_spatial)}")
    print(f"Spatial Max: {jnp.max(k_spatial)}")
    print(f"Spatial Min: {jnp.min(k_spatial)}")
    
    # Check center value (should be high)
    mid = size // 2
    print(f"Center value [mid, mid]: {k_centered[mid, mid]}")
    
    # Run one step with uniform state
    state = jnp.ones((size, size))
    # If state = 1 everywhere, potential should be 1 everywhere (since kernel sums to 1)
    
    potential_fft = jnp.fft.rfft2(state) * k_fft
    potential = jnp.fft.irfft2(potential_fft)
    
    print(f"Potential (Uniform Input) Mean: {jnp.mean(potential)}")
    print(f"Potential (Uniform Input) Min: {jnp.min(potential)}")
    
    if jnp.min(potential) < 0.99:
        print("❌ FAIL: Convolution failed (potential != 1 for uniform input)")
    else:
        print("✅ PASS: Convolution logic seems basic ok")

if __name__ == "__main__":
    debug_kernel()
