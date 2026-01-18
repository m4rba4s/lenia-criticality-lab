
import sys
import os
sys.path.append(os.getcwd())
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import time

from src.metrics_jax import compute_lyapunov_batch, compute_te_pairwise
from src.engine_jax import get_kernel_fft, get_default_params

def benchmark_lyapunov():
    print("="*40)
    print("Benchmarking Batch Lyapunov...")
    
    batch_size = 100
    size = 128
    steps = 100
    
    print(f"Batch: {batch_size}, Grid: {size}x{size}, Steps: {steps}")
    
    # Setup
    key = random.PRNGKey(0)
    states = random.uniform(key, (batch_size, size, size))
    params = get_default_params()
    k_fft = get_kernel_fft(size, 13, params.k_peak, params.k_width)
    
    # Warmup JIT
    print("Compiling...")
    start_comp = time.time()
    _, _ = compute_lyapunov_batch(states, k_fft, params, 10, 10)
    jax.block_until_ready(states)
    end_comp = time.time()
    print(f"Compilation: {end_comp - start_comp:.3f}s")
    
    # Run
    start = time.time()
    # Using 10 warmup, 100 measure for speed test
    exps, _ = compute_lyapunov_batch(states, k_fft, params, 10, steps)
    exps.block_until_ready()
    end = time.time()
    
    duration = end - start
    total_ops = batch_size * steps
    
    print(f"Duration: {duration:.3f}s")
    print(f"Throughput: {total_ops / duration:.0f} sim-steps/sec")
    print(f"Effective Sim Speed: {(total_ops / duration):.0f} FPS")
    print(f"Result mean: {jnp.mean(exps):.4f}")

def benchmark_te():
    print("\n" + "="*40)
    print("Benchmarking Transfer Entropy (pairwise)...")
    
    T = 1000
    n_pairs = 1000
    
    print(f"Time series length: {T}, Pairs: {n_pairs}")
    
    # Generate random data
    key = random.PRNGKey(1)
    x = random.normal(key, (n_pairs, T))
    y = random.normal(key, (n_pairs, T))
    
    # JIT the batch version
    # Map over the 0-th axis of inputs
    batch_te = jax.vmap(compute_te_pairwise, in_axes=(0, 0, None, None, None))
    
    # Warmup
    print("Compiling...")
    start_comp = time.time()
    _ = batch_te(x, y, 8, 1, 1)
    jax.block_until_ready(x)
    end_comp = time.time()
    print(f"Compilation: {end_comp - start_comp:.3f}s")
    
    # Run
    start = time.time()
    results = batch_te(x, y, 8, 1, 1)
    results.block_until_ready()
    end = time.time()
    
    duration = end - start
    print(f"Duration: {duration:.3f}s")
    print("Pairs/sec: {:.0f}".format(n_pairs / duration))
    print(f"Mean TE (Random): {jnp.mean(results):.4f}")
    
    # Validation: Causal link
    print("\nValidating Causality (Y = X shifted)...")
    x = random.normal(key, (10, T))
    # y = x shifted by 1
    y = jnp.roll(x, 1, axis=1)
    
    # Compute TE(X->Y) with lag=1
    te_causal = batch_te(x, y, 8, 1, 1)
    print(f"TE(X->Y) Causal: {jnp.mean(te_causal):.4f}")
    
    # Should be high (Max entropy of discrete X approx log2(8) = 3)
    if jnp.mean(te_causal) > 2.0:
        print("✅ PASS: Causality detected!")
    else:
        print("❌ FAIL: TE too low for perfect copy.")

if __name__ == "__main__":
    benchmark_lyapunov()
    benchmark_te()
