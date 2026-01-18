
import sys
import os
import time
sys.path.append(os.getcwd())
import jax
import jax.numpy as jnp
from src.engine_jax import LeniaJAX

def run_jax_benchmark(steps=1000, size=128):
    print(f"Benchmarking JAX Engine (Grid: {size}x{size}, Steps: {steps})...")
    print(f"Device: {jax.devices()[0]}")
    
    # Init
    start_init = time.time()
    sim = LeniaJAX(size=size)
    # Force compilation
    sim.step() 
    sim.state.block_until_ready()
    end_init = time.time()
    print(f"Initialization + JIT Compilation: {end_init - start_init:.3f}s")
    
    # Run loop
    start = time.time()
    # using scan internally
    sim.run(steps)
    sim.state.block_until_ready() # Wait for GPU
    end = time.time()
    
    duration = end - start
    fps = steps / duration
    
    print(f"Duration: {duration:.3f}s")
    print(f"FPS: {fps:.2f}")
    
    return fps

if __name__ == "__main__":
    run_jax_benchmark(steps=2000, size=128)
