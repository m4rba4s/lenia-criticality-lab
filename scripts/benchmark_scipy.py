
import time
import psutil
import sys
import os
sys.path.append(os.getcwd())  # Add current directory to path
import numpy as np
from src.simulation import LeniaSimulation, LeniaConfig

def run_benchmark(steps=1000, size=128):
    print(f"Benchmarking Scipy Engine (Grid: {size}x{size}, Steps: {steps})...")
    
    config = LeniaConfig(grid_size=size, seed=42)
    sim = LeniaSimulation(config)
    
    # Warmup
    sim.run(10)
    
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024
    
    start = time.time()
    for _ in range(steps):
        sim.step()
    end = time.time()
    
    mem_after = process.memory_info().rss / 1024 / 1024
    
    duration = end - start
    fps = steps / duration
    
    print(f"Duration: {duration:.3f}s")
    print(f"FPS: {fps:.2f}")
    print(f"Memory: {mem_before:.1f}MB -> {mem_after:.1f}MB")
    return fps

if __name__ == "__main__":
    run_benchmark(steps=500, size=128)
    # run_benchmark(steps=100, size=256)
