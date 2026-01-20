"""
Locomotion Training for Neuro-Lenia using Simple Evolutionary Strategy.
Hand-rolled ES (no external dependency issues).
"""

import sys
import os
sys.path.append(os.getcwd())

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from src.neuro_lenia import LeniaRNN

def compute_com(grid):
    """Compute Center of Mass (CoM) of a 2D grid."""
    if grid.ndim == 3:
        grid = grid[0]
    
    H, W = grid.shape
    total_mass = jnp.sum(grid) + 1e-10
    
    y_coords = jnp.linspace(0, 1, H)
    x_coords = jnp.linspace(0, 1, W)
    
    x_com = jnp.sum(grid * x_coords[None, :]) / total_mass
    y_com = jnp.sum(grid * y_coords[:, None]) / total_mass
    
    return x_com, y_com

def create_initial_blob(size, center_x=0.3, radius=0.1):
    """Create an initial blob on the left side of the grid."""
    x = jnp.linspace(0, 1, size)
    xx, yy = jnp.meshgrid(x, x)
    
    r = jnp.sqrt((xx - center_x)**2 + (yy - 0.5)**2)
    blob = jnp.exp(-r**2 / (2 * radius**2))
    
    return blob[None, ...]

def train_locomotion_es():
    print("=" * 50)
    print("Phase 6: Locomotion RL — Simple ES")
    print("=" * 50)
    
    size = 64
    simulation_steps = 50
    population_size = 20
    generations = 30
    sigma_noise = 0.02  # Mutation strength
    
    key = jax.random.PRNGKey(42)
    init_state = create_initial_blob(size, center_x=0.25, radius=0.12)
    
    # We optimize 2 params: mu and sigma
    # Start with known good values
    best_params = jnp.array([0.18, 0.04])  # [mu, sigma]
    
    # Bounds
    param_min = jnp.array([0.05, 0.005])
    param_max = jnp.array([0.5, 0.15])
    
    @jax.jit
    def evaluate(params, rng_key):
        """Evaluate fitness of params. Returns displacement (higher = better)."""
        mu_val = jnp.clip(params[0], param_min[0], param_max[0])
        sigma_val = jnp.clip(params[1], param_min[1], param_max[1])
        
        model = LeniaRNN(rng_key, steps=simulation_steps)
        model = eqx.tree_at(lambda m: m.cell.mu, model, jnp.array([mu_val]))
        model = eqx.tree_at(lambda m: m.cell.sigma, model, jnp.array([sigma_val]))
        
        final_state, _ = model(init_state)
        
        x_init, _ = compute_com(init_state)
        x_final, _ = compute_com(final_state)
        displacement = x_final - x_init
        
        final_mass = jnp.sum(final_state)
        init_mass = jnp.sum(init_state)
        
        # Penalty for death
        alive = jnp.where(final_mass > 0.1 * init_mass, 1.0, 0.0)
        
        # Reward: displacement * alive (0 if dead)
        reward = displacement * alive
        
        return reward
    
    print(f"Population: {population_size}, Generations: {generations}")
    print(f"Starting params: Mu={best_params[0]:.3f}, Sigma={best_params[1]:.3f}")
    print("-" * 50)
    
    best_fitness = -float('inf')
    
    for gen in range(generations):
        key, noise_key, eval_key = jax.random.split(key, 3)
        
        # Generate population by adding noise to best
        noise = jax.random.normal(noise_key, (population_size, 2)) * sigma_noise
        population = best_params + noise
        population = jnp.clip(population, param_min, param_max)
        
        # Evaluate all
        eval_keys = jax.random.split(eval_key, population_size)
        fitness = jax.vmap(evaluate)(population, eval_keys)
        
        # Select best
        best_idx = jnp.argmax(fitness)
        
        if fitness[best_idx] > best_fitness:
            best_fitness = fitness[best_idx]
            best_params = population[best_idx]
        
        if gen % 5 == 0:
            print(f"Gen {gen:3d}: Best Fitness={float(best_fitness):.4f} | Mu={float(best_params[0]):.4f}, Sigma={float(best_params[1]):.4f}")
    
    print("=" * 50)
    print("Evolution Complete!")
    print(f"Best Params: Mu={float(best_params[0]):.4f}, Sigma={float(best_params[1]):.4f}")
    print(f"Best Reward (Displacement): {float(best_fitness):.4f}")
    
    # Final evaluation
    key, eval_key = jax.random.split(key)
    model = LeniaRNN(eval_key, steps=simulation_steps)
    model = eqx.tree_at(lambda m: m.cell.mu, model, jnp.array([best_params[0]]))
    model = eqx.tree_at(lambda m: m.cell.sigma, model, jnp.array([best_params[1]]))
    
    final_state, _ = model(init_state)
    x_init, _ = compute_com(init_state)
    x_final, _ = compute_com(final_state)
    
    print(f"\nFinal CoM: {float(x_init):.3f} -> {float(x_final):.3f}")
    print(f"Total movement: {float(x_final - x_init):.4f}")
    
    if x_final > x_init + 0.02:
        print("\n[SUCCESS] The creature evolved to move RIGHT!")
    elif x_final < x_init - 0.02:
        print("\n[INTERESTING] The creature moved LEFT")
    else:
        print("\n[PARTIAL] Minimal movement.")

if __name__ == "__main__":
    train_locomotion_es()
