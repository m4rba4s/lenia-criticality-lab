"""
Auto-Lenia: Open-Ended Evolution of Creatures.
Uses Genetic Algorithm to evolve morphology and physics parameters
for survival, locomotion, and novelty.
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
    """Compute Center of Mass."""
    if grid.ndim == 3:
        grid = grid[0]
    H, W = grid.shape
    total_mass = jnp.sum(grid) + 1e-10
    x_coords = jnp.linspace(0, 1, W)
    x_com = jnp.sum(grid * x_coords[None, :]) / total_mass
    return x_com

def create_creature(key, size, genotype):
    """
    Create a creature from genotype.
    Genotype: [center_x, center_y, radius, num_blobs, mu, sigma]
    All operations must be JAX-compatible (no Python int).
    """
    center_x = genotype[0]
    center_y = genotype[1]
    radius = jnp.clip(genotype[2], 0.05, 0.2)
    num_blobs = jnp.clip(genotype[3], 1.0, 3.0)  # Keep as float
    
    x = jnp.linspace(0, 1, size)
    xx, yy = jnp.meshgrid(x, x)
    
    # Main blob
    r = jnp.sqrt((xx - center_x)**2 + (yy - center_y)**2)
    blob = jnp.exp(-r**2 / (2 * radius**2))
    
    # Second blob (weighted by whether num_blobs >= 2)
    r2 = jnp.sqrt((xx - center_x - 0.15)**2 + (yy - center_y)**2)
    blob2 = 0.7 * jnp.exp(-r2**2 / (2 * (radius * 0.8)**2))
    weight2 = jnp.where(num_blobs >= 2.0, 1.0, 0.0)
    blob = blob + blob2 * weight2
    
    # Third blob (weighted by whether num_blobs >= 3)
    r3 = jnp.sqrt((xx - center_x + 0.15)**2 + (yy - center_y)**2)
    blob3 = 0.7 * jnp.exp(-r3**2 / (2 * (radius * 0.8)**2))
    weight3 = jnp.where(num_blobs >= 3.0, 1.0, 0.0)
    blob = blob + blob3 * weight3
    
    return jnp.clip(blob, 0.0, 1.0)[None, ...]

def evaluate_creature(genotype, key, size=64, steps=80):
    """
    Evaluate fitness of a creature.
    Returns: (survival_score, displacement, novelty_hash)
    """
    mu = jnp.clip(genotype[4], 0.05, 0.9)  # Expanded to cover logic regime
    sigma = jnp.clip(genotype[5], 0.005, 0.15)
    
    init_state = create_creature(key, size, genotype)
    init_mass = jnp.sum(init_state)
    x_init = compute_com(init_state)
    
    model = LeniaRNN(key, steps=steps)
    model = eqx.tree_at(lambda m: m.cell.mu, model, jnp.array([mu]))
    model = eqx.tree_at(lambda m: m.cell.sigma, model, jnp.array([sigma]))
    
    final_state, history = model(init_state)
    final_mass = jnp.sum(final_state)
    x_final = compute_com(final_state)
    
    # Survival score (0-1)
    survival = jnp.clip(final_mass / (init_mass + 1e-6), 0.0, 1.0)
    
    # Displacement score
    displacement = x_final - x_init
    
    # Combined fitness
    fitness = survival * 0.5 + jnp.clip(displacement, -0.5, 0.5) * 0.5
    
    return fitness, survival, displacement

def evolve_creatures():
    print("=" * 60)
    print("Auto-Lenia: Open-Ended Evolution")
    print("=" * 60)
    
    population_size = 30
    generations = 20
    mutation_rate = 0.1
    elite_count = 5
    
    # Genotype: [center_x, center_y, radius, num_blobs, mu, sigma]
    genotype_size = 6
    
    # Bounds (expanded mu to cover 0.8 logic regime)
    bounds_min = jnp.array([0.2, 0.3, 0.05, 1.0, 0.05, 0.005])
    bounds_max = jnp.array([0.4, 0.7, 0.2, 3.0, 0.9, 0.15])
    
    key = jax.random.PRNGKey(42)
    
    # Initialize population
    key, init_key = jax.random.split(key)
    population = jax.random.uniform(init_key, (population_size, genotype_size))
    population = bounds_min + population * (bounds_max - bounds_min)
    
    @jax.jit
    def eval_batch(pop, keys):
        return jax.vmap(lambda g, k: evaluate_creature(g, k)[0])(pop, keys)
    
    print(f"Population: {population_size}, Generations: {generations}")
    print(f"Genotype: [center_x, center_y, radius, num_blobs, mu, sigma]")
    print("-" * 60)
    
    best_ever = None
    best_fitness_ever = -float('inf')
    
    for gen in range(generations):
        key, eval_key, select_key, mutate_key = jax.random.split(key, 4)
        
        # Evaluate
        eval_keys = jax.random.split(eval_key, population_size)
        fitness = eval_batch(population, eval_keys)
        
        # Track best
        best_idx = jnp.argmax(fitness)
        best_fitness = float(fitness[best_idx])
        
        if best_fitness > best_fitness_ever:
            best_fitness_ever = best_fitness
            best_ever = population[best_idx]
        
        if gen % 5 == 0:
            print(f"Gen {gen:3d}: Best={best_fitness:.4f} | Mu={float(population[best_idx, 4]):.3f}, Sigma={float(population[best_idx, 5]):.3f}")
        
        # Selection (tournament)
        sorted_indices = jnp.argsort(-fitness)  # Descending
        elites = population[sorted_indices[:elite_count]]
        
        # Vectorized Crossover & Mutation (no Python loop)
        num_offspring = population_size - elite_count
        
        # Generate all random keys at once
        key, cross_key, mut_key, parent_key = jax.random.split(key, 4)
        parent_indices = jax.random.randint(parent_key, (num_offspring, 2), 0, elite_count)
        
        # Select parents
        parents1 = elites[parent_indices[:, 0]]  # (num_offspring, genotype_size)
        parents2 = elites[parent_indices[:, 1]]  # (num_offspring, genotype_size)
        
        # Uniform crossover (vectorized)
        cross_masks = jax.random.bernoulli(cross_key, 0.5, (num_offspring, genotype_size))
        children = jnp.where(cross_masks, parents1, parents2)
        
        # Mutation (vectorized)
        noise = jax.random.normal(mut_key, (num_offspring, genotype_size)) * mutation_rate
        children = children + noise * (bounds_max - bounds_min)
        children = jnp.clip(children, bounds_min, bounds_max)
        
        population = jnp.concatenate([elites, children], axis=0)
    
    print("=" * 60)
    print("Evolution Complete!")
    print(f"Best Fitness: {best_fitness_ever:.4f}")
    print(f"Best Genotype:")
    print(f"  center_x={float(best_ever[0]):.3f}")
    print(f"  center_y={float(best_ever[1]):.3f}")
    print(f"  radius={float(best_ever[2]):.3f}")
    print(f"  num_blobs={int(best_ever[3])}")
    print(f"  mu={float(best_ever[4]):.4f}")
    print(f"  sigma={float(best_ever[5]):.4f}")
    
    # Final evaluation
    key, final_key = jax.random.split(key)
    fitness, survival, displacement = evaluate_creature(best_ever, final_key)
    print(f"\nFinal Eval:")
    print(f"  Survival: {float(survival):.2%}")
    print(f"  Displacement: {float(displacement):.4f}")

if __name__ == "__main__":
    evolve_creatures()
