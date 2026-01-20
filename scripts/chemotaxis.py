"""
Phase 9: Chemotaxis - Sensory-Motor Experiment.
Goal: Teach Lenia creature to navigate towards "food" using only physics.

Mechanism:
- Food Field: A gradient on the grid (e.g., high on left, low on right)
- Sensory Term: Growth modified by local food concentration
- Result: Creature should migrate towards higher food areas

Formula:
    Growth = StandardGrowth + alpha * (State * FoodField)
"""

import sys
import os
sys.path.append(os.getcwd())

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from src.neuro_lenia import LeniaRNN

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "size": 64,
    "mu": 0.26,
    "sigma": 0.05,
    "steps": 100,
    "n_trials": 10,
    "alpha": 0.2,  # Stronger sensory coupling
}

def create_creature(size, center_x=0.5, center_y=0.5, radius=0.12):
    """Create initial creature (stable soliton)."""
    x = jnp.linspace(0, 1, size)
    xx, yy = jnp.meshgrid(x, x)
    r = jnp.sqrt((xx - center_x)**2 + (yy - center_y)**2)
    blob = jnp.exp(-r**2 / (2 * radius**2))
    
    # Add asymmetric mass to make it stable (from Phase 8 findings)
    bump = 0.3 * jnp.exp(-((xx - center_x + 0.1)**2 + (yy - center_y)**2) / (2 * 0.08**2))
    return jnp.clip(blob + bump, 0.0, 1.0)

def create_food_gradient(size, direction="left"):
    """
    Create food field.
    - "left": High food on left, low on right
    - "right": High food on right, low on left
    - "center": High food in center
    """
    x = jnp.linspace(0, 1, size)
    xx, yy = jnp.meshgrid(x, x)
    
    if direction == "left":
        food = 1.0 - xx  # High on left (x=0), low on right (x=1)
    elif direction == "right":
        food = xx  # High on right
    elif direction == "center":
        food = 1.0 - jnp.abs(xx - 0.5) * 2  # High in center
    else:
        food = jnp.ones((size, size))
    
    return food

def compute_com(grid):
    """Compute Center of Mass."""
    if grid.ndim == 3:
        grid = grid[0]
    H, W = grid.shape
    total_mass = jnp.sum(grid) + 1e-10
    x_coords = jnp.linspace(0, 1, W)
    x_com = jnp.sum(grid * x_coords[None, :]) / total_mass
    return x_com

def step_with_food(state, kernel_fft, params, food_field, alpha):
    """
    Modified Lenia step with sensory input.
    
    The growth function is biased by local food concentration:
    Growth = StandardGrowth + alpha * State * FoodField
    
    This creates chemotaxis: cells grow more in high-food areas.
    """
    from src.engine_jax import growth_gaussian
    
    # Standard convolution
    state_fft = jnp.fft.rfft2(state)
    potential_fft = state_fft * kernel_fft
    potential = jnp.fft.irfft2(potential_fft, s=state.shape)
    
    # Standard growth
    standard_growth = growth_gaussian(potential, params.mu, params.sigma)
    
    # Sensory term: boost growth proportional to food
    sensory_growth = alpha * state * food_field
    
    # Total growth
    total_growth = standard_growth + sensory_growth
    
    # Update
    new_state = state + params.dt * total_growth
    return jnp.clip(new_state, 0.0, 1.0)

def run_chemotaxis_trial(key, food_direction, start_x=0.5):
    """Run one chemotaxis trial."""
    from src.engine_jax import get_default_params, get_kernel_fft
    
    size = CONFIG["size"]
    params = get_default_params()._replace(mu=CONFIG["mu"], sigma=CONFIG["sigma"])
    R = 13
    kernel_fft = get_kernel_fft(size, R, params.k_peak, params.k_width)
    
    # Create creature and food
    state = create_creature(size, center_x=start_x)
    food_field = create_food_gradient(size, direction=food_direction)
    
    x_init = float(compute_com(state))
    init_mass = float(jnp.sum(state))
    
    # Run simulation with sensory input
    for _ in range(CONFIG["steps"]):
        state = step_with_food(state, kernel_fft, params, food_field, CONFIG["alpha"])
    
    x_final = float(compute_com(state))
    final_mass = float(jnp.sum(state))
    survived = final_mass > 0.1 * init_mass
    
    displacement = x_final - x_init
    return displacement, survived

def run_chemotaxis_experiment():
    print("=" * 70)
    print("Phase 9: Chemotaxis Experiment")
    print("=" * 70)
    print(f"Parameters: mu={CONFIG['mu']}, sigma={CONFIG['sigma']}")
    print(f"Alpha (sensory coupling): {CONFIG['alpha']}")
    print(f"Steps: {CONFIG['steps']}")
    print("-" * 70)
    
    key = jax.random.PRNGKey(42)
    
    # ========================================
    # Experiment A: Food on LEFT (creature should move LEFT)
    # ========================================
    print("\n[A] Food on LEFT (creature starts at center)...")
    left_displacements = []
    for i in range(CONFIG["n_trials"]):
        key, trial_key = jax.random.split(key)
        displacement, survived = run_chemotaxis_trial(trial_key, "left", start_x=0.5)
        if survived:
            left_displacements.append(displacement)
    
    print(f"  Survival: {len(left_displacements)}/{CONFIG['n_trials']}")
    if left_displacements:
        print(f"  Mean displacement: {np.mean(left_displacements):.4f}")
        print(f"  Direction: {'LEFT' if np.mean(left_displacements) < 0 else 'RIGHT'}")
    
    # ========================================
    # Experiment B: Food on RIGHT (creature should move RIGHT)
    # ========================================
    print("\n[B] Food on RIGHT (creature starts at center)...")
    right_displacements = []
    for i in range(CONFIG["n_trials"]):
        key, trial_key = jax.random.split(key)
        displacement, survived = run_chemotaxis_trial(trial_key, "right", start_x=0.5)
        if survived:
            right_displacements.append(displacement)
    
    print(f"  Survival: {len(right_displacements)}/{CONFIG['n_trials']}")
    if right_displacements:
        print(f"  Mean displacement: {np.mean(right_displacements):.4f}")
        print(f"  Direction: {'LEFT' if np.mean(right_displacements) < 0 else 'RIGHT'}")
    
    # ========================================
    # Experiment C: NO food gradient (control)
    # ========================================
    print("\n[C] NO food gradient (uniform - control)...")
    control_displacements = []
    for i in range(CONFIG["n_trials"]):
        key, trial_key = jax.random.split(key)
        displacement, survived = run_chemotaxis_trial(trial_key, "uniform", start_x=0.5)
        if survived:
            control_displacements.append(displacement)
    
    print(f"  Survival: {len(control_displacements)}/{CONFIG['n_trials']}")
    if control_displacements:
        print(f"  Mean displacement: {np.mean(control_displacements):.4f}")
    
    # ========================================
    # Analysis
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if left_displacements and right_displacements:
        left_mean = np.mean(left_displacements)
        right_mean = np.mean(right_displacements)
        
        print(f"Food LEFT → Displacement: {left_mean:.4f}")
        print(f"Food RIGHT → Displacement: {right_mean:.4f}")
        
        if left_mean < 0 and right_mean > 0:
            print("\n[CHEMOTAXIS CONFIRMED!]")
            print("Creature navigates TOWARDS food gradient.")
        elif left_mean < right_mean:
            print("\n[PARTIAL] Creature biased by food, but not perfectly directed.")
        else:
            print("\n[INCONCLUSIVE] Food gradient effect not detected.")

if __name__ == "__main__":
    run_chemotaxis_experiment()
