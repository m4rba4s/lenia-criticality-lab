"""
Phase 8: The Assassin Experiment.
Goal: Prove we can KILL a living soliton with an anti-bump (RESET).

Hypothesis:
- SET (right bump) → Life
- RESET (left bump or negative bump) → Death

If true, we have a functional RS-Trigger:
- Input A (Set): Poke right → Survive
- Input B (Reset): Poke left → Die
"""

import sys
import os
sys.path.append(os.getcwd())

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from scipy import stats

from src.neuro_lenia import LeniaRNN

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "size": 64,
    "mu": 0.26,
    "sigma": 0.05,
    "steps_stabilize": 30,
    "steps_after_reset": 50,
    "n_trials": 20,
    "set_strength": 0.3,  # Right bump (positive)
    "reset_strength": 0.8,  # Stronger assassin
}

def create_blob(size, center_x=0.5, center_y=0.5, radius=0.12):
    """Create symmetric initial blob."""
    x = jnp.linspace(0, 1, size)
    xx, yy = jnp.meshgrid(x, x)
    r = jnp.sqrt((xx - center_x)**2 + (yy - center_y)**2)
    blob = jnp.exp(-r**2 / (2 * radius**2))
    return blob[None, ...]

def apply_set_bump(grid, strength):
    """SET: Add mass to right side (creates asymmetry that enables survival)."""
    if grid.ndim == 3:
        grid = grid[0]
    H, W = grid.shape
    x = jnp.linspace(0, 1, W)
    y = jnp.linspace(0, 1, H)
    xx, yy = jnp.meshgrid(x, y)
    bump = strength * jnp.exp(-((xx - 0.6)**2 + (yy - 0.5)**2) / (2 * 0.08**2))
    return jnp.clip(grid + bump, 0.0, 1.0)[None, ...]

def apply_reset_bump(grid, strength, mode="left"):
    """
    RESET: Try to kill the soliton.
    Modes:
    - "left": Add mass to left side (restore symmetry)
    - "negative": Remove mass from right side (anti-bump)
    - "center": Add mass to center (disrupt structure)
    """
    if grid.ndim == 3:
        grid = grid[0]
    H, W = grid.shape
    x = jnp.linspace(0, 1, W)
    y = jnp.linspace(0, 1, H)
    xx, yy = jnp.meshgrid(x, y)
    
    if mode == "left":
        # Left bump to restore symmetry
        bump = strength * jnp.exp(-((xx - 0.4)**2 + (yy - 0.5)**2) / (2 * 0.08**2))
        return jnp.clip(grid + bump, 0.0, 1.0)[None, ...]
    elif mode == "negative":
        # Remove mass from right side
        anti_bump = strength * jnp.exp(-((xx - 0.6)**2 + (yy - 0.5)**2) / (2 * 0.08**2))
        return jnp.clip(grid - anti_bump, 0.0, 1.0)[None, ...]
    elif mode == "center":
        # Disruptive center bump
        bump = strength * jnp.exp(-((xx - 0.5)**2 + (yy - 0.5)**2) / (2 * 0.06**2))
        return jnp.clip(grid + bump, 0.0, 1.0)[None, ...]

def run_trial(key, do_set=True, do_reset=False, reset_mode="left"):
    """
    Run one trial.
    Returns: (survived_after_set, survived_after_reset, final_mass)
    """
    size = CONFIG["size"]
    
    # Create models
    model_stab = LeniaRNN(key, steps=CONFIG["steps_stabilize"])
    model_stab = eqx.tree_at(lambda m: m.cell.mu, model_stab, jnp.array([CONFIG["mu"]]))
    model_stab = eqx.tree_at(lambda m: m.cell.sigma, model_stab, jnp.array([CONFIG["sigma"]]))
    
    model_post = LeniaRNN(key, steps=CONFIG["steps_after_reset"])
    model_post = eqx.tree_at(lambda m: m.cell.mu, model_post, jnp.array([CONFIG["mu"]]))
    model_post = eqx.tree_at(lambda m: m.cell.sigma, model_post, jnp.array([CONFIG["sigma"]]))
    
    # Initial state
    state = create_blob(size)
    
    # SET phase
    if do_set:
        state = apply_set_bump(state, CONFIG["set_strength"])
    
    # Stabilize
    state, _ = model_stab(state)
    mass_after_set = float(jnp.sum(state))
    survived_set = mass_after_set > 1.0
    
    # RESET phase (The Assassin)
    if do_reset and survived_set:
        state = apply_reset_bump(state, CONFIG["reset_strength"], mode=reset_mode)
    
    # Evolve after reset
    state, _ = model_post(state)
    final_mass = float(jnp.sum(state))
    survived_reset = final_mass > 1.0
    
    return survived_set, survived_reset, final_mass

def run_assassin_experiment():
    print("=" * 70)
    print("Phase 8: The Assassin Experiment")
    print("=" * 70)
    print(f"Parameters: mu={CONFIG['mu']}, sigma={CONFIG['sigma']}")
    print(f"SET strength: {CONFIG['set_strength']}, RESET strength: {CONFIG['reset_strength']}")
    print("-" * 70)
    
    key = jax.random.PRNGKey(42)
    
    # ========================================
    # Experiment A: SET only (control - should survive)
    # ========================================
    print("\n[A] SET only (no RESET)...")
    set_only_survival = 0
    for i in range(CONFIG["n_trials"]):
        key, trial_key = jax.random.split(key)
        survived_set, survived_final, _ = run_trial(trial_key, do_set=True, do_reset=False)
        if survived_final:
            set_only_survival += 1
    print(f"  Survival: {set_only_survival}/{CONFIG['n_trials']} ({set_only_survival/CONFIG['n_trials']:.0%})")
    
    # ========================================
    # Experiment B: SET then RESET (left bump)
    # ========================================
    print("\n[B] SET then RESET (left bump - restore symmetry)...")
    reset_left_survival = 0
    for i in range(CONFIG["n_trials"]):
        key, trial_key = jax.random.split(key)
        survived_set, survived_final, _ = run_trial(trial_key, do_set=True, do_reset=True, reset_mode="left")
        if survived_final:
            reset_left_survival += 1
    print(f"  Survival: {reset_left_survival}/{CONFIG['n_trials']} ({reset_left_survival/CONFIG['n_trials']:.0%})")
    
    # ========================================
    # Experiment C: SET then RESET (negative - anti-bump)
    # ========================================
    print("\n[C] SET then RESET (anti-bump - remove mass)...")
    reset_neg_survival = 0
    for i in range(CONFIG["n_trials"]):
        key, trial_key = jax.random.split(key)
        survived_set, survived_final, _ = run_trial(trial_key, do_set=True, do_reset=True, reset_mode="negative")
        if survived_final:
            reset_neg_survival += 1
    print(f"  Survival: {reset_neg_survival}/{CONFIG['n_trials']} ({reset_neg_survival/CONFIG['n_trials']:.0%})")
    
    # ========================================
    # Experiment D: SET then RESET (center - disrupt)
    # ========================================
    print("\n[D] SET then RESET (center bump - disrupt)...")
    reset_center_survival = 0
    for i in range(CONFIG["n_trials"]):
        key, trial_key = jax.random.split(key)
        survived_set, survived_final, _ = run_trial(trial_key, do_set=True, do_reset=True, reset_mode="center")
        if survived_final:
            reset_center_survival += 1
    print(f"  Survival: {reset_center_survival}/{CONFIG['n_trials']} ({reset_center_survival/CONFIG['n_trials']:.0%})")
    
    # ========================================
    # Analysis
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"SET only:        {set_only_survival}/{CONFIG['n_trials']} alive")
    print(f"SET + LEFT:      {reset_left_survival}/{CONFIG['n_trials']} alive")
    print(f"SET + ANTI-BUMP: {reset_neg_survival}/{CONFIG['n_trials']} alive")
    print(f"SET + CENTER:    {reset_center_survival}/{CONFIG['n_trials']} alive")
    
    # Find the best assassin
    best_kill = min(reset_left_survival, reset_neg_survival, reset_center_survival)
    
    if set_only_survival > 15 and best_kill < set_only_survival - 5:
        print("\n[RS-TRIGGER CONFIRMED!]")
        print("SET creates life, RESET induces death.")
        print("This is a functional 1-bit memory cell.")
    elif set_only_survival > 15 and best_kill >= set_only_survival - 5:
        print("\n[PARTIAL] RESET effect weak. May need stronger perturbation.")
    else:
        print("\n[INCONCLUSIVE] SET survival too low for meaningful comparison.")

if __name__ == "__main__":
    run_assassin_experiment()
