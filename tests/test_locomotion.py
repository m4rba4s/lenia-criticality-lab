"""
Unit tests for Locomotion / Soft Robotics functionality.
"""

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

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

def create_blob(size, center_x=0.25):
    """Create test blob."""
    x = jnp.linspace(0, 1, size)
    xx, yy = jnp.meshgrid(x, x)
    r = jnp.sqrt((xx - center_x)**2 + (yy - 0.5)**2)
    blob = jnp.exp(-r**2 / (2 * 0.12**2))
    return blob[None, ...]

def test_com_calculation():
    """Verify CoM calculation is correct."""
    size = 64
    # Blob on left
    blob_left = create_blob(size, center_x=0.2)
    com_left = compute_com(blob_left)
    assert 0.15 < com_left < 0.25
    
    # Blob on right
    blob_right = create_blob(size, center_x=0.8)
    com_right = compute_com(blob_right)
    assert 0.75 < com_right < 0.85

def test_locomotion_params_produce_movement():
    """Verify known locomotion params produce rightward movement."""
    key = jax.random.PRNGKey(42)
    size = 64
    steps = 50
    
    # Best params from evolution
    mu_best = 0.155
    sigma_best = 0.030
    
    model = LeniaRNN(key, steps=steps)
    model = eqx.tree_at(lambda m: m.cell.mu, model, jnp.array([mu_best]))
    model = eqx.tree_at(lambda m: m.cell.sigma, model, jnp.array([sigma_best]))
    
    init_state = create_blob(size, center_x=0.25)
    final_state, _ = model(init_state)
    
    x_init = compute_com(init_state)
    x_final = compute_com(final_state)
    displacement = x_final - x_init
    
    # Should move right by at least 0.1
    assert displacement > 0.1, f"Expected rightward movement, got {displacement}"
    
    # Should be alive (mass > 0)
    assert jnp.sum(final_state) > 0.1 * jnp.sum(init_state)

def test_dead_params_produce_no_movement():
    """Verify bad params (death) produce zero/negative reward."""
    key = jax.random.PRNGKey(42)
    size = 64
    steps = 50
    
    # Extreme params that cause death
    mu_dead = 0.01
    sigma_dead = 0.001
    
    model = LeniaRNN(key, steps=steps)
    model = eqx.tree_at(lambda m: m.cell.mu, model, jnp.array([mu_dead]))
    model = eqx.tree_at(lambda m: m.cell.sigma, model, jnp.array([sigma_dead]))
    
    init_state = create_blob(size, center_x=0.25)
    final_state, _ = model(init_state)
    
    # Should be dead or nearly dead
    mass_ratio = jnp.sum(final_state) / jnp.sum(init_state)
    assert mass_ratio < 0.5, f"Expected death, but mass ratio is {mass_ratio}"
