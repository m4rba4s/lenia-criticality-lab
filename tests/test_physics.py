
import pytest
import jax
import jax.numpy as jnp
from jax import random
import numpy as np

from src.engine_jax import step, get_kernel_fft, JAXLeniaParams, get_default_params

@pytest.fixture
def params():
    return get_default_params()

@pytest.fixture
def kernel_fft(params):
    return get_kernel_fft(64, 13, params.k_peak, params.k_width)

def test_determinism(params, kernel_fft):
    """Running twice with same inputs should give identical outputs."""
    key = random.PRNGKey(0)
    state = random.uniform(key, (64, 64))
    
    res1 = step(state, kernel_fft, params)
    res2 = step(state, kernel_fft, params)
    
    assert jnp.allclose(res1, res2)

def test_empty_world(params, kernel_fft):
    """Empty world should stay empty (no spontaneous generation)."""
    state = jnp.zeros((64, 64))
    res = step(state, kernel_fft, params)
    assert jnp.allclose(res, 0.0)

def test_rotational_symmetry(params, kernel_fft):
    """
    Simulating a rotated state should be equivalent to 
    simulating the state and then rotating the result.
    (Isotropic kernel assumption)
    """
    key = random.PRNGKey(1)
    state = random.uniform(key, (64, 64))
    
    # 1. Rotate then Step
    state_rot = jnp.rot90(state)
    res_rot = step(state_rot, kernel_fft, params)
    
    # 2. Step then Rotate
    res_orig = step(state, kernel_fft, params)
    res_orig_rot = jnp.rot90(res_orig)
    
    # Allow small numerical error from FFT
    assert jnp.allclose(res_rot, res_orig_rot, atol=1e-5)

def test_translation_invariance(params, kernel_fft):
    """Simulating a shifted state should match shifted output (Toroidal boundary)."""
    key = random.PRNGKey(2)
    state = random.uniform(key, (64, 64))
    
    # Shift by (10, 10)
    shift = 10
    state_shift = jnp.roll(state, (shift, shift), axis=(0, 1))
    
    res_shift = step(state_shift, kernel_fft, params)
    
    res_orig = step(state, kernel_fft, params)
    res_orig_shift = jnp.roll(res_orig, (shift, shift), axis=(0, 1))
    
    assert jnp.allclose(res_shift, res_orig_shift, atol=1e-5)
