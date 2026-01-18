
import pytest
import jax.numpy as jnp
from jax import random
import numpy as np

from src.metrics_jax import compute_te_pairwise, compute_te_significance, compute_lyapunov_batch
from src.engine_jax import get_default_params, get_kernel_fft

def test_te_independence():
    """TE between independent random variables should be low."""
    key = random.PRNGKey(0)
    x = random.randint(key, (1000,), 0, 100) / 100.0
    y = random.randint(key, (1000,), 0, 100) / 100.0 # Independent
    
    # Run slightly larger bin count to ensure we don't accidentally match
    te = compute_te_pairwise(x, y, n_bins=4)
    # Ideally 0, but bias exists. Should be small.
    assert te < 0.1

def test_te_copy():
    """TE from X to copy of X should be high (Self-Entropy)."""
    key = random.PRNGKey(1)
    x = random.normal(key, (1000,))
    # Normalize to 0-1
    x = (x - jnp.min(x)) / (jnp.max(x) - jnp.min(x))
    
    # Y is lagged copy
    y = jnp.roll(x, 1)
    
    te = compute_te_pairwise(x, y, n_bins=8, lag=1)
    
    # Entropy of uniform-ish random variable with 8 bins is log2(8)=3.
    # Normal distribution is less, maybe ~2.
    assert te > 1.0

def test_significance_test():
    """Verify significance testing returns proper structure and plausible values."""
    key = random.PRNGKey(2)
    k1, k2, k_stat = random.split(key, 3)
    x = random.normal(k1, (500,))
    y = random.normal(k2, (500,)) # Independent
    
    te, p_val, z = compute_te_significance(k_stat, x, y, n_surrogates=20)
    
    assert isinstance(te, float) or isinstance(te, jnp.ndarray)
    assert 0.0 <= p_val <= 1.0
    # For random data, z-score should be small
    assert abs(z) < 3.0

def test_lyapunov_convergence():
    """Check that Lyapunov function generally runs and returns valid shapes."""
    batch_size = 2
    size = 32
    params = get_default_params()
    k_fft = get_kernel_fft(size, 13, 0.5, 0.15)
    
    key = random.PRNGKey(3)
    states = random.uniform(key, (batch_size, size, size))
    
    exps, history = compute_lyapunov_batch(states, k_fft, params, warmup_steps=10, measure_steps=50, renorm_interval=10)
    
    assert exps.shape == (batch_size,)
    assert history.shape[1] == batch_size
    # Check not NaN
    assert not jnp.any(jnp.isnan(exps))
