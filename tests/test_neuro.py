
import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from src.neuro_lenia import LeniaLayer, LeniaRNN

def test_layer_shape():
    """Verify LeniaLayer preserves grid shape."""
    key = jax.random.PRNGKey(0)
    layer = LeniaLayer(key)
    
    # Input: (1, 64, 64)
    x = jnp.zeros((1, 64, 64))
    out = layer(x)
    
    assert out.shape == x.shape
    assert not jnp.any(jnp.isnan(out))

def test_rnn_unroll():
    """Verify RNN unrolls for N steps and returns history."""
    key = jax.random.PRNGKey(1)
    steps = 10
    model = LeniaRNN(key, steps=steps)
    
    x = jnp.zeros((1, 32, 32))
    final, history = model(x)
    
    assert final.shape == x.shape
    # History stack: (steps, 1, H, W) depending on scan output structure
    # Our scan returns (new_state, new_state), so history is stack of states
    assert history.shape == (steps, 1, 32, 32)

def test_differentiability():
    """Verify gradients flow through the model (params are updated)."""
    key = jax.random.PRNGKey(2)
    model = LeniaRNN(key, steps=5)
    
    x = jax.random.uniform(key, (1, 32, 32))
    target = jax.random.uniform(key, (1, 32, 32))
    
    @eqx.filter_jit
    def loss_fn(m, x, y):
        pred, _ = m(x)
        return jnp.mean((pred - y)**2)
    
    # Check gradients exist
    grads = eqx.filter_grad(loss_fn)(model, x, target)
    
    # Check gradient for Mu is not zero (unless optimal, which is unlikely random)
    mu_grad = grads.cell.mu
    assert mu_grad is not None
    assert not jnp.allclose(mu_grad, 0.0)
    
    # Check kernel weights grad
    kernel_grad = grads.cell.conv.weight
    assert kernel_grad is not None
    assert jnp.max(jnp.abs(kernel_grad)) > 0.0
