"""
JAX-Accelerated Metrics for Lenia
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial
from typing import Tuple, NamedTuple, Callable, Optional, Dict

from src.engine_jax import step, JAXLeniaParams

# =============================================================================
# LYAPUNOV EXPONENT
# =============================================================================

@partial(jit, static_argnames=['warmup_steps', 'measure_steps', 'renorm_interval'])
def compute_lyapunov_batch(
    states: jnp.ndarray, # [Batch, H, W]
    kernel_fft: jnp.ndarray,
    params: JAXLeniaParams,
    warmup_steps: int = 100,
    measure_steps: int = 500,
    renorm_interval: int = 10,
    epsilon: float = 1e-8
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute Lyapunov exponent for a batch of states.
    
    Returns:
        exponent: Estimated LE for each batch item
        history: Convergence history
    """
    
    # 1. Warmup
    def warmup_step(state, _):
        return step(state, kernel_fft, params), None
        
    states_warm, _ = lax.scan(warmup_step, states, None, length=warmup_steps)
    
    # 2. Create perturbation
    # We need randomness, but to keep it simple and deterministic per batch, 
    # we can use a fixed perturbation pattern or pass keys. 
    # Ideally should pass keys. For now, use a simple checkerboard or deterministic noise.
    # A simple way to get "random-like" perturbation without carrying keys:
    # use sin(state * huge_number)
    noise = jnp.sin(states_warm * 12345.6789)
    # Project to orthogonal? No, just normalize.
    noise_norm = jnp.linalg.norm(noise, axis=(1, 2), keepdims=True)
    perturbation = noise / (noise_norm + 1e-10) * epsilon
    perturbation = perturbation.reshape(states.shape)
    
    state_ref = states_warm
    state_pert = jnp.clip(state_ref + perturbation, 0.0, 1.0)
    
    # 3. Measurement Loop
    n_renorms = measure_steps // renorm_interval
    
    def measure_cycle(carry, _):
        ref, pert = carry
        
        # Evolve interval
        def evolve_step(pair, _):
            r, p = pair
            r_next = step(r, kernel_fft, params)
            p_next = step(p, kernel_fft, params)
            return (r_next, p_next), None
            
        (ref_next, pert_next), _ = lax.scan(evolve_step, (ref, pert), None, length=renorm_interval)
        
        # Measure separation
        diff = pert_next - ref_next
        dist = jnp.linalg.norm(diff, axis=(1, 2), keepdims=True)
        
        # Log stretch
        # Handle collapse (dist ~= 0) -> log(0) -> -inf. Clip small.
        # squeeze for log (we want [Batch])
        dist_scalar = jnp.squeeze(dist)
        dist_scalar = jnp.maximum(dist_scalar, 1e-15)
        log_stretch = jnp.log(dist_scalar / epsilon)
        
        # Renormalize
        # dist is [Batch, 1, 1], so simple division works
        # Prevent division by zero
        dist_safe = jnp.maximum(dist, 1e-15)
        pert_norm = ref_next + (diff / dist_safe) * epsilon
        pert_norm = jnp.clip(pert_norm, 0.0, 1.0)

        
        return (ref_next, pert_norm), log_stretch

    _, log_stretches = lax.scan(measure_cycle, (state_ref, state_pert), None, length=n_renorms)
    
    # log_stretches shape: [n_renorms, batch]
    # Mean over time
    exponents = jnp.mean(log_stretches, axis=0) / renorm_interval
    
    return exponents, log_stretches


# =============================================================================
# TRANSFER ENTROPY
# =============================================================================

def histogram_entropy(x_idx: jnp.ndarray, n_bins: int) -> float:
    """Compute entropy of discrete indices using bincount."""
    counts = jnp.bincount(x_idx, minlength=n_bins, length=n_bins)
    max_val = counts.sum()
    probs = counts / (max_val + 1e-10)
    # Filter zeros to avoid log(0)
    probs = jnp.where(probs > 0, probs, 1.0)
    return -jnp.sum(probs * jnp.log2(probs))

@partial(jit, static_argnames=['n_bins', 'history_length', 'lag'])
def compute_te_pairwise(
    x: jnp.ndarray, # Time series [T]
    y: jnp.ndarray, # Time series [T]
    n_bins: int = 8,
    history_length: int = 1,
    lag: int = 1
) -> float:
    """
    Compute TE(X->Y) for dense time series.
    JIT-compatible implementation using integer packing.
    """
    # 1. Discretize
    def discretize(arr):
        # Normalize simple min-max
        arr_min, arr_max = jnp.min(arr), jnp.max(arr)
        # Avoid div by zero
        scale = jnp.where(arr_max > arr_min, 1.0 / (arr_max - arr_min), 0.0)
        norm = (arr - arr_min) * scale
        return jnp.clip((norm * n_bins).astype(jnp.int32), 0, n_bins - 1)
        
    x_d = discretize(x)
    y_d = discretize(y)
    T = x.shape[0]
    
    # 2. Construct embeddings
    # We need:
    # Y_future: Y(t+lag)
    # Y_past: Y(t), Y(t-1)...
    # X_past: X(t), X(t-1)...
    
    # Valid range
    t_start = history_length
    t_end = T - lag
    
    # We can effectively just slice windows
    # Since history_length is usually small (1-3), we can manually pack
    # Pack into single integer for bincounting
    # Index = Y_fut + Y_past*B + X_past*B^2 ...
    
    # For history_length=1:
    # Key = Y_fut + Y_curr * B + X_curr * B^2
    # B = n_bins
    
    # Prepare slices
    # Y_future: from [start+lag : end+lag]
    sl_fut = y_d[t_start+lag : t_end+lag]
    
    # Past: current time t
    sl_y_past = y_d[t_start : t_end]
    sl_x_past = x_d[t_start : t_end]
    
    # Construct packed indices
    # H(Y_fut | Y_past) = H(Y_fut, Y_past) - H(Y_past)
    idx_y_fut_past = sl_fut + sl_y_past * n_bins
    idx_y_past = sl_y_past
    
    # H(Y_fut | Y_past, X_past) = H(Y_fut, Y_past, X_past) - H(Y_past, X_past)
    idx_y_fut_past_x = sl_fut + sl_y_past * n_bins + sl_x_past * (n_bins**2)
    idx_y_past_x = sl_y_past + sl_x_past * n_bins
    
    # Entropies
    # Define max bins for each case
    max_bins_2d = n_bins * n_bins
    max_bins_3d = n_bins * n_bins * n_bins
    
    h_y_fut_past = histogram_entropy(idx_y_fut_past, max_bins_2d)
    h_y_past = histogram_entropy(idx_y_past, n_bins)
    
    h_cond_1 = h_y_fut_past - h_y_past
    
    h_y_fut_past_x = histogram_entropy(idx_y_fut_past_x, max_bins_3d)
    h_y_past_x = histogram_entropy(idx_y_past_x, max_bins_2d)
    
    h_cond_2 = h_y_fut_past_x - h_y_past_x
    
    te = h_cond_1 - h_cond_2
    
    # If history_length > 1, this logic needs general packing loop.
    # For now, hardcode history_length=1 support or use scan for packing.
    # Given the requirements, let's stick to k=1 for super speed or expand if needed.
    

    return jnp.maximum(te, 0.0)


@partial(jit, static_argnames=['n_bins', 'n_surrogates'])
def compute_te_significance(
    key: jnp.ndarray,
    x: jnp.ndarray, # [T]
    y: jnp.ndarray, # [T]
    n_bins: int = 8,
    n_surrogates: int = 100
) -> Tuple[float, float, float]:
    """
    Compute Transfer Entropy with Statistical Significance (Surrogate Data Test).
    
    Generates surrogates by circular time-shifting X relative to Y, destroying 
    specific temporal correlations while preserving marginal statistics.
    
    Returns:
        te_original: The actual TE value.
        p_value: Prob(TE_random >= TE_original). Low value (<0.01) implies significance.
        z_score: (TE_original - mean(TE_surrogates)) / std(TE_surrogates).
    """
    # 1. Compute original TE
    te_real = compute_te_pairwise(x, y, n_bins)
    
    # 2. Generate random shifts for surrogates
    # We want shifts that are not 0 (original) or small (correlation length).
    # Shift range: [T//4, 3T//4] roughly, or just random uniform.
    T = x.shape[0]
    shifts = jax.random.randint(key, (n_surrogates,), minval=10, maxval=T-10)
    
    # 3. Compute TE for each surrogate
    def measure_surrogate(shift):
        x_shuffled = jnp.roll(x, shift)
        return compute_te_pairwise(x_shuffled, y, n_bins)
        
    te_surrogates = vmap(measure_surrogate)(shifts)
    
    # 4. Statistics
    # P-value: Fraction of surrogates >= original
    # We add 1 to num and den to be conservative (pseudo-count)
    n_greater = jnp.sum(te_surrogates >= te_real)
    p_value = (n_greater + 1.0) / (n_surrogates + 1.0)
    
    mean_null = jnp.mean(te_surrogates)
    std_null = jnp.std(te_surrogates)
    
    z_score = (te_real - mean_null) / (std_null + 1e-10)
    
    return te_real, p_value, z_score

