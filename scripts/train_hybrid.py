
import sys
import os
sys.path.append(os.getcwd())

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np

from src.neuro_lenia import LeniaRNN

def create_target_pattern(size):
    # Create a target "Goal" pattern: Two distinct spots
    x = jnp.linspace(-1, 1, size)
    xx, yy = jnp.meshgrid(x, x)
    
    # Spot 1
    r1 = jnp.sqrt((xx - 0.3)**2 + (yy - 0.3)**2)
    s1 = jnp.exp(-r1**2 / 0.05)
    
    # Spot 2
    r2 = jnp.sqrt((xx + 0.3)**2 + (yy + 0.3)**2)
    s2 = jnp.exp(-r2**2 / 0.05)
    
    return jnp.clip(s1 + s2, 0.0, 1.0)[..., None] # H, W, 1

def train_neuro_lenia():
    print("Initializing Neuro-Lenia Training...")
    
    size = 64
    batch_size = 1
    lr = 0.01
    epochs = 200
    
    # Target: The model must evolve specific noisy inputs into this Target Pattern
    target_img = create_target_pattern(size)
    batched_target = jnp.expand_dims(target_img, 0) # [1, H, W, 1]
    
    # Model
    model = LeniaRNN(steps=40)
    key = jax.random.PRNGKey(42)
    
    # Init params
    dummy_input = jnp.zeros((1, size, size, 1))
    params = model.init(key, dummy_input)
    
    # Optimizer
    tx = optax.adam(lr)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    # Loss: MSE between Final State and Target
    @jax.jit
    def train_step(state, batch_input, batch_target):
        def loss_fn(p):
            final_pred, _ = state.apply_fn(p, batch_input)
            loss = jnp.mean((final_pred - batch_target) ** 2)
            return loss
        
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss

    print("Starting Optimization Loop...")
    
    for epoch in range(epochs):
        # Data: Target + Noise
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, batched_target.shape) * 0.5
        batch_input = jnp.clip(batched_target + noise, 0.0, 1.0)
        
        state, loss = train_step(state, batch_input, batched_target)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.6f}")
            
    print("Training Complete.")
    
    # Verify Inference
    print("Testing Trained Model...")
    final_out, _ = state.apply_fn(state.params, batch_input)
    mse = jnp.mean((final_out - batched_target)**2)
    print(f"Final Test MSE: {mse:.6f}")
    
    # Check learnt params
    mu = state.params['params']['LeniaLayer_0']['mu']
    sigma = state.params['params']['LeniaLayer_0']['sigma']
    print(f"Learnt Growth Params: Mu={mu}, Sigma={sigma}")
    
    if mse < 0.05:
        print("✅ SUCCESS: Neuro-Lenia learned to stabilize the pattern!")
    else:
        print("⚠️ RESULT: Convergence incomplete.")

if __name__ == "__main__":
    train_neuro_lenia()
