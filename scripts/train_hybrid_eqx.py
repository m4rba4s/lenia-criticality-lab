
import sys
import os
sys.path.append(os.getcwd())

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

from src.neuro_lenia import LeniaRNN

def create_target_pattern(size):
    x = jnp.linspace(-1, 1, size)
    xx, yy = jnp.meshgrid(x, x)
    
    # Spot 1
    r1 = jnp.sqrt((xx - 0.3)**2 + (yy - 0.3)**2)
    s1 = jnp.exp(-r1**2 / 0.05)
    
    # Spot 2
    r2 = jnp.sqrt((xx + 0.3)**2 + (yy + 0.3)**2)
    s2 = jnp.exp(-r2**2 / 0.05)
    
    return jnp.clip(s1 + s2, 0.0, 1.0)[None, ...] # (1, H, W)

def train_neuro_lenia():
    print("Initializing Neuro-Lenia (Equinox) Training...")
    
    size = 64
    lr = 0.01
    epochs = 200
    
    # Target
    target_img = create_target_pattern(size)
    # Batch dim handled by vmap if needed, but here simple loop
    
    key = jax.random.PRNGKey(42)
    model = LeniaRNN(key, steps=40)
    
    # Optimizer
    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    
    @eqx.filter_jit
    def train_step(model, opt_state, batch_input, batch_target):
        def loss_fn(m):
            final_pred, _ = m(batch_input)
            loss = jnp.mean((final_pred - batch_target) ** 2)
            return loss
        
        # Only diff arrays
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        
        return model, opt_state, loss

    print("Starting Optimization Loop...")
    
    for epoch in range(epochs):
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, target_img.shape) * 0.5
        batch_input = jnp.clip(target_img + noise, 0.0, 1.0)
        
        model, opt_state, loss = train_step(model, opt_state, batch_input, target_img)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.6f}")
            
    print("Training Complete.")
    
    # Check params
    print(f"Learnt Mu: {model.cell.mu}")
    print(f"Learnt Sigma: {model.cell.sigma}")
    
    # Test
    final, _ = model(batch_input)
    mse = jnp.mean((final - target_img)**2)
    print(f"Final MSE: {mse:.6f}")
    
    if mse < 0.05:
        print("✅ SUCCESS: Neuro-Lenia learned to stabilize!")
    else:
        print("⚠️ RESULT: Convergence incomplete.")

if __name__ == "__main__":
    train_neuro_lenia()
