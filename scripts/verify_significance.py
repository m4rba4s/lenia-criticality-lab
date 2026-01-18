
import sys
import os
sys.path.append(os.getcwd())
import jax
import jax.numpy as jnp
from jax import random
import numpy as np

from src.metrics_jax import compute_te_significance

def verify_significance():
    print("="*50)
    print("Verifying Statistical Significance (Surrogate Test)")
    print("="*50)
    
    key = random.PRNGKey(42)
    T = 1000
    
    # 1. Null Case: Independent Noise
    print("\n[Case 1] Independent Noise (Should FAIL to reject Null)")
    key, k1, k2, k3 = random.split(key, 4)
    x = random.normal(k1, (T,))
    y = random.normal(k2, (T,))
    
    # Warmup
    print("Compiling...")
    _ = compute_te_significance(k3, x, y, 8, 100)
    
    # Run
    te, p_val, z = compute_te_significance(k3, x, y, 8, 100)
    print(f"TE: {te:.4f} | P-value: {p_val:.4f} | Z-score: {z:.4f}")
    
    if p_val > 0.05:
        print("✅ PASS: Null hypothesis NOT rejected (Correct)")
    else:
        print("⚠️ WARN: Null hypothesis rejected (False Positive?)")
        
    # 2. Causal Case: Y = X shifted
    print("\n[Case 2] Causal Link (Should REJECT Null)")
    x = random.normal(k1, (T,))
    y = jnp.roll(x, 1) # Lag=1 match
    
    te, p_val, z = compute_te_significance(k3, x, y, 8, 1000) # More surrogates for precision
    print(f"TE: {te:.4f} | P-value: {p_val:.4f} | Z-score: {z:.4f}")
    
    if p_val < 0.01:
        print("✅ PASS: Null hypothesis REJECTED (Correct, p < 0.01)")
    else:
        print("❌ FAIL: Failed to detect significant causality")

if __name__ == "__main__":
    verify_significance()
