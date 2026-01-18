
import sys
import os
sys.path.append(os.getcwd())
from src.search_jax import CriticalitySeeker

def verify_search():
    print("="*40)
    print("Verifying Differentiable Search")
    print("Target: Mu ~ 0.15, Sigma ~ 0.015")
    print("="*40)
    
    seeker = CriticalitySeeker(size=64)
    
    # Start from "boring" dead/stable regime
    # Mu=0.10 usually dies quickly
    start_mu = 0.12
    start_sigma = 0.02
    
    final_params, history = seeker.find_critical_parameters(
        start_mu=start_mu, 
        start_sigma=start_sigma, 
        steps=200
    )
    
    mu, sigma = final_params
    print("\n" + "="*40)
    print(f"Final Result: Mu={mu:.4f}, Sigma={sigma:.4f}")
    
    # Check if we landed in the "Soliton" sweet spot
    # Known range: Mu [0.13, 0.17], Sigma [0.013, 0.017]
    if 0.13 <= mu <= 0.17 and 0.012 <= sigma <= 0.018:
        print("✅ SUCCESS: Recovered critical regime!")
    else:
        print("⚠️ RESULT: Landed outside standard soliton range.")
        print("   (This might still be interesting, or loss needs tuning)")

if __name__ == "__main__":
    verify_search()
