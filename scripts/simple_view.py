#!/usr/bin/env python3
"""
Simple Elite Viewer - guaranteed to work
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src.simulation import LeniaSimulation, LeniaConfig

# Load elite
elite_path = Path(__file__).parent.parent / 'experiments' / 'elite_species.json'
if not elite_path.exists():
    # Use hardcoded best species
    elite = [
        {'mu': 0.1585, 'sigma': 0.01885, 'lyapunov': -0.000124, 'classification': 'stable'},
        {'mu': 0.1672, 'sigma': 0.02346, 'lyapunov': 0.0351, 'classification': 'chaotic'},
        {'mu': 0.1715, 'sigma': 0.02423, 'lyapunov': 0.0398, 'classification': 'chaotic'},
    ]
else:
    with open(elite_path) as f:
        elite = json.load(f)

print(f"Loaded {len(elite)} species")

# Pick species (change this number to view different ones)
SPECIES_IDX = 0  # 0 = most critical, try 1,2,3... for chaotic ones

species = elite[SPECIES_IDX]
print(f"\nViewing species #{SPECIES_IDX}:")
print(f"  mu = {species['mu']}")
print(f"  sigma = {species['sigma']}")
print(f"  lyapunov = {species.get('lyapunov', 'N/A')}")
print(f"  type = {species.get('classification', 'unknown')}")

# Create simulation
config = LeniaConfig(
    grid_size=256,
    mu=species['mu'],
    sigma=species['sigma'],
    seed=42,
    init_size=0.2,
    init_density=0.5,
)
sim = LeniaSimulation(config)

# Setup plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('#1a1a2e')

im = ax1.imshow(sim.world, cmap='viridis', vmin=0, vmax=1, animated=True)
ax1.axis('off')
ax1.set_title('Lenia', color='white')

mass_history = []
line, = ax2.plot([], [], 'g-', lw=2)
ax2.set_xlim(0, 500)
ax2.set_ylim(0, 500)
ax2.set_facecolor('#2a2a4e')
ax2.set_xlabel('Step', color='white')
ax2.set_ylabel('Mass', color='white')
ax2.tick_params(colors='white')

def init():
    return im, line

def update(frame):
    # Run 3 steps per frame for speed
    for _ in range(3):
        sim.step()

    mass_history.append(sim.mass())

    im.set_array(sim.world)
    ax1.set_title(f'Step {sim.step_count} | Mass={sim.mass():.0f}', color='white')

    line.set_data(range(len(mass_history)), mass_history)
    if len(mass_history) > 10:
        ax2.set_ylim(0, max(mass_history) * 1.2)
        ax2.set_xlim(0, max(500, len(mass_history)))

    return im, line

print("\nStarting animation... Close window to exit.")
ani = FuncAnimation(fig, update, init_func=init, frames=2000, interval=30, blit=True)
plt.tight_layout()
plt.show()
