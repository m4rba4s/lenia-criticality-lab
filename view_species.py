#!/usr/bin/env python3
"""
🧬 ELITE SPECIES VIEWER v2 — с автокалибровкой
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from scipy.signal import fftconvolve

from src.simulation import LeniaSimulation, LeniaConfig


def calibrate_init(mu, sigma, grid_size=150):
    """
    Автокалибровка: подбираем init_density чтобы potential ≈ mu
    """
    for density in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        for size in [0.2, 0.25, 0.3]:
            config = LeniaConfig(
                grid_size=grid_size,
                mu=mu, sigma=sigma,
                seed=42,
                init_size=size,
                init_density=density,
            )
            sim = LeniaSimulation(config)

            # Проверяем выживание
            for _ in range(80):
                sim.step()

            if sim.mass() > 30:
                return density, size

    # Fallback
    return 0.5, 0.25


class SpeciesViewer:
    def __init__(self, species_list):
        self.species_list = species_list
        self.current_species = None
        self.sim = None
        self.running = True
        self.mass_history = []
        self.calibration_cache = {}

        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(14, 7))
        self.fig.patch.set_facecolor('#0a0a15')

        self.ax_sim = self.fig.add_axes([0.02, 0.15, 0.55, 0.8])
        self.ax_mass = self.fig.add_axes([0.60, 0.55, 0.38, 0.35])
        self.ax_info = self.fig.add_axes([0.60, 0.15, 0.38, 0.35])
        self.ax_info.axis('off')

        self.btn_next = Button(plt.axes([0.60, 0.02, 0.12, 0.06]),
                               '🎲 RANDOM', color='#2a4a6a', hovercolor='#3a6a9a')
        self.btn_pause = Button(plt.axes([0.73, 0.02, 0.08, 0.06]),
                                '⏸', color='#4a2a2a', hovercolor='#7a3a3a')
        self.btn_critical = Button(plt.axes([0.82, 0.02, 0.08, 0.06]),
                                   '💎', color='#4a4a2a', hovercolor='#7a7a3a')
        self.btn_chaos = Button(plt.axes([0.91, 0.02, 0.08, 0.06]),
                                '🔥', color='#6a4a2a', hovercolor='#9a6a3a')

        self.btn_next.on_clicked(self.load_random)
        self.btn_pause.on_clicked(self.toggle_pause)
        self.btn_critical.on_clicked(self.load_critical)
        self.btn_chaos.on_clicked(self.load_chaotic)

        # Pre-calibrate a few species
        print("Калибрую начальные условия...")
        self._precalibrate()

        self.load_random(None)

    def _precalibrate(self):
        """Pre-calibrate init params for all species"""
        for i, sp in enumerate(self.species_list[:10]):  # First 10
            key = (round(sp['mu'], 4), round(sp['sigma'], 5))
            if key not in self.calibration_cache:
                density, size = calibrate_init(sp['mu'], sp['sigma'])
                self.calibration_cache[key] = (density, size)
                print(f"  {i+1}/{min(10, len(self.species_list))}: μ={sp['mu']:.4f} → density={density:.2f}")

    def load_random(self, event):
        self.current_species = random.choice(self.species_list)
        self._init_simulation()

    def load_critical(self, event):
        """Load critical species (λ ≈ 0)"""
        critical = [s for s in self.species_list
                    if s.get('lyapunov') is not None and abs(s['lyapunov']) < 0.015]
        if critical:
            self.current_species = random.choice(critical)
        else:
            self.current_species = self.species_list[0]
        self._init_simulation()

    def load_chaotic(self, event):
        """Load chaotic species (λ > 0)"""
        chaotic = [s for s in self.species_list
                   if s.get('lyapunov') is not None and s['lyapunov'] > 0.02]
        if chaotic:
            self.current_species = random.choice(chaotic)
        else:
            self.current_species = max(self.species_list,
                                       key=lambda x: x.get('lyapunov') or 0)
        self._init_simulation()

    def _init_simulation(self):
        sp = self.current_species
        key = (round(sp['mu'], 4), round(sp['sigma'], 5))

        # Get calibrated params or calibrate now
        if key in self.calibration_cache:
            density, size = self.calibration_cache[key]
        else:
            print(f"Калибрую μ={sp['mu']:.4f}...")
            density, size = calibrate_init(sp['mu'], sp['sigma'])
            self.calibration_cache[key] = (density, size)

        config = LeniaConfig(
            grid_size=150,
            mu=sp['mu'],
            sigma=sp['sigma'],
            seed=random.randint(0, 10000),
            init_size=size,
            init_density=density,
        )

        self.sim = LeniaSimulation(config)
        self.mass_history = [self.sim.mass()]
        self.running = True
        self._update_info()

    def _update_info(self):
        self.ax_info.clear()
        self.ax_info.axis('off')

        sp = self.current_species
        lyap = sp.get('lyapunov') or 0
        cls = sp.get('classification', 'unknown').upper()

        if cls == 'CHAOTIC' or lyap > 0.02:
            icon, color, desc = '🔥', '#ff6b6b', 'CHAOTIC'
        elif abs(lyap) < 0.015:
            icon, color, desc = '💎', '#ffd700', 'CRITICAL'
        else:
            icon, color, desc = '🧬', '#6bff6b', 'STABLE'

        info = f"""{icon} {desc}

μ  = {sp['mu']:.5f}
σ  = {sp['sigma']:.6f}
λ  = {lyap:+.5f}

"""
        if lyap > 0.02:
            info += "⚡ Хаос: может умереть"
        elif lyap < -0.02:
            info += "❄️ Порядок: стабильно"
        else:
            info += "⚖️ EDGE OF CHAOS!"

        self.ax_info.text(0.05, 0.95, info, fontsize=11, color=color,
                         transform=self.ax_info.transAxes,
                         fontfamily='monospace', verticalalignment='top')

    def toggle_pause(self, event):
        self.running = not self.running

    def update(self, frame):
        if not self.running or self.sim is None:
            return []

        for _ in range(2):
            self.sim.step()
        self.mass_history.append(self.sim.mass())

        if len(self.mass_history) > 500:
            self.mass_history = self.mass_history[-500:]

        # Simulation view
        self.ax_sim.clear()
        self.ax_sim.imshow(self.sim.world, cmap='viridis', vmin=0, vmax=1)
        self.ax_sim.axis('off')

        status = "ALIVE" if self.sim.mass() > 10 else "DEAD"
        color = '#4f8' if status == "ALIVE" else '#f44'
        self.ax_sim.set_title(f"Step {self.sim.step_count} | Mass: {self.sim.mass():.0f} | {status}",
                             color=color, fontsize=12, fontweight='bold')

        # Mass history
        self.ax_mass.clear()
        self.ax_mass.fill_between(range(len(self.mass_history)),
                                  self.mass_history, alpha=0.3, color='#4a9')
        self.ax_mass.plot(self.mass_history, color='#4fc', linewidth=1.5)
        self.ax_mass.set_facecolor('#151525')
        self.ax_mass.set_xlim(0, max(500, len(self.mass_history)))
        if self.mass_history:
            self.ax_mass.set_ylim(0, max(max(self.mass_history) * 1.2, 50))
        self.ax_mass.set_ylabel('Mass', color='#888', fontsize=9)
        self.ax_mass.tick_params(colors='#666', labelsize=8)

        return []

    def run(self):
        self.ani = FuncAnimation(self.fig, self.update, interval=50, blit=False)
        plt.show()


def main():
    elite_path = Path(__file__).parent / 'experiments' / 'elite_species.json'

    if not elite_path.exists():
        print("Создаю дефолтный генофонд...")
        species_list = [
            {'mu': 0.1585, 'sigma': 0.01885, 'lyapunov': -0.0001, 'classification': 'stable'},
            {'mu': 0.1672, 'sigma': 0.01885, 'lyapunov': 0.0006, 'classification': 'stable'},
            {'mu': 0.1497, 'sigma': 0.01654, 'lyapunov': 0.0016, 'classification': 'stable'},
            {'mu': 0.1715, 'sigma': 0.02115, 'lyapunov': 0.0018, 'classification': 'stable'},
            {'mu': 0.1541, 'sigma': 0.01808, 'lyapunov': 0.0258, 'classification': 'chaotic'},
        ]
    else:
        with open(elite_path) as f:
            species_list = json.load(f)

    print(f"""
╔═══════════════════════════════════════════════╗
║  🧬 ELITE SPECIES VIEWER v2                   ║
╠═══════════════════════════════════════════════╣
║  Loaded: {len(species_list):3d} species                        ║
║                                               ║
║  🎲 = Random   💎 = Critical   🔥 = Chaotic   ║
║  ⏸  = Pause                                   ║
╚═══════════════════════════════════════════════╝
    """)

    viewer = SpeciesViewer(species_list)
    viewer.run()


if __name__ == '__main__':
    main()
