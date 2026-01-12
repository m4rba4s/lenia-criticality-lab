#!/usr/bin/env python3
"""
Elite Species Viewer

Run discovered critical and chaotic species in real-time.
Uses matplotlib animation for visualization.

Usage:
    python view_elite.py                    # View all elite species
    python view_elite.py --species 0        # View specific species by index
    python view_elite.py --chaos            # View only chaotic species
    python view_elite.py --critical         # View only critical species
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider

from src.simulation import LeniaSimulation, LeniaConfig


class EliteViewer:
    """Interactive viewer for elite Lenia species."""

    def __init__(self, elite_data: list, start_idx: int = 0):
        self.elite = elite_data
        self.current_idx = start_idx
        self.sim = None
        self.running = True
        self.speed = 1

        # Setup figure
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.patch.set_facecolor('#1a1a2e')

        # Main simulation display
        self.ax_main = self.fig.add_axes([0.05, 0.25, 0.6, 0.7])
        self.ax_main.set_facecolor('#1a1a2e')

        # Info panel
        self.ax_info = self.fig.add_axes([0.68, 0.25, 0.3, 0.7])
        self.ax_info.set_facecolor('#1a1a2e')
        self.ax_info.axis('off')

        # History graph
        self.ax_history = self.fig.add_axes([0.05, 0.08, 0.6, 0.12])
        self.ax_history.set_facecolor('#2a2a4e')

        # Buttons
        self.btn_prev = Button(plt.axes([0.68, 0.08, 0.1, 0.05]), '< Prev',
                               color='#2a9d8f', hovercolor='#3ab5a5')
        self.btn_next = Button(plt.axes([0.79, 0.08, 0.1, 0.05]), 'Next >',
                               color='#2a9d8f', hovercolor='#3ab5a5')
        self.btn_pause = Button(plt.axes([0.90, 0.08, 0.08, 0.05]), 'Pause',
                                color='#e63946', hovercolor='#ff4d5a')

        self.btn_prev.on_clicked(self.prev_species)
        self.btn_next.on_clicked(self.next_species)
        self.btn_pause.on_clicked(self.toggle_pause)

        # Speed slider
        self.ax_speed = plt.axes([0.68, 0.15, 0.28, 0.03])
        self.slider_speed = Slider(self.ax_speed, 'Speed', 1, 10, valinit=1, valstep=1)
        self.slider_speed.on_changed(self.update_speed)

        # State
        self.mass_history = []
        self.im = None

        # Load first species
        self.load_species(self.current_idx)

    def load_species(self, idx: int):
        """Load and initialize a species."""
        if idx < 0 or idx >= len(self.elite):
            return

        self.current_idx = idx
        species = self.elite[idx]

        # Create simulation
        config = LeniaConfig(
            grid_size=256,
            mu=species['mu'],
            sigma=species['sigma'],
            seed=42,
            init_size=0.2,
            init_density=0.5,
        )
        self.sim = LeniaSimulation(config)
        self.mass_history = [self.sim.mass()]

        # Update display
        self.update_info()

    def update_info(self):
        """Update info panel."""
        self.ax_info.clear()
        self.ax_info.axis('off')

        species = self.elite[self.current_idx]
        lyap = species.get('lyapunov', 0) or 0
        classification = species.get('classification', 'unknown')

        # Color based on type
        if classification == 'chaotic':
            title_color = '#e63946'
            icon = '🔥'
        elif abs(lyap) < 0.015:
            title_color = '#ffd700'
            icon = '💎'
        else:
            title_color = '#2a9d8f'
            icon = '🧬'

        lines = [
            (f"{icon} SPECIES #{self.current_idx + 1}/{len(self.elite)}", 18, title_color),
            ("", 12, 'white'),
            (f"Classification: {classification.upper()}", 14, 'white'),
            ("", 12, 'white'),
            ("─" * 25, 10, '#555'),
            ("PARAMETERS", 12, '#888'),
            (f"  μ (growth center) = {species['mu']:.4f}", 11, '#aaa'),
            (f"  σ (growth width)  = {species['sigma']:.5f}", 11, '#aaa'),
            ("", 10, 'white'),
            ("─" * 25, 10, '#555'),
            ("DYNAMICS", 12, '#888'),
            (f"  Lyapunov λ = {lyap:+.5f}", 11, '#aaa'),
            ("", 8, 'white'),
        ]

        # Lyapunov interpretation
        if lyap > 0.02:
            lines.append(("  ⚡ CHAOTIC (λ > 0)", 11, '#e63946'))
            lines.append(("  Exponential divergence", 9, '#888'))
        elif lyap < -0.02:
            lines.append(("  ❄️  ORDERED (λ < 0)", 11, '#457b9d'))
            lines.append(("  Perturbations decay", 9, '#888'))
        else:
            lines.append(("  ⚖️  CRITICAL (λ ≈ 0)", 11, '#ffd700'))
            lines.append(("  Edge of chaos!", 9, '#888'))

        y = 0.95
        for text, size, color in lines:
            self.ax_info.text(0.05, y, text, fontsize=size, color=color,
                              transform=self.ax_info.transAxes, family='monospace')
            y -= 0.07

        # Controls hint
        self.ax_info.text(0.05, 0.05, "← → : Switch species\nSPACE: Pause/Resume",
                          fontsize=9, color='#666', transform=self.ax_info.transAxes,
                          family='monospace')

    def update(self, frame):
        """Animation update function."""
        if not self.running or self.sim is None:
            return

        # Run simulation steps
        for _ in range(self.speed):
            self.sim.step()

        self.mass_history.append(self.sim.mass())
        if len(self.mass_history) > 300:
            self.mass_history = self.mass_history[-300:]

        # Update main view
        self.ax_main.clear()
        self.ax_main.imshow(self.sim.world, cmap='viridis', vmin=0, vmax=1)
        self.ax_main.axis('off')
        self.ax_main.set_title(f"Step {self.sim.step_count}", color='white', fontsize=10)

        # Update history
        self.ax_history.clear()
        self.ax_history.plot(self.mass_history, color='#2a9d8f', linewidth=1)
        self.ax_history.fill_between(range(len(self.mass_history)), self.mass_history,
                                     alpha=0.3, color='#2a9d8f')
        self.ax_history.set_xlim(0, 300)
        self.ax_history.set_ylabel('Mass', color='#888', fontsize=8)
        self.ax_history.tick_params(colors='#888', labelsize=7)
        self.ax_history.spines['top'].set_visible(False)
        self.ax_history.spines['right'].set_visible(False)
        self.ax_history.spines['bottom'].set_color('#444')
        self.ax_history.spines['left'].set_color('#444')

    def prev_species(self, event):
        self.load_species((self.current_idx - 1) % len(self.elite))

    def next_species(self, event):
        self.load_species((self.current_idx + 1) % len(self.elite))

    def toggle_pause(self, event):
        self.running = not self.running
        self.btn_pause.label.set_text('Resume' if not self.running else 'Pause')

    def update_speed(self, val):
        self.speed = int(val)

    def on_key(self, event):
        if event.key == 'left':
            self.prev_species(None)
        elif event.key == 'right':
            self.next_species(None)
        elif event.key == ' ':
            self.toggle_pause(None)
        elif event.key == 'escape':
            plt.close()

    def run(self):
        """Start the viewer."""
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        ani = animation.FuncAnimation(self.fig, self.update, interval=50, blit=False)
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="View elite Lenia species")
    parser.add_argument('--species', '-s', type=int, default=0, help='Species index to start with')
    parser.add_argument('--chaos', action='store_true', help='Show only chaotic species')
    parser.add_argument('--critical', action='store_true', help='Show only critical species')
    parser.add_argument('--file', '-f', type=str, default='experiments/elite_species.json',
                        help='Path to elite species JSON')
    args = parser.parse_args()

    # Load elite species
    elite_path = Path(__file__).parent.parent / args.file
    if not elite_path.exists():
        print(f"Error: {elite_path} not found")
        print("Run the phase diagram experiment first!")
        sys.exit(1)

    with open(elite_path) as f:
        elite = json.load(f)

    print(f"Loaded {len(elite)} elite species")

    # Filter if requested
    if args.chaos:
        elite = [s for s in elite if s.get('classification') == 'chaotic']
        print(f"Filtered to {len(elite)} chaotic species")
    elif args.critical:
        elite = [s for s in elite if abs(s.get('lyapunov', 1)) < 0.015]
        print(f"Filtered to {len(elite)} critical species")

    if not elite:
        print("No species to display!")
        sys.exit(1)

    # Print quick reference
    print("\nElite Species:")
    for i, s in enumerate(elite[:10]):
        lyap = s.get('lyapunov', 0) or 0
        cls = s.get('classification', '?')
        icon = '🔥' if cls == 'chaotic' else ('💎' if abs(lyap) < 0.015 else '🧬')
        print(f"  {i}: {icon} μ={s['mu']:.4f} σ={s['sigma']:.5f} λ={lyap:+.4f} [{cls}]")
    if len(elite) > 10:
        print(f"  ... and {len(elite) - 10} more")

    print("\nControls: ← → switch species, SPACE pause, ESC quit")
    print("Starting viewer...")

    viewer = EliteViewer(elite, start_idx=args.species)
    viewer.run()


if __name__ == '__main__':
    main()
