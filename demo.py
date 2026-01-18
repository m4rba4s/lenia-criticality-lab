#!/usr/bin/env python3
"""
LENIA CRITICALITY - INTERACTIVE DEMOS

Launch interactive visualizations of key research findings.

Usage:
    python demo.py              # Show menu
    python demo.py nand         # NAND gate demonstration
    python demo.py phase        # Phase space explorer
    python demo.py signal       # Signal propagation demo
    python demo.py species      # Elite species viewer
"""

import sys
import subprocess
from pathlib import Path


DEMOS = {
    'nand': {
        'script': 'demo_nand_gate.py',
        'title': 'NAND Gate Demonstration',
        'desc': 'Emergent NAND-like computation via self-repair threshold',
        'icon': '🔌',
    },
    'phase': {
        'script': 'demo_phase_explorer.py',
        'title': 'Phase Space Explorer',
        'desc': 'Interactive exploration of order/chaos/critical regimes',
        'icon': '🗺️',
    },
    'signal': {
        'script': 'demo_signal_propagation.py',
        'title': 'Signal Propagation',
        'desc': 'Perturbation waves and lagged correlations',
        'icon': '📡',
    },
    'species': {
        'script': 'view_species.py',
        'title': 'Elite Species Viewer',
        'desc': 'View discovered critical and chaotic organisms',
        'icon': '🧬',
    },
}


def print_banner():
    print("""
\033[36m╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ██╗     ███████╗███╗   ██╗██╗ █████╗                          ║
║   ██║     ██╔════╝████╗  ██║██║██╔══██╗                         ║
║   ██║     █████╗  ██╔██╗ ██║██║███████║                         ║
║   ██║     ██╔══╝  ██║╚██╗██║██║██╔══██║                         ║
║   ███████╗███████╗██║ ╚████║██║██║  ██║                         ║
║   ╚══════╝╚══════╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝                         ║
║                                                                  ║
║   \033[33mEMERGENT COMPUTATION AT THE EDGE OF CHAOS\033[36m                      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝\033[0m
""")


def print_menu():
    print("\033[1mAVAILABLE DEMONSTRATIONS:\033[0m\n")

    for key, demo in DEMOS.items():
        print(f"  \033[33m{demo['icon']} {key:8}\033[0m  {demo['title']}")
        print(f"             \033[90m{demo['desc']}\033[0m\n")

    print("\033[1mUSAGE:\033[0m")
    print("  python demo.py <name>     Run specific demo")
    print("  python demo.py nand       Example: NAND gate demo")
    print()


def run_demo(name: str):
    if name not in DEMOS:
        print(f"\033[31mError: Unknown demo '{name}'\033[0m")
        print(f"Available: {', '.join(DEMOS.keys())}")
        return 1

    demo = DEMOS[name]
    script = Path(__file__).parent / demo['script']

    if not script.exists():
        print(f"\033[31mError: Script not found: {script}\033[0m")
        return 1

    print(f"\n\033[36m{demo['icon']} Launching {demo['title']}...\033[0m\n")

    try:
        subprocess.run([sys.executable, str(script)], check=True)
    except KeyboardInterrupt:
        print("\n\033[33mDemo interrupted.\033[0m")
    except subprocess.CalledProcessError as e:
        print(f"\033[31mDemo exited with error: {e}\033[0m")
        return 1

    return 0


def main():
    print_banner()

    if len(sys.argv) < 2:
        print_menu()

        # Interactive selection
        print("\033[1mSelect demo (or 'q' to quit):\033[0m ", end='')
        try:
            choice = input().strip().lower()
            if choice == 'q' or choice == 'quit':
                print("Goodbye!")
                return 0
            if choice:
                return run_demo(choice)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            return 0
    else:
        demo_name = sys.argv[1].lower()
        if demo_name in ['-h', '--help', 'help']:
            print_menu()
            return 0
        return run_demo(demo_name)


if __name__ == '__main__':
    sys.exit(main())
