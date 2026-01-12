#!/usr/bin/env python3
"""
Command-line interface for running Lenia criticality experiments.

Usage:
    python run_experiment.py --experiment phase_diagram --resolution 30
    python run_experiment.py --experiment lyapunov --resolution 20
    python run_experiment.py --experiment critical_line --mu 0.15
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiment import ExperimentRunner, ExperimentConfig, Experiment
from src.analysis import setup_publication_style, plot_criticality_signatures, summarize_results


def main():
    parser = argparse.ArgumentParser(
        description="Run Lenia criticality experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (low resolution)
  python run_experiment.py --experiment phase_diagram --resolution 10 --workers 2

  # Full phase diagram
  python run_experiment.py --experiment phase_diagram --resolution 40 --workers 8

  # Lyapunov-focused scan
  python run_experiment.py --experiment lyapunov --resolution 25

  # Critical line analysis at specific mu
  python run_experiment.py --experiment critical_line --mu 0.15 --resolution 50
        """
    )

    parser.add_argument('--experiment', '-e', type=str, required=True,
                        choices=['phase_diagram', 'lyapunov', 'critical_line', 'custom'],
                        help='Type of experiment to run')

    parser.add_argument('--resolution', '-r', type=int, default=30,
                        help='Grid resolution (default: 30)')

    parser.add_argument('--mu', type=float, default=0.15,
                        help='Fixed mu for critical_line experiment')

    parser.add_argument('--mu-range', type=float, nargs=2, default=[0.08, 0.25],
                        help='Mu range (default: 0.08 0.25)')

    parser.add_argument('--sigma-range', type=float, nargs=2, default=[0.005, 0.035],
                        help='Sigma range (default: 0.005 0.035)')

    parser.add_argument('--grid-size', type=int, default=128,
                        help='Simulation grid size (default: 128)')

    parser.add_argument('--workers', '-w', type=int, default=4,
                        help='Number of parallel workers (default: 4)')

    parser.add_argument('--output', '-o', type=str, default='experiments',
                        help='Output directory (default: experiments)')

    parser.add_argument('--no-lyapunov', action='store_true',
                        help='Skip Lyapunov calculation (faster)')

    parser.add_argument('--serial', action='store_true',
                        help='Run serially (no parallelism, for debugging)')

    parser.add_argument('--plot', action='store_true',
                        help='Generate plots after experiment')

    args = parser.parse_args()

    # Run experiment
    if args.experiment == 'phase_diagram':
        df = ExperimentRunner.phase_diagram(
            name='phase_diagram',
            mu_range=tuple(args.mu_range),
            sigma_range=tuple(args.sigma_range),
            resolution=args.resolution,
            grid_size=args.grid_size,
            n_workers=args.workers,
            output_dir=args.output,
            compute_lyapunov=not args.no_lyapunov,
        )

    elif args.experiment == 'lyapunov':
        df = ExperimentRunner.lyapunov_scan(
            name='lyapunov_scan',
            mu_range=tuple(args.mu_range),
            sigma_range=tuple(args.sigma_range),
            resolution=args.resolution,
            grid_size=args.grid_size,
            n_workers=args.workers,
            output_dir=args.output,
        )

    elif args.experiment == 'critical_line':
        df = ExperimentRunner.criticality_line(
            mu_center=args.mu,
            sigma_range=tuple(args.sigma_range),
            resolution=args.resolution,
            grid_size=args.grid_size,
            n_workers=args.workers,
            output_dir=args.output,
        )

    else:
        print(f"Unknown experiment: {args.experiment}")
        sys.exit(1)

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    summary = summarize_results(df)
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Plot if requested
    if args.plot:
        setup_publication_style()
        fig = plot_criticality_signatures(df)
        plot_path = Path(args.output) / f"{args.experiment}_summary.png"
        fig.savefig(plot_path, dpi=150)
        print(f"\nPlot saved to: {plot_path}")


if __name__ == '__main__':
    main()
