"""
Lenia Criticality Research Package

Investigating edge-of-chaos dynamics in continuous cellular automata.
"""

__version__ = "0.1.0"
__author__ = "Lenia Research"

# Core modules (no external dependencies beyond numpy/scipy)
from .simulation import LeniaSimulation, LeniaConfig
from .metrics import LyapunovEstimator, SpatialAnalyzer, InformationMetrics

# Experiment framework requires pandas - import conditionally
def _get_experiment_classes():
    from .experiment import Experiment, ExperimentRunner
    return Experiment, ExperimentRunner

def _get_analysis_functions():
    from .analysis import (setup_publication_style, plot_phase_diagram,
                           plot_lyapunov_diagram, summarize_results)
    return setup_publication_style, plot_phase_diagram, plot_lyapunov_diagram, summarize_results
