"""nolab -- single shared library for the Neural Observer v2 package.

One solver, one observation family, one observer loop, one metric
definition.  Everything in scripts/ and exp14/ imports from here; the
per-experiment copies that existed in v1 are gone.
"""

from .solver import (Grid, KolmogorovFlow, SOLVER_VERSION,
                     solver_fingerprint, equilibration_time)
from .observations import (SpectralObs, PointObs, FixedPointObs,
                           ShellGainObs, VelocityPointObs,
                           NonlinearPointObs, PointShellObs,
                           covering_radius, anisotropy_index)
from .observer import run_observer, random_observer_state
from .metrics import sync_rate, bootstrap_Nc, first_consistent_K
from .configs import (uniform, ground_tracks, clustered, blind_half,
                      lattice, GENERATORS)
from .letkf import LETKF, run_letkf, LETKFDivergence, gaspari_cohn
from .harness import get_truth, truth_path, trial, save, load_results
from .configs_fix import stripes_exact, corridor, lattice_m
from .geometry_fix import (corridor_diagnostics, delta_K, band_coupling,
                           layout_report)
from .metrics_fix import (sync_rate_multi, aggregate_mu, Nc_vs_cstar)

__all__ = ["Grid", "KolmogorovFlow", "SOLVER_VERSION", "solver_fingerprint",
           "SpectralObs", "PointObs", "FixedPointObs", "ShellGainObs",
           "VelocityPointObs", "NonlinearPointObs", "PointShellObs",
           "covering_radius", "anisotropy_index", "run_observer",
           "random_observer_state", "sync_rate", "bootstrap_Nc",
           "first_consistent_K", "uniform", "ground_tracks", "clustered",
           "blind_half", "lattice", "GENERATORS", "LETKF", "run_letkf",
           "LETKFDivergence", "gaspari_cohn", "get_truth", "truth_path", "stripes_exact", "corridor", "lattice_m",
           "corridor_diagnostics", "delta_K", "band_coupling",
           "layout_report", "sync_rate_multi", "aggregate_mu",
           "Nc_vs_cstar",
           "trial", "save", "load_results", "equilibration_time"]
