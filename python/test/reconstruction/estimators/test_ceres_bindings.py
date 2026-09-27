"""Port of upstream pycolmap estimators/ceres_bindings_test.py (Open3D style).

The upstream accesses the surface as ``pycolmap._core.pyceres``; the fork
exposes it as ``reconstruction.pyceres``. Extended with numeric checks on the
options/summary surface that the upstream suite only touches existence-wise.
"""
import os

import pytest


def test_pyceres_submodule_exists(recon):
    assert hasattr(recon, "pyceres")


def test_minimizer_type_enum(recon):
    pyceres = recon.pyceres
    assert {
        k: int(v) for k, v in pyceres.MinimizerType.__members__.items()
    } == {
        "LINE_SEARCH": 0,
        "TRUST_REGION": 1,
    }


def test_linear_solver_type_enum(recon):
    pyceres = recon.pyceres
    assert {
        k: int(v) for k, v in pyceres.LinearSolverType.__members__.items()
    } == {
        "DENSE_NORMAL_CHOLESKY": 0,
        "DENSE_QR": 1,
        "SPARSE_NORMAL_CHOLESKY": 2,
        "DENSE_SCHUR": 3,
        "SPARSE_SCHUR": 4,
        "ITERATIVE_SCHUR": 5,
        "CGNR": 6,
    }


def test_trust_region_strategy_type_enum(recon):
    pyceres = recon.pyceres
    assert {
        k: int(v)
        for k, v in pyceres.TrustRegionStrategyType.__members__.items()
    } == {
        "LEVENBERG_MARQUARDT": 0,
        "DOGLEG": 1,
    }


def test_logging_type_enum(recon):
    pyceres = recon.pyceres
    assert {k: int(v) for k, v in pyceres.LoggingType.__members__.items()} == {
        "SILENT": 0,
        "PER_MINIMIZER_ITERATION": 1,
    }


def test_termination_type_enum(recon):
    pyceres = recon.pyceres
    assert {
        k: int(v) for k, v in pyceres.TerminationType.__members__.items()
    } == {
        "CONVERGENCE": 0,
        "NO_CONVERGENCE": 1,
        "FAILURE": 2,
        "USER_SUCCESS": 3,
        "USER_FAILURE": 4,
    }


def test_solver_options_default_init(recon):
    options = recon.pyceres.SolverOptions()
    assert options is not None


def test_solver_options_is_valid(recon):
    options = recon.pyceres.SolverOptions()
    valid, error = options.IsValid()
    assert valid is True
    assert error == ""


def test_solver_options_roundtrip(recon):
    options = recon.pyceres.SolverOptions()
    options.max_num_iterations = 123
    assert options.max_num_iterations == 123
    options.function_tolerance = 1e-9
    assert options.function_tolerance == pytest.approx(1e-9)
    options.minimizer_type = recon.pyceres.MinimizerType.LINE_SEARCH
    assert options.minimizer_type == recon.pyceres.MinimizerType.LINE_SEARCH
    options.linear_solver_type = recon.pyceres.LinearSolverType.DENSE_QR
    assert options.linear_solver_type == recon.pyceres.LinearSolverType.DENSE_QR


def test_solver_options_num_threads_effective(recon):
    """Non-positive thread counts resolve to the hardware concurrency
    (upstream GetEffectiveNumThreads semantics)."""
    options = recon.pyceres.SolverOptions()
    options.num_threads = -1
    assert options.num_threads == os.cpu_count()
    options.num_threads = 2
    assert options.num_threads == 2


def test_solver_summary_default_init(recon):
    summary = recon.pyceres.SolverSummary()
    assert summary is not None


def test_solver_summary_brief_report(recon):
    summary = recon.pyceres.SolverSummary()
    report = summary.BriefReport()
    assert isinstance(report, str)


def test_solver_summary_full_report(recon):
    summary = recon.pyceres.SolverSummary()
    report = summary.FullReport()
    assert isinstance(report, str)


def test_solver_summary_is_solution_usable(recon):
    summary = recon.pyceres.SolverSummary()
    result = summary.IsSolutionUsable()
    assert isinstance(result, bool)
    assert result is False


def test_iteration_summary_accessible(recon):
    summary = recon.pyceres.IterationSummary()
    assert summary is not None
