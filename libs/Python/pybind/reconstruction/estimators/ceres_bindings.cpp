// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/reconstruction/estimators/ceres_bindings.h"

#include <ceres/ceres.h>

#include <string>
#include <utility>

#include "pybind/reconstruction/enum_ctor.h"
#include "util/threading.h"

namespace cloudViewer {
namespace reconstruction {
namespace estimators {
namespace {

using colmap::GetEffectiveNumThreads;

void BindCeresTypes(py::module& m) {
    // clang-format off
    auto ownt = py::enum_<ceres::Ownership>(m, "Ownership", py::module_local())
                    .value("DO_NOT_TAKE_OWNERSHIP",
                           ceres::Ownership::DO_NOT_TAKE_OWNERSHIP)
                    .value("TAKE_OWNERSHIP", ceres::Ownership::TAKE_OWNERSHIP)
                    .export_values();
    AddStringToEnumConstructor(ownt);

    auto mt =
            py::enum_<ceres::MinimizerType>(m, "MinimizerType",
                                            py::module_local())
                    .value("LINE_SEARCH", ceres::MinimizerType::LINE_SEARCH)
                    .value("TRUST_REGION",
                           ceres::MinimizerType::TRUST_REGION);
    AddStringToEnumConstructor(mt);

    auto linesearcht =
            py::enum_<ceres::LineSearchType>(m, "LineSearchType",
                                             py::module_local())
                    .value("ARMIJO", ceres::LineSearchType::ARMIJO)
                    .value("WOLFE", ceres::LineSearchType::WOLFE);
    AddStringToEnumConstructor(linesearcht);

    auto lsdt = py::enum_<ceres::LineSearchDirectionType>(
                        m, "LineSearchDirectionType", py::module_local())
                        .value("BFGS", ceres::LineSearchDirectionType::BFGS)
                        .value("LBFGS", ceres::LineSearchDirectionType::LBFGS)
                        .value("NONLINEAR_CONJUGATE_GRADIENT",
                               ceres::LineSearchDirectionType::
                                       NONLINEAR_CONJUGATE_GRADIENT)
                        .value("STEEPEST_DESCENT",
                               ceres::LineSearchDirectionType::
                                       STEEPEST_DESCENT);
    AddStringToEnumConstructor(lsdt);

    auto lsit = py::enum_<ceres::LineSearchInterpolationType>(
                        m, "LineSearchInterpolationType",
                        py::module_local())
                        .value("BISECTION",
                               ceres::LineSearchInterpolationType::BISECTION)
                        .value("CUBIC",
                               ceres::LineSearchInterpolationType::CUBIC)
                        .value("QUADRATIC",
                               ceres::LineSearchInterpolationType::QUADRATIC);
    AddStringToEnumConstructor(lsit);

    auto ncgt = py::enum_<ceres::NonlinearConjugateGradientType>(
                        m, "NonlinearConjugateGradientType",
                        py::module_local())
                        .value("FLETCHER_REEVES",
                               ceres::NonlinearConjugateGradientType::
                                       FLETCHER_REEVES)
                        .value("HESTENES_STIEFEL",
                               ceres::NonlinearConjugateGradientType::
                                       HESTENES_STIEFEL)
                        .value("POLAK_RIBIERE",
                               ceres::NonlinearConjugateGradientType::
                                       POLAK_RIBIERE);
    AddStringToEnumConstructor(ncgt);

    auto linsolt =
            py::enum_<ceres::LinearSolverType>(m, "LinearSolverType",
                                               py::module_local())
                    .value("DENSE_NORMAL_CHOLESKY",
                           ceres::LinearSolverType::DENSE_NORMAL_CHOLESKY)
                    .value("DENSE_QR", ceres::LinearSolverType::DENSE_QR)
                    .value("SPARSE_NORMAL_CHOLESKY",
                           ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY)
                    .value("DENSE_SCHUR",
                           ceres::LinearSolverType::DENSE_SCHUR)
                    .value("SPARSE_SCHUR",
                           ceres::LinearSolverType::SPARSE_SCHUR)
                    .value("ITERATIVE_SCHUR",
                           ceres::LinearSolverType::ITERATIVE_SCHUR)
                    .value("CGNR", ceres::LinearSolverType::CGNR);
    AddStringToEnumConstructor(linsolt);

    auto dogt =
            py::enum_<ceres::DoglegType>(m, "DoglegType", py::module_local())
                    .value("TRADITIONAL_DOGLEG",
                           ceres::DoglegType::TRADITIONAL_DOGLEG)
                    .value("SUBSPACE_DOGLEG",
                           ceres::DoglegType::SUBSPACE_DOGLEG);
    AddStringToEnumConstructor(dogt);

    auto trst = py::enum_<ceres::TrustRegionStrategyType>(
                        m, "TrustRegionStrategyType", py::module_local())
                        .value("LEVENBERG_MARQUARDT",
                               ceres::TrustRegionStrategyType::
                                       LEVENBERG_MARQUARDT)
                        .value("DOGLEG",
                               ceres::TrustRegionStrategyType::DOGLEG);
    AddStringToEnumConstructor(trst);

    auto prt = py::enum_<ceres::PreconditionerType>(
                       m, "PreconditionerType", py::module_local())
                       .value("IDENTITY",
                              ceres::PreconditionerType::IDENTITY)
                       .value("JACOBI", ceres::PreconditionerType::JACOBI)
                       .value("SCHUR_JACOBI",
                              ceres::PreconditionerType::SCHUR_JACOBI)
                       .value("CLUSTER_JACOBI",
                              ceres::PreconditionerType::CLUSTER_JACOBI)
                       .value("CLUSTER_TRIDIAGONAL",
                              ceres::PreconditionerType::CLUSTER_TRIDIAGONAL);
    AddStringToEnumConstructor(prt);

    auto vct = py::enum_<ceres::VisibilityClusteringType>(
                       m, "VisibilityClusteringType", py::module_local())
                       .value("CANONICAL_VIEWS",
                              ceres::VisibilityClusteringType::
                                      CANONICAL_VIEWS)
                       .value("SINGLE_LINKAGE",
                              ceres::VisibilityClusteringType::
                                      SINGLE_LINKAGE);
    AddStringToEnumConstructor(vct);

    auto dlalt = py::enum_<ceres::DenseLinearAlgebraLibraryType>(
                         m, "DenseLinearAlgebraLibraryType",
                         py::module_local())
                         .value("EIGEN",
                                ceres::DenseLinearAlgebraLibraryType::EIGEN)
                         .value("LAPACK",
                                ceres::DenseLinearAlgebraLibraryType::LAPACK)
                         .value("CUDA",
                                ceres::DenseLinearAlgebraLibraryType::CUDA);
    AddStringToEnumConstructor(dlalt);

    auto slalt = py::enum_<ceres::SparseLinearAlgebraLibraryType>(
                         m, "SparseLinearAlgebraLibraryType",
                         py::module_local())
                         .value("SUITE_SPARSE",
                                ceres::SparseLinearAlgebraLibraryType::
                                        SUITE_SPARSE)
                         .value("EIGEN_SPARSE",
                                ceres::SparseLinearAlgebraLibraryType::
                                        EIGEN_SPARSE)
                         .value("ACCELERATE_SPARSE",
                                ceres::SparseLinearAlgebraLibraryType::
                                        ACCELERATE_SPARSE)
                         .value("CUDA_SPARSE",
                                ceres::SparseLinearAlgebraLibraryType::
                                        CUDA_SPARSE)
                         .value("NO_SPARSE",
                                ceres::SparseLinearAlgebraLibraryType::
                                        NO_SPARSE);
    AddStringToEnumConstructor(slalt);

    auto logt =
            py::enum_<ceres::LoggingType>(m, "LoggingType",
                                          py::module_local())
                    .value("SILENT", ceres::LoggingType::SILENT)
                    .value("PER_MINIMIZER_ITERATION",
                           ceres::LoggingType::PER_MINIMIZER_ITERATION);
    AddStringToEnumConstructor(logt);

    auto cbrt =
            py::enum_<ceres::CallbackReturnType>(m, "CallbackReturnType",
                                                 py::module_local())
                    .value("SOLVER_CONTINUE",
                           ceres::CallbackReturnType::SOLVER_CONTINUE)
                    .value("SOLVER_ABORT",
                           ceres::CallbackReturnType::SOLVER_ABORT)
                    .value("SOLVER_TERMINATE_SUCCESSFULLY",
                           ceres::CallbackReturnType::
                                   SOLVER_TERMINATE_SUCCESSFULLY);
    AddStringToEnumConstructor(cbrt);

    auto dft = py::enum_<ceres::DumpFormatType>(m, "DumpFormatType",
                                                py::module_local())
                       .value("CONSOLE", ceres::DumpFormatType::CONSOLE)
                       .value("TEXTFILE", ceres::DumpFormatType::TEXTFILE);
    AddStringToEnumConstructor(dft);

    auto termt =
            py::enum_<ceres::TerminationType>(m, "TerminationType",
                                              py::module_local())
                    .value("CONVERGENCE",
                           ceres::TerminationType::CONVERGENCE)
                    .value("NO_CONVERGENCE",
                           ceres::TerminationType::NO_CONVERGENCE)
                    .value("FAILURE", ceres::TerminationType::FAILURE)
                    .value("USER_SUCCESS",
                           ceres::TerminationType::USER_SUCCESS)
                    .value("USER_FAILURE",
                           ceres::TerminationType::USER_FAILURE);
    AddStringToEnumConstructor(termt);
    // clang-format on
}

void BindCeresSolver(py::module& m) {
    using Options = ceres::Solver::Options;
    // Upstream pycolmap parity (src/pycolmap/estimators/ceres_bindings.cc):
    // the full ceres solver configuration surface. Note that the fork engine
    // merges a higher-level projection of this into
    // reconstruction.options.BundleAdjustmentOptions; this raw surface is the
    // upstream escape hatch.
    py::class_<Options> py_options(m, "SolverOptions", py::module_local(),
                                   "Configuration of the ceres solver.");
    py_options.def(py::init<>(), "Default constructor.")
            .def(py::init<const Options&>(), py::arg("options"),
                 "Copy constructor.")
            .def(
                    "IsValid",
                    [](const Options& self) {
                        std::string error;
                        const bool valid = self.IsValid(&error);
                        return std::make_pair(valid, error);
                    },
                    "Validate the option values; returns a (valid, "
                    "error_message) pair.")
            .def_readwrite("minimizer_type", &Options::minimizer_type,
                           "MinimizerType: (Default ``TRUST_REGION``) "
                           "Minimization algorithm to use.")
            .def_readwrite("line_search_direction_type",
                           &Options::line_search_direction_type,
                           "LineSearchDirectionType: (Default ``LBFGS``) "
                           "Direction for the line search step.")
            .def_readwrite("line_search_type", &Options::line_search_type,
                           "LineSearchType: (Default ``WOLFE``) Line search "
                           "termination criterion.")
            .def_readwrite("nonlinear_conjugate_gradient_type",
                           &Options::nonlinear_conjugate_gradient_type,
                           "NonlinearConjugateGradientType: (Default "
                           "``FLETCHER_REEVES``) Updates for the "
                           "nonlinear conjugate gradient method.")
            .def_readwrite("max_lbfgs_rank", &Options::max_lbfgs_rank,
                           "int: (Default ``20``) Rank of the L-BFGS "
                           "approximation.")
            .def_readwrite("use_approximate_eigenvalue_bfgs_scaling",
                           &Options::use_approximate_eigenvalue_bfgs_scaling,
                           "bool: (Default ``true``) Approximate eigenvalue "
                           "scaling for BFGS.")
            .def_readwrite("line_search_interpolation_type",
                           &Options::line_search_interpolation_type,
                           "LineSearchInterpolationType: (Default ``CUBIC``) "
                           "Interpolation for the line search minimum.")
            .def_readwrite("min_line_search_step_size",
                           &Options::min_line_search_step_size,
                           "double: (Default ``1e-9``) Smallest step size.")
            .def_readwrite("line_search_sufficient_function_decrease",
                           &Options::line_search_sufficient_function_decrease,
                           "double: (Default ``1e-4``) Armijo-Wolfe "
                           "sufficient decrease.")
            .def_readwrite("max_line_search_step_contraction",
                           &Options::max_line_search_step_contraction,
                           "double: (Default ``1e-3``) Maximum contraction.")
            .def_readwrite("min_line_search_step_contraction",
                           &Options::min_line_search_step_contraction,
                           "double: (Default ``0.6``) Minimum contraction.")
            .def_readwrite("max_num_line_search_step_size_iterations",
                           &Options::max_num_line_search_step_size_iterations,
                           "int: (Default ``20``) Maximum step size "
                           "iterations.")
            .def_readwrite("max_num_line_search_direction_restarts",
                           &Options::max_num_line_search_direction_restarts,
                           "int: (Default ``5``) Maximum restarts.")
            .def_readwrite("line_search_sufficient_curvature_decrease",
                           &Options::line_search_sufficient_curvature_decrease,
                           "double: (Default ``0.9``) Wolfe curvature "
                           "condition.")
            .def_readwrite("max_line_search_step_expansion",
                           &Options::max_line_search_step_expansion,
                           "double: (Default ``10.0``) Maximum expansion.")
            .def_readwrite("trust_region_strategy_type",
                           &Options::trust_region_strategy_type,
                           "TrustRegionStrategyType: (Default "
                           "``LEVENBERG_MARQUARDT``).")
            .def_readwrite("dogleg_type", &Options::dogleg_type,
                           "DoglegType: (Default ``TRADITIONAL_DOGLEG``).")
            .def_readwrite("use_nonmonotonic_steps",
                           &Options::use_nonmonotonic_steps,
                           "bool: (Default ``false``) Accept non-monotonic "
                           "steps.")
            .def_readwrite("max_consecutive_nonmonotonic_steps",
                           &Options::max_consecutive_nonmonotonic_steps,
                           "int: (Default ``5``) Maximum consecutive "
                           "non-monotonic steps.")
            .def_readwrite("max_num_iterations", &Options::max_num_iterations,
                           "int: (Default ``50``) Maximum number of solver "
                           "iterations.")
            .def_readwrite("max_solver_time_in_seconds",
                           &Options::max_solver_time_in_seconds,
                           "double: (Default ``1e9``) Wall-clock solver "
                           "budget in seconds.")
            .def_property(
                    "num_threads",
                    [](const Options& self) { return self.num_threads; },
                    [](Options& self, int n_threads) {
                        // Upstream parity: 0 or negative thread counts are
                        // resolved against the hardware concurrency.
                        self.num_threads = GetEffectiveNumThreads(n_threads);
                    },
                    "int: (Default ``-1``) Number of solver threads; "
                    "non-positive values resolve to the hardware "
                    "concurrency.")
            .def_readwrite("initial_trust_region_radius",
                           &Options::initial_trust_region_radius,
                           "double: (Default ``1e4``).")
            .def_readwrite("max_trust_region_radius",
                           &Options::max_trust_region_radius,
                           "double: (Default ``1e8``).")
            .def_readwrite("min_trust_region_radius",
                           &Options::min_trust_region_radius,
                           "double: (Default ``1e-32``).")
            .def_readwrite("min_relative_decrease",
                           &Options::min_relative_decrease,
                           "double: (Default ``1e-3``) Step acceptance "
                           "threshold.")
            .def_readwrite("min_lm_diagonal", &Options::min_lm_diagonal,
                           "double: (Default ``1e6``) Levenberg-Marquardt "
                           "lower bound.")
            .def_readwrite("max_lm_diagonal", &Options::max_lm_diagonal,
                           "double: (Default ``1e-6``) Levenberg-Marquardt "
                           "upper bound.")
            .def_readwrite("max_num_consecutive_invalid_steps",
                           &Options::max_num_consecutive_invalid_steps,
                           "int: (Default ``5``).")
            .def_readwrite("function_tolerance", &Options::function_tolerance,
                           "double: (Default ``1e-6``).")
            .def_readwrite("gradient_tolerance", &Options::gradient_tolerance,
                           "double: (Default ``1e-10``).")
            .def_readwrite("parameter_tolerance", &Options::parameter_tolerance,
                           "double: (Default ``1e-8``).")
            .def_readwrite("linear_solver_type", &Options::linear_solver_type,
                           "LinearSolverType: (Default ``SPARSE_NORMAL_"
                           "CHOLESKY``).")
            .def_readwrite("preconditioner_type", &Options::preconditioner_type,
                           "PreconditionerType: (Default ``JACOBI``).")
            .def_readwrite("visibility_clustering_type",
                           &Options::visibility_clustering_type,
                           "VisibilityClusteringType: (Default "
                           "``CANONICAL_VIEWS``).")
            .def_readwrite("dense_linear_algebra_library_type",
                           &Options::dense_linear_algebra_library_type,
                           "DenseLinearAlgebraLibraryType: (Default "
                           "``EIGEN``).")
            .def_readwrite("sparse_linear_algebra_library_type",
                           &Options::sparse_linear_algebra_library_type,
                           "SparseLinearAlgebraLibraryType: (Default "
                           "``EIGEN_SPARSE``).")
            .def_readwrite("use_explicit_schur_complement",
                           &Options::use_explicit_schur_complement,
                           "bool: (Default ``false``).")
            .def_readwrite("dynamic_sparsity", &Options::dynamic_sparsity,
                           "bool: (Default ``false``).")
            .def_readwrite("use_inner_iterations",
                           &Options::use_inner_iterations,
                           "bool: (Default ``false``).")
            .def_readwrite("inner_iteration_tolerance",
                           &Options::inner_iteration_tolerance,
                           "double: (Default ``1e-4``).")
            .def_readwrite("min_linear_solver_iterations",
                           &Options::min_linear_solver_iterations,
                           "int: (Default ``0``).")
            .def_readwrite("max_linear_solver_iterations",
                           &Options::max_linear_solver_iterations,
                           "int: (Default ``500``).")
            .def_readwrite("eta", &Options::eta, "double: (Default ``1e-2``).")
            .def_readwrite("jacobi_scaling", &Options::jacobi_scaling,
                           "bool: (Default ``true``).")
            .def_readwrite("logging_type", &Options::logging_type,
                           "LoggingType: (Default ``SILENT``).")
            .def_readwrite("minimizer_progress_to_stdout",
                           &Options::minimizer_progress_to_stdout,
                           "bool: (Default ``false``).")
            .def_readwrite("trust_region_problem_dump_directory",
                           &Options::trust_region_problem_dump_directory,
                           "str: (Default ``''``).")
            .def_readwrite("trust_region_problem_dump_format_type",
                           &Options::trust_region_problem_dump_format_type,
                           "DumpFormatType: (Default ``CONSOLE``).")
            .def_readwrite("check_gradients", &Options::check_gradients,
                           "bool: (Default ``false``).")
            .def_readwrite("gradient_check_relative_precision",
                           &Options::gradient_check_relative_precision,
                           "double: (Default ``1e-8``).")
            .def_readwrite(
                    "gradient_check_numeric_derivative_relative_step_size",
                    &Options::
                            gradient_check_numeric_derivative_relative_step_size,
                    "double: (Default ``1e-6``).")
            .def_readwrite("update_state_every_iteration",
                           &Options::update_state_every_iteration,
                           "bool: (Default ``false``).");

    using Summary = ceres::Solver::Summary;
    py::class_<Summary> py_summary(m, "SolverSummary", py::module_local(),
                                   "Result summary of a ceres solve.");
    py_summary.def(py::init<>(), "Default constructor.")
            .def(py::init<const Summary&>(), py::arg("summary"),
                 "Copy constructor.")
            .def("BriefReport", &Summary::BriefReport,
                 "A brief one-line text description of the solve.")
            .def("FullReport", &Summary::FullReport,
                 "A full multiline description of the solve.")
            .def("IsSolutionUsable", &Summary::IsSolutionUsable,
                 "Whether the solution is usable (solve succeeded or "
                 "converged).")
            .def_readwrite("minimizer_type", &Summary::minimizer_type,
                           "MinimizerType used.")
            .def_readwrite("termination_type", &Summary::termination_type,
                           "TerminationType of the solve.")
            .def_readwrite("message", &Summary::message,
                           "str: Human-readable termination reason.")
            .def_readwrite("initial_cost", &Summary::initial_cost,
                           "double: Cost at the start of the solve.")
            .def_readwrite("final_cost", &Summary::final_cost,
                           "double: Cost at the end of the solve.")
            .def_readwrite("fixed_cost", &Summary::fixed_cost,
                           "double: Constant cost part.")
            .def_readwrite("num_successful_steps",
                           &Summary::num_successful_steps,
                           "int: Number of successful steps.")
            .def_readwrite("num_unsuccessful_steps",
                           &Summary::num_unsuccessful_steps,
                           "int: Number of unsuccessful steps.")
            .def_readwrite("num_inner_iteration_steps",
                           &Summary::num_inner_iteration_steps,
                           "int: Number of inner iteration steps.")
            .def_readwrite("num_line_search_steps",
                           &Summary::num_line_search_steps,
                           "int: Number of line search steps.")
            .def_readwrite("preprocessor_time_in_seconds",
                           &Summary::preprocessor_time_in_seconds,
                           "double: Preprocessing time.")
            .def_readwrite("minimizer_time_in_seconds",
                           &Summary::minimizer_time_in_seconds,
                           "double: Minimization time.")
            .def_readwrite("postprocessor_time_in_seconds",
                           &Summary::postprocessor_time_in_seconds,
                           "double: Postprocessing time.")
            .def_readwrite("total_time_in_seconds",
                           &Summary::total_time_in_seconds,
                           "double: Total solve time.")
            .def_readwrite("linear_solver_time_in_seconds",
                           &Summary::linear_solver_time_in_seconds,
                           "double: Time spent in the linear solver.")
            .def_readwrite("num_linear_solves", &Summary::num_linear_solves,
                           "int: Number of linear solves.")
            .def_readwrite("residual_evaluation_time_in_seconds",
                           &Summary::residual_evaluation_time_in_seconds,
                           "double: Time spent evaluating residuals.")
            .def_readwrite("num_residual_evaluations",
                           &Summary::num_residual_evaluations,
                           "int: Number of residual evaluations.")
            .def_readwrite("jacobian_evaluation_time_in_seconds",
                           &Summary::jacobian_evaluation_time_in_seconds,
                           "double: Time spent evaluating jacobians.")
            .def_readwrite("num_jacobian_evaluations",
                           &Summary::num_jacobian_evaluations,
                           "int: Number of jacobian evaluations.")
            .def_readwrite("inner_iteration_time_in_seconds",
                           &Summary::inner_iteration_time_in_seconds,
                           "double: Time spent in inner iterations.")
            .def_readwrite(
                    "line_search_cost_evaluation_time_in_seconds",
                    &Summary::line_search_cost_evaluation_time_in_seconds,
                    "double: Line search residual evaluation time.")
            .def_readwrite(
                    "line_search_gradient_evaluation_time_in_seconds",
                    &Summary::line_search_gradient_evaluation_time_in_seconds,
                    "double: Line search gradient evaluation time.")
            .def_readwrite(
                    "line_search_polynomial_minimization_time_in_seconds",
                    &Summary::
                            line_search_polynomial_minimization_time_in_seconds,
                    "double: Line search polynomial minimization time.")
            .def_readwrite("line_search_total_time_in_seconds",
                           &Summary::line_search_total_time_in_seconds,
                           "double: Total line search time.")
            .def_readwrite("num_parameter_blocks",
                           &Summary::num_parameter_blocks,
                           "int: Number of parameter blocks.")
            .def_readwrite("num_parameters", &Summary::num_parameters,
                           "int: Number of parameters.")
            .def_readwrite("num_effective_parameters",
                           &Summary::num_effective_parameters,
                           "int: Number of effective (tangent-space) "
                           "parameters.")
            .def_readwrite("num_residual_blocks", &Summary::num_residual_blocks,
                           "int: Number of residual blocks.")
            .def_readwrite("num_residuals", &Summary::num_residuals,
                           "int: Number of residuals.")
            .def_readwrite("num_parameter_blocks_reduced",
                           &Summary::num_parameter_blocks_reduced,
                           "int: Number of parameter blocks after "
                           "reduction.")
            .def_readwrite("num_parameters_reduced",
                           &Summary::num_parameters_reduced,
                           "int: Number of parameters after reduction.")
            .def_readwrite("num_effective_parameters_reduced",
                           &Summary::num_effective_parameters_reduced,
                           "int: Number of effective parameters after "
                           "reduction.")
            .def_readwrite("num_residual_blocks_reduced",
                           &Summary::num_residual_blocks_reduced,
                           "int: Number of residual blocks after reduction.")
            .def_readwrite("num_residuals_reduced",
                           &Summary::num_residuals_reduced,
                           "int: Number of residuals after reduction.")
            .def_readwrite("is_constrained", &Summary::is_constrained,
                           "bool: Whether the problem has constraints.")
            .def_readwrite("num_threads_given", &Summary::num_threads_given,
                           "int: Requested number of threads.")
            .def_readwrite("num_threads_used", &Summary::num_threads_used,
                           "int: Actually used number of threads.")
            .def_readwrite("linear_solver_type_given",
                           &Summary::linear_solver_type_given,
                           "LinearSolverType requested.")
            .def_readwrite("linear_solver_type_used",
                           &Summary::linear_solver_type_used,
                           "LinearSolverType actually used.")
            .def_readwrite("schur_structure_given",
                           &Summary::schur_structure_given,
                           "str: Requested Schur structure.")
            .def_readwrite("schur_structure_used",
                           &Summary::schur_structure_used,
                           "str: Schur structure actually used.")
            .def_readwrite("inner_iterations_given",
                           &Summary::inner_iterations_given,
                           "bool: Whether inner iterations were requested.")
            .def_readwrite("inner_iterations_used",
                           &Summary::inner_iterations_used,
                           "bool: Whether inner iterations were used.")
            .def_readwrite("preconditioner_type_given",
                           &Summary::preconditioner_type_given,
                           "PreconditionerType requested.")
            .def_readwrite("preconditioner_type_used",
                           &Summary::preconditioner_type_used,
                           "PreconditionerType actually used.")
            .def_readwrite("visibility_clustering_type",
                           &Summary::visibility_clustering_type,
                           "VisibilityClusteringType used.")
            .def_readwrite("trust_region_strategy_type",
                           &Summary::trust_region_strategy_type,
                           "TrustRegionStrategyType used.")
            .def_readwrite("dogleg_type", &Summary::dogleg_type,
                           "DoglegType used.")
            .def_readwrite("dense_linear_algebra_library_type",
                           &Summary::dense_linear_algebra_library_type,
                           "DenseLinearAlgebraLibraryType used.")
            .def_readwrite("sparse_linear_algebra_library_type",
                           &Summary::sparse_linear_algebra_library_type,
                           "SparseLinearAlgebraLibraryType used.")
            .def_readwrite("line_search_direction_type",
                           &Summary::line_search_direction_type,
                           "LineSearchDirectionType used.")
            .def_readwrite("line_search_type", &Summary::line_search_type,
                           "LineSearchType used.")
            .def_readwrite("line_search_interpolation_type",
                           &Summary::line_search_interpolation_type,
                           "LineSearchInterpolationType used.")
            .def_readwrite("nonlinear_conjugate_gradient_type",
                           &Summary::nonlinear_conjugate_gradient_type,
                           "NonlinearConjugateGradientType used.")
            .def_readwrite("max_lbfgs_rank", &Summary::max_lbfgs_rank,
                           "int: L-BFGS rank used.");

    using IterSummary = ceres::IterationSummary;
    py::class_<IterSummary> py_iter_summary(
            m, "IterationSummary", py::module_local(),
            "Per-iteration solver progress report.");
    py_iter_summary.def(py::init<>(), "Default constructor.")
            .def(py::init<const IterSummary&>(), py::arg("summary"),
                 "Copy constructor.")
            .def_readonly("iteration", &IterSummary::iteration,
                          "int: Current iteration number.")
            .def_readonly("step_is_valid", &IterSummary::step_is_valid,
                          "bool: Whether the step was valid.")
            .def_readonly("step_is_nonmonotonic",
                          &IterSummary::step_is_nonmonotonic,
                          "bool: Whether the step was non-monotonic.")
            .def_readonly("step_is_successful",
                          &IterSummary::step_is_successful,
                          "bool: Whether the step was accepted.")
            .def_readonly("cost", &IterSummary::cost,
                          "double: Cost at this iteration.")
            .def_readonly("cost_change", &IterSummary::cost_change,
                          "double: Cost change relative to the previous "
                          "iteration.")
            .def_readonly("gradient_max_norm", &IterSummary::gradient_max_norm,
                          "double: Max norm of the gradient.")
            .def_readonly("gradient_norm", &IterSummary::gradient_norm,
                          "double: Norm of the gradient.")
            .def_readonly("step_norm", &IterSummary::step_norm,
                          "double: Norm of the parameter update.")
            .def_readonly("relative_decrease", &IterSummary::relative_decrease,
                          "double: Relative cost decrease.")
            .def_readonly("trust_region_radius",
                          &IterSummary::trust_region_radius,
                          "double: Trust region radius after the step.")
            .def_readonly("eta", &IterSummary::eta, "double: Eta for the step.")
            .def_readonly("step_size", &IterSummary::step_size,
                          "double: Line search step size.")
            .def_readonly("line_search_function_evaluations",
                          &IterSummary::line_search_function_evaluations,
                          "int: Function evaluations in the line search.")
            .def_readonly("line_search_gradient_evaluations",
                          &IterSummary::line_search_gradient_evaluations,
                          "int: Gradient evaluations in the line search.")
            .def_readonly("line_search_iterations",
                          &IterSummary::line_search_iterations,
                          "int: Line search iterations.")
            .def_readonly("linear_solver_iterations",
                          &IterSummary::linear_solver_iterations,
                          "int: Linear solver iterations.")
            .def_readonly("iteration_time_in_seconds",
                          &IterSummary::iteration_time_in_seconds,
                          "double: Time of this iteration.")
            .def_readonly("step_solver_time_in_seconds",
                          &IterSummary::step_solver_time_in_seconds,
                          "double: Time to solve the step.")
            .def_readonly("cumulative_time_in_seconds",
                          &IterSummary::cumulative_time_in_seconds,
                          "double: Cumulative solve time.");
}

}  // namespace

void pybind_ceres(py::module& m) {
    py::module m_pyceres = m.def_submodule(
            "pyceres",
            "Embedded ceres-solver binding surface (upstream pycolmap parity; "
            "branched from pyceres core types/solver).");
    BindCeresTypes(m_pyceres);
    BindCeresSolver(m_pyceres);
    // Upstream relies on an optional pyceres install to register this base
    // class, which leaves the cost-function factories uncallable when it is
    // absent. Register it here so the factories in the cost_functions
    // submodule return usable objects.
    py::class_<ceres::CostFunction>(
            m_pyceres, "CostFunction", py::module_local(),
            "Base class of all ceres cost functions (residual + jacobian "
            "evaluation).")
            .def("num_residuals", &ceres::CostFunction::num_residuals,
                 "Number of residuals of the cost function.")
            .def(
                    "parameter_block_sizes",
                    [](const ceres::CostFunction& self) {
                        return self.parameter_block_sizes();
                    },
                    "Sizes of the parameter blocks the cost function "
                    "evaluates.");
}

}  // namespace estimators
}  // namespace reconstruction
}  // namespace cloudViewer
