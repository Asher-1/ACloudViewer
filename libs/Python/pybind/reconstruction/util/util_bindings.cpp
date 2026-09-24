// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/reconstruction/util/util_bindings.h"

#include <glog/logging.h>

#include "util/cancellation.h"
#include "util/logging.h"
#include "util/timer.h"
#include "util/version.h"

namespace cloudViewer {
namespace reconstruction {
namespace util {

namespace {

// Upstream pycolmap parity (src/pycolmap/util/logging.cc): dummy class
// carrying the glog severity levels, with the glog FLAGS exposed as static
// attributes and the log-message helpers as static methods.
struct PyLogging {
    enum class LogSeverity {
        GLOG_INFO = google::GLOG_INFO,
        GLOG_WARNING = google::GLOG_WARNING,
        GLOG_ERROR = google::GLOG_ERROR,
        GLOG_FATAL = google::GLOG_FATAL,
    };
};  // dummy class

// Upstream parity: log messages carry the Python call site instead of the
// binding source location.
std::pair<std::string, int> GetPythonCallFrame() {
    const auto frame = py::module_::import("sys").attr("_getframe")(0);
    const std::string file = py::str(frame.attr("f_code").attr("co_filename"));
    const std::string function = py::str(frame.attr("f_code").attr("co_name"));
    const int line = py::int_(frame.attr("f_lineno"));
    return std::make_pair(file + ":" + function, line);
}

}  // namespace

void pybind_util(py::module& m) {
    py::module m_submodule =
            m.def_submodule("util",
                            "Utility bindings (Timer, logging, "
                            "cancellation; upstream pycolmap/util "
                            "parity).");

    // Timer (upstream pycolmap/util/timer.cc).
    py::class_<colmap::Timer> py_timer(m_submodule, "Timer",
                                       "Wall-clock stopwatch.");
    py_timer.def(py::init<>())
            .def("start", &colmap::Timer::Start, "Start the timer.")
            .def("restart", &colmap::Timer::Restart,
                 "Restart the timer from zero.")
            .def("pause", &colmap::Timer::Pause, "Pause the running timer.")
            .def("resume", &colmap::Timer::Resume, "Resume a paused timer.")
            .def("reset", &colmap::Timer::Reset,
                 "Reset the timer to zero without stopping it.")
            .def("elapsed_micro_seconds", &colmap::Timer::ElapsedMicroSeconds,
                 "Elapsed time in microseconds.")
            .def("elapsed_seconds", &colmap::Timer::ElapsedSeconds,
                 "Elapsed time in seconds.")
            .def("elapsed_minutes", &colmap::Timer::ElapsedMinutes,
                 "Elapsed time in minutes.")
            .def("elapsed_hours", &colmap::Timer::ElapsedHours,
                 "Elapsed time in hours.")
            .def("print_seconds", &colmap::Timer::PrintSeconds,
                 "Log the elapsed time in seconds.")
            .def("print_minutes", &colmap::Timer::PrintMinutes,
                 "Log the elapsed time in minutes.")
            .def("print_hours", &colmap::Timer::PrintHours,
                 "Log the elapsed time in hours.");

    // Logging (upstream pycolmap/util/logging.cc).
    py::class_<PyLogging> py_logging(m_submodule, "logging",
                                     "glog control surface.",
                                     py::module_local());
    py::enum_<PyLogging::LogSeverity> py_severity(py_logging, "Level");
    py_severity.value("INFO", PyLogging::LogSeverity::GLOG_INFO)
            .value("WARNING", PyLogging::LogSeverity::GLOG_WARNING)
            .value("ERROR", PyLogging::LogSeverity::GLOG_ERROR)
            .value("FATAL", PyLogging::LogSeverity::GLOG_FATAL)
            .export_values();

    py_logging.def_readwrite_static("minloglevel", &FLAGS_minloglevel)
            .def_readwrite_static("stderrthreshold", &FLAGS_stderrthreshold)
            .def_readwrite_static("log_dir", &FLAGS_log_dir)
            .def_readwrite_static("logtostderr", &FLAGS_logtostderr)
            .def_readwrite_static("alsologtostderr", &FLAGS_alsologtostderr)
            .def_readwrite_static("verbose_level", &FLAGS_v)
            .def_static(
                    "set_log_destination",
                    [](const PyLogging::LogSeverity severity,
                       const std::string& path) {
                        google::SetLogDestination(
                                static_cast<google::LogSeverity>(severity),
                                path.c_str());
                    },
                    "level"_a, "path"_a,
                    "Set the base path for log files at the given severity.")
            .def_static(
                    "verbose",
                    [](const int level, const std::string& message) {
                        if (VLOG_IS_ON(level)) {
                            const auto frame = GetPythonCallFrame();
                            google::LogMessage(frame.first.c_str(),
                                               frame.second)
                                            .stream()
                                    << message;
                        }
                    },
                    "level"_a, "message"_a,
                    "Log a verbose message if the verbosity level is high "
                    "enough.")
            .def_static(
                    "info",
                    [](const std::string& message) {
                        const auto frame = GetPythonCallFrame();
                        google::LogMessage(frame.first.c_str(), frame.second)
                                        .stream()
                                << message;
                    },
                    "message"_a, "Log an informational message.")
            .def_static(
                    "warning",
                    [](const std::string& message) {
                        const auto frame = GetPythonCallFrame();
                        google::LogMessage(frame.first.c_str(), frame.second)
                                        .stream()
                                << message;
                    },
                    "message"_a, "Log a warning message.")
            .def_static(
                    "error",
                    [](const std::string& message) {
                        const auto frame = GetPythonCallFrame();
                        google::LogMessage(frame.first.c_str(), frame.second)
                                        .stream()
                                << message;
                    },
                    "message"_a, "Log an error message.")
            .def_static(
                    "fatal",
                    [](const std::string& message) {
                        const auto frame = GetPythonCallFrame();
                        google::LogMessage(frame.first.c_str(), frame.second)
                                        .stream()
                                << message;
                    },
                    "message"_a,
                    "Log a fatal error and terminate the program.");

    // Cancellation (upstream pycolmap/util/cancellation.cc).
    py::class_<colmap::CancellationToken,
               std::shared_ptr<colmap::CancellationToken>>(
            m_submodule, "CancellationToken",
            "Thread-safe, single-use cancellation token for cooperative "
            "cancellation of long-running operations.")
            .def(py::init<>())
            .def("cancel", &colmap::CancellationToken::Cancel,
                 "Request cancellation; the token stays cancelled.")
            .def_property_readonly("is_cancelled",
                                   &colmap::CancellationToken::IsCancelled,
                                   "Whether cancellation was requested.");

    // Version (upstream exposes the library version on the module).
    m_submodule
            .def("get_version", &colmap::GetVersionInfo,
                 "The engine version string (COLMAP lineage).")
            .def("get_build_info", &colmap::GetBuildInfo,
                 "The engine build configuration string.");
}

}  // namespace util
}  // namespace reconstruction
}  // namespace cloudViewer
