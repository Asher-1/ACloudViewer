// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "core/system/Config.hpp"

#include <mutex>

std::mutex gLogMutex;

namespace sibr {

LogExit::LogExit(void) : lock(gLogMutex) {}

void LogExit::operator<<=(const std::ostream& /*stream*/) {
    // do exit, only profit a the rules of 'operator precedence'
    // to be executed after operator << when writing to the stream
    // itself.
    // So that this class is evaluated after writing the output and
    // it will exit (see dtor)
    // exit(EXIT_FAILURE);
    throw std::runtime_error("See log for message errors");
}

DebugScopeProfiler::~DebugScopeProfiler(void) {
    double t = double(clock() - _t0) / CLOCKS_PER_SEC;
    SIBR_LOG << "[PROFILER] Scope '" << _name << "' completed in " << t
             << "sec." << std::endl;
}

DebugScopeProfiler::DebugScopeProfiler(const std::string& name) : _name(name) {
    _t0 = clock();
}

}  // namespace sibr