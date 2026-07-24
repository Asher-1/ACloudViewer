// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <CVLog.h>

#include "util/logging.h"

//! Verbose reconstruction / DA3 pipeline diagnostics (debug builds only).
#define RECON_LOG_DEBUG(...) CVLog::PrintDebug(__VA_ARGS__)

//! User-visible status (also emitted via glog for the reconstruction Log
//! widget).
#define RECON_LOG_INFO(...)                      \
    do {                                         \
        LOG(INFO) << StringPrintf(__VA_ARGS__);  \
        CVLog::Print(StringPrintf(__VA_ARGS__)); \
    } while (0)

//! User-visible warnings (quality gates, fallbacks, skipped stages).
#define RECON_LOG_WARN(...) CVLog::Warning(__VA_ARGS__)

//! Hard failures that stop or skip a stage.
#define RECON_LOG_ERROR(...) CVLog::Error(__VA_ARGS__)
