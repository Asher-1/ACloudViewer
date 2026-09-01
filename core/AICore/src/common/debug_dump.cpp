// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Implementation of the sanctioned debug dump-path accessors. Whitelisted in
// tests/check_no_env_getenv.sh; see the header comment for the policy.

#include "common/debug_dump.hpp"

#include <cstdlib>

namespace aicore {
namespace debug {

const char* savpe_dump_path() { return std::getenv("AICORE_SAVPE_DUMP"); }

const char* savpe_mask_dump_path() {
    return std::getenv("AICORE_SAVPE_DUMP_MASK");
}

bool savpe_debug_enabled() {
    return std::getenv("AICORE_SAVPE_DEBUG") != nullptr;
}

}  // namespace debug
}  // namespace aicore
