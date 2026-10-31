// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Upstream COLMAP dbb41680 parity test helpers. The legacy TEST_NAME guard
// was removed with the googletest migration (decision D6): upstream
// test files include this header without defining TEST_NAME and keep
// compiling unchanged.

#include <gtest/gtest.h>

#include <filesystem>
#include <mutex>
#include <set>
#include <sstream>
#include <string>

#include "util/logging.h"

namespace colmap {

// Creates (and cleans up from previous runs) a per-test temporary directory
// rooted at the system temp path. Mirrors the upstream util/testing.cc
// implementation so ported upstream tests can call CreateTestDir() directly.
inline std::filesystem::path CreateTestDir() {
    const testing::TestInfo* test_info = THROW_CHECK_NOTNULL(
            testing::UnitTest::GetInstance()->current_test_info());
    std::ostringstream test_name_stream;
    test_name_stream << test_info->test_suite_name() << "."
                     << test_info->name();
    const std::string test_name = test_name_stream.str();

    const std::filesystem::path test_dir =
            std::filesystem::temp_directory_path() / "colmap_test_data" /
            test_name;
    LOG(INFO) << "Creating test directory: " << test_dir;

    // Create directory once. Cleanup artifacts from previous test runs.
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static std::set<std::string> existing_test_names;
    if (existing_test_names.count(test_name) == 0) {
        if (std::filesystem::is_directory(test_dir)) {
            std::filesystem::remove_all(test_dir);
        }
        std::filesystem::create_directories(test_dir);
    }
    existing_test_names.insert(test_name);

    return test_dir;
}

}  // namespace colmap
