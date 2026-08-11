// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdlib>
#include <string>

#include "common/test_macros.hpp"
#include "data_root_util.hpp"

static int failures = 0;

int main() {
    const std::string home_root = aicore::locate_data_root();
    AICORE_CHECK(!home_root.empty());

    const std::string da3 = aicore::extract_model_dir("da3_models");
    AICORE_CHECK(da3.find("da3_models") != std::string::npos);
    AICORE_CHECK(da3.find(home_root) != std::string::npos);

    const std::string gauss = aicore::extract_model_dir("freesplatter_models");
    AICORE_CHECK(gauss.find("freesplatter_models") != std::string::npos);

#ifdef _WIN32
    _putenv_s("CLOUDVIEWER_DATA_ROOT", "/tmp/aicore_test_data_root");
#else
    setenv("CLOUDVIEWER_DATA_ROOT", "/tmp/aicore_test_data_root", 1);
#endif
    AICORE_CHECK(aicore::locate_data_root() == "/tmp/aicore_test_data_root");
    AICORE_CHECK(aicore::extract_model_dir("da3_models") ==
                 "/tmp/aicore_test_data_root/extract/da3_models");

    std::fprintf(stderr, "data_root ok (default=%s)\n", home_root.c_str());
    return failures == 0 ? 0 : 1;
}
