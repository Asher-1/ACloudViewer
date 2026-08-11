// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "data_root_util.hpp"

#include <FileSystem.h>

#include <cstdlib>
#include <filesystem>

namespace aicore {

std::string locate_data_root() {
    if (const char* env_p = std::getenv("CLOUDVIEWER_DATA_ROOT")) {
        if (env_p[0] != '\0') {
            return std::string(env_p);
        }
    }
    return cloudViewer::utility::filesystem::GetHomeDirectory() +
           "/cloudViewer_data";
}

std::string extract_model_dir(const char* sub_dir) {
    return (std::filesystem::path(locate_data_root()) / "extract" / sub_dir)
            .lexically_normal()
            .string();
}

}  // namespace aicore
