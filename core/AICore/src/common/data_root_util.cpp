// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "common/data_root_util.hpp"

#include <FileSystem.h>

#include <filesystem>

namespace aicore {

std::string locate_data_root() {
    return cloudViewer::utility::filesystem::LocateDataRoot();
}

std::string extract_model_dir(const char* sub_dir) {
    return (std::filesystem::u8path(locate_data_root()) / "extract" /
            std::filesystem::u8path(sub_dir))
            .lexically_normal()
            .generic_u8string();
}

}  // namespace aicore
