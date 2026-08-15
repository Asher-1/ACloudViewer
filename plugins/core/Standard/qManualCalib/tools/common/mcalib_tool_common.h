// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <CVLog.h>

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace mcalib {
namespace tools {

inline void collectFiles(const fs::path& dir,
                         const std::string& ext,
                         std::vector<fs::path>& out) {
    if (!fs::exists(dir)) return;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() == ext) {
            out.push_back(entry.path());
        }
    }
    std::sort(out.begin(), out.end());
}

inline void collectBagFiles(const fs::path& dir, std::vector<fs::path>& out) {
    collectFiles(dir, ".bag", out);
}

inline void collectConfigDirs(const fs::path& root,
                              const std::string& cfg_name,
                              std::vector<fs::path>& out) {
    if (!fs::exists(root)) return;
    if (fs::exists(root / cfg_name)) {
        out.push_back(root);
        return;
    }
    for (const auto& entry : fs::directory_iterator(root)) {
        if (!entry.is_directory()) continue;
        if (fs::exists(entry.path() / cfg_name)) {
            out.push_back(entry.path());
        }
    }
    std::sort(out.begin(), out.end());
}

inline void collectImages(const fs::path& dir, std::vector<fs::path>& out) {
    collectFiles(dir, ".jpg", out);
    if (out.empty()) collectFiles(dir, ".png", out);
    if (out.empty()) collectFiles(dir, ".jpeg", out);
}

}  // namespace tools
}  // namespace mcalib
