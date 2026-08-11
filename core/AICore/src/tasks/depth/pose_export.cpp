// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pose_export.hpp"

#include <cstdio>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif  // !_WIN32

namespace aicore {
namespace depth {

//! Open an output file with safe (non-world-writable) permissions.
static std::FILE* open_output_file(const std::string& path) {
#ifdef _WIN32
    return std::fopen(path.c_str(), "wb");
#else
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return nullptr;
    return ::fdopen(fd, "wb");
#endif  // _WIN32
}

bool write_pose_json(const std::string& path,
                     const std::array<float, 12>& ext,
                     const std::array<float, 9>& intr) {
    std::FILE* f = open_output_file(path);
    if (!f) return false;
    std::fprintf(f, "{\n  \"extrinsics\": [\n");
    for (int r = 0; r < 3; ++r) {
        std::fprintf(f, "    [%.8g, %.8g, %.8g, %.8g]%s\n", ext[r * 4 + 0],
                     ext[r * 4 + 1], ext[r * 4 + 2], ext[r * 4 + 3],
                     r < 2 ? "," : "");
    }
    std::fprintf(f, "  ],\n  \"intrinsics\": [\n");
    for (int r = 0; r < 3; ++r) {
        std::fprintf(f, "    [%.8g, %.8g, %.8g]%s\n", intr[r * 3 + 0],
                     intr[r * 3 + 1], intr[r * 3 + 2], r < 2 ? "," : "");
    }
    std::fprintf(f, "  ]\n}\n");
    std::fclose(f);
    return true;
}

}  // namespace depth
}  // namespace aicore
