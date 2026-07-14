#include "pose_export.hpp"
#include <cstdio>

namespace da {

bool write_pose_json(const std::string& path, const std::array<float,12>& ext, const std::array<float,9>& intr){
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fprintf(f, "{\n  \"extrinsics\": [\n");
    for (int r = 0; r < 3; ++r){
        std::fprintf(f, "    [%.8g, %.8g, %.8g, %.8g]%s\n",
                     ext[r*4+0], ext[r*4+1], ext[r*4+2], ext[r*4+3], r < 2 ? "," : "");
    }
    std::fprintf(f, "  ],\n  \"intrinsics\": [\n");
    for (int r = 0; r < 3; ++r){
        std::fprintf(f, "    [%.8g, %.8g, %.8g]%s\n",
                     intr[r*3+0], intr[r*3+1], intr[r*3+2], r < 2 ? "," : "");
    }
    std::fprintf(f, "  ]\n}\n");
    std::fclose(f);
    return true;
}

} // namespace da
