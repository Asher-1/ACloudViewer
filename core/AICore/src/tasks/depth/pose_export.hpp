#pragma once
#include <string>
#include <array>
namespace aicore {
namespace depth {
// Write {"extrinsics":[[..4],[..4],[..4]], "intrinsics":[[..3],[..3],[..3]]} as JSON.
bool write_pose_json(const std::string& path, const std::array<float,12>& ext, const std::array<float,9>& intr);
} // namespace depth
} // namespace aicore
