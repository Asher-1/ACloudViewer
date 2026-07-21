#pragma once
#include "model_loader.hpp"
#include "backend.hpp"
#include <array>
#include <vector>
namespace aicore {
namespace depth {
class CamPose {
public:
    CamPose(ModelLoader& ml, Backend& be) : ml_(ml), be_(be) {}
    // cam_token: 1536 floats (the layer-11 camera token). Outputs row-major.
    bool pose(const std::vector<float>& cam_token, int H, int W,
              std::array<float,9>& pose_enc,
              std::array<float,12>& extrinsics,   // 3x4 row-major
              std::array<float,9>& intrinsics);   // 3x3 row-major
private:
    ModelLoader& ml_; Backend& be_;
};
} // namespace depth
} // namespace aicore
