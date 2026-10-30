#pragma once

#include "../model.h"
#include "point_mlp.h"

#include <vector>

namespace openpcdet {

struct pointrcnn final : model {
    std::vector<std::vector<point_mlp_weights>> backbone_sa;
    std::vector<point_mlp_weights> backbone_fp;
    point_mlp_weights point_cls;
    point_mlp_weights point_box;
    point_mlp_weights roi_xyz_up;
    point_mlp_weights roi_merge;
    std::vector<point_mlp_weights> roi_sa;
    point_mlp_weights roi_cls;
    point_mlp_weights roi_reg;

    bool load(const std::string & path, const std::string & device, int threads) override;
    bool forward(const float * points, int32_t count, int32_t dimension,
                 std::vector<detection> & detections,
                 const tap_sink & sink = {}) override;
};

} // namespace openpcdet
