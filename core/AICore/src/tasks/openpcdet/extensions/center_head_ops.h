#pragma once

#include "../model.h"

#include <cstdint>
#include <string>
#include <vector>

namespace openpcdet::ext {

// Layout: row-major HWC by default (location = y*width + x, channels adjacent).
// With `whc_layout` set, data is GGML WHC (x + width*y + width*height*c), which
// makes the decode loops walk location contiguously (stride 1 instead of C).
// `heatmap` contains logits. Regression fields are shared across classes within
// a head, matching OpenPCDet CenterHead.
struct center_head_view {
    const float * heatmap = nullptr;
    const float * center = nullptr;
    const float * center_z = nullptr;
    const float * dimensions = nullptr;
    const float * rotation = nullptr; // [cos, sin]
    const float * velocity = nullptr; // optional [vx, vy]
    const float * iou = nullptr;      // optional raw IoU logits, one per row
    // Optional sparse coordinates [active_count, 3] = batch,y,x. When set,
    // decoding uses these rows instead of a dense height*width grid.
    const int32_t * coordinates = nullptr;
    int active_count = 0;
    int height = 0;
    int width = 0;
    int class_count = 0;
    bool heatmap_is_probability = false;
    bool dimensions_are_linear = false;
    bool whc_layout = false;
    std::vector<int32_t> class_mapping; // zero-based global class ids
};

struct center_decode_params {
    std::vector<float> point_cloud_range;
    std::vector<float> voxel_size;
    int feature_map_stride = 1;
    int max_objects = 500;
    float score_threshold = 0.1f;
    std::vector<float> post_center_limit_range;
    float nms_threshold = 0.2f;
    int nms_pre_maxsize = 1000;
    int nms_post_maxsize = 83;
    // VoxelNeXt Waymo IoU branch. Arrays are indexed by zero-based global
    // class id and must all have the same length when `head.iou` is set.
    std::vector<float> iou_rectifier;
    std::vector<float> class_nms_thresholds;
    std::vector<int32_t> class_nms_pre_maxsizes;
    std::vector<int32_t> class_nms_post_maxsizes;
};

bool decode_center_head(const center_head_view & head,
                        const center_decode_params & params,
                        std::vector<detection> & output,
                        std::string & error);

} // namespace openpcdet::ext
