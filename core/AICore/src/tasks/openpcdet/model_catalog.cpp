// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstring>
#include <string>
#include <vector>

#include "aicore/openpcdet_capi.h"

namespace {

static constexpr const char* kDownloadBase =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "OpenPCDet-GGUF/";

// Published OpenPCDet-GGUF release models (detector checkpoints x 3
// quantizations), aligned with the upstream OpenPCDet-GGML release inventory.
struct ModelVariant {
    const char* file_stem;
    const char* display_name;
    const char* family_note;
};

static constexpr ModelVariant kVariants[] = {
        {"kitti_pointpillar",
         "OpenPCDet KITTI PointPillars",
         "PointPillar / AnchorHeadSingle, KITTI (3 classes, 4-dim points)"},
        {"nuscenes_pointpillar_multihead",
         "OpenPCDet nuScenes PointPillars Multihead",
         "PointPillar / AnchorHeadMulti, nuScenes (10 classes, 5-dim points)"},
        {"nuscenes_dynpillar_centerpoint",
         "OpenPCDet nuScenes CenterPoint (Dynamic Pillar)",
         "PointPillar / CenterHead, nuScenes (10 classes, 5-dim points)"},
        {"nuscenes_centerpoint_voxel0075",
         "OpenPCDet nuScenes CenterPoint (Voxel 0.075m)",
         "SECOND voxel / CenterHead, nuScenes (10 classes, 5-dim points)"},
        {"second_7862",
         "OpenPCDet KITTI SECOND",
         "SECOND / VoxelBackBone8x sparse, KITTI (3 classes, 4-dim points)"},
        {"kitti_second_iou",
         "OpenPCDet KITTI SECOND-IoU",
         "SECOND + SECONDHead RoI refinement, KITTI (3 classes, 4-dim points)"},
        {"nuscenes_second_multihead",
         "OpenPCDet nuScenes SECOND Multihead",
         "SECOND residual / AnchorHeadMulti, nuScenes (10 classes, 5-dim)"},
        {"kitti_pointrcnn",
         "OpenPCDet KITTI PointRCNN",
         "PointRCNN / PointNet++ RPN + RCNN, KITTI (3 classes, 4-dim points)"},
        {"kitti_pointrcnn_iou",
         "OpenPCDet KITTI PointRCNN-IoU",
         "PointRCNN + IoU head refinement, KITTI (3 classes, 4-dim points)"},
        {"kitti_pvrcnn",
         "OpenPCDet KITTI PV-RCNN",
         "PV-RCNN / voxel sparse + point branch + RCNN, KITTI (3 classes)"},
        {"kitti_voxelrcnn_car",
         "OpenPCDet KITTI Voxel-RCNN (Car)",
         "Voxel-RCNN / sparse backbone + voxel RoI, KITTI (car class)"},
        {"kitti_voxelrcnn_focal_multimodal",
         "OpenPCDet KITTI Voxel-RCNN Focal Multimodal",
         "Voxel-RCNN focal sparse multimodal fusion, KITTI (3 classes)"},
        {"kitti_parta2_anchor",
         "OpenPCDet KITTI Part-A2",
         "Part-A2 / sparse backbone + part-aware refine, KITTI (3 classes)"},
        {"kitti_caddn",
         "OpenPCDet KITTI CaDDN",
         "CaDDN / gridded LiDAR feature + 2D decode, KITTI (3 classes)"},
        {"nuscenes_transfusion_lidar",
         "OpenPCDet nuScenes TransFusion (LiDAR)",
         "TransFusion / sparse transformer decoder, nuScenes (5-dim points)"},
        {"nuscenes_voxelnext",
         "OpenPCDet nuScenes VoxelNeXt",
         "VoxelNeXt / voxel-centric sparse head, nuScenes (5-dim points)"},
        {"nuscenes_voxelnext_doubleflip",
         "OpenPCDet nuScenes VoxelNeXt Double-Flip",
         "VoxelNeXt double-flip TTA inference, nuScenes (5-dim points)"},
        {"nuscenes_bevfusion",
         "OpenPCDet nuScenes BEVFusion (LiDAR)",
         "BEVFusion / multi-branch pillar fusion, nuScenes (5-dim points)"},
};

static constexpr const char* kQuantSuffixes[] = {"f32", "f16", "q8_0"};

static constexpr const char* kQuantNotes[] = {
        "F32 \xe2\x80\x94 full precision reference",
        "F16 \xe2\x80\x94 half precision (recommended)",
        "Q8_0 \xe2\x80\x94 dense convs 8-bit; sparse backbone stays F32",
};

static constexpr int kVariantCount =
        static_cast<int>(sizeof(kVariants) / sizeof(kVariants[0]));
static constexpr int kQuantCount =
        static_cast<int>(sizeof(kQuantSuffixes) / sizeof(kQuantSuffixes[0]));

// MSVC names the POSIX helper "_strdup"; keep a portable wrapper so the
// catalog builds warning-clean on all three platforms.
static char* dupString(const char* s) {
#ifdef _MSC_VER
    return _strdup(s);
#else
    return strdup(s);
#endif
}

struct ModelRow {
    const char* filename;
    const char* download_url;
    const char* display_name;
    const char* quant_note;
    const char* family_note;
    const char* license_note;
};

static std::vector<ModelRow> buildModels() {
    std::vector<ModelRow> rows;
    rows.reserve(static_cast<size_t>(kVariantCount) * kQuantCount);
    for (int vi = 0; vi < kVariantCount; ++vi) {
        for (int qi = 0; qi < kQuantCount; ++qi) {
            std::string filename = std::string(kVariants[vi].file_stem) + "." +
                                   kQuantSuffixes[qi] + ".gguf";
            std::string url = std::string(kDownloadBase) + filename;
            std::string display = std::string(kVariants[vi].display_name) +
                                  " \xe2\x80\x94 " + kQuantNotes[qi];
            rows.push_back({dupString(filename.c_str()), dupString(url.c_str()),
                            dupString(display.c_str()),
                            dupString(kQuantNotes[qi]),
                            dupString(kVariants[vi].family_note),
                            "Apache-2.0 (OpenPCDet)"});
        }
    }
    return rows;
}

static const std::vector<ModelRow> kModels = buildModels();

static aicore_openpcdet_model_entry toEntry(const ModelRow& row) {
    return {row.filename, row.download_url, row.display_name,
            row.quant_note, row.family_note, row.license_note};
}

}  // namespace

AICORE_CAPI int aicore_openpcdet_model_count(void) {
    return static_cast<int>(kModels.size());
}

AICORE_CAPI int aicore_openpcdet_model_default_index(void) {
    // The first F16 row (KITTI PointPillars F16): the declared default carries
    // the same "(recommended)" marker as its quant note, so declaration and
    // user-facing label cannot drift.
    for (size_t i = 0; i < kModels.size(); ++i) {
        if (kModels[i].quant_note &&
            std::strstr(kModels[i].quant_note, "(recommended)")) {
            return static_cast<int>(i);
        }
    }
    return 0;
}

AICORE_CAPI const aicore_openpcdet_model_entry* aicore_openpcdet_model_at(
        int index) {
    static thread_local aicore_openpcdet_model_entry entry{};
    if (index < 0 || index >= static_cast<int>(kModels.size())) return nullptr;
    entry = toEntry(kModels[static_cast<size_t>(index)]);
    return &entry;
}

AICORE_CAPI const aicore_openpcdet_model_entry*
aicore_openpcdet_model_by_filename(const char* filename) {
    if (filename == nullptr || filename[0] == '\0') return nullptr;
    for (size_t i = 0; i < kModels.size(); ++i) {
        if (std::strcmp(kModels[i].filename, filename) == 0) {
            return aicore_openpcdet_model_at(static_cast<int>(i));
        }
    }
    return nullptr;
}

AICORE_CAPI const char* aicore_openpcdet_model_download_base(void) {
    return kDownloadBase;
}
