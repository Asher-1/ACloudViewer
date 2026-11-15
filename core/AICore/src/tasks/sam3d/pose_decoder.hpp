// Native implementation of the pose-decoding contract in pipeline.yaml.
//
// SAM 3D Objects predicts a normalized 6D rotation together with log-scale
// and translation heads.  The public pipeline config selects
// ScaleShiftInvariant, so conversion into the input-camera frame must also
// apply the point-map scene scale and shift.
#pragma once

#include <array>
#include <string>

namespace sam3d {

struct NativeInstancePose {
    // PyTorch3D convention: scalar first quaternion (w, x, y, z).
    std::array<float, 4> rotation_wxyz{};
    std::array<float, 3> translation{};
    std::array<float, 3> scale{};
    // The SS model emits this log-space head.  ScaleShiftInvariant decodes it
    // with exp(), but its official conversion intentionally does not consume
    // it when constructing the final instance transform.
    float translation_scale = 1.0f;
    std::array<float, 3> scene_scale{};
    std::array<float, 3> scene_shift{};
};

// Match sam3d_objects.pipeline.inference_utils.pose_decoder("ScaleShiftInvariant")
// for one batch element.  All arguments contain F32 values in the same order
// as their exported SAMT tensors.
//
// downsample_factor reproduces the official post-decode rescale
// (inference_pipeline.py: "Rescaling scale by <factor> after downsampling",
// scale *= downsample_factor): when the sparse support exceeds max_coords the
// official pipeline decodes on the downsampled grid and compensates the
// instance scale by the same factor.  It multiplies the decoded scale only -
// the rotation is decomposed before the rescale, exactly like the official
// order (pose_decoder first, scale rescale after).  Standalone decoders pass
// the default 1.
bool decode_scale_shift_invariant_pose(const float normalized_rotation_6d[6],
                                       const float log_scale[3],
                                       const float translation[3],
                                       float log_translation_scale,
                                       const float scene_scale[3],
                                       const float scene_shift[3],
                                       NativeInstancePose& output,
                                       std::string& error,
                                       int downsample_factor = 1);

// Serialize an independently comparable, stable JSON pose artifact.  This is
// deliberately separate from GLB export: the official inference pipeline
// returns pose metadata but does not bake it into decode_slat's local mesh.
bool write_native_pose_json(const std::string& path, const NativeInstancePose& pose,
                            std::string& error);

}  // namespace sam3d
