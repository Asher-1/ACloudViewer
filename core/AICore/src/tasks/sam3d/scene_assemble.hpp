// Native multi-object scene assembly: the official demo_multi_object flow
// (N single-object reconstructions, make_scene pose application,
// concatenation, optional normalization) ported from the upstream
// sam-3d-objects-ggml scene_assemble module. No renderer and no orbit
// cameras here: ACloudViewer consumes the composed splat set directly.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "pose_decoder.hpp"  // NativeInstancePose

namespace sam3d {

// Activated gaussian splat representation (the upstream GaussianSplatSet):
// positions/sh0 are the raw interchange values, opacities = sigmoid(logit),
// scales = exp(log_scale), rotations = normalized wxyz quaternion.
struct SceneSplatSet {
    std::vector<float> positions;  // 3 * P
    std::vector<float> sh0;        // 3 * P
    std::vector<float> opacities;  // P
    std::vector<float> scales;     // 3 * P
    std::vector<float> rotations;  // 4 * P

    size_t size() const { return opacities.size(); }
    bool valid() const;
};

// One composed object's PLY-semantic interchange input. Every pointer is a
// borrowed caller-owned array; the values are exactly the binary Gaussian
// PLY row fields (see write_gaussian_ply), with centers in the world domain
// (PLY x/y/z). The pose is the official ScaleShiftInvariant receipt decoded
// by the pipeline (write_native_pose_json's top-level rotation/translation/
// scale fields).
struct SceneObjectInput {
    size_t splat_count = 0;
    const float* centers = nullptr;        // 3 * N
    const float* sh0 = nullptr;            // 3 * N
    const float* opacity_logit = nullptr;  // N
    const float* log_scale = nullptr;      // 3 * N
    const float* rot_ply = nullptr;        // 4 * N (unnormalized)
    const NativeInstancePose* pose = nullptr;
};

// Activate one object's interchange rows exactly like the upstream Gaussian
// PLY loader (sigmoid / exp / quaternion normalize) and apply the official
// make_scene pose semantics to the activated representation.
bool activate_and_pose_object(const SceneObjectInput& object,
                              SceneSplatSet& out, std::string& error);

// Concatenate an object's splats onto the accumulating scene.
void append_splat_set(SceneSplatSet& scene, const SceneSplatSet& object);

// Official normalized_gaussian: uniform rescale and centering driven by the
// opacity > 0.9 active bounds (fix_alignment=False, no axis permutation).
bool normalize_scene(SceneSplatSet& scene, std::string& error);

// Full composition: per-object activation + pose, concatenation, optional
// normalization. Mirrors the upstream scene-assemble command.
bool compose_scene(const SceneObjectInput* objects, size_t object_count,
                   bool normalize, SceneSplatSet& out, std::string& error);

}  // namespace sam3d
