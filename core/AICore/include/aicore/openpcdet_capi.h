// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// OpenPCDet LiDAR 3D object detection C API (in-tree port of
// OpenPCDet-GGML: https://github.com/Asher-1/OpenPCDet-GGML).
//
// The engine lives in core/AICore/src/tasks/openpcdet/ and runs on stock ggml
// ops only (get_rows + mul_mat sparse convolution, no custom operators), with
// CPU / CUDA / Vulkan backends selected through the shared AICore runtime
// lease registry. The host side owns voxelization, VFE, rulebook generation,
// box decoding and rotated NMS; the ggml graph covers the sparse/dense
// backbone and the detection heads.
//
// Phase-1 supported architectures (selected from GGUF
// `general.architecture`): "openpcdet.pointpillar" (incl. multihead /
// CenterHead pillar variants) and "openpcdet.second" / "openpcdet.voxel"
// (VoxelBackBone8x / VoxelResBackBone8x sparse backbones).

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "aicore/export.h"
#include "aicore/pipeline_timing.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Returns the ABI version of the OpenPCDet C API (bump on breaking ABI
 *  changes). */
AICORE_CAPI int aicore_openpcdet_abi_version(void);

typedef struct aicore_openpcdet_ctx aicore_openpcdet_ctx;
typedef struct aicore_openpcdet_options aicore_openpcdet_options;

/** Creates a default options struct (device "auto", threads 0 = backend
 *  default, thresholds from GGUF metadata). Release with
 *  aicore_openpcdet_options_free. */
AICORE_CAPI aicore_openpcdet_options* aicore_openpcdet_options_new(void);
/** Releases an options struct created by aicore_openpcdet_options_new. */
AICORE_CAPI void aicore_openpcdet_options_free(aicore_openpcdet_options* opts);
/** Selects the inference device: NULL or "auto", "cpu", "gpu", "vulkan"
 *  (optionally ":N"), "cuda" (Linux/Windows). */
AICORE_CAPI void aicore_openpcdet_options_set_device(
        aicore_openpcdet_options* opts, const char* device);
/** CPU thread count; <= 0 picks the backend default. */
AICORE_CAPI void aicore_openpcdet_options_set_threads(
        aicore_openpcdet_options* opts, int n_threads);
/** Overrides the detection score threshold; NaN (the default) keeps the
 *  GGUF-declared value. */
AICORE_CAPI void aicore_openpcdet_options_set_score_thresh(
        aicore_openpcdet_options* opts, float t);
/** Overrides the rotated NMS IoU threshold; NaN keeps the GGUF value. */
AICORE_CAPI void aicore_openpcdet_options_set_nms_thresh(
        aicore_openpcdet_options* opts, float t);
/** Overrides the maximum number of returned boxes; <= 0 keeps the GGUF
 *  value. */
AICORE_CAPI void aicore_openpcdet_options_set_max_boxes(
        aicore_openpcdet_options* opts, int n);
/** Enables per-stage tap dumping (diagnostics) into \p dir as
 *  <name>.f32 + <name>.meta.json pairs matching the upstream
 *  compare_taps.py format; NULL or "" disables it. */
AICORE_CAPI void aicore_openpcdet_options_set_dump_taps_dir(
        aicore_openpcdet_options* opts, const char* dir);

/** Load an OpenPCDet GGUF model. Returns NULL on failure; inspect
 *  aicore_openpcdet_last_error() on the NULL-safe wrapper of the returned
 *  context for the reason. */
AICORE_CAPI aicore_openpcdet_ctx* aicore_openpcdet_load_opts(
        const char* gguf_path, const aicore_openpcdet_options* opts);
/** Releases a context returned by aicore_openpcdet_load_opts; safe on NULL. */
AICORE_CAPI void aicore_openpcdet_free(aicore_openpcdet_ctx* ctx);
/** Returns 1 only when the context owns a successfully loaded model. */
AICORE_CAPI int aicore_openpcdet_is_ready(const aicore_openpcdet_ctx* ctx);
/** Returns the last error message of the context (empty when none). */
AICORE_CAPI const char* aicore_openpcdet_last_error(
        const aicore_openpcdet_ctx* ctx);

/** Releases any buffer returned by an aicore_openpcdet_* function. Safe on
 *  NULL. */
AICORE_CAPI void aicore_openpcdet_free_buffer(void* p);

/** Model geometry, sized from GGUF metadata so callers can validate inputs
 *  without hardcoding dataset constants. */
typedef struct aicore_openpcdet_geometry {
    int32_t abi_version;
    int32_t num_point_features;  // expected point dim (4 = x,y,z,intensity)
    int32_t grid_size_x;         // BEV/voxel grid x
    int32_t grid_size_y;
    int32_t grid_size_z;  // 1 for pillar models
    int32_t max_points_per_voxel;
    int32_t max_voxels;
    int32_t num_class;
    float voxel_size_x;
    float voxel_size_y;
    float voxel_size_z;
    float point_cloud_range[6];  // x0, y0, z0, x1, y1, z1
    char arch[32];               // e.g. "openpcdet.pointpillar"
} aicore_openpcdet_geometry;

/** Fills \p out with the loaded model's geometry. Returns 0 on success,
 *  -1 on invalid arguments. */
AICORE_CAPI int aicore_openpcdet_geometry_of(const aicore_openpcdet_ctx* ctx,
                                             aicore_openpcdet_geometry* out);

/** One detected 3D box (LiDAR frame). */
typedef struct aicore_openpcdet_detection {
    float x, y, z;     // box center
    float dx, dy, dz;  // box size
    float heading;     // box yaw
    float score;
    int32_t label;  // 1-indexed class (OpenPCDet convention)
} aicore_openpcdet_detection;

/** Typed hot-path detection on a borrowed, caller-owned row-major point
 *  buffer [n_points, point_dim] float32 (point_dim must match the model's
 *  num_point_features). Populates the internal detection store; read records
 *  with aicore_openpcdet_detection_count / _at. Returns the detection count
 *  (>= 0) on success and -1 on failure (see last_error). */
AICORE_CAPI int aicore_openpcdet_detect_points(aicore_openpcdet_ctx* ctx,
                                               const float* points,
                                               int32_t n_points,
                                               int32_t point_dim);

/** One multi-modal inference scene (BEVFusion). Every buffer is borrowed,
 *  caller-owned and stays valid for the duration of the detect_scene call.
 *  The camera payload matches the upstream dump_bevfusion_scene.py layout:
 *  images are packed NHWC float32 [num_cameras, h, w, channels]; each matrix
 *  buffer is row-major float32 [num_cameras, 4, 4]; the augmentations are
 *  float32 [4, 4]. */
typedef struct aicore_openpcdet_scene {
    int32_t abi_version;  // AICORE_OPENPCDET_SCENE_ABI_VERSION
    int32_t reserved;
    const float* points;
    int32_t n_points;
    int32_t point_dim;
    const float* images;
    int32_t num_cameras;
    int32_t image_height;
    int32_t image_width;
    int32_t image_channels;
    const float* lidar_to_image;
    const float* camera_to_lidar;
    const float* camera_intrinsics;
    const float* image_augmentation;
    const float* lidar_augmentation;
} aicore_openpcdet_scene;

/** Scene ABI version of this build. */
#define AICORE_OPENPCDET_SCENE_ABI_VERSION 1

/** Loads an upstream scene directory (images.bin + the four camera matrices,
 *  lidar_aug.bin and scene.json) into caller-freed scene buffers. On success
 *  \p out owns the decoded buffers; release them with
 *  aicore_openpcdet_free_scene. Returns 0 on success and -1 on failure
 *  (error text in \p error when non-NULL). */
AICORE_CAPI int aicore_openpcdet_load_scene_dir(const char* scene_dir,
                                                aicore_openpcdet_scene* out,
                                                char** error);

/** Releases scene buffers produced by aicore_openpcdet_load_scene_dir. Safe
 *  on NULL. */
AICORE_CAPI void aicore_openpcdet_free_scene(aicore_openpcdet_scene* scene);

/** Multi-modal hot-path detection on a borrowed scene (BEVFusion models;
 *  point-only models reject it — they have no camera branch). Same result
 *  access as aicore_openpcdet_detect_points. Returns the detection count
 *  (>= 0) on success and -1 on failure (see last_error). */
AICORE_CAPI int aicore_openpcdet_detect_scene(
        aicore_openpcdet_ctx* ctx, const aicore_openpcdet_scene* scene);

/** Number of detections from the most recent detect call; -1 on invalid
 *  context. */
AICORE_CAPI int aicore_openpcdet_detection_count(
        const aicore_openpcdet_ctx* ctx);
/** Borrowed detection record from the most recent detect call; valid until
 *  the next detect call or context free. */
AICORE_CAPI int aicore_openpcdet_detection_at(const aicore_openpcdet_ctx* ctx,
                                              int index,
                                              aicore_openpcdet_detection* out);
/** Common pipeline timings of the most recent detect call. Consumers must
 *  inspect valid_fields; unmeasured stages stay zero with their bit clear. */
AICORE_CAPI int aicore_openpcdet_last_pipeline_timings(
        const aicore_openpcdet_ctx* ctx, aicore_pipeline_timings* out);

/** Backend-RESOLVED device name ("CUDA0", "Vulkan0", "cpu", ...). Differs
 *  from the requested device when the GPU lease can't be acquired — surfaces
 *  silent CPU fallbacks. Owned by ctx. */
AICORE_CAPI const char* aicore_openpcdet_context_device(
        aicore_openpcdet_ctx* ctx);
/** Effective CPU thread count after the auto (<=0) resolution. */
AICORE_CAPI int aicore_openpcdet_context_threads(aicore_openpcdet_ctx* ctx);

/** Returns a JSON summary of the loaded model. Caller frees with
 *  aicore_openpcdet_free_buffer. */
AICORE_CAPI char* aicore_openpcdet_info_json(aicore_openpcdet_ctx* ctx);

/** Warms up the backend for `device`; returns 0 on success. */
AICORE_CAPI int aicore_openpcdet_warmup_backend(const char* device);
/** Releases process-wide OpenPCDet backend resources (idempotent). */
AICORE_CAPI void aicore_openpcdet_shutdown(void);
/** Returns the local model cache directory. Caller frees with
 *  aicore_openpcdet_free_buffer. */
AICORE_CAPI char* aicore_openpcdet_model_cache_dir(void);

/** Published GGUF catalog (cloudViewer_downloads OpenPCDet-GGUF release). */
typedef struct aicore_openpcdet_model_entry {
    const char* filename;
    const char* download_url;
    const char* display_name;
    const char* quant_note;
    const char* family_note;  // detector family / dataset summary
    const char* license_note;
} aicore_openpcdet_model_entry;

/** Number of published catalog entries. */
AICORE_CAPI int aicore_openpcdet_model_count(void);
/** Returns the catalog entry at `index` (NULL when out of range). */
AICORE_CAPI const aicore_openpcdet_model_entry* aicore_openpcdet_model_at(
        int index);
/** Index of the entry the catalog declares as its default — the row carrying
 *  the visible "(recommended)" marker. */
AICORE_CAPI int aicore_openpcdet_model_default_index(void);
/** Returns the catalog entry whose filename matches (NULL when not found). */
AICORE_CAPI const aicore_openpcdet_model_entry*
aicore_openpcdet_model_by_filename(const char* filename);
/** Returns the base URL of the published model release. */
AICORE_CAPI const char* aicore_openpcdet_model_download_base(void);

#ifdef __cplusplus
}
#endif
