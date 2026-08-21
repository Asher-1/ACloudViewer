// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Depth Anything (V1/V2/V3) monocular / multi-view metric depth + pose C API.
//
// The ggml engine under core/AICore/src/tasks/depth/ implements the Depth
// Anything family architectures ("Depth Anything V2" / "Depth Anything V3",
// https://github.com/DepthAnything); checkpoints are exported to GGUF via
// core/AICore/scripts/export_gguf_variants.py.

#pragma once
#include <stddef.h>

#include "aicore/export.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef struct aicore_depth_ctx aicore_depth_ctx;
typedef struct aicore_depth_options aicore_depth_options;
/* ABI version. 3: dense/points APIs. 4: nested two-branch model. 5: explicit
   per-context device selection without process-global environment mutation.
   6: options-based loading. aicore_depth_load_opts /
   aicore_depth_load_nested_opts replace the four flat-argument load variants
   (load / load_device / load_nested / load_nested_device), and the former
   DA_FUSED / DA3_FORCE_JOINT_MV / DA_PROFILE environment toggles became
   explicit aicore_depth_options setters — AICore reads no environment
   variables for logic control. 7: dense / multi-view / colmap outputs are
   packaged into result structs (aicore_depth_dense_result,
   aicore_depth_multiview_result, aicore_depth_multiview_data) instead of
   flat out-parameter lists; buffers are released with
   aicore_depth_free_buffer. */
AICORE_CAPI int aicore_depth_abi_version(void);
/* Options handle (opaque, heap-allocated, value semantics). All setters are
   no-ops on NULL and ignore NULL/invalid values, keeping whatever was set
   before. Defaults reproduce the historical "environment unset" behavior:
   device "auto", threads 0 (backend default), fused graph ON, joint
   multiview OFF, profile logging OFF. */
AICORE_CAPI aicore_depth_options* aicore_depth_options_new(void);
/** Releases an options struct created by aicore_depth_options_new. */
AICORE_CAPI void aicore_depth_options_free(aicore_depth_options* opts);
/* "auto|cpu|cuda|opencl|vulkan|metal" (backend family or instance name such
   as "CUDA0"). NULL/empty keeps the current value. */
AICORE_CAPI void aicore_depth_options_set_device(aicore_depth_options* opts,
                                                 const char* device);
/* 0 = backend default thread count. */
AICORE_CAPI void aicore_depth_options_set_threads(aicore_depth_options* opts,
                                                  int n_threads);
/* Fused backbone+head ONE-graph path (default ON): features stay
   device-resident. 0 restores the original two-graph path (backbone -> host
   features -> head) kept for A/B and debugging. cat_token=false models use
   the unfused path regardless. */
AICORE_CAPI void aicore_depth_options_set_fused_graph(
        aicore_depth_options* opts, int enabled);
/* Joint multiview (default OFF): cross-view global attention for small view
   sets. The default sequential per-view inference keeps the VRAM peak O(1)
   in the view count. */
AICORE_CAPI void aicore_depth_options_set_force_joint_multiview(
        aicore_depth_options* opts, int enabled);
/* Per-stage timing logs (default OFF). */
AICORE_CAPI void aicore_depth_options_set_profile_logging(
        aicore_depth_options* opts, int enabled);
/* Keep the graph allocator high-water between inference calls (default OFF).
   OFF drops allocators after every graph so VRAM stays at the single-graph
   peak (the right default for multi-view / one-shot jobs). ON keeps them for
   repeated same-shape inference (video frames) at the cost of the high-water
   mark. */
AICORE_CAPI void aicore_depth_options_set_keep_graph_buffers(
        aicore_depth_options* opts, int enabled);
/* Load a depth model. opts may be NULL (all defaults). NULL on failure. */
AICORE_CAPI aicore_depth_ctx* aicore_depth_load_opts(
        const char* gguf_path, const aicore_depth_options* opts);
/* Load a NESTED metric model from its two branches: the anyview (GIANT) GGUF
   and the metric (ViT-L + DPT/sky) GGUF. The returned ctx runs the nested
   metric alignment: aicore_depth_depth_dense / aicore_depth_depth_path /
   aicore_depth_pose_path all produce the final metric-scale depth + scaled
   extrinsics (is_metric=1, conf/sky = NULL). opts may be NULL. NULL on
   failure. */
AICORE_CAPI aicore_depth_ctx* aicore_depth_load_nested_opts(
        const char* anyview_gguf,
        const char* metric_gguf,
        const aicore_depth_options* opts);
/** Releases a context returned by aicore_depth_load*; safe on NULL. */
AICORE_CAPI void aicore_depth_free(aicore_depth_ctx* ctx); /* safe on NULL */
/** True only when a context has a loaded depth engine. */
AICORE_CAPI int aicore_depth_is_ready(const aicore_depth_ctx* ctx);
/* malloc'd JSON describing model config; free via aicore_depth_free_buffer. */
AICORE_CAPI char* aicore_depth_info_json(aicore_depth_ctx* ctx);
/** Releases any buffer returned by an aicore_depth_* function (string, float
 *  array or byte array; unified entry point). Safe on NULL. */
AICORE_CAPI void aicore_depth_free_buffer(void* p);
/** Returns the last error message of the context (empty when none). */
AICORE_CAPI const char* aicore_depth_last_error(aicore_depth_ctx* ctx);
/* Active ggml backend after load (e.g. "cpu", "CUDA0", "Vulkan0"). Empty when
   ctx is null. Pointer valid until aicore_depth_free(ctx). */
AICORE_CAPI const char* aicore_depth_device_name(
        aicore_depth_ctx* ctx); /* owned by ctx, "" if none */
/* Dense per-pixel output for a single image (owned by the caller; release
   with aicore_depth_dense_result_free or aicore_depth_free_buffer).
     - DualDPT model (camera-pose capable): depth + conf are filled, sky = NULL,
       ext[12] (3x4 row-major) + intr[9] (3x3) filled.
     - mono model (DA3MONO): depth + sky are filled, conf = NULL, ext/intr
       zeroed (mono has no camera pose).
     - nested model (aicore_depth_load_nested): depth = final metric-scale
       depth, conf = sky = NULL, ext/intr = scaled extrinsics/intrinsics.
   is_metric = 1 for metric/nested/mono variants (best-effort from config),
   else 0. On failure the struct is zeroed and -1 is returned (see
   aicore_depth_last_error). */
typedef struct aicore_depth_dense_result {
    int width;     /* processed width  */
    int height;    /* processed height */
    float* depth;  /* malloc'd [H*W] row-major; NULL when not produced */
    float* conf;   /* malloc'd [H*W] row-major; NULL when not produced */
    float* sky;    /* malloc'd [H*W] row-major; NULL when not produced */
    float ext[12]; /* 3x4 row-major; zeroed when the model has no pose */
    float intr[9]; /* 3x3 row-major; zeroed when the model has no pose */
    int is_metric; /* 1 for metric/nested/mono variants, else 0 */
} aicore_depth_dense_result;

/** Releases buffers owned by a result returned by aicore_depth_depth_dense.
 *  Safe on NULL; does not free the struct itself (it is caller-allocated). */
AICORE_CAPI void aicore_depth_dense_result_free(aicore_depth_dense_result* r);

/* Multi-view depth+pose output (owned by the caller; release with
   aicore_depth_multiview_result_free). depth is malloc'd [n*H*W] float
   (view-major, row-major per view); ext is [n*12] 3x4 row-major; intr is
   [n*9] 3x3 row-major. */
typedef struct aicore_depth_multiview_result {
    int width;    /* processed width  */
    int height;   /* processed height */
    int n_views;  /* number of views   */
    float* depth; /* malloc'd [n*H*W] view-major */
    float* ext;   /* malloc'd [n*12] 3x4 row-major */
    float* intr;  /* malloc'd [n*9] 3x3 row-major */
} aicore_depth_multiview_result;

/** Releases buffers owned by a result returned by
 *  aicore_depth_depth_pose_multi. Safe on NULL. */
AICORE_CAPI void aicore_depth_multiview_result_free(
        aicore_depth_multiview_result* r);

/* Borrowed multi-view depth+pose data for aicore_depth_write_colmap_from_
   multiview (the caller keeps the arrays alive for the call). depth is
   [n*H*W] row-major; ext is [n*12] 3x4 row-major; intr is [n*9] 3x3. */
typedef struct aicore_depth_multiview_data {
    int n_views;        /* number of views */
    int height;         /* processed height */
    int width;          /* processed width */
    const float* depth; /* borrowed [n*H*W] row-major */
    const float* ext;   /* borrowed [n*12] 3x4 row-major */
    const float* intr;  /* borrowed [n*9] 3x3 row-major */
} aicore_depth_multiview_data;

/* Run depth on an image file. On success writes *out_h,*out_w and returns a
   malloc'd float[H*W] depth map (row-major); caller frees via
   aicore_depth_free_buffer. NULL on error. */
AICORE_CAPI float* aicore_depth_depth_path(aicore_depth_ctx* ctx,
                                           const char* image_path,
                                           int* out_h,
                                           int* out_w);

/* Run pose; fills ext[12] (3x4 row-major) and intr[9] (3x3). Returns 0 ok, -1
 * error. */
AICORE_CAPI int aicore_depth_pose_path(aicore_depth_ctx* ctx,
                                       const char* image_path,
                                       float out_ext[12],
                                       float out_intr[9]);
/* Multi-view depth+pose. n_images paths. Fills the caller-owned result
   struct: malloc'd depth [n*H*W] (view-major) plus per-view ext [n*12] and
   intr [n*9]. Release with aicore_depth_multiview_result_free. Returns 0 ok,
   -1 error. */
AICORE_CAPI int aicore_depth_depth_pose_multi(
        aicore_depth_ctx* ctx,
        const char** image_paths,
        int n_images,
        aicore_depth_multiview_result* out);
/* Single-image 3D export. Runs the native depth+pose pipeline, captures the
   processed-resolution RGB colors, and writes a glTF-2.0 binary point cloud to
   out_glb. Returns 0 ok, -1 error (see aicore_depth_last_error). */
AICORE_CAPI int aicore_depth_export_glb(aicore_depth_ctx* ctx,
                                        const char* image_path,
                                        const char* out_glb);
/* Single-image 3D export to a COLMAP sparse model (cameras/images/points3D) in
   directory out_dir. binary != 0 => .bin (default); 0 => .txt. Returns 0 ok, -1
   error. */
AICORE_CAPI int aicore_depth_export_colmap(aicore_depth_ctx* ctx,
                                           const char* image_path,
                                           const char* out_dir,
                                           int binary);
/* Multi-view COLMAP sparse export: multiview depth+pose, back-project to
   points3D. image_paths has n_images entries; writes under out_dir. Returns 0
   ok, -1 error. */
AICORE_CAPI int aicore_depth_export_colmap_multi(aicore_depth_ctx* ctx,
                                                 const char** image_paths,
                                                 int n_images,
                                                 const char* out_dir,
                                                 int binary);
/* Same as aicore_depth_export_colmap_multi but image_names[i] is COLMAP
   Image.Name() (relative to the image root). NULL names fall back to
   basename(image_paths[i]). */
AICORE_CAPI int aicore_depth_export_colmap_multi_named(aicore_depth_ctx* ctx,
                                                       const char** image_paths,
                                                       const char** image_names,
                                                       int n_images,
                                                       const char* out_dir,
                                                       int binary);
/* Write COLMAP sparse from an existing multiview depth+pose result (no
   re-inference). data.depth is n*h*w row-major; data.ext is n*12 row-major
   3x4; data.intr is n*9 row-major 3x3 (see aicore_depth_multiview_data). */
AICORE_CAPI int aicore_depth_write_colmap_from_multiview(
        aicore_depth_ctx* ctx,
        const char** image_paths,
        const char** image_names,
        const aicore_depth_multiview_data* data,
        const char* out_dir,
        int binary);

/* Dense per-pixel output for a single image (see aicore_depth_dense_result).
   Returns 0 ok, -1 error. Any of the out_* members may stay NULL when not
   produced by the model; out_ext/out_intr are zeroed when the model has no
   camera pose. */
AICORE_CAPI int aicore_depth_depth_dense(aicore_depth_ctx* ctx,
                                         const char* image_path,
                                         aicore_depth_dense_result* out);

/* Single-image 3D point cloud (DualDPT/pose-capable models only; returns -1 for
   mono models with a clear last_error). Runs depth+pose+processed-RGB,
   back-projects to world space keeping pixels with conf >= conf_thresh. On
   success sets *out_n and writes a malloc'd *out_xyz[3*N float] + *out_rgb[3*N
   uint8]; free xyz / rgb via aicore_depth_free_buffer. Returns 0 ok, -1 error.
 */
AICORE_CAPI int aicore_depth_points(aicore_depth_ctx* ctx,
                                    const char* image_path,
                                    float conf_thresh,
                                    int* out_n,
                                    float** out_xyz,
                                    unsigned char** out_rgb);

/* Default cross-platform GGUF model cache directory (UTF-8). Free with
 * aicore_depth_free_buffer. */
AICORE_CAPI char* aicore_depth_model_cache_dir(void);
/* Override preprocess longest-side target before inference (0 = model default).
 * Clamped to >= patch_size. Safe to call on NULL ctx (no-op). */
AICORE_CAPI void aicore_depth_set_img_resize_target(aicore_depth_ctx* ctx,
                                                    int target);
/* Drop ggml graph buffers and (when GPU offloading) device-resident weights.
 * Call between sequential per-view inferences to keep VRAM peak O(1). */
AICORE_CAPI void aicore_depth_release_gpu_working_memory(aicore_depth_ctx* ctx);
/* Cap preprocess long-edge from free GPU VRAM (single-view activation peak).
 * Returns min(requested, vram-safe cap). No-op on CPU / when ctx is NULL. */
AICORE_CAPI int aicore_depth_cap_img_resize_target(aicore_depth_ctx* ctx,
                                                   int requested);
/* Lightweight main-thread backend warmup: register ggml backends and clear
 * sticky CUDA errors. Returns 0 on success. */
AICORE_CAPI int aicore_depth_warmup_backend(const char* device);
/* Single-image 3D Gaussian reconstruction (anyview/GIANT branch). On success
   sets *out_h,*out_w,*out_n and malloc'd flat arrays (caller frees via
   aicore_depth_free_buffer): *out_means[n*3], *out_scales[n*3],
   *out_harmonics[n*3*9], *out_opacities[n]. Returns 0 ok, -1 error. */
AICORE_CAPI int aicore_depth_reconstruct_path(const char* gguf_path,
                                              int n_threads,
                                              const char* image_path,
                                              int* out_h,
                                              int* out_w,
                                              int* out_n,
                                              float** out_means,
                                              float** out_scales,
                                              float** out_harmonics,
                                              float** out_opacities);
/* Quantize GGUF matmul weights to type (f16/q8_0/q6_k/q5_k/q4_k). Returns 0
 * ok, -1 error. */
AICORE_CAPI int aicore_depth_quantize_gguf(const char* in_gguf,
                                           const char* out_gguf,
                                           const char* type);
#ifdef __cplusplus
}
#endif
