// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SAM 3D Objects (image -> 3D) — public C ABI.
//
// Pipeline (ported from the sam-3d-objects-ggml runtime):
//   decoded image (+ binary alpha/mask) -> native MoGe point map ->
//   native condition preprocessing -> DINOv2 + PointPatch conditioners ->
//   SparseStructure flow + decoder -> SLat flow -> Gaussian decoder
//   (Gaussian PLY export) and optional 101-channel mesh decoder
//   (FlexiCubes surface, returned as a typed result).
//
// Model files are the published GGUF exports of
// https://huggingface.co/Asher-1/SAM_3D_OBJECTS_GGUF, resolved through
// aicore_sam3d_model_* below. The runtime catalog (URLs, sizes, SHA-256) is
// owned by this task; consumers must not maintain a second model table.
//
// Contract notes:
//  - Decoded images cross the boundary as borrowed aicore_image_view values
//    with their real row stride. The binary object mask is expressed either
//    by the view's alpha channel or by a separate mask view; both follow the
//    official `mask > 0` semantics.
//  - The Gaussian PLY is an export artifact written to the caller-provided
//    path. The mesh decoder result is returned in memory as a typed struct.
//  - All inference work is serialized per context; contexts may run in
//    parallel (the GPU queue is shared through the process backend registry).
#ifndef AICORE_SAM3D_CAPI_H
#define AICORE_SAM3D_CAPI_H

#include <stdint.h>

#include "aicore/export.h"
#include "aicore/image_view.h"
#include "aicore/pipeline_timing.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AICORE_SAM3D_ABI_VERSION 1

typedef struct aicore_sam3d_ctx aicore_sam3d_ctx;
typedef struct aicore_sam3d_options aicore_sam3d_options;
typedef struct aicore_sam3d_result aicore_sam3d_result;

AICORE_CAPI int aicore_sam3d_abi_version(void);

// ---- Options ---------------------------------------------------------------
//
// device: "auto" (default), "cpu", "cuda", "vulkan" — same resolution rules
// as the other AICore tasks (downgrade reasons surface in backend_note).
// dtype:  AICORE_SAM3D_DTYPE_F16 | Q8_0 | Q4_K selects the published
//         `<stage>-<dtype>.gguf` family inside models_dir.
typedef enum aicore_sam3d_dtype {
    AICORE_SAM3D_DTYPE_F16 = 0,
    AICORE_SAM3D_DTYPE_Q8_0 = 1,
    AICORE_SAM3D_DTYPE_Q4_K = 2,
} aicore_sam3d_dtype;

AICORE_CAPI aicore_sam3d_options* aicore_sam3d_options_new(void);
AICORE_CAPI void aicore_sam3d_options_free(aicore_sam3d_options* options);

// Directory holding the published GGUF files (<stage>-<dtype>.gguf layout).
AICORE_CAPI void aicore_sam3d_options_set_models_dir(
        aicore_sam3d_options* options, const char* models_dir);
// Optional explicit MoGe GGUF path; defaults to
// <models_dir>/moge_vitl-f16.gguf.
AICORE_CAPI void aicore_sam3d_options_set_moge_gguf(
        aicore_sam3d_options* options, const char* moge_gguf);
AICORE_CAPI void aicore_sam3d_options_set_dtype(aicore_sam3d_options* options,
                                                aicore_sam3d_dtype dtype);
AICORE_CAPI void aicore_sam3d_options_set_device(aicore_sam3d_options* options,
                                                 const char* device);
AICORE_CAPI void aicore_sam3d_options_set_threads(aicore_sam3d_options* options,
                                                  int n_threads);
// Diffusion sampler configuration. steps <= 0 falls back to the official
// trajectory gate (25 for both stages); seed 0 means "use 42" only as an
// unset sentinel — pass the exact seed for reproducible geometry.
AICORE_CAPI void aicore_sam3d_options_set_seed(aicore_sam3d_options* options,
                                               int seed);
AICORE_CAPI void aicore_sam3d_options_set_steps(aicore_sam3d_options* options,
                                                int ss_steps,
                                                int slat_steps);
// Strict F32 SS-backbone attention (default off — the F16-KV flash
// throughput path is the production default; strict is the parity
// diagnostic) and the portable (graph-decomposed) Gaussian-decoder
// attention.
AICORE_CAPI void aicore_sam3d_options_set_strict_ss_attention(
        aicore_sam3d_options* options, int strict);
AICORE_CAPI void aicore_sam3d_options_set_gs_portable_attention(
        aicore_sam3d_options* options, int portable);
// Cross-backend Philox sampling contract for CPU/Vulkan parity replays
// (0 = generate the documented per-shape default; a positive value makes the
// request self-contained instead of relying on a diagnostic dump).
AICORE_CAPI void aicore_sam3d_options_set_philox_blocks(
        aicore_sam3d_options* options, uint32_t blocks);
// Force the MoGe forward even when the RGB content matches the previous
// request (timing runs; production batches should keep the reuse).
AICORE_CAPI void aicore_sam3d_options_set_disable_moge_cache(
        aicore_sam3d_options* options, int disable);
// Accuracy-diagnostic inputs for stage bisects against the upstream
// sam-3d-objects-ggml reference:
//   noise_dir: immutable official stage directory with initial SS/SLat
//       noise (sam3d-cli rng-dump / image-to-3d --noise-dir layout). When
//       empty the request generates its own noise from seed.
//   conditions_out: caller-owned condition directory. When non-empty the
//       ss_input_*.samt condition tensors are written there (upstream
//       sam3d-cli --conditions-out layout, byte-compatible) and kept after
//       the session finishes; when empty a temporary directory is used and
//       removed. The upstream directory can then be replayed through
//       options_set_conditions_out + generate to attribute any divergence
//       to kernel execution rather than conditioning.
AICORE_CAPI void aicore_sam3d_options_set_noise_dir(
        aicore_sam3d_options* options, const char* noise_dir);
AICORE_CAPI void aicore_sam3d_options_set_conditions_out(
        aicore_sam3d_options* options, const char* conditions_out);

// ---- Context lifecycle -----------------------------------------------------
//
// Loads nothing heavy; model weights are streamed per stage by the pipeline.
// Returns NULL on contract errors (err receives the message when non-NULL);
// a ready context answers is_ready() == 1 and carries queryable errors.
AICORE_CAPI aicore_sam3d_ctx* aicore_sam3d_load_opts(
        const aicore_sam3d_options* options, char* err, size_t err_size);
AICORE_CAPI void aicore_sam3d_free(aicore_sam3d_ctx* ctx);
AICORE_CAPI int aicore_sam3d_is_ready(const aicore_sam3d_ctx* ctx);
AICORE_CAPI const char* aicore_sam3d_last_error(const aicore_sam3d_ctx* ctx);
AICORE_CAPI const char* aicore_sam3d_backend(const aicore_sam3d_ctx* ctx);
// Human-readable device downgrade/explanation note; NULL when none.
AICORE_CAPI const char* aicore_sam3d_backend_note(const aicore_sam3d_ctx* ctx);

// Bitfield of AICORE_SAM3D_CAP_* values.
AICORE_CAPI int aicore_sam3d_caps(const aicore_sam3d_ctx* ctx);
#define AICORE_SAM3D_CAP_GAUSSIAN 0x1u
#define AICORE_SAM3D_CAP_MESH 0x2u

// ---- Inference -------------------------------------------------------------
//
// image: borrowed decoded pixels; RGB8/RGBA8/GRAY8/BGR8/BGRA8 with the real
//        row_stride_bytes. GRAY8 is expanded to RGB; alpha semantics apply
//        when the format carries alpha.
// mask:  optional borrowed binary-mask view (> 0 -> object). GRAY8 or an
//        alpha-carrying format; overrides the image alpha channel.
// out_ply: destination path for the Gaussian PLY export (required).
// decode_mesh: when non-zero, also run the FlexiCubes mesh decoder
//        (requires slat_decoder_mesh-<dtype>.gguf in models_dir).
// progress: optional stage callback. stage is one of the
//        AICORE_SAM3D_STAGE_* values, step/total refine the current stage
//        when meaningful (total 0 = stage-level event only).
//
// Not thread-safe on one ctx; concurrent calls must serialize externally.
typedef void (*aicore_sam3d_progress_fn)(void* user,
                                         int stage,
                                         int step,
                                         int total);
#define AICORE_SAM3D_STAGE_LOAD 0
#define AICORE_SAM3D_STAGE_POINTMAP 1
#define AICORE_SAM3D_STAGE_CONDITION 2
#define AICORE_SAM3D_STAGE_SS_FLOW 3
#define AICORE_SAM3D_STAGE_SS_DECODE 4
#define AICORE_SAM3D_STAGE_SLAT_FLOW 5
#define AICORE_SAM3D_STAGE_GS_DECODE 6
#define AICORE_SAM3D_STAGE_MESH_DECODE 7

AICORE_CAPI aicore_sam3d_result* aicore_sam3d_generate(
        aicore_sam3d_ctx* ctx,
        const aicore_image_view* image,
        const aicore_image_view* mask,
        const char* out_ply,
        int decode_mesh,
        aicore_sam3d_progress_fn progress,
        void* user);

// ---- Typed result ----------------------------------------------------------
//
// mesh accessors return NULL / 0 when the mesh decoder did not run. Vertices
// and triangles stay owned by the result; copy before releasing it.
AICORE_CAPI int aicore_sam3d_result_has_mesh(const aicore_sam3d_result* result);
AICORE_CAPI int aicore_sam3d_result_mesh_vertex_count(
        const aicore_sam3d_result* result);
AICORE_CAPI int aicore_sam3d_result_mesh_triangle_count(
        const aicore_sam3d_result* result);
AICORE_CAPI const float* aicore_sam3d_result_mesh_vertices(
        const aicore_sam3d_result* result);
AICORE_CAPI const float* aicore_sam3d_result_mesh_normals(
        const aicore_sam3d_result* result);
AICORE_CAPI const uint32_t* aicore_sam3d_result_mesh_triangles(
        const aicore_sam3d_result* result);
// Number of active voxels behind the sample (diagnostic).
AICORE_CAPI int aicore_sam3d_result_voxel_count(
        const aicore_sam3d_result* result);
// Gaussian splat count encoded in the exported PLY (0 when unknown).
AICORE_CAPI int64_t
aicore_sam3d_result_gaussian_count(const aicore_sam3d_result* result);
AICORE_CAPI void aicore_sam3d_result_free(aicore_sam3d_result* result);

// Common timing contract: one entry per generate call. e2e_ms is measured;
// the stage split follows the aicore_pipeline_timings semantics with only
// measurable stages marked in valid_fields.
AICORE_CAPI int aicore_sam3d_last_pipeline_timings(
        const aicore_sam3d_ctx* ctx, aicore_pipeline_timings* timings);

// Universal release for any buffer handed out by this API (currently none —
// results own their storage — but kept for ABI stability).
AICORE_CAPI void aicore_sam3d_free_buffer(void* buffer);

// Idempotent process-level cleanup; delegates to aicore_runtime_shutdown and
// never destroys live contexts.
AICORE_CAPI void aicore_sam3d_shutdown(void);

// ---- Runtime model catalog -------------------------------------------------
//
// The published Asher-1/SAM_3D_OBJECTS_GGUF inventory (filename, download
// URL, size, SHA-256) is the single source of truth for cache validation and
// downloads; qSam3d must read it only through these accessors.
typedef struct aicore_sam3d_model_entry {
    const char* filename;
    const char* download_url;
    const char* display_name;
    const char* quant_note;
    const char* license_note;
    const char* role;  // "moge" | "ss_generator" | "ss_decoder" |
                       // "slat_generator" | "slat_decoder_gs" |
                       // "slat_decoder_gs_4" | "slat_decoder_mesh"
    int64_t size_bytes;
    const char* sha256;
} aicore_sam3d_model_entry;

AICORE_CAPI int aicore_sam3d_model_count(void);
AICORE_CAPI const aicore_sam3d_model_entry* aicore_sam3d_model_at(int index);
AICORE_CAPI const aicore_sam3d_model_entry* aicore_sam3d_model_by_filename(
        const char* filename);
AICORE_CAPI const char* aicore_sam3d_model_download_base(void);
// Preferred cache directory (shared data root); owned by the library.
AICORE_CAPI const char* aicore_sam3d_model_cache_dir(void);

AICORE_CAPI const char* aicore_sam3d_info_json(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AICORE_SAM3D_CAPI_H
