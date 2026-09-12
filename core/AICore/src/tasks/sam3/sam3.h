// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

/*
** ── DLL Export Decoration ────────────────────────────────────────────────
**
** SAM3_API mirrors LLAMA_API / SD_API: tag every public function (and the
** opaque-type deleters) so MSVC emits the correct __declspec when sam3 is
** built / consumed as a shared library. For static builds the macro expands
** to nothing and there is no ABI surface.
**
** Define SAM3_SHARED on both build and consumer side when sam3 is a DLL;
** additionally define SAM3_BUILD only on the side that is compiling sam3.cpp
** (the CMake target sets this via target_compile_definitions).
*/
#ifdef SAM3_SHARED
#if defined(_WIN32) && !defined(__MINGW32__)
#ifdef SAM3_BUILD
#define SAM3_API __declspec(dllexport)
#else
#define SAM3_API __declspec(dllimport)
#endif
#else
#define SAM3_API __attribute__((visibility("default")))
#endif
#else
#define SAM3_API
#endif

/*
** ── Version ─────────────────────────────────────────────────────────────
*/

#define SAM3_VERSION_MAJOR 1
#define SAM3_VERSION_MINOR 0
#define SAM3_VERSION_PATCH 0
#define SAM3_VERSION "1.0.0"

/*
** ── Forward Declarations ─────────────────────────────────────────────────
*/

struct sam3_model;
struct sam3_state;
struct sam3_tracker;

/* Custom deleters so unique_ptr works with forward-declared opaque types.
** The struct itself is dllexport'd so the out-of-line operator() (which calls
** sam3_free_state / sam3_tracker_reset + delete) is reachable across the DLL
** boundary — the deleter needs the full type of sam3_state / sam3_tracker for
** `delete`, which only exists inside sam3.cpp. */
struct SAM3_API sam3_state_deleter {
    void operator()(sam3_state* p) const;
};
struct SAM3_API sam3_tracker_deleter {
    void operator()(sam3_tracker* p) const;
};

using sam3_state_ptr = std::unique_ptr<sam3_state, sam3_state_deleter>;
using sam3_tracker_ptr = std::unique_ptr<sam3_tracker, sam3_tracker_deleter>;

/*
** ── Model Type ──────────────────────────────────────────────────────────
*/

enum sam3_model_type {
    SAM3_MODEL_SAM3 = 0,         // Full SAM3 (ViT + detector + tracker)
    SAM3_MODEL_SAM3_VISUAL = 1,  // SAM3 visual-only (ViT + tracker, no text)
    SAM3_MODEL_SAM2 = 2,         // SAM2 (Hiera + tracker, no text/detector)
};

/*****************************************************************************
** Public Data Types
**
** Geometry primitives, images, masks, and detection results.
*****************************************************************************/

struct sam3_point {
    float x;
    float y;
};

struct sam3_box {
    float x0;  // top-left x
    float y0;  // top-left y
    float x1;  // bottom-right x
    float y1;  // bottom-right y
};

struct sam3_image {
    int width = 0;
    int height = 0;
    int channels = 3;
    std::vector<uint8_t> data;
};

struct sam3_mask {
    int width = 0;
    int height = 0;
    float iou_score = 0.0f;
    float obj_score = 0.0f;
    int instance_id = -1;
    std::vector<uint8_t> data;  // binary mask (0 or 255)
};

struct sam3_detection {
    sam3_box box;
    float score = 0.0f;
    float iou_score = 0.0f;
    int instance_id = -1;
    sam3_mask mask;
    std::vector<float> sam_token;  // raw SAM decoder output token (for obj_ptr)
};

struct sam3_result {
    std::vector<sam3_detection> detections;
};

/*****************************************************************************
** Parameters
**
** Configuration for model loading, segmentation, and video tracking.
*****************************************************************************/

// Backend device selection. AUTO follows the platform auto order (Metal ->
// CPU on macOS, CUDA -> CPU on Linux/Windows; Vulkan is deliberately skipped
// in AUTO — see sam3_backend_init) and uses the first one that initialises.
// `use_gpu = false` still forces CPU for backwards compatibility.
enum sam3_device {
    SAM3_DEVICE_AUTO = 0,
    SAM3_DEVICE_CPU = 1,
    SAM3_DEVICE_CUDA = 2,
    SAM3_DEVICE_VULKAN = 3,
    SAM3_DEVICE_METAL = 4,
};

struct sam3_params {
    std::string model_path;
    int n_threads = 4;
    bool use_gpu = true;
    sam3_device device = SAM3_DEVICE_AUTO;  // explicit device; AUTO follows
                                            // the platform auto order
    int seed = 42;
    int encode_img_size = 0;  // 0 = model default; override input resolution
};

struct sam3_tensor_info {
    int64_t ne[4] = {0, 0, 0, 0};
    uint64_t nb[4] = {0, 0, 0, 0};
    int type = 0;
    int op = 0;
    bool is_contiguous = false;
};

enum sam3_vit_block_stage {
    SAM3_VIT_BLOCK_STAGE_NORM1 = 0,
    SAM3_VIT_BLOCK_STAGE_WINDOW_PART,
    SAM3_VIT_BLOCK_STAGE_QKV_PROJ,
    SAM3_VIT_BLOCK_STAGE_ATTN_CORE,
    SAM3_VIT_BLOCK_STAGE_ATTN_PROJ,
    SAM3_VIT_BLOCK_STAGE_WINDOW_UNPART,
    SAM3_VIT_BLOCK_STAGE_NORM2,
    SAM3_VIT_BLOCK_STAGE_MLP_FC1,
    SAM3_VIT_BLOCK_STAGE_MLP_GELU,
    SAM3_VIT_BLOCK_STAGE_MLP_FC2,
    SAM3_VIT_BLOCK_STAGE_MLP,
};

enum sam3_vit_prefix_stage {
    SAM3_VIT_PREFIX_STAGE_PATCH_EMBED = 0,
    SAM3_VIT_PREFIX_STAGE_PATCH_IM2COL,
    SAM3_VIT_PREFIX_STAGE_PATCH_MULMAT_RAW,
    SAM3_VIT_PREFIX_STAGE_PATCH_MULMAT,
    SAM3_VIT_PREFIX_STAGE_POS_ADD,
    SAM3_VIT_PREFIX_STAGE_LN_PRE_NORM,
    SAM3_VIT_PREFIX_STAGE_LN_PRE,
};

struct sam3_pcs_params {
    std::string text_prompt;
    std::vector<sam3_box> pos_exemplars;
    std::vector<sam3_box> neg_exemplars;
    float score_threshold = 0.5f;
    float nms_threshold = 0.1f;
};

struct sam3_pvs_params {
    std::vector<sam3_point> pos_points;
    std::vector<sam3_point> neg_points;
    sam3_box box = {0, 0, 0, 0};
    bool use_box = false;
    bool multimask = false;
};

struct sam3_video_params {
    std::string text_prompt;
    float score_threshold = 0.5f;
    float nms_threshold = 0.1f;
    float assoc_iou_threshold = 0.1f;
    int hotstart_delay = 15;
    int max_keep_alive = 30;
    int recondition_every = 16;
    int fill_hole_area = 16;
};

struct sam3_video_info {
    int width = 0;
    int height = 0;
    int n_frames = 0;
    float fps = 0.0f;
};

/*****************************************************************************
** Public API
**
** Model lifecycle, image encoding, segmentation, and video tracking.
*****************************************************************************/

/*
** ── Model Lifecycle ──────────────────────────────────────────────────────
*/

/*
** Load a SAM3 model from the file specified in params.model_path.
** Returns nullptr on failure.
*/
SAM3_API std::shared_ptr<sam3_model> sam3_load_model(const sam3_params& params);

/* Free all resources held by a loaded model. */
SAM3_API void sam3_free_model(sam3_model& model);

/* Returns true if the model was loaded as visual-only (no text/detector path).
** SAM2 models are always considered visual-only. */
SAM3_API bool sam3_is_visual_only(const sam3_model& model);

/* Returns the model type (SAM2 or SAM3). */
SAM3_API sam3_model_type sam3_get_model_type(const sam3_model& model);

/* Returns the human-readable name of the active inference backend
** (e.g. "CUDA", "Metal", "Vulkan", "CPU"). Useful for reporting which
** device a model actually runs on. */
SAM3_API const char* sam3_backend_name(const sam3_model& model);

/*
** ── Inference State ──────────────────────────────────────────────────────
*/

/* Allocate inference state (backbone caches, PE buffers). */
SAM3_API sam3_state_ptr sam3_create_state(const sam3_model& model,
                                          const sam3_params& params);

/* Free inference state and its GPU buffers. */
SAM3_API void sam3_free_state(sam3_state& state);

/*
** ── Image Backbone ───────────────────────────────────────────────────────
*/

/*
** Encode an image through the ViT backbone and FPN neck.
** Call once per image before segmentation or tracking.
** Returns true on success, false on failure.
*/
SAM3_API bool sam3_encode_image(sam3_state& state,
                                const sam3_model& model,
                                const sam3_image& image);

/*
** Encode only the image features consumed by point/box segmentation and
** tracking. For full SAM3 checkpoints this skips the detector neck; use
** sam3_encode_image() when sam3_segment_pcs() will consume the image.
*/
SAM3_API bool sam3_encode_image_pvs(sam3_state& state,
                                    const sam3_model& model,
                                    const sam3_image& image);

/*
** ── Image Segmentation ──────────────────────────────────────────────────
*/

/* Segment using text prompt + exemplar boxes (PCS path). */
SAM3_API sam3_result sam3_segment_pcs(sam3_state& state,
                                      const sam3_model& model,
                                      const sam3_pcs_params& params);

/* Segment using point/box prompts (PVS path). */
SAM3_API sam3_result sam3_segment_pvs(sam3_state& state,
                                      const sam3_model& model,
                                      const sam3_pvs_params& params);

/* Cross-DLL allocator pairing: drain a sam3_result's vectors inside the DLL
** that allocated them. When sam3 is built as a shared library and the caller
** lives in a different module (e.g. an Unreal Engine plugin that overrides
** global ::operator new/delete), letting ~sam3_result fire on the caller
** side inlines std::vector<>::~vector() into the caller's TU and routes
** ::operator delete[] through the caller's allocator — which doesn't
** recognize blocks created here. Symptom: a heap-canary fatal in the host
** allocator (e.g. UE's FMallocBinned2 "Attempt to realloc an unrecognized
** block"). Call this immediately after extracting data; the result is left
** valid-but-empty so the caller's eventual ~sam3_result is a no-op. */
SAM3_API void sam3_free_result(sam3_result& result) noexcept;

/*
** ── Video Tracking ──────────────────────────────────────────────────────
*/

/* Create a tracker for text-prompted video segmentation. */
SAM3_API sam3_tracker_ptr sam3_create_tracker(const sam3_model& model,
                                              const sam3_video_params& params);

/* Change the text prompt of an existing tracker (empty disables text
** detection; the tracker re-reads it on every frame). */
SAM3_API void sam3_tracker_set_text_prompt(sam3_tracker& tracker,
                                           const std::string& text);

/* Encode a frame, detect objects, and update tracked instances. */
SAM3_API sam3_result sam3_track_frame(sam3_tracker& tracker,
                                      sam3_state& state,
                                      const sam3_model& model,
                                      const sam3_image& frame);

/* Refine a tracked instance with interactive point prompts. */
SAM3_API bool sam3_refine_instance(sam3_tracker& tracker,
                                   sam3_state& state,
                                   const sam3_model& model,
                                   int instance_id,
                                   const std::vector<sam3_point>& pos_points,
                                   const std::vector<sam3_point>& neg_points);

/*
** Add a new instance to the tracker from PVS prompts (points/box) on the
** current frame.  The image must already be encoded (via sam3_track_frame
** or sam3_encode_image).  Returns assigned instance_id, or -1 on failure.
*/
SAM3_API int sam3_tracker_add_instance(sam3_tracker& tracker,
                                       sam3_state& state,
                                       const sam3_model& model,
                                       const sam3_pvs_params& pvs_params);

/*
** Add a new instance to the tracker from an existing binary mask on the
** current frame, bypassing the PVS prompt encoder / mask decoder.  The image
** must already be encoded (via sam3_track_frame, sam3_propagate_frame, or
** sam3_encode_image).  The mask (0/255, any resolution) is resampled to the
** tracker's memory resolution and written straight into the memory bank as a
** conditioning frame, so propagation tracks exactly the supplied mask rather
** than a mask re-derived from points/box.
**
** The prompt path produces no SAM decoder token, so the object pointer is
** instead obtained by running one propagation decode against the just-written
** conditioning memory (the same SAM token every tracked frame yields) and
** projecting it; this gives propagation a real appearance cue to re-localize
** with.  obj_score is the confidence stored for the seed frame (1.0 = fully
** trusted user mask).
**
** Returns the assigned instance_id, or -1 on failure (empty mask).
*/
SAM3_API int sam3_tracker_add_instance_from_mask(sam3_tracker& tracker,
                                                 sam3_state& state,
                                                 const sam3_model& model,
                                                 const sam3_mask& mask,
                                                 float obj_score = 1.0f);

/* Return the current frame index of the tracker. */
SAM3_API int sam3_tracker_frame_index(const sam3_tracker& tracker);

/* Reset the tracker, clearing all instances and memory. */
SAM3_API void sam3_tracker_reset(sam3_tracker& tracker);

/*
** ── Visual-Only Video Tracking ──────────────────────────────────────────
*/

struct sam3_visual_track_params {
    float assoc_iou_threshold = 0.1f;
    int max_keep_alive = 30;
    int recondition_every = 16;
    int fill_hole_area = 16;
};

/*
** Create a tracker for visual-only models.  Instances are added manually
** via sam3_tracker_add_instance().
*/
SAM3_API sam3_tracker_ptr sam3_create_visual_tracker(
        const sam3_model& model, const sam3_visual_track_params& params);

/*
** Propagate all tracked instances to the next frame (no detection step).
** The image is encoded, then each tracked instance is propagated via
** memory attention + SAM mask decode, and the memory bank is updated.
*/
SAM3_API sam3_result sam3_propagate_frame(sam3_tracker& tracker,
                                          sam3_state& state,
                                          const sam3_model& model,
                                          const sam3_image& frame);

/*
** ── Utility ─────────────────────────────────────────────────────────────
*/

/*****************************************************************************
** Test and Debug API
**
** Standalone tokenizer and debug utilities.
** These functions are intended for testing and development only.
*****************************************************************************/

SAM3_API bool sam3_test_load_tokenizer(const std::string& model_path);
SAM3_API std::vector<int32_t> sam3_test_tokenize(const std::string& text);

/*
** ── Debug ────────────────────────────────────────────────────────────────
*/

/* Query metadata for a named state tensor without dumping its payload. */
SAM3_API bool sam3_get_state_tensor_info(const sam3_state& state,
                                         const std::string& tensor_name,
                                         sam3_tensor_info& info);

/* Query metadata for a named model tensor. */
SAM3_API bool sam3_get_model_tensor_info(const sam3_model& model,
                                         const std::string& tensor_name,
                                         sam3_tensor_info& info);

/*
** Encode an image from pre-preprocessed float data (CHW layout, already
** resized and normalized).  Bypasses C++ preprocessing so that numerical
** comparisons against the Python reference are not polluted by resize
** implementation differences.
*/
SAM3_API bool sam3_encode_image_from_preprocessed(sam3_state& state,
                                                  const sam3_model& model,
                                                  const float* chw_data,
                                                  int img_size);

/*
** Test-only: run ONLY the ViT encoder from preprocessed float data and keep
** only the requested intermediate tensors alive for dumping/comparison.
*/
SAM3_API bool sam3_encode_vit_from_preprocessed_selective(
        sam3_state& state,
        const sam3_model& model,
        const float* chw_data,
        int img_size,
        const std::vector<std::string>& output_tensors);

/*
** Test-only: run the exact ViT prefix up to the tensor entering block 0
** (patch embed + pos embed + ln_pre).
*/
SAM3_API bool sam3_test_run_vit_block0_input(const sam3_model& model,
                                             const float* chw_data,
                                             int img_size,
                                             std::vector<float>& output_data,
                                             int64_t output_ne[4],
                                             int n_threads = 4);

/*
** Test-only: run an exact ViT prefix sub-stage on the real model tensors using
** the model's backend. PATCH_EMBED expects image input [W,H,3,1]. Later stages
** expect feature input [E,W,H,1] in ggml 4D layout.
*/
SAM3_API bool sam3_test_run_vit_prefix_stage(const sam3_model& model,
                                             sam3_vit_prefix_stage stage,
                                             const float* input_data,
                                             const int64_t input_ne[4],
                                             std::vector<float>& output_data,
                                             int64_t output_ne[4],
                                             int n_threads = 4);
SAM3_API bool sam3_test_run_patch_mulmat_host_ref(
        const sam3_model& model,
        const float* input_data,
        const int64_t input_ne[4],
        bool use_double_accum,
        std::vector<float>& output_data,
        int64_t output_ne[4]);
SAM3_API bool sam3_test_run_vit_block_linear_host_ref(
        const sam3_model& model,
        int block_idx,
        sam3_vit_block_stage stage,
        const float* input_data,
        const int64_t input_ne[4],
        bool use_double_accum,
        std::vector<float>& output_data,
        int64_t output_ne[4]);

/*
** Test-only: run an exact ViT block sub-stage on the real model tensors using
** the model's backend. The input is always provided as F32 in ggml 4D layout.
*/
SAM3_API bool sam3_test_run_vit_block_stage(const sam3_model& model,
                                            int block_idx,
                                            sam3_vit_block_stage stage,
                                            const float* input_data,
                                            const int64_t input_ne[4],
                                            std::vector<float>& output_data,
                                            int64_t output_ne[4],
                                            int n_threads = 4);
