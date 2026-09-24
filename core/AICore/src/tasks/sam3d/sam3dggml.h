// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SAM 3D Objects inference on ggml — public C API surface (C++17).
//
// The library loads GGUF exports of the official SAM 3D Objects generative
// checkpoints and runs the conditioned native graph:
//   DINOv2 + PointPatchEmbed conditioners -> SparseStructure DiT -> SS decoder
//   -> SLat DiT -> Gaussian decoder -> PLY export.
//
// The conditioned entry point accepts a directory produced by
// scripts/dump_e2e_stages.py. `run_image_to_3d` is the native raw-image
// counterpart: it executes MoGe, preprocessing, GGML generation, and the
// optional native PBR branch without invoking Python at runtime.
#ifndef SAM3D_GGML_H
#define SAM3D_GGML_H

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "e2e_options.hpp"  // Sam3dArtifacts

namespace aicore {
namespace runtime {
class BackendLease;
}  // namespace runtime
}  // namespace aicore

namespace sam3d {

struct RgbaImage;  // asset_io.hpp; in-memory bridge input (see below)

struct CliOptions {
    // model directory containing the converted GGUF files
    std::string models_dir = "cpp_ggml/models/gguf";
    // backend selection: auto | cpu | cuda | vulkan
    std::string backend = "auto";
    int n_threads = 8;
    int ss_steps = 25;
    int slat_steps = 25;
    int ss_cfg_strength = 7;
    float ss_cfg_strength_pm = 0.0f;
    int slat_cfg_strength = 5;
    int seed = 42;
    // Reserved for a future native condition-input API. The current session
    // consumes condition_dir exclusively.
    std::string pointmap_bin;
    // Directory produced by scripts/dump_e2e_stages.py.  This is the
    // deterministic, parity-testable input form for the C++ session.
    std::string condition_dir;
    // Reserved source-image metadata. These fields are not consumed by the
    // current native session; image preprocessing happens before condition_dir.
    std::string image_png;
    std::string mask_png;
    // outputs
    std::string out_ply;  // gaussian splat PLY output
    std::string
            out_gs_bin;    // raw gaussian parameters for python-side rendering
    std::string out_meta;  // json with timing + counts
    std::string dump_dir;  // if set: dump intermediate tensors for parity
    // benchmarking
    bool bench = false;
    int warmup = 5;
    int iters = 20;
};

struct RunStats {
    double load_ms = 0;
    double pointmap_ms = 0;
    double condition_ms = 0;
    double ss_flow_ms = 0;
    double ss_decode_ms = 0;
    double slat_flow_ms = 0;
    double gs_decode_ms = 0;
    int n_active_voxels = 0;
    int n_gaussians = 0;
    // Session batch diagnostics: true when this request skipped the MoGe
    // forward because the RGB content matched the previous request.
    bool moge_pointmap_reused = false;
};

struct RunOutput {
    RunStats stats;
    std::vector<int> coords;  // n_voxels * 3
    std::vector<float> gs;    // n_gaussians packed attributes (see gs_io.h)
};

// Run the full pipeline. Returns nullptr-terminated error string on failure
// inside the returned object (ok == false).
struct RunResult {
    bool ok = false;
    std::string error;
    RunOutput output;
};

// Native raw-image entry point. The image is always required; a supplied mask
// replaces its alpha channel with the official binary-mask semantics. A
// textured PBR GLB additionally requires a CUDA build with the explicitly
// licensed native PBR dependencies enabled. Vulkan callers can instead export
// the raw FlexiCubes mesh and hand it, together with out_ply, to a separate
// CUDA pbr-assemble process.
struct ImageTo3DOptions {
    std::string models_dir = "cpp_ggml/models/gguf";
    std::string moge_model = "cpp_ggml/models/gguf/moge_vitl-f16.gguf";
    std::string backend = "auto";
    std::string dtype = "f16";
    std::string image_path;
    std::string mask_path;
    std::string out_ply;
    std::string out_pbr;
    // Raw FlexiCubes mesh in SAMT layout ([3, V] F32 and [3, F] I32). These
    // must be requested together. They are the stable Vulkan-to-CUDA PBR
    // interchange boundary, before cleanup, UV generation and texture baking.
    std::string out_mesh_vertices;
    std::string out_mesh_faces;
    // Optional JSON representation of the official ScaleShiftInvariant pose.
    // The final GLB remains in decoder-local coordinates, matching the Python
    // pipeline; this preserves pose as a separately reusable output.
    std::string out_pose;
    // Optional JSON summary of the dtypes and operations present in every
    // native end-to-end graph constructed for this request.
    std::string out_dtype_contract;
    // Optional in-memory artifact sink (C API path): splat centers/colors and
    // the FlexiCubes mesh are delivered here; out_ply and the mesh SAMT
    // exports become optional when it is set.
    Sam3dArtifacts* artifacts = nullptr;
    // Run the FlexiCubes mesh stage for the artifact sink without mesh file
    // exports (zero file IO). Ignored when artifacts is null.
    bool decode_mesh = false;
    // Capture the scene-composer interchange attributes (PLY-semantic splat
    // rows + the official pose receipt) into the artifact sink. Ignored when
    // artifacts is null.
    bool scene_attributes = false;
    // Optional immutable official stage directory containing initial SS and
    // SLat noise. This is an accuracy-diagnostic input, not a production
    // sampling mode: a normal request must generate its own noise from seed.
    std::string noise_dir;
    // If empty, the implementation owns a unique temporary condition
    // directory and removes it after the session finishes.
    std::string conditions_out;
    int n_threads = 8;
    int seed = 42;
    // Explicit F32 SS attention for an accuracy-first replay. The official
    // SS backbone is an F32 model, so this is the default contract;
    // --ss-attention normal (CLI) opts out for A/B bisects.
    bool strict_ss_attention = true;
    // A/B bisect: portable attention in the GS decoder even on CUDA.
    bool gs_portable_attention = false;
    // Vulkan/CPU path: PyTorch CUDA Philox distribution-block contract from
    // `sam3d-cli rng-dump`. 0 falls back to the documented environment
    // variable; a positive value makes the request self-contained.
    unsigned philox_blocks = 0;
    // Trajectory length for the SS and SLat flow samplers. The repository
    // trajectory gate fixes 25 steps for both; the official deployment
    // pipeline (pipeline.yaml) distills SS (ShortCut) to 2 steps and SLat to
    // 12, which is the caliber the scene timings can be compared against.
    int ss_steps = 25;
    int slat_steps = 25;
    // Hot-timing diagnostic: force the MoGe forward on every request even
    // when the RGB content matches the previous one. Timing runs set this so
    // the measured latency includes the real per-request MoGe cost instead
    // of a cache hit that a production batch would take.
    bool disable_moge_pointmap_cache = false;

    // Optional stage-level progress reporting (single-threaded invocation,
    // fired from the session thread while the request runs).
    std::function<void(int stage, int step, int total)> progress;

    // In-memory input bridge (ACloudViewer borrowed image view). When set,
    // image_path/mask_path are ignored: the RGBA pixels are used directly
    // and the alpha channel already carries the official binary-mask
    // semantics. Numerics are identical to the decoded-file path.
    std::shared_ptr<const RgbaImage> image_override;
};

// Execute the full native conditioned generation graph and write Gaussian PLY.
// opts.condition_dir and opts.out_ply are required.
RunResult run_pipeline(const CliOptions& opts);

// Execute image -> MoGe point map -> native condition preprocessing -> GGML
// generation -> optional native PBR export. No Python or Torch process is
// started by this API.
RunResult run_image_to_3d(const ImageTo3DOptions& opts);

// Reusable session for multi-mask batches over one source image. init() loads
// the MoGe weights once and fixes the model/backend/dtype/thread settings;
// every run() serves one request with complete isolation (fresh noise from
// its seed, request-scoped activations and outputs). Repeated requests with
// the same RGB content reuse the RGB-only MoGe point map; the mask merge and
// condition preprocessing always run per request. The single-request
// run_image_to_3d above is a thin one-shot session wrapper with identical
// numerics.
class ImageTo3DSession {
public:
    ImageTo3DSession();
    ~ImageTo3DSession();
    ImageTo3DSession(const ImageTo3DSession&) = delete;
    ImageTo3DSession& operator=(const ImageTo3DSession&) = delete;

    // Prepares the shared resources from the request template. The models,
    // backend, dtype and thread settings recorded here are fixed for the
    // session; run() rejects requests that change any of them.
    bool init(const ImageTo3DOptions& config, std::string& error);

    // Executes one request. image_path/mask_path/seed/output paths may vary
    // per request; models_dir/moge_model/backend/dtype/n_threads must match
    // the init configuration.
    RunResult run(const ImageTo3DOptions& opts);

    // Process-shared physical backend handles used by the session (ACloudViewer
    // runtime registry leases). The C ABI caller locks these around run().
    std::vector<aicore::runtime::BackendLease> leases() const;

    // Resolved backend name ("cuda", "vulkan", "cpu"), for reporting.
    std::string backend_name() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Version of the graph format emitted by the converter understood here.
inline constexpr const char* kGraphFormatVersion = "1";

}  // namespace sam3d

#endif  // SAM3D_GGML_H
