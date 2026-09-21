// e2e pipeline options: every knob is an explicit field filled by the CLI
// from argv flags. The pipeline reads no process environment, keeping the
// stage contracts discoverable, unit-testable and free of leaked-environment
// failure modes.
#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace sam3d {

// Stage indices reported through E2eOptions::progress. They mirror the
// aicore_sam3d progress contract (stage-level events; step/total optional).
enum E2eStage {
    kStageCondition = 2,
    kStageSsFlow = 3,
    kStageSsDecode = 4,
    kStageSlatFlow = 5,
    kStageGsDecode = 6,
    kStageMeshDecode = 7,
};

struct E2eOptions {
    std::string models_dir;
    std::string cond_dir;
    std::string noise_dir;
    std::string dbg_dir;
    std::string out_ply;
    std::string out_pbr;
    std::string out_mesh_vertices;
    std::string out_mesh_faces;
    std::string out_pose;
    std::string out_dtype_contract;
    std::string backend = "auto";
    int threads = 8;
    unsigned seed = 0;
    // per-stage weight formats (empty inherits `dtype`)
    std::string dtype = "f32";
    std::string ss_dtype;
    std::string ss_decoder_dtype;
    std::string slat_dtype;
    std::string gs_dtype;
    std::string mesh_dtype;
    // stage selection ("" = full pipeline)
    std::string stage;
    // condition chain
    bool skip_cond = false;          // fused tokens from SAMT
    bool fuser_only = false;         // fuser graph, tokens external
    std::string ss_cond_path;
    std::string slat_cond_path;
    // Condition chain: the image-to-3D production path always sets this; the
    // e2e command exposes it as --cond-manual-attention for parity bisects.
    bool cond_manual_attention = false;
    bool dino_dbg = false;
    std::string dino_dbg_out;
    std::string debug_stage;
    // SS sampling
    int ss_steps = 25;
    bool ss_flow_only = false;
    bool ss_strict_attention = true;
    bool verify = false;
    // coordinates
    bool reference_coords = false;
    std::string coords_path;
    // SLat sampling
    // 25 steps is the repository's trajectory-gate contract; the official
    // deployment pipeline (pipeline.yaml) runs the distilled 12-step schedule,
    // so the step count is configurable for the deployment-caliber A/B.
    int slat_steps = 25;
    bool dump_slat_steps = false;
    bool slat_flow_only = false;
    bool vulkan_ss_table_cache = false;
    bool vulkan_table_cache = false;
    int debug_slat_forwards = 0;
    bool debug_once = false;
    std::string debug_slat_output;
    // A/B bisect: quantized projection weights consumed directly
    bool keep_quant_gemm = false;
    // A/B bisect: portable attention in the GS decoder even on CUDA
    bool gs_portable_attention = false;
    // Vulkan/CPU path: PyTorch CUDA Philox distribution-block contract
    // obtained from `sam3d-cli rng-dump` (0 = derive from the legacy env,
    // required to be set explicitly through --philox-blocks).
    uint32_t philox_blocks = 0;
    // Session reuse: when set, every stage borrows this backend instead of
    // creating and destroying its own. The caller owns the backend and must
    // outlive the cmd_e2e call; stage weights are still uploaded and released
    // per stage exactly as with owned backends. A Backend* is stored raw to
    // keep this header dependency-free.
    void* shared_backend = nullptr;

    // Optional stage-level progress reporting. Never invoked from multiple
    // threads: cmd_e2e is single-threaded across its stage boundaries.
    std::function<void(int stage, int step, int total)> progress;
};

int cmd_e2e(const E2eOptions& opt);

}  // namespace sam3d
