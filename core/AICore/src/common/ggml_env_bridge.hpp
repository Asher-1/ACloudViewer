// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Explicit bridge for ggml-side environment configuration.
//
// ggml upstream (GGML_VK_*) and our ggml patches (RMBG_VK_*,
// RMBG_CUDA_CONV_TF32) read a handful of environment variables when a backend
// instance is created — there is no runtime API for them. This bridge is the
// ONLY place in AICore that writes those variables:
//
//   explicit options -> GgmlEnvOverrides -> apply_ggml_env_overrides()
//
// The direction of control is "explicit interface drives env", never "env
// drives logic". AICore's own code paths read no environment variables.
//
// Semantics per field (matching the historical setenv/unsetenv behavior):
//   nullopt / nullopt string -> leave the variable untouched (shell wins)
//   true / ""                -> set ("1" for bools; "" clears the string var)
//   false / "value"          -> unset / set to "value"
//
// Application is immediate and process-global (last writer wins), exactly
// like the setenv calls it replaces. ggml snapshots these variables when a
// backend instance is CREATED, so for deterministic results a task must
// apply its overrides before its first context creation; applying after the
// backends were loaded prints a warning because existing instances keep
// their snapshot.

#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace aicore {

struct GgmlEnvOverrides {
    // ggml-vulkan instance switches (upstream interface).
    std::optional<bool> vk_disable_f16;
    std::optional<bool> vk_disable_coopmat;
    std::optional<bool> vk_disable_coopmat2;
    std::optional<bool> vk_disable_integer_dot_product;

    // ggml-metal instance switches (upstream interface, macOS only).
    std::optional<bool> metal_graph_optimize_disable;
    std::optional<bool> metal_fusion_disable;

    // rmbg_merged patch switches (read by the patched custom ops).
    std::optional<bool> rmbg_vk_scalar_direct_conv;
    std::optional<std::string> rmbg_vk_coopmat_matmul;  // nullopt = untouched,
    // "" = clear, non-empty = set
    std::optional<bool> rmbg_cuda_conv_tf32;

    // cuBLAS TF32 switch. NOTE: process-global by nature (same as the
    // historical setenv) — it also affects any other cuBLAS user in the
    // process.
    std::optional<bool> nvidia_tf32_override;
};

/** Write the overrides into the environment immediately. Warns when the
 *  ggml backends have already been loaded (existing backend instances keep
 *  their variable snapshot; new instances see the new values). */
void apply_ggml_env_overrides(const GgmlEnvOverrides& overrides);

/** Snapshot of ggml-side variables for callers that must scope an override
 *  to a backend-creation window (see restore_ggml_env_snapshot). Env reads
 *  stay inside this bridge. */
struct GgmlEnvSnapshot {
    // key -> (was_set, value_when_snapshotted)
    std::vector<std::pair<std::string, std::pair<bool, std::string>>> saved;
};
GgmlEnvSnapshot take_ggml_env_snapshot(const std::vector<std::string>& keys);

/** Restore a snapshot taken before apply_ggml_env_overrides(): variables that
 *  were unset go back to unset, set ones get their old value back. */
void restore_ggml_env_snapshot(const GgmlEnvSnapshot& snapshot);

/** Internal: records that ggml backends have been registered, so later
 *  apply_ggml_env_overrides() calls can warn about the snapshot semantics.
 *  Called by ggml_common::load_backends_once(). */
void mark_ggml_backends_loaded();

}  // namespace aicore
