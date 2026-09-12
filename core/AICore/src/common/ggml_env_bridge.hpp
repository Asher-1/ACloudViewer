// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Explicit bridge for ggml-side environment configuration.
//
// ggml upstream (GGML_VK_*, GGML_METAL_*) reads a handful of environment
// variables when a backend instance is created — there is no runtime API for
// them. This bridge is the ONLY place in AICore that writes those variables:
//
//   explicit options -> GgmlEnvOverrides -> apply_ggml_env_overrides()
//
// The direction of control is "explicit interface drives env", never "env
// drives logic". AICore's own code paths read no environment variables, and
// task behavior must never flow through this bridge: the RMBG math profile
// (formerly translated here into RMBG_VK_* / RMBG_CUDA_CONV_TF32 variables)
// is now carried as explicit GraphOptions and baked into the graph as
// output-name marks the ggml patch routes on.
//
// Semantics per field (matching the historical setenv/unsetenv behavior):
//   nullopt          -> leave the variable untouched (shell wins)
//   true             -> set ("1" for bools)
//   false            -> unset
//
// Application is immediate and process-global (last writer wins), exactly
// like the setenv calls it replaces. ggml snapshots these variables when a
// backend instance is CREATED, so for deterministic results a caller must
// apply its overrides before its first context creation; applying after the
// backends were loaded prints a warning because existing instances keep
// their snapshot.
//
// Task modules never touch this header (enforced by
// tests/check_no_env_getenv.sh); they use plain option values only.

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

/** Process-wide Vulkan runtime defaults, applied once before any ggml
 *  device is initialized (ggml-vulkan snapshots instance-level variables at
 *  first use). Enables the GGML_VK_ALLOW_SYSMEM_FALLBACK tier: when a
 *  device allocation fails mid-run — VRAM exhausted by the desktop, the
 *  host 3D viewport, or a larger-than-expected activation — the affected
 *  buffer degrades to host-visible memory instead of throwing. Normal runs
 *  never touch the fallback tier, so this only turns "failed run" into
 *  "slower run". An explicit shell setting of the variable always wins. */
void apply_vulkan_runtime_defaults();

/** Route ggml backend warnings/errors (memory fallbacks, allocation
 *  failures, device issues) into the AICore application log. Without this
 *  bridge ggml only prints to stderr, which GUI users never see — the
 *  Vulkan host-memory fallback in particular is fully silent there and was
 *  indistinguishable from a hang. Installed once in
 *  ggml_common::load_backends_once() before any backend registration; INFO
 *  and below stay on stderr so the app log keeps a low noise floor. */
void install_ggml_log_bridge();

/** Internal: records that ggml backends have been registered, so later
 *  apply_ggml_env_overrides() calls can warn about the snapshot semantics.
 *  Called by ggml_common::load_backends_once(). */
void mark_ggml_backends_loaded();

}  // namespace aicore
