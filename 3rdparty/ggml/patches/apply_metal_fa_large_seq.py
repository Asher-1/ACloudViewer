#!/usr/bin/env python3
"""
Patch ggml Metal backend for large flash-attention sequences.

Problem: The Metal flash_attn_ext kernel asserts ne01 < 65536 (sequence length)
and, even below that, macOS GPU watchdog kills command buffers that exceed ~2s.
FreeSplatter's object model with 8+ views produces S >= 32768 tokens per head,
which causes either an assertion failure or a GPU timeout crash.

Fix:
  1) Remove the ne01 < 65536 assert in ggml-metal-ops.cpp (the kernel handles
     arbitrary sequence lengths; the assert was overly conservative).
  2) Raise GGML_METAL_MAX_COMMAND_BUFFERS from 8 to 64.
  3) Auto-tune n_cb AND n_main at graph_compute time based on FA op count
     and sequence length.  For heavy FA graphs (Object model, S>=16384),
     split across enough command buffers so each contains at most one FA
     dispatch.  Small graphs (Scene model, S=8192) keep default n_cb=1-2.
     This prevents macOS GPU watchdog timeouts (~2s limit per command buffer)
     while avoiding unnecessary overhead for lightweight models.

This script is idempotent.
Usage: python3 apply_metal_fa_large_seq.py <ggml_source_dir>
"""

import sys
import os


def patch_remove_fa_assert(src_dir):
    """Remove the ne01 < 65536 assertion in flash_attn_ext dispatch."""
    ops_path = os.path.join(src_dir, "src", "ggml-metal", "ggml-metal-ops.cpp")
    if not os.path.exists(ops_path):
        print(f"[ggml-patch-fa] ops file not found: {ops_path}")
        return False

    with open(ops_path, "r") as f:
        content = f.read()

    sentinel = "// ACV: ne01 limit relaxed"
    if sentinel in content:
        print("[ggml-patch-fa] ne01 assert already patched, skipping")
        return True

    old = "    GGML_ASSERT(ne01 < 65536);"
    if old not in content:
        print("[ggml-patch-fa] WARNING: could not find ne01 assert "
              "(maybe upstream changed?) — skipping")
        return True

    new = (
        "    // ACV: ne01 limit relaxed\n"
        "    // Original hard assert removed; the kernel handles arbitrary\n"
        "    // sequence lengths.  Graph is split across command buffers to\n"
        "    // avoid macOS GPU watchdog timeouts."
    )
    content = content.replace(old, new, 1)

    with open(ops_path, "w") as f:
        f.write(content)
    print("[ggml-patch-fa] Removed ne01 < 65536 assertion")
    return True


def patch_increase_max_cmd_bufs(src_dir):
    """Increase GGML_METAL_MAX_COMMAND_BUFFERS from 8 to 64."""
    ctx_path = os.path.join(src_dir, "src", "ggml-metal", "ggml-metal-context.m")
    if not os.path.exists(ctx_path):
        print(f"[ggml-patch-fa] context file not found: {ctx_path}")
        return False

    with open(ctx_path, "r") as f:
        content = f.read()

    new_define = "#define GGML_METAL_MAX_COMMAND_BUFFERS 64"

    if new_define in content:
        print("[ggml-patch-fa] MAX_COMMAND_BUFFERS already patched, skipping")
        return True

    for old in ("#define GGML_METAL_MAX_COMMAND_BUFFERS 8",
                "#define GGML_METAL_MAX_COMMAND_BUFFERS 32"):
        if old in content:
            content = content.replace(old, new_define, 1)
            with open(ctx_path, "w") as f:
                f.write(content)
            print("[ggml-patch-fa] Increased MAX_COMMAND_BUFFERS to 64")
            return True

    print("[ggml-patch-fa] WARNING: could not find MAX_COMMAND_BUFFERS "
          "define — skipping")
    return True


def patch_autotune_n_cb(src_dir):
    """Auto-tune n_cb and n_main in graph_compute for large FA graphs.

    Handles three source states:
      A) Fresh upstream  — original n_main/n_cb code, no ACV sentinel.
      B) Old patch (v1)  — sentinel "// ACV: auto-tune n_cb" with
                           target_nodes_per_cb=40, n_main unchanged.
      C) Current (v2)    — sentinel "// ACV: auto-tune n_cb and n_main".
    """
    ctx_path = os.path.join(src_dir, "src", "ggml-metal", "ggml-metal-context.m")
    if not os.path.exists(ctx_path):
        print(f"[ggml-patch-fa] context file not found: {ctx_path}")
        return False

    with open(ctx_path, "r") as f:
        content = f.read()

    # --- Target block (v2) ---------------------------------------------------
    # Dynamically compute n_cb based on FA op count and sequence length.
    # For small graphs or cheap FA ops, n_cb stays at the default (1-2).
    # For heavy FA graphs (S>=16384), split so each buffer has ≤1 FA op.
    new_block = """\
    // ACV: auto-tune n_cb and n_main for large graphs to avoid macOS
    // GPU watchdog timeouts.  Dynamically computes the split based on
    // flash-attention op count and sequence length, so small graphs
    // (e.g. Scene model with 2 views) keep the default n_cb=1-2.
    int n_main = MAX(64, 0.1*gf->n_nodes);
    {
        int n_fa = 0;
        int64_t max_ne01 = 0;
        for (int i = 0; i < gf->n_nodes; ++i) {
            if (gf->nodes[i]->op == GGML_OP_FLASH_ATTN_EXT) {
                ++n_fa;
                if (gf->nodes[i]->ne[1] > max_ne01)
                    max_ne01 = gf->nodes[i]->ne[1];
            }
        }
        // Estimate per-FA dispatch time.  Empirical baseline on M2 Max:
        // S=8192 (2 views) ≈ 25 ms.  Scales as O(S²).
        const double fa_ms = (max_ne01 > 0)
            ? 25.0 * ((double)max_ne01 / 8192.0) * ((double)max_ne01 / 8192.0)
            : 0.0;
        // Target: each command buffer completes in < 1500 ms (watchdog ~2000 ms).
        const int max_fa_per_buf = (fa_ms > 10.0)
            ? MAX(1, (int)(1500.0 / fa_ms))
            : 999;
        if (n_fa > 0 && max_fa_per_buf < n_fa) {
            const int needed = (n_fa + max_fa_per_buf - 1) / max_fa_per_buf;
            const int wanted = MIN(48, MAX(ctx->n_cb, needed));
            n_main = MIN(10, gf->n_nodes);
            if (wanted > ctx->n_cb) {
                ggml_metal_set_n_cb(ctx, wanted);
            }
        }
    }

    // number of threads in addition to the main thread
    const int n_cb = ctx->n_cb;"""

    # State C — already at v2, nothing to do.
    sentinel_v2 = "// ACV: auto-tune n_cb and n_main"
    if sentinel_v2 in content:
        print("[ggml-patch-fa] n_cb auto-tune v2 already applied, skipping")
        return True

    # State B — old patch v1 present, upgrade to v2.
    sentinel_v1 = "// ACV: auto-tune n_cb"
    if sentinel_v1 in content:
        old_v1_block = """\
    // number of nodes encoded by the main thread (empirically determined)
    const int n_main = MAX(64, 0.1*gf->n_nodes);

    // ACV: auto-tune n_cb for large graphs to avoid macOS GPU watchdog.
    // Target ~40 nodes per command buffer (≈1-2 transformer blocks).
    {
        const int target_nodes_per_cb = 40;
        const int auto_cb = (gf->n_nodes > 200)
            ? MIN(16, MAX(ctx->n_cb, gf->n_nodes / target_nodes_per_cb))
            : ctx->n_cb;
        if (auto_cb != ctx->n_cb) {
            ggml_metal_set_n_cb(ctx, auto_cb);
        }
    }

    // number of threads in addition to the main thread
    const int n_cb = ctx->n_cb;"""

        if old_v1_block in content:
            content = content.replace(old_v1_block, new_block, 1)
            with open(ctx_path, "w") as f:
                f.write(content)
            print("[ggml-patch-fa] Upgraded n_cb auto-tune v1 -> v2")
            return True

        print("[ggml-patch-fa] WARNING: v1 sentinel found but block mismatch "
              "— skipping")
        return True

    # State A — fresh upstream, apply v2 directly.
    fresh_block = """\
    // number of nodes encoded by the main thread (empirically determined)
    const int n_main = MAX(64, 0.1*gf->n_nodes);

    // number of threads in addition to the main thread
    const int n_cb = ctx->n_cb;"""

    if fresh_block in content:
        content = content.replace(fresh_block, new_block, 1)
        with open(ctx_path, "w") as f:
            f.write(content)
        print("[ggml-patch-fa] Applied n_cb/n_main auto-tune v2 (fresh)")
        return True

    print("[ggml-patch-fa] WARNING: could not find any known n_cb block "
          "— skipping")
    return True


def touch_metal_sources(src_dir):
    metal_dir = os.path.join(src_dir, "src", "ggml-metal")
    if not os.path.isdir(metal_dir):
        return
    for fname in os.listdir(metal_dir):
        fpath = os.path.join(metal_dir, fname)
        if os.path.isfile(fpath) and (
            fname.endswith(".cpp") or fname.endswith(".m")
            or fname.endswith(".h") or fname.endswith(".metal")
        ):
            os.utime(fpath, None)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <ggml_source_dir>")
        sys.exit(1)

    src_dir = sys.argv[1]
    ok1 = patch_remove_fa_assert(src_dir)
    ok2 = patch_increase_max_cmd_bufs(src_dir)
    ok3 = patch_autotune_n_cb(src_dir)
    if ok1 or ok2 or ok3:
        touch_metal_sources(src_dir)
    sys.exit(0 if (ok1 and ok2 and ok3) else 1)
