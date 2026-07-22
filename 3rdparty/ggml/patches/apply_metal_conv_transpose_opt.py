#!/usr/bin/env python3
"""
Patch ggml Metal conv_transpose_2d kernel for improved performance.

Problem: The original kernel dispatches one threadgroup per output pixel with
threads mapped to (KW, KH) and a serial IC loop + serial reduce. For large
tensors (DA3 DPT head: IC=256, stride=4, output ~684x684x256) this creates
~120M threadgroups and hangs the GPU on macOS due to command buffer timeouts.

Fix: Remap threads to parallelize over IC (input channels) with tree reduction.
Each threadgroup still produces one output pixel, but now up to 256 threads
cooperatively accumulate across IC with O(log n) shared memory reduction.

This script is idempotent: it checks whether the patch has already been applied.
Usage: python3 apply_metal_conv_transpose_opt.py <ggml_source_dir>
"""

import sys
import os

def patch_metal_shader(src_dir):
    metal_path = os.path.join(src_dir, "src", "ggml-metal", "ggml-metal.metal")
    if not os.path.exists(metal_path):
        print(f"[ggml-patch] Metal shader not found: {metal_path}")
        return False

    with open(metal_path, "r") as f:
        content = f.read()

    # Check if already patched
    if "for (int64_t in_c = tid; in_c < args.IC; in_c += num_threads)" in content:
        print("[ggml-patch] Metal conv_transpose_2d already patched, skipping")
        return True

    old_kernel_body = """\
    const int64_t kw = tpitg[0];
    const int64_t kh = tpitg[1];

    float v = 0.0f;

    for (int64_t in_c = 0; in_c < args.IC; in_c++) {
        int64_t in_y = out_y - kh;

        if (in_y < 0 || in_y % args.s0) continue;

        in_y /= args.s0;

        if (in_y >= args.IH) continue;

        int64_t in_x = out_x - kw;

        if (in_x < 0 || in_x % args.s0) continue;

        in_x /= args.s0;

        if (in_x >= args.IW) continue;

        const int64_t input_idx = (args.IW * args.IH) * in_c + (args.IW) * in_y + in_x;
        const int64_t kernel_idx = (args.KH * args.KW * args.OC) * in_c + (args.KH * args.KW) * out_c + (args.KW) * kh + kw;

        v += (float)src0[kernel_idx] * src1[input_idx];
    }

    const uint tid = tpitg.y * ntg.x + tpitg.x;
    shared_sum[tid] = v;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        const uint num_threads = ntg.x * ntg.y;
        for (uint i = 0; i < num_threads; i++) {
            total += shared_sum[i];
        }

        device float * dst_ptr = (device float *) (dst + out_x*args.nb0 + out_y * args.nb1 + out_c*args.nb2);
        dst_ptr[0] = total;
    }"""

    new_kernel_body = """\
    const int64_t tid = tpitg[0];
    const int64_t num_threads = ntg[0];

    float v = 0.0f;

    for (int64_t in_c = tid; in_c < args.IC; in_c += num_threads) {
        for (int64_t kh = 0; kh < args.KH; kh++) {
            int64_t in_y = out_y - kh;

            if (in_y < 0 || in_y % args.s0) continue;

            in_y /= args.s0;

            if (in_y >= args.IH) continue;

            for (int64_t kw = 0; kw < args.KW; kw++) {
                int64_t in_x = out_x - kw;

                if (in_x < 0 || in_x % args.s0) continue;

                in_x /= args.s0;

                if (in_x >= args.IW) continue;

                const int64_t input_idx = (args.IW * args.IH) * in_c + (args.IW) * in_y + in_x;
                const int64_t kernel_idx = (args.KH * args.KW * args.OC) * in_c + (args.KH * args.KW) * out_c + (args.KW) * kh + kw;

                v += (float)src0[kernel_idx] * src1[input_idx];
            }
        }
    }

    shared_sum[tid] = v;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int64_t s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        device float * dst_ptr = (device float *) (dst + out_x*args.nb0 + out_y * args.nb1 + out_c*args.nb2);
        dst_ptr[0] = shared_sum[0];
    }"""

    if old_kernel_body not in content:
        print("[ggml-patch] ERROR: could not find original kernel body in Metal shader")
        return False

    content = content.replace(old_kernel_body, new_kernel_body)
    with open(metal_path, "w") as f:
        f.write(content)
    print("[ggml-patch] Metal shader patched successfully")
    return True


def patch_ops_dispatch(src_dir):
    ops_path = os.path.join(src_dir, "src", "ggml-metal", "ggml-metal-ops.cpp")
    if not os.path.exists(ops_path):
        print(f"[ggml-patch] ops file not found: {ops_path}")
        return False

    with open(ops_path, "r") as f:
        content = f.read()

    if "nth_ic" in content:
        print("[ggml-patch] ops dispatch already patched, skipping")
        return True

    old_dispatch = """\
    // Metal requires buffer size to be multiple of 16 bytes
    const size_t smem = GGML_PAD(KW * KH * sizeof(float), 16);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, OW, OH, OC, KW, KH, 1);"""

    new_dispatch = """\
    // Threads per threadgroup parallelize over IC (input channels).
    // Tree reduction in shared memory replaces the old serial accumulation.
    const int32_t nth_ic = std::min(IC, 256);
    const size_t smem = GGML_PAD(nth_ic * sizeof(float), 16);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, OW, OH, OC, nth_ic, 1, 1);"""

    if old_dispatch not in content:
        print("[ggml-patch] ERROR: could not find original dispatch in ops file")
        return False

    content = content.replace(old_dispatch, new_dispatch)
    with open(ops_path, "w") as f:
        f.write(content)
    print("[ggml-patch] ops dispatch patched successfully")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <ggml_source_dir>")
        sys.exit(1)

    src_dir = sys.argv[1]
    ok1 = patch_metal_shader(src_dir)
    ok2 = patch_ops_dispatch(src_dir)
    sys.exit(0 if (ok1 and ok2) else 1)
