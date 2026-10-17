// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// WHITEBOX diagnostic (bisection tool, not a release gate): run the SAME
// world-family GGUF once on CPU and once on the GPU, then compare EVERY
// compute-graph node's output checksum to locate the FIRST divergent op.
//
// Assets (location-only env vars; unset => skip with 77):
//   AICORE_TEST_YOLO_GGUF           a single world/yoloe GGUF
//   AICORE_TEST_YOLO_IMAGE          probe image
//   AICORE_TEST_YOLO_CLASSES        comma-separated class list
//   AICORE_TEST_YOLO_TEXT_MODEL     text-encoder GGUF for the class list
//   AICORE_TEST_YOLO_PARITY_DEVICE  gpu device request (default "cuda")

#include <QImage>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml.h"
#include "tasks/yolo/yolo_clip_text_graph.hpp"
#include "tasks/yolo/yolo_gguf_loader.hpp"
#include "tasks/yolo/yolo_graph.hpp"
#include "tasks/yolo/yolo_image.hpp"
#include "tasks/yolo/yolo_mobileclip_graph.hpp"

namespace {

// Download one backend tensor and upcast to f32 (f16 flows store most
// inter-op values as GGML_TYPE_F16 — skip only non-float types). Returns an
// empty vector for unsupported types.
std::vector<float> to_f32(ggml_tensor* t) {
    std::vector<float> out;
    if (t == nullptr || t->buffer == nullptr) return out;  // not allocated
    const size_t n = (size_t)ggml_nelements(t);
    if (n == 0) return out;
    if (t->type == GGML_TYPE_F32) {
        out.resize(n);
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        out.resize(n);
        for (size_t i = 0; i < n; ++i) out[i] = ggml_fp16_to_fp32(tmp[i]);
    }
    return out;
}

std::vector<float> encode_classes(const char* gguf,
                                  const char* classes,
                                  const char* text_model,
                                  int* out_nc) {
    std::vector<std::string> names;
    std::string item;
    for (const char* p = classes;; ++p) {
        if (*p == ',' || *p == '\0') {
            names.push_back(item);
            item.clear();
            if (*p == '\0') break;
        } else {
            item.push_back(*p);
        }
    }
    if (names.empty()) return {};
    const size_t dim = 512;
    std::vector<float> embed(names.size() * dim, 0.0f);
    const bool yoloe = std::strstr(gguf, "yoloe") != nullptr;
    if (yoloe) {
        mobileclip::Session* ms = mobileclip::create_session(text_model, 0);
        if (ms == nullptr) return {};
        for (size_t i = 0; i < names.size(); ++i) {
            if (!mobileclip::encode_string(ms, names[i].c_str(),
                                           embed.data() + i * dim)) {
                mobileclip::free_session(ms);
                return {};
            }
        }
        mobileclip::free_session(ms);
    } else {
        clip::TextSession* cs = clip::text_create_session(text_model, 0);
        if (cs == nullptr) return {};
        for (size_t i = 0; i < names.size(); ++i) {
            if (!clip::text_encode_string(cs, names[i].c_str(),
                                          embed.data() + i * dim)) {
                clip::text_free_session(cs);
                return {};
            }
        }
        clip::text_free_session(cs);
    }
    *out_nc = (int)names.size();
    return embed;
}

yolo::Session* make_session(const char* gguf,
                            const char* device,
                            const std::vector<float>& embed,
                            int nc,
                            const yolo::LetterboxInfo& info) {
    yolo::SessionOptions opts;
    opts.threads = 4;
    opts.keep_all_ops = true;  // expose per-user-op outputs for bisection
    if (nc > 0) opts.world_nc = nc;
    yolo::Session* s = yolo::create_session(gguf, device, opts);
    if (s == nullptr) return nullptr;
    if (!embed.empty() && !yolo::session_set_text(s, embed.data())) {
        yolo::free_session(s);
        return nullptr;
    }
    if (!yolo::session_ensure_canvas(s, info.imgsz_w, info.imgsz_h)) {
        yolo::free_session(s);
        return nullptr;
    }
    return s;
}

}  // namespace

int main() {
    const char* gguf = std::getenv("AICORE_TEST_YOLO_GGUF");
    const char* image = std::getenv("AICORE_TEST_YOLO_IMAGE");
    const char* classes = std::getenv("AICORE_TEST_YOLO_CLASSES");
    const char* text_model = std::getenv("AICORE_TEST_YOLO_TEXT_MODEL");
    const char* device_env = std::getenv("AICORE_TEST_YOLO_PARITY_DEVICE");
    const std::string gpu_device = device_env != nullptr ? device_env : "cuda";
    if (gguf == nullptr || image == nullptr) {
        std::printf("[optrace] skipped: AICORE_TEST_YOLO_GGUF/IMAGE needed\n");
        return 77;
    }

    // Probe image -> letterbox canvas (imgsz from the model metadata).
    yolo::ModelMeta meta = yolo::read_gguf_meta(gguf);
    QImage img(QString::fromUtf8(image));
    if (img.isNull() || meta.imgsz <= 0) {
        std::printf("[optrace] failed to load image / metadata\n");
        return 1;
    }
    QImage rgb888 = img.convertToFormat(QImage::Format_RGB888);
    std::vector<uint8_t> rgb((size_t)rgb888.width() * rgb888.height() * 3);
    for (int y = 0; y < rgb888.height(); ++y) {
        std::memcpy(rgb.data() + (size_t)y * rgb888.width() * 3,
                    rgb888.constScanLine(y), (size_t)rgb888.width() * 3);
    }
    yolo::LetterboxInfo info;
    std::vector<float> canvas;
    yolo::letterbox_image(
            yolo::Image{rgb888.width(), rgb888.height(), rgb.data()},
            meta.imgsz, info, canvas);

    // Shared class embedding (encoded once on CPU; identical bytes for both
    // sessions).
    int nc = 0;
    std::vector<float> embed;
    if (classes != nullptr && classes[0] != '\0' && text_model != nullptr) {
        embed = encode_classes(gguf, classes, text_model, &nc);
        if (embed.empty()) {
            std::printf("[optrace] text encoding failed\n");
            return 1;
        }
    }

    yolo::Session* cpu = make_session(gguf, "cpu", embed, nc, info);
    yolo::Session* gpu =
            make_session(gguf, gpu_device.c_str(), embed, nc, info);
    if (cpu == nullptr || gpu == nullptr) {
        std::printf("[optrace] skipped: session load failed (%s / %s)\n",
                    cpu != nullptr ? "cpu ok" : "cpu failed",
                    gpu != nullptr ? "gpu ok" : "gpu failed");
        yolo::free_session(cpu);
        yolo::free_session(gpu);
        return 77;
    }
    if (!yolo::session_run(cpu, canvas.data()) ||
        !yolo::session_run(gpu, canvas.data())) {
        std::printf("[optrace] inference failed\n");
        yolo::free_session(cpu);
        yolo::free_session(gpu);
        return 1;
    }

    // Compare per USER-OP outputs (not graph nodes — the CUDA f16 flow
    // builds extra cast nodes, so node indices differ across devices).
    // EVERY op is printed (error-propagation profile); the tool reports the
    // FIRST op whose output leaves the tolerance band, but keeps scanning so
    // a rounding snowball is distinguishable from a single bad kernel.
    const size_t n_ops = cpu->model.ops.size();
    std::printf("[optrace] user ops: cpu=%zu gpu=%zu\n", n_ops,
                gpu->model.ops.size());
    int first_bad = -1;
    size_t n_compared = 0, n_null = 0;
    for (size_t i = 0; i < n_ops; ++i) {
        ggml_tensor* ta = cpu->op_values[i];
        ggml_tensor* tb = gpu->op_values[i];
        if (ta == nullptr || tb == nullptr) {
            ++n_null;
            continue;
        }
        const std::vector<float> ha = to_f32(ta);
        const std::vector<float> hb = to_f32(tb);
        if (ha.empty() || hb.empty()) continue;  // non-float type
        if (ha.size() != hb.size()) {
            std::printf(
                    "[optrace] op %zu (%s): element count differs "
                    "(%zu vs %zu)\n",
                    i, cpu->model.ops[i].type.c_str(), ha.size(), hb.size());
            if (first_bad < 0) first_bad = (int)i;
            continue;
        }
        double sa = 0, sb = 0;
        long long bad = 0;
        float mx = 0;
        for (size_t k = 0; k < ha.size(); ++k) {
            sa += ha[k];
            sb += hb[k];
            const float d = std::abs(ha[k] - hb[k]);
            if (d > mx) mx = d;
            if (!(std::isfinite(ha[k]) == std::isfinite(hb[k])) || d > 1.0f)
                ++bad;
        }
        const double denom =
                std::max(1e-9, std::max(std::abs(sa), std::abs(sb)));
        const double rel = std::abs(sa - sb) / denom;
        std::printf(
                "[optrace] op %3zu %-18s n=%9zu rel=%.3e maxdiff=%.4f "
                "big_errs=%lld%s\n",
                i, cpu->model.ops[i].type.c_str(), ha.size(), rel, mx, bad,
                (first_bad < 0 && (rel > 1e-2 || bad > 0))
                        ? "  <-- FIRST OUT OF BAND"
                        : "");
        if (first_bad < 0 && (rel > 1e-2 || bad > 0)) first_bad = (int)i;
        ++n_compared;
    }
    std::printf("[optrace] compared %zu/%zu ops (%zu null)\n", n_compared,
                n_ops, n_null);
    if (first_bad < 0) {
        std::printf("[optrace] all user-op outputs match\n");
    } else {
        std::printf("[optrace] FIRST DIVERGENCE at user-op %d (%s)\n",
                    first_bad, cpu->model.ops[first_bad].type.c_str());
    }

    // Also compare the SESSION-LEVEL outputs the C API actually consumes
    // (post f16->f32 cast on GPU). A mismatch here means the divergence
    // enters before postprocess; matching values here point INTO the
    // decode path (anchors/DFL/mask-coefficient slicing) instead.
    auto cmp_final = [&](const char* tag, ggml_tensor* a, ggml_tensor* b) {
        if (a == nullptr || b == nullptr) {
            std::printf("[optrace] final %s: missing tensor\n", tag);
            return;
        }
        if (ggml_nelements(a) != ggml_nelements(b)) {
            std::printf(
                    "[optrace] final %s: nelements differ (%lld vs %lld, "
                    "%s vs %s)\n",
                    tag, (long long)ggml_nelements(a),
                    (long long)ggml_nelements(b), ggml_type_name(a->type),
                    ggml_type_name(b->type));
            return;
        }
        std::vector<float> va = to_f32(a);
        std::vector<float> vb = to_f32(b);
        if (va.empty() || vb.empty() || va.size() != vb.size()) {
            std::printf("[optrace] final %s: unreadable (%s vs %s)\n", tag,
                        ggml_type_name(a->type), ggml_type_name(b->type));
            return;
        }
        double sa = 0, sb = 0;
        float mx = 0;
        long long bad = 0;
        for (size_t k = 0; k < va.size(); ++k) {
            sa += va[k];
            sb += vb[k];
            const float d = std::abs(va[k] - vb[k]);
            if (d > mx) mx = d;
            if (d > 1.0f || !(std::isfinite(va[k]) == std::isfinite(vb[k])))
                ++bad;
        }
        const double rel = std::abs(sa - sb) /
                           std::max(1e-9, std::max(std::abs(sa), std::abs(sb)));
        std::printf(
                "[optrace] final %-6s n=%9zu (%s) rel=%.3e maxdiff=%.4f "
                "big_errs=%lld\n",
                tag, va.size(), ggml_type_name(a->type), rel, mx, bad);
    };
    cmp_final("output", cpu->output, gpu->output);
    cmp_final("proto", cpu->output_proto, gpu->output_proto);

    // Layout/buffer forensics for the proto chain (which backend owns each
    // tensor, is the cast source contiguous, where does the op-value live).
    auto describe = [&](const char* tag, ggml_tensor* t) {
        if (t == nullptr) {
            std::printf("[optrace] %s: <null>\n", tag);
            return;
        }
        std::printf(
                "[optrace] %-14s type=%-4s ne=[%lld,%lld,%lld,%lld] "
                "cont=%d buf=%s\n",
                tag, ggml_type_name(t->type), (long long)t->ne[0],
                (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3],
                ggml_is_contiguous(t),
                t->buffer != nullptr ? ggml_backend_buffer_name(t->buffer)
                                     : "<none>");
    };
    std::printf("[optrace] proto chain: gpu f32->src[0] is the f16 op value\n");
    describe("cpu proto f32", cpu->output_proto);
    describe("gpu proto f32", gpu->output_proto);
    if (gpu->output_proto != nullptr && gpu->output_proto->src[0] != nullptr)
        describe("gpu proto f16", gpu->output_proto->src[0]);
    if (cpu->output_proto != nullptr && cpu->output_proto->src[0] != nullptr)
        describe("cpu proto f16", cpu->output_proto->src[0]);
    describe("cpu output", cpu->output);
    describe("gpu output", gpu->output);
    // Peek the first elements: an all-zero / bit-pattern garbage f32 proto
    // next to a correct f16 source means the cast node never executed.
    auto peek = [&](const char* tag, ggml_tensor* t, int k) {
        std::vector<float> v = to_f32(t);
        if (v.empty()) return;
        std::printf("[optrace] peek %-14s [0..%d):", tag, k);
        for (int i = 0; i < k && i < (int)v.size(); ++i)
            std::printf(" %.4f", v[i]);
        std::printf("\n");
    };
    peek("cpu proto f32", cpu->output_proto, 8);
    peek("gpu proto f32", gpu->output_proto, 8);
    if (gpu->output_proto != nullptr && gpu->output_proto->src[0] != nullptr)
        peek("gpu proto f16", gpu->output_proto->src[0], 8);
    // Which USER-OP produced the proto? Pointer-match it against op_values;
    // when it MISSES, walk the src chain — a cast fed by an INNER graph node
    // (not the stored op value) means the op-loop wrapped the branch output
    // in extra nodes on this backend.
    bool proto_ptr_matched = false;
    for (size_t i = 0; i < gpu->op_values.size(); ++i) {
        if (gpu->op_values[i] == gpu->output_proto->src[0]) {
            proto_ptr_matched = true;
            std::printf("[optrace] proto source = op %zu (%s)\n", i,
                        gpu->model.ops[i].type.c_str());
            peek("cpu opval", cpu->op_values[i], 8);
            peek("gpu opval", gpu->op_values[i], 8);
            break;
        }
    }
    if (!proto_ptr_matched) {
        std::printf(
                "[optrace] proto source is NOT a stored op value — inner "
                "graph node!\n");
        ggml_tensor* t = gpu->output_proto->src[0];
        for (int depth = 0; depth < 4 && t != nullptr; ++depth) {
            std::printf(
                    "[optrace]   src[%d] name='%s' op=%s type=%s "
                    "ne=[%lld,%lld,%lld,%lld]\n",
                    depth, t->name ? t->name : "", ggml_op_desc(t),
                    ggml_type_name(t->type), (long long)t->ne[0],
                    (long long)t->ne[1], (long long)t->ne[2],
                    (long long)t->ne[3]);
            t = t->src[0];
        }
    }
    yolo::free_session(cpu);
    yolo::free_session(gpu);
    return first_bad < 0 ? 0 : 1;
}
