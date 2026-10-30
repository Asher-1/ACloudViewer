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
#include <limits>
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
    if (const char* mode = std::getenv("AICORE_TEST_YOLO_ATTENTION_ACCUMULATION")) {
        // The test bridges an explicit option; production sessions never read
        // environment variables.
        if (std::strcmp(mode, "0") == 0)
            opts.attention_accumulation = AICORE_YOLO_ATTENTION_UPSTREAM;
        else if (std::strcmp(mode, "1") == 0)
            opts.attention_accumulation = AICORE_YOLO_ATTENTION_F32;
        else
            return nullptr;
    }
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

// Keeps only the current layer's real sources; captured after that layer
// completes and before the next layer reuses the scratch buffer.
struct ReplayCapture {
    ggml_tensor* node = nullptr;
    std::vector<uint8_t> bytes[3];
    std::vector<float> values[3], output;
    bool valid = false;
};

bool capture_replay(ggml_tensor* tensor, bool ask, void* opaque) {
    auto& capture = *static_cast<ReplayCapture*>(opaque);
    if (ask) return tensor == capture.node;
    if (tensor != capture.node) return true;
    capture.valid = true;
    for (int i = 0; i < 3; ++i) if (tensor->src[i]) {
        auto* source = tensor->src[i];
        capture.valid &= ggml_is_contiguous(source);
        capture.values[i] = to_f32(source);
        capture.valid &= !capture.values[i].empty();
        capture.bytes[i].resize(ggml_nbytes(source));
        ggml_backend_tensor_get(source, capture.bytes[i].data(), 0, capture.bytes[i].size());
    }
    capture.output = to_f32(tensor);
    capture.valid &= !capture.output.empty();
    return capture.valid;
}

// Diagnostic, not a release gate: pins the layer's real inputs and weights to
// separate propagated error from error this layer adds.
// The oracle's O(N*OC*OH*OW*IC*KH*KW) multiply-adds follow the convolution
// definition and never call a ggml convolution.
bool replay_conv(yolo::Session* session, size_t index,
                 yolo::Session* source_session, const ReplayCapture& capture) {
    if (index >= session->op_values.size() || !session->backend.gpu) return false;
    ggml_tensor* node = session->op_values[index];
    if (!node || node->op != GGML_OP_CONV_2D || !node->src[0] || !node->src[1]) return false;
    // A cross-GPU replay pins the reference's own inputs so propagated
    // upstream error is not attributed to this layer.
    ggml_tensor* source_node = node;
    if (source_session) {
        if (index >= source_session->op_values.size()) return false;
        source_node = source_session->op_values[index];
        if (!source_node || source_node->op != node->op || source_node->type != node->type ||
            !ggml_are_same_shape(source_node, node) ||
            std::memcmp(source_node->op_params, node->op_params, 8 * sizeof(int32_t)) != 0)
            return false;
    }
    ggml_tensor* original[] = {source_node->src[0], source_node->src[1], source_node->src[2]};
    if (!capture.valid || capture.node != source_node) return false;
    const auto& w = capture.values[0];
    const auto& x = capture.values[1];
    const auto& bias = capture.values[2];
    const auto& original_output = capture.output;
    auto* ctx = ggml_init({1024 * 1024, nullptr, true});
    if (!ctx) return false;
    ggml_tensor* sources[3] = {};
    for (int i = 0; i < 3; ++i) if (original[i]) sources[i] = ggml_dup_tensor(ctx, original[i]);
    int params[6];
    for (int i = 0; i < 6; ++i) params[i] = node->op_params[i];
    const bool fused = node->op_params[7] == 1;
    auto* out = fused ? ggml_conv_2d_direct_bias_silu(ctx, sources[0], sources[1], sources[2],
            params[0], params[1], params[2], params[3], params[4], params[5]) :
            ggml_conv_2d_direct(ctx, sources[0], sources[1], params[0], params[1],
                               params[2], params[3], params[4], params[5]);
    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    auto* buffer = ggml_backend_alloc_ctx_tensors(ctx, session->backend.gpu);
    if (!buffer) { ggml_free(ctx); return false; }
    for (int i = 0; i < 3; ++i) if (sources[i]) {
        // Use the snapshot taken during compute; an intermediate cast may
        // already have been reused by the allocator after the full graph ends.
        if (capture.bytes[i].size() != ggml_nbytes(sources[i])) {
            ggml_backend_buffer_free(buffer); ggml_free(ctx); return false;
        }
        ggml_backend_tensor_set(sources[i], capture.bytes[i].data(), 0, capture.bytes[i].size());
    }
    const bool computed = ggml_backend_graph_compute(session->backend.gpu, graph) == GGML_STATUS_SUCCESS;
    const auto actual = computed ? to_f32(out) : std::vector<float>{};
    bool valid = !actual.empty() && actual.size() == original_output.size();
    double squared = 0, reference_squared = 0;
    size_t candidate_better = 0, reference_better = 0, equal_outputs = 0;
    float maximum = 0, repeat_error = 0, reference_maximum = 0;
    size_t worst = 0;
    float worst_expected = 0;
    const int64_t kw = original[0]->ne[0], kh = original[0]->ne[1];
    const int64_t iw = original[1]->ne[0], ih = original[1]->ne[1], ic = original[1]->ne[2];
    const int64_t ow = out->ne[0], oh = out->ne[1], oc = out->ne[2];
    for (size_t i = 0; valid && i < actual.size(); ++i) {
        const int64_t px = i % ow, py = (i / ow) % oh;
        const int64_t channel = (i / (ow * oh)) % oc, batch = i / (ow * oh * oc);
        double sum = 0;
        for (int64_t c = 0; c < ic; ++c) for (int64_t ky = 0; ky < kh; ++ky)
        for (int64_t kx = 0; kx < kw; ++kx) {
            const int64_t sx = px * params[0] - params[2] + kx * params[4];
            const int64_t sy = py * params[1] - params[3] + ky * params[5];
            if (sx >= 0 && sx < iw && sy >= 0 && sy < ih)
                sum += double(x[((batch * ic + c) * ih + sy) * iw + sx]) *
                       w[((channel * ic + c) * kh + ky) * kw + kx];
        }
        if (fused) {
            sum += bias[channel];
            sum /= 1.0 + std::exp(-sum);
        }
        float expected = float(sum);
        if (out->type == GGML_TYPE_F16) expected = ggml_fp16_to_fp32(ggml_fp32_to_fp16(expected));
        const float error = std::abs(actual[i] - expected);
        valid &= std::isfinite(actual[i]) && std::isfinite(expected);
        squared += double(error) * error;
        const float reference_error = std::abs(original_output[i] - expected);
        reference_squared += double(reference_error) * reference_error;
        reference_maximum = std::max(reference_maximum, reference_error);
        candidate_better += error < reference_error;
        reference_better += reference_error < error;
        equal_outputs += actual[i] == original_output[i];
        if (error > maximum) { maximum = error; worst = i; worst_expected = expected; }
        repeat_error = std::max(repeat_error, std::abs(actual[i] - original_output[i]));
    }
    std::printf("[replay] input_origin=%s op=%zu W=%s X=%s Y=%s fused=%d kernel=%lldx%lld IC=%lld OC=%lld "
                "n=%zu oracle_max=%.9g oracle_rms=%.9g comparison_max=%.9g worst=%zu actual=%.9g expected=%.9g valid=%d\n",
                source_session ? "reference" : "candidate", index,
                ggml_type_name(original[0]->type), ggml_type_name(original[1]->type),
                ggml_type_name(out->type), fused, (long long)kw, (long long)kh,
                (long long)ic, (long long)oc, actual.size(), maximum,
                actual.empty() ? 0 : std::sqrt(squared / actual.size()), repeat_error,
                worst, actual.empty() ? 0 : actual[worst], worst_expected, valid);
    if (source_session)
        std::printf("[replay] reference_oracle_max=%.9g reference_oracle_rms=%.9g "
                    "candidate_better=%zu reference_better=%zu equal_outputs=%zu/%zu\n",
                    reference_maximum, actual.empty() ? 0 : std::sqrt(reference_squared / actual.size()),
                    candidate_better, reference_better, equal_outputs, actual.size());
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return valid;
}

ggml_tensor* select_matmul(yolo::Session* session, size_t ordinal) {
    for (int i = 0; i < ggml_graph_n_nodes(session->graph); ++i) {
        auto* node = ggml_graph_node(session->graph, i);
        if (node->op == GGML_OP_MUL_MAT && ordinal-- == 0) return node;
    }
    return nullptr;
}

// Pins the real inputs seen during compute; an element-wise double dot product
// separates accumulation precision from upstream error.
// O(batch*M*N*K) follows the matrix-multiplication definition; the output
// snapshot holds at most 8M elements (32 MiB).
bool replay_matmul(yolo::Session* session, size_t ordinal, const ReplayCapture& capture) {
    auto* node = select_matmul(session, ordinal);
    if (!node || !capture.valid || !session->backend.gpu || !capture.node ||
        capture.node->op != GGML_OP_MUL_MAT || !ggml_are_same_shape(node, capture.node) ||
        node->type != GGML_TYPE_F32 || capture.output.size() > 8 * 1024 * 1024) return false;
    auto* a = capture.node->src[0];
    auto* b = capture.node->src[1];
    if (!a || !b || !ggml_are_same_shape(a, node->src[0]) ||
        !ggml_are_same_shape(b, node->src[1]) || a->type != node->src[0]->type ||
        b->type != node->src[1]->type || b->ne[2] % a->ne[2] || b->ne[3] % a->ne[3]) return false;
    const int64_t K = a->ne[0], M = a->ne[1], N = b->ne[1];
    std::vector<float> expected(capture.output.size());
    float reference_max = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const int64_t m = i % M, n = (i / M) % N;
        const int64_t batch2 = (i / (M * N)) % b->ne[2], batch3 = i / (M * N * b->ne[2]);
        const int64_t a2 = batch2 / (b->ne[2] / a->ne[2]), a3 = batch3 / (b->ne[3] / a->ne[3]);
        const size_t ai = ((a3 * a->ne[2] + a2) * M + m) * K;
        const size_t bi = ((batch3 * b->ne[2] + batch2) * N + n) * K;
        if (ai + K > capture.values[0].size() || bi + K > capture.values[1].size()) return false;
        double sum = 0;
        for (int64_t k = 0; k < K; ++k)
            sum += double(capture.values[0][ai + k]) * capture.values[1][bi + k];
        expected[i] = float(sum);
        if (!std::isfinite(expected[i])) return false;
        reference_max = std::max(reference_max, std::abs(capture.output[i] - expected[i]));
    }
    bool all_valid = true;
    for (bool f32 : {false, true}) {
        auto* ctx = ggml_init({1024 * 1024, nullptr, true});
        if (!ctx) return false;
        auto* ac = ggml_dup_tensor(ctx, a);
        auto* bc = ggml_dup_tensor(ctx, b);
        ggml_set_name(ac, a->name);
        ggml_set_name(bc, b->name);
        auto* out = ggml_mul_mat(ctx, ac, bc);
        std::memcpy(out->op_params, capture.node->op_params, sizeof(out->op_params));
        ggml_set_name(out, capture.node->name);
        if (f32 && !ggml_prec_set_acc(out, GGML_PREC_F32)) {
            ggml_free(ctx);
            return false;
        }
        auto* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        auto* buffer = ggml_backend_alloc_ctx_tensors(ctx, session->backend.gpu);
        if (!buffer) { ggml_free(ctx); return false; }
        ggml_backend_tensor_set(ac, capture.bytes[0].data(), 0, capture.bytes[0].size());
        ggml_backend_tensor_set(bc, capture.bytes[1].data(), 0, capture.bytes[1].size());
        std::vector<float> poison(expected.size(), std::numeric_limits<float>::quiet_NaN());
        std::vector<float> first;
        float maximum = 0, comparison = 0, repeat_error = 0;
        double squared = 0;
        bool valid = true;
        for (int repeat = 0; repeat < 3; ++repeat) {
            ggml_backend_tensor_set(out, poison.data(), 0, poison.size() * sizeof(float));
            valid &= ggml_backend_graph_compute(session->backend.gpu, graph) == GGML_STATUS_SUCCESS;
            const auto actual = valid ? to_f32(out) : std::vector<float>{};
            valid &= actual.size() == expected.size();
            for (size_t i = 0; valid && i < actual.size(); ++i) {
                valid &= std::isfinite(actual[i]);
                const float error = std::abs(actual[i] - expected[i]);
                maximum = std::max(maximum, error);
                comparison = std::max(comparison, std::abs(actual[i] - capture.output[i]));
                if (repeat == 0) squared += double(error) * error;
                else repeat_error = std::max(repeat_error, std::abs(actual[i] - first[i]));
            }
            if (repeat == 0) first = actual;
        }
        valid &= repeat_error == 0;
        std::printf("[matmul-replay] ordinal=%zu precision=%s A=%s B=%s K=%lld M=%lld N=%lld "
                    "n=%zu oracle_max=%.9g oracle_rms=%.9g reference_oracle_max=%.9g "
                    "comparison_max=%.9g repeat_error=%.9g valid=%d\n", ordinal, f32 ? "f32" : "default",
                    ggml_type_name(a->type), ggml_type_name(b->type), (long long)K, (long long)M, (long long)N,
                    expected.size(), maximum, std::sqrt(squared / expected.size()), reference_max,
                    comparison, repeat_error, valid);
        all_valid &= valid;
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    return all_valid;
}

}  // namespace

int main() {
    const char* gguf = std::getenv("AICORE_TEST_YOLO_GGUF");
    const char* image = std::getenv("AICORE_TEST_YOLO_IMAGE");
    const char* classes = std::getenv("AICORE_TEST_YOLO_CLASSES");
    const char* text_model = std::getenv("AICORE_TEST_YOLO_TEXT_MODEL");
    const char* device_env = std::getenv("AICORE_TEST_YOLO_PARITY_DEVICE");
    const std::string gpu_device = device_env != nullptr ? device_env : "cuda";
    const char* reference_env = std::getenv("AICORE_TEST_YOLO_REFERENCE_DEVICE");
    const std::string reference_device = reference_env ? reference_env : "cpu";
    std::printf("[optrace] reference=%s candidate=%s; parity diagnostic, not independent truth\n",
                reference_device.c_str(), gpu_device.c_str());
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

    yolo::Session* reference = make_session(gguf, reference_device.c_str(), embed, nc, info);
    yolo::Session* gpu =
            make_session(gguf, gpu_device.c_str(), embed, nc, info);
    if (reference == nullptr || gpu == nullptr) {
        std::printf("[optrace] skipped: session load failed (%s / %s)\n",
                    reference != nullptr ? "reference ok" : "reference failed",
                    gpu != nullptr ? "gpu ok" : "gpu failed");
        yolo::free_session(reference);
        yolo::free_session(gpu);
        return 77;
    }
    ReplayCapture reference_capture, candidate_capture;
    const char* replay_op = std::getenv("AICORE_TEST_YOLO_REPLAY_OP");
    const char* replay_mm = std::getenv("AICORE_TEST_YOLO_REPLAY_MATMUL");
    if (replay_op && replay_mm) {
        yolo::free_session(reference);
        yolo::free_session(gpu);
        return 1;
    }
    const char* replay_index = replay_mm ? replay_mm : replay_op;
    if (const char* replay = replay_index) {
        char* end = nullptr;
        const long index = std::strtol(replay, &end, 10);
        bool installed = end != replay && *end == '\0' && index >= 0;
        const bool cross = std::getenv("AICORE_TEST_YOLO_REPLAY_REFERENCE_INPUT") != nullptr;
        auto install = [&](yolo::Session* session, ReplayCapture& capture) {
            if (!installed || !session->backend.sched) return false;
            if (replay_mm) capture.node = select_matmul(session, static_cast<size_t>(index));
            else if (static_cast<size_t>(index) < session->op_values.size())
                capture.node = session->op_values[index];
            if (!capture.node || capture.node->op != (replay_mm ? GGML_OP_MUL_MAT : GGML_OP_CONV_2D)) return false;
            ggml_backend_sched_set_eval_callback(session->backend.sched, capture_replay, &capture);
            return true;
        };
        installed = installed && install(gpu, candidate_capture);
        if (cross) installed = installed && install(reference, reference_capture);
        if (!installed) {
            std::fprintf(stderr, "[replay] cannot capture the real layer inputs; no numeric conclusion is allowed\n");
            yolo::free_session(reference);
            yolo::free_session(gpu);
            return 1;
        }
    }
    if (!yolo::session_run(reference, canvas.data()) ||
        !yolo::session_run(gpu, canvas.data())) {
        std::printf("[optrace] inference failed\n");
        yolo::free_session(reference);
        yolo::free_session(gpu);
        return 1;
    }

    if (const char* replay = replay_index) {
        char* end = nullptr;
        const long index = std::strtol(replay, &end, 10);
        bool valid = end != replay && *end == '\0' && index >= 0;
        const bool cross = std::getenv("AICORE_TEST_YOLO_REPLAY_REFERENCE_INPUT") != nullptr;
        const auto& capture = cross ? reference_capture : candidate_capture;
        valid &= capture.valid;
        // op0's known image cast is a lifetime regression assertion: it must
        // consume the exact half conversion of the original image.
        if (valid && !replay_mm && index == 0) {
            const auto& captured = capture.values[1];
            valid = captured.size() == canvas.size();
            for (size_t i = 0; valid && i < captured.size(); ++i) {
                const float expected = capture.node->src[1]->type == GGML_TYPE_F16
                    ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(canvas[i])) : canvas[i];
                valid &= captured[i] == expected;
            }
            if (!valid) std::fprintf(stderr, "[replay] the image input is stale; it does not qualify for oracle comparison\n");
        }
        valid = valid && (replay_mm ? replay_matmul(gpu, static_cast<size_t>(index), capture) :
            replay_conv(gpu, static_cast<size_t>(index), cross ? reference : nullptr, capture));
        if (reference->backend.sched) ggml_backend_sched_set_eval_callback(reference->backend.sched, nullptr, nullptr);
        if (gpu->backend.sched) ggml_backend_sched_set_eval_callback(gpu->backend.sched, nullptr, nullptr);
        yolo::free_session(reference);
        yolo::free_session(gpu);
        return valid ? 0 : 1;
    }

    // Compare per USER-OP outputs (not graph nodes — the CUDA f16 flow
    // builds extra cast nodes, so node indices differ across devices).
    // EVERY op is printed (error-propagation profile); the tool reports the
    // FIRST op whose output leaves the tolerance band, but keeps scanning so
    // a rounding snowball is distinguishable from a single bad kernel.
    // stdout is fully buffered under redirection; a crash mid-loop used to
    // lose every unflushed line (the bisection tail vanished together with
    // the segfault). Unbuffered stdout keeps the last compared op visible.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const size_t n_ops = reference->model.ops.size();
    std::printf("[optrace] user ops: reference=%zu candidate=%zu\n", n_ops,
                gpu->model.ops.size());
    int first_bad = -1;
    size_t n_compared = 0, n_null = 0;
    for (size_t i = 0; i < n_ops; ++i) {
        // op_values holds one entry per emitted op; a plan that skips or
        // fuses trailing ops makes the vector shorter than model.ops —
        // indexing past it used to read out of bounds and crash.
        if (i >= reference->op_values.size() || i >= gpu->op_values.size()) {
            std::printf(
                    "[optrace] op %3zu skipped: op_values shorter "
                    "(reference=%zu candidate=%zu)\n",
                    i, reference->op_values.size(), gpu->op_values.size());
            break;
        }
        ggml_tensor* ta = reference->op_values[i];
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
                    i, reference->model.ops[i].type.c_str(), ha.size(), hb.size());
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
                "[optrace] op %3zu %-18s ne=[%lld,%lld,%lld,%lld] n=%9zu "
                "rel=%.3e maxdiff=%.4f big_errs=%lld%s\n",
                i, reference->model.ops[i].type.c_str(), (long long)ta->ne[0],
                (long long)ta->ne[1], (long long)ta->ne[2],
                (long long)ta->ne[3], ha.size(), rel, mx, bad,
                (first_bad < 0 && (rel > 1e-2 || bad > 0))
                        ? "  <-- FIRST OUT OF BAND"
                        : "");
        if (first_bad < 0 && (rel > 1e-2 || bad > 0)) first_bad = (int)i;
        ++n_compared;
    }
    std::printf("[optrace] compared %zu/%zu ops (%zu null)\n", n_compared,
                n_ops, n_null);
    if (first_bad < 0) {
        std::printf("[optrace] within the diagnostic threshold; this is not bitwise equality or an independent accuracy gate\n");
    } else {
        std::printf("[optrace] FIRST DIVERGENCE at user-op %d (%s)\n",
                    first_bad, reference->model.ops[first_bad].type.c_str());
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
    cmp_final("output", reference->output, gpu->output);
    cmp_final("proto", reference->output_proto, gpu->output_proto);

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
    describe("reference proto", reference->output_proto);
    describe("gpu proto f32", gpu->output_proto);
    if (gpu->output_proto != nullptr && gpu->output_proto->src[0] != nullptr)
        describe("gpu proto f16", gpu->output_proto->src[0]);
    if (reference->output_proto != nullptr && reference->output_proto->src[0] != nullptr)
        describe("reference source", reference->output_proto->src[0]);
    describe("reference output", reference->output);
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
    peek("reference proto", reference->output_proto, 8);
    peek("gpu proto f32", gpu->output_proto, 8);
    if (gpu->output_proto != nullptr && gpu->output_proto->src[0] != nullptr)
        peek("gpu proto f16", gpu->output_proto->src[0], 8);
    // Which USER-OP produced the proto? Pointer-match it against op_values;
    // when it MISSES, walk the src chain — a cast fed by an INNER graph node
    // (not the stored op value) means the op-loop wrapped the branch output
    // in extra nodes on this backend.
    // Detect-only models have no proto/mask head at all: output_proto is
    // null and dereferencing it here used to segfault AFTER the full op
    // comparison had already completed.
    bool proto_ptr_matched = false;
    if (gpu->output_proto != nullptr) {
        for (size_t i = 0; i < gpu->op_values.size(); ++i) {
            if (gpu->op_values[i] == gpu->output_proto->src[0]) {
                proto_ptr_matched = true;
                std::printf("[optrace] proto source = op %zu (%s)\n", i,
                            gpu->model.ops[i].type.c_str());
                peek("reference opval", reference->op_values[i], 8);
                peek("gpu opval", gpu->op_values[i], 8);
                break;
            }
        }
    }
    if (!proto_ptr_matched && gpu->output_proto != nullptr) {
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
    yolo::free_session(reference);
    yolo::free_session(gpu);
    return first_bad < 0 ? 0 : 1;
}
