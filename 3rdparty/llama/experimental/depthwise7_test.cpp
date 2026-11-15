#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct Shape {
    int w, h, c, n, stride, pad, dilation;
    bool cwhn;
};

static bool run(ggml_backend_t backend, Shape s, ggml_type wt, bool bias,
                int prefix, bool bench, FILE * dump) {
    const int ow = (s.w + 2 * s.pad - s.dilation * 6 - 1) / s.stride + 1;
    const int oh = (s.h + 2 * s.pad - s.dilation * 6 - 1) / s.stride + 1;
    const size_t nx = size_t(s.w) * s.h * s.c * s.n;
    const size_t ny = size_t(ow) * oh * s.c * s.n;
    const size_t nk = size_t(49) * s.c;
    ggml_context * ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    if (!ctx) return false;
    ggml_tensor * xs = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nx + prefix + 4);
    ggml_tensor * ks = ggml_new_tensor_1d(ctx, wt, nk + prefix + 4);
    ggml_tensor * bs = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, s.c + 2);
    ggml_tensor * ys = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ny + prefix + 4);
    const size_t kt = ggml_type_size(wt);
    ggml_tensor * x = s.cwhn
            ? ggml_permute(ctx, ggml_view_4d(ctx, xs, s.c, s.w, s.h, s.n,
                    s.c * 4, s.c * s.w * 4, s.c * s.w * s.h * 4, prefix * 4), 2, 0, 1, 3)
            : ggml_view_4d(ctx, xs, s.w, s.h, s.c, s.n,
                    s.w * 4, s.w * s.h * 4, s.w * s.h * s.c * 4, prefix * 4);
    ggml_tensor * k = s.cwhn
            ? ggml_permute(ctx, ggml_view_4d(ctx, ks, s.c, 7, 7, 1,
                    s.c * kt, s.c * 7 * kt, s.c * 49 * kt, prefix * kt), 3, 0, 1, 2)
            : ggml_view_4d(ctx, ks, 7, 7, 1, s.c, 7 * kt, 49 * kt, 49 * kt, prefix * kt);
    if (s.cwhn) k->nb[2] = kt;
    ggml_tensor * b = ggml_view_1d(ctx, bs, s.c, sizeof(float));
    ggml_tensor * y = bias
            ? ggml_conv_2d_dw_direct_bias(ctx, k, x, b, s.stride, s.stride, s.pad, s.pad, s.dilation, s.dilation)
            : ggml_conv_2d_dw_direct(ctx, k, x, s.stride, s.stride, s.pad, s.pad, s.dilation, s.dilation);
    y->view_src = ys;
    y->view_offs = prefix * sizeof(float);
    ggml_set_output(y);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    if (!ggml_backend_supports_op(backend, y)) { ggml_free(ctx); return false; }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { ggml_free(ctx); return false; }
    std::vector<float> vx(nx + prefix + 4, -98765.0f), vk(nk + prefix + 4, -17.0f);
    std::vector<float> vb(s.c + 2, -17.0f), vy(ny + prefix + 4, -98765.0f);
    for (size_t i = 0; i < nx; ++i) vx[prefix + i] = float(int(i * 37 % 991) - 495) / 997.0f;
    for (size_t i = 0; i < nk; ++i) vk[prefix + i] = float(int(i * 19 % 127) - 63) / 257.0f;
    for (int c = 0; c < s.c; ++c) vb[c + 1] = float(c % 7 - 3) / 8.0f;
    ggml_backend_tensor_set(xs, vx.data(), 0, vx.size() * sizeof(float));
    if (wt == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(vk.size());
        ggml_fp32_to_fp16_row(vk.data(), half.data(), vk.size());
        ggml_backend_tensor_set(ks, half.data(), 0, half.size() * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(half.data(), vk.data(), vk.size());
    } else ggml_backend_tensor_set(ks, vk.data(), 0, vk.size() * sizeof(float));
    ggml_backend_tensor_set(bs, vb.data(), 0, vb.size() * sizeof(float));
    bool valid = true;
    std::vector<float> first, times;
    const int warmup = bench ? 20 : 0, repeats = bench ? 100 : 3;
    for (int rep = -warmup; rep < repeats; ++rep) {
        if (!bench) ggml_backend_tensor_set(ys, vy.data(), 0, vy.size() * sizeof(float));
        const auto start = std::chrono::steady_clock::now();
        valid &= ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        ggml_backend_synchronize(backend);
        const auto stop = std::chrono::steady_clock::now();
        if (rep >= 0) times.push_back(std::chrono::duration<float, std::milli>(stop - start).count());
        if (bench && rep != repeats - 1) continue;
        ggml_backend_tensor_get(ys, vy.data(), 0, vy.size() * sizeof(float));
        if (!bench) {
            for (int i = 0; i < prefix; ++i) valid &= vy[i] == -98765.0f;
            for (size_t i = prefix + ny; i < vy.size(); ++i) valid &= vy[i] == -98765.0f;
        }
        std::vector<float> result(vy.begin() + prefix, vy.begin() + prefix + ny);
        for (float value : result) valid &= std::isfinite(value);
        if (rep == 0 || bench) first = result;
        else valid &= std::memcmp(first.data(), result.data(), ny * sizeof(float)) == 0;
    }
    float max_error = 0;
    for (int n = 0; n < s.n; ++n) for (int c = 0; c < s.c; ++c)
    for (int py = 0; py < oh; ++py) for (int px = 0; px < ow; ++px) {
        float sum = 0;
        for (int ky = 0; ky < 7; ++ky) for (int kx = 0; kx < 7; ++kx) {
            const int iy = py * s.stride + ky * s.dilation - s.pad;
            const int ix = px * s.stride + kx * s.dilation - s.pad;
            if (iy < 0 || iy >= s.h || ix < 0 || ix >= s.w) continue;
            const size_t xi = s.cwhn ? ((size_t(n) * s.h + iy) * s.w + ix) * s.c + c
                                    : ((size_t(n) * s.c + c) * s.h + iy) * s.w + ix;
            const size_t ki = s.cwhn ? size_t(ky * 7 + kx) * s.c + c : size_t(c) * 49 + ky * 7 + kx;
            sum = std::fma(vx[prefix + xi], vk[prefix + ki], sum);
        }
        if (bias) sum += vb[c + 1];
        const size_t yi = s.cwhn ? ((size_t(n) * oh + py) * ow + px) * s.c + c
                                : ((size_t(n) * s.c + c) * oh + py) * ow + px;
        max_error = std::max(max_error, std::abs(first[yi] - sum));
        valid &= std::abs(first[yi] - sum) <= 1e-5f;
    }
    std::printf("DW7 w=%d h=%d c=%d n=%d s=%d p=%d d=%d cwhn=%d wt=%s bias=%d prefix=%d max_error=%.9g valid=%d\n",
                s.w, s.h, s.c, s.n, s.stride, s.pad, s.dilation, s.cwhn,
                ggml_type_name(wt), bias, prefix, max_error, valid);
    if (bench) {
        // O(m log m): p50/p95, m=100.
        std::sort(times.begin(), times.end());
        std::printf("BENCH w=%d c=%d p50_ms=%.9g p95_ms=%.9g warmup=%d iterations=%d\n",
                    s.w, s.c, times[49], times[94], warmup, repeats);
    }
    if (dump) valid &= std::fwrite(first.data(), sizeof(float), first.size(), dump) == first.size();
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return valid;
}

int main(int argc, char ** argv) {
    if (argc != 5) return 2;
    ggml_backend_load_all_from_path(argv[1]);
    ggml_backend_dev_t device = ggml_backend_dev_by_name(argv[2]);
    if (!device) return 77;
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (!backend) return 1;
    const std::string mode = argv[3];
    FILE * dump = std::strcmp(argv[4], "-") ? std::fopen(argv[4], "wb") : nullptr;
    if (std::strcmp(argv[4], "-") && !dump) { ggml_backend_free(backend); return 2; }
    int errors = 0, cases = 0;
    if (mode == "trace" || mode == "fallback") {
        errors += !run(backend, {28, 28, 96, 1, 1, 3, 1, true}, GGML_TYPE_F32, true,
                       mode == "trace" ? 4 : 1, false, dump);
        cases = 1;
    } else if (mode == "bench") {
        for (Shape s : {Shape{112, 112, 192, 1, 1, 3, 1, true},
                        Shape{56, 56, 384, 1, 1, 3, 1, true},
                        Shape{28, 28, 768, 1, 1, 3, 1, true},
                        Shape{14, 14, 1536, 1, 1, 3, 1, true}}) {
            errors += !run(backend, s, GGML_TYPE_F32, true, 0, true, dump);
            ++cases;
        }
    } else if (mode == "matrix") {
        for (Shape s : {Shape{28, 28, 96, 1, 1, 3, 1, true},
                        Shape{17, 13, 12, 2, 1, 3, 1, true},
                        Shape{5, 4, 8, 1, 1, 3, 1, true},
                        Shape{20, 11, 16, 1, 1, 1, 1, true},
                        Shape{21, 21, 32, 1, 1, 0, 1, true},
                        Shape{15, 15, 6, 1, 1, 3, 1, true},
                        Shape{28, 28, 64, 1, 2, 3, 1, true},
                        Shape{28, 28, 64, 1, 1, 6, 2, true},
                        Shape{28, 28, 32, 1, 1, 3, 1, false}})
        for (ggml_type wt : {GGML_TYPE_F32, GGML_TYPE_F16})
        for (bool bias : {false, true}) for (int prefix : {0, 1, 4}) {
            errors += !run(backend, s, wt, bias, prefix, false, dump);
            ++cases;
        }
    } else errors = 1;
    if (dump) errors += std::fclose(dump) != 0;
    ggml_backend_free(backend);
    std::printf("DW7 matrix cases=%d failures=%d\n", cases, errors);
    return errors ? 1 : 0;
}
