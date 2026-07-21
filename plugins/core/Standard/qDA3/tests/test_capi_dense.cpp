// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Verifies the dense C-API: aicore_depth_depth_dense + aicore_depth_points for
// a DualDPT model (base f32) and, if available, a mono model (mono-large f32).
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "aicore/depth_capi.h"

static bool finite_all(const float* p, int n) {
    for (int i = 0; i < n; ++i)
        if (!std::isfinite(p[i])) return false;
    return true;
}

// DualDPT: depth+conf present, sky NULL, plausible pose; points return N>0.
static bool test_dualdpt(const char* gguf, const char* png) {
    aicore_depth_ctx* c = aicore_depth_load(gguf, 1);
    if (!c) {
        std::fprintf(stderr, "dualdpt: load failed\n");
        return false;
    }
    int H = 0, W = 0, is_metric = -1;
    float *depth = nullptr, *conf = nullptr, *sky = nullptr;
    float ext[12], intr[9];
    int r = aicore_depth_depth_dense(c, png, &H, &W, &depth, &conf, &sky, ext,
                                     intr, &is_metric);
    bool ok = (r == 0) && H > 0 && W > 0 && depth && conf && !sky;
    if (ok) ok = finite_all(depth, H * W) && finite_all(conf, H * W);
    // Plausible pose: intrinsics fx,fy > 0; extrinsics finite.
    if (ok)
        ok = (intr[0] > 0.f) && (intr[4] > 0.f) && finite_all(ext, 12) &&
             finite_all(intr, 9);
    if (ok) ok = (is_metric == 0);  // base is a relative (non-metric) DualDPT
    std::fprintf(stderr,
                 "dualdpt dense: r=%d %dx%d depth=%p conf=%p sky=%p fx=%.3f "
                 "fy=%.3f is_metric=%d -> %s\n",
                 r, W, H, (void*)depth, (void*)conf, (void*)sky, intr[0],
                 intr[4], is_metric, ok ? "OK" : "FAIL");
    aicore_depth_free_floats(depth);
    aicore_depth_free_floats(conf);
    aicore_depth_free_floats(sky);

    int N = 0;
    float* xyz = nullptr;
    unsigned char* rgb = nullptr;
    int rp = aicore_depth_points(c, png, 0.0f, &N, &xyz, &rgb);
    bool okp = (rp == 0) && N > 0 && xyz && rgb && finite_all(xyz, 3 * N);
    std::fprintf(stderr, "dualdpt points: r=%d N=%d xyz=%p rgb=%p -> %s\n", rp,
                 N, (void*)xyz, (void*)rgb, okp ? "OK" : "FAIL");
    aicore_depth_free_floats(xyz);
    aicore_depth_free_bytes(rgb);
    aicore_depth_free(c);
    return ok && okp;
}

// mono: depth+sky present, conf NULL; points return -1.
static bool test_mono(const char* gguf, const char* png) {
    aicore_depth_ctx* c = aicore_depth_load(gguf, 1);
    if (!c) {
        std::fprintf(stderr, "mono: load failed\n");
        return false;
    }
    int H = 0, W = 0, is_metric = -1;
    float *depth = nullptr, *conf = nullptr, *sky = nullptr;
    float ext[12], intr[9];
    int r = aicore_depth_depth_dense(c, png, &H, &W, &depth, &conf, &sky, ext,
                                     intr, &is_metric);
    bool ok = (r == 0) && H > 0 && W > 0 && depth && sky && !conf;
    if (ok) ok = finite_all(depth, H * W) && finite_all(sky, H * W);
    if (ok) ok = (is_metric == 1);  // DA3MONO is metric
    std::fprintf(stderr,
                 "mono dense: r=%d %dx%d depth=%p conf=%p sky=%p is_metric=%d "
                 "-> %s\n",
                 r, W, H, (void*)depth, (void*)conf, (void*)sky, is_metric,
                 ok ? "OK" : "FAIL");
    aicore_depth_free_floats(depth);
    aicore_depth_free_floats(conf);
    aicore_depth_free_floats(sky);

    int N = 0;
    float* xyz = nullptr;
    unsigned char* rgb = nullptr;
    int rp = aicore_depth_points(c, png, 0.0f, &N, &xyz, &rgb);
    bool okp = (rp == -1) && !xyz && !rgb;  // mono has no pose
    std::fprintf(stderr, "mono points: r=%d (expect -1) err=\"%s\" -> %s\n", rp,
                 aicore_depth_last_error(c), okp ? "OK" : "FAIL");
    aicore_depth_free_floats(xyz);
    aicore_depth_free_bytes(rgb);
    aicore_depth_free(c);
    return ok && okp;
}

// nested: two-branch metric model via aicore_depth_load_nested. depth present,
// conf and sky NULL, is_metric==1, plausible pose.
static bool test_nested(const char* anyview,
                        const char* metric,
                        const char* png) {
    aicore_depth_ctx* c = aicore_depth_load_nested(anyview, metric, 1);
    if (!c) {
        std::fprintf(stderr, "nested: load failed\n");
        return false;
    }
    int H = 0, W = 0, is_metric = -1;
    float *depth = nullptr, *conf = nullptr, *sky = nullptr;
    float ext[12], intr[9];
    int r = aicore_depth_depth_dense(c, png, &H, &W, &depth, &conf, &sky, ext,
                                     intr, &is_metric);
    bool ok = (r == 0) && H > 0 && W > 0 && depth && !conf && !sky;
    if (ok) ok = finite_all(depth, H * W);
    if (ok)
        ok = (intr[0] > 0.f) && (intr[4] > 0.f) && finite_all(ext, 12) &&
             finite_all(intr, 9);
    if (ok) ok = (is_metric == 1);
    std::fprintf(stderr,
                 "nested dense: r=%d %dx%d depth=%p conf=%p sky=%p fx=%.3f "
                 "is_metric=%d -> %s\n",
                 r, W, H, (void*)depth, (void*)conf, (void*)sky, intr[0],
                 is_metric, ok ? "OK" : "FAIL");
    aicore_depth_free_floats(depth);
    aicore_depth_free_floats(conf);
    aicore_depth_free_floats(sky);
    aicore_depth_free(c);
    return ok;
}

int main() {
    const char* gguf = std::getenv("DA_TEST_GGUF");  // base f32 (DualDPT)
    const char* png = std::getenv("DA_TEST_NATIVE_PNG");
    if (!gguf || !png) return 77;  // skip if fixtures absent
    bool ok = test_dualdpt(gguf, png);

    const char* nanyview = std::getenv("DA_TEST_GGUF_NESTED_ANYVIEW");
    const char* nmetric = std::getenv("DA_TEST_GGUF_NESTED_METRIC");
    if (nanyview && nmetric) {
        ok = test_nested(nanyview, nmetric, png) && ok;
    } else {
        std::fprintf(stderr, "nested env not set, skipping nested checks\n");
    }

    const char* mono = std::getenv("DA_TEST_GGUF_MONO");
    const char* mpng = std::getenv("DA_TEST_MONO_PNG");
    if (mono && mpng) {
        FILE* f = std::fopen(mono, "rb");
        if (f) {
            std::fclose(f);
            ok = test_mono(mono, mpng) && ok;
        } else
            std::fprintf(stderr,
                         "mono gguf absent (%s), skipping mono checks\n", mono);
    } else {
        std::fprintf(stderr, "mono env not set, skipping mono checks\n");
    }
    return ok ? 0 : 1;
}
