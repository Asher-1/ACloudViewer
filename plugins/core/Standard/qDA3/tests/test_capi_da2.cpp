// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Verifies the C-API routes a Depth Anything V2 (relative ViT-L) GGUF through
// the DA2 depth-only path: aicore_depth_depth_dense returns depth only (no
// conf/sky/pose), is_metric==0 for the relative model, and
// aicore_depth_pose_path returns -1 (no pose).
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "aicore/depth_capi.h"

static bool finite_all(const float* p, int n) {
    for (int i = 0; i < n; ++i)
        if (!std::isfinite(p[i])) return false;
    return true;
}

int main() {
    const char* gguf =
            std::getenv("DA_TEST_GGUF_DA2");      // relative ViT-L DA2 GGUF
    if (!gguf) return 77;                         // skip if fixture absent
    const char* png = "assets/samples/desk.jpg";  // WORKING_DIRECTORY = DA_ROOT
    FILE* f = std::fopen(png, "rb");
    if (!f) {
        std::fprintf(stderr, "sample image %s absent, skipping\n", png);
        return 77;
    }
    std::fclose(f);

    aicore_depth_ctx* c = aicore_depth_load(gguf, 1);
    if (!c) {
        std::fprintf(stderr, "da2: load failed\n");
        return 1;
    }

    int H = 0, W = 0, is_metric = -1;
    float *depth = nullptr, *conf = nullptr, *sky = nullptr;
    float ext[12], intr[9];
    int r = aicore_depth_depth_dense(c, png, &H, &W, &depth, &conf, &sky, ext,
                                     intr, &is_metric);
    bool ok = (r == 0) && H > 0 && W > 0 && depth && !conf && !sky;
    if (ok) ok = (H * W > 0) && finite_all(depth, H * W);
    if (ok) ok = (is_metric == 0);  // relative DA2 -> non-metric
    std::fprintf(stderr,
                 "da2 dense: r=%d %dx%d depth=%p conf=%p sky=%p is_metric=%d "
                 "-> %s\n",
                 r, W, H, (void*)depth, (void*)conf, (void*)sky, is_metric,
                 ok ? "OK" : "FAIL");
    aicore_depth_free_floats(depth);
    aicore_depth_free_floats(conf);
    aicore_depth_free_floats(sky);

    // DA2 has no camera pose: pose_path must fail.
    int rp = aicore_depth_pose_path(c, png, ext, intr);
    bool okp = (rp == -1);
    std::fprintf(stderr, "da2 pose: r=%d (expect -1) err=\"%s\" -> %s\n", rp,
                 aicore_depth_last_error(c), okp ? "OK" : "FAIL");

    aicore_depth_free(c);
    return (ok && okp) ? 0 : 1;
}
