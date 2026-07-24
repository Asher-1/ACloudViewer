// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Gate the C++ real upper_bound_resize preprocessing against the upstream
// DA3 InputProcessor on a REAL non-square image (640x427 -> 504x336, pure cv2
// INTER_AREA downscale). Reveals cv2-interpolation parity of resize_area().
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "image_io.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
#include "preprocess.hpp"

int main() {
    const char* base = std::getenv("AICORE_TEST_DEPTH_BASELINE_PREPROC");
    if (!base) {
        std::fprintf(stderr,
                     "[preproc_real] no AICORE_TEST_DEPTH_BASELINE_PREPROC -> SKIP\n");
        return 77;
    }

    // The raw PNG lives next to the baseline (dumps/). Allow an override.
    std::string png = "dumps/preproc_real_input.png";
    if (const char* p = std::getenv("AICORE_TEST_DEPTH_PREPROC_IMAGE")) png = p;

    aicore::depth::Image img;
    if (!aicore::depth::load_image_rgb(png, img)) {
        std::fprintf(stderr, "[preproc_real] cannot load %s -> SKIP\n",
                     png.c_str());
        return 77;
    }

    // Reference processed tensor + expected output size from the baseline gguf.
    std::vector<float> ref;
    std::vector<int64_t> s;
    if (!da_parity::load_baseline(base, "proc_image", ref, s)) return 77;
    uint32_t out_h = da_parity::load_kv_u32(base, "preproc.out_h");
    uint32_t out_w = da_parity::load_kv_u32(base, "preproc.out_w");

    // The InputProcessor normalize + resize policy are fixed constants; build a
    // Config so the gate is self-contained (no full model gguf required).
    aicore::depth::Config cfg;
    cfg.patch_size = 14;
    cfg.img_mean = {0.485f, 0.456f, 0.406f};
    cfg.img_std = {0.229f, 0.224f, 0.225f};
    cfg.img_resize_target = 504;
    cfg.img_resize_mode = "upper_bound";

    aicore::depth::Preprocessed p;
    if (!aicore::depth::preprocess_real(img, cfg, p)) {
        std::fprintf(stderr, "[preproc_real] preprocess_real failed\n");
        return 1;
    }

    std::fprintf(stderr,
                 "[preproc_real] in=%dx%d -> out=%dx%d (ref %ux%u) "
                 "scale=(%.5f,%.5f)\n",
                 img.w, img.h, p.W, p.H, out_w, out_h, p.scale_w, p.scale_h);
    if ((uint32_t)p.H != out_h || (uint32_t)p.W != out_w) {
        std::fprintf(stderr,
                     "[preproc_real] size mismatch got %dx%d ref %ux%u\n", p.W,
                     p.H, out_w, out_h);
        return 1;
    }

    // Honest tolerance: the resize quantizes to uint8 then normalizes, so a
    // single off-by-one uint8 vs cv2 maps to ~1/255/std ~= 1.7e-2 in normalized
    // units at a pixel. We require mean|d| < 1e-2 (sub-pixel-fraction average
    // agreement) and a max|d| that stays within a few uint8 LSBs (< 5e-2). The
    // numbers are printed regardless so any INTER_AREA mismatch is visible.
    bool ok = da_parity::compare(p.chw, ref, "preproc_real", 5e-2f, 0.f);
    return ok ? 0 : 1;
}
