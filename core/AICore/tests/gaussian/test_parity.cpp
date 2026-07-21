// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Per-layer parity gate (ctest label "model"). Runs the engine and stream-
// compares every tap against the float64 PyTorch reference fixtures, in the
// engine's own emission order, holding only one tap in memory at a time.
//
// Skips (exit 77) unless both env vars are set:
//   FREE_SPLATTER_GGUF      path to a converted .gguf (f32 recommended)
//   FREE_SPLATTER_FIXTURES  dir with images.f32 + <tap>.f32
//   (scripts/hf_dump.py)
// Optional:
//   FREE_SPLATTER_DEVICE         cpu|vulkan|cuda     (default cpu)
//   FREE_SPLATTER_PARITY_COS     min per-row cosine  (default 0.9999,
//   f32-strict) FREE_SPLATTER_PARITY_ROWERR  max per-row norm err (default
//   1e-2) FREE_SPLATTER_MAX_BLOCKS     cap blocks (engine-internal) for a fast
//   subset
// For an f16 or Vulkan gguf, widen the gates (e.g. COS=0.999 ROWERR=0.05).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "model.h"

static std::vector<float> read_f32(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<float> v(n / (std::streamsize)sizeof(float));
    f.read((char *)v.data(), n);
    return v;
}

static double env_d(const char *k, double dflt) {
    const char *v = std::getenv(k);
    return v ? std::atof(v) : dflt;
}

int main() {
    const char *gguf = std::getenv("FREE_SPLATTER_GGUF");
    const char *fix = std::getenv("FREE_SPLATTER_FIXTURES");
    if (!gguf || !fix) {
        std::fprintf(
                stderr,
                "SKIP: set FREE_SPLATTER_GGUF and FREE_SPLATTER_FIXTURES\n");
        return 77;
    }
    const std::string fixdir = fix;
    const char *dev = std::getenv("FREE_SPLATTER_DEVICE");
    const double cos_gate = env_d("FREE_SPLATTER_PARITY_COS", 0.9999);
    const double rowerr_gate = env_d("FREE_SPLATTER_PARITY_ROWERR", 1e-2);

    std::vector<float> images = read_f32(fixdir + "/images.f32");
    if (images.empty()) {
        std::fprintf(stderr, "no %s/images.f32\n", fix);
        return 1;
    }

    aicore::gaussian::model m;
    if (!m.load(gguf, dev ? dev : "cpu", 0)) {
        std::fprintf(stderr, "load failed: %s\n", m.error.c_str());
        return 1;
    }
    const aicore::gaussian::hparams &hp = m.hp();
    const int64_t per_view =
            (int64_t)hp.in_channels * hp.image_size * hp.image_size;
    if ((int64_t)images.size() % per_view != 0) {
        std::fprintf(stderr, "images.f32 size not a whole number of views\n");
        return 1;
    }
    const int32_t n_views = (int32_t)(images.size() / per_view);

    // In a capped-depth subset run the head sits on a truncated transformer, so
    // its taps cannot match the full-depth reference -- compare block taps
    // only.
    const char *mb = std::getenv("FREE_SPLATTER_MAX_BLOCKS");
    const bool partial = mb && std::atoi(mb) > 0 && std::atoi(mb) < hp.n_layer;
    auto is_head_tap = [](const std::string &n) {
        return n == "result_norm" || n == "head_logits" ||
               n == "gaussians_raw" || n == "gaussians";
    };

    int compared = 0, skipped = 0;
    double worst_cos = 1.0, worst_rowerr = 0.0;
    std::string worst_cos_tap, worst_rowerr_tap;

    auto sink = [&](const std::string &name, const float *data, int64_t rows,
                    int64_t cols) {
        if (partial && is_head_tap(name)) {
            skipped++;
            return;
        }
        const std::vector<float> ref = read_f32(fixdir + "/" + name + ".f32");
        if ((int64_t)ref.size() != rows * cols) {
            skipped++;
            return;
        }
        double min_cos = 1.0, max_re = 0.0;
        for (int64_t r = 0; r < rows; r++) {
            const float *a = data + r * cols;
            const float *b = ref.data() + r * cols;
            double dot = 0, na = 0, nb = 0, maxabs = 0, maxref = 1.0;
            for (int64_t c = 0; c < cols; c++) {
                dot += (double)a[c] * b[c];
                na += (double)a[c] * a[c];
                nb += (double)b[c] * b[c];
                const double d = std::fabs((double)a[c] - b[c]);
                if (d > maxabs) maxabs = d;
                if (std::fabs((double)b[c]) > maxref)
                    maxref = std::fabs((double)b[c]);
            }
            const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
            const double re = maxabs / maxref;
            if (cos < min_cos) min_cos = cos;
            if (re > max_re) max_re = re;
        }
        compared++;
        if (min_cos < worst_cos) {
            worst_cos = min_cos;
            worst_cos_tap = name;
        }
        if (max_re > worst_rowerr) {
            worst_rowerr = max_re;
            worst_rowerr_tap = name;
        }
    };

    std::vector<float> out;
    if (!m.forward(images.data(), n_views, out, sink)) {
        std::fprintf(stderr, "forward failed: %s\n", m.error.c_str());
        return 1;
    }

    std::printf(
            "parity: %d taps compared (%d skipped, no fixture), worst_cos=%.6f "
            "(%s) "
            "worst_rowerr=%.3e (%s)\n",
            compared, skipped, worst_cos, worst_cos_tap.c_str(), worst_rowerr,
            worst_rowerr_tap.c_str());
    if (compared == 0) {
        std::fprintf(stderr, "no taps had matching fixtures\n");
        return 1;
    }

    const bool ok = worst_cos >= cos_gate && worst_rowerr <= rowerr_gate;
    std::printf(ok ? "test_parity: ok (gates cos>=%.5f rowerr<=%.1e)\n"
                   : "test_parity: FAIL (gates cos>=%.5f rowerr<=%.1e)\n",
                cos_gate, rowerr_gate);
    return ok ? 0 : 1;
}
