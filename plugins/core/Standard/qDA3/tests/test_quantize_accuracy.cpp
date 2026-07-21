// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "engine.hpp"
#include "image_io.hpp"
#include "parity.hpp"
#include "quantize.hpp"

static double corr(const std::vector<float>& a, const std::vector<float>& b) {
    double ma = 0, mb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        ma += a[i];
        mb += b[i];
    }
    ma /= a.size();
    mb /= b.size();
    double na = 0, nb = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double x = a[i] - ma, y = b[i] - mb;
        na += x * x;
        nb += y * y;
        d += x * y;
    }
    return d / (std::sqrt(na * nb) + 1e-12);
}
static bool run_one(const char* gguf,
                    const char* base,
                    const std::string& type,
                    double& dmax,
                    double& dcorr,
                    double& emax) {
    std::string out = std::string(gguf) + "." + type + ".acc.gguf";
    if (!aicore::depth::quantize_gguf(gguf, out, type)) {
        std::fprintf(stderr, "quantize %s failed\n", type.c_str());
        return false;
    }
    auto eng = aicore::depth::Engine::load(out, 0);
    if (!eng) {
        std::remove(out.c_str());
        return false;
    }
    std::vector<float> raw, refd, refe;
    std::vector<int64_t> s;
    da_parity::load_baseline(base, "raw_image", raw, s);
    da_parity::load_baseline(base, "head_depth", refd, s);
    da_parity::load_baseline(base, "extrinsics", refe, s);
    aicore::depth::Image img;
    img.w = 224;
    img.h = 224;
    img.rgb.resize(224 * 224 * 3);
    for (size_t i = 0; i < img.rgb.size(); ++i)
        img.rgb[i] = (unsigned char)(raw[i] + 0.5f);
    std::vector<float> depth, conf;
    std::array<float, 12> ext;
    std::array<float, 9> intr;
    int H, W;
    bool ok = eng->depth_pose(img, depth, conf, ext, intr, H, W);
    std::remove(out.c_str());
    if (!ok) return false;
    dmax = 0;
    for (size_t i = 0; i < depth.size(); ++i)
        dmax = std::max(dmax, (double)std::fabs(depth[i] - refd[i]));
    dcorr = corr(depth, refd);
    emax = 0;
    for (int i = 0; i < 12; ++i)
        emax = std::max(emax, (double)std::fabs(ext[i] - refe[i]));
    return true;
}
int main() {
    const char* gguf = std::getenv("DA_TEST_GGUF");
    const char* base = std::getenv("DA_TEST_BASELINE");
    if (!gguf || !base) return 77;
    double q8d, q8c, q8e, q4d, q4c, q4e;
    if (!run_one(gguf, base, "q8_0", q8d, q8c, q8e)) return 1;
    if (!run_one(gguf, base, "q4_k", q4d, q4c, q4e)) return 1;
    std::fprintf(stderr, "q8_0: depth max|d|=%.4e corr=%.6f ext max|d|=%.4e\n",
                 q8d, q8c, q8e);
    std::fprintf(stderr, "q4_k: depth max|d|=%.4e corr=%.6f ext max|d|=%.4e\n",
                 q4d, q4c, q4e);
    bool ok = (q8d < 2e-2) && (q4c > 0.99);
    std::fprintf(stderr, "quant accuracy -> %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
