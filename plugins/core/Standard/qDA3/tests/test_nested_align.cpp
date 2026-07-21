// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// M6-T3 gate: nested metric alignment host math (src/nested.cpp). Loads the
// anyview+metric branch inputs from the nested baseline dump, runs
// NestedAligner, and compares depth_final / scale_factor / extrinsics_final vs
// the reference at 2e-3 (rel). This isolates the alignment math from the slow
// backbones. SKIP (77) if DA_TEST_BASELINE_NESTED is absent.
#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "nested.hpp"
#include "parity.hpp"

int main() {
    const char* base = std::getenv("DA_TEST_BASELINE_NESTED");
    if (!base) return 77;
    const int H = 224, W = 224;

    auto loadv = [&](const char* name, std::vector<float>& out) -> bool {
        std::vector<int64_t> s;
        return da_parity::load_baseline(base, name, out, s);
    };

    aicore::depth::AnyviewOut any;
    aicore::depth::MetricOut metric;
    std::vector<float> ext_any, intr_any;
    if (!loadv("depth_any", any.depth)) return 1;
    if (!loadv("depth_conf_any", any.depth_conf)) return 1;
    if (!loadv("extrinsics_any", ext_any)) return 1;
    if (!loadv("intrinsics_any", intr_any)) return 1;
    if (!loadv("depth_metric_raw", metric.depth)) return 1;
    if (!loadv("sky", metric.sky)) return 1;
    if (ext_any.size() != 12 || intr_any.size() != 9) {
        std::fprintf(stderr, "bad extrinsics/intrinsics shapes %zu %zu\n",
                     ext_any.size(), intr_any.size());
        return 1;
    }
    for (int i = 0; i < 12; ++i) any.extrinsics[i] = ext_any[i];
    for (int i = 0; i < 9; ++i) any.intrinsics[i] = intr_any[i];

    aicore::depth::NestedOut res =
            aicore::depth::NestedAligner::align(any, metric, H, W);

    bool ok = true;
    {
        std::vector<float> ref;
        if (!loadv("depth_final", ref)) return 1;
        ok &= da_parity::compare(res.depth, ref, "depth_final", 2e-3f, 2e-3f);
    }
    {
        std::vector<float> ref;
        if (!loadv("scale_factor", ref)) return 1;
        std::vector<float> got = {res.scale_factor};
        ok &= da_parity::compare(got, ref, "scale_factor", 2e-3f, 2e-3f);
    }
    {
        std::vector<float> ref;
        if (!loadv("extrinsics_final", ref)) return 1;
        std::vector<float> got(res.extrinsics.begin(), res.extrinsics.end());
        ok &= da_parity::compare(got, ref, "extrinsics_final", 2e-3f, 2e-3f);
    }
    return ok ? 0 : 1;
}
