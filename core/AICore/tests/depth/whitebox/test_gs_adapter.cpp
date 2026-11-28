// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// M5-T4 gate: GaussianAdapter (host geometry -> world-space 3D Gaussians).
// Feeds the dumped giant raw_gs/gs_conf/depth/extrinsics/intrinsics into the
// pure-host GsAdapter and gates each output attribute (means/scales/rotations/
// harmonics/opacities) against the dumped reference at 2e-3 (atol+rtol).
// SKIP (77) if the giant baseline is absent.
#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <string>
#include <vector>

#include "tasks/depth/gs_adapter.hpp"
#include "tests/depth/whitebox/parity.hpp"

TEST(DepthWhitebox, GsAdapter) {
    const char* base = std::getenv("AICORE_TEST_DEPTH_BASELINE_GIANT");
    if (!base) GTEST_SKIP() << "Missing configured model/reference assets";
    const int H = 224, W = 224;
    std::vector<int64_t> s;

    std::vector<float> raw_gs, gs_conf, depth, ext_v, intr_v;
    if (!da_parity::load_baseline(base, "raw_gs", raw_gs, s))
        FAIL() << "Configured reference asset could not be loaded";
    if (!da_parity::load_baseline(base, "gs_conf", gs_conf, s))
        FAIL() << "Required operation or numeric contract failed";
    if (!da_parity::load_baseline(base, "depth_g", depth, s))
        FAIL() << "Required operation or numeric contract failed";
    if (!da_parity::load_baseline(base, "extrinsics_g", ext_v, s))
        FAIL() << "Required operation or numeric contract failed";
    if (!da_parity::load_baseline(base, "intrinsics_g", intr_v, s))
        FAIL() << "Required operation or numeric contract failed";
    if (ext_v.size() != 12 || intr_v.size() != 9) {
        std::fprintf(stderr, "unexpected ext/intr sizes %zu %zu\n",
                     ext_v.size(), intr_v.size());
        FAIL() << "Required operation or numeric contract failed";
    }
    std::array<float, 12> ext;
    for (int i = 0; i < 12; i++) ext[i] = ext_v[i];
    std::array<float, 9> intr;
    for (int i = 0; i < 9; i++) intr[i] = intr_v[i];

    aicore::depth::GsAdapter ad;
    aicore::depth::Gaussians g;
    if (!ad.build(raw_gs, depth, gs_conf, ext, intr, H, W, g)) {
        std::fprintf(stderr, "GsAdapter::build failed\n");
        FAIL() << "Required operation or numeric contract failed";
    }

    {
        std::vector<float> ref;
        if (da_parity::load_baseline(base, "gs_means", ref, s))
            EXPECT_TRUE(
                    da_parity::compare(g.means, ref, "gs_means", 2e-3f, 2e-3f));
        else
            FAIL() << "Required reference tensor is absent";
    }
    {
        std::vector<float> ref;
        if (da_parity::load_baseline(base, "gs_scales", ref, s))
            EXPECT_TRUE(da_parity::compare(g.scales, ref, "gs_scales", 2e-3f,
                                           2e-3f));
        else
            FAIL() << "Required reference tensor is absent";
    }
    {
        std::vector<float> ref;
        if (da_parity::load_baseline(base, "gs_rotations", ref, s))
            EXPECT_TRUE(da_parity::compare(g.rotations, ref, "gs_rotations",
                                           2e-3f, 2e-3f));
        else
            FAIL() << "Required reference tensor is absent";
    }
    {
        std::vector<float> ref;
        if (da_parity::load_baseline(base, "gs_harmonics", ref, s))
            EXPECT_TRUE(da_parity::compare(g.harmonics, ref, "gs_harmonics",
                                           2e-3f, 2e-3f));
        else
            FAIL() << "Required reference tensor is absent";
    }
    {
        std::vector<float> ref;
        if (da_parity::load_baseline(base, "gs_opacities", ref, s))
            EXPECT_TRUE(da_parity::compare(g.opacities, ref, "gs_opacities",
                                           2e-3f, 2e-3f));
        else
            FAIL() << "Required reference tensor is absent";
    }
}
