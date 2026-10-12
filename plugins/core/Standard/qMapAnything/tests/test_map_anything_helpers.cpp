// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Focused helper tests for qMapAnything: the pure pose/depth->cloud math
// (no model load) and the catalog->variant mapping (gtest, same pattern as
// test_qyolo_helpers).

#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QSet>
#include <QTemporaryDir>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include "MapAnythingGeometry.h"
#include "MapAnythingModelCatalog.h"
#include "MapAnythingReconstruction.h"
#include "MapAnythingSceneMeta.h"

using namespace MapAnythingGeometry;

TEST(QMapAnythingGeometry, QuatIdentity) {
    const float q[4] = {0, 0, 0, 1};
    float R[9];
    quat_xyzw_to_mat3(q, R);
    const float I[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    for (int i = 0; i < 9; ++i) {
        ASSERT_NEAR(R[i], I[i], 1e-6f);
    }
}

TEST(QMapAnythingGeometry, Quat90DegZ) {
    // +90 deg around Z (xyzw) -> X axis maps onto Y.
    const float s = std::sqrt(2.0f) / 2.0f;
    const float q[4] = {0, 0, s, s};
    float R[9];
    quat_xyzw_to_mat3(q, R);
    const float x[3] = {1, 0, 0};
    float y[3];
    for (int r = 0; r < 3; ++r) {
        y[r] = R[r * 3 + 0] * x[0] + R[r * 3 + 1] * x[1] + R[r * 3 + 2] * x[2];
    }
    ASSERT_NEAR(y[0], 0.0f, 1e-6f);
    ASSERT_NEAR(y[1], 1.0f, 1e-6f);
    ASSERT_NEAR(y[2], 0.0f, 1e-6f);
}

TEST(QMapAnythingGeometry, FovIntrinsics) {
    // 90 deg horizontal FoV over a 100 px wide grid -> fx = (W/2)/tan(45deg)
    // = 50; cx/cy land in the grid center.
    float k[4];
    fov_to_intrinsics(1.0f, float(M_PI) / 2.0f, 100, 80, k);
    ASSERT_NEAR(k[0], 50.0f, 1e-4f);
    ASSERT_NEAR(k[2], 50.0f, 1e-5f);
    ASSERT_NEAR(k[3], 40.0f, 1e-5f);
}

TEST(QMapAnythingGeometry, UnprojectIdentityCamera) {
    // Camera at the world origin: the image-center depth d unprojects to
    // (0, 0, d) in the world (identity w2c: world = R^T*(cam - T)).
    const float pose9[9] = {0, 0, 0, 0, 0, 0, 1, 1.0f, 1.0f};
    float R[9], t[3], k[4];
    pose_enc_to_w2c(pose9, R, t);
    fov_to_intrinsics(pose9[7], pose9[8], 100, 100, k);
    float world[3];
    unproject_depth_pixel(R, t, k, 50, 50, 2.0f, world);
    ASSERT_NEAR(world[0], 0.0f, 1e-5f);
    ASSERT_NEAR(world[1], 0.0f, 1e-5f);
    ASSERT_NEAR(world[2], 2.0f, 1e-5f);
}

TEST(QMapAnythingGeometry, UnprojectInvertsW2c) {
    // 90-degree yaw (quat xyzw = (0, sin45, 0, cos45)) + translation.
    // The upstream pose_enc.py defines [R|T] as camera-from-world, so a
    // camera point is mapped by world = R^T * (cam - T).
    const float s45 = 0.70710678f;
    const float pose9[9] = {1, 2, 3, 0, s45, 0, s45, 1.0f, 1.0f};
    float R[9], t[3], k[4];
    pose_enc_to_w2c(pose9, R, t);
    fov_to_intrinsics(pose9[7], pose9[8], 100, 100, k);
    float world[3];
    unproject_depth_pixel(R, t, k, 50, 50, 2.0f, world);
    ASSERT_NEAR(world[0], 1.0f, 1e-4f);
    ASSERT_NEAR(world[1], -2.0f, 1e-4f);
    ASSERT_NEAR(world[2], -1.0f, 1e-4f);
}

TEST(QMapAnythingGeometry, SigmoidExtremes) {
    ASSERT_FLOAT_EQ(sigmoid(0.0f), 0.5f);
    ASSERT_GT(sigmoid(100.0f), 0.99f);
    ASSERT_LT(sigmoid(-100.0f), 0.01f);
}

TEST(QMapAnythingGeometry, ConfPriorityVoxelKeepsHighestConfidence) {
    // Points 0 and 1 share one 1 m voxel; the higher-confidence point 1
    // must survive regardless of its rank in the descending-confidence pass.
    float pts[] = {0.1f, 0.1f, 0.1f, 0.2f, 0.2f, 0.2f, 1.1f, 0.1f, 0.1f};
    unsigned char cols[] = {10, 0, 0, 20, 0, 0, 30, 0, 0};
    float conf[] = {1.0f, 3.0f, 2.0f};
    const size_t count = conf_priority_voxel(pts, cols, conf, 3, 1.0f);
    ASSERT_EQ(count, 2U);
    EXPECT_FLOAT_EQ(pts[0], 0.2f);
    EXPECT_EQ(cols[0], 20);
    EXPECT_FLOAT_EQ(pts[3], 1.1f);
    EXPECT_EQ(cols[3], 30);
}

TEST(QMapAnythingGeometry, ConfPriorityVoxelReportsOriginalSurvivors) {
    float pts[] = {0.1f, 0.1f, 0.1f, 0.2f, 0.2f, 0.2f, 1.1f, 0.1f, 0.1f};
    unsigned char cols[] = {10, 0, 0, 20, 0, 0, 30, 0, 0};
    float conf[] = {1.0f, 3.0f, 2.0f};
    std::vector<size_t> survivors;
    const size_t count =
            conf_priority_voxel(pts, cols, conf, 3, 1.0f, &survivors);
    ASSERT_EQ(count, 2U);
    ASSERT_EQ(survivors.size(), 2U);
    EXPECT_EQ(survivors[0], 1U);
    EXPECT_EQ(survivors[1], 2U);
}

TEST(QMapAnythingGeometry, C2wTransform) {
    // Pure translation: world = cam + t.
    const float pose16[16] = {1, 0, 0, 5, 0, 1, 0, -2, 0, 0, 1, 3, 0, 0, 0, 1};
    const float cam[3] = {1, 1, 1};
    float world[3];
    transform_by_c2w(pose16, cam, world);
    ASSERT_FLOAT_EQ(world[0], 6.0f);
    ASSERT_FLOAT_EQ(world[1], -1.0f);
    ASSERT_FLOAT_EQ(world[2], 4.0f);
}

TEST(QMapAnythingCatalog, VariantMapping) {
    const auto models = MapAnythingHelpers::catalogModels();
    if (models.isEmpty()) {
        GTEST_SKIP() << "AICore catalog unavailable in this build";
    }
    // Quantization layout mirrors the published HF repo: 36 rows total -
    // omega family + mapanything ship all five quants; pi3/vggt-1b dropped
    // q6_K, pi3x/dust3r dropped q5_K.
    EXPECT_EQ(models.size(), 36);
    const QStringList variants = MapAnythingHelpers::catalogVariants();
    EXPECT_EQ(variants.size(), 8);
    // The catalog contract: first variant = the vggt-omega 512 accuracy
    // pick; its rows carry the "(recommended)" quant marker.
    EXPECT_EQ(variants.first(), QStringLiteral("vggt-omega-1b-512"));
    const auto omega = MapAnythingHelpers::modelsForVariant(
            QStringLiteral("vggt-omega-1b-512"));
    EXPECT_EQ(omega.size(), 5);
    bool hasRecommended = false;
    for (const auto& m : omega) {
        if (m.quantNote.contains(QStringLiteral("(recommended)"))) {
            hasRecommended = true;
        }
    }
    EXPECT_TRUE(hasRecommended);
}

// Regression guard for the combo bug: displayName alone is constant across
// the quantization rows of one variant, so the display label MUST differ
// per row (it embeds the quant note) and the default row must be the
// recommended q8_0 pick.
TEST(QMapAnythingCatalog, DisplayLabelsDistinguishQuantizations) {
    const auto models = MapAnythingHelpers::catalogModels();
    if (models.isEmpty()) {
        GTEST_SKIP() << "AICore catalog unavailable in this build";
    }
    for (const auto& variant : MapAnythingHelpers::catalogVariants()) {
        const auto rows = MapAnythingHelpers::modelsForVariant(variant);
        ASSERT_GE(rows.size(), 4);
        QSet<QString> labels;
        QSet<QString> quants;
        for (const auto& m : rows) {
            const QString label = MapAnythingHelpers::modelDisplayLabel(m);
            EXPECT_FALSE(label.isEmpty());
            // The label must carry row-level info beyond the variant-level
            // displayName (the exact bug: 4 identical combo items).
            EXPECT_TRUE(label.contains(m.quantNote))
                    << "label missing quant note: " << label.toStdString();
            labels.insert(label);
            for (const QString& q :
                 {QStringLiteral("f16"), QStringLiteral("f32"),
                  QStringLiteral("q8_0"), QStringLiteral("q6_K"),
                  QStringLiteral("q5_K")}) {
                if (m.filename.contains(q)) quants.insert(q);
            }
        }
        EXPECT_EQ(labels.size(), rows.size())
                << "indistinguishable combo labels for variant "
                << variant.toStdString();
        // The published layout DIVERGED per variant (HF update): omega
        // family + mapanything ship all five quants, pi3/vggt-1b dropped
        // q6_K, pi3x/dust3r dropped q5_K. Pin the exact per-variant set —
        // any catalog drift from the remote repo trips this.
        static const QHash<QString, QSet<QString>> expected = {
                {QStringLiteral("vggt-omega-1b-512"),
                 {QStringLiteral("f16"), QStringLiteral("q8_0"),
                  QStringLiteral("q6_K"), QStringLiteral("q5_K"),
                  QStringLiteral("f32")}},
                {QStringLiteral("vggt-omega-1b-416-reproduce"),
                 {QStringLiteral("f16"), QStringLiteral("q8_0"),
                  QStringLiteral("q6_K"), QStringLiteral("q5_K"),
                  QStringLiteral("f32")}},
                {QStringLiteral("vggt-omega-1b-256-text"),
                 {QStringLiteral("f16"), QStringLiteral("q8_0"),
                  QStringLiteral("q6_K"), QStringLiteral("q5_K"),
                  QStringLiteral("f32")}},
                {QStringLiteral("mapanything"),
                 {QStringLiteral("f16"), QStringLiteral("q8_0"),
                  QStringLiteral("q6_K"), QStringLiteral("q5_K"),
                  QStringLiteral("f32")}},
                {QStringLiteral("pi3"),
                 {QStringLiteral("f16"), QStringLiteral("q8_0"),
                  QStringLiteral("q5_K"), QStringLiteral("f32")}},
                {QStringLiteral("pi3x"),
                 {QStringLiteral("f16"), QStringLiteral("q8_0"),
                  QStringLiteral("q6_K"), QStringLiteral("f32")}},
                {QStringLiteral("vggt-1b"),
                 {QStringLiteral("f16"), QStringLiteral("q8_0"),
                  QStringLiteral("q5_K"), QStringLiteral("f32")}},
                {QStringLiteral("dust3r"),
                 {QStringLiteral("f16"), QStringLiteral("q8_0"),
                  QStringLiteral("q6_K"), QStringLiteral("f32")}},
        };
        ASSERT_TRUE(expected.contains(variant))
                << "unknown variant in test table: " << variant.toStdString();
        EXPECT_EQ(quants, expected.value(variant))
                << "quantization layout drift for variant "
                << variant.toStdString();
    }
    // The catalog default row is a published filename with a recommended
    // (q8_0) quant note.
    const int def = MapAnythingHelpers::defaultModelIndex();
    ASSERT_GE(def, 0);
    ASSERT_LT(def, models.size());
    EXPECT_TRUE(
            models[def].quantNote.contains(QStringLiteral("(recommended)")));
    EXPECT_TRUE(models[def].filename.contains(QStringLiteral("q8_0")));
}

// Official end-to-end pipeline parity contract (per-architecture demo
// defaults, verified against the upstream sources): the catalog must carry
// each family's official resize mode, confidence-filter semantics and
// default threshold — the dialog and the headless command seed from these
// rows, so any drift here silently changes the pipeline config.
TEST(QMapAnythingCatalog, OfficialPipelineDefaults) {
    const auto models = MapAnythingHelpers::catalogModels();
    if (models.isEmpty()) {
        GTEST_SKIP() << "AICore catalog unavailable in this build";
    }
    auto rowOf = [&](const QString& filename) -> const MapAnythingModelEntry* {
        for (const auto& m : models)
            if (m.filename == filename) return &m;
        return nullptr;
    };
    struct Expectation {
        const char* filename;
        const char* variant;
        const char* resize;
        const char* filter;
        float minConf;
    };
    const Expectation kExpected[] = {
            // vggt-omega demo_gradio: balanced 512/16 + raw conf >= 3.0
            {"vggt-omega-1b-512-q8_0.gguf", "vggt-omega-1b-512", "balanced",
             "normalized", 0.67f},
            // Original VGGT preprocessing is black center-pad square 518.
            // Reconstruction uses the upstream e2e floor max(2, p20): on the
            // normalized (c-1)/c scale, 0.5 is exactly the c >= 2 floor.
            {"vggt-1b-q8_0.gguf", "vggt-1b", "square", "normalized", 0.5f},
            // MapAnything gradio: balanced + conf percentile 3.0
            {"mapanything-q8_0.gguf", "mapanything", "balanced", "percentile",
             3.0f},
            // Pi3: balanced; its conf head is already normalized
            {"pi3-q8_0.gguf", "pi3", "balanced", "normalized", 0.5f},
            // DUSt3R: official pair-wise pipeline is strictly 2 views
            {"dust3r-q8_0.gguf", "dust3r", "balanced", "normalized", 0.5f},
    };
    for (const auto& e : kExpected) {
        const auto* m = rowOf(QString::fromUtf8(e.filename));
        ASSERT_NE(m, nullptr) << e.filename;
        EXPECT_EQ(m->variant.toStdString(), e.variant) << e.filename;
        EXPECT_EQ(m->defaultResizeMode.toStdString(), e.resize) << e.filename;
        EXPECT_EQ(m->confFilter.toStdString(), e.filter) << e.filename;
        EXPECT_FLOAT_EQ(m->defaultMinConf, e.minConf) << e.filename;
    }
}

// ---- shared GUI/headless driver helpers ------------------------------------

TEST(QMapAnythingGeometry, PercentileIsLinearInterpolated) {
    // np.percentile / torch.quantile default (linear): rank p/100*(m-1).
    const float v[] = {4.0f, 1.0f, NAN, 3.0f, 2.0f};
    EXPECT_FLOAT_EQ(percentile_value(v, 5, 50.0f), 2.5f);
    EXPECT_FLOAT_EQ(percentile_value(v, 5, 0.0f), 1.0f);
    EXPECT_FLOAT_EQ(percentile_value(v, 5, 100.0f), 4.0f);
    EXPECT_FLOAT_EQ(percentile_value(v, 5, 20.0f), 1.6f);
    const float none[] = {NAN, INFINITY};
    EXPECT_TRUE(std::isnan(percentile_value(none, 2, 50.0f)));
}

TEST(QMapAnythingGeometry, OfficialConfFloorClampsAtTwo) {
    const float low[] = {1.0f, 1.1f, 1.2f};
    EXPECT_FLOAT_EQ(official_conf_floor(low, 3, 20.0f), 2.0f);
    const float high[] = {3.0f, 4.0f, 5.0f, 6.0f};
    EXPECT_FLOAT_EQ(official_conf_floor(high, 4, 20.0f), 3.6f);
}

TEST(QMapAnythingGeometry, PlanChunkStartsMatchesUpstreamWindow) {
    EXPECT_EQ(plan_chunk_starts(38, 64, 2), std::vector<int>({0}));
    EXPECT_EQ(plan_chunk_starts(38, 38, 2), std::vector<int>({0}));
    // range(0, 38, 14) -> 0, 14, 28 -> clamp 28 to 38-16 = 22.
    EXPECT_EQ(plan_chunk_starts(38, 16, 2), std::vector<int>({0, 14, 22}));
    EXPECT_EQ(plan_chunk_starts(20, 16, 2), std::vector<int>({0, 4}));
    // Every later chunk shares >= overlap frames with its predecessor.
    const std::vector<int> s = plan_chunk_starts(101, 7, 3);
    for (size_t i = 1; i < s.size(); ++i) EXPECT_LE(s[i] - s[i - 1], 7 - 3);
    EXPECT_EQ(s.back(), 101 - 7);
    EXPECT_TRUE(plan_chunk_starts(0, 16, 2).empty());
}

namespace {
void applySim3(const Sim3& s, const float* p, float* out) {
    apply_sim3(s.R, s.t, s.s, p, 1, out);
}
Sim3 testSim3(float yaw, float scale, float tx, float ty, float tz) {
    Sim3 s;
    s.s = scale;
    const float c = std::cos(yaw), si = std::sin(yaw);
    const float R[9] = {c, -si, 0, si, c, 0, 0, 0, 1};
    std::copy(R, R + 9, s.R);
    s.t[0] = tx;
    s.t[1] = ty;
    s.t[2] = tz;
    return s;
}
}  // namespace

TEST(QMapAnythingGeometry, Sim3ComposeAppliesRightFirst) {
    const Sim3 a = testSim3(0.3f, 2.0f, 1, -2, 0.5f);
    const Sim3 b = testSim3(-1.1f, 0.5f, -3, 0.25f, 4);
    const float p[3] = {0.7f, -1.3f, 2.2f};
    float bp[3], abp[3], cp[3];
    applySim3(b, p, bp);
    applySim3(a, bp, abp);
    applySim3(sim3_compose(a, b), p, cp);
    for (int k = 0; k < 3; ++k) EXPECT_NEAR(cp[k], abp[k], 1e-5f);
}

TEST(QMapAnythingGeometry, Sim3MovesCameraCenterAndKeepsRotation) {
    const Sim3 s = testSim3(0.9f, 3.0f, 1, 2, 3);
    const float s45 = 0.70710678f;
    const float pose9[9] = {1, 2, 3, 0, s45, 0, s45, 1.0f, 1.0f};
    float c2w[16], moved[16];
    pose_enc_to_c2w(pose9, c2w);
    sim3_apply_c2w(s, c2w, moved);
    const float center[3] = {c2w[3], c2w[7], c2w[11]};
    float expect[3];
    applySim3(s, center, expect);
    EXPECT_NEAR(moved[3], expect[0], 1e-5f);
    EXPECT_NEAR(moved[7], expect[1], 1e-5f);
    EXPECT_NEAR(moved[11], expect[2], 1e-5f);
    // Rotation block stays orthonormal (scale acts on the position only).
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            float dot = 0.0f;
            for (int k = 0; k < 3; ++k)
                dot += moved[4 * k + i] * moved[4 * k + j];
            EXPECT_NEAR(dot, i == j ? 1.0f : 0.0f, 1e-5f);
        }
    }
    EXPECT_FLOAT_EQ(moved[15], 1.0f);
}

TEST(QMapAnythingGeometry, PoseEncC2wInvertsW2c) {
    const float pose9[9] = {0.4f,  -1.0f,     2.5f, 0.1f, 0.2f,
                            -0.3f, 0.927362f, 1.0f, 1.0f};
    float R[9], t[3], c2w[16];
    pose_enc_to_w2c(pose9, R, t);
    pose_enc_to_c2w(pose9, c2w);
    const float x[3] = {1.5f, -0.5f, 3.0f};
    float cam[3], back[3];
    for (int r = 0; r < 3; ++r) {
        cam[r] = R[3 * r] * x[0] + R[3 * r + 1] * x[1] + R[3 * r + 2] * x[2] +
                 t[r];
    }
    transform_by_c2w(c2w, cam, back);
    for (int k = 0; k < 3; ++k) EXPECT_NEAR(back[k], x[k], 1e-5f);
}

TEST(QMapAnythingGeometry, RobustSim3RecoversTransformDespiteOutlier) {
    const Sim3 truth = testSim3(0.6f, 1.7f, -2, 0.5f, 3);
    std::vector<float> src, dst;
    for (int i = 0; i < 40; ++i) {
        const float p[3] = {float(i % 5), float((i / 5) % 4) * 0.7f,
                            float(i % 3) - 1.0f + 0.1f * i};
        float q[3];
        applySim3(truth, p, q);
        if (i == 7) q[0] += 50.0f;  // gross outlier
        src.insert(src.end(), p, p + 3);
        dst.insert(dst.end(), q, q + 3);
    }
    Sim3 fit;
    ASSERT_EQ(robust_sim3(src.data(), dst.data(), 40, &fit.s, fit.R, fit.t), 0);
    EXPECT_NEAR(fit.s, truth.s, 1e-3f);
    for (int k = 0; k < 9; ++k) EXPECT_NEAR(fit.R[k], truth.R[k], 1e-3f);
    for (int k = 0; k < 3; ++k) EXPECT_NEAR(fit.t[k], truth.t[k], 1e-2f);
}

namespace {
namespace MR = MapAnythingReconstruction;
// Cloud with one point per confidence, x = conf; population == the points.
MR::Cloud makeCloud(const std::vector<std::vector<float>>& perView) {
    MR::Cloud c;
    c.populationOffset.push_back(0);
    for (size_t v = 0; v < perView.size(); ++v) {
        for (float conf : perView[v]) {
            c.xyz.insert(c.xyz.end(), {conf, 0.0f, 0.0f});
            c.rgb.insert(c.rgb.end(), {1, 2, 3});
            c.conf.push_back(conf);
            c.view.push_back((int32_t)v);
            c.confPopulation.push_back(conf);
        }
        c.populationOffset.push_back(c.confPopulation.size());
    }
    return c;
}
}  // namespace

TEST(QMapAnythingFilter, MapAnythingIsPerViewQuantileStrict) {
    // Per-view p50 = 2.5 / 25: a scene-wide threshold (p50 = 7) would
    // erase view 0 entirely - the old "0 points" defect class.
    MR::Cloud c = makeCloud({{1, 2, 3, 4}, {10, 20, 30, 40}});
    MR::FilterOptions o;
    o.minConf = 50.0f;
    const MR::FilterReport r = MR::filter_scene(c, "mapanything", o, true);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.rule, "percentile/view");
    EXPECT_TRUE(std::isnan(r.confFloor));
    EXPECT_EQ(r.input, 8U);
    ASSERT_EQ(r.kept, 4U);
    EXPECT_EQ(c.conf, std::vector<float>({3, 4, 30, 40}));
    EXPECT_EQ(c.view, std::vector<int32_t>({0, 0, 1, 1}));
    // Strict '>': p0 is the per-view minimum, which is itself dropped.
    MR::Cloud c0 = makeCloud({{1, 2, 3, 4}, {10, 20, 30, 40}});
    o.minConf = 0.0f;
    EXPECT_EQ(MR::filter_scene(c0, "mapanything", o, false).kept, 6U);
}

TEST(QMapAnythingFilter, OneExpFamiliesUseSceneFloorThenNormalizedGate) {
    MR::Cloud c = makeCloud({{1.5f, 2.5f}, {3.0f, 5.0f}});
    MR::FilterOptions o;
    o.minConf = 0.0f;
    o.confPercentile = 20.0f;  // linear p20 = 2.1 > 2
    const MR::FilterReport r = MR::filter_scene(c, "vggt_omega", o, false);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.rule, "floor+normalized");
    EXPECT_NEAR(r.confFloor, 2.1f, 1e-6f);
    EXPECT_EQ(r.kept, 3U);
    EXPECT_FALSE(r.voxelApplied);  // learned frame: metric voxel is off
    // (c-1)/c >= 0.7: only 5 (0.8) survives; 3 (0.667) does not.
    MR::Cloud c2 = makeCloud({{1.5f, 2.5f}, {3.0f, 5.0f}});
    o.minConf = 0.7f;
    EXPECT_EQ(MR::filter_scene(c2, "dust3r", o, false).kept, 1U);
}

TEST(QMapAnythingFilter, Pi3UsesStrictSigmoidOnLogits) {
    MR::Cloud c = makeCloud({{-1.0f, 0.0f, 2.0f}});
    MR::FilterOptions o;
    o.minConf = 0.5f;
    const MR::FilterReport r = MR::filter_scene(c, "pi3x", o, false);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.rule, "sigmoid");
    EXPECT_EQ(c.conf, std::vector<float>({2.0f}));
}

TEST(QMapAnythingFilter, UnknownArchitectureFailsClosed) {
    MR::Cloud c = makeCloud({{1, 2}});
    const MR::FilterReport r = MR::filter_scene(c, "mystery", {}, true);
    EXPECT_FALSE(r.ok);
    EXPECT_EQ(c.size(), 2U);  // untouched
}

TEST(QMapAnythingFilter, VoxelOnlyInMetricFrame) {
    MR::Cloud c;
    // Two points in one 0.02 voxel (first has the higher conf) + one apart.
    c.xyz = {0.001f, 0, 0, 0.002f, 0, 0, 1.0f, 0, 0};
    c.rgb = {1, 1, 1, 2, 2, 2, 3, 3, 3};
    c.view = {0, 0, 0};
    MR::FilterReport r = MR::filter_scene(c, "vggt", {}, false);
    EXPECT_EQ(r.rule, "none");
    EXPECT_FALSE(r.voxelApplied);
    r = MR::filter_scene(c, "vggt", {}, true);
    ASSERT_TRUE(r.voxelApplied);
    EXPECT_EQ(r.fused.size(), 2U);
    EXPECT_EQ(r.kept, 3U);
}

TEST(QMapAnythingFilter, WritePlyHeaderAndPayload) {
    MR::Cloud c = makeCloud({{1, 2, 3}});
    const std::string path =
            ::testing::TempDir() + "qmapanything_write_ply_test.ply";
    const std::vector<size_t> idx = {0, 2};
    ASSERT_TRUE(MR::write_ply(path, c, &idx));
    std::ifstream in(path, std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());
    const std::string marker = "end_header\n";
    const size_t body = bytes.find(marker);
    ASSERT_NE(body, std::string::npos);
    EXPECT_NE(bytes.find("element vertex 2\n"), std::string::npos);
    EXPECT_EQ(bytes.size() - body - marker.size(), 2U * (3 * 4 + 3));
    float x1 = 0.0f;
    std::memcpy(&x1, bytes.data() + body + marker.size() + 15, 4);
    EXPECT_FLOAT_EQ(x1, 3.0f);  // second record is index 2
    std::remove(path.c_str());
}

TEST(QMapAnythingGeometry, MedianIsNumpyMedian) {
    EXPECT_FLOAT_EQ(median({3, 1, 2}), 2.0f);
    EXPECT_FLOAT_EQ(median({4, 1, 3, 2}), 2.5f);  // mean of the two middles
    EXPECT_TRUE(std::isnan(median({})));
}

TEST(QMapAnythingAnchor, MapsCamerasAndCloudIntoReferenceFrame) {
    // Model frame = reference frame seen through `truth`^-1: the anchor must
    // recover `truth` from the camera centers alone and move the cloud and
    // every camera with it. View 2 has no reference (NaN) and is skipped.
    const Sim3 truth = testSim3(0.4f, 11.3f, 4, -5.6f, 1.7f);
    MR::Result r;
    r.views = 5;
    r.c2w.assign(5 * 16, 0.0f);
    std::vector<float> ref(5 * 3);
    const float centers[5][3] = {
            {0, 0, 0}, {1, 0, 0.2f}, {2, 0.5f, 0}, {0.3f, 1, 1}, {1, 1, 2}};
    for (int v = 0; v < 5; ++v) {
        float* M = &r.c2w[(size_t)v * 16];
        M[0] = M[5] = M[10] = M[15] = 1.0f;
        M[3] = centers[v][0];
        M[7] = centers[v][1];
        M[11] = centers[v][2];
        applySim3(truth, centers[v], &ref[(size_t)v * 3]);
    }
    ref[6] = std::numeric_limits<float>::quiet_NaN();
    r.cloud.xyz = {0.5f, 0.5f, 0.5f};
    float expect[3];
    applySim3(truth, r.cloud.xyz.data(), expect);

    MR::ReferenceAnchor a;
    ASSERT_TRUE(MR::anchor_to_reference(r, ref, &a));
    EXPECT_EQ(a.pairs, 4);
    EXPECT_NEAR(a.sim3.s, truth.s, 1e-3f);
    EXPECT_LT(a.residualMedian, 1e-3f);
    for (int k = 0; k < 3; ++k) EXPECT_NEAR(r.cloud.xyz[k], expect[k], 1e-3f);
    float c2[3];
    applySim3(truth, centers[2], c2);  // unreferenced view moves too
    EXPECT_NEAR(r.c2w[2 * 16 + 3], c2[0], 1e-3f);
    EXPECT_NEAR(r.c2w[2 * 16 + 7], c2[1], 1e-3f);
    EXPECT_NEAR(r.c2w[2 * 16 + 11], c2[2], 1e-3f);

    // < 3 usable pairs: refused, result untouched.
    MR::Result few = r;
    std::vector<float> sparse(5 * 3, std::numeric_limits<float>::quiet_NaN());
    std::copy(ref.begin(), ref.begin() + 6, sparse.begin());
    EXPECT_FALSE(MR::anchor_to_reference(few, sparse, nullptr));
    EXPECT_EQ(few.cloud.xyz, r.cloud.xyz);
}

TEST(QMapAnythingGeometry, Sim3InverseUndoes) {
    const Sim3 a = testSim3(0.7f, 2.5f, 1, -2, 3);
    const Sim3 id = sim3_compose(sim3_inverse(a), a);
    const float p[3] = {0.4f, -1.2f, 2.6f};
    float q[3];
    applySim3(id, p, q);
    for (int k = 0; k < 3; ++k) EXPECT_NEAR(q[k], p[k], 1e-5f);
}

TEST(QMapAnythingAnchor, StreamedSceneAnchorsEachChunkOnItsOwn) {
    // upstream e2e_full_scene.py --align gt: two chunks (views 0-2, 2-4)
    // whose model frames differ by a 1.4x scale drift. The chain left
    // chunk 1 in chunk 0's frame through an imperfect boundary fit
    // (`chain`); the per-chunk anchor must still put every view exactly on
    // its reference, which one Sim3 for the whole scene cannot.
    const float ref[5][3] = {
            {0, 0, 0}, {1, 0, 0.2f}, {2, 0.5f, 0}, {0.3f, 1, 1}, {1, 1, 2}};
    const Sim3 toChunk[2] = {testSim3(0.3f, 0.1f, 1, 2, 3),
                             testSim3(-0.5f, 0.14f, -1, 0, 2)};
    // Boundary fit = true chunk1 -> chunk0 map, 5% / 0.02 rad off.
    const Sim3 chain =
            sim3_compose(testSim3(0.02f, 1.05f, 0.01f, 0, 0),
                         sim3_compose(toChunk[0], sim3_inverse(toChunk[1])));
    MR::Result r;
    r.views = 5;
    r.c2w.assign(5 * 16, 0.0f);
    r.chunks.resize(2);
    for (int ci = 0; ci < 2; ++ci) {
        MR::Chunk& c = r.chunks[(size_t)ci];
        c.start = ci * 2;
        c.count = 3;
        if (ci == 1) c.toGlobal = chain;
        c.centers.resize(9);
        for (int i = 0; i < 3; ++i)
            applySim3(toChunk[ci], ref[c.start + i], &c.centers[(size_t)i * 3]);
    }
    // One cloud point per view at its camera, from its first chunk.
    for (int g = 0; g < 5; ++g) {
        const int ci = g < 3 ? 0 : 1;
        const MR::Chunk& c = r.chunks[(size_t)ci];
        float p[3];
        applySim3(c.toGlobal, &c.centers[(size_t)(g - c.start) * 3], p);
        float* M = &r.c2w[(size_t)g * 16];
        M[0] = M[5] = M[10] = M[15] = 1.0f;
        M[3] = p[0];
        M[7] = p[1];
        M[11] = p[2];
        r.cloud.xyz.insert(r.cloud.xyz.end(), p, p + 3);
        r.cloud.view.push_back(g);
    }
    std::vector<float> refs(&ref[0][0], &ref[0][0] + 15);

    MR::Result scene = r;
    MR::ReferenceAnchor a;
    std::string error;
    ASSERT_TRUE(MR::anchor_to_reference(r, refs, &a, &error)) << error;
    EXPECT_TRUE(a.perChunk);
    EXPECT_TRUE(a.perChunkRefused.empty());
    ASSERT_EQ(a.chunks.size(), 2u);
    EXPECT_EQ(a.chunks[0].pairs, 3);
    EXPECT_NEAR(a.chunks[0].sim3.s, 10.0f, 1e-2f);
    EXPECT_NEAR(a.chunks[1].sim3.s, 1.0f / 0.14f, 1e-2f);
    EXPECT_EQ(a.pairs, 5);
    EXPECT_LT(a.residualMedian, 1e-4f);
    for (int g = 0; g < 5; ++g) {
        for (int k = 0; k < 3; ++k) {
            EXPECT_NEAR(r.c2w[(size_t)g * 16 + 3 + 4 * k], ref[g][k], 1e-4f);
            EXPECT_NEAR(r.cloud.xyz[(size_t)g * 3 + k], ref[g][k], 1e-4f);
        }
    }
    EXPECT_NEAR(r.chunks[1].toGlobal.s, a.chunks[1].sim3.s, 1e-6f);

    // Chunk 1 with only 2 references: all-or-nothing -> one Sim3 for the
    // chained scene, reason recorded; that fit cannot absorb the drift.
    std::vector<float> partial = refs;
    partial[4 * 3] = std::numeric_limits<float>::quiet_NaN();
    ASSERT_TRUE(MR::anchor_to_reference(scene, partial, &a, &error)) << error;
    EXPECT_FALSE(a.perChunk);
    EXPECT_TRUE(a.chunks.empty());
    EXPECT_NE(a.perChunkRefused.find("chunk 1 (frames 2..4)"),
              std::string::npos)
            << a.perChunkRefused;
    EXPECT_GT(a.residualMedian, 1e-2f);
}

TEST(QMapAnythingAnchor, SceneMetaDiscoveryAndCenters) {
    QTemporaryDir tmp;
    ASSERT_TRUE(tmp.isValid());
    const QDir root(tmp.path());
    ASSERT_TRUE(root.mkpath(QStringLiteral("images")));
    ASSERT_TRUE(root.mkpath(QStringLiteral("other")));
    const QStringList images = {root.filePath("images/DSC_1.png"),
                                root.filePath("images/DSC_2.v2.png")};
    // No scene_meta.json yet -> model frame.
    EXPECT_TRUE(MapAnythingSceneMeta::discover(images).isEmpty());
    QFile meta(root.filePath("scene_meta.json"));
    ASSERT_TRUE(meta.open(QIODevice::WriteOnly));
    meta.write(R"({"frames":[
        {"frame_name":"DSC_1","transform_matrix":[[1,0,0,1.5],[0,1,0,-2],[0,0,1,3],[0,0,0,1]]},
        {"frame_name":"DSC_9","transform_matrix":[[1,0,0,9],[0,1,0,9],[0,0,1,9],[0,0,0,1]]},
        {"frame_name":"DSC_2.v2","transform_matrix":[[1,0,0,4],[0,1,0,5],[0,0,1,6],[0,0,0,1]]}]})");
    meta.close();
    const QString found = MapAnythingSceneMeta::discover(images);
    EXPECT_EQ(QFileInfo(found).canonicalFilePath(),
              QFileInfo(meta.fileName()).canonicalFilePath());
    // Mixed directories or a directory not named images/ -> no discovery.
    EXPECT_TRUE(MapAnythingSceneMeta::discover(
                        {images[0], root.filePath("other/DSC_3.png")})
                        .isEmpty());
    EXPECT_TRUE(MapAnythingSceneMeta::discover({root.filePath("other/a.png")})
                        .isEmpty());

    std::vector<float> c;
    QString error;
    const QStringList three =
            images + QStringList{root.filePath("images/x.png")};
    ASSERT_TRUE(MapAnythingSceneMeta::loadCenters(found, three, &c, &error));
    ASSERT_EQ(c.size(), 9U);
    EXPECT_EQ(std::vector<float>(c.begin(), c.begin() + 6),
              std::vector<float>({1.5f, -2, 3, 4, 5, 6}));
    EXPECT_TRUE(std::isnan(c[6]));  // frame absent from the meta
    EXPECT_FALSE(MapAnythingSceneMeta::loadCenters(
            root.filePath("missing.json"), images, &c, &error));
    EXPECT_FALSE(error.isEmpty());
}

namespace {

QString writeText(const QDir& dir, const QString& name, const char* text) {
    QFile f(dir.filePath(name));
    f.open(QIODevice::WriteOnly | QIODevice::Truncate);
    f.write(text);
    return f.fileName();
}

}  // namespace

TEST(QMapAnythingAnchor, Wgs84MatchesColmapGpsTransform) {
    // COLMAP src/colmap/geometry/gps_test.cc EllipsoidToECEFWGS84 vectors.
    const double lla[2][3] = {{48 + 8. / 60 + 51.70361 / 3600,
                               11 + 34. / 60 + 10.51777 / 3600, 561.1851},
                              {48 + 8. / 60 + 52.40575 / 3600,
                               11 + 34. / 60 + 11.77179 / 3600, 561.1509}};
    const double ecef[2][3] = {
            {4.177239709042750e6, 0.855153779923415e6, 4.728267404769168e6},
            {4.177218660452103e6, 0.855175931344048e6, 4.728281850382507e6}};
    for (int i = 0; i < 2; ++i) {
        double xyz[3];
        MapAnythingSceneMeta::wgs84ToEcef(lla[i], xyz);
        for (int k = 0; k < 3; ++k) EXPECT_NEAR(xyz[k], ecef[i][k], 1e-6);
    }
    // ENU about point 0 (COLMAP EllipsoidToENUWGS84): the rotated ECEF
    // difference, and the origin maps to zero.
    double enu[3];
    MapAnythingSceneMeta::wgs84ToEnu(lla[0], lla[0], enu);
    for (double e : enu) EXPECT_NEAR(e, 0.0, 1e-9);
    MapAnythingSceneMeta::wgs84ToEnu(lla[1], lla[0], enu);
    const double d[3] = {ecef[1][0] - ecef[0][0], ecef[1][1] - ecef[0][1],
                         ecef[1][2] - ecef[0][2]};
    EXPECT_NEAR(std::sqrt(enu[0] * enu[0] + enu[1] * enu[1] + enu[2] * enu[2]),
                std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]), 1e-6);
    EXPECT_GT(enu[0], 0.0);  // point 1 lies east (larger longitude) ...
    EXPECT_GT(enu[1], 0.0);  // ... and north (larger latitude)
}

TEST(QMapAnythingAnchor, ColmapReferenceFileContract) {
    QTemporaryDir tmp;
    ASSERT_TRUE(tmp.isValid());
    const QDir root(tmp.path());
    ASSERT_TRUE(root.mkpath(QStringLiteral("a")));
    ASSERT_TRUE(root.mkpath(QStringLiteral("b")));
    const QStringList images = {
            root.filePath("a/f1.jpg"), root.filePath("a/f2.jpg"),
            root.filePath("b/f1.jpg"), root.filePath("a/f3.jpg")};
    MapAnythingSceneMeta::ReferenceCenters r;
    QString error;

    // Bare names, trailing paths, comments, blank lines, trailing tokens and
    // unknown images (ignored, as in COLMAP).
    const QString ok = writeText(root, "ok.txt",
                                 "# name X Y Z\n"
                                 "\n"
                                 "a/f1.jpg 1 2 3\n"
                                 "f2.jpg  4\t5 6 extra\n"
                                 "b/f1.jpg -1 -2 -3\n"
                                 "other.jpg 9 9 9\n");
    ASSERT_TRUE(MapAnythingSceneMeta::loadColmapReferences(ok, images, false,
                                                           &r, &error))
            << error.toStdString();
    EXPECT_EQ(r.matched, 3);
    ASSERT_EQ(r.centers.size(), 12U);
    EXPECT_EQ(std::vector<float>(r.centers.begin(), r.centers.begin() + 9),
              std::vector<float>({1, 2, 3, 4, 5, 6, -1, -2, -3}));
    EXPECT_TRUE(std::isnan(r.centers[9]));
    EXPECT_EQ(r.offset[0], 0.0);

    // Fail closed: ambiguous name, duplicate image, malformed, < 3 matched,
    // GPS out of range.
    EXPECT_FALSE(MapAnythingSceneMeta::loadColmapReferences(
            writeText(root, "amb.txt", "f1.jpg 1 2 3\n"), images, false, &r,
            &error));
    EXPECT_TRUE(error.contains(QStringLiteral("matches both")));
    EXPECT_FALSE(MapAnythingSceneMeta::loadColmapReferences(
            writeText(root, "dup.txt",
                      "a/f1.jpg 1 2 3\nf2.jpg 1 2 3\na/f1.jpg 1 2 3\n"),
            images, false, &r, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("already has")));
    EXPECT_FALSE(MapAnythingSceneMeta::loadColmapReferences(
            writeText(root, "bad.txt", "a/f1.jpg 1 two 3\n"), images, false, &r,
            &error));
    EXPECT_TRUE(error.contains(QStringLiteral(":1:")));
    EXPECT_FALSE(MapAnythingSceneMeta::loadColmapReferences(
            writeText(root, "few.txt", "a/f1.jpg 1 2 3\nf2.jpg 1 2 3\n"),
            images, false, &r, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("at least 3")));
    EXPECT_FALSE(MapAnythingSceneMeta::loadColmapReferences(
            writeText(root, "gpsbad.txt",
                      "a/f1.jpg 4000000 2 3\nf2.jpg 1 2 3\nf3.jpg 1 2 3\n"),
            images, true, &r, &error));
    EXPECT_FALSE(MapAnythingSceneMeta::loadColmapReferences(
            root.filePath("missing.txt"), images, false, &r, &error));

    // Large cartesian coordinates (UTM-like): per-axis offset from the FIRST
    // line, truncated to hundreds; locals keep sub-mm precision.
    const QString utm = writeText(root, "utm.txt",
                                  "a/f1.jpg 500123.4567 4100987.6543 12.5\n"
                                  "f2.jpg 500124.4567 4100988.6543 13.5\n"
                                  "f3.jpg 500125.0 4100986.0 14.0\n");
    ASSERT_TRUE(MapAnythingSceneMeta::loadColmapReferences(utm, images, false,
                                                           &r, &error));
    EXPECT_EQ(r.offset[0], 500100.0);
    EXPECT_EQ(r.offset[1], 4100900.0);
    EXPECT_EQ(r.offset[2], 0.0);  // below the threshold: kept as given
    EXPECT_NEAR(r.centers[0], 23.4567f, 1e-5f);
    EXPECT_NEAR(r.centers[1], 87.6543f, 1e-5f);
    EXPECT_NEAR(r.centers[3 * 3 + 0], 25.0f, 1e-5f);

    // GPS: local ENU about the first line (COLMAP --alignment_type enu).
    const QString gps = writeText(root, "gps.txt",
                                  "f3.jpg 48.0 11.0 500\n"
                                  "a/f1.jpg 48.001 11.0 500\n"
                                  "f2.jpg 48.0 11.001 510\n");
    ASSERT_TRUE(MapAnythingSceneMeta::loadColmapReferences(gps, images, true,
                                                           &r, &error));
    EXPECT_TRUE(r.gps);
    EXPECT_EQ(r.enuOrigin[0], 48.0);
    for (int k = 0; k < 3; ++k) EXPECT_NEAR(r.centers[9 + k], 0.0f, 1e-6f);
    EXPECT_NEAR(r.centers[0], 0.0f, 1e-3f);   // due north: no east
    EXPECT_NEAR(r.centers[1], 111.2f, 0.3f);  // 0.001 deg lat ~ 111.2 m
    EXPECT_NEAR(r.centers[3], 74.6f, 0.3f);   // 0.001 deg lon * cos(48)
    EXPECT_NEAR(r.centers[5], 10.0f, 0.01f);  // altitude difference
}

TEST(QMapAnythingAnchor, RefusesCollinearReferenceCenters) {
    // Cameras along a straight walk (x axis) with centimetre jitter across
    // it, noisier predictions: the roll about x is unobservable.
    MR::Result r;
    r.views = 8;
    r.c2w.assign(8 * 16, 0.0f);
    std::vector<float> ref(8 * 3);
    for (int v = 0; v < 8; ++v) {
        float* M = &r.c2w[(size_t)v * 16];
        M[0] = M[5] = M[10] = M[15] = 1.0f;
        const float jitter = (v % 2 ? 0.01f : -0.01f);
        M[3] = (float)v;
        M[7] = (v % 3 - 1) * 0.05f;
        M[11] = ((v + 1) % 3 - 1) * 0.05f;
        ref[(size_t)v * 3] = (float)v;
        ref[(size_t)v * 3 + 1] = jitter;
        ref[(size_t)v * 3 + 2] = -jitter;
    }
    r.cloud.xyz = {0.5f, 0.5f, 0.5f};
    const MR::Result before = r;
    MR::ReferenceAnchor a;
    std::string error;
    EXPECT_FALSE(MR::anchor_to_reference(r, ref, &a, &error));
    EXPECT_NE(error.find("collinear"), std::string::npos) << error;
    EXPECT_EQ(r.cloud.xyz, before.cloud.xyz);  // untouched

    // The same path bent sideways by 1 m is well posed.
    for (int v = 0; v < 8; ++v) {
        const float bend = (v % 4 < 2 ? 0.0f : 1.0f);
        r.c2w[(size_t)v * 16 + 7] = bend;
        r.c2w[(size_t)v * 16 + 11] = 0.0f;
        ref[(size_t)v * 3 + 1] = bend;
        ref[(size_t)v * 3 + 2] = 0.0f;
    }
    ASSERT_TRUE(MR::anchor_to_reference(r, ref, &a, &error)) << error;
    EXPECT_GT(a.crossSpread, 0.4f);
    EXPECT_LT(a.residualMedian, 1e-4f);
}

TEST(QMapAnythingReconstruction, WritePlyDoubleWithOffset) {
    MR::Cloud c = makeCloud({{1, 2, 3}});
    const std::string path =
            ::testing::TempDir() + "qmapanything_write_ply_offset.ply";
    const double offset[3] = {500100.0, 4100900.0, 0.0};
    ASSERT_TRUE(MR::write_ply(path, c, nullptr, offset));
    std::ifstream in(path, std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());
    const std::string marker = "end_header\n";
    const size_t body = bytes.find(marker);
    ASSERT_NE(body, std::string::npos);
    EXPECT_NE(bytes.find("property double x\n"), std::string::npos);
    EXPECT_EQ(bytes.size() - body - marker.size(), c.size() * (3 * 8 + 3));
    double p[3];
    std::memcpy(p, bytes.data() + body + marker.size(), sizeof(p));
    EXPECT_EQ(p[0], double(c.xyz[0]) + offset[0]);
    EXPECT_EQ(p[1], double(c.xyz[1]) + offset[1]);
    EXPECT_EQ(p[2], double(c.xyz[2]));
    std::remove(path.c_str());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
