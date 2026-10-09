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

#include <QHash>
#include <QSet>
#include <cmath>
#include <cstdint>

#include "MapAnythingGeometry.h"
#include "MapAnythingModelCatalog.h"

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
            // facebook/VGGT demo_colmap: center-pad square 518 + conf >= 5.0
            {"vggt-1b-q8_0.gguf", "vggt-1b", "square", "normalized", 0.8f},
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

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
