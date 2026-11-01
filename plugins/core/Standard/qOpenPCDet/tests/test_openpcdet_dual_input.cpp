// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Pure-logic unit tests for the qOpenPCDet helpers (point-file parsing, CSV
// serialization, class-name mapping, box wireframe geometry, catalog
// mirror). No GGUF model required.

#include <ecvPluginDbNaming.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>
#include <ecvScalarField.h>
#include <gtest/gtest.h>

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QRegularExpression>
#include <QTemporaryDir>
#include <cmath>
#include <cstring>
#include <random>
#include <set>

#include "OpenPCDetDbExport.h"
#include "OpenPCDetModelCatalog.h"
#include "OpenPCDetWorker.h"
#include "ecvTestDataRepository.h"

#include "aicore/openpcdet_capi.h"

namespace {

bool SameEdge(const OpenPCDetHelpers::EdgePair& a,
              const OpenPCDetHelpers::EdgePair& b) {
    return (a.a == b.a && a.b == b.b) || (a.a == b.b && a.b == b.a);
}

}  // namespace

TEST(OpenPCDetHelpers, ParsePointsBinBasic) {
    // 2 points x 4 features (x y z intensity).
    const float raw[8] = {1.f, 2.f, 3.f, 0.5f, -4.f, -5.f, -6.f, 0.25f};
    QByteArray bytes(reinterpret_cast<const char*>(raw), sizeof(raw));

    std::vector<float> points;
    int count = 0;
    ASSERT_TRUE(OpenPCDetHelpers::parsePointsBin(bytes, 4, &points, &count,
                                                 nullptr));
    EXPECT_EQ(count, 2);
    ASSERT_EQ(points.size(), 8u);
    EXPECT_FLOAT_EQ(points[0], 1.0f);
    EXPECT_FLOAT_EQ(points[3], 0.5f);
    EXPECT_FLOAT_EQ(points[4], -4.0f);
    EXPECT_FLOAT_EQ(points[7], 0.25f);
}

TEST(OpenPCDetHelpers, ParsePointsBinBadSize) {
    std::vector<float> points;
    int count = 0;
    QString error;
    // 7 bytes can never be a whole number of float32 features.
    EXPECT_FALSE(OpenPCDetHelpers::parsePointsBin(QByteArray(7, '\0'), 4,
                                                  &points, &count, &error));
    EXPECT_FALSE(error.isEmpty());
    // Empty file is rejected.
    EXPECT_FALSE(OpenPCDetHelpers::parsePointsBin(QByteArray(), 4, &points,
                                                  &count, &error));
    // Invalid dim is rejected.
    EXPECT_FALSE(OpenPCDetHelpers::parsePointsBin(QByteArray(16, '\0'), 0,
                                                  &points, &count, &error));
}

TEST(OpenPCDetHelpers, ParsePointsTxtBasicAndPadding) {
    // One 3-value line (padded to dim 4) and one 4-value line.
    const QString text = QStringLiteral(
            "# comment line\n"
            "1 2 3\n"
            "\n"
            "4 5 6 0.75\n");
    std::vector<float> points;
    int count = 0;
    ASSERT_TRUE(OpenPCDetHelpers::parsePointsTxt(text, 4, &points, &count,
                                                 nullptr));
    EXPECT_EQ(count, 2);
    ASSERT_EQ(points.size(), 8u);
    EXPECT_FLOAT_EQ(points[0], 1.0f);
    EXPECT_FLOAT_EQ(points[3], 0.0f);  // padded feature
    EXPECT_FLOAT_EQ(points[4], 4.0f);
    EXPECT_FLOAT_EQ(points[7], 0.75f);
}

TEST(OpenPCDetHelpers, ParsePointsTxtFiveFeatures) {
    // nuScenes-style 5-dim input: extra features zero-filled when absent.
    const QString text = QStringLiteral("1 2 3 0.5\n");
    std::vector<float> points;
    int count = 0;
    ASSERT_TRUE(OpenPCDetHelpers::parsePointsTxt(text, 5, &points, &count,
                                                 nullptr));
    EXPECT_EQ(count, 1);
    ASSERT_EQ(points.size(), 5u);
    EXPECT_FLOAT_EQ(points[3], 0.5f);
    EXPECT_FLOAT_EQ(points[4], 0.0f);
}

TEST(OpenPCDetHelpers, ParsePointsTxtMalformed) {
    std::vector<float> points;
    int count = 0;
    QString error;
    // Fewer than 3 values is rejected.
    EXPECT_FALSE(OpenPCDetHelpers::parsePointsTxt(QStringLiteral("1 2\n"), 4,
                                                  &points, &count, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("Line 1")));
    // Non-numeric token is rejected.
    EXPECT_FALSE(OpenPCDetHelpers::parsePointsTxt(QStringLiteral("1 2 abc 4\n"),
                                                  4, &points, &count, &error));
    // No points at all is rejected.
    EXPECT_FALSE(OpenPCDetHelpers::parsePointsTxt(
            QStringLiteral("# only a comment\n"), 4, &points, &count, &error));
}

TEST(OpenPCDetHelpers, ClassNameKittiMapping) {
    EXPECT_EQ(OpenPCDetHelpers::className(1, 3), QStringLiteral("Car"));
    EXPECT_EQ(OpenPCDetHelpers::className(2, 3), QStringLiteral("Pedestrian"));
    EXPECT_EQ(OpenPCDetHelpers::className(3, 3), QStringLiteral("Cyclist"));
    // Out-of-range labels fall back to the generic form.
    EXPECT_EQ(OpenPCDetHelpers::className(0, 3), QStringLiteral("class 0"));
    EXPECT_EQ(OpenPCDetHelpers::className(4, 3), QStringLiteral("class 4"));
}

TEST(OpenPCDetHelpers, ClassNameGenericFallback) {
    // Non-KITTI layouts (e.g. nuScenes, 10 classes) have no name table in
    // the C API, so labels stay generic.
    EXPECT_EQ(OpenPCDetHelpers::className(1, 10), QStringLiteral("class 1"));
    EXPECT_EQ(OpenPCDetHelpers::className(10, 10), QStringLiteral("class 10"));
}

TEST(OpenPCDetHelpers, PerClassCounts) {
    OpenPCDetDetection a;
    a.label = 1;
    OpenPCDetDetection b;
    b.label = 3;
    OpenPCDetDetection c;
    c.label = 1;
    const std::map<int32_t, int> counts =
            OpenPCDetHelpers::perClassCounts({a, b, c});
    ASSERT_EQ(counts.size(), 2u);
    EXPECT_EQ(counts.at(1), 2);
    EXPECT_EQ(counts.at(3), 1);
}

TEST(OpenPCDetHelpers, CsvRoundTrip) {
    EXPECT_EQ(OpenPCDetHelpers::csvHeader(),
              QStringLiteral("x,y,z,dx,dy,dz,heading,score,label"));

    OpenPCDetDetection d;
    d.x = 1.5f;
    d.y = -2.25f;
    d.z = 3.0f;
    d.dx = 4.0f;
    d.dy = 2.0f;
    d.dz = 1.5f;
    d.heading = 0.5f;
    d.score = 0.97f;
    d.label = 1;
    const QString row = OpenPCDetHelpers::detectionToCsvRow(d);
    const QStringList cols = row.split(QRegularExpression(QStringLiteral(",")));
    ASSERT_EQ(cols.size(), 9);
    EXPECT_FLOAT_EQ(cols[0].toFloat(), 1.5f);
    EXPECT_FLOAT_EQ(cols[1].toFloat(), -2.25f);
    EXPECT_FLOAT_EQ(cols[7].toFloat(), 0.97f);
    EXPECT_EQ(cols[8], QStringLiteral("1"));
}

TEST(OpenPCDetHelpers, BoxCornersAxisAligned) {
    OpenPCDetDetection d;
    d.x = 10.0f;
    d.y = 20.0f;
    d.z = 30.0f;
    d.dx = 4.0f;
    d.dy = 2.0f;
    d.dz = 1.0f;
    d.heading = 0.0f;

    float corners[24];
    OpenPCDetHelpers::boxCorners(d, corners);

    // Corner 0 = (-hx, -hy, -hz) offsets from the center.
    EXPECT_FLOAT_EQ(corners[0], 8.0f);
    EXPECT_FLOAT_EQ(corners[1], 19.0f);
    EXPECT_FLOAT_EQ(corners[2], 29.5f);
    // Corner 7 = (+hx, +hy, +hz).
    EXPECT_FLOAT_EQ(corners[21], 12.0f);
    EXPECT_FLOAT_EQ(corners[22], 21.0f);
    EXPECT_FLOAT_EQ(corners[23], 30.5f);
}

TEST(OpenPCDetHelpers, BoxCornersHeadingRotation) {
    OpenPCDetDetection d;
    d.x = 0.0f;
    d.y = 0.0f;
    d.z = 0.0f;
    d.dx = 2.0f;
    d.dy = 0.0f;
    d.dz = 2.0f;
    d.heading = 1.5707963f;  // +90 deg about +Z

    float corners[24];
    OpenPCDetHelpers::boxCorners(d, corners);

    // Corner 1 local (+1, 0, ·) rotates to (0, +1, ·) at +90 deg.
    EXPECT_NEAR(corners[3], 0.0f, 1e-5f);
    EXPECT_NEAR(corners[4], 1.0f, 1e-5f);
    // Corner 0 local (-1, 0, ·) rotates to (0, -1, ·).
    EXPECT_NEAR(corners[0], 0.0f, 1e-5f);
    EXPECT_NEAR(corners[1], -1.0f, 1e-5f);
}

TEST(OpenPCDetHelpers, BoxEdgesAreTheTwelveCubeEdges) {
    const std::vector<OpenPCDetHelpers::EdgePair> edges =
            OpenPCDetHelpers::boxEdges();
    ASSERT_EQ(edges.size(), 12u);
    std::set<std::pair<int, int>> unique;
    for (const OpenPCDetHelpers::EdgePair& e : edges) {
        EXPECT_NE(e.a, e.b);
        // Edges differ in exactly one axis bit (true cube adjacency).
        const int diff = e.a ^ e.b;
        EXPECT_TRUE(diff == 1 || diff == 2 || diff == 4);
        unique.insert({std::min(e.a, e.b), std::max(e.a, e.b)});
    }
    EXPECT_EQ(unique.size(), 12u);
}

TEST(OpenPCDetHelpers, BoxWireCoversAllEdges) {
    const std::vector<int> wire = OpenPCDetHelpers::boxWireIndices();
    const std::vector<OpenPCDetHelpers::EdgePair> edges =
            OpenPCDetHelpers::boxEdges();

    // Every consecutive pair of the walk must be a real cube edge...
    for (size_t i = 0; i + 1 < wire.size(); ++i) {
        const OpenPCDetHelpers::EdgePair step{wire[i], wire[i + 1]};
        bool found = false;
        for (const OpenPCDetHelpers::EdgePair& e : edges) {
            if (SameEdge(step, e)) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "segment " << i << " is not a cube edge";
    }
    // ...and together they must cover all 12 edges.
    std::set<std::pair<int, int>> covered;
    for (size_t i = 0; i + 1 < wire.size(); ++i) {
        covered.insert({std::min(wire[i], wire[i + 1]),
                        std::max(wire[i], wire[i + 1])});
    }
    EXPECT_EQ(covered.size(), 12u);
}

TEST(OpenPCDetHelpers, ModelDisplayLabelDoesNotDuplicateQuantNote) {
    OpenPCDetModelEntry entry;
    entry.displayName = QStringLiteral(
            "OpenPCDet KITTI PointPillars — F16 — half precision "
            "(recommended)");
    entry.quantNote = QStringLiteral("half precision");
    const QString label = OpenPCDetHelpers::modelDisplayLabel(entry);
    EXPECT_EQ(label.count(entry.quantNote), 1);

    entry.displayName = QStringLiteral("OpenPCDet KITTI PointPillars");
    EXPECT_EQ(OpenPCDetHelpers::modelDisplayLabel(entry).count(entry.quantNote),
              1);
}

TEST(OpenPCDetHelpers, QuantSuffixesMatchCatalogOrder) {
    const QStringList quants = OpenPCDetHelpers::quantSuffixes();
    ASSERT_EQ(quants.size(), 3);
    EXPECT_EQ(quants[0], QStringLiteral("f32"));
    EXPECT_EQ(quants[1], QStringLiteral("f16"));
    EXPECT_EQ(quants[2], QStringLiteral("q8_0"));
}

TEST(OpenPCDetHelpers, BuildModelFilename) {
    QString out;
    ASSERT_TRUE(OpenPCDetHelpers::buildModelFilename(
            QStringLiteral("kitti_pointpillar"), QStringLiteral("f16"), &out));
    EXPECT_EQ(out, QStringLiteral("kitti_pointpillar.f16.gguf"));

    // Every checkpoint stem x quant combination must exist in the catalog.
    const QVector<OpenPCDetHelpers::OpenPCDetCheckpoint> checkpoints =
            OpenPCDetHelpers::catalogCheckpoints();
    for (const OpenPCDetHelpers::OpenPCDetCheckpoint& cp : checkpoints) {
        for (const QString& quant : OpenPCDetHelpers::quantSuffixes()) {
            ASSERT_TRUE(
                    OpenPCDetHelpers::buildModelFilename(cp.stem, quant, &out));
            OpenPCDetModelEntry entry;
            EXPECT_TRUE(OpenPCDetHelpers::findModelByFilename(out, &entry))
                    << "missing catalog row " << out.toStdString();
        }
    }
    // Invalid inputs are rejected.
    EXPECT_FALSE(OpenPCDetHelpers::buildModelFilename(
            QString(), QStringLiteral("f16"), &out));
    EXPECT_FALSE(OpenPCDetHelpers::buildModelFilename(
            QStringLiteral("kitti_pointpillar"), QStringLiteral("q4_k"), &out));
}

TEST(OpenPCDetHelpers, InputKindRouting) {
    // The multi-modal checkpoints route to scene inference.
    EXPECT_EQ(OpenPCDetHelpers::inputKindForFilename(
                      QStringLiteral("kitti_caddn.f16.gguf")),
              OpenPCDetHelpers::InputKind::Scene);
    EXPECT_EQ(OpenPCDetHelpers::inputKindForFilename(
                      QStringLiteral("nuscenes_bevfusion.q8_0.gguf")),
              OpenPCDetHelpers::InputKind::Scene);
    EXPECT_EQ(OpenPCDetHelpers::inputKindForFilename(QStringLiteral(
                      "kitti_voxelrcnn_focal_multimodal.q8_0.gguf")),
              OpenPCDetHelpers::InputKind::Scene);
    // All LiDAR-only checkpoints (a sample of each family) route to the
    // point path.
    for (const QString& filename :
         {QStringLiteral("kitti_pointpillar.f32.gguf"),
          QStringLiteral("second_7862.f16.gguf"),
          QStringLiteral("kitti_voxelrcnn_car.q8_0.gguf"),
          QStringLiteral("nuscenes_pointpillar_multihead.f16.gguf"),
          QStringLiteral("nuscenes_voxelnext_doubleflip.f32.gguf"),
          QStringLiteral("nuscenes_transfusion_lidar.q8_0.gguf")}) {
        EXPECT_EQ(OpenPCDetHelpers::inputKindForFilename(filename),
                  OpenPCDetHelpers::InputKind::Points)
                << filename.toStdString();
    }
}

TEST(OpenPCDetHelpers, CatalogCheckpointsGrouping) {
    const QVector<OpenPCDetHelpers::OpenPCDetCheckpoint> checkpoints =
            OpenPCDetHelpers::catalogCheckpoints();
    // 18 detector checkpoints x 3 quantizations = the 54 catalog rows.
    ASSERT_EQ(checkpoints.size(), 18);

    int sceneCount = 0;
    int kittiCount = 0;
    int nuScenesCount = 0;
    std::set<QString> stems;
    for (const OpenPCDetHelpers::OpenPCDetCheckpoint& cp : checkpoints) {
        // No checkpoint repeats and every checkpoint carries exactly the
        // three quantization rows.
        EXPECT_TRUE(stems.insert(cp.stem).second) << cp.stem.toStdString();
        ASSERT_EQ(cp.quants.size(), 3);
        EXPECT_FALSE(cp.displayName.isEmpty());
        EXPECT_EQ(cp.displayName.left(10), QStringLiteral("OpenPCDet "));
        // The display name is quant-agnostic (no " — <quant>" tail,
        // em dash U+2014).
        EXPECT_FALSE(cp.displayName.contains(QChar(0x2014)));

        // Input-kind + dataset grouping (dataset follows the family note,
        // matching the dialog's task-list sections).
        const bool isScene = OpenPCDetHelpers::inputKindForFilename(
                                     cp.stem + QStringLiteral(".f16.gguf")) ==
                             OpenPCDetHelpers::InputKind::Scene;
        if (isScene) {
            sceneCount += 1;
        } else if (cp.familyNote.contains(QStringLiteral("KITTI"))) {
            kittiCount += 1;
        } else {
            EXPECT_TRUE(cp.familyNote.contains(QStringLiteral("nuScenes")))
                    << cp.familyNote.toStdString();
            nuScenesCount += 1;
        }
        // Each row belongs to this checkpoint (prefix-consistent filenames).
        for (const OpenPCDetModelEntry& e : cp.quants) {
            EXPECT_TRUE(e.filename.startsWith(cp.stem + QStringLiteral(".")))
                    << e.filename.toStdString();
        }
    }
    EXPECT_EQ(kittiCount, 8);     // 8 KITTI LiDAR-only checkpoints
    EXPECT_EQ(nuScenesCount, 7);  // 7 nuScenes LiDAR checkpoints
    EXPECT_EQ(sceneCount, 3);     // kitti_caddn + nuscenes_bevfusion +
                                  // kitti_voxelrcnn_focal_multimodal
    // The default catalog row's checkpoint owns the recommended f16 model.
    const int defaultIdx = OpenPCDetHelpers::catalogDefaultIndex();
    const QVector<OpenPCDetModelEntry> all = OpenPCDetHelpers::catalogModels();
    ASSERT_GE(defaultIdx, 0);
    ASSERT_LT(defaultIdx, all.size());
    const int cpIdx = OpenPCDetHelpers::checkpointIndexForFilename(
            all[defaultIdx].filename);
    EXPECT_GE(cpIdx, 0);
    EXPECT_LT(cpIdx, checkpoints.size());
}

TEST(OpenPCDetHelpers, SceneCameraPreviewConversion) {
    // One 1x2 RGB frame, camera 0 of a 2-camera payload (NHWC).
    const float frames[2 * 1 * 2 * 3] = {
            0.0f, 0.5f, 1.0f, 1.0f, 0.0f, 0.25f,  // camera 0: two pixels
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,   // camera 1: black
    };
    const QImage img =
            OpenPCDetHelpers::sceneCameraPreview(frames, 2, 1, 2, 3, 0);
    ASSERT_FALSE(img.isNull());
    EXPECT_EQ(img.width(), 2);
    EXPECT_EQ(img.height(), 1);
    EXPECT_EQ(img.format(), QImage::Format_RGB888);
    const QRgb p0 = img.pixel(0, 0);
    EXPECT_EQ(qRed(p0), 0);
    EXPECT_EQ(qGreen(p0), 128);  // 0.5 * 255 + 0.5 rounds to 128
    EXPECT_EQ(qBlue(p0), 255);
    const QRgb p1 = img.pixel(1, 0);
    EXPECT_EQ(qRed(p1), 255);
    EXPECT_EQ(qGreen(p1), 0);
    EXPECT_EQ(qBlue(p1), 64);  // 0.25 * 255 + 0.5 = 64.25 -> 64

    // Grayscale payload expands to gray RGB.
    const float gray[1 * 1 * 1] = {0.5f};
    const QImage gimg =
            OpenPCDetHelpers::sceneCameraPreview(gray, 1, 1, 1, 1, 0);
    ASSERT_FALSE(gimg.isNull());
    const QRgb gp = gimg.pixel(0, 0);
    EXPECT_EQ(qRed(gp), qGreen(gp));
    EXPECT_EQ(qGreen(gp), qBlue(gp));

    // Invalid payloads return a null image.
    EXPECT_TRUE(OpenPCDetHelpers::sceneCameraPreview(nullptr, 1, 1, 1, 3, 0)
                        .isNull());
    EXPECT_TRUE(OpenPCDetHelpers::sceneCameraPreview(frames, 2, 1, 2, 3, 2)
                        .isNull());  // camera out of range
    EXPECT_TRUE(OpenPCDetHelpers::sceneCameraPreview(frames, 2, 1, 2, 5, 0)
                        .isNull());  // unsupported channel count
}

TEST(OpenPCDetHelpers, ReadScenePreviewFromDirectory) {
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());
    // A minimal upstream scene payload: scene.json + a 2x1 RGB images.bin.
    {
        QFile json(QDir(dir.path()).filePath(QStringLiteral("scene.json")));
        ASSERT_TRUE(json.open(QIODevice::WriteOnly | QIODevice::Text));
        json.write(
                "{\"num_cameras\": 2, \"image_height\": 1, "
                "\"image_width\": 2, \"image_channels\": 3}");
        json.close();
        QFile images(QDir(dir.path()).filePath(QStringLiteral("images.bin")));
        ASSERT_TRUE(images.open(QIODevice::WriteOnly));
        const float frame[2 * 1 * 2 * 3] = {};
        images.write(reinterpret_cast<const char*>(frame), sizeof(frame));
    }
    QImage preview;
    QString error;
    EXPECT_TRUE(
            OpenPCDetHelpers::readScenePreview(dir.path(), &preview, &error));
    EXPECT_TRUE(error.isEmpty());
    EXPECT_EQ(preview.width(), 2);
    EXPECT_EQ(preview.height(), 1);

    // Broken payloads are rejected with a message.
    QTemporaryDir emptyDir;
    ASSERT_TRUE(emptyDir.isValid());
    error.clear();
    EXPECT_FALSE(OpenPCDetHelpers::readScenePreview(emptyDir.path(), nullptr,
                                                    &error));
    EXPECT_FALSE(error.isEmpty());
    error.clear();
    EXPECT_FALSE(OpenPCDetHelpers::readScenePreview(
            QStringLiteral("/no/such/scene/dir"), nullptr, &error));
    EXPECT_FALSE(error.isEmpty());
}

TEST(OpenPCDetHelpers, CatalogMirror) {
    // 54 models = 18 detector checkpoints x 3 quantizations (f32, f16,
    // q8_0), mirroring the AICore OpenPCDet catalog.
    const QVector<OpenPCDetModelEntry> all = OpenPCDetHelpers::catalogModels();
    ASSERT_EQ(all.size(), 54);
    EXPECT_GE(OpenPCDetHelpers::catalogDefaultIndex(), 0);
    EXPECT_LT(OpenPCDetHelpers::catalogDefaultIndex(), all.size());
    for (const OpenPCDetModelEntry& e : all) {
        EXPECT_FALSE(e.filename.isEmpty());
        EXPECT_TRUE(e.downloadUrl.startsWith(QStringLiteral("https://")));
        EXPECT_TRUE(e.downloadUrl.contains(QStringLiteral("OpenPCDet-GGUF")));
        EXPECT_FALSE(e.displayName.isEmpty());
    }

    OpenPCDetModelEntry entry;
    EXPECT_TRUE(OpenPCDetHelpers::findModelByFilename(
            QStringLiteral("kitti_pointpillar.f16.gguf"), &entry));
    EXPECT_TRUE(entry.filename == QStringLiteral("kitti_pointpillar.f16.gguf"));
    EXPECT_FALSE(OpenPCDetHelpers::findModelByFilename(
            QStringLiteral("does-not-exist.gguf"), nullptr));
}

TEST(OpenPCDetHelpers, ClassColorMapping) {
    // nuScenes devkit palette at 10 classes.
    auto car = OpenPCDetHelpers::classColor(1, 10);
    EXPECT_EQ(car.r, 255);
    EXPECT_EQ(car.g, 158);
    EXPECT_EQ(car.b, 0);
    auto ped = OpenPCDetHelpers::classColor(9, 10);
    EXPECT_EQ(ped.r, 0);
    EXPECT_EQ(ped.g, 0);
    EXPECT_EQ(ped.b, 230);
    // KITTI demo colors at 3 classes.
    auto kittiCar = OpenPCDetHelpers::classColor(1, 3);
    EXPECT_EQ(kittiCar.r, 230);
    EXPECT_EQ(kittiCar.g, 60);
    EXPECT_EQ(kittiCar.b, 60);
    // Deterministic fallback outside the known layouts.
    auto fallback = OpenPCDetHelpers::classColor(2, 5);
    auto fallbackAgain = OpenPCDetHelpers::classColor(2, 5);
    EXPECT_EQ(fallback.r, fallbackAgain.r);
    EXPECT_EQ(fallback.g, fallbackAgain.g);
    EXPECT_EQ(fallback.b, fallbackAgain.b);
}

TEST(OpenPCDetHelpers, PointInBoxGeometry) {
    OpenPCDetDetection d;
    d.x = 10.0f;
    d.y = 20.0f;
    d.z = 1.0f;
    d.dx = 4.0f;
    d.dy = 2.0f;
    d.dz = 2.0f;
    d.heading = 0.0f;
    // Axis-aligned box: center inside, faces outside, corners outside.
    EXPECT_TRUE(OpenPCDetHelpers::pointInBox(d, 10.0f, 20.0f, 1.0f));
    EXPECT_TRUE(OpenPCDetHelpers::pointInBox(d, 11.9f, 20.9f, 1.9f));
    EXPECT_FALSE(OpenPCDetHelpers::pointInBox(d, 12.1f, 20.0f, 1.0f));
    EXPECT_FALSE(OpenPCDetHelpers::pointInBox(d, 10.0f, 20.0f, 2.1f));
    // 90-degree heading swaps the effective dx/dy extents.
    d.heading = static_cast<float>(M_PI) / 2.0f;
    EXPECT_TRUE(OpenPCDetHelpers::pointInBox(d, 10.0f, 21.9f, 1.0f));
    EXPECT_FALSE(OpenPCDetHelpers::pointInBox(d, 11.9f, 20.0f, 1.0f));
}

// ---------------------------------------------------------------------------
// "Use test data" panel mapping + bundle resolution (openpcdet_test_data)
// ---------------------------------------------------------------------------

namespace {

constexpr const char* kKittiFrame = "kitti_000008.bin";
constexpr const char* kKittiSampled = "kitti_000008_sampled16384.bin";
constexpr const char* kNuscenesSweep = "nuscenes_n015_lidar.bin";

}  // namespace

TEST(OpenPCDetTestData, MappingPointPanels) {
    // KITTI point panels take the raw frame...
    for (const QString& stem :
         {QStringLiteral("kitti_pointpillar"),
          QStringLiteral("kitti_second_iou"), QStringLiteral("second_7862"),
          QStringLiteral("kitti_pvrcnn"), QStringLiteral("kitti_parta2_anchor"),
          QStringLiteral("kitti_voxelrcnn_car")}) {
        const OpenPCDetHelpers::OpenPCDetTestDataFiles files =
                OpenPCDetHelpers::testDataFilesForCheckpoint(stem);
        EXPECT_EQ(files.pointsFile, QLatin1String(kKittiFrame))
                << stem.toStdString();
        EXPECT_TRUE(files.sceneDirName.isEmpty()) << stem.toStdString();
        EXPECT_TRUE(files.scenePointsFile.isEmpty()) << stem.toStdString();
    }
    // ...the PointRCNN checkpoints their pre-sampled 16384-point variant...
    for (const QString& stem : {QStringLiteral("kitti_pointrcnn"),
                                QStringLiteral("kitti_pointrcnn_iou")}) {
        const OpenPCDetHelpers::OpenPCDetTestDataFiles files =
                OpenPCDetHelpers::testDataFilesForCheckpoint(stem);
        EXPECT_EQ(files.pointsFile, QLatin1String(kKittiSampled))
                << stem.toStdString();
    }
    // ...and every nuScenes point panel the n015 5-feature sweep.
    for (const QString& stem :
         {QStringLiteral("nuscenes_pointpillar_multihead"),
          QStringLiteral("nuscenes_dynpillar_centerpoint"),
          QStringLiteral("nuscenes_second_multihead"),
          QStringLiteral("nuscenes_centerpoint_voxel0075"),
          QStringLiteral("nuscenes_voxelnext"),
          QStringLiteral("nuscenes_voxelnext_doubleflip"),
          QStringLiteral("nuscenes_transfusion_lidar")}) {
        const OpenPCDetHelpers::OpenPCDetTestDataFiles files =
                OpenPCDetHelpers::testDataFilesForCheckpoint(stem);
        EXPECT_EQ(files.pointsFile, QLatin1String(kNuscenesSweep))
                << stem.toStdString();
    }
}

TEST(OpenPCDetTestData, MappingScenePanels) {
    // CaDDN + focal multimodal: KITTI single-camera scene + the raw frame
    // as the LiDAR branch (the AICore focal-scenario pairing).
    for (const QString& stem :
         {QStringLiteral("kitti_caddn"),
          QStringLiteral("kitti_voxelrcnn_focal_multimodal")}) {
        const OpenPCDetHelpers::OpenPCDetTestDataFiles files =
                OpenPCDetHelpers::testDataFilesForCheckpoint(stem);
        EXPECT_EQ(files.sceneDirName, QStringLiteral("scene_caddn"))
                << stem.toStdString();
        EXPECT_EQ(files.scenePointsFile, QLatin1String(kKittiFrame))
                << stem.toStdString();
        EXPECT_TRUE(files.pointsFile.isEmpty()) << stem.toStdString();
    }
    // BEVFusion: 6-camera scene + the nuScenes LiDAR branch.
    const OpenPCDetHelpers::OpenPCDetTestDataFiles bevfusion =
            OpenPCDetHelpers::testDataFilesForCheckpoint(
                    QStringLiteral("nuscenes_bevfusion"));
    EXPECT_EQ(bevfusion.sceneDirName, QStringLiteral("scene_bevfusion"));
    EXPECT_EQ(bevfusion.scenePointsFile, QLatin1String(kNuscenesSweep));
    EXPECT_TRUE(bevfusion.pointsFile.isEmpty());
}

TEST(OpenPCDetTestData, MappingCoversEveryCheckpoint) {
    const QVector<OpenPCDetHelpers::OpenPCDetCheckpoint> checkpoints =
            OpenPCDetHelpers::catalogCheckpoints();
    ASSERT_EQ(checkpoints.size(), 18);
    int pointPanels = 0;
    int scenePanels = 0;
    for (const OpenPCDetHelpers::OpenPCDetCheckpoint& cp : checkpoints) {
        const OpenPCDetHelpers::OpenPCDetTestDataFiles files =
                OpenPCDetHelpers::testDataFilesForCheckpoint(cp.stem);
        // Every checkpoint maps to exactly one input branch, and scene
        // panels always carry their LiDAR-branch file.
        const bool isScene = !files.sceneDirName.isEmpty();
        EXPECT_EQ(!files.scenePointsFile.isEmpty(), isScene);
        if (isScene) {
            EXPECT_TRUE(files.pointsFile.isEmpty());
            EXPECT_FALSE(files.scenePointsFile.isEmpty());
            ++scenePanels;
        } else {
            EXPECT_FALSE(files.pointsFile.isEmpty());
            EXPECT_TRUE(files.sceneDirName.isEmpty());
            EXPECT_TRUE(files.scenePointsFile.isEmpty());
            ++pointPanels;
        }
    }
    EXPECT_EQ(pointPanels, 15);
    EXPECT_EQ(scenePanels, 3);
}

TEST(OpenPCDetTestData, ResolveErrorsOnMissingBundle) {
    OpenPCDetHelpers::OpenPCDetTestDataFiles files;
    QString error;
    // No bundle root at all.
    EXPECT_FALSE(OpenPCDetHelpers::resolveTestDataPaths(
            QStringLiteral("/no/such/openpcdet/bundle"),
            QStringLiteral("kitti_pointpillar"), &files, &error));
    EXPECT_FALSE(error.isEmpty());
    // Empty root.
    error.clear();
    EXPECT_FALSE(OpenPCDetHelpers::resolveTestDataPaths(
            QString(), QStringLiteral("kitti_pointpillar"), &files, &error));
    EXPECT_FALSE(error.isEmpty());
}

TEST(OpenPCDetTestData, ResolveErrorsOnMissingMember) {
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());
    // Bundle root exists but no members were extracted yet.
    OpenPCDetHelpers::OpenPCDetTestDataFiles files;
    QString error;
    EXPECT_FALSE(OpenPCDetHelpers::resolveTestDataPaths(
            dir.path(), QStringLiteral("kitti_pointpillar"), &files, &error));
    EXPECT_TRUE(error.contains(QLatin1String(kKittiFrame)))
            << error.toStdString();

    // Partial extraction (only the KITTI frame): KITTI panels resolve, the
    // nuScenes / scene panels still fail on their missing members.
    ASSERT_TRUE(QDir(dir.path()).mkpath(QStringLiteral(".")));
    QFile kitti(QDir(dir.path()).filePath(QLatin1String(kKittiFrame)));
    ASSERT_TRUE(kitti.open(QIODevice::WriteOnly));
    kitti.write("data");
    kitti.close();

    error.clear();
    EXPECT_FALSE(OpenPCDetHelpers::resolveTestDataPaths(
            dir.path(), QStringLiteral("nuscenes_voxelnext"), &files, &error));
    EXPECT_TRUE(error.contains(QLatin1String(kNuscenesSweep)))
            << error.toStdString();
    error.clear();
    EXPECT_FALSE(OpenPCDetHelpers::resolveTestDataPaths(
            dir.path(), QStringLiteral("nuscenes_bevfusion"), &files, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("scene_bevfusion")))
            << error.toStdString();
}

TEST(OpenPCDetTestData, ResolveFillsAbsolutePaths) {
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());
    // Full bundle layout: three LiDAR files + the two scene directories
    // (only the members the mapping consumes are named precisely).
    const auto touch = [&dir](const QString& rel) {
        const QString path = QDir(dir.path()).filePath(rel);
        ASSERT_TRUE(QDir().mkpath(QFileInfo(path).absolutePath()));
        QFile f(path);
        ASSERT_TRUE(f.open(QIODevice::WriteOnly));
        f.write("x");
    };
    touch(QLatin1String(kKittiFrame));
    touch(QLatin1String(kKittiSampled));
    touch(QLatin1String(kNuscenesSweep));
    touch(QStringLiteral("scene_caddn/scene.json"));
    touch(QStringLiteral("scene_bevfusion/scene.json"));

    const QString root = QDir(dir.path()).absolutePath();
    OpenPCDetHelpers::OpenPCDetTestDataFiles files;
    ASSERT_TRUE(OpenPCDetHelpers::resolveTestDataPaths(
            root, QStringLiteral("kitti_pointrcnn"), &files, nullptr));
    EXPECT_EQ(files.pointsFile,
              QDir(root).filePath(QLatin1String(kKittiSampled)));

    ASSERT_TRUE(OpenPCDetHelpers::resolveTestDataPaths(
            root, QStringLiteral("nuscenes_bevfusion"), &files, nullptr));
    EXPECT_EQ(files.sceneDirName,
              QDir(root).filePath(QStringLiteral("scene_bevfusion")));
    EXPECT_EQ(files.scenePointsFile,
              QDir(root).filePath(QLatin1String(kNuscenesSweep)));

    ASSERT_TRUE(OpenPCDetHelpers::resolveTestDataPaths(
            root, QStringLiteral("kitti_caddn"), &files, nullptr));
    EXPECT_EQ(files.sceneDirName,
              QDir(root).filePath(QStringLiteral("scene_caddn")));
    EXPECT_EQ(files.scenePointsFile,
              QDir(root).filePath(QLatin1String(kKittiFrame)));
}

TEST(OpenPCDetTestData, RepositoryMetadataAndLayout) {
    // The repository honors CLOUDVIEWER_DATA_ROOT (the cold-start test
    // harness points it at a scratch directory; the download itself is
    // network-bound and stays untested here, mirroring the qFaceDetect
    // test policy).
    QTemporaryDir dataRoot;
    ASSERT_TRUE(dataRoot.isValid());
    const QString savedRoot = qEnvironmentVariable("CLOUDVIEWER_DATA_ROOT");
    qputenv("CLOUDVIEWER_DATA_ROOT", dataRoot.path().toUtf8());

    using Kind = ecvTestDataRepository::Dataset;
    const Kind kind = Kind::OpenPCDetSample;
    EXPECT_EQ(ecvTestDataRepository::dataRoot(), dataRoot.path());

    const auto info = ecvTestDataRepository::getDatasetInfo(kind);
    EXPECT_EQ(info.kind, kind);
    EXPECT_EQ(info.displayName, QStringLiteral("OpenPCDet detection sample"));
    EXPECT_EQ(info.zipFileName, QStringLiteral("openpcdet_test_data.zip"));
    EXPECT_EQ(info.extractDirName, QStringLiteral("openpcdet_test_data"));
    EXPECT_EQ(info.extractGroupName, QStringLiteral("openpcdet_test_data"));
    EXPECT_EQ(info.downloadUrl,
              QStringLiteral("https://github.com/Asher-1/cloudViewer_downloads/"
                             "releases/download/OpenPCDet-GGUF/"
                             "openpcdet_test_data.zip"));
    EXPECT_EQ(QString::fromLatin1(info.anchor.digestHex),
              QStringLiteral("f1f185a385ddd0b440ed5699be47169eb89b7a0107e29"
                             "fd26e33ed81a07939dd"));

    // Grouped layout: zip under <root>/download, members under
    // <root>/extract/<group>/<zip root>.
    EXPECT_EQ(ecvTestDataRepository::zipPath(kind),
              QDir(ecvTestDataRepository::downloadDir())
                      .filePath(QStringLiteral("openpcdet_test_data.zip")));
    EXPECT_EQ(ecvTestDataRepository::extractPath(kind),
              QDir(QDir(ecvTestDataRepository::extractDir())
                           .filePath(QStringLiteral("openpcdet_test_data")))
                      .filePath(QStringLiteral("openpcdet_test_data")));

    // Nothing extracted in the scratch root: unavailable, and the member
    // lookup comes back empty.
    EXPECT_FALSE(ecvTestDataRepository::instance().isDatasetAvailable(kind));
    EXPECT_TRUE(ecvTestDataRepository::findDatasetFile(
                        kind, QStringLiteral("kitti_000008.bin"))
                        .isEmpty());

    if (savedRoot.isEmpty()) {
        qunsetenv("CLOUDVIEWER_DATA_ROOT");
    } else {
        qputenv("CLOUDVIEWER_DATA_ROOT", savedRoot.toUtf8());
    }
}

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// ---- Dual-input parity & DB-export metadata (automated acceptance) ----

namespace {

/// Builds N pseudo-random points [x,y,z,intensity] with a fixed seed so the
/// file path and the DB-tree path see identical inputs.
std::vector<float> makeSyntheticPoints(int n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> xyz(-50.0f, 50.0f);
    std::uniform_real_distribution<float> inten(0.0f, 1.0f);
    std::vector<float> pts(static_cast<size_t>(n) * 4);
    for (int i = 0; i < n; ++i) {
        pts[i * 4 + 0] = xyz(rng);
        pts[i * 4 + 1] = xyz(rng);
        pts[i * 4 + 2] = xyz(rng);
        pts[i * 4 + 3] = inten(rng);
    }
    return pts;
}

OpenPCDetRunResult makeTwoBoxResult() {
    OpenPCDetRunResult r;
    r.modelPath = QStringLiteral("/models/kitti_pointpillar.f32.gguf");
    r.numClass = 3;
    r.runtimeMs = 12.5;
    r.resolvedDevice = QStringLiteral("cpu");
    OpenPCDetDetection car;
    car.x = 0.f;
    car.y = 0.f;
    car.z = 1.f;
    car.dx = 4.f;
    car.dy = 2.f;
    car.dz = 2.f;
    car.heading = 0.f;
    car.score = 0.97f;
    car.label = 1;
    car.className = QStringLiteral("Car");
    OpenPCDetDetection ped = car;
    ped.x = 10.f;
    ped.y = 0.f;
    ped.dx = 1.f;
    ped.dy = 1.f;
    ped.score = 0.55f;
    ped.label = 2;
    ped.className = QStringLiteral("Pedestrian");
    r.detections = {car, ped};
    return r;
}

}  // namespace

TEST(OpenPCDetDualInput, FileAndDbTreeBuffersAreBitIdentical) {
    const int n = 500;
    const std::vector<float> pts = makeSyntheticPoints(n, 42);

    // Path 1 — file input: write a KITTI-style .bin and parse it back.
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());
    const QString binPath = dir.filePath(QStringLiteral("dual_input.bin"));
    QFile bin(binPath);
    ASSERT_TRUE(bin.open(QIODevice::WriteOnly));
    bin.write(reinterpret_cast<const char*>(pts.data()),
              static_cast<qint64>(pts.size() * sizeof(float)));
    bin.close();

    std::vector<float> fileBuffer;
    int fileCount = 0;
    QString error;
    QFile reader(binPath);
    ASSERT_TRUE(reader.open(QIODevice::ReadOnly));
    const QByteArray bytes = reader.readAll();
    reader.close();
    ASSERT_FALSE(bytes.isEmpty());
    ASSERT_TRUE(OpenPCDetHelpers::parsePointsBin(bytes, 4, &fileBuffer,
                                                 &fileCount, &error))
            << error.toStdString();

    // Path 2 — DB-tree input: the same points as a ccPointCloud with an
    // intensity scalar field (what firstSelectedCloud() hands the worker).
    ccPointCloud cloud(QStringLiteral("dbtree"));
    ASSERT_TRUE(cloud.reserve(n));
    for (int i = 0; i < n; ++i) {
        cloud.addPoint(
                CCVector3(pts[i * 4 + 0], pts[i * 4 + 1], pts[i * 4 + 2]));
    }
    auto* intensity = new ccScalarField("intensity");
    ASSERT_TRUE(intensity->reserveSafe(static_cast<unsigned>(n)));
    for (int i = 0; i < n; ++i) {
        intensity->addElement(pts[i * 4 + 3]);
    }
    intensity->computeMinAndMax();
    ASSERT_GE(cloud.addScalarField(intensity), 0);

    std::vector<float> cloudBuffer;
    int cloudCount = 0;
    ASSERT_TRUE(OpenPCDetWorker::extractCloudPoints(&cloud, 4, &cloudBuffer,
                                                    &cloudCount, &error))
            << error.toStdString();

    ASSERT_EQ(fileCount, n);
    ASSERT_EQ(cloudCount, n);
    ASSERT_EQ(fileBuffer.size(), cloudBuffer.size());
    EXPECT_EQ(0, std::memcmp(fileBuffer.data(), cloudBuffer.data(),
                             fileBuffer.size() * sizeof(float)))
            << "file input and DB-tree input diverge — the two Run paths "
               "would produce different detections";
}

TEST(OpenPCDetDualInput, ExtractCloudPointsCarriesFifthFeatureField) {
    // nuScenes 5-feature models: the ring/velocity channel must survive as
    // the uniformly named "OpenPCDet feature 4" scalar field instead of
    // being silently zero-filled for DB-tree input.
    constexpr int n = 3;
    const float ring[n] = {7.5f, -2.0f, 0.25f};
    ccPointCloud cloud(QStringLiteral("nuscenes5"));
    ASSERT_TRUE(cloud.reserve(n));
    for (int i = 0; i < n; ++i) {
        cloud.addPoint(CCVector3(1.0f * i, 2.0f, 3.0f));
    }
    auto* intensity = new ccScalarField("intensity");
    ASSERT_TRUE(intensity->reserveSafe(static_cast<unsigned>(n)));
    for (int i = 0; i < n; ++i) {
        intensity->addElement(0.5f);
    }
    intensity->computeMinAndMax();
    ASSERT_GE(cloud.addScalarField(intensity), 0);
    auto* feature4 = new ccScalarField("OpenPCDet feature 4");
    ASSERT_TRUE(feature4->reserveSafe(static_cast<unsigned>(n)));
    for (int i = 0; i < n; ++i) {
        feature4->addElement(ring[i]);
    }
    feature4->computeMinAndMax();
    ASSERT_GE(cloud.addScalarField(feature4), 0);

    std::vector<float> points;
    int count = 0;
    QString error;
    ASSERT_TRUE(OpenPCDetWorker::extractCloudPoints(&cloud, 5, &points, &count,
                                                    &error))
            << error.toStdString();
    ASSERT_EQ(count, n);
    ASSERT_EQ(points.size(), static_cast<size_t>(n) * 5);
    for (int i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(points[i * 5 + 3], 0.5f);
        EXPECT_FLOAT_EQ(points[i * 5 + 4], ring[i]) << "point " << i;
    }

    // Without the field the documented zero-fill fallback stays in effect.
    ccPointCloud plain(QStringLiteral("nuscenes_no_feature4"));
    ASSERT_TRUE(plain.reserve(n));
    for (int i = 0; i < n; ++i) {
        plain.addPoint(CCVector3(1.0f * i, 2.0f, 3.0f));
    }
    auto* plainIntensity = new ccScalarField("intensity");
    ASSERT_TRUE(plainIntensity->reserveSafe(static_cast<unsigned>(n)));
    for (int i = 0; i < n; ++i) {
        plainIntensity->addElement(0.5f);
    }
    plainIntensity->computeMinAndMax();
    ASSERT_GE(plain.addScalarField(plainIntensity), 0);
    std::vector<float> plainPoints;
    int plainCount = 0;
    ASSERT_TRUE(OpenPCDetWorker::extractCloudPoints(&plain, 5, &plainPoints,
                                                    &plainCount, &error))
            << error.toStdString();
    ASSERT_EQ(plainCount, n);
    for (int i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(plainPoints[i * 5 + 4], 0.0f);
    }
}

TEST(OpenPCDetDualInput, DbExportWireframeMetadata) {
    const OpenPCDetRunResult result = makeTwoBoxResult();
    ccHObject* group = BuildDetectionGroup(result, QStringLiteral("pp"));
    ASSERT_NE(group, nullptr);

    EXPECT_TRUE(group->getMetaData(QStringLiteral("OpenPCDet")).toBool());
    EXPECT_EQ(group->getMetaData(QStringLiteral("OpenPCDet/Model")),
              QStringLiteral("kitti_pointpillar.f32.gguf"));
    EXPECT_EQ(group->getMetaData(QStringLiteral("OpenPCDet/Count")),
              static_cast<qlonglong>(2));
    EXPECT_EQ(group->getMetaData(QStringLiteral("Device")),
              QStringLiteral("cpu"));
    EXPECT_EQ(group->getMetaData(QStringLiteral("Runtime (ms)")), 12.5);

    const std::vector<int> wire = OpenPCDetHelpers::boxWireIndices();
    ASSERT_EQ(group->getChildrenNumber(), 2u);
    for (unsigned i = 0; i < 2; ++i) {
        const OpenPCDetDetection& d = result.detections[i];
        auto* box = static_cast<ccPolyline*>(group->getChild(i));
        EXPECT_EQ(box->getName(), QStringLiteral("%1 %2")
                                          .arg(d.className)
                                          .arg(d.score, 0, 'f', 2));
        EXPECT_EQ(box->size(), wire.size());
        const OpenPCDetHelpers::OpenPCDetClassColor cls =
                OpenPCDetHelpers::classColor(d.label, result.numClass);
        const ecvColor::Rgb got = box->getColor();
        EXPECT_EQ(got.r, cls.r);
        EXPECT_EQ(got.g, cls.g);
        EXPECT_EQ(got.b, cls.b);
        EXPECT_EQ(box->getMetaData(QStringLiteral("OpenPCDet/Label")),
                  static_cast<qlonglong>(d.label));
        EXPECT_EQ(box->getMetaData(QStringLiteral("OpenPCDet/Score")),
                  static_cast<double>(d.score));
        EXPECT_EQ(box->getMetaData(QStringLiteral("OpenPCDet/Size")),
                  QStringLiteral("[%1,%2,%3]")
                          .arg(d.dx, 0, 'f', 3)
                          .arg(d.dy, 0, 'f', 3)
                          .arg(d.dz, 0, 'f', 3));
        EXPECT_EQ(box->getMetaData(QStringLiteral("OpenPCDet/Heading")),
                  static_cast<double>(d.heading));
    }
    delete group;
}

TEST(OpenPCDetDualInput, DbExportAnnotatedCloudMetadataAndTint) {
    // 8 points: 1 deep inside the car box, 1 outside, 1 inside the
    // pedestrian box, and 5 more spread far away.
    ccPointCloud source(QStringLiteral("src"));
    const int n = 8;
    ASSERT_TRUE(source.reserve(n));
    const CCVector3 pts[n] = {
            {0.5f, 0.2f, 1.0f},   // in car
            {30.f, 30.f, 0.f},    // outside
            {10.2f, 0.1f, 1.0f},  // in pedestrian
            {-20.f, 5.f, 0.f},    // outside
            {25.f, -25.f, 0.f},   // outside
            {0.1f, 40.f, 0.f},    // outside
            {-40.f, -0.5f, 0.f},  // outside
            {12.f, 12.f, 12.f},   // outside
    };
    for (int i = 0; i < n; ++i) source.addPoint(pts[i]);

    const OpenPCDetRunResult result = makeTwoBoxResult();
    unsigned tinted = 0;
    QString error;
    ccPointCloud* annotated = BuildAnnotatedCloud(
            &source, result, QStringLiteral("pp"), &tinted, &error);
    ASSERT_NE(annotated, nullptr) << error.toStdString();
    EXPECT_EQ(annotated->size(), n);
    EXPECT_EQ(tinted, 2u);
    EXPECT_EQ(annotated->getMetaData(QStringLiteral("OpenPCDet/TintedPoints")),
              static_cast<qlonglong>(2));
    EXPECT_EQ(annotated->getMetaData(QStringLiteral("OpenPCDet/Count")),
              static_cast<qlonglong>(2));
    EXPECT_EQ(annotated->getMetaData(QStringLiteral("OpenPCDet/Model")),
              QStringLiteral("kitti_pointpillar.f32.gguf"));
    EXPECT_TRUE(annotated->colorsShown());

    const OpenPCDetHelpers::OpenPCDetClassColor carColor =
            OpenPCDetHelpers::classColor(1, 3);
    const OpenPCDetHelpers::OpenPCDetClassColor pedColor =
            OpenPCDetHelpers::classColor(2, 3);
    for (int i = 0; i < n; ++i) {
        const ecvColor::Rgb c = annotated->getPointColor(i);
        if (i == 0) {
            EXPECT_EQ(c.r, carColor.r);
            EXPECT_EQ(c.g, carColor.g);
            EXPECT_EQ(c.b, carColor.b);
        } else if (i == 2) {
            EXPECT_EQ(c.r, pedColor.r);
            EXPECT_EQ(c.g, pedColor.g);
            EXPECT_EQ(c.b, pedColor.b);
        } else {
            EXPECT_EQ(c.r, 255);
            EXPECT_EQ(c.g, 255);
            EXPECT_EQ(c.b, 255);
        }
    }
    delete annotated;
}

/** Captures the worker's resultReady payload on the worker thread
 *  (direct connection), mirroring the plugin's queued onResultReady. */
class ResultCatcher : public QObject {
    Q_OBJECT
public:
    OpenPCDetRunResult result;
    bool got = false;
public slots:
    void onResult(const OpenPCDetRunResult& r) {
        result = r;
        got = true;
    }
};

#include "test_openpcdet_dual_input.moc"

TEST(OpenPCDetDualInput, EndToEndRealInferenceAndExport) {
    // Real cached assets (the layout the AICore model cache and the
    // sample-data download populate). Skip — do not fail — when the
    // machine has not populated them.
    const QString extractRoot = ecvTestDataRepository::extractDir();
    const QString model = extractRoot + QStringLiteral(
                                                "/openpcdet_models/"
                                                "kitti_pointpillar.f32.gguf");
    const QString bin =
            extractRoot + QStringLiteral("/openpcdet_inputs/kitti_000008.bin");
    if (!QFileInfo::exists(model) || !QFileInfo::exists(bin)) {
        GTEST_SKIP() << "OpenPCDet assets not cached under "
                     << extractRoot.toStdString();
    }

    // ---- DB-tree input: the ccPointCloud exactly as the sample-fill
    // path hands it to the worker (points + intensity scalar field). ----
    QFile binFile(bin);
    ASSERT_TRUE(binFile.open(QIODevice::ReadOnly));
    const QByteArray bytes = binFile.readAll();
    binFile.close();
    ASSERT_FALSE(bytes.isEmpty());

    std::vector<float> fileBuffer;
    int fileCount = 0;
    QString error;
    ASSERT_TRUE(OpenPCDetHelpers::parsePointsBin(bytes, 4, &fileBuffer,
                                                 &fileCount, &error))
            << error.toStdString();
    ASSERT_GT(fileCount, 0);

    ccPointCloud cloud(QStringLiteral("kitti_000008"));
    ASSERT_TRUE(cloud.reserve(static_cast<unsigned>(fileCount)));
    for (int i = 0; i < fileCount; ++i) {
        cloud.addPoint(CCVector3(fileBuffer[i * 4 + 0], fileBuffer[i * 4 + 1],
                                 fileBuffer[i * 4 + 2]));
    }
    auto* intensity = new ccScalarField("intensity");
    ASSERT_TRUE(intensity->reserveSafe(static_cast<unsigned>(fileCount)));
    for (int i = 0; i < fileCount; ++i) {
        intensity->addElement(fileBuffer[i * 4 + 3]);
    }
    intensity->computeMinAndMax();
    ASSERT_GE(cloud.addScalarField(intensity), 0);

    // ---- Run the real worker: once with the DB-tree cloud, once with
    // the raw file, mirroring the two GUI input paths. ----
    auto runWorker = [&](const OpenPCDetWorker::Settings& ws, ccPointCloud* src,
                         OpenPCDetRunResult* out) {
        OpenPCDetWorker worker(ws, src);
        ResultCatcher catcher;
        QObject::connect(&worker, &OpenPCDetWorker::resultReady, &catcher,
                         &ResultCatcher::onResult, Qt::DirectConnection);
        worker.start();
        const bool finished = worker.wait(180000);
        worker.releaseContextOnMainThread();
        if (catcher.got) *out = catcher.result;
        return finished && catcher.got;
    };

    OpenPCDetWorker::Settings cloudWs;
    cloudWs.modelPath = model;
    cloudWs.device = QStringLiteral("cpu");
    cloudWs.threads = 4;
    OpenPCDetWorker::Settings fileWs = cloudWs;
    fileWs.pointsFile = bin;

    OpenPCDetRunResult cloudResult;
    OpenPCDetRunResult fileResult;
    ASSERT_TRUE(runWorker(cloudWs, &cloud, &cloudResult))
            << "worker run (DB-tree cloud input) produced no result";
    ASSERT_TRUE(runWorker(fileWs, nullptr, &fileResult))
            << "worker run (file input) produced no result";

    // ---- Dual-path parity on REAL data through the REAL engine. ----
    ASSERT_FALSE(cloudResult.detections.isEmpty())
            << "kitti_pointpillar found no object in KITTI 000008";
    ASSERT_EQ(fileResult.detections.size(), cloudResult.detections.size());
    for (int i = 0; i < cloudResult.detections.size(); ++i) {
        const OpenPCDetDetection& a = cloudResult.detections[i];
        const OpenPCDetDetection& b = fileResult.detections[i];
        EXPECT_EQ(a.label, b.label) << "box " << i;
        EXPECT_EQ(a.className, b.className) << "box " << i;
        EXPECT_FLOAT_EQ(a.x, b.x) << "box " << i;
        EXPECT_FLOAT_EQ(a.y, b.y) << "box " << i;
        EXPECT_FLOAT_EQ(a.z, b.z) << "box " << i;
        EXPECT_FLOAT_EQ(a.dx, b.dx) << "box " << i;
        EXPECT_FLOAT_EQ(a.dy, b.dy) << "box " << i;
        EXPECT_FLOAT_EQ(a.dz, b.dz) << "box " << i;
        EXPECT_FLOAT_EQ(a.heading, b.heading) << "box " << i;
        EXPECT_FLOAT_EQ(a.score, b.score) << "box " << i;
    }
    EXPECT_EQ(cloudResult.numPoints, fileResult.numPoints);

    // ---- Export flow exactly as onResultReady consumes it. ----
    const QString modelTag = ecvPluginDbNaming::modelTagFromFilename(model);
    ccHObject* group = BuildDetectionGroup(cloudResult, modelTag);
    ASSERT_NE(group, nullptr);
    EXPECT_TRUE(group->getMetaData(QStringLiteral("OpenPCDet")).toBool());
    EXPECT_EQ(group->getMetaData(QStringLiteral("OpenPCDet/Model")),
              QFileInfo(model).fileName());
    EXPECT_EQ(group->getMetaData(QStringLiteral("OpenPCDet/Count")),
              static_cast<qlonglong>(cloudResult.detections.size()));
    EXPECT_GT(group->getMetaData(QStringLiteral("Runtime (ms)")).toDouble(),
              0.0);
    ASSERT_EQ(group->getChildrenNumber(),
              static_cast<unsigned>(cloudResult.detections.size()));
    const std::vector<int> wire = OpenPCDetHelpers::boxWireIndices();
    for (int i = 0; i < cloudResult.detections.size(); ++i) {
        const OpenPCDetDetection& d = cloudResult.detections[i];
        auto* box = static_cast<ccPolyline*>(
                group->getChild(static_cast<unsigned>(i)));
        EXPECT_EQ(box->size(), wire.size()) << "box " << i;
        EXPECT_EQ(box->getMetaData(QStringLiteral("OpenPCDet/Label")),
                  static_cast<qlonglong>(d.label))
                << "box " << i;
        const double score =
                box->getMetaData(QStringLiteral("OpenPCDet/Score")).toDouble();
        EXPECT_GT(score, 0.0) << "box " << i;
        EXPECT_LE(score, 1.0) << "box " << i;
        const OpenPCDetHelpers::OpenPCDetClassColor cls =
                OpenPCDetHelpers::classColor(d.label, cloudResult.numClass);
        const ecvColor::Rgb got = box->getColor();
        EXPECT_EQ(got.r, cls.r) << "box " << i;
        EXPECT_EQ(got.g, cls.g) << "box " << i;
        EXPECT_EQ(got.b, cls.b) << "box " << i;
    }

    unsigned tinted = 0;
    error.clear();
    ccPointCloud* annotated =
            BuildAnnotatedCloud(&cloud, cloudResult, modelTag, &tinted, &error);
    ASSERT_NE(annotated, nullptr) << error.toStdString();
    EXPECT_EQ(annotated->size(), cloud.size());
    EXPECT_GT(tinted, 0u)
            << "real detections must tint at least one LiDAR point";
    EXPECT_EQ(annotated->getMetaData(QStringLiteral("OpenPCDet/TintedPoints")),
              static_cast<qlonglong>(tinted));
    EXPECT_TRUE(annotated->hasColors());
    EXPECT_EQ(annotated->getMetaData(QStringLiteral("OpenPCDet/Model")),
              QFileInfo(model).fileName())
            << "annotated cloud must carry the model filename (the "
               "runInference modelPath fix)";
    delete annotated;
    delete group;
}

/** Explicit full-matrix mode (plan §6): enumerate every published
 *  checkpoint × quantization through the REAL worker on the CPU backend,
 *  verifying pairing, declared defaults (through the metadata API), typed
 *  finite boxes, and file/DB dual-path parity per row.
 *
 *  Gated behind ACV_OPENPCDET_FULL_MATRIX=1: the default CTest run stays
 *  small, while the explicit mode FAILS — never skips — on any missing
 *  asset, so a green explicit run is per-model evidence, not a sample.
 *  GPU/Vulkan backend rows are reported by validate_all.py / the bench
 *  runner; this test pins the CPU path deterministically. */
TEST(OpenPCDetDualInput, FullCatalogMatrix) {
    const bool full = qgetenv("ACV_OPENPCDET_FULL_MATRIX") == "1";
    if (!full) {
        GTEST_SKIP() << "set ACV_OPENPCDET_FULL_MATRIX=1 for the "
                        "checkpoint-by-checkpoint matrix";
    }
    const QString extractRoot = ecvTestDataRepository::extractDir();
    const QString modelsRoot = extractRoot + QStringLiteral("/openpcdet_models");
    // The pinned fixture bundle carries the point frames AND the two
    // multi-modal scene dirs; the model-cache input dir carries point frames
    // only, so the bundle wins when present.
    const QString fixtureRoot = extractRoot + QStringLiteral(
            "/openpcdet_test_data/openpcdet_test_data");
    const QString inputsRoot =
            QDir(fixtureRoot).exists()
                    ? fixtureRoot
                    : extractRoot + QStringLiteral("/openpcdet_inputs");

    const QVector<OpenPCDetHelpers::OpenPCDetCheckpoint> checkpoints =
            OpenPCDetHelpers::catalogCheckpoints();
    ASSERT_EQ(checkpoints.size(), 18)
            << "the published checkpoint set changed — update this matrix";

    auto failMissing = [&](const QString& path) {
        ADD_FAILURE() << "missing asset in explicit mode: "
                      << path.toStdString();
        return false;
    };

    int rows = 0;
    for (const auto& cp : checkpoints) {
        // Input identity (plan §4): the pairing must map each checkpoint
        // family to its own frame — no cross-dataset borrowing.
        const OpenPCDetHelpers::OpenPCDetTestDataFiles files =
                OpenPCDetHelpers::testDataFilesForCheckpoint(cp.stem);
        const bool nuScenes = cp.stem.startsWith(QStringLiteral("nuscenes"));
        ASSERT_FALSE(files.pointsFile.isEmpty() &&
                     files.scenePointsFile.isEmpty())
                << cp.stem.toStdString()
                << " has neither a point file nor a scene LiDAR branch";
        if (nuScenes) {
            EXPECT_TRUE(files.pointsFile.endsWith(
                                QStringLiteral("nuscenes_n015_lidar.bin")) ||
                        files.scenePointsFile.endsWith(
                                QStringLiteral("nuscenes_n015_lidar.bin")))
                    << cp.stem.toStdString();
        } else if (cp.stem.contains(QStringLiteral("pointrcnn"))) {
            EXPECT_EQ(files.pointsFile,
                      QStringLiteral("kitti_000008_sampled16384.bin"))
                    << cp.stem.toStdString();
        }

        const QString inputName = !files.pointsFile.isEmpty()
                                          ? files.pointsFile
                                          : files.scenePointsFile;
        const QString inputPath = inputsRoot + QLatin1Char('/') + inputName;
        if (!QFileInfo::exists(inputPath)) {
            failMissing(inputPath);
            continue;
        }
        // Multi-modal rows run through detect_scene: resolve the pinned
        // scene dir next to the point frames.
        const bool sceneRow = !files.sceneDirName.isEmpty();
        QString scenePath;
        if (sceneRow) {
            scenePath = inputsRoot + QLatin1Char('/') + files.sceneDirName;
            if (!QFileInfo::exists(scenePath + QStringLiteral("/scene.json"))) {
                failMissing(scenePath);
                continue;
            }
        }

        for (const OpenPCDetModelEntry& e : cp.quants) {
            const QString modelPath = modelsRoot + QLatin1Char('/') + e.filename;
            if (!QFileInfo::exists(modelPath)) {
                failMissing(modelPath);
                continue;
            }
            // Declared metadata drives the parse dimension — the same facts
            // the dialog consumes (read path must not initialize backends).
            aicore_openpcdet_model_info info{};
            info.struct_size = sizeof(info);
            info.abi_version = AICORE_OPENPCDET_MODEL_INFO_ABI_VERSION;
            ASSERT_EQ(aicore_openpcdet_read_model_info(
                              modelPath.toUtf8().constData(), &info, nullptr),
                      0)
                    << e.filename.toStdString();
            ASSERT_GT(info.num_point_features, 0);

            QFile binFile(inputPath);
            ASSERT_TRUE(binFile.open(QIODevice::ReadOnly));
            const QByteArray bytes = binFile.readAll();
            binFile.close();
            std::vector<float> fileBuffer;
            int fileCount = 0;
            QString error;
            ASSERT_TRUE(OpenPCDetHelpers::parsePointsBin(
                    bytes, info.num_point_features, &fileBuffer, &fileCount,
                    &error))
                    << error.toStdString();
            ASSERT_GT(fileCount, 0);

            ccPointCloud cloud(QStringLiteral("matrix_%1").arg(cp.stem));
            ASSERT_TRUE(cloud.reserve(static_cast<unsigned>(fileCount)));
            for (int i = 0; i < fileCount; ++i) {
                const float* row = fileBuffer.data() +
                                   static_cast<size_t>(i) *
                                           info.num_point_features;
                cloud.addPoint(CCVector3(row[0], row[1], row[2]));
            }
            auto* intensity = new ccScalarField("intensity");
            ASSERT_TRUE(intensity->reserveSafe(
                    static_cast<unsigned>(fileCount)));
            for (int i = 0; i < fileCount; ++i) {
                const float* row = fileBuffer.data() +
                                   static_cast<size_t>(i) *
                                           info.num_point_features;
                intensity->addElement(info.num_point_features >= 4 ? row[3]
                                                                   : 0.0f);
            }
            intensity->computeMinAndMax();
            ASSERT_GE(cloud.addScalarField(intensity), 0);

            auto runWorker = [&](const OpenPCDetWorker::Settings& ws,
                                 ccPointCloud* src, OpenPCDetRunResult* out) {
                OpenPCDetWorker worker(ws, src);
                ResultCatcher catcher;
                QObject::connect(&worker, &OpenPCDetWorker::resultReady,
                                 &catcher, &ResultCatcher::onResult,
                                 Qt::DirectConnection);
                QStringList workerLog;
                QObject::connect(
                        &worker, &OpenPCDetWorker::logMessage, &worker,
                        [&](const QString& m) { workerLog << m; },
                        Qt::DirectConnection);
                QObject::connect(
                        &worker, &OpenPCDetWorker::debugMessage, &worker,
                        [&](const QString& m) { workerLog << m; },
                        Qt::DirectConnection);
                worker.start();
                const bool finished = worker.wait(300000);
                worker.releaseContextOnMainThread();
                if (catcher.got) *out = catcher.result;
                const bool ok = finished && catcher.got;
                if (!ok) {
                    std::fprintf(stderr, "[matrix-fail] worker log:\n");
                    for (const QString& line : workerLog) {
                        std::fprintf(stderr, "  %s\n",
                                     line.toUtf8().constData());
                    }
                }
                return ok;
            };

            OpenPCDetWorker::Settings cloudWs;
            cloudWs.modelPath = modelPath;
            cloudWs.device = QStringLiteral("cpu");
            if (sceneRow) cloudWs.sceneDir = scenePath;
            OpenPCDetWorker::Settings fileWs = cloudWs;
            fileWs.pointsFile = inputPath;

            OpenPCDetRunResult cloudResult;
            OpenPCDetRunResult fileResult;
            ASSERT_TRUE(runWorker(cloudWs, &cloud, &cloudResult))
                    << e.filename.toStdString();
            ASSERT_TRUE(runWorker(fileWs, nullptr, &fileResult))
                    << e.filename.toStdString();

            // Finite typed boxes: label range, score range, geometry.
            ASSERT_EQ(cloudResult.detections.size(),
                      fileResult.detections.size())
                    << e.filename.toStdString();
            for (int i = 0; i < cloudResult.detections.size(); ++i) {
                const OpenPCDetDetection& a = cloudResult.detections[i];
                const OpenPCDetDetection& b = fileResult.detections[i];
                EXPECT_EQ(a.label, b.label) << "box " << i;
                EXPECT_FLOAT_EQ(a.x, b.x) << "box " << i;
                EXPECT_FLOAT_EQ(a.y, b.y) << "box " << i;
                EXPECT_FLOAT_EQ(a.z, b.z) << "box " << i;
                EXPECT_FLOAT_EQ(a.heading, b.heading) << "box " << i;
                EXPECT_FLOAT_EQ(a.score, b.score) << "box " << i;
                EXPECT_GE(a.label, 1);
                EXPECT_LE(a.label, cloudResult.numClass);
                EXPECT_TRUE(std::isfinite(a.x) && std::isfinite(a.y) &&
                            std::isfinite(a.z))
                        << "box " << i;
                EXPECT_GT(a.dx, 0.0f);
                EXPECT_GT(a.dy, 0.0f);
                EXPECT_GT(a.dz, 0.0f);
                EXPECT_GE(a.score, 0.0f);
                EXPECT_LE(a.score, 1.0f);
            }
            EXPECT_EQ(cloudResult.numPoints, fileResult.numPoints);
            EXPECT_EQ(cloudResult.pointDim, info.num_point_features);
            EXPECT_EQ(cloudResult.fromScene, sceneRow)
                    << e.filename.toStdString();

            std::printf(
                "[matrix] %s boxes=%d points=%d dim=%d %lld ms\n",
                e.filename.toUtf8().constData(),
                cloudResult.detections.size(), cloudResult.numPoints,
                cloudResult.pointDim,
                static_cast<long long>(cloudResult.runtimeMs));
            ++rows;
        }
    }
    EXPECT_EQ(rows, 54) << "explicit matrix must cover the full "
                           "18 checkpoints x 3 quantizations";
}

TEST(OpenPCDetDualInput, ThresholdOverrideExplicitZeroSemantics) {
    // thresholdsExplicit=true owns the values verbatim: an explicit
    // score/NMS of 0 must reach the C API as a real override (debug-visible),
    // while the legacy sentinel path must stay override-free for the same
    // zeros. Asset-gated like the E2E test.
    const QString extractRoot = ecvTestDataRepository::extractDir();
    const QString model = extractRoot + QStringLiteral(
                                                "/openpcdet_models/"
                                                "kitti_pointpillar.f16.gguf");
    const QString bin = extractRoot + QStringLiteral(
                                                "/openpcdet_inputs/"
                                                "kitti_000008.bin");
    if (!QFileInfo::exists(model) || !QFileInfo::exists(bin)) {
        GTEST_SKIP() << "OpenPCDet assets not cached under "
                     << extractRoot.toStdString();
    }

    auto run = [&](bool explicitFlag, QStringList* debugOut) {
        OpenPCDetWorker::Settings ws;
        ws.modelPath = model;
        ws.device = QStringLiteral("cpu");
        ws.pointsFile = bin;
        ws.thresholdsExplicit = explicitFlag;
        OpenPCDetWorker worker(ws, nullptr);
        QStringList lines;
        QObject::connect(
                &worker, &OpenPCDetWorker::debugMessage, &worker,
                [&](const QString& m) { lines << m; },
                Qt::DirectConnection);
        worker.start();
        ASSERT_TRUE(worker.wait(180000));
        worker.releaseContextOnMainThread();
        *debugOut = lines;
    };

    QStringList explicitLogs;
    QStringList legacyLogs;
    run(true, &explicitLogs);
    run(false, &legacyLogs);

    bool explicitScoreZero = false;
    for (const QString& line : explicitLogs) {
        if (line.contains(QStringLiteral("Score threshold override: 0.00")))
            explicitScoreZero = true;
    }
    EXPECT_TRUE(explicitScoreZero)
            << "explicit thresholds must forward a 0 score override";
    for (const QString& line : legacyLogs) {
        EXPECT_FALSE(line.contains(QStringLiteral("threshold override")))
                << "legacy sentinel zeros must not become overrides: "
                << line.toStdString();
    }
}

namespace {

// Shared run helper for the DB-frame tests below: starts one worker and
// captures its result (direct connection on the worker thread).
bool runOneWorker(const OpenPCDetWorker::Settings& ws,
                  ccPointCloud* src,
                  OpenPCDetRunResult* out) {
    OpenPCDetWorker worker(ws, src);
    ResultCatcher catcher;
    QObject::connect(&worker, &OpenPCDetWorker::resultReady, &catcher,
                     &ResultCatcher::onResult, Qt::DirectConnection);
    worker.start();
    const bool finished = worker.wait(180000);
    worker.releaseContextOnMainThread();
    if (catcher.got) *out = catcher.result;
    return finished && catcher.got;
}

}  // namespace

TEST(OpenPCDetDualInput, DbCloudInputLiftsToLiDARFrame) {
    // A georeferenced DB cloud stores LOCAL coordinates (ecvShiftedObject:
    // Plocal = (Pglobal + shift) * scale). The engine must see the ORIGINAL
    // LiDAR frame (Pglobal = Plocal/scale - shift); a worker that feeds the
    // stored local coordinates drops every point outside the declared
    // point_cloud_range and finds nothing.
    const QString extractRoot = ecvTestDataRepository::extractDir();
    const QString model = extractRoot + QStringLiteral(
                                                "/openpcdet_models/"
                                                "kitti_pointpillar.f16.gguf");
    const QString bin = extractRoot + QStringLiteral(
                                                "/openpcdet_inputs/"
                                                "kitti_000008.bin");
    if (!QFileInfo::exists(model) || !QFileInfo::exists(bin)) {
        GTEST_SKIP() << "OpenPCDet assets not cached under "
                     << extractRoot.toStdString();
    }

    // Reference answer: the raw file in the native LiDAR frame.
    OpenPCDetWorker::Settings fileWs;
    fileWs.modelPath = model;
    fileWs.device = QStringLiteral("cpu");
    fileWs.pointsFile = bin;
    OpenPCDetRunResult fileResult;
    ASSERT_TRUE(runOneWorker(fileWs, nullptr, &fileResult));
    ASSERT_FALSE(fileResult.detections.isEmpty());

    // The same frame stored in a shifted/scaled local DB frame. Scale 2.0 is
    // a power of two, so the round trip Pglobal -> Plocal -> Pglobal is
    // exact and box-count parity with the reference run is meaningful.
    QFile binFile(bin);
    ASSERT_TRUE(binFile.open(QIODevice::ReadOnly));
    const QByteArray bytes = binFile.readAll();
    binFile.close();
    std::vector<float> pts;
    int count = 0;
    QString error;
    ASSERT_TRUE(OpenPCDetHelpers::parsePointsBin(bytes, 4, &pts, &count,
                                                 &error))
            << error.toStdString();
    ASSERT_GT(count, 0);
    const double scale = 2.0;
    const CCVector3d shift(1000.0, 2000.0, 3000.0);
    auto* cloud = new ccPointCloud(QStringLiteral("shifted_kitti"));
    ASSERT_TRUE(cloud->reserve(static_cast<unsigned>(count)));
    for (int i = 0; i < count; ++i) {
        const CCVector3d pg(pts[i * 4 + 0], pts[i * 4 + 1], pts[i * 4 + 2]);
        const CCVector3d pl = (pg + shift) * scale;
        cloud->addPoint(CCVector3(pl.toPC()));
    }
    auto* intensity = new ccScalarField("intensity");
    ASSERT_TRUE(intensity->reserveSafe(static_cast<unsigned>(count)));
    for (int i = 0; i < count; ++i) intensity->addElement(pts[i * 4 + 3]);
    intensity->computeMinAndMax();
    ASSERT_GE(cloud->addScalarField(intensity), 0);
    cloud->setGlobalScale(scale);
    cloud->setGlobalShift(shift);

    OpenPCDetWorker::Settings ws;
    ws.modelPath = model;
    ws.device = QStringLiteral("cpu");
    OpenPCDetRunResult result;
    ASSERT_TRUE(runOneWorker(ws, cloud, &result));
    delete cloud;

    ASSERT_FALSE(result.detections.isEmpty())
            << "worker must lift stored local points back to the LiDAR "
               "frame (Pglobal = Plocal/scale - shift)";
    EXPECT_EQ(result.detections.size(), fileResult.detections.size())
            << "lifted input must reproduce the raw-file detections";
}

TEST(OpenPCDetDualInput, WorkerSnapshotsInputAtConstruction) {
    // The worker takes a VALUE snapshot of the DB cloud at construction; a
    // later mutation of the live cloud (the user editing/deleting it while a
    // run is queued) must not leak into the running inference. A pointer-
    // based worker reads the mutated coordinates and finds nothing.
    const QString extractRoot = ecvTestDataRepository::extractDir();
    const QString model = extractRoot + QStringLiteral(
                                                "/openpcdet_models/"
                                                "kitti_pointpillar.f16.gguf");
    const QString bin = extractRoot + QStringLiteral(
                                                "/openpcdet_inputs/"
                                                "kitti_000008.bin");
    if (!QFileInfo::exists(model) || !QFileInfo::exists(bin)) {
        GTEST_SKIP() << "OpenPCDet assets not cached under "
                     << extractRoot.toStdString();
    }

    QFile binFile(bin);
    ASSERT_TRUE(binFile.open(QIODevice::ReadOnly));
    const QByteArray bytes = binFile.readAll();
    binFile.close();
    std::vector<float> pts;
    int count = 0;
    QString error;
    ASSERT_TRUE(OpenPCDetHelpers::parsePointsBin(bytes, 4, &pts, &count,
                                                 &error))
            << error.toStdString();

    auto* cloud = new ccPointCloud(QStringLiteral("snapshot_kitti"));
    ASSERT_TRUE(cloud->reserve(static_cast<unsigned>(count)));
    for (int i = 0; i < count; ++i) {
        cloud->addPoint(
                CCVector3(pts[i * 4 + 0], pts[i * 4 + 1], pts[i * 4 + 2]));
    }
    auto* intensity = new ccScalarField("intensity");
    ASSERT_TRUE(intensity->reserveSafe(static_cast<unsigned>(count)));
    for (int i = 0; i < count; ++i) intensity->addElement(pts[i * 4 + 3]);
    intensity->computeMinAndMax();
    ASSERT_GE(cloud->addScalarField(intensity), 0);

    OpenPCDetWorker::Settings ws;
    ws.modelPath = model;
    ws.device = QStringLiteral("cpu");
    OpenPCDetWorker worker(ws, cloud);  // snapshot happens HERE
    // Mutate the live cloud after construction (well-defined: the cloud is
    // alive) — far outside any LiDAR range.
    cloud->Translate(CCVector3(9000.f, 9000.f, 9000.f));

    // Run THE SAME worker instance that holds the construction-time
    // snapshot.
    ResultCatcher catcher;
    QObject::connect(&worker, &OpenPCDetWorker::resultReady, &catcher,
                     &ResultCatcher::onResult, Qt::DirectConnection);
    worker.start();
    ASSERT_TRUE(worker.wait(180000));
    worker.releaseContextOnMainThread();
    delete cloud;
    ASSERT_TRUE(catcher.got) << "snapshotted worker produced no result";
    const OpenPCDetRunResult result = catcher.result;
    ASSERT_FALSE(result.detections.isEmpty())
            << "the construction-time snapshot must survive later live-cloud "
               "mutations";
}
