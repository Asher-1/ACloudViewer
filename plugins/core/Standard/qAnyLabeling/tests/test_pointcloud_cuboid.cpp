// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <QCoreApplication>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QTemporaryDir>
#include <cstdio>

#include "PointCloudDocument.h"

#define REQUIRE(x)                                               \
    do {                                                         \
        if (!(x)) {                                              \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); \
            return 1;                                            \
        }                                                        \
    } while (false)

int runCuboidProbe(int argc, char** argv) {
    QString error;
    if (argc == 4 && QString::fromLocal8Bit(argv[1]) == "--mesh-probe") {
        QFile input(QString::fromLocal8Bit(argv[2]));
        REQUIRE(input.open(QIODevice::ReadOnly));
        QJsonArray results;
        for (const auto& value :
             QJsonDocument::fromJson(input.readAll()).array()) {
            const auto item = value.toObject();
            QVector<PointCloudCuboid> boxes;
            for (const auto& entry : item["boxes"].toArray()) {
                PointCloudCuboid box;
                REQUIRE(PointCloudCuboid::fromJson(entry.toObject(), &box,
                                                   &error));
                boxes.append(box);
            }
            QHash<quint16, QColor> colors;
            const auto palette = item["colors"].toObject();
            for (auto it = palette.begin(); it != palette.end(); ++it)
                colors.insert(quint16(it.key().toInt()),
                              QColor(it.value().toString()));
            PointCloudCuboid preview;
            if (item.contains("preview"))
                REQUIRE(PointCloudCuboid::fromJson(item["preview"].toObject(),
                                                   &preview, &error));
            PointCloudCuboidMesh mesh;
            REQUIRE(pointCloudCuboidMesh(
                    boxes, item["selected"].toInt(), colors,
                    item.contains("preview") ? &preview : nullptr,
                    item["orthographic"].toBool(), &mesh, &error));
            QJsonArray vertices;
            for (int i = 0; i < mesh.vertices.size(); i += 7) {
                QJsonArray vertex;
                for (int axis = 0; axis < 7; ++axis)
                    vertex.append(mesh.vertices[i + axis]);
                vertices.append(vertex);
            }
            results.append(QJsonObject{{"face_vertices", mesh.faceVertices},
                                       {"vertices", vertices}});
        }
        QFile output(QString::fromLocal8Bit(argv[3]));
        REQUIRE(output.open(QIODevice::WriteOnly));
        output.write(QJsonDocument(results).toJson());
        return 0;
    }
    if (argc == 5 && QString::fromLocal8Bit(argv[1]) == "--roundtrip") {
        QVector<PointCloudCuboid> boxes;
        const QString source = QString::fromLocal8Bit(argv[4]);
        REQUIRE(loadPointCloudCuboids(QString::fromLocal8Bit(argv[2]), source,
                                      &boxes, &error));
        REQUIRE(savePointCloudCuboids(QString::fromLocal8Bit(argv[3]), source,
                                      boxes, &error));
        return 0;
    }
    if (argc == 4 && QString::fromLocal8Bit(argv[1]) == "--probe") {
        QFile input(QString::fromLocal8Bit(argv[2]));
        REQUIRE(input.open(QIODevice::ReadOnly));
        const auto tests = QJsonDocument::fromJson(input.readAll()).array();
        QJsonArray results;
        for (const auto& test : tests) {
            const auto t = test.toObject();
            PointCloudCuboid box;
            if (!PointCloudCuboid::fromJson(t["box"].toObject(), &box,
                                            &error)) {
                results.append(QJsonObject{{"valid", false}});
                continue;
            }
            QJsonArray contains;
            QVector<QVector4D> points;
            for (const auto& point : t["points"].toArray()) {
                const auto p = point.toArray();
                const CuboidVector v{
                        {p[0].toDouble(), p[1].toDouble(), p[2].toDouble()}};
                contains.append(box.contains(v));
                points.append(
                        QVector4D(float(v[0]), float(v[1]), float(v[2]), 0));
            }
            for (const auto& op : t["operations"].toArray()) {
                const auto o = op.toObject();
                if (o["type"] == QJsonValue("resize")) {
                    const auto s = o["signs"].toArray(),
                               d = o["delta"].toArray();
                    box = box.resized(
                            {{s[0].toInt(), s[1].toInt(), s[2].toInt()}},
                            {{d[0].toDouble(), d[1].toDouble(),
                              d[2].toDouble()}});
                } else if (o["type"] == QJsonValue("rotate"))
                    box = box.rotated(o["axis"].toInt(), o["angle"].toDouble());
            }
            QJsonArray corners;
            for (const auto& p : box.corners())
                corners.append(QJsonArray{p[0], p[1], p[2]});
            QJsonObject result{{"valid", true},
                               {"box", box.toJson()},
                               {"corners", corners},
                               {"contains", contains}};
            PointCloudCuboid fitted;
            if (box.fitted(points, &fitted, &error, false))
                result["fitted"] = fitted.toJson();
            results.append(result);
        }
        QFile output(QString::fromLocal8Bit(argv[3]));
        REQUIRE(output.open(QIODevice::WriteOnly));
        output.write(QJsonDocument(results).toJson());
        return 0;
    }
    return 0;
}

TEST(PointCloudCuboidDocument, TracksMixedUndoLocksAndSavedBaselines) {
    QString error;
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());
    PointCloudFrame frame;
    frame.path = dir.filePath("sample.bin");
    frame.labelPath = dir.filePath("sample.label");
    frame.points = {QVector4D(-1, -1, -1, 0), QVector4D(1, 1, 1, 0)};
    frame.labels = {0, 0};
    PointCloudDocument doc;
    doc.setFrame(frame);
    PointCloudCuboid box;
    ASSERT_TRUE(!doc.dirty() && doc.nextCuboidId() == 1);
    ASSERT_TRUE(doc.setCuboid(box, &error) && doc.cuboidsDirty() &&
                !doc.labelsDirty());
    ASSERT_TRUE(doc.assign({0}, 2, 3));
    box.locked = true;
    ASSERT_TRUE(doc.setCuboid(box, &error));
    auto illegal = box;
    illegal.center[0] = 2;
    ASSERT_TRUE(!doc.setCuboid(illegal, &error));
    ASSERT_TRUE(!doc.deleteCuboid(1, &error));
    box.locked = false;
    ASSERT_TRUE(doc.setCuboid(box, &error));
    ASSERT_TRUE(doc.undo() && doc.frame().cuboids[0].locked);
    ASSERT_TRUE(doc.undo() && !doc.frame().cuboids[0].locked);
    ASSERT_TRUE(doc.undo() && doc.frame().labels[0] == 0);
    ASSERT_TRUE(doc.undo() && doc.frame().cuboids.isEmpty() && !doc.dirty());
    ASSERT_TRUE(doc.redo() && doc.redo());
    const auto path = dir.filePath("sample.cuboids.json");
    ASSERT_TRUE(savePointCloudCuboids(path, frame.path, doc.frame().cuboids,
                                      &error));
    doc.markCuboidsSaved(path);
    ASSERT_TRUE(!doc.cuboidsDirty() && doc.labelsDirty());
    QVector<PointCloudCuboid> loaded;
    ASSERT_TRUE(loadPointCloudCuboids(path, frame.path, &loaded, &error));
    ASSERT_TRUE(loaded == doc.frame().cuboids);
    ASSERT_TRUE(!loadPointCloudCuboids(path, dir.filePath("other.bin"), &loaded,
                                       &error));
    ASSERT_TRUE(loaded == doc.frame().cuboids);
    ASSERT_TRUE(!savePointCloudCuboids(path, frame.path, {box, box}, &error));
    ASSERT_TRUE(!savePointCloudCuboids(frame.path, frame.path, {box}, &error));
    ASSERT_TRUE(doc.deleteCuboid(1, &error) && doc.cuboidsDirty());
    ASSERT_TRUE(savePointCloudCuboids(path, frame.path, doc.frame().cuboids,
                                      &error));
    ASSERT_TRUE(loadPointCloudCuboids(path, frame.path, &loaded, &error) &&
                loaded.isEmpty());
    const auto labelsBeforeImport = doc.frame().labels;
    ASSERT_TRUE(doc.replaceCuboids({box}, &error));
    doc.markCuboidsSaved(path);
    ASSERT_TRUE(doc.labelsDirty() && !doc.cuboidsDirty() &&
                doc.frame().labels == labelsBeforeImport);
    ASSERT_TRUE(!doc.replaceCuboids({box, box}, &error) &&
                doc.frame().cuboids == QVector<PointCloudCuboid>{box});
    ASSERT_TRUE(doc.undo() && doc.frame().cuboids.isEmpty() &&
                doc.cuboidsDirty() && doc.labelsDirty());
    ASSERT_TRUE(doc.redo() && !doc.cuboidsDirty() &&
                doc.frame().labels == labelsBeforeImport);
}

TEST(PointCloudCuboidSchema, RejectsInvalidFieldTypesAndClassIds) {
    QString error;
    PointCloudCuboid box;
    auto invalid = box.toJson();
    invalid["locked"] = 1;
    PointCloudCuboid untouched;
    ASSERT_TRUE(!PointCloudCuboid::fromJson(invalid, &untouched, &error));
    invalid = box.toJson();
    invalid["class_id"] = 0;
    ASSERT_TRUE(!PointCloudCuboid::fromJson(invalid, &untouched, &error));
    invalid = box.toJson();
    invalid["extra"] = 1;
    ASSERT_TRUE(!PointCloudCuboid::fromJson(invalid, &untouched, &error));
}

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    if (argc > 1 && !QString::fromLocal8Bit(argv[1]).startsWith("--gtest"))
        return runCuboidProbe(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
