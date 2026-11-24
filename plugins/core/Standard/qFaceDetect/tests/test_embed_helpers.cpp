// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <QFile>
#include <QSqlDatabase>
#include <QSqlQuery>
#include <QTemporaryDir>
#include <QThread>
#include <QUuid>
#include <QVariant>
#include <atomic>
#include <cmath>
#include <thread>

#include "FaceDetectEmbedHelpers.h"
#include "FaceRegistryStore.h"

static FaceDetectBox makeBox(
        float x1, float y1, float x2, float y2, float score) {
    FaceDetectBox box;
    box.x1 = x1;
    box.y1 = y1;
    box.x2 = x2;
    box.y2 = y2;
    box.score = score;
    return box;
}

TEST(FaceDetectEmbedHelpers, ParseDetectJson) {
    const QByteArray json =
            R"({"faces":[{"score":0.91,"box":[10,20,110,120],"landmarks":[[30,40],[50,40],[70,50],[40,70],[60,70]]}]})";
    const auto faces = FaceDetectEmbed::parseDetectJson(json);
    EXPECT_TRUE(faces.size() == 1);
    EXPECT_TRUE(std::abs(faces[0].score - 0.91f) < 1e-4f);
    EXPECT_TRUE(std::abs(faces[0].x1 - 10.f) < 1e-4f);
    EXPECT_TRUE(std::abs(faces[0].landmarks[0][0] - 30.f) < 1e-4f);
}

TEST(FaceDetectEmbedHelpers, ParseAnalyzeJson) {
    const QByteArray json =
            R"({"faces":[{"score":0.88,"box":[0,0,50,50],"age":29,"gender":"F"}]})";
    const auto faces = FaceDetectEmbed::parseAnalyzeJson(json);
    EXPECT_TRUE(faces.size() == 1);
    EXPECT_TRUE(faces[0].age == 29);
    EXPECT_TRUE(faces[0].gender == 'F');
}

TEST(FaceDetectEmbedHelpers, ParseDenseJson) {
    const QByteArray json =
            R"({"faces":[{"score":0.95,"box":[1,2,3,4],"landmarks_5":[[10,11],[12,13],[14,15],[16,17],[18,19]],"landmarks_2d":[[1,2],[3,4]],"landmarks_3d":[[1,2,3]]}]})";
    const auto faces = FaceDetectEmbed::parseDenseJson(json);
    EXPECT_TRUE(faces.size() == 1);
    EXPECT_TRUE(faces[0].denseLandmarks2d.size() == 2);
    EXPECT_TRUE(faces[0].denseLandmarks3d.size() == 1);
    EXPECT_TRUE(std::abs(faces[0].landmarks[0][0] - 10.f) < 1e-4f);
}

TEST(FaceDetectEmbedHelpers, FilterAndScale) {
    std::vector<FaceDetectBox> faces = {
            makeBox(0, 0, 10, 10, 0.9f),
            makeBox(0, 0, 10, 10, 0.3f),
    };
    FaceDetectEmbed::filterFacesByScore(&faces, 0.5f);
    EXPECT_TRUE(faces.size() == 1);
    EXPECT_TRUE(faces[0].score > 0.5f);

    faces = {makeBox(100, 100, 200, 200, 0.8f)};
    FaceDetectEmbed::scaleFaceBoxes(&faces, 2.f);
    EXPECT_TRUE(std::abs(faces[0].x2 - 100.f) < 1e-4f);
}

TEST(FaceDetectEmbedHelpers, OffsetFaceBoxes) {
    // Regression: the old implementation wrote "box.y1 += box.x2 += dx",
    // which corrupted y1 with the translated x2 value.
    std::vector<FaceDetectBox> faces = {makeBox(10, 20, 30, 40, 0.9f)};
    faces[0].landmarks[0][0] = 12.f;
    faces[0].landmarks[0][1] = 22.f;
    FaceDetectEmbed::offsetFaceBoxes(&faces, 5.f, 7.f);
    EXPECT_TRUE(std::abs(faces[0].x1 - 15.f) < 1e-4f);
    EXPECT_TRUE(std::abs(faces[0].y1 - 27.f) < 1e-4f);
    EXPECT_TRUE(std::abs(faces[0].x2 - 35.f) < 1e-4f);
    EXPECT_TRUE(std::abs(faces[0].y2 - 47.f) < 1e-4f);
    EXPECT_TRUE(std::abs(faces[0].landmarks[0][0] - 17.f) < 1e-4f);
    EXPECT_TRUE(std::abs(faces[0].landmarks[0][1] - 29.f) < 1e-4f);
    // Corrupt-old-behavior check: y1 must not be y1 + (x2 + dx).
    EXPECT_TRUE(std::abs(faces[0].y1 - (20.f + 30.f + 5.f)) > 1e-4f);
}

TEST(FaceDetectEmbedHelpers, ExpandFaceBox) {
    const FaceDetectBox box = makeBox(40, 40, 60, 60, 0.9f);
    const FaceDetectBox expanded = FaceDetectEmbed::expandFaceBox(
            box, FaceDetectEmbed::kDefaultCropMarginRatio, 100, 100);
    EXPECT_TRUE(expanded.x1 < box.x1);
    EXPECT_TRUE(expanded.y1 < box.y1);
    EXPECT_TRUE(expanded.x2 > box.x2);
    EXPECT_TRUE(expanded.y2 > box.y2);
    EXPECT_TRUE(expanded.x1 >= 0.f);
    EXPECT_TRUE(expanded.x2 <= 100.f);
}

TEST(FaceDetectEmbedHelpers, FormatLabels) {
    const QString match =
            FaceDetectEmbed::formatMatchLabel(QStringLiteral("Ross"), 0.321f);
    EXPECT_TRUE(match.contains(QStringLiteral("Ross")));
    EXPECT_TRUE(match.contains(QStringLiteral("d=0.321")));

    const QString miss =
            FaceDetectEmbed::formatNoMatchLabel(0.812f, QStringLiteral("Joey"));
    EXPECT_TRUE(miss.contains(QStringLiteral("NO MATCH")));
    EXPECT_TRUE(miss.contains(QStringLiteral("Joey")));
}

static std::vector<float> unitVec(int dim, float value) {
    std::vector<float> v(static_cast<size_t>(dim), 0.f);
    if (dim > 0) v[0] = value;
    return v;
}

TEST(FaceDetectEmbedHelpers, RegistryStoreMatch) {
    QTemporaryDir tmp;
    EXPECT_TRUE(tmp.isValid());
    const QString dbPath = tmp.filePath(QStringLiteral("registry.db"));

    FaceRegistryStore store(dbPath);
    EXPECT_TRUE(store.open());

    FaceRegistryEntry a;
    a.name = QStringLiteral("Alice");
    a.embedding = unitVec(4, 1.f);
    a.embedDim = 4;
    EXPECT_TRUE(store.addEntry(std::move(a)));

    FaceRegistryEntry b;
    b.name = QStringLiteral("Bob");
    b.embedding = unitVec(4, 0.f);
    b.embedDim = 4;
    b.embedding[1] = 1.f;
    EXPECT_TRUE(store.addEntry(std::move(b)));

    EXPECT_TRUE(store.entries().size() == 2);

    const auto match = store.bestMatch(unitVec(4, 1.f), 0.5f);
    EXPECT_TRUE(match.has_value());
    EXPECT_TRUE(match->entry.name == QStringLiteral("Alice"));

    const float dist =
            FaceRegistryStore::cosineDistance(unitVec(4, 1.f), unitVec(4, 1.f));
    EXPECT_TRUE(dist < 1e-5f);
}

TEST(FaceDetectEmbedHelpers, RegistryConcurrentRead) {
    QTemporaryDir tmp;
    EXPECT_TRUE(tmp.isValid());
    FaceRegistryStore store(tmp.filePath(QStringLiteral("registry.db")));
    EXPECT_TRUE(store.open());

    FaceRegistryEntry e;
    e.name = QStringLiteral("Concurrent");
    e.embedding = unitVec(8, 1.f);
    e.embedDim = 8;
    EXPECT_TRUE(store.addEntry(std::move(e)));

    std::atomic<bool> reader_ok{true};
    std::thread reader([&]() {
        for (int i = 0; i < 200; ++i) {
            const auto match = store.bestMatch(unitVec(8, 1.f), 1.f);
            if (!match || match->entry.name != QStringLiteral("Concurrent")) {
                reader_ok = false;
                break;
            }
        }
    });

    for (int i = 0; i < 50; ++i) {
        FaceRegistryEntry upd;
        upd.name = QStringLiteral("Concurrent");
        upd.embedding = unitVec(8, 1.f);
        upd.embedDim = 8;
        store.addEntry(std::move(upd));
    }
    reader.join();
    EXPECT_TRUE(reader_ok.load());
}

TEST(FaceDetectEmbedHelpers, RegistryMigratesLegacyJson) {
    QTemporaryDir tmp;
    EXPECT_TRUE(tmp.isValid());
    QFile legacy(tmp.filePath(QStringLiteral("face_registry.json")));
    EXPECT_TRUE(legacy.open(QIODevice::WriteOnly));
    const QByteArray json =
            R"({"entries":[{"id":"legacy-id","name":"Legacy User","model":"arcface","created":"2024-01-01T00:00:00Z","embedding":[1,0,0,0]}]})";
    EXPECT_TRUE(legacy.write(json) == json.size());
    legacy.close();

    FaceRegistryStore store(tmp.filePath(QStringLiteral("registry.db")));
    EXPECT_TRUE(store.open());
    const auto entries = store.entries();
    EXPECT_TRUE(entries.size() == 1);
    if (!entries.empty()) {
        EXPECT_TRUE(entries.front().id == QStringLiteral("legacy-id"));
    }
    EXPECT_TRUE(QFile::exists(
            tmp.filePath(QStringLiteral("face_registry.json.migrated"))));
}

TEST(FaceDetectEmbedHelpers, RegistryRejectsInvalidEmbeddings) {
    QTemporaryDir tmp;
    EXPECT_TRUE(tmp.isValid());
    const QString dbPath = tmp.filePath(QStringLiteral("registry.db"));
    FaceRegistryStore store(dbPath);
    EXPECT_TRUE(store.open());

    FaceRegistryEntry empty;
    empty.name = QStringLiteral("Empty");
    EXPECT_TRUE(!store.addEntry(std::move(empty)));

    FaceRegistryEntry nonFinite;
    nonFinite.name = QStringLiteral("Non finite");
    nonFinite.embedding = {1.f, std::nanf("")};
    EXPECT_TRUE(!store.addEntry(std::move(nonFinite)));
}

TEST(FaceDetectEmbedHelpers, RegistryIgnoresMalformedBlob) {
    QTemporaryDir tmp;
    EXPECT_TRUE(tmp.isValid());
    const QString dbPath = tmp.filePath(QStringLiteral("registry.db"));
    const QString connection =
            QStringLiteral("malformed_test_") + QUuid::createUuid().toString();
    {
        QSqlDatabase db = QSqlDatabase::addDatabase(QStringLiteral("QSQLITE"),
                                                    connection);
        db.setDatabaseName(dbPath);
        EXPECT_TRUE(db.open());
        QSqlQuery query(db);
        EXPECT_TRUE(query.exec(QStringLiteral(
                "CREATE TABLE faces (id TEXT PRIMARY KEY,name TEXT NOT NULL,"
                "model TEXT,dim INTEGER NOT NULL,created TEXT,"
                "embedding BLOB NOT NULL,thumb BLOB)")));
        query.prepare(QStringLiteral(
                "INSERT INTO faces(id,name,dim,embedding) VALUES(?,?,?,?)"));
        query.addBindValue(QStringLiteral("bad"));
        query.addBindValue(QStringLiteral("Bad blob"));
        query.addBindValue(1);
        query.addBindValue(QByteArray("abc", 3));
        EXPECT_TRUE(query.exec());
        db.close();
    }
    QSqlDatabase::removeDatabase(connection);

    FaceRegistryStore store(dbPath);
    EXPECT_TRUE(store.open());
    EXPECT_TRUE(store.entries().empty());
}
