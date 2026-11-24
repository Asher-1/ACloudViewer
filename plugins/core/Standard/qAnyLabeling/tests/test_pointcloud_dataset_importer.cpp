// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>
#include <cstdio>

#include "PointCloudDatasetExporter.h"
#include "PointCloudDatasetImporter.h"

namespace {

QByteArray read(const QString& path) {
    QFile file(path);
    return file.open(QIODevice::ReadOnly) ? file.readAll() : QByteArray();
}

bool write(const QString& path, const QByteArray& bytes) {
    QFile file(path);
    return file.open(QIODevice::WriteOnly) && file.write(bytes) == bytes.size();
}
}  // namespace

int readDatasetProbe(int argc, char** argv) {
    const auto arguments = QCoreApplication::arguments();
    if (argc == 4 && arguments.at(1) == QStringLiteral("--read")) {
        PointCloudDatasetImportResult result;
        QString error;
        if (!importPointCloudDataset(arguments.at(2), arguments.at(3), &result,
                                     {}, &error)) {
            std::fprintf(stderr, "%s\n", qPrintable(error));
            return 1;
        }
        QJsonArray classes;
        QJsonArray frames;
        for (const auto& item : result.classes)
            classes.append(QJsonObject{
                    {QStringLiteral("id"), item.id},
                    {QStringLiteral("name"), item.name},
                    {QStringLiteral("color"), item.color.name().toUpper()}});
        for (const auto& frame : result.frames) {
            QJsonArray boxes;
            for (const auto& box : frame.cuboids) boxes.append(box.toJson());
            frames.append(QJsonObject{
                    {QStringLiteral("name"), frame.name.isEmpty()
                                                     ? QJsonValue()
                                                     : QJsonValue(frame.name)},
                    {QStringLiteral("index"), frame.index},
                    {QStringLiteral("cuboids"), boxes}});
        }
        const QByteArray json =
                QJsonDocument(QJsonObject{{QStringLiteral("classes"), classes},
                                          {QStringLiteral("frames"), frames}})
                        .toJson();
        std::fwrite(json.constData(), 1, size_t(json.size()), stdout);
        return 0;
    }
    return 0;
}

TEST(PointCloudDatasetImporter, RoundTripFormatsAndAtomicApplyRollback) {
    QTemporaryDir scratch;
    ASSERT_TRUE(scratch.isValid())
            << (QStringLiteral("temporary folder unavailable")).toStdString();
    const bool fixture =
            QCoreApplication::arguments().size() == 3 &&
            QCoreApplication::arguments().at(1) == QStringLiteral("--fixture");
    QDir temporary(fixture ? QFileInfo(QCoreApplication::arguments().at(2))
                                     .absoluteFilePath()
                           : scratch.path());
    ASSERT_TRUE(QDir().mkpath(temporary.path()))
            << (QStringLiteral("fixture folder unavailable")).toStdString();

    PointCloudFrame frame;
    frame.path = temporary.filePath(QStringLiteral("frame_01.bin"));
    frame.points = {QVector4D(1.0f, 2.0f, 3.0f, 0.5f)};
    PointCloudCuboid cuboid;
    cuboid.id = 7;
    cuboid.classId = 1;
    cuboid.center = {{1.0, 2.0, 3.0}};
    cuboid.size = {{4.0, 2.0, 1.5}};
    cuboid.rotation = {{0.1, 0.2, 0.3}};
    cuboid.occluded = true;
    cuboid.locked = true;
    frame.cuboids = {cuboid};

    PointCloudDatasetExportOptions options;
    options.frames = {frame};
    options.classNames.insert(1, QStringLiteral("car"));
    options.classColors.insert(1, QColor(QStringLiteral("#123456")));
    PointCloudCuboid importedCuboid = cuboid;
    importedCuboid.id = 1;
    for (const QString& format :
         {QStringLiteral("datumaro"), QStringLiteral("kitti_raw"),
          QStringLiteral("sly_pointcloud")}) {
        options.format = format;
        options.outputPath =
                temporary.filePath(format + QStringLiteral(".zip"));
        QString error;
        ASSERT_TRUE(exportPointCloudDataset(options, {}, &error))
                << (QStringLiteral("%1 export failed: %2").arg(format, error))
                           .toStdString();
        PointCloudDatasetImportResult imported;
        PointCloudCuboid expectedCuboid = importedCuboid;
        if (format == QStringLiteral("kitti_raw"))
            expectedCuboid.locked = false;
        int lastProgress = -1;
        ASSERT_TRUE(importPointCloudDataset(
                options.outputPath, format, &imported,
                [&](int current, int total) {
                    if (total <= 0 || current < lastProgress || current > total)
                        return false;
                    lastProgress = current;
                    return true;
                },
                &error))
                << (QStringLiteral("%1 import failed: %2").arg(format, error))
                           .toStdString();
        ASSERT_TRUE(imported.classes.size() == 1 &&
                    imported.classes.at(0).name == QStringLiteral("car"))
                << (QStringLiteral("%1 class mismatch").arg(format))
                           .toStdString();
        ASSERT_TRUE(imported.frames.size() == 1 &&
                    imported.frames.at(0).name == QStringLiteral("frame_01"))
                << (QStringLiteral("%1 frame mismatch").arg(format))
                           .toStdString();
        ASSERT_TRUE(imported.frames.at(0).cuboids.size() == 1 &&
                    imported.frames.at(0).cuboids.at(0) == expectedCuboid)
                << (QStringLiteral("%1 cuboid mismatch: %2").arg(format, error))
                           .toStdString();
        ASSERT_TRUE(lastProgress >= 1)
                << (QStringLiteral("%1 progress missing").arg(format))
                           .toStdString();
    }

    PointCloudDatasetExportOptions cancelledOptions;
    cancelledOptions.outputPath =
            temporary.filePath(QStringLiteral("cancelled.zip"));
    cancelledOptions.format = QStringLiteral("datumaro");
    cancelledOptions.frames = {frame};
    cancelledOptions.classNames.insert(1, QStringLiteral("car"));
    QString error;
    ASSERT_TRUE(exportPointCloudDataset(cancelledOptions, {}, &error))
            << (QStringLiteral("cancelled fixture export failed: %1")
                        .arg(error))
                       .toStdString();
    PointCloudDatasetImportResult imported;
    ASSERT_TRUE(!importPointCloudDataset(
            cancelledOptions.outputPath, cancelledOptions.format, &imported,
            [](int, int) { return false; }, &error))
            << (QStringLiteral("import cancellation must fail")).toStdString();
    ASSERT_TRUE(
            error.contains(QStringLiteral("cancelled"), Qt::CaseInsensitive))
            << (QStringLiteral("import cancellation error missing"))
                       .toStdString();

    const QString labelPath =
            temporary.filePath(QStringLiteral("frame_01.label"));
    const QString cuboidPath = pointCloudCuboidPath(labelPath);
    const QString configPath =
            temporary.filePath(QStringLiteral("classes.json"));
    PointCloudCuboid oldCuboid = cuboid;
    oldCuboid.classId =
            40;  // Reserve an ID even if it has no class definition.
    ASSERT_TRUE(
            write(frame.path, QByteArray("cloud input")) &&
            write(labelPath, QByteArray("segmentation input")) &&
            savePointCloudCuboids(cuboidPath, frame.path, {oldCuboid}, &error))
            << (QStringLiteral("import target fixtures failed: %1").arg(error))
                       .toStdString();
    const QByteArray beforeCuboids = read(cuboidPath);
    const QByteArray beforeLabels = read(labelPath);
    const QByteArray beforeCloud = read(frame.path);
    const QVector<PointCloudDatasetImportClass> classes{
            {0, QStringLiteral("Unlabeled"), QColor(QStringLiteral("#808080"))},
            {1, QStringLiteral("pedestrian"),
             QColor(QStringLiteral("#FF0000"))}};
    const QJsonArray segmentation{
            QJsonObject{{QStringLiteral("id"), 0},
                        {QStringLiteral("name"), QStringLiteral("Background")},
                        {QStringLiteral("color"), QStringLiteral("#010203")}},
            QJsonObject{{QStringLiteral("id"), 9},
                        {QStringLiteral("name"), QStringLiteral("Road")},
                        {QStringLiteral("color"), QStringLiteral("#345678")}}};
    const QByteArray beforeConfig =
            QJsonDocument(QJsonObject{{QStringLiteral("version"), 1},
                                      {QStringLiteral("segmentation"),
                                       QJsonObject{{QStringLiteral("classes"),
                                                    segmentation}}}})
                    .toJson();
    ASSERT_TRUE(write(configPath, beforeConfig))
            << (QStringLiteral("class config fixture failed")).toStdString();
    const PointCloudDatasetImportTarget target{
            frame.path, labelPath, cuboidPath, {}, false};
    PointCloudDatasetImportPlan plan;
    ASSERT_TRUE(preparePointCloudDatasetImport(
            cancelledOptions.outputPath, QStringLiteral("datumaro"), {target},
            classes, configPath, &plan, {}, &error))
            << (QStringLiteral("import preparation failed: %1").arg(error))
                       .toStdString();
    ASSERT_TRUE(plan.changes.size() == 2 && plan.targets.size() == 1 &&
                plan.classes.size() == 2 && plan.previousObjects == 1 &&
                plan.newClasses == 1 &&
                plan.targets.at(0).cuboids.at(0).classId == 41)
            << (QStringLiteral("class merge must preserve reserved IDs and "
                               "count replaced objects"))
                       .toStdString();
    ASSERT_TRUE(read(cuboidPath) == beforeCuboids &&
                read(configPath) == beforeConfig)
            << (QStringLiteral("preparation must not mutate files"))
                       .toStdString();
    if (fixture) {
        QJsonArray definitions;
        QJsonArray changes;
        for (const auto& entry : plan.classes)
            definitions.append(QJsonObject{
                    {QStringLiteral("id"), entry.id},
                    {QStringLiteral("name"), entry.name},
                    {QStringLiteral("color"), entry.color.name().toUpper()}});
        for (const auto& change : plan.changes)
            changes.append(QJsonObject{
                    {QStringLiteral("path"), change.path},
                    {QStringLiteral("existed"), change.existed},
                    {QStringLiteral("before"),
                     QString::fromLatin1(change.before.toBase64())},
                    {QStringLiteral("after"),
                     QJsonDocument::fromJson(change.after).object()}});
        const QJsonObject report{
                {QStringLiteral("classes"), definitions},
                {QStringLiteral("changes"), changes},
                {QStringLiteral("previous_objects"), plan.previousObjects},
                {QStringLiteral("new_classes"), plan.newClasses}};
        ASSERT_TRUE(write(temporary.filePath(QStringLiteral("cpp-plan.json")),
                          QJsonDocument(report).toJson()))
                << (QStringLiteral("could not write import-plan fixture"))
                           .toStdString();
    }

    // Cancel after the sidecar has been replaced, then after all replacements.
    for (int cancelAt : {1, 2}) {
        ASSERT_TRUE(!applyPointCloudDatasetImport(
                plan,
                [cancelAt](int current, int) { return current < cancelAt; },
                &error))
                << (QStringLiteral("mid-commit cancellation must fail"))
                           .toStdString();
        ASSERT_TRUE(read(cuboidPath) == beforeCuboids &&
                    read(configPath) == beforeConfig &&
                    read(labelPath) == beforeLabels &&
                    read(frame.path) == beforeCloud)
                << (QStringLiteral(
                            "cancelled import must roll back sidecar and "
                            "config"))
                           .toStdString();
        const QStringList leftovers =
                QDir(temporary.path())
                        .entryList({QStringLiteral("*.import.*"),
                                    QStringLiteral("*.backup.*")},
                                   QDir::Files | QDir::Hidden);
        ASSERT_TRUE(leftovers.isEmpty())
                << (QStringLiteral("cancelled import leaked staged files"))
                           .toStdString();
    }
    ASSERT_TRUE(write(configPath, QByteArray("external edit")))
            << (QStringLiteral("external edit fixture failed")).toStdString();
    ASSERT_TRUE(!applyPointCloudDatasetImport(plan, {}, &error))
            << (QStringLiteral("stale import plan must be rejected"))
                       .toStdString();
    ASSERT_TRUE(read(cuboidPath) == beforeCuboids &&
                read(configPath) == QByteArray("external edit"))
            << (QStringLiteral("stale plan must preserve external changes"))
                       .toStdString();
    ASSERT_TRUE(write(configPath, beforeConfig))
            << (QStringLiteral("config restore failed")).toStdString();
    bool editedDuringCommit = false;
    ASSERT_TRUE(!applyPointCloudDatasetImport(
            plan,
            [&](int current, int) {
                if (current == 1 && !editedDuringCommit)
                    editedDuringCommit =
                            write(configPath, QByteArray("concurrent edit"));
                return true;
            },
            &error))
            << (QStringLiteral("concurrent file changes must be rejected"))
                       .toStdString();
    ASSERT_TRUE(editedDuringCommit && read(cuboidPath) == beforeCuboids &&
                read(configPath) == QByteArray("concurrent edit"))
            << (QStringLiteral("concurrent edit must survive while prior "
                               "sidecar is rolled back"))
                       .toStdString();
    ASSERT_TRUE(write(configPath, beforeConfig))
            << (QStringLiteral("concurrent config restore failed"))
                       .toStdString();
    ASSERT_TRUE(applyPointCloudDatasetImport(plan, {}, &error))
            << (QStringLiteral("import apply failed: %1").arg(error))
                       .toStdString();
    QVector<PointCloudCuboid> loaded;
    const QJsonObject savedConfig =
            QJsonDocument::fromJson(read(configPath)).object();
    ASSERT_TRUE(
            loadPointCloudCuboids(cuboidPath, frame.path, &loaded, &error) &&
            loaded == plan.targets.at(0).cuboids)
            << (QStringLiteral(
                        "applied sidecar must load with remapped classes"))
                       .toStdString();
    ASSERT_TRUE(savedConfig.value(QStringLiteral("segmentation"))
                        .toObject()
                        .value(QStringLiteral("classes"))
                        .toArray() == segmentation)
            << (QStringLiteral("detection import must preserve segmentation "
                               "classes"))
                       .toStdString();
    ASSERT_TRUE(savedConfig.value(QStringLiteral("detection"))
                        .toObject()
                        .value(QStringLiteral("classes"))
                        .toArray()
                        .size() == 2)
            << (QStringLiteral("grouped detection class config missing"))
                       .toStdString();
    ASSERT_TRUE(read(labelPath) == beforeLabels &&
                read(frame.path) == beforeCloud)
            << (QStringLiteral("import must preserve point payload and "
                               "segmentation labels"))
                       .toStdString();

    // A plan with no existing destinations also removes newly created files on
    // rollback.
    auto newTarget = target;
    newTarget.cuboidPath =
            temporary.filePath(QStringLiteral("new.cuboids.json"));
    const QString newConfig = temporary.filePath(QStringLiteral("new.json"));
    PointCloudDatasetImportPlan newPlan;
    ASSERT_TRUE(preparePointCloudDatasetImport(
            cancelledOptions.outputPath, QStringLiteral("datumaro"),
            {newTarget}, classes, newConfig, &newPlan, {}, &error))
            << (QStringLiteral("new destination preparation failed: %1")
                        .arg(error))
                       .toStdString();
    ASSERT_TRUE(!applyPointCloudDatasetImport(
            newPlan, [](int current, int total) { return current < total; },
            &error))
            << (QStringLiteral("final cancellation must fail")).toStdString();
    ASSERT_TRUE(!QFile::exists(newTarget.cuboidPath) &&
                !QFile::exists(newConfig))
            << (QStringLiteral(
                        "rollback must remove newly created import files"))
                       .toStdString();

    auto duplicateClasses = classes;
    duplicateClasses.append(
            {2, QStringLiteral("pedestrian"), QColor(Qt::green)});
    auto mismatch = target;
    mismatch.path = temporary.filePath(QStringLiteral("unrelated.bin"));
    auto ambiguous = target;
    ambiguous.path = temporary.filePath(QStringLiteral("another/frame_01.ply"));
    ASSERT_TRUE(!preparePointCloudDatasetImport(
            cancelledOptions.outputPath, QStringLiteral("datumaro"), {target},
            duplicateClasses, configPath, &newPlan, {}, &error))
            << (QStringLiteral(
                        "duplicate current class names must be rejected"))
                       .toStdString();
    ASSERT_TRUE(!preparePointCloudDatasetImport(
            cancelledOptions.outputPath, QStringLiteral("datumaro"), {mismatch},
            classes, configPath, &newPlan, {}, &error))
            << (QStringLiteral("unmatched frame must be rejected"))
                       .toStdString();
    ASSERT_TRUE(!preparePointCloudDatasetImport(
            cancelledOptions.outputPath, QStringLiteral("datumaro"),
            {target, ambiguous}, classes, configPath, &newPlan, {}, &error))
            << (QStringLiteral("ambiguous frame stem must be rejected"))
                       .toStdString();
    ASSERT_TRUE(!preparePointCloudDatasetImport(
            cancelledOptions.outputPath, QStringLiteral("datumaro"), {target},
            classes, target.cuboidPath, &newPlan, {}, &error))
            << (QStringLiteral("duplicate sidecar/config destination must be "
                               "rejected"))
                       .toStdString();
    ASSERT_TRUE(!preparePointCloudDatasetImport(
            cancelledOptions.outputPath, QStringLiteral("datumaro"), {target},
            classes, configPath, &newPlan, [](int, int) { return false; },
            &error))
            << (QStringLiteral("preparation cancellation must fail"))
                       .toStdString();

    auto knownClasses = classes;
    knownClasses.append({5, QStringLiteral("car"), QColor(Qt::green)});
    ASSERT_TRUE(preparePointCloudDatasetImport(
            cancelledOptions.outputPath, QStringLiteral("datumaro"), {target},
            knownClasses, QString(), &newPlan, {}, &error))
            << (QStringLiteral("existing named class preparation failed: %1")
                        .arg(error))
                       .toStdString();
    ASSERT_TRUE(newPlan.newClasses == 0 &&
                newPlan.targets.at(0).cuboids.at(0).classId == 5)
            << (QStringLiteral("matching class name must reuse its current ID"))
                       .toStdString();
    QByteArray corrupted = read(cancelledOptions.outputPath);
    const int central = corrupted.indexOf(QByteArray("PK\001\002", 4));
    ASSERT_TRUE(central >= 0)
            << (QStringLiteral("ZIP central entry missing")).toStdString();
    corrupted[central + 16] = char(quint8(corrupted.at(central + 16)) ^ 1);
    const QString corruptedPath =
            temporary.filePath(QStringLiteral("corrupted.zip"));
    ASSERT_TRUE(write(corruptedPath, corrupted))
            << (QStringLiteral("corrupted ZIP fixture failed")).toStdString();
    ASSERT_TRUE(!importPointCloudDataset(corruptedPath,
                                         QStringLiteral("datumaro"), &imported,
                                         {}, &error) &&
                error.contains(QStringLiteral("checksum")))
            << (QStringLiteral("corrupted ZIP checksum must be rejected"))
                       .toStdString();
}

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    if (argc == 4 && app.arguments().at(1) == "--read")
        return readDatasetProbe(argc, argv);
    const bool fixture = argc == 3 && app.arguments().at(1) == "--fixture";
    if (fixture) {
        // Preserve the fixture generator command while the same assertions run
        // under GoogleTest. Fixture consumers read files, not test-runner text.
        int testArgc = 1;
        ::testing::InitGoogleTest(&testArgc, argv);
    } else
        ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
