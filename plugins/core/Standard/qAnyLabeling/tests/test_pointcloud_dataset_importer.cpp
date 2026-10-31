// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

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
bool check(bool value, const QString& message) {
    if (!value) std::fprintf(stderr, "%s\n", qPrintable(message));
    return value;
}

QByteArray read(const QString& path) {
    QFile file(path);
    return file.open(QIODevice::ReadOnly) ? file.readAll() : QByteArray();
}

bool write(const QString& path, const QByteArray& bytes) {
    QFile file(path);
    return file.open(QIODevice::WriteOnly) && file.write(bytes) == bytes.size();
}
}  // namespace

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    if (argc == 4 && app.arguments().at(1) == QStringLiteral("--read")) {
        PointCloudDatasetImportResult result;
        QString error;
        if (!importPointCloudDataset(app.arguments().at(2),
                                     app.arguments().at(3), &result, {},
                                     &error)) {
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
    QTemporaryDir scratch;
    if (!check(scratch.isValid(),
               QStringLiteral("temporary folder unavailable")))
        return 1;
    const bool fixture =
            argc == 3 && app.arguments().at(1) == QStringLiteral("--fixture");
    QDir temporary(fixture ? QFileInfo(app.arguments().at(2)).absoluteFilePath()
                           : scratch.path());
    if (!check(QDir().mkpath(temporary.path()),
               QStringLiteral("fixture folder unavailable")))
        return 1;

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
        if (!check(exportPointCloudDataset(options, {}, &error),
                   QStringLiteral("%1 export failed: %2").arg(format, error)))
            return 1;
        PointCloudDatasetImportResult imported;
        PointCloudCuboid expectedCuboid = importedCuboid;
        if (format == QStringLiteral("kitti_raw"))
            expectedCuboid.locked = false;
        int lastProgress = -1;
        if (!check(importPointCloudDataset(
                           options.outputPath, format, &imported,
                           [&](int current, int total) {
                               if (total <= 0 || current < lastProgress ||
                                   current > total)
                                   return false;
                               lastProgress = current;
                               return true;
                           },
                           &error),
                   QStringLiteral("%1 import failed: %2").arg(format, error)) ||
            !check(imported.classes.size() == 1 &&
                           imported.classes.at(0).name == QStringLiteral("car"),
                   QStringLiteral("%1 class mismatch").arg(format)) ||
            !check(imported.frames.size() == 1 &&
                           imported.frames.at(0).name ==
                                   QStringLiteral("frame_01"),
                   QStringLiteral("%1 frame mismatch").arg(format)) ||
            !check(imported.frames.at(0).cuboids.size() == 1 &&
                           imported.frames.at(0).cuboids.at(0) ==
                                   expectedCuboid,
                   QStringLiteral("%1 cuboid mismatch: %2")
                           .arg(format, error)) ||
            !check(lastProgress >= 1,
                   QStringLiteral("%1 progress missing").arg(format)))
            return 1;
    }

    PointCloudDatasetExportOptions cancelledOptions;
    cancelledOptions.outputPath =
            temporary.filePath(QStringLiteral("cancelled.zip"));
    cancelledOptions.format = QStringLiteral("datumaro");
    cancelledOptions.frames = {frame};
    cancelledOptions.classNames.insert(1, QStringLiteral("car"));
    QString error;
    if (!check(exportPointCloudDataset(cancelledOptions, {}, &error),
               QStringLiteral("cancelled fixture export failed: %1")
                       .arg(error)))
        return 1;
    PointCloudDatasetImportResult imported;
    if (!check(!importPointCloudDataset(
                       cancelledOptions.outputPath, cancelledOptions.format,
                       &imported, [](int, int) { return false; }, &error),
               QStringLiteral("import cancellation must fail")) ||
        !check(error.contains(QStringLiteral("cancelled"), Qt::CaseInsensitive),
               QStringLiteral("import cancellation error missing")))
        return 1;

    const QString labelPath =
            temporary.filePath(QStringLiteral("frame_01.label"));
    const QString cuboidPath = pointCloudCuboidPath(labelPath);
    const QString configPath =
            temporary.filePath(QStringLiteral("classes.json"));
    PointCloudCuboid oldCuboid = cuboid;
    oldCuboid.classId =
            40;  // Reserve an ID even if it has no class definition.
    if (!check(write(frame.path, QByteArray("cloud input")) &&
                       write(labelPath, QByteArray("segmentation input")) &&
                       savePointCloudCuboids(cuboidPath, frame.path,
                                             {oldCuboid}, &error),
               QStringLiteral("import target fixtures failed: %1").arg(error)))
        return 1;
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
    if (!check(write(configPath, beforeConfig),
               QStringLiteral("class config fixture failed")))
        return 1;
    const PointCloudDatasetImportTarget target{
            frame.path, labelPath, cuboidPath, {}, false};
    PointCloudDatasetImportPlan plan;
    if (!check(preparePointCloudDatasetImport(
                       cancelledOptions.outputPath, QStringLiteral("datumaro"),
                       {target}, classes, configPath, &plan, {}, &error),
               QStringLiteral("import preparation failed: %1").arg(error)) ||
        !check(plan.changes.size() == 2 && plan.targets.size() == 1 &&
                       plan.classes.size() == 2 && plan.previousObjects == 1 &&
                       plan.newClasses == 1 &&
                       plan.targets.at(0).cuboids.at(0).classId == 41,
               QStringLiteral("class merge must preserve reserved IDs and "
                              "count replaced objects")) ||
        !check(read(cuboidPath) == beforeCuboids &&
                       read(configPath) == beforeConfig,
               QStringLiteral("preparation must not mutate files")))
        return 1;
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
        if (!check(write(temporary.filePath(QStringLiteral("cpp-plan.json")),
                         QJsonDocument(report).toJson()),
                   QStringLiteral("could not write import-plan fixture")))
            return 1;
    }

    // Cancel after the sidecar has been replaced, then after all replacements.
    for (int cancelAt : {1, 2}) {
        if (!check(!applyPointCloudDatasetImport(
                           plan,
                           [cancelAt](int current, int) {
                               return current < cancelAt;
                           },
                           &error),
                   QStringLiteral("mid-commit cancellation must fail")) ||
            !check(read(cuboidPath) == beforeCuboids &&
                           read(configPath) == beforeConfig &&
                           read(labelPath) == beforeLabels &&
                           read(frame.path) == beforeCloud,
                   QStringLiteral("cancelled import must roll back sidecar and "
                                  "config")))
            return 1;
        const QStringList leftovers =
                QDir(temporary.path())
                        .entryList({QStringLiteral("*.import.*"),
                                    QStringLiteral("*.backup.*")},
                                   QDir::Files | QDir::Hidden);
        if (!check(leftovers.isEmpty(),
                   QStringLiteral("cancelled import leaked staged files")))
            return 1;
    }
    if (!check(write(configPath, QByteArray("external edit")),
               QStringLiteral("external edit fixture failed")) ||
        !check(!applyPointCloudDatasetImport(plan, {}, &error),
               QStringLiteral("stale import plan must be rejected")) ||
        !check(read(cuboidPath) == beforeCuboids &&
                       read(configPath) == QByteArray("external edit"),
               QStringLiteral("stale plan must preserve external changes")) ||
        !check(write(configPath, beforeConfig),
               QStringLiteral("config restore failed")))
        return 1;
    bool editedDuringCommit = false;
    if (!check(!applyPointCloudDatasetImport(
                       plan,
                       [&](int current, int) {
                           if (current == 1 && !editedDuringCommit)
                               editedDuringCommit =
                                       write(configPath,
                                             QByteArray("concurrent edit"));
                           return true;
                       },
                       &error),
               QStringLiteral("concurrent file changes must be rejected")) ||
        !check(editedDuringCommit && read(cuboidPath) == beforeCuboids &&
                       read(configPath) == QByteArray("concurrent edit"),
               QStringLiteral("concurrent edit must survive while prior "
                              "sidecar is rolled back")) ||
        !check(write(configPath, beforeConfig),
               QStringLiteral("concurrent config restore failed")) ||
        !check(applyPointCloudDatasetImport(plan, {}, &error),
               QStringLiteral("import apply failed: %1").arg(error)))
        return 1;
    QVector<PointCloudCuboid> loaded;
    const QJsonObject savedConfig =
            QJsonDocument::fromJson(read(configPath)).object();
    if (!check(loadPointCloudCuboids(cuboidPath, frame.path, &loaded, &error) &&
                       loaded == plan.targets.at(0).cuboids,
               QStringLiteral(
                       "applied sidecar must load with remapped classes")) ||
        !check(savedConfig.value(QStringLiteral("segmentation"))
                               .toObject()
                               .value(QStringLiteral("classes"))
                               .toArray() == segmentation,
               QStringLiteral("detection import must preserve segmentation "
                              "classes")) ||
        !check(savedConfig.value(QStringLiteral("detection"))
                               .toObject()
                               .value(QStringLiteral("classes"))
                               .toArray()
                               .size() == 2,
               QStringLiteral("grouped detection class config missing")) ||
        !check(read(labelPath) == beforeLabels &&
                       read(frame.path) == beforeCloud,
               QStringLiteral("import must preserve point payload and "
                              "segmentation labels")))
        return 1;

    // A plan with no existing destinations also removes newly created files on
    // rollback.
    auto newTarget = target;
    newTarget.cuboidPath =
            temporary.filePath(QStringLiteral("new.cuboids.json"));
    const QString newConfig = temporary.filePath(QStringLiteral("new.json"));
    PointCloudDatasetImportPlan newPlan;
    if (!check(preparePointCloudDatasetImport(
                       cancelledOptions.outputPath, QStringLiteral("datumaro"),
                       {newTarget}, classes, newConfig, &newPlan, {}, &error),
               QStringLiteral("new destination preparation failed: %1")
                       .arg(error)) ||
        !check(!applyPointCloudDatasetImport(
                       newPlan,
                       [](int current, int total) { return current < total; },
                       &error),
               QStringLiteral("final cancellation must fail")) ||
        !check(!QFile::exists(newTarget.cuboidPath) &&
                       !QFile::exists(newConfig),
               QStringLiteral(
                       "rollback must remove newly created import files")))
        return 1;

    auto duplicateClasses = classes;
    duplicateClasses.append(
            {2, QStringLiteral("pedestrian"), QColor(Qt::green)});
    auto mismatch = target;
    mismatch.path = temporary.filePath(QStringLiteral("unrelated.bin"));
    auto ambiguous = target;
    ambiguous.path = temporary.filePath(QStringLiteral("another/frame_01.ply"));
    if (!check(!preparePointCloudDatasetImport(
                       cancelledOptions.outputPath, QStringLiteral("datumaro"),
                       {target}, duplicateClasses, configPath, &newPlan, {},
                       &error),
               QStringLiteral(
                       "duplicate current class names must be rejected")) ||
        !check(!preparePointCloudDatasetImport(
                       cancelledOptions.outputPath, QStringLiteral("datumaro"),
                       {mismatch}, classes, configPath, &newPlan, {}, &error),
               QStringLiteral("unmatched frame must be rejected")) ||
        !check(!preparePointCloudDatasetImport(
                       cancelledOptions.outputPath, QStringLiteral("datumaro"),
                       {target, ambiguous}, classes, configPath, &newPlan, {},
                       &error),
               QStringLiteral("ambiguous frame stem must be rejected")) ||
        !check(!preparePointCloudDatasetImport(
                       cancelledOptions.outputPath, QStringLiteral("datumaro"),
                       {target}, classes, target.cuboidPath, &newPlan, {},
                       &error),
               QStringLiteral("duplicate sidecar/config destination must be "
                              "rejected")) ||
        !check(!preparePointCloudDatasetImport(
                       cancelledOptions.outputPath, QStringLiteral("datumaro"),
                       {target}, classes, configPath, &newPlan,
                       [](int, int) { return false; }, &error),
               QStringLiteral("preparation cancellation must fail")))
        return 1;

    auto knownClasses = classes;
    knownClasses.append({5, QStringLiteral("car"), QColor(Qt::green)});
    if (!check(preparePointCloudDatasetImport(
                       cancelledOptions.outputPath, QStringLiteral("datumaro"),
                       {target}, knownClasses, QString(), &newPlan, {}, &error),
               QStringLiteral("existing named class preparation failed: %1")
                       .arg(error)) ||
        !check(newPlan.newClasses == 0 &&
                       newPlan.targets.at(0).cuboids.at(0).classId == 5,
               QStringLiteral("matching class name must reuse its current ID")))
        return 1;
    QByteArray corrupted = read(cancelledOptions.outputPath);
    const int central = corrupted.indexOf(QByteArray("PK\001\002", 4));
    if (!check(central >= 0, QStringLiteral("ZIP central entry missing")))
        return 1;
    corrupted[central + 16] = char(quint8(corrupted.at(central + 16)) ^ 1);
    const QString corruptedPath =
            temporary.filePath(QStringLiteral("corrupted.zip"));
    if (!check(write(corruptedPath, corrupted),
               QStringLiteral("corrupted ZIP fixture failed")) ||
        !check(!importPointCloudDataset(corruptedPath,
                                        QStringLiteral("datumaro"), &imported,
                                        {}, &error) &&
                       error.contains(QStringLiteral("checksum")),
               QStringLiteral("corrupted ZIP checksum must be rejected")))
        return 1;

    std::puts(
            "PASS point-cloud Datumaro/KITTI/Sly import, frame/class merge, "
            "atomic apply and rollback");
    return 0;
}
