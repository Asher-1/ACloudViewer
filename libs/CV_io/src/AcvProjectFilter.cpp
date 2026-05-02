// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "AcvProjectFilter.h"

#include "BinFilter.h"

// CV_DB_LIB
#include <CVLog.h>
#include <ecvHObject.h>
#include <ecvViewManager.h>

// Qt
#include <QDataStream>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryFile>

const QString AcvProjectFilter::ACV_MAGIC = QStringLiteral("ACV_PROJECT");

AcvProjectFilter::AcvProjectFilter()
    : FileIOFilter({"_ACloudViewer Project Filter",
                    0.5f,
                    QStringList{"acv"},
                    "acv",
                    QStringList{GetFileFilter()},
                    QStringList{GetFileFilter()},
                    Import | Export | BuiltIn}) {}

bool AcvProjectFilter::canSave(CV_CLASS_ENUM type,
                                bool& multiple,
                                bool& exclusive) const {
    if (type == CV_TYPES::HIERARCHY_OBJECT || type == CV_TYPES::POINT_CLOUD ||
        type == CV_TYPES::MESH || type == CV_TYPES::POLY_LINE ||
        type == CV_TYPES::FACET || type == CV_TYPES::SENSOR ||
        type == CV_TYPES::PRIMITIVE || type == CV_TYPES::IMAGE ||
        type == CV_TYPES::LABEL_2D || type == CV_TYPES::VIEWPORT_2D_LABEL ||
        type == CV_TYPES::VIEWPORT_2D_OBJECT ||
        type == CV_TYPES::CAMERA_SENSOR || type == CV_TYPES::GBL_SENSOR) {
        multiple = true;
        exclusive = false;
        return true;
    }
    return false;
}

CC_FILE_ERROR AcvProjectFilter::saveToFile(ccHObject* entity,
                                            const QString& filename,
                                            const SaveParameters& parameters) {
    if (!entity) {
        return CC_FERR_BAD_ARGUMENT;
    }

    // -- 1. Serialize entities to a temp BIN file --
    QTemporaryFile tempBin;
    if (!tempBin.open()) {
        CVLog::Warning("[AcvProject] Failed to create temporary file");
        return CC_FERR_WRITING;
    }

    CC_FILE_ERROR binResult = BinFilter::SaveFileV2(tempBin, entity);
    if (binResult != CC_FERR_NO_ERROR) {
        CVLog::Warning("[AcvProject] Failed to save entities");
        return binResult;
    }
    tempBin.flush();

    // Read BIN data into memory
    tempBin.seek(0);
    QByteArray entityData = tempBin.readAll();
    tempBin.close();

    // -- 2. Build JSON metadata --
    QJsonObject metadata;

    // Manifest
    QJsonObject manifest;
    manifest["format_version"] = static_cast<int>(ACV_FORMAT_VERSION);
    manifest["app_version"] = QStringLiteral("2.0");
    manifest["entity_count"] = static_cast<int>(entity->getChildrenNumber());
    manifest["created"] =
            QDateTime::currentDateTime().toString(Qt::ISODate);
    metadata["manifest"] = manifest;

    // View/layout state from ecvViewManager
    auto& vm = ecvViewManager::instance();

    auto geometryProvider = [](ecvGenericGLDisplay* view) -> QJsonObject {
        if (!view) return {};
        QJsonObject geo;
        QWidget* w = view->asWidget();
        if (w) {
            geo["width"] = w->width();
            geo["height"] = w->height();
        }
        return geo;
    };

    QJsonObject viewLayout = vm.saveLayout(geometryProvider);
    metadata["views"] = viewLayout;

    // -- 3. Write .acv container --
    QFile outFile(filename);
    if (!outFile.open(QIODevice::WriteOnly)) {
        CVLog::Warning(
                QStringLiteral("[AcvProject] Cannot open '%1' for writing")
                        .arg(filename));
        return CC_FERR_WRITING;
    }

    QDataStream out(&outFile);
    out.setVersion(QDataStream::Qt_5_9);
    out.setByteOrder(QDataStream::BigEndian);

    out << ACV_MAGIC;
    out << ACV_FORMAT_VERSION;

    QByteArray metaBytes =
            QJsonDocument(metadata).toJson(QJsonDocument::Compact);
    out << metaBytes;
    out << entityData;

    if (out.status() != QDataStream::Ok) {
        CVLog::Warning("[AcvProject] Write error");
        outFile.close();
        return CC_FERR_WRITING;
    }

    outFile.close();

    CVLog::Print(QStringLiteral("[AcvProject] Project saved to '%1' (%2 bytes)")
                         .arg(filename)
                         .arg(outFile.size()));

    return CC_FERR_NO_ERROR;
}

CC_FILE_ERROR AcvProjectFilter::loadFile(const QString& filename,
                                          ccHObject& container,
                                          LoadParameters& parameters) {
    QFile inFile(filename);
    if (!inFile.open(QIODevice::ReadOnly)) {
        CVLog::Warning(
                QStringLiteral("[AcvProject] Cannot open '%1' for reading")
                        .arg(filename));
        return CC_FERR_READING;
    }

    QDataStream in(&inFile);
    in.setVersion(QDataStream::Qt_5_9);
    in.setByteOrder(QDataStream::BigEndian);

    // -- 1. Verify magic --
    QString magic;
    in >> magic;
    if (magic != ACV_MAGIC) {
        CVLog::Warning("[AcvProject] Invalid file format (bad magic)");
        return CC_FERR_WRONG_FILE_TYPE;
    }

    quint32 version = 0;
    in >> version;
    if (version > ACV_FORMAT_VERSION) {
        CVLog::Warning(
                QStringLiteral("[AcvProject] File version %1 is newer than "
                               "supported version %2")
                        .arg(version)
                        .arg(ACV_FORMAT_VERSION));
        return CC_FERR_WRONG_FILE_TYPE;
    }

    // -- 2. Read metadata JSON --
    QByteArray metaBytes;
    in >> metaBytes;

    QJsonParseError parseErr;
    QJsonDocument metaDoc = QJsonDocument::fromJson(metaBytes, &parseErr);
    if (metaDoc.isNull()) {
        CVLog::Warning(
                QStringLiteral("[AcvProject] Metadata parse error: %1")
                        .arg(parseErr.errorString()));
        return CC_FERR_READING;
    }
    QJsonObject metadata = metaDoc.object();

    // -- 3. Read entity data --
    QByteArray entityData;
    in >> entityData;
    inFile.close();

    if (entityData.isEmpty()) {
        CVLog::Warning("[AcvProject] No entity data in project file");
        return CC_FERR_NO_LOAD;
    }

    // -- 4. Load entities via BinFilter from temp file --
    QTemporaryFile tempBin;
    if (!tempBin.open()) {
        CVLog::Warning("[AcvProject] Failed to create temporary file");
        return CC_FERR_READING;
    }
    tempBin.write(entityData);
    tempBin.flush();
    tempBin.seek(0);

    CC_FILE_ERROR binResult = BinFilter::LoadFileV2(
            tempBin, container, 0, true, parameters.parentWidget);
    tempBin.close();

    if (binResult != CC_FERR_NO_ERROR) {
        CVLog::Warning("[AcvProject] Failed to load entities from project");
        return binResult;
    }

    // -- 5. Restore view/layout state --
    if (metadata.contains("views")) {
        QJsonObject viewLayout = metadata["views"].toObject();
        auto& vm = ecvViewManager::instance();
        vm.restoreLayout(viewLayout, [](const QJsonObject&) {
            // View restoration is handled by the app layer after loading;
            // we just store the state for MainWindow to pick up.
        });
    }

    CVLog::Print(QStringLiteral("[AcvProject] Project loaded from '%1'")
                         .arg(filename));

    return CC_FERR_NO_ERROR;
}
