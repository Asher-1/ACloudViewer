// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qDeepLSD.h"

#include <LineSet.h>
#include <ecvImage.h>
#include <ecvMainAppInterface.h>
#include <ecvPluginDbNaming.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QMainWindow>
#include <QMessageBox>
#include <QUuid>

#include "ecvPersistentSettings.h"

#ifdef AICore_ENABLED
#include "aicore/backend_capi.h"
#include "aicore/runtime_capi.h"
#endif

namespace {

bool isDeepLSDOutputImage(const ccImage* img) {
    if (!img) return false;
    if (img->getName().startsWith(QStringLiteral("DeepLSD_"))) return true;
    return img->getMetaData(QStringLiteral("DeepLSD")).isValid();
}

void applyDeepLSDEntityMetadata(ccHObject* entity,
                                const DeepLSDDialog::Settings& settings,
                                const DeepLSDRunResult& result) {
    if (!entity) return;
    const QString device = result.resolvedDevice.isEmpty()
                                   ? settings.device
                                   : result.resolvedDevice;
    const int imageW = result.width > 0 ? result.width : result.originalWidth;
    const int imageH =
            result.height > 0 ? result.height : result.originalHeight;
    entity->setMetaData(QStringLiteral("Device"), device);
    entity->setMetaData(QStringLiteral("Model"),
                        QFileInfo(settings.modelPath).fileName());
    entity->setMetaData(QStringLiteral("MinSegmentQuality"),
                        settings.minSegmentScore);
    if (imageW > 0 && imageH > 0) {
        entity->setMetaData(QStringLiteral("ImageSize"),
                            QStringLiteral("%1x%2").arg(imageW).arg(imageH));
    }
}

}  // namespace

qDeepLSD::qDeepLSD(QObject* parent)
    : QObject(parent), ccStdPluginInterface(":/CC/plugin/qDeepLSD/info.json") {
    ecvPS::registerSettingsGroup(QStringLiteral("qDeepLSD"));
    m_action = new QAction(tr("DeepLSD Wireframe"), this);
    m_action->setToolTip(tr("Extract line wireframe fields with DeepLSD GGML"));
    m_action->setIcon(QIcon(":/CC/plugin/qDeepLSD/images/qDeepLSD.svg"));
    connect(m_action, &QAction::triggered, this, &qDeepLSD::showDialog);
}

QList<QAction*> qDeepLSD::getActions() { return {m_action}; }

void qDeepLSD::onNewSelection(const ccHObject::Container& selectedEntities) {
    m_selectedEntities = selectedEntities;
    if (!m_dialog || !m_dialog->isVisible()) return;
    const QStringList names = selectedDbImageNames();
    if (!names.isEmpty()) {
        m_dialog->applyDbTreeSelection(names);
    }
}

ccImage* qDeepLSD::findDbImage(const QString& name) const {
    if (!m_app) return nullptr;
    ccHObject* root = m_app->dbRootObject();
    if (!root) return nullptr;
    ccHObject::Container images;
    root->filterChildren(images, true, CV_TYPES::IMAGE, false);
    for (ccHObject* obj : images) {
        if (obj && obj->getName() == name) {
            return dynamic_cast<ccImage*>(obj);
        }
    }
    return nullptr;
}

QStringList qDeepLSD::selectedDbImageNames() const {
    QStringList names;
    for (ccHObject* obj : m_selectedEntities) {
        if (!obj || !obj->isA(CV_TYPES::IMAGE)) continue;
        ccImage* img = dynamic_cast<ccImage*>(obj);
        if (img && isDeepLSDOutputImage(img)) continue;
        names.append(obj->getName());
    }
    return names;
}

bool qDeepLSD::resolveInputPath(const QString& rawPath,
                                QString& outPath,
                                QString* errorMsg) {
    outPath.clear();
    if (rawPath.startsWith(QStringLiteral("db://"))) {
        const QString name = rawPath.mid(5);
        ccImage* img = findDbImage(name);
        if (!img || img->data().isNull()) {
            if (errorMsg) {
                *errorMsg = tr("DB image not found or empty: %1").arg(name);
            }
            return false;
        }
        const QString tmpDir = DeepLSDDialog::modelCacheDir() + "/../tmp";
        QDir().mkpath(tmpDir);
        outPath = tmpDir + "/deeplsd-" +
                  QUuid::createUuid().toString(QUuid::WithoutBraces) + ".png";
        if (!img->data().save(outPath)) {
            if (errorMsg) {
                *errorMsg = tr("Failed to export DB image: %1").arg(name);
            }
            return false;
        }
        m_stagedInputFiles << outPath;
        return true;
    }
    if (QFile::exists(rawPath)) {
        outPath = rawPath;
        return true;
    }
    if (errorMsg) {
        *errorMsg = tr("Input file not found: %1").arg(rawPath);
    }
    return false;
}

void qDeepLSD::clearStagedInputFiles() {
    for (const QString& path : m_stagedInputFiles) {
        QFile::remove(path);
    }
    m_stagedInputFiles.clear();
}

void qDeepLSD::refreshDbImages() {
    if (!m_app || !m_dialog) return;
    ccHObject* root = m_app->dbRootObject();
    if (!root) {
        m_dialog->setDbImages({});
        return;
    }
    ccHObject::Container images;
    root->filterChildren(images, true, CV_TYPES::IMAGE, false);
    QList<DeepLSDDialog::DbImageEntry> entries;
    for (ccHObject* obj : images) {
        if (!obj || !obj->isEnabled()) continue;
        ccImage* img = dynamic_cast<ccImage*>(obj);
        if (!img || isDeepLSDOutputImage(img)) continue;
        DeepLSDDialog::DbImageEntry entry;
        entry.name = obj->getName();
        entry.preview = img->data();
        entries.append(entry);
    }
    m_dialog->setDbImages(entries);
}

void qDeepLSD::showDialog() {
    if (!m_app) return;
    if (!m_dialog) {
        m_dialog = new DeepLSDDialog(
                static_cast<QWidget*>(m_app->getMainWindow()));
        m_dialog->setAppInterface(m_app);
        connect(m_dialog, &DeepLSDDialog::runRequested, this,
                &qDeepLSD::executeTask);
        connect(m_dialog, &DeepLSDDialog::cancelRequested, this,
                &qDeepLSD::cancelTask);
        connect(m_dialog, &DeepLSDDialog::refreshDbImagesRequested, this,
                [this]() { refreshDbImages(); });
    }
    m_dialog->refreshModelList();
    refreshDbImages();
    const QStringList selectedNames = selectedDbImageNames();
    if (!selectedNames.isEmpty()) {
        m_dialog->applyDbTreeSelection(selectedNames);
    }
    m_dialog->show();
    m_dialog->raise();
    m_dialog->activateWindow();
}

void qDeepLSD::executeTask(const DeepLSDDialog::Settings& settings) {
    if (m_worker && m_worker->isRunning()) return;

    if (m_worker) {
        m_worker->disconnect(this);
        m_worker->disconnect(m_dialog);
        m_worker->releaseContextOnMainThread();
        m_worker->deleteLater();
        m_worker = nullptr;
    }
    clearStagedInputFiles();

    if (settings.modelPath.isEmpty()) {
        m_dialog->appendLog(tr("[Error] Model required."));
        return;
    }

    QString resolvedPath;
    QString err;
    if (!resolveInputPath(settings.inputPath, resolvedPath, &err)) {
        m_dialog->appendLog(err);
        return;
    }
    QString workerDevice = settings.device;
#ifdef AICore_ENABLED
    if (aicore_warmup_backend(workerDevice.toUtf8().constData()) != 0) {
        if (aicore_is_gpu_device(workerDevice.toUtf8().constData())) {
            workerDevice = QStringLiteral("cpu");
            m_dialog->appendLog(
                    tr("[DeepLSD] GPU backend unavailable — using CPU for this "
                       "run."));
        }
    }
#endif

    m_currentSettings = settings;
    m_worker = new DeepLSDWorker(
            DeepLSDWorker::Settings{settings.modelPath, resolvedPath,
                                    settings.threads, workerDevice,
                                    settings.minSegmentScore,
                                    settings.addDistanceOverlayToDb},
            this);
    connect(m_worker, &DeepLSDWorker::logMessage, m_dialog,
            &DeepLSDDialog::appendLog);
    connect(m_worker, &DeepLSDWorker::progressUpdate, m_dialog,
            &DeepLSDDialog::setProgress);
    connect(m_worker, &DeepLSDWorker::resultReady, this,
            &qDeepLSD::onResultReady);
    connect(m_worker, &DeepLSDWorker::taskFinished, this,
            &qDeepLSD::onTaskFinished);
    m_dialog->setRunning(true);
    m_worker->start();
}

void qDeepLSD::cancelTask() {
    if (m_worker && m_worker->isRunning()) m_worker->requestTaskCancel();
}

void qDeepLSD::onResultReady(const DeepLSDRunResult& result) {
    if (!m_app) return;

    ccHObject* lastAdded = nullptr;
    int exportCount = 0;

    if (m_currentSettings.addLineVizToDb &&
        !result.lineVisualization.isNull()) {
        const QString deviceTag = ecvPluginDbNaming::deviceTagFromName(
                result.resolvedDevice.isEmpty() ? m_currentSettings.device
                                                : result.resolvedDevice);
        const QString name = ecvPluginDbNaming::makeUnique(
                QStringLiteral("DeepLSD_%1_%2")
                        .arg(result.imageName, deviceTag),
                m_app);
        auto* img = new ccImage(result.lineVisualization, name);
        img->setMetaData(QStringLiteral("DeepLSD"), true);
        img->setMetaData(QStringLiteral("DeepLSDExport"),
                         QStringLiteral("lines"));
        img->setMetaData(QStringLiteral("Runtime (ms)"), result.runtimeMs);
        img->setMetaData(QStringLiteral("Source"), result.imagePath);
        img->setMetaData(QStringLiteral("SegmentCount"),
                         static_cast<qlonglong>(result.segments.size()));
        applyDeepLSDEntityMetadata(img, m_currentSettings, result);
        m_app->addToDB(img, true, true, false, true);
        lastAdded = img;
        ++exportCount;
        m_dialog->appendLog(
                tr("[DeepLSD] Added line visualization '%1'.").arg(name));
    }

    if (m_currentSettings.addDistanceOverlayToDb &&
        !result.distanceOverlay.isNull()) {
        const QString deviceTag = ecvPluginDbNaming::deviceTagFromName(
                result.resolvedDevice.isEmpty() ? m_currentSettings.device
                                                : result.resolvedDevice);
        const QString name = ecvPluginDbNaming::makeUnique(
                QStringLiteral("DeepLSD_df_%1_%2")
                        .arg(result.imageName, deviceTag),
                m_app);
        auto* img = new ccImage(result.distanceOverlay, name);
        img->setMetaData(QStringLiteral("DeepLSD"), true);
        img->setMetaData(QStringLiteral("DeepLSDExport"),
                         QStringLiteral("distance_field"));
        img->setMetaData(QStringLiteral("Runtime (ms)"), result.runtimeMs);
        img->setMetaData(QStringLiteral("Source"), result.imagePath);
        applyDeepLSDEntityMetadata(img, m_currentSettings, result);
        m_app->addToDB(img, true, true, false, true);
        lastAdded = img;
        ++exportCount;
        m_dialog->appendLog(
                tr("[DeepLSD] Added distance-field overlay '%1'.").arg(name));
    }

    if (m_currentSettings.exportPolylinesToDb && !result.segments.empty()) {
        const QString deviceTag = ecvPluginDbNaming::deviceTagFromName(
                result.resolvedDevice.isEmpty() ? m_currentSettings.device
                                                : result.resolvedDevice);
        const QString entityName = ecvPluginDbNaming::makeUnique(
                QStringLiteral("DeepLSD_lines_%1_%2")
                        .arg(result.imageName, deviceTag),
                m_app);
        auto* lines = new cloudViewer::geometry::LineSet(
                entityName.toUtf8().constData());
        lines->setMetaData(QStringLiteral("DeepLSD"), true);
        lines->setMetaData(QStringLiteral("Source"), result.imagePath);
        applyDeepLSDEntityMetadata(lines, m_currentSettings, result);

        lines->points_.reserve(result.segments.size() * 2);
        lines->lines_.reserve(result.segments.size());
        for (size_t i = 0; i < result.segments.size(); ++i) {
            const DeepLSDLineSegment& seg = result.segments[i];
            const int i0 = static_cast<int>(lines->points_.size());
            lines->points_.emplace_back(seg.x1, seg.y1, 0.0);
            lines->points_.emplace_back(seg.x2, seg.y2, 0.0);
            lines->lines_.emplace_back(i0, i0 + 1);
        }

        lines->set2DMode(true);
        lines->setColor(ecvColor::Rgb(DeepLSDLineStyle::kRed,
                                      DeepLSDLineStyle::kGreen,
                                      DeepLSDLineStyle::kBlue));
        lines->setWidth(1);
        lines->showColors(true);

        m_app->addToDB(lines, true, true, false, true);
        lastAdded = lines;
        ++exportCount;
        m_dialog->appendLog(
                tr("[DeepLSD] Exported %1 segments as LineSet '%2'.")
                        .arg(lines->segmentCount())
                        .arg(entityName));
    }

    if (exportCount == 0) {
        m_dialog->appendLog(tr("[DeepLSD] Done (no DB export selected)."));
        return;
    }

    if (lastAdded) {
        m_app->setSelectedInDB(lastAdded, true);
    }
}

void qDeepLSD::onTaskFinished(bool success) {
    m_dialog->setRunning(false);
    if (m_worker) {
        m_worker->releaseContextOnMainThread();
        m_worker->deleteLater();
        m_worker = nullptr;
    }
    clearStagedInputFiles();
    if (!success) m_dialog->appendLog(tr("[Error] Task failed."));
}
