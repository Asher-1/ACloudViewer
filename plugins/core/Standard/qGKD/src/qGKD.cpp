// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qGKD.h"

#include <ecvImage.h>
#include <ecvMainAppInterface.h>
#include <ecvPluginDbNaming.h>

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QMainWindow>
#include <QMessageBox>
#include <QSettings>
#include <QTimer>
#include <QUuid>

#include "ecvPersistentSettings.h"

#ifdef AICore_ENABLED
#include "aicore/backend_capi.h"
#include "aicore/runtime_capi.h"
#endif

namespace {

bool isGKDOutputImage(const ccImage* img) {
    if (!img) return false;
    if (img->getName().startsWith(QStringLiteral("GKD_"))) return true;
    return img->getMetaData(QStringLiteral("GKD")).isValid();
}

}  // namespace

qGKD::qGKD(QObject* parent)
    : QObject(parent), ccStdPluginInterface(":/CC/plugin/qGKD/info.json") {
    ecvPS::registerSettingsGroup(QStringLiteral("qGKD"));
    qRegisterMetaType<GKDRunResult>("GKDRunResult");
    qRegisterMetaType<GKDWorker::Settings>("GKDWorker::Settings");
    m_action = new QAction(tr("GKD Keypoint Detection"), this);
    m_action->setToolTip(
            tr("GKDT general keypoint detection (text / visual / multi-object) "
               "via GGML"));
    m_action->setIcon(QIcon(":/CC/plugin/qGKD/images/qGKD.svg"));
    connect(m_action, &QAction::triggered, this, &qGKD::showDialog);

    m_inferenceHeartbeat = new QTimer(this);
    m_inferenceHeartbeat->setInterval(10000);
    connect(m_inferenceHeartbeat, &QTimer::timeout, this, [this]() {
        if (!m_worker || !m_worker->isRunning() || !m_dialog) return;
        m_inferenceElapsedSeconds += 10;
        m_dialog->appendLog(tr("[GKD] Task is running (%1 s elapsed)...")
                                    .arg(m_inferenceElapsedSeconds));
    });
}

QList<QAction*> qGKD::getActions() { return {m_action}; }

void qGKD::onNewSelection(const ccHObject::Container& selectedEntities) {
    m_selectedEntities = selectedEntities;
    if (!m_dialog || !m_dialog->isVisible()) return;
    const QStringList names = selectedDbImageNames();
    if (!names.isEmpty()) {
        m_dialog->applyDbTreeSelection(names);
    }
}

ccImage* qGKD::findDbImage(const QString& name) const {
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

QStringList qGKD::selectedDbImageNames() const {
    QStringList names;
    for (ccHObject* obj : m_selectedEntities) {
        if (!obj) continue;
        if (obj->isA(CV_TYPES::IMAGE)) {
            ccImage* img = dynamic_cast<ccImage*>(obj);
            if (!img || img->data().isNull()) continue;
            if (isGKDOutputImage(img)) continue;
            names.append(obj->getName());
        } else if (obj->isGroup()) {
            ccHObject::Container images;
            obj->filterChildren(images, true, CV_TYPES::IMAGE, false);
            for (ccHObject* child : images) {
                if (!child) continue;
                ccImage* img = dynamic_cast<ccImage*>(child);
                if (!img || img->data().isNull()) continue;
                if (isGKDOutputImage(img)) continue;
                names.append(child->getName());
            }
        }
    }
    return names;
}

bool qGKD::resolveInputPath(const QString& rawPath,
                            QString* outPath,
                            QString* errorMsg) {
    outPath->clear();
    if (rawPath.startsWith(QStringLiteral("db://"))) {
        const QString name = rawPath.mid(5);
        ccImage* img = findDbImage(name);
        if (!img) {
            if (errorMsg) *errorMsg = tr("DB image not found: %1").arg(name);
            return false;
        }
        if (img->data().isNull()) {
            if (errorMsg) {
                *errorMsg = tr("DB image has no pixel data: %1").arg(name);
            }
            return false;
        }
        const QString tmpDir = GKDDialog::modelCacheDir() + "/../tmp";
        QDir().mkpath(tmpDir);
        *outPath = tmpDir + "/gkd-" +
                   QUuid::createUuid().toString(QUuid::WithoutBraces) + ".png";
        if (!img->data().save(*outPath)) {
            if (errorMsg) {
                *errorMsg = tr("Failed to export DB image: %1").arg(name);
            }
            return false;
        }
        m_stagedInputFiles << *outPath;
        return true;
    }
    if (QFile::exists(rawPath)) {
        *outPath = rawPath;
        return true;
    }
    if (errorMsg) *errorMsg = tr("Input file not found: %1").arg(rawPath);
    return false;
}

void qGKD::clearStagedInputFiles() {
    for (const QString& path : m_stagedInputFiles) {
        QFile::remove(path);
    }
    m_stagedInputFiles.clear();
}

void qGKD::refreshDbImages() {
    if (!m_app || !m_dialog) return;
    ccHObject* root = m_app->dbRootObject();
    if (!root) {
        m_dialog->setDbImages({});
        return;
    }
    ccHObject::Container images;
    root->filterChildren(images, true, CV_TYPES::IMAGE, false);
    QList<GKDImageEntry> entries;
    for (ccHObject* obj : images) {
        if (!obj || !obj->isEnabled()) continue;
        ccImage* img = dynamic_cast<ccImage*>(obj);
        if (!img || img->data().isNull() || isGKDOutputImage(img)) continue;
        GKDImageEntry entry;
        entry.name = obj->getName();
        entry.preview = img->data();
        entries.append(entry);
    }
    m_dialog->setDbImages(entries);
}

void qGKD::showDialog() {
    if (!m_app) return;
    if (!m_dialog) {
        m_dialog = new GKDDialog(static_cast<QWidget*>(m_app->getMainWindow()));
        m_dialog->setAppInterface(m_app);
        connect(m_dialog, &GKDDialog::runRequested, this, &qGKD::executeTask);
        connect(m_dialog, &GKDDialog::cancelRequested, this, &qGKD::cancelTask);
        connect(m_dialog, &GKDDialog::refreshDbImagesRequested, this,
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

void qGKD::executeTask(const GKDWorker::Settings& settings) {
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
    if (!resolveInputPath(settings.inputPath, &resolvedPath, &err)) {
        clearStagedInputFiles();
        m_dialog->appendLog(err);
        return;
    }

    GKDWorker::Settings workerSettings = settings;
    workerSettings.inputPath = resolvedPath;

    QString workerDevice = settings.device;
#ifdef AICore_ENABLED
    if (aicore_warmup_backend(workerDevice.toUtf8().constData()) != 0) {
        if (aicore_is_gpu_device(workerDevice.toUtf8().constData())) {
            workerDevice = QStringLiteral("cpu");
            m_dialog->appendLog(
                    tr("[GKD] GPU backend unavailable — using CPU for "
                       "this run."));
        }
    }
#endif
    workerSettings.device = workerDevice;

    m_currentSettings = settings;
    m_currentSettings.device = workerDevice;
    m_currentSettings.inputPath = resolvedPath;

    m_worker = new GKDWorker(workerSettings, this);
    connect(m_worker, &GKDWorker::logMessage, m_dialog, &GKDDialog::appendLog,
            Qt::QueuedConnection);
    connect(m_worker, &GKDWorker::taskStage, m_dialog, &GKDDialog::setTaskStage,
            Qt::QueuedConnection);
    connect(m_worker, &GKDWorker::resultReady, this, &qGKD::onResultReady,
            Qt::QueuedConnection);
    connect(m_worker, &GKDWorker::taskFinished, this, &qGKD::onTaskFinished,
            Qt::QueuedConnection);
    m_dialog->setRunning(true);
    m_inferenceElapsedSeconds = 0;
    m_inferenceHeartbeat->start();
    m_worker->start();
}

void qGKD::cancelTask() {
    if (m_worker && m_worker->isRunning()) m_worker->requestTaskCancel();
}

void qGKD::onResultReady(const GKDRunResult& result) {
    if (!m_app) return;
    if (m_currentSettings.addResultToDb && !result.renderedImage.isNull()) {
        addResultToDb(result, m_currentSettings);
    }
    if (!m_currentSettings.savePngDir.isEmpty()) {
        saveResultPng(result);
    }
    if (!m_currentSettings.addResultToDb &&
        m_currentSettings.savePngDir.isEmpty()) {
        m_dialog->appendLog(tr("[GKD] Done (no output selected)."));
    }
}

void qGKD::addResultToDb(const GKDRunResult& result,
                         const GKDWorker::Settings& settings) {
    if (!m_app || result.renderedImage.isNull()) return;

    const QString deviceTag = ecvPluginDbNaming::deviceTagFromName(
            result.resolvedDevice.isEmpty() ? settings.device
                                            : result.resolvedDevice);
    const QString modelTag =
            ecvPluginDbNaming::modelTagFromFilename(settings.modelPath);
    const QString name = ecvPluginDbNaming::makeUnique(
            QStringLiteral("GKD_%1_%2_%3")
                    .arg(modelTag, result.mode, deviceTag),
            m_app);
    auto* img = new ccImage(result.renderedImage, name);
    img->setMetaData(QStringLiteral("GKD"), true);
    img->setMetaData(QStringLiteral("GKD/Mode"), result.mode);
    img->setMetaData(QStringLiteral("GKD/Keypoints"), result.totalKeypoints);
    img->setMetaData(QStringLiteral("GKD/Objects"), result.sets.size());
    img->setMetaData(QStringLiteral("Runtime (ms)"), result.runtimeMs);
    img->setMetaData(QStringLiteral("GKD/Preprocess (ms)"),
                     result.preprocessMs);
    img->setMetaData(QStringLiteral("GKD/Postprocess (ms)"),
                     result.postprocessMs);
    img->setMetaData(QStringLiteral("GKD/Total (ms)"), result.totalRuntimeMs);
    if (!result.imagePath.isEmpty()) {
        img->setMetaData(QStringLiteral("Source"), result.imagePath);
    }
    if (!result.resolvedDevice.isEmpty()) {
        img->setMetaData(QStringLiteral("Device"), result.resolvedDevice);
    }
    if (!result.backend.isEmpty()) {
        img->setMetaData(QStringLiteral("Backend"), result.backend);
    }
    img->setMetaData(QStringLiteral("Model"),
                     QFileInfo(settings.modelPath).fileName());
    if (!result.infoJson.isEmpty()) {
        img->setMetaData(QStringLiteral("GKD/Info"),
                         QString::fromUtf8(result.infoJson));
    }
    m_app->addToDB(img, true, true, false, true);
    m_app->setSelectedInDB(img, true);
    m_dialog->appendLog(tr("[GKD] Added result image '%1'.").arg(name));
}

void qGKD::saveResultPng(const GKDRunResult& result) {
    if (result.renderedImage.isNull() ||
        m_currentSettings.savePngDir.isEmpty()) {
        return;
    }
    QDir dir(m_currentSettings.savePngDir);
    if (!dir.exists() && !dir.mkpath(QStringLiteral("."))) {
        m_dialog->appendLog(tr("[GKD] Cannot create output directory: %1")
                                    .arg(m_currentSettings.savePngDir));
        return;
    }
    const QString base = QFileInfo(result.imageName).completeBaseName();
    QString filePath =
            dir.filePath(QStringLiteral("GKD_%1_%2.png")
                                 .arg(base.isEmpty() ? tr("result") : base)
                                 .arg(QDateTime::currentDateTime().toString(
                                         QStringLiteral("yyyyMMdd_HHmmss"))));
    if (result.renderedImage.save(filePath)) {
        m_dialog->appendLog(tr("[GKD] Saved PNG: %1").arg(filePath));
    } else {
        m_dialog->appendLog(tr("[GKD] Failed to save PNG: %1").arg(filePath));
    }
}

void qGKD::onTaskFinished(bool success) {
    m_inferenceHeartbeat->stop();
    m_dialog->setRunning(false);
    if (m_worker) {
        m_worker->releaseContextOnMainThread();
        m_worker->deleteLater();
        m_worker = nullptr;
    }
    clearStagedInputFiles();
    if (!success) m_dialog->appendLog(tr("[Error] Task failed."));
}
