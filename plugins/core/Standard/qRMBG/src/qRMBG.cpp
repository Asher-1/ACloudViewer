// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qRMBG.h"

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
#include <QUuid>

#include "ecvPersistentSettings.h"

#ifdef AICore_ENABLED
#include "aicore/backend_capi.h"
#include "aicore/runtime_capi.h"
#endif

namespace {

bool isRMBGOutputImage(const ccImage* img) {
    if (!img) return false;
    if (img->getName().startsWith(QStringLiteral("RMBG_"))) return true;
    return img->getMetaData(QStringLiteral("RMBG")).isValid();
}

}  // namespace

qRMBG::qRMBG(QObject* parent)
    : QObject(parent),
      ccStdPluginInterface(":/CC/plugin/qRMBG/info.json") {
    ecvPS::registerSettingsGroup(QStringLiteral("qRMBG"));
    m_action = new QAction(tr("RMBG Remove Background"), this);
    m_action->setToolTip(tr("RMBG-2.0 background removal (GGML)"));
    m_action->setIcon(QIcon(":/CC/plugin/qRMBG/images/qRMBG.svg"));
    connect(m_action, &QAction::triggered, this, &qRMBG::showDialog);
}

QList<QAction*> qRMBG::getActions() { return {m_action}; }

void qRMBG::onNewSelection(const ccHObject::Container& selectedEntities) {
    m_selectedEntities = selectedEntities;
    if (!m_dialog || !m_dialog->isVisible()) return;
    const QStringList names = selectedDbImageNames();
    if (!names.isEmpty()) {
        m_dialog->applyDbTreeSelection(names);
    }
}

ccImage* qRMBG::findDbImage(const QString& name) const {
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

QStringList qRMBG::selectedDbImageNames() const {
    QStringList names;
    for (ccHObject* obj : m_selectedEntities) {
        if (!obj) continue;
        if (obj->isA(CV_TYPES::IMAGE)) {
            ccImage* img = dynamic_cast<ccImage*>(obj);
            if (!img || img->data().isNull()) continue;
            if (isRMBGOutputImage(img)) continue;
            names.append(obj->getName());
        } else if (obj->isGroup()) {
            ccHObject::Container images;
            obj->filterChildren(images, true, CV_TYPES::IMAGE, false);
            for (ccHObject* child : images) {
                if (!child) continue;
                ccImage* img = dynamic_cast<ccImage*>(child);
                if (!img || img->data().isNull()) continue;
                if (isRMBGOutputImage(img)) continue;
                names.append(child->getName());
            }
        }
    }
    return names;
}

bool qRMBG::resolveInputPath(const QString& rawPath,
                             QString& outPath,
                             QString* errorMsg) {
    outPath.clear();
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
        const QString tmpDir = RMBGDialog::modelCacheDir() + "/../tmp";
        QDir().mkpath(tmpDir);
        outPath = tmpDir + "/rmbg-" +
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
    if (errorMsg) *errorMsg = tr("Input file not found: %1").arg(rawPath);
    return false;
}

void qRMBG::clearStagedInputFiles() {
    for (const QString& path : m_stagedInputFiles) {
        QFile::remove(path);
    }
    m_stagedInputFiles.clear();
}

void qRMBG::refreshDbImages() {
    if (!m_app || !m_dialog) return;
    ccHObject* root = m_app->dbRootObject();
    if (!root) {
        m_dialog->setDbImages({});
        return;
    }
    ccHObject::Container images;
    root->filterChildren(images, true, CV_TYPES::IMAGE, false);
    QList<RMBGDialog::DbImageEntry> entries;
    for (ccHObject* obj : images) {
        if (!obj || !obj->isEnabled()) continue;
        ccImage* img = dynamic_cast<ccImage*>(obj);
        if (!img || img->data().isNull() || isRMBGOutputImage(img)) continue;
        RMBGDialog::DbImageEntry entry;
        entry.name = obj->getName();
        entry.preview = img->data();
        entries.append(entry);
    }
    m_dialog->setDbImages(entries);
}

void qRMBG::showDialog() {
    if (!m_app) return;
    if (!m_dialog) {
        m_dialog = new RMBGDialog(static_cast<QWidget*>(m_app->getMainWindow()));
        connect(m_dialog, &RMBGDialog::runRequested, this,
                &qRMBG::executeTask);
        connect(m_dialog, &RMBGDialog::cancelRequested, this,
                &qRMBG::cancelTask);
        connect(m_dialog, &RMBGDialog::refreshDbImagesRequested, this,
                [this]() { refreshDbImages(); });
        connect(m_dialog, &RMBGDialog::liveCaptureReady, this,
                [this](const RMBGRunResult& result) {
                    RMBGDialog::Settings s = m_dialog->getSettings();
                    addResultToDb(result, s, tr("live"));
                    saveResultPng(result, tr("live"));
                });
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

void qRMBG::executeTask(const RMBGDialog::Settings& settings) {
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
        clearStagedInputFiles();
        m_dialog->appendLog(err);
        return;
    }

    RMBGDialog::Settings workerSettings = settings;
    workerSettings.inputPath = resolvedPath;

    QString workerDevice = settings.device;
#ifdef AICore_ENABLED
    if (aicore_warmup_backend(workerDevice.toUtf8().constData()) != 0) {
        if (aicore_is_gpu_device(workerDevice.toUtf8().constData())) {
            workerDevice = QStringLiteral("cpu");
            m_dialog->appendLog(
                    tr("[RMBG] GPU backend unavailable — using CPU for "
                       "this run."));
        }
    }
#endif

    RMBGWorker::Settings ws;
    ws.modelPath = settings.modelPath;
    ws.inputPath = workerSettings.inputPath;
    ws.threads = settings.threads;
    ws.device = workerDevice;

    m_currentSettings = settings;
    m_worker = new RMBGWorker(ws, this);
    connect(m_worker, &RMBGWorker::logMessage, m_dialog,
            &RMBGDialog::appendLog);
    connect(m_worker, &RMBGWorker::progressUpdate, m_dialog,
            &RMBGDialog::setProgress);
    connect(m_worker, &RMBGWorker::resultReady, this,
            &qRMBG::onResultReady);
    connect(m_worker, &RMBGWorker::taskFinished, this,
            &qRMBG::onTaskFinished);
    m_dialog->setRunning(true);
    m_worker->start();
}

void qRMBG::cancelTask() {
    if (m_worker && m_worker->isRunning()) m_worker->requestTaskCancel();
}

void qRMBG::onResultReady(const RMBGRunResult& result) {
    if (!m_app) return;

    const bool addToDb =
            m_currentSettings.addResultToDb && !result.resultImage.isNull();
    if (addToDb) {
        addResultToDb(result, m_currentSettings, result.imageName);
    }
    if (!m_currentSettings.savePngDir.isEmpty()) {
        saveResultPng(result, result.imageName);
    }
    if (!addToDb && m_currentSettings.savePngDir.isEmpty()) {
        m_dialog->appendLog(tr("[RMBG] Done (no output selected)."));
    }
}

void qRMBG::addResultToDb(const RMBGRunResult& result,
                          const RMBGDialog::Settings& settings,
                          const QString& sourceLabel) {
    if (!m_app || result.resultImage.isNull()) return;

    const QString deviceTag = ecvPluginDbNaming::deviceTagFromName(
            result.resolvedDevice.isEmpty() ? settings.device
                                            : result.resolvedDevice);
    const QString name = ecvPluginDbNaming::makeUnique(
            QStringLiteral("RMBG_%1_%2")
                    .arg(sourceLabel, deviceTag),
            m_app);
    auto* img = new ccImage(result.resultImage, name);
    img->setMetaData(QStringLiteral("RMBG"), true);
    img->setMetaData(QStringLiteral("RMBG/AlphaMean"), result.alphaMean);
    img->setMetaData(QStringLiteral("RMBG/ForegroundRatio"),
                     result.foregroundRatio);
    img->setMetaData(QStringLiteral("Runtime (ms)"), result.runtimeMs);
    if (!result.imagePath.isEmpty()) {
        img->setMetaData(QStringLiteral("Source"), result.imagePath);
    } else {
        img->setMetaData(QStringLiteral("Source"), sourceLabel);
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
        img->setMetaData(QStringLiteral("RMBG/Info"),
                         QString::fromUtf8(result.infoJson));
    }
    m_app->addToDB(img, true, true, false, true);
    m_app->setSelectedInDB(img, true);
    m_dialog->appendLog(tr("[RMBG] Added result image '%1'.").arg(name));
}

void qRMBG::saveResultPng(const RMBGRunResult& result,
                          const QString& sourceLabel) {
    if (result.resultImage.isNull() ||
        m_currentSettings.savePngDir.isEmpty()) {
        return;
    }
    QDir dir(m_currentSettings.savePngDir);
    if (!dir.exists() && !dir.mkpath(QStringLiteral("."))) {
        m_dialog->appendLog(tr("[RMBG] Cannot create output directory: %1")
                                    .arg(m_currentSettings.savePngDir));
        return;
    }
    const QString base = QFileInfo(result.imageName).completeBaseName();
    QString filePath = dir.filePath(
            QStringLiteral("RMBG_%1_%2.png")
                    .arg(base.isEmpty() ? sourceLabel : base)
                    .arg(QDateTime::currentDateTime().toString(
                            QStringLiteral("yyyyMMdd_HHmmss"))));
    if (result.resultImage.save(filePath)) {
        m_dialog->appendLog(tr("[RMBG] Saved PNG: %1").arg(filePath));
    } else {
        m_dialog->appendLog(tr("[RMBG] Failed to save PNG: %1").arg(filePath));
    }
}

void qRMBG::onTaskFinished(bool success) {
    m_dialog->setRunning(false);
    if (m_worker) {
        m_worker->releaseContextOnMainThread();
        m_worker->deleteLater();
        m_worker = nullptr;
    }
    clearStagedInputFiles();
    if (!success) m_dialog->appendLog(tr("[Error] Task failed."));
}
