// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qDeepLSD.h"

#include <ecvImage.h>
#include <ecvMainAppInterface.h>
#include <ecvPluginDbNaming.h>

#include <QDir>
#include <QFile>
#include <QMainWindow>
#include <QMessageBox>

namespace {

bool isDeepLSDOutputImage(const ccImage* img) {
    if (!img) return false;
    if (img->getName().startsWith(QStringLiteral("DeepLSD_"))) return true;
    return img->getMetaData(QStringLiteral("DeepLSD")).isValid();
}

}  // namespace

qDeepLSD::qDeepLSD(QObject* parent)
    : QObject(parent), ccStdPluginInterface(":/CC/plugin/qDeepLSD/info.json") {
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
                                QString* errorMsg) const {
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
        outPath = tmpDir + "/" + name + ".png";
        if (!img->data().save(outPath)) {
            if (errorMsg) {
                *errorMsg = tr("Failed to export DB image: %1").arg(name);
            }
            return false;
        }
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

    QString resolvedPath;
    QString err;
    if (!resolveInputPath(settings.inputPath, resolvedPath, &err)) {
        m_dialog->appendLog(err);
        return;
    }
    if (settings.modelPath.isEmpty()) {
        m_dialog->appendLog(tr("[Error] Model required."));
        return;
    }

    m_currentSettings = settings;
    m_worker = new DeepLSDWorker(
            DeepLSDWorker::Settings{settings.modelPath, resolvedPath,
                                    settings.threads, settings.device},
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
    if (m_worker && m_worker->isRunning()) m_worker->requestInterruption();
}

void qDeepLSD::onResultReady(const DeepLSDRunResult& result) {
    if (!m_app || result.overlay.isNull()) return;
    if (!m_currentSettings.addResultToDb) {
        m_dialog->appendLog(tr("[DeepLSD] Done (DB export disabled)."));
        return;
    }
    const QString name = ecvPluginDbNaming::makeUnique(
            QStringLiteral("DeepLSD_%1").arg(result.imageName), m_app);
    auto* img = new ccImage(result.overlay, name);
    img->setMetaData(QStringLiteral("DeepLSD"), true);
    img->setMetaData(QStringLiteral("Runtime (ms)"), result.runtimeMs);
    img->setMetaData(QStringLiteral("Source"), result.imagePath);
    m_app->addToDB(img, true, true, false, true);
    m_app->setSelectedInDB(img, true);
    m_dialog->appendLog(tr("[DeepLSD] Added '%1' to DB.").arg(name));
}

void qDeepLSD::onTaskFinished(bool success) {
    m_dialog->setRunning(false);
    if (m_worker) {
        m_worker->releaseContextOnMainThread();
        m_worker->deleteLater();
        m_worker = nullptr;
    }
    if (!success) m_dialog->appendLog(tr("[Error] Task failed."));
}
