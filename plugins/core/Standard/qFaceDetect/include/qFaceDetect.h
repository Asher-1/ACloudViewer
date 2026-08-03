// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ecvHObject.h>
#include <ecvStdPluginInterface.h>

#include <QAction>

#include "FaceDetectDialog.h"
#include "FaceDetectWorker.h"

class ccImage;

class qFaceDetect : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "cvcorp.cloudviewer.plugin.qFaceDetect" FILE
                          "../info.json")

public:
    explicit qFaceDetect(QObject* parent = nullptr);

    void onNewSelection(const ccHObject::Container& selectedEntities) override;
    QList<QAction*> getActions() override;

private slots:
    void showDialog();
    void executeTask(const FaceDetectDialog::Settings& settings);
    void cancelTask();
    void onResultReady(const FaceDetectRunResult& result);
    void onTaskFinished(bool success);

private:
    ccImage* findDbImage(const QString& name) const;
    QStringList selectedDbImageNames() const;
    bool resolveInputPath(const QString& rawPath,
                          QString& outPath,
                          QString* errorMsg) const;
    void refreshDbImages();
    void addResultToDb(const FaceDetectRunResult& result,
                       const FaceDetectDialog::Settings& settings,
                       const QString& sourceLabel);
    void onAuthVisualizationReady(const QImage& annotated, const QString& summary);

    QAction* m_action = nullptr;
    FaceDetectDialog* m_dialog = nullptr;
    FaceDetectWorker* m_worker = nullptr;
    FaceDetectDialog::Settings m_currentSettings;
    ccHObject::Container m_selectedEntities;
};
