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

#include "DeepLSDDialog.h"
#include "DeepLSDWorker.h"

class ccImage;

class qDeepLSD : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "cvcorp.cloudviewer.plugin.qDeepLSD" FILE
                          "../info.json")

public:
    explicit qDeepLSD(QObject* parent = nullptr);

    void onNewSelection(const ccHObject::Container& selectedEntities) override;
    QList<QAction*> getActions() override;

private slots:
    void showDialog();
    void executeTask(const DeepLSDDialog::Settings& settings);
    void cancelTask();
    void onResultReady(const DeepLSDRunResult& result);
    void onTaskFinished(bool success);

private:
    ccImage* findDbImage(const QString& name) const;
    QStringList selectedDbImageNames() const;
    bool resolveInputPath(const QString& rawPath,
                          QString& outPath,
                          QString* errorMsg);
    void clearStagedInputFiles();
    void refreshDbImages();

    QAction* m_action = nullptr;
    DeepLSDDialog* m_dialog = nullptr;
    DeepLSDWorker* m_worker = nullptr;
    DeepLSDDialog::Settings m_currentSettings;
    QStringList m_stagedInputFiles;
    ccHObject::Container m_selectedEntities;
};
