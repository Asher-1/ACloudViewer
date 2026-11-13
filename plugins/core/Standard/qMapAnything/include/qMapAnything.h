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

#include "MapAnythingDialog.h"
#include "MapAnythingWorker.h"

class ccImage;
class ccCommandLineInterface;

class qMapAnything : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "cvcorp.cloudviewer.plugin.qMapAnything" FILE
                          "../info.json")

public:
    explicit qMapAnything(QObject* parent = nullptr);

    void onNewSelection(const ccHObject::Container& selectedEntities) override;
    QList<QAction*> getActions() override;
    /** Headless CLI: registers -MAP_ANYTHING_RECONSTRUCT (multi-view
     *  reconstruction, merged world-frame PLY + RESULT_JSON). */
    void registerCommands(ccCommandLineInterface* cmd) override;

private slots:
    void showDialog();
    void executeTask(const MapAnythingDialog::Settings& settings);
    void cancelTask();
    void onResultReady(const MapAnythingResult& result);
    void onTaskFinished(bool success);
    void onWorkerProgress(int current, int total);

private:
    void addResultToDb(const MapAnythingResult& result);
    /** Turns db:// entries of the shared input list into staged PNG files
     *  (one export per DB image) and returns the resolved run paths. */
    bool resolveInputPaths(MapAnythingDialog::Settings& settings);
    void clearStagedInputFiles();
    ccImage* findDbImage(const QString& name) const;
    QStringList selectedDbImageNames() const;
    void refreshDbImages();

    QAction* m_action = nullptr;
    MapAnythingDialog* m_dialog = nullptr;
    MapAnythingWorker* m_worker = nullptr;
    MapAnythingDialog::Settings m_currentSettings;
    QStringList m_stagedInputFiles;
    ccHObject::Container m_selectedEntities;
};
