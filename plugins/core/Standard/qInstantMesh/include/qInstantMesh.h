// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ecvStdPluginInterface.h>

#include <QAction>
#include <QByteArray>
#include <QPointer>
#include <QTimer>
#include <QVector>

#include "InstantMeshDialog.h"
#include "InstantMeshWorker.h"

class ccCommandLineInterface;

//! InstantMesh image-to-3D plugin (Zero123++ multi-view diffusion +
//! FlexiCubes geometry, native GGML inference via AICore).
class qInstantMesh : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "cvcorp.cloudviewer.plugin.qInstantMesh" FILE
                          "../info.json")

public:
    explicit qInstantMesh(QObject* parent = nullptr);
    ~qInstantMesh() override;

    // ------ ccStdPluginInterface ------
    void onNewSelection(const ccHObject::Container& selectedEntities) override;
    QList<QAction*> getActions() override;
    void registerCommands(ccCommandLineInterface* cmd) override;

protected slots:
    void onNewAction();
    void onRunRequested(const InstantMeshDialog::Settings& settings);
    void onCancelRequested();
    void onResultReady(const InstantMeshRunResult& result);
    void onTaskFinished(bool success);

private:
    void startWorker(const InstantMeshDialog::Settings& settings);
    void addResultToDb(const InstantMeshRunResult& result,
                       const InstantMeshDialog::Settings& settings);
    ccMesh* buildResultMesh(const InstantMeshRunResult& result,
                            const QString& name);
    ccMesh* buildVertexColorMesh(const QVector<float>& verts,
                                 const QVector<float>& vcolors,
                                 const QVector<int>& tris,
                                 const QString& name);

    QAction* m_action = nullptr;
    QPointer<InstantMeshDialog> m_dialog;
    InstantMeshWorker* m_worker = nullptr;
    uint64_t m_generation = 0;
    QString m_source;
    bool m_acceptResults = false;
    bool m_taskSucceeded = false;
};
