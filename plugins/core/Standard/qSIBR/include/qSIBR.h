// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QMap>

#include "ecvStdPluginInterface.h"

class QMenu;
class SIBRViewerThread;
struct SIBRViewerOptions;

class qSIBR : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "ecvcorp.cloudviewer.plugin.qSIBR" FILE
                          "../info.json")

public:
    explicit qSIBR(QObject* parent = nullptr);
    virtual ~qSIBR();

    void onNewSelection(const ccHObject::Container& selectedEntities) override;
    QList<QAction*> getActions() override;

protected slots:
    void launchULRViewer();
    void launchULRv2Viewer();
    void launchTexturedMeshViewer();
    void launchPointBasedViewer();
    void launchGaussianViewer();
    void launchRemoteGaussianViewer();

    // Dataset tools
    void launchPrepareColmap4Sibr();
    void launchTonemapper();
    void launchUnwrapMesh();
    void launchTextureMesh();
    void launchClippingPlanes();
    void launchCropFromCenter();
    void launchNvmToSIBR();
    void launchDistordCrop();
    void launchCameraConverter();
    void launchAlignMeshes();

    // Context-sensitive quick action
    void launchQuickView();

private:
    void launchViewer(SIBRViewerThread* thread);
    void launchSimpleViewer(int viewerMode,
                            int dialogType,
                            const QString& prefix);
    void launchDatasetTool(const QString& toolName);
    void launchDatasetToolWithArgs(const QString& toolName,
                                   const QStringList& args);

    void onViewerStarted(const QString& modeName);
    void onViewerFinished(const QString& modeName, int exitCode);
    void onViewerError(const QString& error);
    void onViewerLog(const QString& message);
    void onViewerResultReady(const QString& resultPath,
                             const QString& description);

    QString detectEntitySourcePath(ccHObject* entity) const;
    static bool looksLikeGaussianModelDir(const QString& dirPath);

    QAction* m_actionULR = nullptr;
    QAction* m_actionULRv2 = nullptr;
    QAction* m_actionTexturedMesh = nullptr;
    QAction* m_actionPointBased = nullptr;
    QAction* m_actionGaussian = nullptr;
    QAction* m_actionRemoteGaussian = nullptr;
    QAction* m_actionDatasetTools = nullptr;
    QMenu* m_datasetToolsMenu = nullptr;

    QMap<QString, SIBRViewerThread*> m_activeViewers;
    SIBRViewerThread* m_activeToolThread = nullptr;

    QAction* m_actionQuickView = nullptr;
    ccHObject::Container m_selectedEntities;
};
