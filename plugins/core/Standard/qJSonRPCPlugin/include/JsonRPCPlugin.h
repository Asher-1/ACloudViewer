// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "ecvStdPluginInterface.h"
// #include <jcon/json_rpc_tcp_server.h>
#include "jsonrpcserver.h"

//! Example qCC plugin
/** Replace 'ExamplePlugin' by your own plugin class name throughout and then
        check 'ExamplePlugin.cpp' for more directions.

        Each plugin requires an info.json file to provide information about
itself - the name, authors, maintainers, icon, etc..

        The one method you are required to implement is 'getActions'. This
should return all actions (QAction objects) for the plugin. ACloudViewer will
        automatically add these with their icons in the plugin toolbar and to
the plugin menu. If	your plugin returns	several actions, CC will create
a dedicated toolbar and a	sub-menu for your plugin. You are responsible
for connecting these actions to	methods in your plugin.

        Use the ccStdPluginInterface::m_app variable for access to most of the
CC components (database, 3D views, console, etc.) - see the ecvMainAppInterface
        class in ecvMainAppInterface.h.
**/
class JsonRPCPlugin : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)

    // Replace "Example" by your plugin name (IID should be unique - let's hope
    // your plugin name is unique ;) The info.json file provides information
    // about the plugin to the loading system and it is displayed in the plugin
    // information dialog.
    Q_PLUGIN_METADATA(IID "ecvcorp.cloudviewer.plugin.JsonRPC" FILE
                          "../info.json")

public:
    explicit JsonRPCPlugin(QObject *parent = nullptr);
    ~JsonRPCPlugin() override = default;

    QList<QAction *> getActions() override;

public slots:
    void triggered(bool checked);
    JsonRPCResult execute(QString method, QMap<QString, QVariant> params);

private:
    // File I/O
    JsonRPCResult rpcOpen(const QMap<QString, QVariant> &params);
    JsonRPCResult rpcExport(const QMap<QString, QVariant> &params);
    JsonRPCResult rpcFileConvert(const QMap<QString, QVariant> &params);
    // Scene tree
    JsonRPCResult rpcSceneList(const QMap<QString, QVariant> &params);
    JsonRPCResult rpcSceneInfo(const QMap<QString, QVariant> &params);
    JsonRPCResult rpcSceneRemove(const QMap<QString, QVariant> &params);
    JsonRPCResult rpcSceneSetVisible(const QMap<QString, QVariant> &params);
    JsonRPCResult rpcSceneSelect(const QMap<QString, QVariant> &params);
    JsonRPCResult rpcClear(const QMap<QString, QVariant> &params);
    // View control
    JsonRPCResult rpcViewSetOrientation(const QMap<QString, QVariant> &params);
    JsonRPCResult rpcViewZoomFit(const QMap<QString, QVariant> &params);
    JsonRPCResult rpcViewRefresh(const QMap<QString, QVariant> &params);
    JsonRPCResult rpcViewSetPerspective(const QMap<QString, QVariant> &params);
    JsonRPCResult rpcViewSetPointSize(const QMap<QString, QVariant> &params);
    // Transform
    JsonRPCResult rpcTransformApply(const QMap<QString, QVariant> &params);
    // Entity properties
    JsonRPCResult rpcEntityRename(const QMap<QString, QVariant> &params);
    JsonRPCResult rpcEntitySetColor(const QMap<QString, QVariant> &params);
    // Mesh processing
    JsonRPCResult rpcMeshSimplify(const QMap<QString, QVariant> &params);
    JsonRPCResult rpcMeshSmooth(const QMap<QString, QVariant> &params);
    JsonRPCResult rpcMeshSubdivide(const QMap<QString, QVariant> &params);
    JsonRPCResult rpcMeshSamplePoints(const QMap<QString, QVariant> &params);
    // Cloud colorization
    JsonRPCResult rpcCloudPaintUniform(const QMap<QString, QVariant> &params);
    JsonRPCResult rpcCloudPaintByHeight(const QMap<QString, QVariant> &params);
    JsonRPCResult rpcCloudPaintByScalarField(
            const QMap<QString, QVariant> &params);
    // Cloud processing
    JsonRPCResult rpcCloudComputeNormals(const QMap<QString, QVariant> &params);
    JsonRPCResult rpcCloudSubsample(const QMap<QString, QVariant> &params);
    JsonRPCResult rpcCloudCrop(const QMap<QString, QVariant> &params);
    JsonRPCResult rpcCloudGetScalarFields(
            const QMap<QString, QVariant> &params);
    // View capture
    JsonRPCResult rpcViewScreenshot(const QMap<QString, QVariant> &params);
    JsonRPCResult rpcViewGetCamera(const QMap<QString, QVariant> &params);
    // Reconstruction (Colmap)
    JsonRPCResult rpcColmapReconstruct(const QMap<QString, QVariant> &params);
    // Introspection
    JsonRPCResult rpcMethodsList(const QMap<QString, QVariant> &params);

    void redraw();

    QAction *m_action{nullptr};
    JsonRPCServer rpc_server;
};
