// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q_MPLANE_PLUGIN_HEADER
#define Q_MPLANE_PLUGIN_HEADER

// std
#include <memory>

// ACloudViewer
#include <ecvPickingListener.h>
#include <ecvPointCloud.h>
#include <ecvStdPluginInterface.h>

// qMPlane
#include "src/ccMPlaneDlgController.h"

#ifdef USE_VLD
// VLD
#include <vld.h>
#endif

class qMPlane : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "ecvcorp.cloudviewer.plugin.MPlane" FILE "info.json")

public:
    explicit qMPlane(QObject *parent = nullptr);
    ~qMPlane() override = default;

    void onNewSelection(const ccHObject::Container &selectedEntities) override;
    QList<QAction *> getActions() override;

protected slots:
    void doAction();

private:
    QAction *m_action = nullptr;
    ccPointCloud *m_selectedCloud = nullptr;
    std::unique_ptr<ccMPlaneDlgController> m_controller;
};
#endif
