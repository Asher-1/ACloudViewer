// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ecvStdPluginInterface.h>

#include <QAction>
#include <QObject>

class SAM3Dialog;

class qSAM3 : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "cvcorp.cloudviewer.plugin.qSAM3" FILE "../info.json")

public:
    explicit qSAM3(QObject* parent = nullptr);
    ~qSAM3() override = default;

    void onNewSelection(const ccHObject::Container& selectedEntities) override;
    QList<QAction*> getActions() override;

private:
    void showDialog();
    void setCurrentDialogSelection();

    QAction* m_action = nullptr;
    SAM3Dialog* m_dialog = nullptr;
};