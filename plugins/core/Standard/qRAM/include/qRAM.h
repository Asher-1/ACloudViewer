// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
#include <ecvStdPluginInterface.h>

#include <QPointer>
class QAction;
class RAMDialog;
class qRAM : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "cvcorp.cloudviewer.plugin.qRAM" FILE "../info.json")
public:
    explicit qRAM(QObject* parent = nullptr);
    ~qRAM() override;
    QList<QAction*> getActions() override;
    void onNewSelection(const ccHObject::Container&) override;

private:
    void showDialog();
    void refreshSelection();
    QAction* m_action = nullptr;
    QPointer<RAMDialog> m_dialog;
    ccHObject::Container m_selection;
};
