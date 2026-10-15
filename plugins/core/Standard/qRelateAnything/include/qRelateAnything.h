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
class RelateAnythingDialog;
class qRelateAnything : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "cvcorp.cloudviewer.plugin.qRelateAnything" FILE
                          "../info.json")
public:
    explicit qRelateAnything(QObject* parent = nullptr);
    ~qRelateAnything() override;
    QList<QAction*> getActions() override;
    void onNewSelection(const ccHObject::Container&) override;

private:
    void showDialog();
    void refreshSelection();
    QAction* m_action = nullptr;
    QPointer<RelateAnythingDialog> m_dialog;
    ccHObject::Container m_selection;
};
