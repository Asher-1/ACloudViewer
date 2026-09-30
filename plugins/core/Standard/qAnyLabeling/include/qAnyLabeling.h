// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ecvStdPluginInterface.h>

#include <QAction>
#include <QPointer>
#include <QTranslator>

class AnnotationWindow;

class qAnyLabeling final : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "cvcorp.cloudviewer.plugin.qAnyLabeling" FILE
                          "../info.json")

public:
    explicit qAnyLabeling(QObject* parent = nullptr);
    QList<QAction*> getActions() override;
    void setMainAppInterface(ecvMainAppInterface* app) override;

private:
    QAction* m_action = nullptr;
    QPointer<AnnotationWindow> m_window;
    // Source language-menu parity: the persisted translation catalog stays
    // installed for the process lifetime.
    QTranslator m_translator;
};
