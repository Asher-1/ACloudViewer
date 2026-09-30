// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qAnyLabeling.h"

#include <QCoreApplication>
#include <QIcon>
#include <QSettings>

#include "AnnotationWindow.h"

qAnyLabeling::qAnyLabeling(QObject* parent)
    : QObject(parent),
      ccStdPluginInterface(":/CC/plugin/qAnyLabeling/info.json") {
    // Source language-menu parity: load the persisted translation catalog
    // before any workspace widget is created. Untranslated strings fall
    // back to the English source text.
    const QString language =
            QSettings()
                    .value(QStringLiteral("qAnyLabeling/language"),
                           QStringLiteral("en_US"))
                    .toString();
    if (language != QStringLiteral("en_US") &&
        m_translator.load(
                QStringLiteral(":/qAnyLabeling/translations/qanylabeling_%1.qm")
                        .arg(language))) {
        QCoreApplication::installTranslator(&m_translator);
    }
    m_action = new QAction(tr("X-AnyLabeling Annotation"), this);
    m_action->setIcon(QIcon(":/CC/plugin/qAnyLabeling/images/icon.png"));
    m_action->setToolTip(tr("Open the X-AnyLabeling annotation workspace"));
    connect(m_action, &QAction::triggered, this, [this]() {
        if (!m_window) {
            m_window = new AnnotationWindow(m_app ? m_app->getMainWindow()
                                                  : nullptr);
            m_window->setAttribute(Qt::WA_DeleteOnClose);
            m_window->setAppInterface(m_app);
        }
        m_window->show();
        m_window->raise();
        m_window->activateWindow();
    });
}

void qAnyLabeling::setMainAppInterface(ecvMainAppInterface* app) {
    ccStdPluginInterface::setMainAppInterface(app);
    if (m_window) m_window->setAppInterface(m_app);
}

QList<QAction*> qAnyLabeling::getActions() { return {m_action}; }
