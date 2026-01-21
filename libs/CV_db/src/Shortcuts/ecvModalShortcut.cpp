// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "Shortcuts/ecvModalShortcut.h"

// app
#include <CVLog.h>

#include <QAction>
#include <QShortcut>
#include <QWidget>

//-----------------------------------------------------------------------------
ecvModalShortcut::ecvModalShortcut(const QKeySequence& key,
                                   QAction* action,
                                   QWidget* parent)
    : Superclass(parent), m_key(key), m_action(action) {
    // Create the underlying QShortcut
    m_shortcut = new QShortcut(key, parent);

    // Connect shortcut activation to our signal
    QObject::connect(m_shortcut, &QShortcut::activated, this,
                     &ecvModalShortcut::activated);

    // If an action is provided, trigger it when activated
    if (m_action) {
        QObject::connect(m_shortcut, &QShortcut::activated, m_action,
                         &QAction::trigger);
    }
}

//-----------------------------------------------------------------------------
ecvModalShortcut::~ecvModalShortcut() {
    Q_EMIT unregister();
    delete m_shortcut;
}

//-----------------------------------------------------------------------------
void ecvModalShortcut::setContextWidget(QWidget* contextWidget,
                                        Qt::ShortcutContext contextArea) {
    bool enabled = this->isEnabled();

    if (m_shortcut) {
        // Check if we're already using this widget and context
        if (qobject_cast<QWidget*>(m_shortcut->parent()) == contextWidget) {
            if (m_shortcut->context() != contextArea) {
                m_shortcut->setContext(contextArea);
            }
            return;
        }
    }

    // To change parents, it's best to start over
    delete m_shortcut;

    if (!contextWidget && contextArea != Qt::ApplicationShortcut) {
        // We need to keep a shortcut around, but don't pay attention
        // to it since the context widget is null.
        m_shortcut = nullptr;
    } else {
        // Create new shortcut with the new parent
        m_shortcut = new QShortcut(m_key, contextWidget);
        m_shortcut->setEnabled(enabled);
        m_shortcut->setContext(contextArea);

        // Reconnect signals
        QObject::connect(m_shortcut, &QShortcut::activated, this,
                         &ecvModalShortcut::activated);
        if (m_action) {
            QObject::connect(m_shortcut, &QShortcut::activated, m_action,
                             &QAction::trigger);
        }
    }
}

//-----------------------------------------------------------------------------
bool ecvModalShortcut::isEnabled() const {
    return m_shortcut ? m_shortcut->isEnabled() : false;
}

//-----------------------------------------------------------------------------
void ecvModalShortcut::setEnabled(bool enable, bool changeFocus) {
    if (!m_shortcut) {
        // Shortcut was destroyed (e.g., parent widget deleted)
        return;
    }

    if (enable) {
        if (!m_shortcut->isEnabled()) {
            // Emit signal first (so ecvKeySequences can disable siblings)
            Q_EMIT enabled();
            m_shortcut->setEnabled(true);

            // If requested, give focus to the context widget
            // so users can immediately use the shortcut
            auto ctxt = m_shortcut->context();
            if ((ctxt == Qt::WidgetShortcut ||
                 ctxt == Qt::WidgetWithChildrenShortcut) &&
                changeFocus) {
                auto* parent = dynamic_cast<QWidget*>(m_shortcut->parent());
                if (parent) {
                    parent->setFocus(Qt::OtherFocusReason);
                }
            }
        }
    } else {
        if (m_shortcut->isEnabled()) {
            m_shortcut->setEnabled(false);
            Q_EMIT disabled();
        }
    }
}

//-----------------------------------------------------------------------------
QKeySequence ecvModalShortcut::keySequence() const { return m_key; }
