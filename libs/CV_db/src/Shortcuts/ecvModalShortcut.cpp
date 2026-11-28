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
    // Note: Default context is WindowShortcut, but we'll set it properly
    // in setContextWidget() if needed
    m_shortcut = new QShortcut(key, parent);

    QObject::connect(m_shortcut, &QShortcut::activated, this,
                     [this]() { Q_EMIT activated(); });

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
    if (m_shortcut && m_shortcut->parent() == contextWidget &&
        m_shortcut->context() == contextArea)
        return;
    const bool wasEnabled = isEnabled();
    // QObject ownership remains with the original owner; only the QShortcut's
    // dispatch context changes. Recreate consistently for Qt 5 and Qt 6.
    delete m_shortcut;
    if (contextWidget) {
        m_shortcut = new QShortcut(m_key, contextWidget);
        m_shortcut->setContext(contextArea);
        m_shortcut->setEnabled(wasEnabled);
        connect(m_shortcut, &QShortcut::activated, this,
                &ecvModalShortcut::activated);
        if (m_action)
            connect(m_shortcut, &QShortcut::activated, m_action,
                    &QAction::trigger);
    }
    if (wasEnabled && !m_shortcut) Q_EMIT disabled();
    Q_EMIT ownershipChanged(false);
}

//-----------------------------------------------------------------------------
bool ecvModalShortcut::isEnabled() const {
    return m_shortcut ? m_shortcut->isEnabled() : false;
}

void ecvModalShortcut::setSuspended(bool suspended) {
    const bool wasEnabled = isEnabled();
    m_suspended = suspended;
    const bool shouldEnable = m_requestedEnabled && !m_suspended;
    if (m_shortcut && wasEnabled != shouldEnable) {
        m_shortcut->setEnabled(shouldEnable);
        if (shouldEnable)
            Q_EMIT enabled();
        else
            Q_EMIT disabled();
    }
}

//-----------------------------------------------------------------------------
void ecvModalShortcut::setEnabled(bool enable, bool changeFocus) {
    m_requestedEnabled = enable;
    // Explicit enable reclaims priority, even while temporarily suspended.
    Q_EMIT ownershipChanged(enable);
    // Also handle initial setup before the registry connects ownershipChanged.
    setSuspended(m_suspended);
    if (isEnabled() && changeFocus) {
        const auto ctxt = m_shortcut->context();
        if (ctxt == Qt::WidgetShortcut ||
            ctxt == Qt::WidgetWithChildrenShortcut) {
            auto* parent = qobject_cast<QWidget*>(m_shortcut->parent());
            if (parent) parent->setFocus(Qt::OtherFocusReason);
        }
    }
}

//-----------------------------------------------------------------------------
QKeySequence ecvModalShortcut::keySequence() const { return m_key; }
