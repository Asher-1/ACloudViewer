// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "Shortcuts/ecvKeySequences.h"

#include "Shortcuts/ecvModalShortcut.h"

// app
#include <CVLog.h>

#include <QAction>
#include <QKeySequence>
#include <QMap>
#include <QPointer>
#include <QSet>

namespace {

/**
 * @brief Internal structure to hold all shortcuts registered for a key
 * sequence.
 */
struct Shortcuts {
    /// All shortcuts (siblings) that share the same key sequence
    QSet<ecvModalShortcut*> Siblings;

    /// Most recently used shortcut (for reordering)
    ecvModalShortcut* mruShortcut = nullptr;

    // TODO: Allow a key-sequence to be "default to last" when a shortcut is
    //       removed/disabled as an option?
    // TODO: Hold tooltip or preference name?
};

/**
 * @brief Global dictionary mapping key sequences to their shortcuts.
 */
struct Dictionary {
    QMap<QKeySequence, Shortcuts> Data;
};

/// Global instance
Dictionary g_keys;

}  // anonymous namespace

//-----------------------------------------------------------------------------
ecvKeySequences& ecvKeySequences::instance() {
    static ecvKeySequences inst(nullptr);
    return inst;
}

//-----------------------------------------------------------------------------
ecvKeySequences::ecvKeySequences(QObject* parent)
    : QObject(parent), m_silence(false) {
    CVLog::PrintVerbose("[ecvKeySequences] Modal shortcut manager initialized");
}

//-----------------------------------------------------------------------------
ecvModalShortcut* ecvKeySequences::active(
        const QKeySequence& keySequence) const {
    ecvModalShortcut* active = nullptr;
    QMap<QKeySequence, Shortcuts>::iterator iter =
            g_keys.Data.find(keySequence);
    if (iter == g_keys.Data.end()) {
        return active;
    }

    for (auto& sibling : iter->Siblings) {
        if (sibling && sibling->isEnabled()) {
            active = sibling;
            break;
        }
    }
    return active;
}

//-----------------------------------------------------------------------------
ecvModalShortcut* ecvKeySequences::addModalShortcut(
        const QKeySequence& keySequence, QAction* action, QWidget* parent) {
    if (keySequence.isEmpty()) {
        CVLog::Warning(
                "[ecvKeySequences] Attempted to register empty key sequence");
        return nullptr;
    }

    // Create the modal shortcut
    auto shortcut = new ecvModalShortcut(keySequence, action, parent);

    // Add to siblings list
    auto& keyData = g_keys.Data[keySequence];
    keyData.Siblings.insert(shortcut);

    // Initially disable (will be enabled below)
    shortcut->setEnabled(false);

    // Connect signals
    QObject::connect(shortcut, &ecvModalShortcut::enabled, this,
                     &ecvKeySequences::disableSiblings);
    // QObject::connect(shortcut, &ecvModalShortcut::disabled,
    //                  this, &ecvKeySequences::enableNextSibling);
    QObject::connect(shortcut, &ecvModalShortcut::unregister, this,
                     &ecvKeySequences::removeModalShortcut);

    // Enable (this will trigger disableSiblings() to ensure only one is active)
    shortcut->setEnabled(true);

    CVLog::PrintVerbose(
            QString("[ecvKeySequences] Registered modal shortcut: %1 (ID: %2)")
                    .arg(keySequence.toString())
                    .arg(shortcut->objectName().isEmpty()
                                 ? "unnamed"
                                 : shortcut->objectName()));

    return shortcut;
}

//-----------------------------------------------------------------------------
void ecvKeySequences::reorder(ecvModalShortcut* target) {
    if (!target) {
        return;
    }

    // Find the key sequence for this shortcut
    QMap<QKeySequence, Shortcuts>::iterator iter =
            g_keys.Data.find(target->keySequence());
    if (iter == g_keys.Data.end()) {
        return;
    }

    // Check if any sibling of target is currently active
    bool hasActiveSibling = false;
    for (auto& sibling : iter->Siblings) {
        if (sibling != target && sibling && sibling->isEnabled()) {
            hasActiveSibling = true;
            break;
        }
    }

    // If no sibling is active, reorder has no effect (as per ParaView
    // documentation)
    if (!hasActiveSibling) {
        return;
    }

    // Mark target as the most recently used shortcut
    // This allows enableNextSibling() to activate it when the current
    // shortcut is disabled
    iter->mruShortcut = target;

    CVLog::PrintVerbose(
            QString("[ecvKeySequences] Reordered shortcut: %1 (MRU set)")
                    .arg(target->keySequence().toString()));
}

//-----------------------------------------------------------------------------
void ecvKeySequences::dumpShortcuts(const QKeySequence& keySequence) const {
    QMap<QKeySequence, Shortcuts>::iterator iter =
            g_keys.Data.find(keySequence);
    if (iter == g_keys.Data.end()) {
        CVLog::Print(
                QString("[ecvKeySequences] No shortcuts registered for: %1")
                        .arg(keySequence.toString()));
        return;
    }

    CVLog::PrintVerbose(QString("[ecvKeySequences] Shortcuts for %1:")
                                .arg(keySequence.toString()));
    for (auto& sibling : iter->Siblings) {
        if (sibling) {
            CVLog::PrintVerbose(QString("  - %1: %2")
                                        .arg(sibling->objectName().isEmpty()
                                                     ? "unnamed"
                                                     : sibling->objectName())
                                        .arg(sibling->isEnabled()
                                                     ? "enabled"
                                                     : "disabled"));
        }
    }
}

//-----------------------------------------------------------------------------
void ecvKeySequences::disableSiblings() {
    if (m_silence) {
        return;
    }

    auto* shortcut = qobject_cast<ecvModalShortcut*>(this->sender());
    if (!shortcut) {
        return;
    }

    QMap<QKeySequence, Shortcuts>::iterator iter =
            g_keys.Data.find(shortcut->keySequence());
    if (iter == g_keys.Data.end()) {
        return;
    }

    // Update MRU shortcut to the one being enabled
    iter->mruShortcut = shortcut;

    // Update MRU shortcut to the one being enabled
    iter->mruShortcut = shortcut;

    // Disable all siblings except the one being enabled
    m_silence = true;
    int disabledCount = 0;
    for (auto& sibling : iter->Siblings) {
        if (sibling != shortcut && sibling->isEnabled()) {
            sibling->setEnabled(false);
            disabledCount++;
        }
    }
    m_silence = false;

    if (disabledCount > 0) {
        CVLog::PrintVerbose(
                QString("[ecvKeySequences] Disabled %1 sibling(s) for %2")
                        .arg(disabledCount)
                        .arg(shortcut->keySequence().toString()));
    }
}

//-----------------------------------------------------------------------------
void ecvKeySequences::enableNextSibling() {
    if (m_silence) {
        return;
    }

    auto* shortcut = qobject_cast<ecvModalShortcut*>(this->sender());
    if (!shortcut) {
        return;
    }

    // Find the key sequence for this shortcut
    QMap<QKeySequence, Shortcuts>::iterator iter =
            g_keys.Data.find(shortcut->keySequence());
    if (iter == g_keys.Data.end()) {
        return;
    }

    // Activate the most-recently-used sibling if available
    // This implements the reordering behavior: when a shortcut is disabled,
    // the MRU shortcut (set via reorder()) becomes the next one to activate
    if (iter->mruShortcut && iter->mruShortcut != shortcut) {
        // Check if the MRU shortcut's parent widget is still valid and enabled
        QWidget* parentWidget =
                qobject_cast<QWidget*>(iter->mruShortcut->parent());
        if (parentWidget && parentWidget->isEnabled() &&
            parentWidget->isVisible()) {
            iter->mruShortcut->setEnabled(true);
            CVLog::PrintVerbose(
                    QString("[ecvKeySequences] Enabled next sibling (MRU): %1")
                            .arg(shortcut->keySequence().toString()));
        } else {
            // MRU shortcut's widget is disabled/hidden, try to find another
            // enabled sibling
            for (auto& sibling : iter->Siblings) {
                if (sibling != shortcut && sibling) {
                    QWidget* siblingParent =
                            qobject_cast<QWidget*>(sibling->parent());
                    if (siblingParent && siblingParent->isEnabled() &&
                        siblingParent->isVisible()) {
                        sibling->setEnabled(true);
                        CVLog::PrintVerbose(QString("[ecvKeySequences] Enabled "
                                                    "next sibling: "
                                                    "%1")
                                                    .arg(shortcut->keySequence()
                                                                 .toString()));
                        break;
                    }
                }
            }
        }
    } else {
        // No MRU shortcut set, try to find any enabled sibling
        for (auto& sibling : iter->Siblings) {
            if (sibling != shortcut && sibling) {
                QWidget* siblingParent =
                        qobject_cast<QWidget*>(sibling->parent());
                if (siblingParent && siblingParent->isEnabled() &&
                    siblingParent->isVisible()) {
                    sibling->setEnabled(true);
                    CVLog::PrintVerbose(
                            QString("[ecvKeySequences] Enabled next sibling: "
                                    "%1")
                                    .arg(shortcut->keySequence().toString()));
                    break;
                }
            }
        }
    }
}

//-----------------------------------------------------------------------------
void ecvKeySequences::removeModalShortcut() {
    auto* shortcut = qobject_cast<ecvModalShortcut*>(this->sender());
    if (!shortcut) {
        return;
    }

    QMap<QKeySequence, Shortcuts>::iterator iter =
            g_keys.Data.find(shortcut->keySequence());
    if (iter == g_keys.Data.end()) {
        return;
    }

    // Remove the shortcut from the key sequence list
    iter->Siblings.remove(shortcut);
    QObject::disconnect(shortcut);

    CVLog::PrintVerbose(
            QString("[ecvKeySequences] Unregistered modal shortcut: %1")
                    .arg(shortcut->keySequence().toString()));

    // If no more siblings, remove the key sequence entry
    if (iter->Siblings.empty()) {
        g_keys.Data.erase(iter);
    }
}
