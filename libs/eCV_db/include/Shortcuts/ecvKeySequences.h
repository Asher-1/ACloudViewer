// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ecvKeySequences_h
#define ecvKeySequences_h

// Local
#include "eCV_db.h"

// Qt
#include <QObject>
#include <QWidget>

class QAction;
class ecvModalShortcut;

/**
 * @brief Manage key sequences used for shortcuts.
 *
 * ParaView-style modal shortcut manager that prevents ambiguous activation
 * by ensuring only one listener is active for each key sequence at a time.
 *
 * Key features:
 * - Automatic mutual exclusion: when one shortcut is enabled, all siblings are
 * disabled
 * - Visual feedback: widgets can show which one currently owns a shortcut
 * - Context awareness: shortcuts can be limited to specific widgets
 *
 * Usage:
 * @code
 * auto* shortcut = ecvKeySequences::instance().addModalShortcut(
 *     QKeySequence(Qt::Key_S), myAction, myWidget);
 * @endcode
 */
class ECV_DB_LIB_API ecvKeySequences : public QObject {
    Q_OBJECT

public:
    /**
     * Get the singleton instance.
     */
    static ecvKeySequences& instance();

    /**
     * Return the active shortcut for a given key sequence (if any).
     */
    ecvModalShortcut* active(const QKeySequence& keySequence) const;

    /**
     * Register a modal shortcut with the manager.
     *
     * @param keySequence The key combination to listen for
     * @param action The action to trigger (optional)
     * @param parent The context widget (optional)
     * @return The created modal shortcut (owned by the manager)
     */
    ecvModalShortcut* addModalShortcut(const QKeySequence& keySequence,
                                       QAction* action,
                                       QWidget* parent);

    /**
     * Ask the manager to reorder shortcuts so that the currently-active
     * one becomes the "next" in line to the passed target.
     *
     * ** NB: This method is currently a placeholder. **
     *
     * Widgets should call this method before invoking
     * ecvModalShortcut::setEnabled in response to user input.
     *
     * This method has no effect if no sibling of target is active
     * at the time it is invoked.
     */
    void reorder(ecvModalShortcut* target);

    /**
     * Dump a list of shortcuts registered for a given key sequence (debug).
     */
    void dumpShortcuts(const QKeySequence& keySequence) const;

protected Q_SLOTS:
    /**
     * Called when a shortcut is enabled to ensure siblings are disabled.
     */
    virtual void disableSiblings();

    /**
     * Not currently used. Intended for use enabling next-most-recently-used
     * shortcut.
     */
    virtual void enableNextSibling();

    /**
     * Called when shortcuts are deleted to disable and unregister them.
     */
    virtual void removeModalShortcut();

protected:
    ecvKeySequences(QObject* parent);
    ~ecvKeySequences() override = default;

    /// Set true in slot implementations to avoid signal/slot recursion.
    bool m_silence;
};

#endif  // ecvKeySequences_h
