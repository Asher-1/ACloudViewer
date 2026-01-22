// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ecvModalShortcut_h
#define ecvModalShortcut_h

// Local
#include "CV_db.h"

// Qt
#include <QKeySequence>
#include <QObject>
#include <QPointer>

class QAction;
class QShortcut;
class QWidget;

/**
 * @brief Manage an action and/or widget's responsivity to a shortcut.
 *
 * This object will add and remove a connection between
 * a widget/action and a QShortcut as required by the ecvKeySequences
 * manager to prevent any ambiguous activations.
 *
 * Key features:
 * - Automatic enable/disable based on ecvKeySequences manager
 * - Context-aware: can be restricted to a specific widget
 * - Signal emission for visual feedback
 *
 * Example:
 * @code
 * // This is typically created by ecvKeySequences::addModalShortcut()
 * auto* modalShortcut = new ecvModalShortcut(
 *     QKeySequence(Qt::Key_S), myAction, myWidget);
 * modalShortcut->setEnabled(true);
 * @endcode
 */
class CV_DB_LIB_API ecvModalShortcut : public QObject {
    Q_OBJECT

public:
    using Superclass = QObject;
    ~ecvModalShortcut() override;

    /**
     * If the shortcut should be restricted to a particular widget
     * (such as a view), use this method to set and update the widget
     * during the life of the ecvModalShortcut.
     *
     * @param contextWidget The widget to restrict the shortcut to
     * @param contextArea The context scope (default: WindowShortcut)
     */
    void setContextWidget(QWidget* contextWidget,
                          Qt::ShortcutContext contextArea = Qt::WindowShortcut);

    /**
     * Check if the shortcut is currently enabled.
     */
    bool isEnabled() const;

    /**
     * Enable or disable the shortcut.
     *
     * @param shouldEnable True to enable, false to disable
     * @param changeFocus If true and enabling, move focus to context widget
     */
    void setEnabled(bool shouldEnable, bool changeFocus = true);

    /**
     * Get the key sequence this shortcut responds to.
     */
    QKeySequence keySequence() const;

Q_SIGNALS:
    /**
     * Called from setEnabled() whenever it is passed true.
     *
     * This is used by ecvKeySequences to disable any sibling shortcuts
     * with the same keysequence.
     *
     * This may also be used by widgets to update their visual state, indicating
     * they are now accepting shortcuts.
     */
    void enabled();

    /**
     * Called from setEnabled() whenever it is passed false.
     *
     * This may be used by widgets to update their visual state, indicating
     * they are no longer accepting shortcuts.
     */
    void disabled();

    /**
     * Called from the destructor.
     *
     * This is used by ecvKeySequences to clean its records.
     */
    void unregister();

    /**
     * Invoked when the key sequence is pressed.
     */
    void activated();

protected:
    friend class ecvKeySequences;

    /**
     * Constructor (protected - use ecvKeySequences::addModalShortcut()).
     */
    ecvModalShortcut(const QKeySequence& key,
                     QAction* action = nullptr,
                     QWidget* parent = nullptr);

    QKeySequence m_key;
    QPointer<QShortcut> m_shortcut;
    QPointer<QAction> m_action;
};

#endif  // ecvModalShortcut_h
