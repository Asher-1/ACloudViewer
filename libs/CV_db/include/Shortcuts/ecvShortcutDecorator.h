// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ecvShortcutDecorator_h
#define ecvShortcutDecorator_h

// Local
#include "CV_db.h"

// Qt
#include <QColor>
#include <QFrame>
#include <QList>
#include <QObject>
#include <QPointer>

class ecvModalShortcut;

/**
 * @brief Decorate a widget by highlighting its frame when keyboard shortcuts
 * are active.
 *
 * This class provides visual feedback for widgets that have modal shortcuts
 * attached. Features:
 * - Highlights the widget's border when shortcuts are active
 * - Allows users to click on the border to activate/deactivate shortcuts
 * - Monitors mouse enter/exit events for visual feedback
 * - Manages multiple shortcuts as a group (all enabled/disabled together)
 *
 * Usage:
 * @code
 * auto* decorator = new ecvShortcutDecorator(myWidget);
 * auto* shortcut = ecvKeySequences::instance().addModalShortcut(...);
 * decorator->addShortcut(shortcut);
 * @endcode
 */
class CV_DB_LIB_API ecvShortcutDecorator : public QObject {
    Q_OBJECT

public:
    using Superclass = QObject;

    /**
     * Create a decorator for the given widget.
     *
     * @param parent The widget to decorate (must be a QFrame or subclass)
     */
    explicit ecvShortcutDecorator(QFrame* parent);

    /**
     * Destructor.
     */
    ~ecvShortcutDecorator() override = default;

    /**
     * Add a shortcut to this decorator.
     * All shortcuts attached to a decorator are enabled/disabled as a group.
     *
     * @param shortcut The modal shortcut to attach
     */
    void addShortcut(ecvModalShortcut* shortcut);

    /**
     * Check if any attached shortcuts are enabled.
     *
     * @return True if at least one shortcut is enabled
     */
    bool isEnabled() const;

public Q_SLOTS:
    /**
     * Enable or disable all attached shortcuts.
     *
     * @param enable True to enable, false to disable
     * @param refocusWhenEnabling If true, focus will shift to the context
     * widget
     */
    virtual void setEnabled(bool enable, bool refocusWhenEnabling = false);

protected Q_SLOTS:
    /**
     * Called when any shortcut is enabled.
     * Ensures all shortcuts are enabled and marks the widget as active.
     */
    virtual void onShortcutEnabled();

    /**
     * Called when any shortcut is disabled.
     * Ensures all shortcuts are disabled and marks the widget as inactive.
     */
    virtual void onShortcutDisabled();

protected:
    /**
     * Get the decorated widget as a QFrame.
     */
    QFrame* decoratedFrame() const;

    /**
     * Monitor mouse events to allow users to enable/disable shortcuts.
     */
    bool eventFilter(QObject* obj, QEvent* event) override;

    /**
     * Show/hide and color the frame border.
     *
     * @param active True if the shortcut is active
     * @param frameColor The color to use for the border
     */
    void markFrame(bool active, const QColor& frameColor);

    /// All the shortcuts that decorate the widget.
    /// These will all be enabled/disabled en banc.
    QList<QPointer<ecvModalShortcut>> m_shortcuts;

    /// Note when the user has pressed the mouse inside the widget and not
    /// released it.
    bool m_pressed;

    /// Prevent recursive signaling inside onShortcutEnabled/onShortcutDisabled.
    bool m_silent;

    /// Should shortcuts set the keyboard focus to their context widget?
    /// This is set to true when users explicitly click on the widget frame
    /// and false otherwise.
    bool m_allowRefocus;
};

#endif  // ecvShortcutDecorator_h
