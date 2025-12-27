// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "Shortcuts/ecvShortcutDecorator.h"

#include "Shortcuts/ecvKeySequences.h"
#include "Shortcuts/ecvModalShortcut.h"

// eCV
#include <CVLog.h>

#include <QEvent>
#include <QFrame>
#include <QKeySequence>

//-----------------------------------------------------------------------------
ecvShortcutDecorator::ecvShortcutDecorator(QFrame* parent)
    : Superclass(parent),
      m_pressed(false),
      m_silent(false),
      m_allowRefocus(false) {
    if (!parent) {
        CVLog::Error("[ecvShortcutDecorator] Cannot decorate null widget!");
        return;
    }

    // Set up the frame for decoration
    parent->setLineWidth(2);
    parent->setFrameStyle(QFrame::NoFrame | QFrame::Plain);
    this->markFrame(false, QColor(0, 0, 0, 0));

    // Install event filter to monitor mouse events
    parent->installEventFilter(this);
}

//-----------------------------------------------------------------------------
void ecvShortcutDecorator::addShortcut(ecvModalShortcut* shortcut) {
    if (!shortcut) {
        CVLog::Warning(
                "[ecvShortcutDecorator] Attempted to add null shortcut!");
        return;
    }

    m_shortcuts.push_back(shortcut);

    // Make border active; color it; enable all shortcuts for this widget
    this->onShortcutEnabled();

    // Connect signals
    QObject::connect(shortcut, &ecvModalShortcut::enabled, this,
                     &ecvShortcutDecorator::onShortcutEnabled);
    QObject::connect(shortcut, &ecvModalShortcut::disabled, this,
                     &ecvShortcutDecorator::onShortcutDisabled);

    CVLog::Print(QString("[ecvShortcutDecorator] Added shortcut: %1")
                         .arg(shortcut->keySequence().toString()));
}

//-----------------------------------------------------------------------------
bool ecvShortcutDecorator::isEnabled() const {
    // All attached shortcuts should have the same state, so just take the first
    // (if any).
    if (m_shortcuts.empty()) {
        return false;
    }

    for (auto& shortcut : m_shortcuts) {
        if (shortcut) {
            return shortcut->isEnabled();
        }
    }

    return false;
}

//-----------------------------------------------------------------------------
void ecvShortcutDecorator::onShortcutEnabled() {
    if (m_silent) {
        return;
    }

    // One shortcut was just enabled... but to mark ourselves
    // as active, we must activate all of them.
    m_silent = true;
    for (auto& shortcut : m_shortcuts) {
        if (shortcut) {
            shortcut->setEnabled(true, m_allowRefocus);
        }
    }
    m_silent = false;
    m_allowRefocus = false;  // Always reset after use.

    // Set the visual style (use link color from palette)
    auto* frame = this->decoratedFrame();
    if (frame) {
        this->markFrame(true, frame->palette().link().color());
    }
}

//-----------------------------------------------------------------------------
void ecvShortcutDecorator::onShortcutDisabled() {
    if (m_silent) {
        return;
    }

    // One shortcut was just disabled... but to mark ourselves
    // as inactive, we must deactivate all of them.
    m_silent = true;
    for (auto& shortcut : m_shortcuts) {
        if (shortcut) {
            shortcut->setEnabled(false);
        }
    }
    m_silent = false;

    // Set the visual style (transparent)
    this->markFrame(false, QColor(0, 0, 0, 0));
}

//-----------------------------------------------------------------------------
void ecvShortcutDecorator::setEnabled(bool enable, bool refocusWhenEnabling) {
    if (enable) {
        m_allowRefocus = refocusWhenEnabling;  // This will be reset inside
                                               // onShortcutEnabled.
        // This has the effect of turning all shortcuts on.
        this->onShortcutEnabled();
    } else {
        // This has the effect of turning all shortcuts off.
        this->onShortcutDisabled();
    }
}

//-----------------------------------------------------------------------------
QFrame* ecvShortcutDecorator::decoratedFrame() const {
    return qobject_cast<QFrame*>(this->parent());
}

//-----------------------------------------------------------------------------
bool ecvShortcutDecorator::eventFilter(QObject* obj, QEvent* event) {
    if (obj == this->parent()) {
        auto* frame = this->decoratedFrame();
        if (!frame) {
            return Superclass::eventFilter(obj, event);
        }

        switch (event->type()) {
            case QEvent::Enter:
                // On mouse enter, make the border darker
                this->markFrame(true, frame->palette().link().color().darker());
                break;

            case QEvent::Leave: {
                // On mouse leave, restore the border to active or transparent
                bool ena = this->isEnabled();
                this->markFrame(ena, ena ? frame->palette().link().color()
                                         : QColor(0, 0, 0, 0));
                m_pressed = false;
            } break;

            case QEvent::MouseButtonPress:
                m_pressed = true;
                break;

            case QEvent::MouseButtonRelease:
                if (m_pressed) {
                    m_pressed = false;

                    // Reorder how shortcuts will cycle so that the previous
                    // sibling is this shortcut's "next".
                    for (auto& shortcut : m_shortcuts) {
                        if (shortcut) {
                            ecvKeySequences::instance().reorder(shortcut);
                        }
                    }

                    // Toggle the enabled state
                    this->setEnabled(!this->isEnabled(), true);

                    // Eat this mouse event:
                    return true;
                }
                break;

            default:
                // do nothing
                break;
        }
    }
    return Superclass::eventFilter(obj, event);
}

//-----------------------------------------------------------------------------
void ecvShortcutDecorator::markFrame(bool active, const QColor& frameColor) {
    (void)active;  // This can be used to modulate line width, frame shape.

    auto* frame = this->decoratedFrame();
    if (!frame) {
        return;
    }

    frame->setFrameShape(QFrame::Box);
    frame->setLineWidth(2);

    // Use stylesheet to set the border color
    QString objectName = frame->objectName();
    if (objectName.isEmpty()) {
        // If no object name, set a temporary one
        objectName = QString("ecvDecoratedFrame_%1")
                             .arg(reinterpret_cast<quintptr>(frame));
        frame->setObjectName(objectName);
    }

    frame->setStyleSheet(
            QString("QFrame#%1 { border: 2px solid rgba(%2, %3, %4, %5); }")
                    .arg(objectName)
                    .arg(frameColor.red())
                    .arg(frameColor.green())
                    .arg(frameColor.blue())
                    .arg(frameColor.alpha()));
}
