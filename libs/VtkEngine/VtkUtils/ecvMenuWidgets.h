// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file ecvMenuWidgets.h
/// @brief Host real button widgets inside QMenu so checkable menu entries stay
/// clickable under every appearance mode.

#include <QAbstractButton>
#include <QCheckBox>
#include <QMenu>
#include <QRadioButton>
#include <QString>
#include <QWidgetAction>
#include <functional>

/// Create a checkable button parented to @p menu without connecting or hosting
/// it yet, so callers can wire cross-references (mutual exclusion) first.
/// The spacing-only stylesheet keeps the themed indicator intact under the
/// Custom (Qlementine) backend and is harmless under Native styles.
template <typename Button>
inline Button* ecvCreateMenuButton(QMenu* menu,
                                   const QString& text,
                                   bool checked) {
    auto* button = new Button(text, menu);
    button->setChecked(checked);
    button->setCursor(Qt::PointingHandCursor);
    button->setStyleSheet(
            QStringLiteral("%1 { padding: 4px 8px; }")
                    .arg(QLatin1String(Button::staticMetaObject.className())));
    return button;
}

/// Host an existing button as a QWidgetAction. The click is then handled by the
/// child widget directly, independent of QMenu's action-trigger path (which the
/// Custom style mediates through a flash/synthetic-event filter that can
/// swallow clicks). QMenu::clear() deletes the QWidgetAction and its hosted
/// widget; do not pre-delete the button.
inline QWidgetAction* ecvHostMenuButton(QMenu* menu, QAbstractButton* button) {
    auto* action = new QWidgetAction(menu);
    action->setDefaultWidget(button);
    menu->addAction(action);
    return action;
}

/// Convenience: create + connect + host a checkable menu button. @p context
/// bounds the connection lifetime; @p onToggled runs on user toggles only (the
/// initial state is applied before the connection is made).
template <typename Button>
inline Button* ecvAddMenuButton(QMenu* menu,
                                const QString& text,
                                bool checked,
                                const QObject* context,
                                std::function<void(bool)> onToggled) {
    auto* button = ecvCreateMenuButton<Button>(menu, text, checked);
    QObject::connect(button, &QAbstractButton::toggled, context,
                     [onToggled](bool value) { onToggled(value); });
    ecvHostMenuButton(menu, button);
    return button;
}