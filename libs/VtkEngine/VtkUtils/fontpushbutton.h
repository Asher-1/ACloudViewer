// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file fontpushbutton.h
/// @brief Push button that opens a font dialog and emits the selected font.

#include <QPushButton>

#include "qVTK.h"

namespace Widgets {

/// @class FontPushButton
/// @brief Button that opens QFontDialog on click and emits fontSelected with
/// the chosen font.
class QVTK_ENGINE_LIB_API FontPushButton : public QPushButton {
    Q_OBJECT
public:
    explicit FontPushButton(QWidget* parent = 0);
    /// @param text Button label text
    /// @param parent Parent widget
    explicit FontPushButton(const QString& text, QWidget* parent = 0);

signals:
    void fontSelected(const QFont& font);

private slots:
    void onClicked();

private:
    void init();
};

}  // namespace Widgets
