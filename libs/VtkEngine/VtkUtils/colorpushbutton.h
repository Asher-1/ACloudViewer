// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file colorpushbutton.h
/// @brief Push button that opens a color picker and emits color changes.

#include "qVTK.h"
#include "qtcolorpicker.h"

namespace Widgets {

/// @class ColorPushButton
/// @brief Color picker button; extends QtColorPicker with colorChanged signal.
class QVTK_ENGINE_LIB_API ColorPushButton : public QtColorPicker {
    Q_OBJECT

public:
    ColorPushButton(QWidget* parent = 0);
    ~ColorPushButton();
    /// @param c Color to set
    void setColor(const QColor& c) { setCurrentColor(c); }
    /// @return Currently selected color
    QColor color() { return currentColor(); }

signals:
    void colorChanged();

private:
    Q_DISABLE_COPY(ColorPushButton)
};

}  // namespace Widgets
