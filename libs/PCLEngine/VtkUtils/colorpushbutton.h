// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "qPCL.h"
#include "qtcolorpicker.h"

namespace Widgets {

class QPCL_ENGINE_LIB_API ColorPushButton : public QtColorPicker {
    Q_OBJECT

public:
    ColorPushButton(QWidget* parent = 0);
    ~ColorPushButton();
    void setColor(const QColor& c) { setCurrentColor(c); }
    QColor color() { return currentColor(); }

signals:
    void colorChanged();

private:
    Q_DISABLE_COPY(ColorPushButton)
};

}  // namespace Widgets
