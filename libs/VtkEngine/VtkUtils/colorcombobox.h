// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file colorcombobox.h
 * @brief Combo box widget for selecting colors from a predefined palette.
 */

#include <QComboBox>

#include "qVTK.h"

namespace Widgets {

/**
 * @class ColorComboBox
 * @brief QComboBox displaying a palette of named colors with color-swatch
 * previews.
 */
class QVTK_ENGINE_LIB_API ColorComboBox : public QComboBox {
    Q_OBJECT

public:
    ColorComboBox(QWidget* parent = 0);
    void setColor(const QColor& c);
    QColor color() const;
    static QList<QColor> colorList();
    static QStringList colorNames();
    static int colorIndex(const QColor& c);
    static QColor color(int colorIndex);
    static QColor defaultColor(int colorIndex);
    static bool isValidColor(const QColor& color);
    static int numPredefinedColors();
    static QStringList defaultColorNames();
    static QList<QColor> defaultColors();

protected:
    void init();
    static const int stColorsCount = 24;
    static const QColor stColors[];

private:
    Q_DISABLE_COPY(ColorComboBox)
};

}  // namespace Widgets
