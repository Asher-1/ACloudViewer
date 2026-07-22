// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "colorpushbutton.h"

#include <QColorDialog>

#include "colorcombobox.h"

namespace Widgets {

/*!
 * \class ColorPushButton
 * \brief ColorPushButton, an extension of QtColorPicker.
 * \ingroup Pictureui
 */

/*!
 * \brief Constructs the color picker button and initializes it.
 * \param parent Parent widget
 */
ColorPushButton::ColorPushButton(QWidget *parent) : QtColorPicker(parent) {
    QStringList color_names = ColorComboBox::defaultColorNames();
    QList<QColor> defaultColors = ColorComboBox::defaultColors();
    for (int i = 0; i < ColorComboBox::numPredefinedColors(); i++)
        insertColor(defaultColors[i], color_names[i]);

    QList<QColor> colors = ColorComboBox::colorList();
    color_names = ColorComboBox::colorNames();
    for (int i = 0; i < colors.count(); i++) {
        QColor c = colors[i];
        if (!defaultColors.contains(c)) insertColor(c, color_names[i]);
    }

    connect(this, SIGNAL(colorChanged(const QColor &)), this,
            SIGNAL(colorChanged()));
}

ColorPushButton::~ColorPushButton() {}

}  // namespace Widgets
