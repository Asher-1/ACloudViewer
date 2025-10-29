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
 * \brief ColorPushButton, 是对QtColorPicker的扩展.
 * \ingroup Pictureui
 */

/*!
 * \brief 构造拾色器按钮类, 初始化拾色器按钮.
 * \param parent, 父窗口
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
