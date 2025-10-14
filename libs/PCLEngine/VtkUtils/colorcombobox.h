// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef COLORCOMBOBOX_H
#define COLORCOMBOBOX_H

#include <QComboBox>

#include "../qPCL.h"

namespace Widgets {

class QPCL_ENGINE_LIB_API ColorComboBox : public QComboBox {
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
#endif  // COLORCOMBOBOX_H
