// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef QTCOLORPICKER_H
#define QTCOLORPICKER_H
#include <QColor>
#include <QEvent>
#include <QFocusEvent>
#include <QLabel>
#include <QPushButton>
#include <QString>

#include "../qPCL.h"

namespace Widgets {

class ColorPickerPopup;

class QPCL_ENGINE_LIB_API QtColorPicker : public QPushButton {
    Q_OBJECT

    Q_PROPERTY(bool colorDialog READ colorDialogEnabled WRITE
                       setColorDialogEnabled)

public:
    QtColorPicker(QWidget *parent = 0,
                  int columns = -1,
                  bool enableColorDialog = true);

    ~QtColorPicker();

    void insertColor(const QColor &color,
                     const QString &text = QString(),
                     int index = -1);

    QColor currentColor() const;

    QColor color(int index) const;

    void setColorDialogEnabled(bool enabled);
    bool colorDialogEnabled() const;

    void setStandardColors();

    static QColor GetColor(const QPoint &pos, bool allowCustomColors = true);

public Q_SLOTS:
    void setCurrentColor(const QColor &col);

Q_SIGNALS:
    void colorChanged(const QColor &);

protected:
    void paintEvent(QPaintEvent *e);

private Q_SLOTS:
    void buttonPressed(bool toggled);
    void popupClosed();

private:
    ColorPickerPopup *popup;
    QColor col;
    bool withColorDialog;
    bool dirty;
    bool firstInserted;
};

}  // namespace Widgets
#endif
