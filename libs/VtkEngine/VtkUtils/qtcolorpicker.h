// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file qtcolorpicker.h
/// @brief Push button with color picker popup and optional color dialog.

#include <QColor>
#include <QEvent>
#include <QFocusEvent>
#include <QLabel>
#include <QPushButton>
#include <QString>

#include "qVTK.h"

namespace Widgets {

class ColorPickerPopup;

/// @class QtColorPicker
/// @brief Push button that shows color grid popup and optional QColorDialog.
class QVTK_ENGINE_LIB_API QtColorPicker : public QPushButton {
    Q_OBJECT

    Q_PROPERTY(bool colorDialog READ colorDialogEnabled WRITE
                       setColorDialogEnabled)

public:
    /// @param parent Parent widget
    /// @param columns Popup grid columns (-1 for auto)
    /// @param enableColorDialog Enable "More colors" dialog
    QtColorPicker(QWidget *parent = 0,
                  int columns = -1,
                  bool enableColorDialog = true);

    ~QtColorPicker();

    /// @param color Color to add
    /// @param text Optional label
    /// @param index Insert position (-1 for append)
    void insertColor(const QColor &color,
                     const QString &text = QString(),
                     int index = -1);

    /// @return Currently selected color
    QColor currentColor() const;

    /// @param index Color index
    /// @return Color at index
    QColor color(int index) const;

    /// @param enabled Enable color dialog
    void setColorDialogEnabled(bool enabled);
    /// @return true if color dialog enabled
    bool colorDialogEnabled() const;

    void setStandardColors();

    /// @param pos Screen position for popup
    /// @param allowCustomColors Allow custom color selection
    /// @return Selected color
    static QColor GetColor(const QPoint &pos, bool allowCustomColors = true);

public Q_SLOTS:
    /// @param col Color to set as current
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
