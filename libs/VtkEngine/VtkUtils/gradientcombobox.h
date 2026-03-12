// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file gradientcombobox.h
/// @brief ComboBox for selecting two-color gradient presets.

#include <QComboBox>

#include "qVTK.h"

class QListWidgetItem;
namespace Widgets {

class GradientComboBoxPrivate;
/// @class GradientComboBox
/// @brief ComboBox for gradient selection; shows color pair previews and emits
/// colorsChanged.
class QVTK_ENGINE_LIB_API GradientComboBox : public QComboBox {
    Q_OBJECT
public:
    explicit GradientComboBox(QWidget* parent = nullptr);
    ~GradientComboBox();

    void setCurrentIndex(int index);
    int currentIndex() const;

    /// @return Name of current gradient
    QString currentName() const;
    /// @return First color of current gradient
    QColor currentColor1() const;
    /// @return Second color of current gradient
    QColor currentColor2() const;

    /// @param index Gradient index
    /// @return Color pair for gradient at index
    static QPair<QColor, QColor> colorPair(int index);

    void showPopup();
    void hidePopup();

signals:
    void colorsChanged(const QString& name,
                       const QColor& clr1,
                       const QColor& clr2);

protected:
    void paintEvent(QPaintEvent* e);
    void resizeEvent(QResizeEvent* e);
    QSize minimumSizeHint() const;

private slots:
    void onActivated(int index);

private:
    GradientComboBoxPrivate* d_ptr;
    Q_DISABLE_COPY(GradientComboBox)
};

}  // namespace Widgets
