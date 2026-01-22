// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Qt
#include <ui_colorGradientDlg.h>

#include <QColor>

//! Dialog to define a color gradient (default, with 2 colors, banding, etc.)
class ccColorGradientDlg : public QDialog, public Ui::ColorGradientDialog {
    Q_OBJECT

public:
    //! Default constructor
    explicit ccColorGradientDlg(QWidget* parent);

    //! Gradient types
    enum GradientType { Default, TwoColors, Banding };

    //! Returns selected gradient type
    GradientType getType() const;
    //! Sets the currently activated type
    void setType(GradientType type);

    //! Returns the two colors of the gradient ('TwoColors' mode)
    void getColors(QColor& first, QColor& second) const;

    //! Returns the frequency of the gradient ('Banding' mode)
    double getBandingFrequency() const;

    //! Returns the ramp dimension
    unsigned char getDimension() const;

protected slots:

    void changeFirstColor();
    void changeSecondColor();
};
