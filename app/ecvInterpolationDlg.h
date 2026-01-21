// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ui_interpolationDlg.h>

// CV_DB_LIB
#include <ecvPointCloudInterpolator.h>

//! Dialog for generic interpolation algorithms
class ccInterpolationDlg : public QDialog, public Ui::InterpolationDlg {
    Q_OBJECT

public:
    //! Default constructor
    explicit ccInterpolationDlg(QWidget* parent = 0);

    ccPointCloudInterpolator::Parameters::Method getInterpolationMethod() const;
    void setInterpolationMethod(
            ccPointCloudInterpolator::Parameters::Method method);

    ccPointCloudInterpolator::Parameters::Algo getInterpolationAlgorithm()
            const;
    void setInterpolationAlgorithm(
            ccPointCloudInterpolator::Parameters::Algo algo);

protected slots:

    void onRadiusUpdated(double);
};
