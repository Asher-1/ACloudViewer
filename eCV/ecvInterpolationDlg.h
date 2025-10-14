// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_INTERPOLATION_DLG_HEADER
#define ECV_INTERPOLATION_DLG_HEADER

#include <ui_interpolationDlg.h>

// ECV_DB_LIB
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

#endif  // ECV_INTERPOLATION_DLG_HEADER
