// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_NOISE_FILTER_DLG_HEADER
#define ECV_NOISE_FILTER_DLG_HEADER

#include <ui_noiseFilterDlg.h>

//! Dialog for noise filtering (based on the distance to the implicit local
//! surface)
class ecvNoiseFilterDlg : public QDialog, public Ui::NoiseFilterDialog {
    Q_OBJECT

public:
    //! Default constructor
    explicit ecvNoiseFilterDlg(QWidget* parent = 0);
};

#endif  // ECV_NOISE_FILTER_DLG_HEADER
