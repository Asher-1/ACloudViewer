// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ui_matchScalesDlg.h>

// Local
#include "ecvLibAlgorithms.h"

//! Scales matching tool dialog
class ccMatchScalesDlg : public QDialog, public Ui::MatchScalesDialog {
    Q_OBJECT

public:
    //! Default constructor
    ccMatchScalesDlg(const ccHObject::Container& entities,
                     int defaultSelectedIndex = 0,
                     QWidget* parent = 0);

    //! Returns selected index
    int getSelectedIndex() const;

    //! Sets the selected matching algorithm
    void setSelectedAlgorithm(
            ccLibAlgorithms::ScaleMatchingAlgorithm algorithm);

    //! Returns the selected matching algorithm
    ccLibAlgorithms::ScaleMatchingAlgorithm getSelectedAlgorithm() const;
};
