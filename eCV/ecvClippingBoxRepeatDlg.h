// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ui_clippingBoxRepeatDlg.h>

// Qt
#include <QDialog>

//! Dialog for managing clipping box based repeated processes
class ccClippingBoxRepeatDlg : public QDialog, public Ui::ClippingBoxRepeatDlg {
    Q_OBJECT

public:
    //! Default constructor
    ccClippingBoxRepeatDlg(bool singleContourMode = false, QWidget* parent = 0);

    //! Sets flat dimension (single contour mode only!)
    void setFlatDim(unsigned char dim);
    //! Sets repeat dimension (multiple contour mode only!)
    void setRepeatDim(unsigned char dim);

protected slots:

    // multi-contour mode
    void onDimChecked(bool);

    // single-contour mode
    void onDimXChecked(bool);
    void onDimYChecked(bool);
    void onDimZChecked(bool);
};
