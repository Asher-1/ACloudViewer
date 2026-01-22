// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// CC_Lib
#include <CVGeom.h>

// Qt
#include <ui_scaleDlg.h>

#include <QDialog>

//! Scale / multiply dialog
class ccScaleDlg : public QDialog, public Ui::ScaleDialog {
    Q_OBJECT

public:
    //! Default constructor
    explicit ccScaleDlg(QWidget* parent = 0);

    //! Returns scales
    CCVector3d getScales() const;

    //! Whether the entity should be 'kept in place' or not
    bool keepInPlace() const;

    //! Whether the Global shift should be rescaled as well
    bool rescaleGlobalShift() const;

    //! Saves state
    void saveState();

protected slots:

    void allDimsAtOnceToggled(bool);
    void fxUpdated(double);

protected:
};
