// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Qt
#include <ecvGLMatrix.h>
#include <ui_matrixDisplayDlg.h>

#include <QWidget>

//! Simple widget to display a 4x4 matrix in various formats
class MatrixDisplayDlg : public QWidget, public Ui::MatrixDisplayDlg {
    Q_OBJECT

public:
    //! Default constructor
    explicit MatrixDisplayDlg(QWidget* parent = 0);

    //! Clears widget
    void clear();

    //! Updates dialog with a given (float) matrix
    void fillDialogWith(const ccGLMatrix& mat);
    //! Updates dialog with a given (double) matrix
    void fillDialogWith(const ccGLMatrixd& mat);

public slots:

    //! Exports current matrix to an ASCII file
    /** Will display a file selection dialog!
     **/
    void exportToASCII();
    //! Exports current matrix to the clipboard
    void exportToClipboard();

protected:
    //! Fills the second part of the dialog
    void fillDialogWith(const CCVector3d& axis,
                        double angle_rad,
                        const CCVector3d& T,
                        int precision);

    //! Matrix
    ccGLMatrixd m_mat;
};
