// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_PRIMITIVE_FACTORY_DLG_HEADER
#define ECV_PRIMITIVE_FACTORY_DLG_HEADER

#include "ui_primitiveFactoryDlg.h"

// Qt
#include <QDialog>

class MainWindow;
class ccGLMatrix;

//! Primitive factory
class ecvPrimitiveFactoryDlg : public QDialog, public Ui::PrimitiveFactoryDlg {
    Q_OBJECT

public:
    //! Default constructor
    explicit ecvPrimitiveFactoryDlg(MainWindow* win);

protected slots:

    //! Creates currently defined primitive
    void createPrimitive();

protected:
    //! Set sphere position from clipboard
    void setSpherePositionFromClipboard();

    //! Set sphere position to origin
    void setSpherePositionToOrigin();

    void setCoordinateSystemBasedOnSelectedObject();

    void onMatrixTextChange();

    void setCSMatrixToIdentity();

    ccGLMatrix getCSMatrix(bool& valid);

protected:
    //! Associated main window
    MainWindow* m_win;
};

#endif  // ECV_PRIMITIVE_FACTORY_DLG_HEADER
