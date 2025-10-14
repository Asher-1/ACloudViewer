// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_SF_FROM_COLOR_DLG_HEADER
#define ECV_SF_FROM_COLOR_DLG_HEADER

#include <ui_scalarFieldFromColorDlg.h>

class ccPointCloud;

//! Dialog to choose 2 scalar fields (SF) and one operation for arithmetics
//! processing
class ccScalarFieldFromColorDlg : public QDialog,
                                  public Ui::scalarFieldFromColorDlg {
    Q_OBJECT

public:
    //! Default constructor
    explicit ccScalarFieldFromColorDlg(QWidget* parent = 0);

    //! Returns if to export R channel as SF
    bool getRStatus();

    //! Returns if to export G channel as SF
    bool getGStatus();

    //! Returns if to export B channel as SF
    bool getBStatus();

    //! Returns if to export Composite channel as SF
    bool getCompositeStatus();
};

#endif  // ECV_SF_FROM_COLOR_DLG_HEADER
