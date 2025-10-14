// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_STATISTICAL_TEST_DLG_HEADER
#define ECV_STATISTICAL_TEST_DLG_HEADER

#include <ui_statisticalTestDlg.h>

//! Dialog for the Local Statistical Test tool
class ccStatisticalTestDlg : public QDialog, public Ui::StatisticalTestDialog {
public:
    //! Default constructor (for distributions with up to 3 parameters)
    ccStatisticalTestDlg(QString param1Label,
                         QString param2Label,
                         QString param3Label = QString(),
                         QString windowTitle = QString(),
                         QWidget* parent = 0);

    //! Returns 1st parameter value
    double getParam1() const;
    //! Returns 2nd parameter value
    double getParam2() const;
    //! Returns 3rd parameter value
    double getParam3() const;

    //! Returns the number of neighbors
    int getNeighborsNumber() const;
    //! Returns the associated probability
    double getProba() const;
};

#endif  // ECV_STATISTICAL_TEST_DLG_HEADER