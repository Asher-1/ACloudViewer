// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_ASK_TWO_DOUBLE_VALUES_DIALOG_HEADER
#define ECV_ASK_TWO_DOUBLE_VALUES_DIALOG_HEADER

#include <ui_askTwoDoubleValuesDlg.h>

//! Dialog to input 2 values with custom labels
class ccAskTwoDoubleValuesDlg : public QDialog,
                                public Ui::AskTwoDoubleValuesDialog {
    Q_OBJECT

public:
    //! Default constructor
    ccAskTwoDoubleValuesDlg(const char* vName1,
                            const char* vName2,
                            double minVal,
                            double maxVal,
                            double defaultVal1,
                            double defaultVal2,
                            int precision = 6,
                            const char* windowTitle = 0,
                            QWidget* parent = 0);
};

#endif  // ECV_ASK_TWO_DOUBLE_VALUES_DIALOG_HEADER
