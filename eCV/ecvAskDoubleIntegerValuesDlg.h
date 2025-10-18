// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ui_askDoubleIntegerValuesDlg.h>

//! Dialog to input 2 values with custom labels
class ecvAskDoubleIntegerValuesDlg : public QDialog,
                                     public Ui::AskDoubleIntegerValuesDialog {
    Q_OBJECT

public:
    //! Default constructor
    ecvAskDoubleIntegerValuesDlg(const char* vName1,
                                 const char* vName2,
                                 double minVal,
                                 double maxVal,
                                 int minInt,
                                 int maxInt,
                                 double defaultVal1,
                                 int defaultVal2,
                                 int precision = 6,
                                 const char* windowTitle = 0,
                                 QWidget* parent = 0);
};
