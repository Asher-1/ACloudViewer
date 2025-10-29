// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ui_askThreeDoubleValuesDlg.h>

// Qt
#include <QString>

//! Generic dialog to query 3 (double) values
class ccAskThreeDoubleValuesDlg : public QDialog,
                                  public Ui::AskThreeDoubleValuesDialog {
    Q_OBJECT

public:
    //! Default constructor
    ccAskThreeDoubleValuesDlg(const QString& vName1,
                              const QString& vName2,
                              const QString& vName3,
                              double minVal,
                              double maxVal,
                              double defaultVal1,
                              double defaultVal2,
                              double defaultVal3,
                              int precision = 6,
                              const QString windowTitle = QString(),
                              QWidget* parent = nullptr);

    //! Enable the checkbox (bottom-left)
    void showCheckbox(const QString& label,
                      bool state,
                      QString tooltip = QString());

    //! Returns
    bool getCheckboxState() const;
};
