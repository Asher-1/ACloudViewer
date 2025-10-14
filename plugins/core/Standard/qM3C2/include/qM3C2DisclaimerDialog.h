// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef M3C2_DISCLAIMER_DIALOG_HEADER
#define M3C2_DISCLAIMER_DIALOG_HEADER

// Qt
#include <QDialog>

class ecvMainAppInterface;

namespace Ui {
class DisclaimerDialog;
}

//! Dialog for displaying the M3C2/UEB disclaimer
class DisclaimerDialog : public QDialog {
public:
    //! Default constructor
    DisclaimerDialog(QWidget* parent = nullptr);
    ~DisclaimerDialog();

    static bool show(ecvMainAppInterface* app);

private:
    // whether disclaimer has already been displayed (and accepted) or not
    static bool s_disclaimerAccepted;

    Ui::DisclaimerDialog* m_ui;
};

#endif  // M3C2_DISCLAIMER_DIALOG_HEADER
