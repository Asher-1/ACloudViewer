// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef VOXFALL_DISCLAIMER_DIALOG_HEADER
#define VOXFALL_DISCLAIMER_DIALOG_HEADER

#include <QDialog>

class ecvMainAppInterface;

namespace Ui {
class DisclaimerDialog;
}

//! Dialog for displaying the VoxFall disclaimer
class DisclaimerDialog : public QDialog {
public:
    DisclaimerDialog(QWidget* parent = nullptr);
    ~DisclaimerDialog();

    static bool show(ecvMainAppInterface* app);

private:
    // whether disclaimer has already been displayed (and accepted) or not
    static bool s_disclaimerAccepted;

    Ui::DisclaimerDialog* m_ui;
};

#endif  // VOXFALL_DISCLAIMER_DIALOG_HEADER
