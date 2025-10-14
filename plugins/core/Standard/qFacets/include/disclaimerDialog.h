// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef QFACET_DISCLAIMER_DIALOG_HEADER
#define QFACET_DISCLAIMER_DIALOG_HEADER

#include <ui_disclaimerDlg.h>

// ECV_PLUGINS
#include <ecvMainAppInterface.h>

// Qt
#include <QMainWindow>

//! Dialog for displaying the BRGM disclaimer
class DisclaimerDialog : public QDialog, public Ui::DisclaimerDialog {
public:
    //! Default constructor
    DisclaimerDialog(QWidget* parent = 0)
        : QDialog(parent), Ui::DisclaimerDialog() {
        setupUi(this);
    }
};

// whether disclaimer has already been displayed (and accepted) or not
static bool s_disclaimerAccepted = false;

static bool ShowDisclaimer(ecvMainAppInterface* app) {
    if (!s_disclaimerAccepted) {
        // if the user "cancels" it, then he refuses the diclaimer!
        s_disclaimerAccepted =
                DisclaimerDialog(app ? app->getMainWindow() : 0).exec();
    }

    return s_disclaimerAccepted;
}

#endif  // QFACET_DISCLAIMER_DIALOG_HEADER
