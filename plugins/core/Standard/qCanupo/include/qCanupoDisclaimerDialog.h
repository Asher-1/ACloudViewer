// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ui_classifyDisclaimerDlg.h>
#include <ui_trainDisclaimerDlg.h>

// ECV_PLUGINS
#include <ecvMainAppInterface.h>

// Qt
#include <QMainWindow>

//! Dialog for displaying the CANUPO/UEB disclaimer
class TrainDisclaimerDialog : public QDialog, public Ui::TrainDisclaimerDialog {
public:
    //! Default constructor
    TrainDisclaimerDialog(QWidget* parent = 0)
        : QDialog(parent), Ui::TrainDisclaimerDialog() {
        setupUi(this);
    }
};

// whether disclaimer has already been displayed (and accepted) or not
static bool s_trainDisclaimerAccepted = false;

static bool ShowTrainDisclaimer(ecvMainAppInterface* app) {
    if (!s_trainDisclaimerAccepted) {
        // if the user "cancels" it, then he refuses the diclaimer!
        s_trainDisclaimerAccepted =
                TrainDisclaimerDialog(app ? app->getMainWindow() : 0).exec();
    }

    return s_trainDisclaimerAccepted;
}

//! Dialog for displaying the CANUPO/UEB disclaimer
class ClassifyDisclaimerDialog : public QDialog,
                                 public Ui::ClassifyDisclaimerDialog {
public:
    //! Default constructor
    ClassifyDisclaimerDialog(QWidget* parent = 0)
        : QDialog(parent), Ui::ClassifyDisclaimerDialog() {
        setupUi(this);
    }
};

// whether disclaimer has already been displayed (and accepted) or not
static bool s_classifyDisclaimerAccepted = false;

static bool ShowClassifyDisclaimer(ecvMainAppInterface* app) {
    if (!s_classifyDisclaimerAccepted) {
        // if the user "cancels" it, then he refuses the diclaimer!
        s_classifyDisclaimerAccepted =
                ClassifyDisclaimerDialog(app ? app->getMainWindow() : 0).exec();
    }

    return s_classifyDisclaimerAccepted;
}
