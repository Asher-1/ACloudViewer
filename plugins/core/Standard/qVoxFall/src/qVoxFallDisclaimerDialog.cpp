// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qVoxFallDisclaimerDialog.h"

#include "ui_disclaimerDlg.h"

// qCC_plugins
#include <ecvMainAppInterface.h>

// Qt
#include <QMainWindow>

bool DisclaimerDialog::s_disclaimerAccepted = false;

DisclaimerDialog::DisclaimerDialog(QWidget *parent)
    : QDialog(parent), m_ui(new Ui::DisclaimerDialog) {
    m_ui->setupUi(this);
}

DisclaimerDialog::~DisclaimerDialog() { delete m_ui; }

bool DisclaimerDialog::show(ecvMainAppInterface *app) {
    if (!s_disclaimerAccepted) {
        // if the user "cancels" it, then he refuses the disclaimer
        s_disclaimerAccepted =
                DisclaimerDialog(app ? app->getMainWindow() : 0).exec();
    }

    return s_disclaimerAccepted;
}
