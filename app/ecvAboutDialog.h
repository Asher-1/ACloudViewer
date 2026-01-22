// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QDialog>

namespace Ui {
class AboutDialog;
}

class ecvAboutDialog : public QDialog {
    Q_OBJECT

public:
    ecvAboutDialog(QWidget *parent = nullptr);
    ~ecvAboutDialog();

private:
    Ui::AboutDialog *mUI;
};
