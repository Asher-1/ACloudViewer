// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ui_MLSDialog.h>

// Qt
#include <QDialog>

class MLSDialog : public QDialog, public Ui::MLSDialog {
    Q_OBJECT

public:
    explicit MLSDialog(QWidget *parent = nullptr);

protected slots:
    void activateMenu(QString name);
    void toggleMethods(bool status);
    void updateSquaredGaussian(double radius);

protected:
    void updateCombo();
    void deactivateAllMethods();
};
