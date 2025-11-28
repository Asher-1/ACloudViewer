// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QDialog>

class ecvMainAppInterface;

namespace Ui {
class G3PointDisclaimer;
}

class G3PointDisclaimer : public QDialog {
    Q_OBJECT

public:
    explicit G3PointDisclaimer(QWidget *parent = nullptr);
    ~G3PointDisclaimer();

    static bool show(ecvMainAppInterface *app);

private:
    // whether disclaimer has already been displayed (and accepted) or not
    static bool s_disclaimerAccepted;

    Ui::G3PointDisclaimer *ui;
};
