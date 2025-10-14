// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QDialog>
#include <QDialogButtonBox>

/*
Simple class for displaying the help dialog.
*/
class ccCompassInfo : public QDialog {
    Q_OBJECT

public:
    explicit ccCompassInfo(QWidget *parent = nullptr);
};
