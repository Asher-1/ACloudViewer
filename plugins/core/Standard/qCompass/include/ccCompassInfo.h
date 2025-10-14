// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_COMPASS_INFO_HEADER
#define ECV_COMPASS_INFO_HEADER

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

#endif  // ECV_COMPASS_INFO_HEADER