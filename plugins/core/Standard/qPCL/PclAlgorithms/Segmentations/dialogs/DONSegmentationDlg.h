// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ui_DONSegmentationDlg.h>

// Qt
#include <QDialog>

// system
#include <vector>

class DONSegmentationDlg : public QDialog, public Ui::DONSegmentationDlg {
public:
    explicit DONSegmentationDlg(QWidget* parent = 0);

    const QString getComparisonField();
    void getComparisonTypes(QStringList& types);
};
