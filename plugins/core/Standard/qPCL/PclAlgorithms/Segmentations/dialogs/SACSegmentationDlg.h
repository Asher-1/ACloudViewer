// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ui_SACSegmentationDlg.h>

// Qt
#include <QDialog>

// system
#include <vector>

class SACSegmentationDlg : public QDialog, public Ui::SACSegmentationDlg {
public:
    explicit SACSegmentationDlg(QWidget* parent = 0);

protected slots:
    void modelsChanged(int currentIndex = 0);

private:
    void initParameters();
    void updateModelTypeComboBox(const QStringList& fields);
    void updateMethodTypeComboBox(const QStringList& fields);
};
