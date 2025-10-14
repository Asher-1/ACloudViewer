// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ui_SIFTExtractDlg.h>

// Qt
#include <QDialog>

// system
#include <vector>

class SIFTExtractDlg : public QDialog, public Ui::SIFTExtractDlg {
public:
    explicit SIFTExtractDlg(QWidget* parent = nullptr);

    void updateComboBox(const std::vector<std::string>& fields);
};
