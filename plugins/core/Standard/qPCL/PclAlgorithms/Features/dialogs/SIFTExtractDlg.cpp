// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "SIFTExtractDlg.h"

SIFTExtractDlg::SIFTExtractDlg(QWidget* parent)
    : QDialog(parent, Qt::Tool), Ui::SIFTExtractDlg() {
    setupUi(this);
}

void SIFTExtractDlg::updateComboBox(const std::vector<std::string>& fields) {
    intensityCombo->clear();
    for (size_t i = 0; i < fields.size(); i++) {
        intensityCombo->addItem(QString(fields[i].c_str()));
    }
}
