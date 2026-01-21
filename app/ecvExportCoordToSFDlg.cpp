// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvExportCoordToSFDlg.h"

ccExportCoordToSFDlg::ccExportCoordToSFDlg(QWidget* parent /*=0*/)
    : QDialog(parent, Qt::Tool), Ui::ExportCoordToSFDlg() {
    setupUi(this);
}

bool ccExportCoordToSFDlg::exportX() const { return xCheckBox->isChecked(); }

bool ccExportCoordToSFDlg::exportY() const { return yCheckBox->isChecked(); }

bool ccExportCoordToSFDlg::exportZ() const { return zCheckBox->isChecked(); }
