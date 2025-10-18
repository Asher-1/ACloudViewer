// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "profileImportDlg.h"

// Qt
#include <QFileDialog>

// System
#include <assert.h>

static double s_joints = 4;  // Estimated mortar joints width in cm

static double s_horizontal = 3;  // Segmentation window in metres (X axis)
static double s_vertical = 2;    // Segmentation window in metres (Z axis)

ProfileImportDlg::ProfileImportDlg(QWidget* parent)
    : QDialog(parent, Qt::Tool), Ui::ProfileImportDlg() {
    setupUi(this);

    connect(buttonBox, SIGNAL(accepted()), this, SLOT(saveSettings));
    jointsSpinBox->setValue(s_joints);
    segmentHSpinBox->setValue(s_horizontal);
    segmentVSpinBox->setValue(s_vertical);
    alignmentCheckBox->setChecked(true);
    alignmentCheckBox->setDisabled(true);
    this->setWindowTitle("Automatic Segmentation plugin");
}

void ProfileImportDlg::saveSettings() {
    s_joints = jointsSpinBox->value();
    s_horizontal = segmentHSpinBox->value();
    s_vertical = segmentVSpinBox->value();
}
