// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ccCSFDlg.h"

// ECV_DB_LIB
#include <ecvOctree.h>

static int MaxIteration = 500;
static double cloth_resolution = 1.5;
static double class_threshold = 0.5;
ccCSFDlg::ccCSFDlg(QWidget* parent) : QDialog(parent), Ui::CSFDialog() {
    setupUi(this);

    connect(buttonBox, &QDialogButtonBox::accepted, this,
            &ccCSFDlg::saveSettings);

    setWindowFlags(Qt::Tool /*Qt::Dialog | Qt::WindowStaysOnTopHint*/);

    MaxIterationSpinBox->setValue(MaxIteration);
    cloth_resolutionSpinBox->setValue(cloth_resolution);
    class_thresholdSpinBox->setValue(class_threshold);
}

void ccCSFDlg::saveSettings() {
    MaxIteration = MaxIterationSpinBox->value();
    cloth_resolution = cloth_resolutionSpinBox->value();
    class_threshold = class_thresholdSpinBox->value();
}