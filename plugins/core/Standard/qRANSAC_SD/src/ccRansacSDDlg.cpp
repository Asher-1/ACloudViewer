// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ccRansacSDDlg.h"

#include <ecvOctree.h>

static int s_minSupport =
        500;  // this is the minimal numer of points required for a primitive
static double s_maxNormalDev_deg =
        25.0;  // maximal normal deviation from ideal shape (in degrees)
static double s_probability = 0.01;  // probability that no better candidate was
                                     // overlooked during sampling

ccRansacSDDlg::ccRansacSDDlg(QWidget* parent)
    : QDialog(parent, Qt::Tool), Ui::RansacSDDialog() {
    setupUi(this);

    connect(buttonBox, &QDialogButtonBox::accepted, this,
            &ccRansacSDDlg::saveSettings);

    supportPointsSpinBox->setValue(s_minSupport);
    maxNormDevAngleSpinBox->setValue(s_maxNormalDev_deg);
    probaDoubleSpinBox->setValue(s_probability);
}

void ccRansacSDDlg::saveSettings() {
    s_minSupport = supportPointsSpinBox->value();
    s_maxNormalDev_deg = maxNormDevAngleSpinBox->value();
    s_probability = probaDoubleSpinBox->value();
}
