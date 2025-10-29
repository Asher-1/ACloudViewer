// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvLabelingDlg.h"

// CV_CORE_LIB
#include <DgmOctree.h>

ccLabelingDlg::ccLabelingDlg(QWidget* parent /*=0*/)
    : QDialog(parent, Qt::Tool), Ui::LabelingDialog() {
    setupUi(this);

    octreeLevelSpinBox->setMaximum(cloudViewer::DgmOctree::MAX_OCTREE_LEVEL);
}

int ccLabelingDlg::getOctreeLevel() { return octreeLevelSpinBox->value(); }

int ccLabelingDlg::getMinPointsNb() { return minPtsSpinBox->value(); }

bool ccLabelingDlg::randomColors() {
    return (randomColorsCheckBox->checkState() == Qt::Checked);
}
