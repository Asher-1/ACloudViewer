// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "SACSegmentationDlg.h"

SACSegmentationDlg::SACSegmentationDlg(QWidget* parent)
    : QDialog(parent, Qt::Tool), Ui::SACSegmentationDlg() {
    setupUi(this);
    initParameters();
    connect(modelTypeCombo,
            static_cast<void (QComboBox::*)(int)>(
                    &QComboBox::currentIndexChanged),
            this,
            static_cast<void (SACSegmentationDlg::*)(int)>(
                    &SACSegmentationDlg::modelsChanged));
}

void SACSegmentationDlg::modelsChanged(int currentIndex) {
    switch (modelTypeCombo->currentIndex()) {
        case 5:   // SACMODEL_CYLINDER
        case 6:   // SACMODEL_CONE
        case 11:  // SACMODEL_NORMAL_PLANE
        case 12:  // SACMODEL_NORMAL_SPHERE
        case 16:  // SACMODEL_NORMAL_PARALLEL_PLANE
        {
            normalDisWeightLabel->setEnabled(true);
            normalDisWeightSpinBox->setEnabled(true);
        } break;
        default: {
            normalDisWeightLabel->setEnabled(false);
            normalDisWeightSpinBox->setEnabled(false);
        } break;
    }
}
void SACSegmentationDlg::updateModelTypeComboBox(const QStringList& fields) {
    modelTypeCombo->clear();
    for (int i = 0; i < fields.size(); i++) {
        modelTypeCombo->addItem(fields[i], i);
        modelTypeCombo->setItemText(i, fields[i]);
    }
}

void SACSegmentationDlg::updateMethodTypeComboBox(const QStringList& fields) {
    methodTypeCombo->clear();
    for (int i = 0; i < fields.size(); i++) {
        methodTypeCombo->addItem(fields[i], i);
        methodTypeCombo->setItemText(i, fields[i]);
    }
}

void SACSegmentationDlg::initParameters() {
    QStringList methodFields;
    QStringList modelFields;
    if (modelFields.isEmpty()) {
        modelFields << tr("SACMODEL_PLANE") << tr("SACMODEL_LINE")
                    << tr("SACMODEL_CIRCLE2D") << tr("SACMODEL_CIRCLE3D")
                    << tr("SACMODEL_SPHERE") << tr("SACMODEL_CYLINDER")
                    << tr("SACMODEL_CONE") << tr("SACMODEL_TORUS")
                    << tr("SACMODEL_PARALLEL_LINE")
                    << tr("SACMODEL_PERPENDICULAR_PLANE")
                    << tr("SACMODEL_PARALLEL_LINES")
                    << tr("SACMODEL_NORMAL_PLANE")
                    << tr("SACMODEL_NORMAL_SPHERE")
                    << tr("SACMODEL_REGISTRATION")
                    << tr("SACMODEL_REGISTRATION_2D")
                    << tr("SACMODEL_PARALLEL_PLANE")
                    << tr("SACMODEL_NORMAL_PARALLEL_PLANE")
                    << tr("SACMODEL_STICK");
    }
    if (methodFields.isEmpty()) {
        methodFields << tr("SAC_RANSAC") << tr("SAC_LMEDS") << tr("SAC_MSAC")
                     << tr("SAC_RRANSAC") << tr("SAC_RMSAC") << tr("SAC_MLESAC")
                     << tr("SAC_PROSAC");
    }

    // update the combo box
    updateModelTypeComboBox(modelFields);
    updateMethodTypeComboBox(methodFields);
}
