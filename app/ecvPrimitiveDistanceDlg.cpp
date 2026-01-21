// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvPrimitiveDistanceDlg.h"

// Qt
#include <QHeaderView>
#include <QMessageBox>

// System
#include <assert.h>

static bool s_signedDist = true;
static bool s_flipNormals = false;
static bool s_treatAsBounded = false;
ecvPrimitiveDistanceDlg::ecvPrimitiveDistanceDlg(QWidget* parent)
    : QDialog(parent, Qt::Tool), Ui::primitiveDistanceDlg() {
    setupUi(this);

    signedDistCheckBox->setChecked(s_signedDist);
    flipNormalsCheckBox->setEnabled(s_signedDist);
    flipNormalsCheckBox->setChecked(s_flipNormals);
    treatPlanesAsBoundedCheckBox->setUpdatesEnabled(false);
    treatPlanesAsBoundedCheckBox->setChecked(s_treatAsBounded);
    connect(cancelButton, &QPushButton::clicked, this,
            &ecvPrimitiveDistanceDlg::cancelAndExit);
    connect(okButton, &QPushButton::clicked, this,
            &ecvPrimitiveDistanceDlg::applyAndExit);
    connect(signedDistCheckBox, &QCheckBox::toggled, this,
            &ecvPrimitiveDistanceDlg::toggleSigned);
}

void ecvPrimitiveDistanceDlg::applyAndExit() {
    s_signedDist = signedDistances();
    s_flipNormals = flipNormals();
    s_treatAsBounded = treatPlanesAsBounded();
    accept();
}

void ecvPrimitiveDistanceDlg::cancelAndExit() { reject(); }

void ecvPrimitiveDistanceDlg::toggleSigned(bool state) {
    flipNormalsCheckBox->setEnabled(state);
}
