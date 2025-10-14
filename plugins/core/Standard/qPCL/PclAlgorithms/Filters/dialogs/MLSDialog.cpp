// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "MLSDialog.h"

#include "../MLSSmoothingUpsampling.h"

// PCL
#include "PclUtils/PCLModules.h"

using namespace PCLModules;

// Qt
#include <QVariant>

MLSDialog::MLSDialog(QWidget *parent) : QDialog(parent), Ui::MLSDialog() {
    setupUi(this);

    updateCombo();

    connect(this->upsampling_method, SIGNAL(currentIndexChanged(QString)), this,
            SLOT(activateMenu(QString)));
    connect(this->search_radius, SIGNAL(valueChanged(double)), this,
            SLOT(updateSquaredGaussian(double)));

    deactivateAllMethods();
}

void MLSDialog::updateCombo() {
    this->upsampling_method->clear();
    this->upsampling_method->addItem(tr("None"), QVariant(MLSParameters::NONE));
    this->upsampling_method->addItem(
            tr("Sample Local Plane"),
            QVariant(MLSParameters::SAMPLE_LOCAL_PLANE));
    this->upsampling_method->addItem(
            tr("Random Uniform Density"),
            QVariant(MLSParameters::RANDOM_UNIFORM_DENSITY));
    this->upsampling_method->addItem(
            tr("Voxel Grid Dilation"),
            QVariant(MLSParameters::VOXEL_GRID_DILATION));
}

void MLSDialog::activateMenu(QString name) {
    deactivateAllMethods();

    if (name == tr("Sample Local Plane")) {
        this->sample_local_plane_method->setEnabled(true);
    } else if (name == tr("Random Uniform Density")) {
        this->random_uniform_density_method->setEnabled(true);
    } else if (name == tr("Voxel Grid Dilation")) {
        this->voxel_grid_dilation_method->setEnabled(true);
    } else {
        deactivateAllMethods();
    }
}

void MLSDialog::deactivateAllMethods() {
    this->sample_local_plane_method->setEnabled(false);
    this->random_uniform_density_method->setEnabled(false);
    this->voxel_grid_dilation_method->setEnabled(false);
}

void MLSDialog::toggleMethods(bool status) {
    if (!status) deactivateAllMethods();
}

void MLSDialog::updateSquaredGaussian(double radius) {
    this->squared_gaussian_parameter->setValue(radius * radius);
}
