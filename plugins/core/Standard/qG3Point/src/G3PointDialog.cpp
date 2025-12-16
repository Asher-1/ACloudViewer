// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "G3PointDialog.h"

#include <QSettings>

#include "ui_G3PointDialog.h"

G3PointDialog::G3PointDialog(QString cloudName, QWidget *parent)
    : QDialog(parent), ui(new Ui::G3PointDialog) {
    ui->setupUi(this);

    setAttribute(Qt::WA_DeleteOnClose, true);
    setWindowFlag(Qt::WindowStaysOnTopHint, true);
    setWindowTitle("G3Point");
    ui->labelCloud->setText(cloudName);

    connect(this->ui->spinBoxkNN, &QSpinBox::editingFinished, this,
            &G3PointDialog::emitKNNChanged);
    connect(this->ui->pushButtonSegment, &QPushButton::clicked, this,
            &G3PointDialog::emitSegment);
    connect(this->ui->pushButtonClusterAndOrClean, &QPushButton::clicked, this,
            &G3PointDialog::emitClusterAndOrClean);
    connect(this->ui->pushButtonGetBorders, &QPushButton::clicked, this,
            &G3PointDialog::emitGetBorders);

    connect(this->ui->radioButtonAll, &QRadioButton::clicked, this,
            &G3PointDialog::emitAllClicked);
    connect(this->ui->radioButtonOnlyOne, &QRadioButton::clicked, this,
            &::G3PointDialog::emitOnlyOneClicked);
    connect(this->ui->spinBoxOnlyOne, qOverload<int>(&QSpinBox::valueChanged),
            this, &::G3PointDialog::emitOnlyOneChanged);

    connect(this->ui->pushButtonFit, &QPushButton::clicked, this,
            &G3PointDialog::emitFit);
    connect(this->ui->pushButtonExportResults, &QPushButton::clicked, this,
            &G3PointDialog::emitExportResults);
    connect(this->ui->pushButtonWolman, &QPushButton::clicked, this,
            &G3PointDialog::emitWolman);
    connect(this->ui->pushButtonAngles, &QPushButton::clicked, this,
            &G3PointDialog::emitAngles);
    connect(this->ui->checkBoxSurfaces, &QCheckBox::clicked, this,
            &G3PointDialog::emitDrawSurfaces);
    connect(this->ui->checkBoxWireframes, &QCheckBox::clicked, this,
            &::G3PointDialog::emitDrawLines);
    connect(this->ui->checkBoxPoints, &QCheckBox::clicked, this,
            &G3PointDialog::emitDrawPoints);
    connect(this->ui->spinBoxGLPointSize,
            qOverload<int>(&QSpinBox::valueChanged), this,
            &::G3PointDialog::emitGLPointSizeChanged);

    connect(this->ui->radioButtonOnlyOne, &QRadioButton::toggled, this,
            &G3PointDialog::enableDrawPointsForOnlyOneGrain);
}

G3PointDialog::~G3PointDialog() { delete ui; }

void G3PointDialog::readSettings() {
    QSettings settings;
    settings.beginGroup("G3Point");

    // kNN
    int kNN = settings.value("kNN", 20).toInt();
    this->ui->spinBoxkNN->setValue(kNN);
    // radius factor
    double radiusFactor = settings.value("radiusFactor", 0.6).toDouble();
    this->ui->doubleSpinBoxRadiusFactor->setValue(radiusFactor);
    // max angle 1
    double maxAngle1 = settings.value("maxAngle1", 60).toDouble();
    this->ui->doubleSpinBoxRadiusFactor->setValue(maxAngle1);
    // wolmanNbIter
    int wolmanNbIter = settings.value("wolmanNbIter", 20).toInt();
    this->ui->spinBoxWolmanNbIter->setValue(wolmanNbIter);
    // anglesNbBins
    int anglesNbBins = settings.value("anglesNbBins", 10).toInt();
    this->ui->spinBoxAnglesNbBins->setValue(anglesNbBins);
}

void G3PointDialog::writeSettings() {
    QSettings settings;
    settings.beginGroup("G3Point");

    // kNN
    settings.setValue("kNN", ui->spinBoxkNN->value());
    // radius factor
    settings.setValue("radiusFactor", ui->doubleSpinBoxRadiusFactor->value());
    // max angle 1
    settings.setValue("maxAngle1", ui->doubleSpinBoxMaxAngle1->value());
    // wolmanNbIter
    settings.setValue("wolmanNbIter", ui->spinBoxWolmanNbIter->value());
    // anglesNbBins
    settings.setValue("anglesNbBins", ui->spinBoxAnglesNbBins->value());
}

double G3PointDialog::getMaxAngle1() {
    return this->ui->doubleSpinBoxMaxAngle1->value();
}

double G3PointDialog::getMaxAngle2() {
    return this->ui->doubleSpinBoxMaxAngle2->value();
}

double G3PointDialog::getMinFlatness() {
    return this->ui->doubleSpinBoxMinFlatness->value();
}

int G3PointDialog::getNMin() { return this->ui->spinBoxNMin->value(); }

int G3PointDialog::getkNN() { return this->ui->spinBoxkNN->value(); }

int G3PointDialog::getWolmanNbIter() {
    return this->ui->spinBoxWolmanNbIter->value();
}

int G3PointDialog::getAnglesNbBins() {
    if (this->ui->radioButtonAutoNbBins->isChecked()) {
        return -1;
    } else {
        return this->ui->spinBoxAnglesNbBins->value();
    }
}

double G3PointDialog::getRadiusFactor() {
    return ui->doubleSpinBoxRadiusFactor->value();
}

bool G3PointDialog::isSteepestSlope() {
    return this->ui->radioButtonSteepestSlope->isChecked();
}

void G3PointDialog::enableClusterAndOrClean(bool state) {
    this->ui->groupBoxClusterAndOrClean->setEnabled(state);
}

bool G3PointDialog::clusterIsChecked() {
    return this->ui->checkBoxClustering->isChecked();
}

bool G3PointDialog::cleanIsChecked() {
    return this->ui->checkBoxCleaning->isChecked();
}

void G3PointDialog::emitSignals() {
    emit allClicked(this->ui->radioButtonAll->isChecked());
    emit onlyOneChanged(this->ui->spinBoxOnlyOne->value());
}

void G3PointDialog::setOnlyOneMax(int max) {
    this->ui->spinBoxOnlyOne->setMaximum(max);
}

void G3PointDialog::enableDrawPointsForOnlyOneGrain(bool state) {
    this->ui->checkBoxPoints->setEnabled(state);
    this->ui->spinBoxGLPointSize->setEnabled(state);
}
