// ##########################################################################
// #                                                                        #
// #                              CLOUDVIEWER                               #
// #                                                                        #
// #  This program is free software; you can redistribute it and/or modify  #
// #  it under the terms of the GNU General Public License as published by  #
// #  the Free Software Foundation; version 2 or later of the License.      #
// #                                                                        #
// #  This program is distributed in the hope that it will be useful,       #
// #  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
// #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
// #  GNU General Public License for more details.                          #
// #                                                                        #
// #          COPYRIGHT: EDF R&D / DAHAI LU                                 #
// #                                                                        #
// ##########################################################################

#include "ecvAlignDlg.h"

#include "MainWindow.h"
#include "ui_alignDlg.h"

// common
#include <ecvQtHelpers.h>

// CV_CORE_LIB
#include <CVPointCloud.h>
#include <CloudSamplingTools.h>
#include <DgmOctree.h>
#include <GeometricalAnalysisTools.h>
#include <ReferenceCloud.h>

// ECV_DB_LIB
#include <ecvDisplayTools.h>
#include <ecvGenericPointCloud.h>
#include <ecvProgressDialog.h>

ccAlignDlg::ccAlignDlg(ccGenericPointCloud* data,
                       ccGenericPointCloud* model,
                       QWidget* parent)
    : QDialog(parent, Qt::Tool), m_ui(new Ui::AlignDialog) {
    m_ui->setupUi(this);

    m_ui->samplingMethod->addItem("None");
    m_ui->samplingMethod->addItem("Random");
    m_ui->samplingMethod->addItem("Space");
    m_ui->samplingMethod->addItem("Octree");
    m_ui->samplingMethod->setCurrentIndex(NONE);

    ccQtHelpers::SetButtonColor(m_ui->dataColorButton, Qt::red);
    ccQtHelpers::SetButtonColor(m_ui->modelColorButton, Qt::yellow);

    dataObject = data;
    modelObject = model;
    setColorsAndLabels();

    changeSamplingMethod(m_ui->samplingMethod->currentIndex());
    toggleNbMaxCandidates(m_ui->isNbCandLimited->isChecked());

    connect(m_ui->swapButton, &QPushButton::clicked, this,
            &ccAlignDlg::swapModelAndData);
    connect(m_ui->modelSample, &QSlider::sliderReleased, this,
            &ccAlignDlg::modelSliderReleased);
    connect(m_ui->dataSample, &QSlider::sliderReleased, this,
            &ccAlignDlg::dataSliderReleased);
    connect(m_ui->deltaEstimation, &QPushButton::clicked, this,
            &ccAlignDlg::estimateDelta);
    connect(m_ui->isNbCandLimited, &QCheckBox::toggled, this,
            &ccAlignDlg::toggleNbMaxCandidates);
    connect(m_ui->samplingMethod,
            static_cast<void (QComboBox::*)(int)>(
                    &QComboBox::currentIndexChanged),
            this, &ccAlignDlg::changeSamplingMethod);
    connect(m_ui->dataSamplingRate,
            static_cast<void (QDoubleSpinBox::*)(double)>(
                    &QDoubleSpinBox::valueChanged),
            this, &ccAlignDlg::dataSamplingRateChanged);
    connect(m_ui->modelSamplingRate,
            static_cast<void (QDoubleSpinBox::*)(double)>(
                    &QDoubleSpinBox::valueChanged),
            this, &ccAlignDlg::modelSamplingRateChanged);
}

ccAlignDlg::~ccAlignDlg() {
    modelObject->enableTempColor(false);
    dataObject->enableTempColor(false);
}

unsigned ccAlignDlg::getNbTries() { return m_ui->nbTries->value(); }

double ccAlignDlg::getOverlap() { return m_ui->overlap->value(); }

double ccAlignDlg::getDelta() { return m_ui->delta->value(); }

ccGenericPointCloud* ccAlignDlg::getModelObject() { return modelObject; }

ccGenericPointCloud* ccAlignDlg::getDataObject() { return dataObject; }

ccAlignDlg::CC_SAMPLING_METHOD ccAlignDlg::getSamplingMethod() {
    return (CC_SAMPLING_METHOD)m_ui->samplingMethod->currentIndex();
}

bool ccAlignDlg::isNumberOfCandidatesLimited() {
    return m_ui->isNbCandLimited->isChecked();
}

unsigned ccAlignDlg::getMaxNumberOfCandidates() {
    return m_ui->nbMaxCandidates->value();
}

cloudViewer::ReferenceCloud* ccAlignDlg::getSampledModel() {
    cloudViewer::ReferenceCloud* sampledCloud = nullptr;

    switch (getSamplingMethod()) {
        case SPACE: {
            cloudViewer::CloudSamplingTools::SFModulationParams modParams(
                    false);
            sampledCloud =
                    cloudViewer::CloudSamplingTools::resampleCloudSpatially(
                            modelObject,
                            static_cast<PointCoordinateType>(
                                    m_ui->modelSamplingRate->value()),
                            modParams);
        } break;
        case OCTREE:
            if (modelObject->getOctree()) {
                sampledCloud = cloudViewer::CloudSamplingTools::
                        subsampleCloudWithOctreeAtLevel(
                                modelObject,
                                static_cast<unsigned char>(
                                        m_ui->modelSamplingRate->value()),
                                cloudViewer::CloudSamplingTools::
                                        NEAREST_POINT_TO_CELL_CENTER,
                                nullptr, modelObject->getOctree().data());
            } else {
                CVLog::Error(
                        "[ccAlignDlg::getSampledModel] Failed to get/compute "
                        "model octree!");
            }
            break;
        case RANDOM: {
            sampledCloud =
                    cloudViewer::CloudSamplingTools::subsampleCloudRandomly(
                            modelObject,
                            static_cast<unsigned>(
                                    m_ui->modelSamplingRate->value()));
        } break;
        default: {
            sampledCloud = new cloudViewer::ReferenceCloud(modelObject);
            if (!sampledCloud->addPointIndex(0, modelObject->size())) {
                delete sampledCloud;
                sampledCloud = nullptr;
                CVLog::Error(
                        "[ccAlignDlg::getSampledModel] Not enough memory!");
            }
        } break;
    }

    return sampledCloud;
}

cloudViewer::ReferenceCloud* ccAlignDlg::getSampledData() {
    cloudViewer::ReferenceCloud* sampledCloud = nullptr;

    switch (getSamplingMethod()) {
        case SPACE: {
            cloudViewer::CloudSamplingTools::SFModulationParams modParams(
                    false);
            sampledCloud =
                    cloudViewer::CloudSamplingTools::resampleCloudSpatially(
                            dataObject,
                            static_cast<PointCoordinateType>(
                                    m_ui->dataSamplingRate->value()),
                            modParams);
        } break;
        case OCTREE:
            if (dataObject->getOctree()) {
                sampledCloud = cloudViewer::CloudSamplingTools::
                        subsampleCloudWithOctreeAtLevel(
                                dataObject,
                                static_cast<unsigned char>(
                                        m_ui->dataSamplingRate->value()),
                                cloudViewer::CloudSamplingTools::
                                        NEAREST_POINT_TO_CELL_CENTER,
                                nullptr, dataObject->getOctree().data());
            } else {
                CVLog::Error(
                        "[ccAlignDlg::getSampledData] Failed to get/compute "
                        "data octree!");
            }
            break;
        case RANDOM: {
            sampledCloud =
                    cloudViewer::CloudSamplingTools::subsampleCloudRandomly(
                            dataObject,
                            (unsigned)(m_ui->dataSamplingRate->value()));
        } break;
        default: {
            sampledCloud = new cloudViewer::ReferenceCloud(dataObject);
            if (!sampledCloud->addPointIndex(0, dataObject->size())) {
                delete sampledCloud;
                sampledCloud = nullptr;
                CVLog::Error("[ccAlignDlg::getSampledData] Not enough memory!");
            }
        } break;
    }

    return sampledCloud;
}

void ccAlignDlg::setColorsAndLabels() {
    if (!modelObject || !dataObject) return;

    m_ui->modelCloud->setText(modelObject->getName());
    modelObject->setVisible(true);
    modelObject->setTempColor(ecvColor::red);

    m_ui->dataCloud->setText(dataObject->getName());
    dataObject->setVisible(true);
    dataObject->setTempColor(ecvColor::yellow);

    ecvDisplayTools::RedrawDisplay(false);
}

// SLOTS
void ccAlignDlg::swapModelAndData() {
    std::swap(dataObject, modelObject);
    setColorsAndLabels();
    changeSamplingMethod(m_ui->samplingMethod->currentIndex());
}

void ccAlignDlg::modelSliderReleased() {
    double rate = static_cast<double>(m_ui->modelSample->sliderPosition()) /
                  m_ui->modelSample->maximum();
    if (getSamplingMethod() == SPACE) rate = 1.0 - rate;
    rate *= m_ui->modelSamplingRate->maximum();
    m_ui->modelSamplingRate->setValue(rate);
    modelSamplingRateChanged(rate);
}

void ccAlignDlg::dataSliderReleased() {
    double rate = static_cast<double>(m_ui->dataSample->sliderPosition()) /
                  m_ui->dataSample->maximum();
    if (getSamplingMethod() == SPACE) rate = 1.0 - rate;
    rate *= m_ui->dataSamplingRate->maximum();
    m_ui->dataSamplingRate->setValue(rate);
    dataSamplingRateChanged(rate);
}

void ccAlignDlg::modelSamplingRateChanged(double value) {
    QString message("An error occurred");

    CC_SAMPLING_METHOD method = getSamplingMethod();
    float rate = static_cast<float>(m_ui->modelSamplingRate->value()) /
                 m_ui->modelSamplingRate->maximum();
    if (method == SPACE) rate = 1.0f - rate;
    m_ui->modelSample->setSliderPosition(
            static_cast<int>(rate * m_ui->modelSample->maximum()));

    switch (method) {
        case SPACE: {
            cloudViewer::ReferenceCloud* tmpCloud =
                    getSampledModel();  // DGM FIXME: wow! you generate a
                                        // spatially sampled cloud just to
                                        // display its size?!
            if (tmpCloud) {
                message = QString("distance units (%1 remaining points)")
                                  .arg(tmpCloud->size());
                delete tmpCloud;
            }
        } break;
        case RANDOM: {
            message = QString("remaining points (%1%)")
                              .arg(rate * 100.0f, 0, 'f', 1);
        } break;
        case OCTREE: {
            cloudViewer::ReferenceCloud* tmpCloud =
                    getSampledModel();  // DGM FIXME: wow! you generate a
                                        // spatially sampled cloud just to
                                        // display its size?!
            if (tmpCloud) {
                message = QString("%1 remaining points").arg(tmpCloud->size());
                delete tmpCloud;
            }
        } break;
        default: {
            unsigned remaining =
                    static_cast<unsigned>(rate * modelObject->size());
            message = QString("%1 remaining points").arg(remaining);
        } break;
    }

    m_ui->modelRemaining->setText(message);
}

void ccAlignDlg::dataSamplingRateChanged(double value) {
    QString message("An error occurred");

    CC_SAMPLING_METHOD method = getSamplingMethod();
    double rate = static_cast<float>(m_ui->dataSamplingRate->value() /
                                     m_ui->dataSamplingRate->maximum());
    if (method == SPACE) rate = 1.0 - rate;
    m_ui->dataSample->setSliderPosition(
            static_cast<int>(rate * m_ui->dataSample->maximum()));

    switch (method) {
        case SPACE: {
            cloudViewer::ReferenceCloud* tmpCloud =
                    getSampledData();  // DGM FIXME: wow! you generate a
                                       // spatially sampled cloud just to
                                       // display its size?!
            if (tmpCloud) {
                message = QString("distance units (%1 remaining points)")
                                  .arg(tmpCloud->size());
                delete tmpCloud;
            }
        } break;
        case RANDOM: {
            message = QString("remaining points (%1%)")
                              .arg(rate * 100.0, 0, 'f', 1);
        } break;
        case OCTREE: {
            cloudViewer::ReferenceCloud* tmpCloud =
                    getSampledData();  // DGM FIXME: wow! you generate a
                                       // spatially sampled cloud just to
                                       // display its size?!
            if (tmpCloud) {
                message = QString("%1 remaining points").arg(tmpCloud->size());
                delete tmpCloud;
            }
        } break;
        default: {
            unsigned remaining =
                    static_cast<unsigned>(rate * dataObject->size());
            message = QString("%1 remaining points").arg(remaining);
        } break;
    }

    m_ui->dataRemaining->setText(message);
}

void ccAlignDlg::estimateDelta() {
    ecvProgressDialog pDlg(false, this);

    cloudViewer::ReferenceCloud* sampledData = getSampledData();

    // we have to work on a copy of the cloud in order to prevent the algorithms
    // from modifying the original cloud.
    cloudViewer::PointCloud cloud;
    {
        cloud.reserve(sampledData->size());
        for (unsigned i = 0; i < sampledData->size(); i++)
            cloud.addPoint(*sampledData->getPoint(i));
        cloud.enableScalarField();
    }

    if (cloudViewer::GeometricalAnalysisTools::ComputeLocalDensityApprox(
                &cloud, cloudViewer::GeometricalAnalysisTools::DENSITY_KNN,
                &pDlg) != cloudViewer::GeometricalAnalysisTools::NoError) {
        CVLog::Error("Failed to compute approx. density");
        return;
    }
    unsigned count = 0;
    double meanDensity = 0;
    double meanSqrDensity = 0;
    for (unsigned i = 0; i < cloud.size(); i++) {
        ScalarType value = cloud.getPointScalarValue(i);
        if (value == value) {
            meanDensity += value;
            meanSqrDensity += static_cast<double>(value) * value;
            count++;
        }
    }

    if (count) {
        meanDensity /= count;
        meanSqrDensity /= count;
    }
    double dev = meanSqrDensity - (meanDensity * meanDensity);

    m_ui->delta->setValue(meanDensity + dev);
    delete sampledData;
}

void ccAlignDlg::changeSamplingMethod(int index) {
    // Reste a changer les textes d'aide
    switch (index) {
        case SPACE: {
            // model
            {
                m_ui->modelSamplingRate->setDecimals(4);
                int oldSliderPos = m_ui->modelSample->sliderPosition();
                CCVector3 bbMin;
                CCVector3 bbMax;
                modelObject->getBoundingBox(bbMin, bbMax);
                double dist = (bbMin - bbMax).norm();
                m_ui->modelSamplingRate->setMaximum(dist);
                m_ui->modelSample->setSliderPosition(oldSliderPos);
                m_ui->modelSamplingRate->setSingleStep(0.01);
                m_ui->modelSamplingRate->setMinimum(0.);
            }
            // data
            {
                m_ui->dataSamplingRate->setDecimals(4);
                int oldSliderPos = m_ui->dataSample->sliderPosition();
                CCVector3 bbMin;
                CCVector3 bbMax;
                dataObject->getBoundingBox(bbMin, bbMax);
                double dist = (bbMin - bbMax).norm();
                m_ui->dataSamplingRate->setMaximum(dist);
                m_ui->dataSample->setSliderPosition(oldSliderPos);
                m_ui->dataSamplingRate->setSingleStep(0.01);
                m_ui->dataSamplingRate->setMinimum(0.);
            }
        } break;
        case RANDOM: {
            // model
            {
                m_ui->modelSamplingRate->setDecimals(0);
                m_ui->modelSamplingRate->setMaximum(
                        static_cast<float>(modelObject->size()));
                m_ui->modelSamplingRate->setSingleStep(1.);
                m_ui->modelSamplingRate->setMinimum(0.);
            }
            // data
            {
                m_ui->dataSamplingRate->setDecimals(0);
                m_ui->dataSamplingRate->setMaximum(
                        static_cast<float>(dataObject->size()));
                m_ui->dataSamplingRate->setSingleStep(1.);
                m_ui->dataSamplingRate->setMinimum(0.);
            }
        } break;
        case OCTREE: {
            // model
            {
                if (!modelObject->getOctree()) modelObject->computeOctree();
                m_ui->modelSamplingRate->setDecimals(0);
                m_ui->modelSamplingRate->setMaximum(static_cast<double>(
                        cloudViewer::DgmOctree::MAX_OCTREE_LEVEL));
                m_ui->modelSamplingRate->setMinimum(1.);
                m_ui->modelSamplingRate->setSingleStep(1.);
            }
            // data
            {
                if (!dataObject->getOctree()) dataObject->computeOctree();
                m_ui->dataSamplingRate->setDecimals(0);
                m_ui->dataSamplingRate->setMaximum(static_cast<double>(
                        cloudViewer::DgmOctree::MAX_OCTREE_LEVEL));
                m_ui->dataSamplingRate->setMinimum(1.);
                m_ui->dataSamplingRate->setSingleStep(1.);
            }
        } break;
        default: {
            // model
            {
                m_ui->modelSamplingRate->setDecimals(2);
                m_ui->modelSamplingRate->setMaximum(100.);
                m_ui->modelSamplingRate->setSingleStep(0.01);
                m_ui->modelSamplingRate->setMinimum(0.);
            }
            // data
            {
                m_ui->dataSamplingRate->setDecimals(2);
                m_ui->dataSamplingRate->setMaximum(100.);
                m_ui->dataSamplingRate->setSingleStep(0.01);
                m_ui->dataSamplingRate->setMinimum(0.);
            }
        } break;
    }

    if (index == NONE) {
        // model
        m_ui->modelSample->setSliderPosition(m_ui->modelSample->maximum());
        m_ui->modelSample->setEnabled(false);
        m_ui->modelSamplingRate->setEnabled(false);
        // data
        m_ui->dataSample->setSliderPosition(m_ui->dataSample->maximum());
        m_ui->dataSample->setEnabled(false);
        m_ui->dataSamplingRate->setEnabled(false);
    } else {
        // model
        m_ui->modelSample->setEnabled(true);
        m_ui->modelSamplingRate->setEnabled(true);
        // data
        m_ui->dataSample->setEnabled(true);
        m_ui->dataSamplingRate->setEnabled(true);
    }

    modelSliderReleased();
    dataSliderReleased();
}

void ccAlignDlg::toggleNbMaxCandidates(bool activ) {
    m_ui->nbMaxCandidates->setEnabled(activ);
}
