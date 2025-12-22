// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "CorrespondenceMatchingDialog.h"

// ECV_DB_LIB
#include <ecvMainAppInterface.h>
#include <ecvPointCloud.h>

// Qt
#include <QApplication>
#include <QComboBox>
#include <QMainWindow>
#include <QPushButton>
#include <QSettings>
#include <QThread>

// Qt5/Qt6 Compatibility
#include <QtCompat.h>

// system
#include <limits>

CorrespondenceMatchingDialog::CorrespondenceMatchingDialog(
        ecvMainAppInterface* app)
    : QDialog(app ? app->getActiveWindow() : 0),
      Ui::CorrespondenceMatchingDialog(),
      m_app(app) {
    setupUi(this);

    int maxThreadCount = QThread::idealThreadCount();
    maxThreadCountSpinBox->setRange(1, maxThreadCount);
    maxThreadCountSpinBox->setSuffix(QString("/%1").arg(maxThreadCount));
    maxThreadCountSpinBox->setValue(maxThreadCount);

    loadParamsFromPersistentSettings();

    connect(model1CloudComboBox, SIGNAL(currentIndexChanged(int)), this,
            SLOT(onCloudChanged(int)));
    connect(model2CloudComboBox, SIGNAL(currentIndexChanged(int)), this,
            SLOT(onCloudChanged(int)));

    onCloudChanged(0);
}

void CorrespondenceMatchingDialog::refreshCloudComboBox() {
    if (m_app) {
        // add list of clouds to the combo-boxes
        ccHObject::Container clouds;
        if (m_app->dbRootObject())
            m_app->dbRootObject()->filterChildren(clouds, true,
                                                  CV_TYPES::POINT_CLOUD);

        unsigned cloudCount = 0;
        model1CloudComboBox->clear();
        model2CloudComboBox->clear();
        evaluationCloudComboBox->clear();
        for (size_t i = 0; i < clouds.size(); ++i) {
            if (clouds[i]->isA(CV_TYPES::POINT_CLOUD))  // as filterChildren
                                                        // only test 'isKindOf'
            {
                QString name = getEntityName(clouds[i]);
                QVariant uniqueID(clouds[i]->getUniqueID());
                model1CloudComboBox->addItem(name, uniqueID);
                model2CloudComboBox->addItem(name, uniqueID);
                evaluationCloudComboBox->addItem(name, uniqueID);
                ++cloudCount;
            }
        }

        // if 3 clouds are loaded, then there's chances that the first one is
        // the global cloud!
        model1CloudComboBox->setCurrentIndex(
                cloudCount > 0 ? (cloudCount > 2 ? 1 : 0) : -1);
        model2CloudComboBox->setCurrentIndex(
                cloudCount > 1 ? (cloudCount > 2 ? 2 : 1) : -1);

        if (cloudCount < 1 && m_app)
            m_app->dispToConsole(
                    tr("You need at least 1 loaded clouds to perform matching"),
                    ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
    }
}

bool CorrespondenceMatchingDialog::validParameters() const {
    int c1 = model1CloudComboBox->currentIndex();
    if (model1checkBox->isChecked()) {
        if (c1 < 0) {
            return false;
        }
    }
    int c2 = model2CloudComboBox->currentIndex();
    if (model2checkBox->isChecked()) {
        if (c2 < 0) {
            return false;
        }
    }

    if (model1checkBox->isChecked() && model2checkBox->isChecked()) {
        if (c1 == c2) return false;
    }

    return true;
}

int CorrespondenceMatchingDialog::getMaxThreadCount() const {
    return maxThreadCountSpinBox->value();
}

bool CorrespondenceMatchingDialog::getVerificationFlag() const {
    return verificationCheckBox->isChecked();
}

float CorrespondenceMatchingDialog::getModelSearchRadius() const {
    return static_cast<float>(modelSearchRadiusSpinBox->value());
}

float CorrespondenceMatchingDialog::getSceneSearchRadius() const {
    return static_cast<float>(sceneSearchRadiusSpinBox->value());
}

float CorrespondenceMatchingDialog::getShotDescriptorRadius() const {
    return static_cast<float>(shotDescriptorRadiusSpinBox->value());
}

float CorrespondenceMatchingDialog::getNormalKSearch() const {
    return static_cast<float>(normalKSearchSpinBox->value());
}

bool CorrespondenceMatchingDialog::isGCActivated() const {
    return GCcheckBox->isChecked() ? true : false;
}

float CorrespondenceMatchingDialog::getGcConsensusSetResolution() const {
    return static_cast<float>(gcResolutionSpinBox->value());
}

float CorrespondenceMatchingDialog::getGcMinClusterSize() const {
    return static_cast<float>(gcMinClusterSizeSpinBox->value());
}

float CorrespondenceMatchingDialog::getHoughLRFRadius() const {
    return static_cast<float>(LRFRadiusSpinBox->value());
}

float CorrespondenceMatchingDialog::getHoughBinSize() const {
    return static_cast<float>(houghBinSizeSpinBox->value());
}

float CorrespondenceMatchingDialog::getHoughThreshold() const {
    return static_cast<float>(houghThresholdSpinBox->value());
}

float CorrespondenceMatchingDialog::getVoxelGridLeafSize() const {
    if (useVoxelGridCheckBox->isChecked()) {
        return static_cast<float>(leafSizeSpinBox->value());
    } else {
        return -1.0f;
    }
}

bool CorrespondenceMatchingDialog::getScales(std::vector<float>& scales) const {
    scales.clear();

    try {
        if (scalesRampRadioButton->isChecked()) {
            double maxScale = maxScaleDoubleSpinBox->value();
            double step = stepScaleDoubleSpinBox->value();
            double minScale = minScaleDoubleSpinBox->value();
            if (maxScale < minScale || maxScale < 0 || step < 1.0e-6)
                return false;
            unsigned stepCount =
                    static_cast<unsigned>(
                            floor((maxScale - minScale) / step + 1.0e-6)) +
                    1;
            scales.resize(stepCount);
            for (unsigned i = 0; i < stepCount; ++i)
                scales[i] = static_cast<float>(maxScale - i * step);
        } else if (scalesListRadioButton->isChecked()) {
            QStringList scaleList = scalesListLineEdit->text().split(
                    ' ', QtCompat::SkipEmptyParts);

            int listSize = scaleList.size();
            scales.resize(listSize);
            for (int i = 0; i < listSize; ++i) {
                bool ok = false;
                float f;
                f = scaleList[i].toFloat(&ok);
                if (!ok) return false;
                scales[i] = f;
            }
        } else {
            return false;
        }
    } catch (const std::bad_alloc&) {
        return false;
    }

    return true;
}

void CorrespondenceMatchingDialog::onCloudChanged(int dummy) {
    buttonBox->button(QDialogButtonBox::Ok)->setEnabled(validParameters());
}

ccPointCloud* CorrespondenceMatchingDialog::getModel1Cloud() {
    // return the cloud currently selected in the combox box
    if (model1checkBox->isChecked()) {
        return getCloudFromCombo(model1CloudComboBox, m_app->dbRootObject());
    } else {
        return nullptr;
    }
}

ccPointCloud* CorrespondenceMatchingDialog::getModel2Cloud() {
    // return the cloud currently selected in the combox box
    if (model2checkBox->isChecked()) {
        return getCloudFromCombo(model2CloudComboBox, m_app->dbRootObject());
    } else {
        return nullptr;
    }
}

ccPointCloud* CorrespondenceMatchingDialog::getModelCloudByIndex(int index) {
    switch (index) {
        case 1:
            return getModel1Cloud();
            break;
        case 2:
            return getModel2Cloud();
            break;
        default:
            return nullptr;
            break;
    }
}

ccPointCloud* CorrespondenceMatchingDialog::getEvaluationCloud() {
    // return the cloud currently selected in the combox box
    return getCloudFromCombo(evaluationCloudComboBox, m_app->dbRootObject());
}

void CorrespondenceMatchingDialog::loadParamsFromPersistentSettings() {
    QSettings settings("templateAlignment");
    settings.beginGroup("Align");

    // read out parameters
    // double minScale =
    // settings.value("MinScale",minScaleDoubleSpinBox->value()).toDouble();
    // double step =
    // settings.value("Step",stepScaleDoubleSpinBox->value()).toDouble(); double
    // maxScale =
    // settings.value("MaxScale",maxScaleDoubleSpinBox->value()).toDouble();
    // QString scalesList =
    // settings.value("ScalesList",scalesListLineEdit->text()).toString(); bool
    // scalesRampEnabled =
    // settings.value("ScalesRampEnabled",scalesRampRadioButton->isChecked()).toBool();

    // unsigned maxPoints =
    // settings.value("MaxPoints",maxPointsSpinBox->value()).toUInt(); int
    // classifParam =
    // settings.value("ClassifParam",paramComboBox->currentIndex()).toInt(); int
    // maxThreadCount = settings.value("MaxThreadCount",
    // maxThreadCountSpinBox->maximum()).toInt();

    ////apply parameters

    // minScaleDoubleSpinBox->setValue(minScale);
    // stepScaleDoubleSpinBox->setValue(step);
    // maxScaleDoubleSpinBox->setValue(maxScale);
    // scalesListLineEdit->setText(scalesList);
    // if (scalesRampEnabled)
    //	scalesRampRadioButton->setChecked(true);
    // else
    //	scalesListRadioButton->setChecked(true);

    // maxPointsSpinBox->setValue(maxPoints);
    // paramComboBox->setCurrentIndex(classifParam);
    // maxThreadCountSpinBox->setValue(maxThreadCount);
}

void CorrespondenceMatchingDialog::saveParamsToPersistentSettings() {
    QSettings settings("templateAlignment");
    settings.beginGroup("Align");

    // save parameters
    // settings.setValue("MinScale", minScaleDoubleSpinBox->value());
    // settings.setValue("Step", stepScaleDoubleSpinBox->value());
    // settings.setValue("MaxScale", maxScaleDoubleSpinBox->value());
    // settings.setValue("ScalesList", scalesListLineEdit->text());
    // settings.setValue("ScalesRampEnabled",
    // scalesRampRadioButton->isChecked());

    // settings.setValue("MaxPoints", maxPointsSpinBox->value());
    // settings.setValue("ClassifParam", paramComboBox->currentIndex());
    // settings.setValue("MaxThreadCount", maxThreadCountSpinBox->value());
}

QString CorrespondenceMatchingDialog::getEntityName(ccHObject* obj) {
    if (!obj) {
        assert(false);
        return QString();
    }

    QString name = obj->getName();
    if (name.isEmpty()) name = tr("unnamed");
    name += QString(" [ID %1]").arg(obj->getUniqueID());

    return name;
}

ccPointCloud* CorrespondenceMatchingDialog::getCloudFromCombo(
        QComboBox* comboBox, ccHObject* dbRoot) {
    assert(comboBox && dbRoot);
    if (!comboBox || !dbRoot) {
        assert(false);
        return nullptr;
    }

    // return the cloud currently selected in the combox box
    int index = comboBox->currentIndex();
    if (index < 0) {
        assert(false);
        return nullptr;
    }
    unsigned uniqueID = comboBox->itemData(index).toUInt();
    ccHObject* item = dbRoot->find(uniqueID);
    if (!item || !item->isA(CV_TYPES::POINT_CLOUD)) {
        assert(false);
        return nullptr;
    }
    return static_cast<ccPointCloud*>(item);
}
