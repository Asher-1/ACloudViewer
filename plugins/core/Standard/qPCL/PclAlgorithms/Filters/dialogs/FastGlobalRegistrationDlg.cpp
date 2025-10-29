// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "FastGlobalRegistrationDlg.h"

// common
#include <ecvQtHelpers.h>

// qCC_db
#include <ecvDisplayTools.h>
#include <ecvOctree.h>
#include <ecvPointCloud.h>

// system
#include <assert.h>

static double s_featureRadius = 0;

FastGlobalRegistrationDialog::FastGlobalRegistrationDialog(
        const std::vector<ccPointCloud*>& allClouds,
        QWidget* parent /*=nullptr*/)
    : QDialog(parent, Qt::Tool),
      Ui::FastGlobalRegistrationDialog(),
      clouds(allClouds),
      referencesCloudUinqueID(ccUniqueIDGenerator::InvalidUniqueID) {
    setupUi(this);

    ccQtHelpers::SetButtonColor(dataColorButton, Qt::red);
    ccQtHelpers::SetButtonColor(modelColorButton, Qt::yellow);

    for (ccPointCloud* cloud : clouds) {
        referenceComboBox->addItem(cloud->getName(), cloud->getUniqueID());
    }

    if (!clouds.empty()) {
        referencesCloudUinqueID = clouds.front()->getUniqueID();
    }

    updateGUI();

    // restore semi-persistent settings
    {
        // semi-persistent options
        double previousRadius = s_featureRadius;
        if (previousRadius == 0) {
            for (ccPointCloud* cloud : clouds) {
                double radius = ccOctree::GuessNaiveRadius(cloud);
                previousRadius = std::max(radius, previousRadius);
            }
        }

        featureRadiusDoubleSpinBox->setValue(previousRadius);
    }

    connect(autoRadiusToolButton, &QToolButton::clicked, this,
            &FastGlobalRegistrationDialog::autoEstimateRadius);
    connect(referenceComboBox, qOverload<int>(&QComboBox::currentIndexChanged),
            this, &FastGlobalRegistrationDialog::referenceEntityChanged);
}

FastGlobalRegistrationDialog::~FastGlobalRegistrationDialog() {
    ecvDisplayTools::SetRedrawRecursive(false);
    for (ccPointCloud* cloud : clouds) {
        cloud->enableTempColor(false);
        cloud->setForceRedraw(true);
    }

    ecvDisplayTools::RedrawDisplay();
}

void FastGlobalRegistrationDialog::saveParameters() const {
    s_featureRadius = getFeatureRadius();
}

ccPointCloud* FastGlobalRegistrationDialog::getReferenceCloud() {
    for (ccPointCloud* cloud : clouds) {
        if (cloud->getUniqueID() == referencesCloudUinqueID) return cloud;
    }

    return nullptr;
}

double FastGlobalRegistrationDialog::getFeatureRadius() const {
    return featureRadiusDoubleSpinBox->value();
}

void FastGlobalRegistrationDialog::updateGUI() {
    if (clouds.size() < 2) return;

    ccPointCloud* referenceCloud = getReferenceCloud();
    if (!referenceCloud) {
        assert(false);
        return;
    }

    // aligned cloud(s)
    ccPointCloud* alignedCloud = nullptr;  // only one of them
    int referenceCloudIndex = -1;
    ecvDisplayTools::SetRedrawRecursive(false);
    for (size_t i = 0; i < clouds.size(); ++i) {
        ccPointCloud* cloud = clouds[i];
        if (cloud->getUniqueID() != referencesCloudUinqueID) {
            alignedCloud = cloud;
            alignedCloud->setVisible(true);
            alignedCloud->setTempColor(ecvColor::red);
            alignedCloud->setForceRedraw(true);
        } else {
            referenceCloudIndex = static_cast<int>(i);
        }
    }
    alignedLineEdit->setText(
            clouds.size() == 2 ? alignedCloud->getName()
                               : tr("%1 other clouds").arg(clouds.size() - 1));

    // reference cloud
    referenceCloud->setVisible(true);
    referenceCloud->setTempColor(ecvColor::yellow);
    referenceCloud->setForceRedraw(true);

    referenceComboBox->blockSignals(true);
    referenceComboBox->setCurrentIndex(referenceCloudIndex);
    referenceComboBox->blockSignals(false);

    ecvDisplayTools::RedrawDisplay();
}

void FastGlobalRegistrationDialog::referenceEntityChanged(int index) {
    referencesCloudUinqueID = referenceComboBox->itemData(index).toUInt();

    updateGUI();
}

void FastGlobalRegistrationDialog::autoEstimateRadius() {
    ccOctree::BestRadiusParams params;
    {
        params.aimedPopulationPerCell = 64;
        params.aimedPopulationRange = 16;
        params.minCellPopulation = 48;
        params.minAboveMinRatio = 0.97;
    }

    PointCoordinateType largestRadius = 0.0;
    for (ccPointCloud* cloud : clouds) {
        PointCoordinateType radius =
                ccOctree::GuessBestRadiusAutoComputeOctree(cloud, params, this);
        if (radius < 0) {
            CVLog::Error(tr("Failed to estimate the radius for cloud %1")
                                 .arg(cloud->getName()));
            return;
        }
        largestRadius = std::max(largestRadius, radius);
    }

    featureRadiusDoubleSpinBox->setValue(largestRadius);
}
