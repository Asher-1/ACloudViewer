// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvGeomFeaturesDlg.h"

// Qt
#include <QDialogButtonBox>
#include <QPushButton>

ccGeomFeaturesDlg::ccGeomFeaturesDlg(QWidget* parent /*=nullptr*/)
    : QDialog(parent, Qt::Tool), Ui::GeomFeaturesDialog() {
    setupUi(this);

    connect(buttonBox->button(QDialogButtonBox::Reset), &QPushButton::clicked,
            this, &ccGeomFeaturesDlg::reset);

    try {
        m_options.push_back(
                Option(roughnessCheckBox,
                       cloudViewer::GeometricalAnalysisTools::Roughness, 0));
        m_options.push_back(
                Option(firstOrderMomentCheckBox,
                       cloudViewer::GeometricalAnalysisTools::MomentOrder1, 0));
        m_options.push_back(
                Option(curvMeanCheckBox,
                       cloudViewer::GeometricalAnalysisTools::Curvature,
                       cloudViewer::Neighbourhood::MEAN_CURV));
        m_options.push_back(
                Option(curvGaussCheckBox,
                       cloudViewer::GeometricalAnalysisTools::Curvature,
                       cloudViewer::Neighbourhood::GAUSSIAN_CURV));
        m_options.push_back(
                Option(curvNCRCheckBox,
                       cloudViewer::GeometricalAnalysisTools::Curvature,
                       cloudViewer::Neighbourhood::NORMAL_CHANGE_RATE));
        m_options.push_back(
                Option(densityKnnCheckBox,
                       cloudViewer::GeometricalAnalysisTools::LocalDensity,
                       cloudViewer::GeometricalAnalysisTools::DENSITY_KNN));
        m_options.push_back(
                Option(densitySurfCheckBox,
                       cloudViewer::GeometricalAnalysisTools::LocalDensity,
                       cloudViewer::GeometricalAnalysisTools::DENSITY_2D));
        m_options.push_back(
                Option(densityVolCheckBox,
                       cloudViewer::GeometricalAnalysisTools::LocalDensity,
                       cloudViewer::GeometricalAnalysisTools::DENSITY_3D));
        m_options.push_back(Option(
                eigSumCheckBox, cloudViewer::GeometricalAnalysisTools::Feature,
                cloudViewer::Neighbourhood::EigenValuesSum));
        m_options.push_back(
                Option(eigOmnivarianceCheckBox,
                       cloudViewer::GeometricalAnalysisTools::Feature,
                       cloudViewer::Neighbourhood::Omnivariance));
        m_options.push_back(
                Option(eigenentropyCheckBox,
                       cloudViewer::GeometricalAnalysisTools::Feature,
                       cloudViewer::Neighbourhood::EigenEntropy));
        m_options.push_back(
                Option(eigAnisotropyCheckBox,
                       cloudViewer::GeometricalAnalysisTools::Feature,
                       cloudViewer::Neighbourhood::Anisotropy));
        m_options.push_back(Option(
                eigPlanarityBox, cloudViewer::GeometricalAnalysisTools::Feature,
                cloudViewer::Neighbourhood::Planarity));
        m_options.push_back(
                Option(eigLinearityCheckBox,
                       cloudViewer::GeometricalAnalysisTools::Feature,
                       cloudViewer::Neighbourhood::Linearity));
        m_options.push_back(Option(
                eigPCA1CheckBox, cloudViewer::GeometricalAnalysisTools::Feature,
                cloudViewer::Neighbourhood::PCA1));
        m_options.push_back(Option(
                eigPCA2CheckBox, cloudViewer::GeometricalAnalysisTools::Feature,
                cloudViewer::Neighbourhood::PCA2));
        m_options.push_back(
                Option(eigSurfaceVarCheckBox,
                       cloudViewer::GeometricalAnalysisTools::Feature,
                       cloudViewer::Neighbourhood::SurfaceVariation));
        m_options.push_back(
                Option(eigSphericityCheckBox,
                       cloudViewer::GeometricalAnalysisTools::Feature,
                       cloudViewer::Neighbourhood::Sphericity));
        m_options.push_back(
                Option(eigVerticalityCheckBox,
                       cloudViewer::GeometricalAnalysisTools::Feature,
                       cloudViewer::Neighbourhood::Verticality));
        m_options.push_back(
                Option(eigenvalue1CheckBox,
                       cloudViewer::GeometricalAnalysisTools::Feature,
                       cloudViewer::Neighbourhood::EigenValue1));
        m_options.push_back(
                Option(eigenvalue2CheckBox,
                       cloudViewer::GeometricalAnalysisTools::Feature,
                       cloudViewer::Neighbourhood::EigenValue2));
        m_options.push_back(
                Option(eigenvalue3CheckBox,
                       cloudViewer::GeometricalAnalysisTools::Feature,
                       cloudViewer::Neighbourhood::EigenValue3));
    } catch (...) {
    }
    m_options.shrink_to_fit();
}

void ccGeomFeaturesDlg::setUpDirection(const CCVector3& upDir) {
    upDirXDoubleSpinBox->setValue(upDir.x);
    upDirYDoubleSpinBox->setValue(upDir.y);
    upDirZDoubleSpinBox->setValue(upDir.z);
    upDirGroupBox->setChecked(true);
}

CCVector3* ccGeomFeaturesDlg::getUpDirection() const {
    if (roughnessCheckBox->isChecked() && upDirGroupBox->isChecked()) {
        static CCVector3 UpDirection(0, 0, 1);
        UpDirection.x =
                static_cast<PointCoordinateType>(upDirXDoubleSpinBox->value());
        UpDirection.y =
                static_cast<PointCoordinateType>(upDirYDoubleSpinBox->value());
        UpDirection.z =
                static_cast<PointCoordinateType>(upDirZDoubleSpinBox->value());
        return &UpDirection;
    } else {
        return nullptr;
    }
}

void ccGeomFeaturesDlg::setSelectedFeatures(
        const ccLibAlgorithms::GeomCharacteristicSet& features) {
    reset();

    for (const ccLibAlgorithms::GeomCharacteristic& f : features) {
        // find the corresponding checkbox
        for (const Option& opt : m_options) {
            if (opt.charac == f.charac && opt.subOption == f.subOption) {
                opt.checkBox->setChecked(true);
                break;
            }
        }
    }
}

bool ccGeomFeaturesDlg::getSelectedFeatures(
        ccLibAlgorithms::GeomCharacteristicSet& features) const {
    features.clear();

    try {
        // test each check-box and add the corresponding feature descriptor if
        // necessary
        for (const Option& opt : m_options) {
            assert(opt.checkBox);
            if (opt.checkBox && opt.checkBox->isChecked())
                features.push_back(opt);
        }
        features.shrink_to_fit();
    } catch (const std::bad_alloc&) {
        return false;
    }
    return true;
}

double ccGeomFeaturesDlg::getRadius() const {
    return radiusDoubleSpinBox->value();
}

void ccGeomFeaturesDlg::setRadius(double r) {
    radiusDoubleSpinBox->setValue(r);
}

void ccGeomFeaturesDlg::reset() {
    for (const Option& opt : m_options) opt.checkBox->setChecked(false);
}
