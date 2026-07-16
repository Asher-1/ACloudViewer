// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "distanceMapGenerationDlg.h"

// local
#include "ccSymbolCloud.h"
#include "dxfProfilesExportDlg.h"
#include "dxfProfilesExporter.h"
#include "qSRAMapWidget.h"

// ECV_PLUGINS
#include <ecvMainAppInterface.h>

// qCC
#include <ecvColorScaleEditorDlg.h>
#include <ecvColorScaleSelector.h>
#include <ecvColorScalesManager.h>

// common
#include <ecvQtHelpers.h>

// Qt5/Qt6 Compatibility
#include <QtCompat.h>

// CV_DB_LIB
#include <ecvColorScale.h>
#include <ecvFileUtils.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>
#include <ecvScalarField.h>

// Qt
#include <QApplication>
#include <QCloseEvent>
#include <QColorDialog>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QLocale>
#include <QMainWindow>
#include <QProgressDialog>
#include <QSettings>
#include <QTextStream>

// system
#include <assert.h>

#include <algorithm>

static QImage CreateColorScaleBarImage(ccColorScale::Shared colorScale,
                                       unsigned steps,
                                       int height = 256) {
    if (!colorScale || height <= 1) return QImage();

    QImage bar(1, height, QImage::Format_ARGB32);
    if (bar.isNull()) return QImage();

    for (int y = 0; y < height; ++y) {
        const double relativePos =
                1.0 - static_cast<double>(y) / static_cast<double>(height - 1);
        const ecvColor::Rgb* rgb =
                colorScale->getColorByRelativePos(relativePos, steps);
        bar.setPixel(0, y, qRgb(rgb->r, rgb->g, rgb->b));
    }

    return bar;
}

static double ConvertAngleFromRad(
        double angle_rad, DistanceMapGenerationDlg::ANGULAR_UNIT destUnit) {
    switch (destUnit) {
        case DistanceMapGenerationDlg::ANG_DEG:  // degrees
            return cloudViewer::RadiansToDegrees(angle_rad);
        case DistanceMapGenerationDlg::ANG_RAD:  // radians
            return angle_rad;
        case DistanceMapGenerationDlg::ANG_GRAD:  // grades
            return angle_rad / M_PI * 200.0;
        default:
            assert(false);
    }

    return 0.0;
}

static double ConvertAngleToRad(
        double angle, DistanceMapGenerationDlg::ANGULAR_UNIT srcUnit) {
    switch (srcUnit) {
        case DistanceMapGenerationDlg::ANG_DEG:  // degrees
            return cloudViewer::DegreesToRadians(angle);
        case DistanceMapGenerationDlg::ANG_RAD:  // radians
            return angle;
        case DistanceMapGenerationDlg::ANG_GRAD:  // grades
            return angle / 200.0 * M_PI;
        default:
            assert(false);
    }

    return 0.0;
}

DistanceMapGenerationDlg::DistanceMapGenerationDlg(
        ccPointCloud* cloud,
        ccScalarField* sf,
        ccPolyline* polyline,
        ecvMainAppInterface* app /*=0*/)
    : QDialog(app ? app->getMainWindow() : 0),
      m_app(app),
      m_cloud(cloud),
      m_profile(polyline),
      m_sf(sf),
      m_map(0),
      m_angularUnits(ANG_GRAD),
      m_window(0),
      m_mapWidget(0),
      m_colorScaleSelector(0),
      m_gridColor(Qt::gray),
      m_symbolColor(Qt::black) {
    setupUi(this);

    assert(m_cloud && m_sf && m_profile);

    // add color ramp selector widget (before calling
    // initFromPersistentSettings!)
    if (m_sf) {
        // create selector widget
        m_colorScaleSelector = new ccColorScaleSelector(
                m_app->getColorScalesManager(), this,
                QString::fromUtf8(":/CC/plugin/qSRA/gearIcon.png"));
        m_colorScaleSelector->init();
        m_colorScaleSelector->setSelectedScale(
                ccColorScalesManager::GetDefaultScale()->getUuid());
        connect(m_colorScaleSelector, SIGNAL(colorScaleSelected(int)), this,
                SLOT(colorScaleChanged(int)));
        connect(m_colorScaleSelector, SIGNAL(colorScaleEditorSummoned()), this,
                SLOT(spawnColorScaleEditor()));
        // add selector to group's layout
        if (!colorRampGroupBox->layout())
            colorRampGroupBox->setLayout(new QHBoxLayout());
        colorRampGroupBox->layout()->addWidget(m_colorScaleSelector);
        colorScaleStepsSpinBox->setRange(ccColorScale::MIN_STEPS,
                                         ccColorScale::MAX_STEPS);
    }

    // init parameters from persistent settings
    initFromPersistentSettings();

    if (m_sf) {
        // we apply the cloud current color scale ONLY if it is not a default
        // one (otherwise we keep the default dialog's one)
        const ccColorScale::Shared& scale = m_sf->getColorScale();
        if (scale && !scale->isLocked())
            m_colorScaleSelector->setSelectedScale(scale->getUuid());
    }

    // profile meta-data
    DistanceMapGenerationTool::ProfileMetaData profileDesc;
    bool validProfile = false;

    // set default dialog values with polyline & cloud information
    if (m_profile) {
        validProfile = DistanceMapGenerationTool::GetPoylineMetaData(
                m_profile, profileDesc);
        if (validProfile) {
            // update the 'Generatrix' tab
            {
                axisDimComboBox->setCurrentIndex(profileDesc.revolDim);

                xOriginDoubleSpinBox->setValue(profileDesc.origin.x);
                yOriginDoubleSpinBox->setValue(profileDesc.origin.y);
                zOriginDoubleSpinBox->setValue(profileDesc.origin.z);
            }

            updateMinAndMaxLimits();
        } else {
            if (m_app)
                m_app->dispToConsole(QString("Invalid profile: can't generate "
                                             "a proper map!"),
                                     ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        }
    }

    // compute min and max height of the points
    // we will 'lock" the max height value with that information
    if (m_cloud) {
        ccBBox bbox = m_cloud->getOwnBB();
        PointCoordinateType hMin = 0, hMax = 0;
        if (bbox.isValid()) {
            hMin = bbox.minCorner().u[profileDesc.revolDim];
            hMax = bbox.maxCorner().u[profileDesc.revolDim];
        }

        if (hMax - hMin <= 0) {
            if (m_app)
                m_app->dispToConsole(
                        QString("Cloud is flat: can't generate a proper map!"),
                        ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        } else {
            hStepDoubleSpinBox->setMaximum(hMax - hMin);
        }
    }

    // add map widget
    {
        m_mapWidget = new qSRAMapWidget(this);
        m_mapWidget->setColorScaleVisible(
                displayColorScaleCheckBox->isChecked());
        m_mapWidget->setLabelFontSize(fontSizeSpinBox->value());

        mapFrame->setLayout(new QHBoxLayout());
        mapFrame->layout()->addWidget(m_mapWidget);
    }

    connect(projectionComboBox, SIGNAL(currentIndexChanged(int)), this,
            SLOT(projectionModeChanged(int)));
    connect(angularUnitComboBox, SIGNAL(currentIndexChanged(int)), this,
            SLOT(angularUnitChanged(int)));
    connect(xStepDoubleSpinBox, SIGNAL(valueChanged(double)), this,
            SLOT(updateGridSteps()));
    connect(hStepDoubleSpinBox, SIGNAL(valueChanged(double)), this,
            SLOT(updateGridSteps()));
    connect(latStepDoubleSpinBox, SIGNAL(valueChanged(double)), this,
            SLOT(updateGridSteps()));
    connect(xMinDoubleSpinBox, SIGNAL(valueChanged(double)), this,
            SLOT(updateGridSteps()));
    connect(xMaxDoubleSpinBox, SIGNAL(valueChanged(double)), this,
            SLOT(updateGridSteps()));
    connect(hMinDoubleSpinBox, SIGNAL(valueChanged(double)), this,
            SLOT(updateGridSteps()));
    connect(hMaxDoubleSpinBox, SIGNAL(valueChanged(double)), this,
            SLOT(updateGridSteps()));
    connect(latMinDoubleSpinBox, SIGNAL(valueChanged(double)), this,
            SLOT(updateGridSteps()));
    connect(latMaxDoubleSpinBox, SIGNAL(valueChanged(double)), this,
            SLOT(updateGridSteps()));
    connect(axisDimComboBox, SIGNAL(currentIndexChanged(int)), this,
            SLOT(updateProfileRevolDim(int)));
    connect(xOriginDoubleSpinBox, SIGNAL(valueChanged(double)), this,
            SLOT(updateProfileOrigin()));
    connect(yOriginDoubleSpinBox, SIGNAL(valueChanged(double)), this,
            SLOT(updateProfileOrigin()));
    connect(zOriginDoubleSpinBox, SIGNAL(valueChanged(double)), this,
            SLOT(updateProfileOrigin()));
    connect(baseRadiusDoubleSpinBox, SIGNAL(valueChanged(double)), this,
            SLOT(baseRadiusChanged(double)));
    connect(heightUnitLineEdit, SIGNAL(editingFinished()), this,
            SLOT(updateHeightUnits()));
    connect(exportCloudPushButton, SIGNAL(clicked()), this,
            SLOT(exportMapAsCloud()));
    connect(exportMeshPushButton, SIGNAL(clicked()), this,
            SLOT(exportMapAsMesh()));
    connect(exportMatrixPushButton, SIGNAL(clicked()), this,
            SLOT(exportMapAsGrid()));
    connect(exportImagePushButton, SIGNAL(clicked()), this,
            SLOT(exportMapAsImage()));
    connect(loadLabelsPushButton, SIGNAL(clicked()), this,
            SLOT(loadOverlaySymbols()));
    connect(clearLabelsPushButton, SIGNAL(clicked()), this,
            SLOT(clearOverlaySymbols()));
    connect(symbolSizeSpinBox, SIGNAL(valueChanged(int)), this,
            SLOT(overlaySymbolsSizeChanged(int)));
    connect(fontSizeSpinBox, SIGNAL(valueChanged(int)), this,
            SLOT(labelFontSizeChanged(int)));
    connect(precisionSpinBox, SIGNAL(valueChanged(int)), this,
            SLOT(labelPrecisionChanged(int)));

    connect(colorScaleStepsSpinBox, SIGNAL(valueChanged(int)), this,
            SLOT(colorRampStepsChanged(int)));
    connect(overlayGridGroupBox, SIGNAL(toggled(bool)), this,
            SLOT(toggleOverlayGrid(bool)));
    connect(scaleXStepDoubleSpinBox, SIGNAL(editingFinished()), this,
            SLOT(updateOverlayGrid()));
    connect(scaleHStepDoubleSpinBox, SIGNAL(editingFinished()), this,
            SLOT(updateOverlayGrid()));
    connect(scaleLatStepDoubleSpinBox, SIGNAL(editingFinished()), this,
            SLOT(updateOverlayGrid()));
    connect(xScaleCheckBox, SIGNAL(clicked()), this, SLOT(updateOverlayGrid()));
    connect(yScaleCheckBox, SIGNAL(clicked()), this, SLOT(updateOverlayGrid()));
    connect(gridColorButton, SIGNAL(clicked()), this, SLOT(changeGridColor()));
    connect(symbolColorButton, SIGNAL(clicked()), this,
            SLOT(changeSymbolColor()));
    connect(displayColorScaleCheckBox, SIGNAL(toggled(bool)), this,
            SLOT(toggleColorScaleDisplay(bool)));
    connect(updateVolumesPushButton, SIGNAL(clicked()), this,
            SLOT(updateVolumes()));

    // DXF profiles export button is only visible/connected if DXF support is
    // enabled!
    if (DxfProfilesExporter::IsEnabled()) {
        connect(exportImageDXFButton, SIGNAL(clicked()), this,
                SLOT(exportProfilesAsDXF()));
    } else {
        exportImageDXFButton->hide();
    }

    // button box
    {
        QPushButton* applyButton = buttonBox->button(QDialogButtonBox::Apply);
        QPushButton* closeButton = buttonBox->button(QDialogButtonBox::Close);
        connect(applyButton, SIGNAL(clicked()), this, SLOT(update()));
        connect(closeButton, SIGNAL(clicked()), this, SLOT(accept()));
        connect(closeButton, &QAbstractButton::clicked, this,
                [this]() { closeEvent(nullptr); });
    }

    angularUnitChanged(m_angularUnits);  // just to be sure
    baseRadiusChanged(0);
    overlaySymbolsColorChanged();
    overlayGridColorChanged();
    labelFontSizeChanged(-1);
    projectionModeChanged(-1);
}

DistanceMapGenerationDlg::~DistanceMapGenerationDlg() {
    qDeleteAll(m_overlaySymbols);
    m_overlaySymbols.clear();
}

void DistanceMapGenerationDlg::closeEvent(QCloseEvent* event) {
    if (event) QDialog::closeEvent(event);
}

void DistanceMapGenerationDlg::updateMinAndMaxLimits() {
    if (m_cloud && m_profile) {
        DistanceMapGenerationTool::ProfileMetaData profileDesc;
        if (DistanceMapGenerationTool::GetPoylineMetaData(m_profile,
                                                          profileDesc)) {
            // compute mean 'radius'
            // as well as min and max 'height'
            double baseRadius = 0.0;
            double minHeight = 0.0;
            double maxHeight = 0.0;
            for (unsigned i = 0; i < m_profile->size(); ++i) {
                const CCVector3* P = m_profile->getPoint(i);
                double radius = P->x;
                double height = P->y + profileDesc.heightShift;
                baseRadius += radius;

                if (i != 0) {
                    if (height < minHeight)
                        minHeight = height;
                    else if (height > maxHeight)
                        maxHeight = height;
                } else {
                    minHeight = maxHeight = height;
                }
            }

            // set default 'base radius'
            if (m_profile->size() != 0) baseRadius /= m_profile->size();
            if (baseRadius == 0.0) baseRadius = 1.0;

            baseRadiusDoubleSpinBox->blockSignals(true);
            baseRadiusDoubleSpinBox->setValue(baseRadius);
            baseRadiusDoubleSpinBox->blockSignals(false);

            // set default min and max height
            hMinDoubleSpinBox->blockSignals(true);
            hMinDoubleSpinBox->setValue(minHeight);
            hMinDoubleSpinBox->blockSignals(false);

            hMaxDoubleSpinBox->blockSignals(true);
            hMaxDoubleSpinBox->setValue(maxHeight);
            hMaxDoubleSpinBox->blockSignals(false);

            // do the same for the latitude

            // compute transformation from the cloud to the surface (of
            // revolution)
            ccGLMatrix cloudToSurfaceOrigin =
                    profileDesc.computeCloudToSurfaceOriginTrans();

            double minLat_rad = 0.0, maxLat_rad = 0.0;
            if (DistanceMapGenerationTool::ComputeMinAndMaxLatitude_rad(
                        m_cloud, minLat_rad, maxLat_rad, cloudToSurfaceOrigin,
                        static_cast<unsigned char>(profileDesc.revolDim))) {
                latMinDoubleSpinBox->blockSignals(true);
                latMinDoubleSpinBox->setValue(
                        ConvertAngleFromRad(minLat_rad, m_angularUnits));
                latMinDoubleSpinBox->blockSignals(false);

                latMaxDoubleSpinBox->blockSignals(true);
                latMaxDoubleSpinBox->setValue(
                        ConvertAngleFromRad(maxLat_rad, m_angularUnits));
                latMaxDoubleSpinBox->blockSignals(false);
            }
        }
    }
}

void DistanceMapGenerationDlg::projectionModeChanged(int) {
    // reset eveything, etc.
    ProjectionMode mode = getProjectionMode();

    clearView();

    // conical mode only
    latLabel->setVisible(mode == PROJ_CONICAL);
    latMinDoubleSpinBox->setVisible(mode == PROJ_CONICAL);
    latMaxDoubleSpinBox->setVisible(mode == PROJ_CONICAL);
    latStepDoubleSpinBox->setVisible(mode == PROJ_CONICAL);
    latStepLabel->setVisible(mode == PROJ_CONICAL);
    scaleLatStepLabel->setVisible(mode == PROJ_CONICAL);
    scaleLatStepDoubleSpinBox->setVisible(mode == PROJ_CONICAL);
    spanRatioFrame->setVisible(mode == PROJ_CONICAL);

    // cylindrical mode only
    yLabel->setVisible(mode == PROJ_CYLINDRICAL);
    hMinDoubleSpinBox->setVisible(mode == PROJ_CYLINDRICAL);
    hMaxDoubleSpinBox->setVisible(mode == PROJ_CYLINDRICAL);
    exportImageDXFButton->setVisible(mode == PROJ_CYLINDRICAL);
    heightStepLabel->setVisible(mode == PROJ_CYLINDRICAL);
    hStepDoubleSpinBox->setVisible(mode == PROJ_CYLINDRICAL);
    scaleHeightStepLabel->setVisible(mode == PROJ_CYLINDRICAL);
    scaleHStepDoubleSpinBox->setVisible(mode == PROJ_CYLINDRICAL);
    heightUnitLineEdit->setVisible(mode == PROJ_CYLINDRICAL);
    yScaleCheckBox->setVisible(mode == PROJ_CYLINDRICAL);

    baseRadiusChanged(0);
    updateGridSteps();

    if (m_map) update();
}

DistanceMapGenerationDlg::ProjectionMode
DistanceMapGenerationDlg::getProjectionMode() const {
    switch (projectionComboBox->currentIndex()) {
        case 0:
            return PROJ_CYLINDRICAL;
        case 1:
            return PROJ_CONICAL;
        default:
            assert(false);
    }

    return PROJ_CYLINDRICAL;
}

DistanceMapGenerationDlg::ANGULAR_UNIT
DistanceMapGenerationDlg::getAngularUnit() const {
    switch (m_angularUnits) {
        case 0:  // degrees
            return ANG_DEG;
        case 1:  // radians
            return ANG_RAD;
        case 2:  // grades
            return ANG_GRAD;
        default:
            assert(false);
    }

    return ANG_DEG;
}

QString DistanceMapGenerationDlg::getAngularUnitString() const {
    switch (m_angularUnits) {
        case 0:  // degrees
            return "deg";
        case 1:  // radians
            return "rad";
        case 2:  // grades
            return "grad";
        default:
            assert(false);
    }

    return "none";
}

QString DistanceMapGenerationDlg::getCondensedAngularUnitString() const {
    switch (m_angularUnits) {
        case 0:  // degrees
            return QChar(0x00B0);
        case 1:  // radians
            return "rd";
        case 2:  // grades
            return "gr";
        default:
            assert(false);
    }

    return "none";
}

double DistanceMapGenerationDlg::getSpinboxAngularValue(
        QDoubleSpinBox* spinBox,
        DistanceMapGenerationDlg::ANGULAR_UNIT destUnit) const {
    // no conversion necessary?
    if (m_angularUnits == destUnit) return spinBox->value();

    // otherwise we convert to radians first
    double angle_rad = ConvertAngleToRad(spinBox->value(), m_angularUnits);
    // then to the destination value
    return ConvertAngleFromRad(angle_rad, destUnit);
}

void DistanceMapGenerationDlg::updateZoom(ccBBox& /*box*/) {
    if (m_mapWidget) m_mapWidget->zoomFit();
}

void DistanceMapGenerationDlg::clearView() {
    if (!m_mapWidget) return;

    m_mapWidget->setMapImage(QImage());
    m_mapWidget->setXLabels({});
    m_mapWidget->setYLabels({});
}

void DistanceMapGenerationDlg::update() {
    if (m_map) {
        if (getProjectionMode() == PROJ_CONICAL) {
            // we must check that the projection parameter have not changed!
            // Otherwise the symbols will be misplaced...
            double yMin, yMax, yStep;
            getGridYValues(yMin, yMax, yStep, ANG_RAD);
            if (!m_map->conical || m_map->yMin != yMin || m_map->yMax != yMax ||
                m_map->conicalSpanRatio !=
                        conicSpanRatioDoubleSpinBox->value()) {
                clearOverlaySymbols();
            }
        } else if (m_map->conical) {
            // we can't keep the symbols when switching the projection mode
            clearOverlaySymbols();
        }
    }

    // release memory
    m_map.clear();

    // clear 3D view
    clearView();

    // update map
    m_map = updateMap();
    // and GUI
    exportGroupBox->setEnabled(m_map != 0);

    // auto update volumes
    updateVolumes();

    if (m_map && m_mapWidget) {
        m_mapWidget->setMapBounds(m_map->xMin, m_map->xMax, m_map->yMin,
                                  m_map->yMax);
        updateMapTexture();
    }

    // update sf names, etc.
    updateHeightUnits();  // already call 'updateOverlayGrid'!
    // update zoom
    ccBBox box;
    if (m_map) {
        box.add(CCVector3(static_cast<PointCoordinateType>(m_map->xMin),
                          static_cast<PointCoordinateType>(m_map->yMin), 0));
        box.add(CCVector3(static_cast<PointCoordinateType>(m_map->xMax),
                          static_cast<PointCoordinateType>(m_map->yMax), 0));
    }
    updateZoom(box);

    saveToPersistentSettings();
}

void DistanceMapGenerationDlg::updateHeightUnits() {
    scaleHStepDoubleSpinBox->setSuffix(QString(" ") +
                                       heightUnitLineEdit->text());

    updateOverlayGrid();
}

void DistanceMapGenerationDlg::updateMapTexture() {
    if (!m_map || !m_colorScaleSelector || !m_mapWidget) return;

    // spawn "update" dialog
    QProgressDialog progressDlg(QString("Updating..."), 0, 0, 0, 0, Qt::Popup);
    progressDlg.setMinimumDuration(0);
    progressDlg.setModal(true);
    progressDlg.show();
    QApplication::processEvents();

    // current color scale
    ccColorScale::Shared colorScale = m_colorScaleSelector->getSelectedScale();
    if (!colorScale) {
        if (m_app)
            m_app->dispToConsole(QString("No color scale chosen!"),
                                 ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }

    const unsigned colorScaleSteps =
            static_cast<unsigned>(colorScaleStepsSpinBox->value());

    // create new texture QImage
    QImage mapImage = DistanceMapGenerationTool::ConvertMapToImage(
            m_map, colorScale, colorScaleSteps);
    if (mapImage.isNull()) {
        if (m_app)
            m_app->dispToConsole(
                    QString("Failed to create map texture! Not enough memory?"),
                    ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }

    m_mapWidget->setMapImage(mapImage);

    QImage scaleBar = CreateColorScaleBarImage(colorScale, colorScaleSteps);
    m_mapWidget->setColorScale(m_map->minVal, m_map->maxVal, scaleBar);
}

void DistanceMapGenerationDlg::colorScaleChanged(int) {
    if (!m_mapWidget || !m_colorScaleSelector) return;

    updateMapTexture();
}

void DistanceMapGenerationDlg::spawnColorScaleEditor() {
    if (!m_app || !m_app->getColorScalesManager()) return;

    ccColorScale::Shared colorScale =
            (m_colorScaleSelector
                     ? m_colorScaleSelector->getSelectedScale()
                     : m_app->getColorScalesManager()->getDefaultScale(
                               ccColorScalesManager::BGYR));
    ccColorScaleEditorDialog cseDlg(m_app->getColorScalesManager(), m_app,
                                    colorScale, m_app->getMainWindow());
    if (cseDlg.exec()) {
        colorScale = cseDlg.getActiveScale();
        if (colorScale && m_colorScaleSelector) {
            m_colorScaleSelector->init();  // in fact it's a 're-init'
            m_colorScaleSelector->setSelectedScale(colorScale->getUuid());
        }

        // save current scale manager state to persistent settings
        m_app->getColorScalesManager()->toPersistentSettings();
    }
}

// helper
static void SetSpinBoxValues(QDoubleSpinBox* spinBox,
                             int decimals,
                             double minVal,
                             double maxVal,
                             double step,
                             double value) {
    if (spinBox) {
        spinBox->setDecimals(decimals);
        spinBox->setRange(minVal, maxVal);
        spinBox->setSingleStep(step);
        spinBox->setValue(value);
    }
}

void DistanceMapGenerationDlg::angularUnitChanged(int index) {
    // backup previous value
    double xStep_rad = getSpinboxAngularValue(xStepDoubleSpinBox, ANG_RAD);
    double xMin_rad = getSpinboxAngularValue(xMinDoubleSpinBox, ANG_RAD);
    double xMax_rad = getSpinboxAngularValue(xMaxDoubleSpinBox, ANG_RAD);
    double scaleXStep_rad =
            getSpinboxAngularValue(scaleXStepDoubleSpinBox, ANG_RAD);

    // same for latitude-related spinboxes
    double latStep_rad = getSpinboxAngularValue(latStepDoubleSpinBox, ANG_RAD);
    double latMin_rad = getSpinboxAngularValue(latMinDoubleSpinBox, ANG_RAD);
    double latMax_rad = getSpinboxAngularValue(latMaxDoubleSpinBox, ANG_RAD);
    double scaleLatStep_rad =
            getSpinboxAngularValue(scaleLatStepDoubleSpinBox, ANG_RAD);

    switch (index) {
        case 0:  // degrees
        {
            m_angularUnits = ANG_DEG;

            SetSpinBoxValues(xStepDoubleSpinBox, 2, 0.01, 360.0, 0.1,
                             cloudViewer::RadiansToDegrees(xStep_rad));
            SetSpinBoxValues(scaleXStepDoubleSpinBox, 2, 0.01, 360.0, 5.0,
                             cloudViewer::RadiansToDegrees(scaleXStep_rad));
            SetSpinBoxValues(xMinDoubleSpinBox, 2, 0.0, 360.0, 5.0,
                             cloudViewer::RadiansToDegrees(xMin_rad));
            SetSpinBoxValues(xMaxDoubleSpinBox, 2, 0.0, 360.0, 5.0,
                             cloudViewer::RadiansToDegrees(xMax_rad));

            SetSpinBoxValues(latStepDoubleSpinBox, 2, 0.01, 89.99, 1.0,
                             cloudViewer::RadiansToDegrees(latStep_rad));
            SetSpinBoxValues(scaleLatStepDoubleSpinBox, 2, 0.01, 89.99, 1.0,
                             cloudViewer::RadiansToDegrees(scaleLatStep_rad));
            SetSpinBoxValues(latMinDoubleSpinBox, 2, -89.99, 89.99, 1.0,
                             cloudViewer::RadiansToDegrees(latMin_rad));
            SetSpinBoxValues(latMaxDoubleSpinBox, 2, -89.99, 89.99, 1.0,
                             cloudViewer::RadiansToDegrees(latMax_rad));

            xMaxDoubleSpinBox->setMaximum(360.0);
            xMaxDoubleSpinBox->setValue(360.0);
            break;
        }

        case 1:  // radians
        {
            m_angularUnits = ANG_RAD;

            double PIx2 = 2.0 * M_PI;
            SetSpinBoxValues(xStepDoubleSpinBox, 4, 0.0001, PIx2, 0.1,
                             xStep_rad);
            SetSpinBoxValues(scaleXStepDoubleSpinBox, 4, 0.0001, PIx2, 0.5,
                             scaleXStep_rad);
            SetSpinBoxValues(xMinDoubleSpinBox, 4, 0.0, PIx2, 0.5, xMin_rad);
            SetSpinBoxValues(xMaxDoubleSpinBox, 4, 0.0, PIx2, 0.5, xMax_rad);

            double PIdiv2 = M_PI / 2.0 - 0.0001;
            SetSpinBoxValues(scaleLatStepDoubleSpinBox, 4, 0.0001, PIdiv2, 0.3,
                             scaleLatStep_rad);
            SetSpinBoxValues(latStepDoubleSpinBox, 4, 0.0001, PIdiv2, 0.3,
                             latStep_rad);
            SetSpinBoxValues(latMinDoubleSpinBox, 4, -PIdiv2, PIdiv2, 0.3,
                             latMin_rad);
            SetSpinBoxValues(latMaxDoubleSpinBox, 4, -PIdiv2, PIdiv2, 0.3,
                             latMax_rad);

            xMaxDoubleSpinBox->setMaximum(PIx2);
            xMaxDoubleSpinBox->setValue(PIx2);
            break;
        }

        case 2:  // grades
        {
            m_angularUnits = ANG_GRAD;

            SetSpinBoxValues(xStepDoubleSpinBox, 2, 0.01, 400.0, 0.1,
                             xStep_rad * 200.0 / M_PI);
            SetSpinBoxValues(scaleXStepDoubleSpinBox, 2, 0.01, 400.0, 5.0,
                             scaleXStep_rad * 200.0 / M_PI);
            SetSpinBoxValues(xMinDoubleSpinBox, 2, 0.0, 400.0, 5.0,
                             xMin_rad * 200.0 / M_PI);
            SetSpinBoxValues(xMaxDoubleSpinBox, 2, 0.0, 400.0, 5.0,
                             xMax_rad * 200.0 / M_PI);

            SetSpinBoxValues(scaleLatStepDoubleSpinBox, 2, 0.01, 99.99, 1.0,
                             scaleLatStep_rad * 200.0 / M_PI);
            SetSpinBoxValues(latStepDoubleSpinBox, 2, 0.01, 99.99, 1.0,
                             latStep_rad * 200.0 / M_PI);
            SetSpinBoxValues(latMinDoubleSpinBox, 2, -99.99, 99.99, 1.0,
                             latMin_rad * 200.0 / M_PI);
            SetSpinBoxValues(latMaxDoubleSpinBox, 2, -99.99, 99.99, 1.0,
                             latMax_rad * 200.0 / M_PI);

            xMaxDoubleSpinBox->setMaximum(400.0);
            xMaxDoubleSpinBox->setValue(400.0);
            break;
        }

        default:  // shouldn't happen!
            assert(false);
    }

    // update spinboxes suffix
    {
        QString suffix = QString(" ") + getAngularUnitString();
        scaleXStepDoubleSpinBox->setSuffix(suffix);
        latMinDoubleSpinBox->setSuffix(suffix);
        latMaxDoubleSpinBox->setSuffix(suffix);
        scaleLatStepDoubleSpinBox->setSuffix(suffix);
    }

    updateOverlayGrid();
}

void DistanceMapGenerationDlg::getGridXValues(
        double& minX,
        double& maxX,
        double& step,
        ANGULAR_UNIT unit /*=ANG_RAD*/) const {
    minX = getSpinboxAngularValue(xMinDoubleSpinBox, unit);
    maxX = getSpinboxAngularValue(xMaxDoubleSpinBox, unit);
    step = getSpinboxAngularValue(xStepDoubleSpinBox, unit);
}

void DistanceMapGenerationDlg::getGridYValues(
        double& minY,
        double& maxY,
        double& step,
        ANGULAR_UNIT unit /*=ANG_RAD*/) const {
    switch (getProjectionMode()) {
        case PROJ_CYLINDRICAL:
            minY = hMinDoubleSpinBox->value();
            maxY = hMaxDoubleSpinBox->value();
            step = hStepDoubleSpinBox->value();
            break;
        case PROJ_CONICAL:
            minY = getSpinboxAngularValue(latMinDoubleSpinBox, unit);
            maxY = getSpinboxAngularValue(latMaxDoubleSpinBox, unit);
            step = getSpinboxAngularValue(latStepDoubleSpinBox, unit);
            break;
        default:
            assert(false);
            break;
    }
}

double DistanceMapGenerationDlg::getScaleYStep(
        ANGULAR_UNIT unit /*=ANG_RAD*/) const {
    if (getProjectionMode() == PROJ_CYLINDRICAL)
        return scaleHStepDoubleSpinBox->value();
    else
        return getSpinboxAngularValue(scaleLatStepDoubleSpinBox, unit);
}

void DistanceMapGenerationDlg::updateProfileRevolDim(int dim) {
    if (!m_profile) {
        assert(false);
        return;
    }

    // update projection dimension
    assert(dim >= 0 && dim < 3);
    DistanceMapGenerationTool::SetPoylineRevolDim(m_profile, dim);
}

void DistanceMapGenerationDlg::updateProfileOrigin() {
    if (!m_profile) {
        assert(false);
        return;
    }

    DistanceMapGenerationTool::ProfileMetaData profileDesc;
    DistanceMapGenerationTool::GetPoylineMetaData(m_profile, profileDesc);

    // update origin
    CCVector3 origin(
            static_cast<PointCoordinateType>(xOriginDoubleSpinBox->value()),
            static_cast<PointCoordinateType>(yOriginDoubleSpinBox->value()),
            static_cast<PointCoordinateType>(zOriginDoubleSpinBox->value()));

    // DGM: we must compensate for the change of shift along the revolution
    // axis!
    double dShift = origin.u[profileDesc.revolDim] -
                    profileDesc.origin.u[profileDesc.revolDim];
    profileDesc.heightShift -= dShift;

    DistanceMapGenerationTool::SetPoylineOrigin(m_profile, origin);
    DistanceMapGenerationTool::SetPolylineHeightShift(m_profile,
                                                      profileDesc.heightShift);

    if (dShift != 0) {
        clearOverlaySymbols();  // symbols placement depend on the origin
                                // position along the revolution axis
    }
    updateMinAndMaxLimits();
}

void DistanceMapGenerationDlg::updateGridSteps() {
    // angular step
    QString xStepsStr;
    {
        double minX, maxX, step;
        getGridXValues(minX, maxX, step, m_angularUnits);
        xStepsStr = (step > 0 ? QString::number(
                                        ceil(std::max(maxX - minX, 0.0) / step))
                              : "inf");
    }

    // Y step
    QString yStepsStr;
    {
        double minY, maxY, step;
        getGridYValues(minY, maxY, step, m_angularUnits);
        yStepsStr = (step > 0 ? QString::number(
                                        ceil(std::max(maxY - minY, 0.0) / step))
                              : "inf");
    }

    gridSizeLabel->setText(QString("%1 x %2").arg(xStepsStr).arg(yStepsStr));
}

double DistanceMapGenerationDlg::getBaseRadius() const {
    return getProjectionMode() == PROJ_CONICAL
                   ? 1.0
                   : baseRadiusDoubleSpinBox->value();
}

void DistanceMapGenerationDlg::baseRadiusChanged(double) {
    // base radius only affects 3D export; no impact on 2D map widget display
}

QString DistanceMapGenerationDlg::getHeightUnitString() const {
    return heightUnitLineEdit->text();
}

DistanceMapGenerationTool::FillStrategyType
DistanceMapGenerationDlg::getFillingStrategy() const {
    switch (fillingStrategyComboxBox->currentIndex()) {
        case 0:
            return DistanceMapGenerationTool::FILL_STRAT_MIN_DIST;
        case 1:
            return DistanceMapGenerationTool::FILL_STRAT_AVG_DIST;
        case 2:
            return DistanceMapGenerationTool::FILL_STRAT_MAX_DIST;
        default:
            return DistanceMapGenerationTool::INVALID_STRATEGY_TYPE;
    }
    return DistanceMapGenerationTool::INVALID_STRATEGY_TYPE;
}

DistanceMapGenerationTool::EmptyCellFillOption
DistanceMapGenerationDlg::getEmptyCellFillingOption() const {
    switch (emptyCellsComboBox->currentIndex()) {
        case 0:
            return DistanceMapGenerationTool::LEAVE_EMPTY;
        case 1:
            return DistanceMapGenerationTool::FILL_WITH_ZERO;
        case 2:
            return DistanceMapGenerationTool::FILL_INTERPOLATE;
        default:
            return DistanceMapGenerationTool::LEAVE_EMPTY;
    }
    return DistanceMapGenerationTool::LEAVE_EMPTY;
}

QSharedPointer<DistanceMapGenerationTool::Map>
DistanceMapGenerationDlg::updateMap() {
    if (!m_cloud || !m_sf || !m_profile) {
        assert(false);
        return QSharedPointer<DistanceMapGenerationTool::Map>(0);
    }

    // profile parameters
    DistanceMapGenerationTool::ProfileMetaData profileDesc;
    if (!DistanceMapGenerationTool::GetPoylineMetaData(m_profile,
                                                       profileDesc)) {
        assert(false);
        return QSharedPointer<DistanceMapGenerationTool::Map>(0);
    }

    // compute transformation from cloud to the surface (of revolution)
    ccGLMatrix cloudToSurface = profileDesc.computeCloudToSurfaceOriginTrans();

    // steps
    double angStep_rad = getSpinboxAngularValue(xStepDoubleSpinBox, ANG_RAD);
    // CW (clockwise) or CCW (counterclockwise)
    bool ccw = ccwCheckBox->isChecked();

    // Y values
    double yMin, yMax, yStep;
    getGridYValues(yMin, yMax, yStep, ANG_RAD);

    // generate map
    return DistanceMapGenerationTool::CreateMap(
            m_cloud, m_sf, cloudToSurface, profileDesc.revolDim, angStep_rad,
            yStep, yMin, yMax, getProjectionMode() == PROJ_CONICAL, ccw,
            getFillingStrategy(), getEmptyCellFillingOption(), m_app);
}

void DistanceMapGenerationDlg::exportMapAsCloud() {
    if (!m_map) {
        if (m_app)
            m_app->dispToConsole(QString("Invalid map! Try to refresh it?"),
                                 ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }
    if (!m_profile) {
        if (m_app)
            m_app->dispToConsole(QString("Invalid profile?!"),
                                 ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }

    double baseRadius = getBaseRadius();

    ccPointCloud* cloud = DistanceMapGenerationTool::ConvertMapToCloud(
            m_map, m_profile, baseRadius);
    if (!cloud) {
        if (m_app)
            m_app->dispToConsole(QString("Failed to convert map to cloud!"),
                                 ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }
    if (m_colorScaleSelector && cloud->getCurrentDisplayedScalarField())
        cloud->getCurrentDisplayedScalarField()->setColorScale(
                m_colorScaleSelector->getSelectedScale());
    cloud->setName(
            m_cloud->getName() +
            QString(".map(%1,%2)").arg(m_map->xSteps).arg(m_map->ySteps));

    if (m_app) m_app->addToDB(cloud);
}

void DistanceMapGenerationDlg::exportMapAsMesh() {
    if (!m_profile || !m_colorScaleSelector) {
        assert(false);
        return;
    }

    if (!m_map) {
        if (m_app)
            m_app->dispToConsole(QString("Invalid map! Try to refresh it?"),
                                 ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }

    // profile parameters
    DistanceMapGenerationTool::ProfileMetaData profileDesc;
    if (!DistanceMapGenerationTool::GetPoylineMetaData(m_profile,
                                                       profileDesc)) {
        assert(false);
        return;
    }

    ccColorScale::Shared colorScale = m_colorScaleSelector->getSelectedScale();

    // create new texture QImage
    QImage mapImage = DistanceMapGenerationTool::ConvertMapToImage(
            m_map, colorScale, colorScaleStepsSpinBox->value());
    if (mapImage.isNull()) {
        if (m_app)
            m_app->dispToConsole(QString("Failed to generate mesh texture! Not "
                                         "enough memory?"),
                                 ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }

    // compute transformation from cloud to the profile (origin)
    ccGLMatrix cloudToProfile = profileDesc.computeCloudToProfileOriginTrans();
    ccMesh* mesh = DistanceMapGenerationTool::ConvertProfileToMesh(
            m_profile, cloudToProfile, m_map->counterclockwise, m_map->xSteps,
            mapImage);

    if (mesh) {
        // mesh->setDisplay_recursive(m_cloud->getDisplay());
        mesh->setName(
                m_cloud->getName() +
                QString(".map(%1,%2)").arg(m_map->xSteps).arg(m_map->ySteps));
        if (m_app) m_app->addToDB(mesh);
    } else {
        if (m_app)
            m_app->dispToConsole(
                    QString("Failed to generate mesh! Not enough memory?"),
                    ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
    }
}

void DistanceMapGenerationDlg::exportMapAsGrid() {
    if (!m_map) {
        if (m_app)
            m_app->dispToConsole(QString("Invalid map! Try to refresh it?"),
                                 ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }

    // persistent settings (default export path)
    QSettings settings;
    settings.beginGroup("qSRA");
    QString path = settings.value("exportPath", ecvFileUtils::defaultDocPath())
                           .toString();

    QString filter("Grid file (*.csv)");

    // open file saving dialog
    QString filename =
            QFileDialog::getSaveFileName(0, "Select output file", path, filter);
    if (filename.isEmpty()) return;

    // save current export path to persistent settings
    settings.setValue("exportPath", QFileInfo(filename).absolutePath());

    QString xUnit = getAngularUnitString();
    double xConversionFactor = ConvertAngleFromRad(1.0, m_angularUnits);
    QString yUnit = getHeightUnitString();
    double yConversionFactor = 1.0;

    if (DistanceMapGenerationTool::SaveMapAsCSVMatrix(
                m_map, filename, xUnit, yUnit, xConversionFactor,
                yConversionFactor, m_app)) {
        if (m_app)
            m_app->dispToConsole(
                    QString("File '%1' saved successfully").arg(filename),
                    ecvMainAppInterface::STD_CONSOLE_MESSAGE);
    }
}

void DistanceMapGenerationDlg::exportMapAsImage() {
    if (!m_mapWidget) return;

    QSettings settings;
    settings.beginGroup("qSRA");
    QString path = settings.value("exportPath", ecvFileUtils::defaultDocPath())
                           .toString();

    QString filename = QFileDialog::getSaveFileName(this, "Select output file",
                                                    path, "Image file (*.png)");
    if (filename.isEmpty()) return;

    settings.setValue("exportPath", QFileInfo(filename).absolutePath());

    QImage image = m_mapWidget->exportAsImage();
    if (!image.save(filename)) {
        if (m_app)
            m_app->dispToConsole(
                    QString("Failed to save file '%1'!").arg(filename),
                    ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
    } else if (m_app) {
        m_app->dispToConsole(
                QString("File '%1' saved successfully").arg(filename),
                ecvMainAppInterface::STD_CONSOLE_MESSAGE);
    }
}

void DistanceMapGenerationDlg::exportProfilesAsDXF() {
    if (!m_map || !m_profile) return;

    DxfProfilesExportDlg dpeDlg(this);

    if (!dpeDlg.exec()) return;

    // profile meta-data (we only need the height shift)
    PointCoordinateType heightShift = 0;
    DistanceMapGenerationTool::GetPolylineHeightShift(m_profile, heightShift);

    DxfProfilesExporter::Parameters params;
    params.legendTheoProfileTitle = dpeDlg.theoNameLineEdit->text();
    params.legendRealProfileTitle = dpeDlg.realNameLineEdit->text();
    params.scaledDevUnits = dpeDlg.scaledDevUnitsLineEdit->text();
    params.devLabelMultCoef = dpeDlg.devValuesScaleDoubleSpinBox->value();
    params.devMagnifyCoef = dpeDlg.magnifyCoefSpinBox->value();
    params.precision = dpeDlg.precisionSpinBox->value();

    /*** vertical profiles ***/

    int angularStepCount = dpeDlg.angularStepsSpinBox->value();
    assert(angularStepCount >= 1);
    // we take the same steps as the overlay grid for labels
    QString vertFilename = dpeDlg.getVertFilename();
    if (!vertFilename.isNull()) {
        // generate profiles titles
        params.profileTitles.clear();
        QString vertProfileBaseTitle = dpeDlg.vertTitleLineEdit->text();
        for (int i = 0; i < angularStepCount; ++i) {
            double angle_rad =
                    static_cast<double>(i) * 2.0 * M_PI / angularStepCount;
            double angle_cur = ConvertAngleFromRad(angle_rad, m_angularUnits);
            params.profileTitles << QString("%1 - %2 %3")
                                            .arg(vertProfileBaseTitle)
                                            .arg(angle_cur)
                                            .arg(getAngularUnitString());
        }

        double heightStep = getScaleYStep();
        if (!DxfProfilesExporter::SaveVerticalProfiles(
                    m_map, m_profile, vertFilename, angularStepCount,
                    heightStep, heightShift, params, m_app)) {
            if (m_app)
                m_app->dispToConsole(
                        QString("Failed to save file '%1'!").arg(vertFilename),
                        ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
            return;
        } else {
            if (m_app)
                m_app->dispToConsole(QString("File '%1' saved successfully")
                                             .arg(vertFilename),
                                     ecvMainAppInterface::STD_CONSOLE_MESSAGE);
        }
    }

    /*** horizontal profiles ***/

    QString horizFilename = dpeDlg.getHorizFilename();
    int heightStepCount = dpeDlg.heightStepsSpinBox->value();
    assert(heightStepCount >= 1);
    // we take the same steps as the overlay grid for labels
    if (!horizFilename.isNull()) {
        // generate profiles titles
        params.profileTitles.clear();
        QString horizProfileBaseTitle = dpeDlg.horizTitleLineEdit->text();
        params.profileTitles
                << QString("%1 - %2 ").arg(horizProfileBaseTitle).arg("%1") +
                           getHeightUnitString();

        double angleStep_rad =
                getSpinboxAngularValue(scaleXStepDoubleSpinBox, ANG_RAD);
        if (!DxfProfilesExporter::SaveHorizontalProfiles(
                    m_map, m_profile, horizFilename, heightStepCount,
                    heightShift, angleStep_rad,
                    ConvertAngleFromRad(1.0, m_angularUnits),
                    getCondensedAngularUnitString(), params, m_app)) {
            if (m_app)
                m_app->dispToConsole(
                        QString("Failed to save file '%1'!").arg(horizFilename),
                        ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
            return;
        } else {
            if (m_app)
                m_app->dispToConsole(QString("File '%1' saved successfully")
                                             .arg(horizFilename),
                                     ecvMainAppInterface::STD_CONSOLE_MESSAGE);
        }
    }
}

void DistanceMapGenerationDlg::loadOverlaySymbols() {
    // need a valid map
    if (!m_map) {
        if (m_app)
            m_app->dispToConsole(QString("Generate a valid map first!"),
                                 ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }

    // profile parameters
    DistanceMapGenerationTool::ProfileMetaData profileDesc;
    if (!DistanceMapGenerationTool::GetPoylineMetaData(m_profile,
                                                       profileDesc)) {
        assert(false);
        return;
    }

    // persistent settings (default import path)
    QSettings settings;
    settings.beginGroup("qSRA");
    QString path = settings.value("importPath", ecvFileUtils::defaultDocPath())
                           .toString();

    QString filter("Symbols (*.txt)");

    // open file loading dialog
    QString filename = QFileDialog::getOpenFileName(0, "Select symbols file",
                                                    path, filter);
    if (filename.isEmpty()) return;

    QFileInfo fileInfo(filename);
    if (!fileInfo.exists())  //?!
    {
        if (m_app)
            m_app->dispToConsole(
                    QString("Failed to find symbol file '%1'?!").arg(filename),
                    ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }

    // save current impoort path to persistent settings
    settings.setValue("importPath", fileInfo.absolutePath());

    ccSymbolCloud* symbolCloud = 0;
    // try to load the file (as a "symbol" point cloud)
    {
        QFile file(filename);
        assert(file.exists());
        if (!file.open(QFile::ReadOnly)) {
            if (m_app)
                m_app->dispToConsole(QString("Failed to open symbol file '%1'!")
                                             .arg(filename),
                                     ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
            return;
        }

        symbolCloud = new ccSymbolCloud(fileInfo.baseName());

        QTextStream stream(&file);
        QString currentLine = stream.readLine();
        bool error = false;
        while (!currentLine.isNull()) {
            QStringList tokens = qtCompatSplitRegex(currentLine, "\\s+",
                                                    QtCompat::SkipEmptyParts);
            if (tokens.size() == 4) {
                bool okX, okY, okZ;
                CCVector3 P(static_cast<PointCoordinateType>(
                                    tokens[1].toDouble(&okX)),
                            static_cast<PointCoordinateType>(
                                    tokens[2].toDouble(&okY)),
                            static_cast<PointCoordinateType>(
                                    tokens[3].toDouble(&okZ)));

                if (!okX || !okY || !okZ) {
                    error = true;
                    break;
                }

                QString label = tokens[0];
                if (symbolCloud->size() == symbolCloud->capacity()) {
                    if (!symbolCloud->reserveThePointsTable(
                                symbolCloud->size() + 64) ||
                        !symbolCloud->reserveLabelArray(symbolCloud->size() +
                                                        64)) {
                        if (m_app)
                            m_app->dispToConsole(
                                    QString("Not enough memory!"),
                                    ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
                        error = true;
                        break;
                    }
                }

                // DGM: warning, for historical reasons height values are
                // expressed relative to the profile origin!
                P.u[profileDesc.revolDim] +=
                        profileDesc.origin.u[profileDesc.revolDim];

                symbolCloud->addPoint(P);
                symbolCloud->addLabel(label);
            }

            // next line
            currentLine = stream.readLine();
        }

        if (symbolCloud->size() == 0) {
            delete symbolCloud;
            symbolCloud = 0;
        } else {
            symbolCloud->shrinkToFit();
        }

        if (error) {
            delete symbolCloud;
            symbolCloud = nullptr;

            if (m_app)
                m_app->dispToConsole(
                        QString("An error occurred while loading the file! "
                                "Result may be incomplete"),
                        ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
            return;
        }
    }

    if (symbolCloud) {
        // unroll the symbol cloud the same way as the input cloud
        if (m_map) {
            // compute transformation from cloud to the surface (of revolution)
            ccGLMatrix cloudToSurface =
                    profileDesc.computeCloudToSurfaceOriginTrans();
            // CW (clockwise) or CCW (counterclockwise)
            bool ccw = ccwCheckBox->isChecked();

            if (getProjectionMode() == PROJ_CYLINDRICAL) {
                DistanceMapGenerationTool::ConvertCloudToCylindrical(
                        symbolCloud, cloudToSurface, profileDesc.revolDim, ccw);
            } else {
                double conicalSpanRatio = conicSpanRatioDoubleSpinBox->value();
                DistanceMapGenerationTool::ConvertCloudToConical(
                        symbolCloud, cloudToSurface, profileDesc.revolDim,
                        m_map->yMin, m_map->yMax, conicalSpanRatio, ccw);
            }
        }
        symbolCloud->setSymbolSize(
                static_cast<double>(symbolSizeSpinBox->value()));
        symbolCloud->setFontSize(fontSizeSpinBox->value());
        symbolCloud->setVisible(true);
        ecvColor::Rgb rgb(static_cast<ColorCompType>(m_symbolColor.red()),
                          static_cast<ColorCompType>(m_symbolColor.green()),
                          static_cast<ColorCompType>(m_symbolColor.blue()));
        symbolCloud->setTempColor(rgb, true);
        m_overlaySymbols.push_back(symbolCloud);

        clearLabelsPushButton->setEnabled(true);
        clearLabelsPushButton->setText(
                QString("Clear (%1)").arg(symbolCloud->size()));

        syncOverlaySymbolsToWidget();
    } else {
        assert(false);
        delete symbolCloud;
        symbolCloud = 0;
    }
}

void DistanceMapGenerationDlg::clearOverlaySymbols() {
    qDeleteAll(m_overlaySymbols);
    m_overlaySymbols.clear();

    clearLabelsPushButton->setEnabled(false);
    clearLabelsPushButton->setText("Clear");

    if (m_mapWidget) m_mapWidget->clearOverlaySymbols();
}

void DistanceMapGenerationDlg::overlaySymbolsSizeChanged(int /*size*/) {
    const double symbolSize = static_cast<double>(symbolSizeSpinBox->value());
    for (ccSymbolCloud* cloud : m_overlaySymbols)
        cloud->setSymbolSize(symbolSize);
    syncOverlaySymbolsToWidget();
}

void DistanceMapGenerationDlg::overlaySymbolsColorChanged() {
    ccQtHelpers::SetButtonColor(symbolColorButton, m_symbolColor);

    ecvColor::Rgb rgb(static_cast<ColorCompType>(m_symbolColor.red()),
                      static_cast<ColorCompType>(m_symbolColor.green()),
                      static_cast<ColorCompType>(m_symbolColor.blue()));

    for (ccSymbolCloud* cloud : m_overlaySymbols)
        cloud->setTempColor(rgb, true);
    syncOverlaySymbolsToWidget();
}

void DistanceMapGenerationDlg::syncOverlaySymbolsToWidget() {
    if (!m_mapWidget) return;

    QVector<qSRAMapWidget::Symbol> symbols;
    for (const ccSymbolCloud* cloud : m_overlaySymbols) {
        if (!cloud->isVisible()) continue;
        double symbolSize = cloud->getSymbolSize();
        for (unsigned i = 0; i < cloud->size(); ++i) {
            const CCVector3* P = cloud->getPoint(i);
            qSRAMapWidget::Symbol sym;
            sym.x = P->x;
            sym.y = P->y;
            sym.size = symbolSize;
            sym.color = m_symbolColor;
            QString lbl = cloud->getLabel(i);
            if (!lbl.isNull()) {
                sym.label = lbl;
            }
            symbols.append(sym);
        }
    }
    m_mapWidget->setOverlaySymbols(symbols);
}

void DistanceMapGenerationDlg::overlayGridColorChanged() {
    ccQtHelpers::SetButtonColor(gridColorButton, m_gridColor);

    if (m_mapWidget) m_mapWidget->setGridColor(m_gridColor);
}

void DistanceMapGenerationDlg::labelFontSizeChanged(int) {
    if (m_mapWidget) m_mapWidget->setLabelFontSize(fontSizeSpinBox->value());
}

void DistanceMapGenerationDlg::labelPrecisionChanged(int /*prec*/) {
    updateOverlayGrid();
}

void DistanceMapGenerationDlg::colorRampStepsChanged(int) {
    colorScaleChanged(-1);  // dummy index, not used
}

void DistanceMapGenerationDlg::updateOverlayGrid() {
    toggleOverlayGrid(overlayGridGroupBox->isChecked());
}

void DistanceMapGenerationDlg::toggleOverlayGrid(bool state) {
    if (!m_mapWidget) return;

    const bool showGrid = state && overlayGridGroupBox->isChecked();
    m_mapWidget->setGridVisible(showGrid);
    m_mapWidget->setGridColor(m_gridColor);

    QVector<qSRAMapWidget::Label> xLabels;
    QVector<qSRAMapWidget::Label> yLabels;

    if (showGrid && m_map) {
        double xMin_rad, xMax_rad, xStep_rad;
        getGridXValues(xMin_rad, xMax_rad, xStep_rad, ANG_RAD);
        double scaleXStep_rad =
                getSpinboxAngularValue(scaleXStepDoubleSpinBox, ANG_RAD);

        double yMin, yMax, yStep;
        getGridYValues(yMin, yMax, yStep, ANG_RAD);
        double scaleYStep = getScaleYStep(ANG_RAD);

        if (scaleXStep_rad == 0 || scaleYStep == 0) {
            if (m_app)
                m_app->dispToConsole(
                        QString("Internal error: invalid step values?!"),
                        ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
            m_mapWidget->setXLabels(xLabels);
            m_mapWidget->setYLabels(yLabels);
            return;
        }

        m_mapWidget->setGridSteps(scaleXStep_rad, scaleYStep);

        const unsigned xStepCount = static_cast<unsigned>(
                ceil(std::max(xMax_rad - xMin_rad, 0.0) / scaleXStep_rad));
        const unsigned yStepCount = static_cast<unsigned>(
                ceil(std::max(yMax - yMin, 0.0) / scaleYStep));

        if (xScaleCheckBox->isChecked()) {
            const QString angularUnitsStr = getCondensedAngularUnitString();
            const int prec = precisionSpinBox->value();
            const int xPrec = (m_angularUnits == ANG_RAD) ? prec : 0;
            for (unsigned i = 0; i <= xStepCount; ++i) {
                const double angle_rad =
                        xMin_rad + static_cast<double>(i) * scaleXStep_rad;
                qSRAMapWidget::Label lbl;
                lbl.position = angle_rad;
                lbl.text = QString("%1%2")
                                   .arg(ConvertAngleFromRad(angle_rad,
                                                            m_angularUnits),
                                        0, 'f', xPrec)
                                   .arg(angularUnitsStr);
                xLabels.push_back(lbl);
            }
        }

        if (yScaleCheckBox->isChecked() &&
            getProjectionMode() != PROJ_CONICAL) {
            const int prec = precisionSpinBox->value();
            for (unsigned i = 0; i <= yStepCount; ++i) {
                const double y = yMin + static_cast<double>(i) * scaleYStep;
                qSRAMapWidget::Label lbl;
                lbl.position = y;
                lbl.text = QString("%1 %2")
                                   .arg(y, 0, 'f', prec)
                                   .arg(getHeightUnitString());
                yLabels.push_back(lbl);
            }
        }
    }

    m_mapWidget->setXLabels(xLabels);
    m_mapWidget->setYLabels(yLabels);
}

void DistanceMapGenerationDlg::changeGridColor() {
    QColor newCol = QColorDialog::getColor(m_gridColor, this);
    if (!newCol.isValid()) return;

    m_gridColor = newCol;

    overlayGridColorChanged();
}

void DistanceMapGenerationDlg::changeSymbolColor() {
    QColor newCol = QColorDialog::getColor(m_symbolColor, this);
    if (!newCol.isValid()) return;

    m_symbolColor = newCol;

    overlaySymbolsColorChanged();
}

void DistanceMapGenerationDlg::toggleColorScaleDisplay(bool state) {
    if (m_mapWidget) m_mapWidget->setColorScaleVisible(state);
}

void DistanceMapGenerationDlg::updateVolumes() {
    if (getProjectionMode() == PROJ_CONICAL) {
        volumeTextEdit->setText("Cylindrical projection mode only!");
        return;
    }

    if (m_map && m_profile) {
        DistanceMapGenerationTool::Measures surfaces;
        DistanceMapGenerationTool::Measures volumes;
        if (DistanceMapGenerationTool::ComputeSurfacesAndVolumes(
                    m_map, m_profile, surfaces, volumes)) {
            QLocale locale(QLocale::English);
            QString text;
            text.append(QString("[Theoretical]\n"));
            text.append(QString("surface = %1\n")
                                .arg(locale.toString(surfaces.theoretical)));
            text.append(QString("volume = %1\n")
                                .arg(locale.toString(volumes.theoretical)));
            text.append(QString("\n"));
            text.append(QString("[Actual]\n"));
            text.append(QString("Surface: %1\n")
                                .arg(locale.toString(surfaces.total)));
            text.append(QString("Volume: %1\n")
                                .arg(locale.toString(volumes.total)));
            text.append(QString("\n"));
            text.append(QString("Positive (deviations) surface:\n%1\n")
                                .arg(locale.toString(surfaces.positive)));
            text.append(QString("Negative (deviations) surface:\n%1\n")
                                .arg(locale.toString(surfaces.negative)));
            text.append(QString("\n"));
            text.append(QString("Positive volume (gain of matter):\n%1\n")
                                .arg(locale.toString(volumes.positive)));
            text.append(QString("Negative volume (loss of matter):\n%1\n")
                                .arg(locale.toString(volumes.negative)));
            text.append(QString("Sum:\n%1\n")
                                .arg(locale.toString(volumes.positive +
                                                     volumes.negative)));
            volumeTextEdit->setText(text);
        } else {
            volumeTextEdit->setText("Volume(s) computation failed!");
        }
    } else {
        if (!m_map)
            volumeTextEdit->setText("No map!");
        else
            volumeTextEdit->setText("No profile defined!");
    }
}

void DistanceMapGenerationDlg::initFromPersistentSettings() {
    QSettings settings;
    settings.beginGroup("DistanceMapGenerationDialog");

    // read parameters
    double conicSpanRatio = settings.value("conicSpanRatio",
                                           conicSpanRatioDoubleSpinBox->value())
                                    .toDouble();
    int angularUnit =
            settings.value("angularUnit", angularUnitComboBox->currentIndex())
                    .toInt();
    QString heightUnit =
            settings.value("heightUnit", heightUnitLineEdit->text()).toString();
    double angularStep =
            settings.value("angularStep", xStepDoubleSpinBox->value())
                    .toDouble();
    double heightStep =
            settings.value("heightStep", hStepDoubleSpinBox->value())
                    .toDouble();
    double latitudeStep =
            settings.value("latitudeStep", latStepDoubleSpinBox->value())
                    .toDouble();
    double scaleAngularStep =
            settings.value("scaleAngularStep", scaleXStepDoubleSpinBox->value())
                    .toDouble();
    double scaleHeightStep =
            settings.value("scaleHeightStep", scaleHStepDoubleSpinBox->value())
                    .toDouble();
    double scaleLatitudeStep =
            settings.value("scaleLatitudeStep",
                           scaleLatStepDoubleSpinBox->value())
                    .toDouble();
    bool ccw = settings.value("CCW", ccwCheckBox->isChecked()).toBool();
    int fillStrategy = settings.value("fillStrategy",
                                      fillingStrategyComboxBox->currentIndex())
                               .toBool();
    int emptyCells =
            settings.value("emptyCells", emptyCellsComboBox->currentIndex())
                    .toInt();
    bool showOverlayGrid =
            settings.value("showOverlayGrid", overlayGridGroupBox->isChecked())
                    .toBool();
    bool showXScale =
            settings.value("showXScale", xScaleCheckBox->isChecked()).toBool();
    bool showYScale =
            settings.value("showYScale", yScaleCheckBox->isChecked()).toBool();
    bool showColorScale = settings.value("showColorScale",
                                         displayColorScaleCheckBox->isChecked())
                                  .toBool();
    QString uuid = settings.value("colorScale", QString()).toString();
    int colorScaleSteps =
            settings.value("colorScaleSteps", colorScaleStepsSpinBox->value())
                    .toInt();
    int symbolSize =
            settings.value("symbolSize", symbolSizeSpinBox->value()).toInt();
    int fontSize = settings.value("fontSize", fontSizeSpinBox->value()).toInt();

    // apply parameters
    conicSpanRatioDoubleSpinBox->setValue(conicSpanRatio);
    angularUnitComboBox->setCurrentIndex(angularUnit);
    angularUnitChanged(angularUnit);  // force update
    heightUnitLineEdit->setText(heightUnit);
    updateHeightUnits();  // force update
    xStepDoubleSpinBox->setValue(angularStep);
    hStepDoubleSpinBox->setValue(heightStep);
    latStepDoubleSpinBox->setValue(latitudeStep);
    scaleXStepDoubleSpinBox->setValue(scaleAngularStep);
    scaleHStepDoubleSpinBox->setValue(scaleHeightStep);
    scaleLatStepDoubleSpinBox->setValue(scaleLatitudeStep);
    ccwCheckBox->setChecked(ccw);
    fillingStrategyComboxBox->setCurrentIndex(fillStrategy);
    emptyCellsComboBox->setCurrentIndex(emptyCells);
    overlayGridGroupBox->setChecked(showOverlayGrid);
    xScaleCheckBox->setChecked(showXScale);
    yScaleCheckBox->setChecked(showYScale);
    displayColorScaleCheckBox->setChecked(showColorScale);
    if (m_colorScaleSelector && !uuid.isNull())
        m_colorScaleSelector->setSelectedScale(uuid);
    colorScaleStepsSpinBox->setValue(colorScaleSteps);
    symbolSizeSpinBox->setValue(symbolSize);
    fontSizeSpinBox->setValue(fontSize);

    settings.endGroup();
}

void DistanceMapGenerationDlg::saveToPersistentSettings() {
    QSettings settings;
    settings.beginGroup("DistanceMapGenerationDialog");

    // write parameters
    settings.setValue("conicSpanRatio", conicSpanRatioDoubleSpinBox->value());
    settings.setValue("angularUnit", angularUnitComboBox->currentIndex());
    settings.setValue("heightUnit", heightUnitLineEdit->text());
    settings.setValue("angularStep", xStepDoubleSpinBox->value());
    settings.setValue("heightStep", hStepDoubleSpinBox->value());
    settings.setValue("latitudeStep", latStepDoubleSpinBox->value());
    settings.setValue("scaleAngularStep", scaleXStepDoubleSpinBox->value());
    settings.setValue("scaleHeightStep", scaleHStepDoubleSpinBox->value());
    settings.setValue("scaleLatitudeStep", scaleLatStepDoubleSpinBox->value());
    settings.setValue("CCW", ccwCheckBox->isChecked());
    settings.setValue("fillStrategy", fillingStrategyComboxBox->currentIndex());
    settings.setValue("emptyCells", emptyCellsComboBox->currentIndex());
    settings.setValue("showOverlayGrid", overlayGridGroupBox->isChecked());
    settings.setValue("showXScale", xScaleCheckBox->isChecked());
    settings.setValue("showYScale", yScaleCheckBox->isChecked());
    settings.setValue("showColorScale", displayColorScaleCheckBox->isChecked());
    if (m_colorScaleSelector) {
        ccColorScale::Shared colorScale =
                m_colorScaleSelector->getSelectedScale();
        if (colorScale) settings.setValue("colorScale", colorScale->getUuid());
    }
    settings.setValue("colorScaleSteps", colorScaleStepsSpinBox->value());
    settings.setValue("symbolSize", symbolSizeSpinBox->value());
    settings.setValue("fontSize", fontSizeSpinBox->value());

    settings.endGroup();
}
