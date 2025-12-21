// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvFilterConfigDialog.h"

// CV_CORE_LIB
#include <CVLog.h>

// VTK
#include <vtkPolyData.h>

// Qt
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QStackedWidget>
#include <QVBoxLayout>

//-----------------------------------------------------------------------------
cvFilterConfigDialog::cvFilterConfigDialog(
        cvSelectionFilter::FilterType filterType,
        vtkPolyData* polyData,
        QWidget* parent)
    : QDialog(parent),
      m_filterType(filterType),
      m_polyData(polyData),
      m_filterTypeCombo(nullptr),
      m_parameterStack(nullptr),
      m_previewButton(nullptr),
      m_okButton(nullptr),
      m_cancelButton(nullptr) {
    setWindowTitle(tr("Configure Selection Filter"));
    setModal(true);
    setupUI();
    loadParameters();
}

//-----------------------------------------------------------------------------
cvFilterConfigDialog::~cvFilterConfigDialog() {}

//-----------------------------------------------------------------------------
void cvFilterConfigDialog::setupUI() {
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(10, 10, 10, 10);
    mainLayout->setSpacing(10);

    // Filter type selection
    QHBoxLayout* typeLayout = new QHBoxLayout();
    typeLayout->addWidget(new QLabel(tr("Filter Type:")));
    
    m_filterTypeCombo = new QComboBox();
    m_filterTypeCombo->addItem(tr("Attribute (Range)"),
                                static_cast<int>(cvSelectionFilter::ATTRIBUTE_RANGE));
    m_filterTypeCombo->addItem(tr("Geometric (Area)"),
                                static_cast<int>(cvSelectionFilter::GEOMETRIC_AREA));
    m_filterTypeCombo->addItem(tr("Geometric (Angle)"),
                                static_cast<int>(cvSelectionFilter::GEOMETRIC_ANGLE));
    m_filterTypeCombo->addItem(tr("Spatial (Bounding Box)"),
                                static_cast<int>(cvSelectionFilter::SPATIAL_BBOX));
    m_filterTypeCombo->addItem(tr("Spatial (Distance)"),
                                static_cast<int>(cvSelectionFilter::SPATIAL_DISTANCE));
    m_filterTypeCombo->addItem(tr("Topology (Neighbors)"),
                                static_cast<int>(cvSelectionFilter::TOPOLOGY_NEIGHBORS));
    
    connect(m_filterTypeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &cvFilterConfigDialog::onFilterTypeChanged);
    
    typeLayout->addWidget(m_filterTypeCombo);
    mainLayout->addLayout(typeLayout);

    // Parameter stack (different UI for each filter type)
    m_parameterStack = new QStackedWidget();
    
    setupAttributeFilterUI();
    setupGeometricFilterUI();
    setupSpatialFilterUI();
    setupTopologyFilterUI();
    
    mainLayout->addWidget(m_parameterStack);

    // Preview and buttons
    QHBoxLayout* buttonLayout = new QHBoxLayout();
    
    m_previewButton = new QPushButton(tr("Preview"));
    m_previewButton->setEnabled(m_polyData != nullptr);
    connect(m_previewButton, &QPushButton::clicked,
            this, &cvFilterConfigDialog::onPreviewClicked);
    buttonLayout->addWidget(m_previewButton);
    
    buttonLayout->addStretch();
    
    QDialogButtonBox* dialogButtons = new QDialogButtonBox(
            QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(dialogButtons, &QDialogButtonBox::accepted,
            this, &cvFilterConfigDialog::onAccepted);
    connect(dialogButtons, &QDialogButtonBox::rejected,
            this, &cvFilterConfigDialog::onRejected);
    buttonLayout->addWidget(dialogButtons);
    
    mainLayout->addLayout(buttonLayout);

    // Set initial filter type
    int index = m_filterTypeCombo->findData(static_cast<int>(m_filterType));
    if (index >= 0) {
        m_filterTypeCombo->setCurrentIndex(index);
    }
}

//-----------------------------------------------------------------------------
void cvFilterConfigDialog::setupAttributeFilterUI() {
    QWidget* attributeWidget = new QWidget();
    QFormLayout* layout = new QFormLayout(attributeWidget);

    // Attribute selection
    m_attributeCombo = new QComboBox();
    m_attributeCombo->addItem(tr("Intensity"));
    m_attributeCombo->addItem(tr("RGB"));
    m_attributeCombo->addItem(tr("Normal"));
    m_attributeCombo->addItem(tr("Curvature"));
    m_attributeCombo->addItem(tr("Custom"));
    layout->addRow(tr("Attribute:"), m_attributeCombo);

    // Operator
    m_operatorCombo = new QComboBox();
    m_operatorCombo->addItem(tr("Greater Than"), ">");
    m_operatorCombo->addItem(tr("Less Than"), "<");
    m_operatorCombo->addItem(tr("Equal To"), "==");
    m_operatorCombo->addItem(tr("Between"), "between");
    layout->addRow(tr("Operator:"), m_operatorCombo);

    // Min/Max values
    m_minValueSpin = new QDoubleSpinBox();
    m_minValueSpin->setRange(-1e6, 1e6);
    m_minValueSpin->setDecimals(3);
    m_minValueSpin->setValue(0.0);
    layout->addRow(tr("Min Value:"), m_minValueSpin);

    m_maxValueSpin = new QDoubleSpinBox();
    m_maxValueSpin->setRange(-1e6, 1e6);
    m_maxValueSpin->setDecimals(3);
    m_maxValueSpin->setValue(100.0);
    layout->addRow(tr("Max Value:"), m_maxValueSpin);

    m_parameterStack->addWidget(attributeWidget);
}

//-----------------------------------------------------------------------------
void cvFilterConfigDialog::setupGeometricFilterUI() {
    QWidget* geometricWidget = new QWidget();
    QVBoxLayout* mainLayout = new QVBoxLayout(geometricWidget);

    // Sphere parameters
    QGroupBox* sphereGroup = new QGroupBox(tr("Sphere Filter"));
    QFormLayout* sphereLayout = new QFormLayout(sphereGroup);

    m_centerXSpin = new QDoubleSpinBox();
    m_centerXSpin->setRange(-1e6, 1e6);
    m_centerXSpin->setDecimals(3);
    sphereLayout->addRow(tr("Center X:"), m_centerXSpin);

    m_centerYSpin = new QDoubleSpinBox();
    m_centerYSpin->setRange(-1e6, 1e6);
    m_centerYSpin->setDecimals(3);
    sphereLayout->addRow(tr("Center Y:"), m_centerYSpin);

    m_centerZSpin = new QDoubleSpinBox();
    m_centerZSpin->setRange(-1e6, 1e6);
    m_centerZSpin->setDecimals(3);
    sphereLayout->addRow(tr("Center Z:"), m_centerZSpin);

    m_radiusSpin = new QDoubleSpinBox();
    m_radiusSpin->setRange(0.0, 1e6);
    m_radiusSpin->setDecimals(3);
    m_radiusSpin->setValue(1.0);
    sphereLayout->addRow(tr("Radius:"), m_radiusSpin);

    mainLayout->addWidget(sphereGroup);

    // Bounding box parameters
    QGroupBox* boxGroup = new QGroupBox(tr("Bounding Box Filter"));
    QFormLayout* boxLayout = new QFormLayout(boxGroup);

    m_xMinSpin = new QDoubleSpinBox();
    m_xMinSpin->setRange(-1e6, 1e6);
    m_xMinSpin->setDecimals(3);
    boxLayout->addRow(tr("X Min:"), m_xMinSpin);

    m_xMaxSpin = new QDoubleSpinBox();
    m_xMaxSpin->setRange(-1e6, 1e6);
    m_xMaxSpin->setDecimals(3);
    boxLayout->addRow(tr("X Max:"), m_xMaxSpin);

    m_yMinSpin = new QDoubleSpinBox();
    m_yMinSpin->setRange(-1e6, 1e6);
    m_yMinSpin->setDecimals(3);
    boxLayout->addRow(tr("Y Min:"), m_yMinSpin);

    m_yMaxSpin = new QDoubleSpinBox();
    m_yMaxSpin->setRange(-1e6, 1e6);
    m_yMaxSpin->setDecimals(3);
    boxLayout->addRow(tr("Y Max:"), m_yMaxSpin);

    m_zMinSpin = new QDoubleSpinBox();
    m_zMinSpin->setRange(-1e6, 1e6);
    m_zMinSpin->setDecimals(3);
    boxLayout->addRow(tr("Z Min:"), m_zMinSpin);

    m_zMaxSpin = new QDoubleSpinBox();
    m_zMaxSpin->setRange(-1e6, 1e6);
    m_zMaxSpin->setDecimals(3);
    boxLayout->addRow(tr("Z Max:"), m_zMaxSpin);

    mainLayout->addWidget(boxGroup);
    mainLayout->addStretch();

    m_parameterStack->addWidget(geometricWidget);
}

//-----------------------------------------------------------------------------
void cvFilterConfigDialog::setupSpatialFilterUI() {
    QWidget* spatialWidget = new QWidget();
    QFormLayout* layout = new QFormLayout(spatialWidget);

    // Proximity type
    m_proximityTypeCombo = new QComboBox();
    m_proximityTypeCombo->addItem(tr("Within Distance"));
    m_proximityTypeCombo->addItem(tr("Nearest Neighbors"));
    layout->addRow(tr("Proximity Type:"), m_proximityTypeCombo);

    // Distance/count
    m_distanceSpin = new QDoubleSpinBox();
    m_distanceSpin->setRange(0.0, 1e6);
    m_distanceSpin->setDecimals(3);
    m_distanceSpin->setValue(1.0);
    layout->addRow(tr("Distance/Count:"), m_distanceSpin);

    m_parameterStack->addWidget(spatialWidget);
}

//-----------------------------------------------------------------------------
void cvFilterConfigDialog::setupTopologyFilterUI() {
    QWidget* topologyWidget = new QWidget();
    QFormLayout* layout = new QFormLayout(topologyWidget);

    // Topology type
    m_topologyTypeCombo = new QComboBox();
    m_topologyTypeCombo->addItem(tr("Connected Components"));
    m_topologyTypeCombo->addItem(tr("Boundary"));
    m_topologyTypeCombo->addItem(tr("Manifold"));
    m_topologyTypeCombo->addItem(tr("Non-Manifold"));
    layout->addRow(tr("Topology Type:"), m_topologyTypeCombo);

    m_parameterStack->addWidget(topologyWidget);
}

//-----------------------------------------------------------------------------
void cvFilterConfigDialog::onFilterTypeChanged(int index) {
    m_filterType = static_cast<cvSelectionFilter::FilterType>(
            m_filterTypeCombo->itemData(index).toInt());
    
    // Switch to appropriate parameter UI
    switch (m_filterType) {
        case cvSelectionFilter::ATTRIBUTE_RANGE:
            m_parameterStack->setCurrentIndex(0);
            break;
        case cvSelectionFilter::GEOMETRIC_AREA:
        case cvSelectionFilter::GEOMETRIC_ANGLE:
        case cvSelectionFilter::SPATIAL_BBOX:
            m_parameterStack->setCurrentIndex(1);
            break;
        case cvSelectionFilter::SPATIAL_DISTANCE:
            m_parameterStack->setCurrentIndex(2);
            break;
        case cvSelectionFilter::TOPOLOGY_NEIGHBORS:
            m_parameterStack->setCurrentIndex(3);
            break;
    }
}

//-----------------------------------------------------------------------------
void cvFilterConfigDialog::onPreviewClicked() {
    // Preview functionality: Apply filter temporarily without closing dialog
    // This would require:
    // 1. Validate current parameters
    // 2. Apply filter to selection
    // 3. Update visualization with preview highlighting
    // 4. Provide "Revert" option to undo preview
    // Note: Not critical for initial release, can be added in future version
    CVLog::Print("[cvFilterConfigDialog] Preview feature available in future version");
    QMessageBox::information(this, tr("Preview"), 
                            tr("Preview functionality will be available in a future version.\n"
                               "For now, please use OK to apply the filter."));
}

//-----------------------------------------------------------------------------
void cvFilterConfigDialog::onAccepted() {
    if (validateParameters()) {
        saveParameters();
        accept();
    }
}

//-----------------------------------------------------------------------------
void cvFilterConfigDialog::onRejected() {
    reject();
}

//-----------------------------------------------------------------------------
void cvFilterConfigDialog::loadParameters() {
    // Load default or saved parameters
    // This would typically load from QSettings or similar
}

//-----------------------------------------------------------------------------
void cvFilterConfigDialog::saveParameters() {
    m_parameters.clear();

    switch (m_filterType) {
        case cvSelectionFilter::ATTRIBUTE_RANGE:
            m_parameters["attribute"] = m_attributeCombo->currentText();
            m_parameters["operator"] = m_operatorCombo->currentData().toString();
            m_parameters["minValue"] = m_minValueSpin->value();
            m_parameters["maxValue"] = m_maxValueSpin->value();
            break;

        case cvSelectionFilter::GEOMETRIC_AREA:
        case cvSelectionFilter::GEOMETRIC_ANGLE:
            m_parameters["minValue"] = m_minValueSpin->value();
            m_parameters["maxValue"] = m_maxValueSpin->value();
            break;

        case cvSelectionFilter::SPATIAL_BBOX:
            m_parameters["xMin"] = m_xMinSpin->value();
            m_parameters["xMax"] = m_xMaxSpin->value();
            m_parameters["yMin"] = m_yMinSpin->value();
            m_parameters["yMax"] = m_yMaxSpin->value();
            m_parameters["zMin"] = m_zMinSpin->value();
            m_parameters["zMax"] = m_zMaxSpin->value();
            break;

        case cvSelectionFilter::SPATIAL_DISTANCE:
            m_parameters["distance"] = m_distanceSpin->value();
            break;

        case cvSelectionFilter::TOPOLOGY_NEIGHBORS:
            m_parameters["neighborCount"] = static_cast<int>(m_distanceSpin->value());
            break;
    }

    CVLog::Print(QString("[cvFilterConfigDialog] Saved %1 parameters")
                         .arg(m_parameters.size()));
}

//-----------------------------------------------------------------------------
bool cvFilterConfigDialog::validateParameters() {
    // Basic validation
    switch (m_filterType) {
        case cvSelectionFilter::ATTRIBUTE_RANGE:
        case cvSelectionFilter::GEOMETRIC_AREA:
        case cvSelectionFilter::GEOMETRIC_ANGLE:
            if (m_minValueSpin->value() > m_maxValueSpin->value()) {
                CVLog::Warning("[cvFilterConfigDialog] Min value must be <= Max value");
                return false;
            }
            break;

        case cvSelectionFilter::SPATIAL_DISTANCE:
            if (m_distanceSpin->value() <= 0.0) {
                CVLog::Warning("[cvFilterConfigDialog] Distance must be positive");
                return false;
            }
            break;

        case cvSelectionFilter::SPATIAL_BBOX:
            if (m_xMinSpin->value() > m_xMaxSpin->value() ||
                m_yMinSpin->value() > m_yMaxSpin->value() ||
                m_zMinSpin->value() > m_zMaxSpin->value()) {
                CVLog::Warning("[cvFilterConfigDialog] Box bounds invalid");
                return false;
            }
            break;

        default:
            break;
    }

    return true;
}

//-----------------------------------------------------------------------------
QMap<QString, QVariant> cvFilterConfigDialog::getParameters() const {
    return m_parameters;
}

//-----------------------------------------------------------------------------
cvSelectionFilter::FilterType cvFilterConfigDialog::getFilterType() const {
    return m_filterType;
}

