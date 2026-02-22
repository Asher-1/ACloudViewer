// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvDistanceTool.h"

#include <VtkUtils/distancewidgetobserver.h>
#include <VtkUtils/signalblocker.h>
#include <VtkUtils/vtkutils.h>
#include <vtkAxisActor2D.h>
#include <vtkCommand.h>
#include <vtkHandleRepresentation.h>
#include <vtkLineRepresentation.h>
#include <vtkMath.h>
#include <vtkPointHandleRepresentation3D.h>
#include <vtkProperty.h>
#include <vtkProperty2D.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkTextProperty.h>

#include <QApplication>
#include <QLayout>
#include <QLayoutItem>
#include <QShortcut>
#include <QSizePolicy>
#include <algorithm>

#include "Tools/PickingTools/cvPointPickingHelper.h"
#include "VTKExtensions/ConstrainedWidgets/cvConstrainedDistanceWidget.h"
#include "VTKExtensions/ConstrainedWidgets/cvConstrainedLineRepresentation.h"
#include "VTKExtensions/ConstrainedWidgets/cvCustomAxisHandleRepresentation.h"
#include "cvMeasurementToolsCommon.h"

// CV_CORE_LIB
#include <CVLog.h>

// CV_DB_LIB
#include <ecv2DLabel.h>
#include <ecvBBox.h>
#include <ecvGenericMesh.h>
#include <ecvGenericPointCloud.h>
#include <ecvHObject.h>
#include <ecvHObjectCaster.h>
#include <ecvPointCloud.h>

namespace {

using namespace cvMeasurementTools;

//! Configure line representation with ParaView-style properties (3D only)
void configureLineRepresentation(cvConstrainedLineRepresentation* rep) {
    if (!rep) return;

    // Configure 3D handles
    auto* h1 = vtkPointHandleRepresentation3D::SafeDownCast(
            rep->GetPoint1Representation());
    auto* h2 = vtkPointHandleRepresentation3D::SafeDownCast(
            rep->GetPoint2Representation());
    auto* hLine = vtkPointHandleRepresentation3D::SafeDownCast(
            rep->GetLineHandleRepresentation());
    configureHandle3D(h1);
    configureHandle3D(h2);
    configureHandle3D(hLine);

    // Configure line properties (matching ParaView's LineProperty)
    if (auto* lineProp = rep->GetLineProperty()) {
        lineProp->SetColor(FOREGROUND_COLOR[0], FOREGROUND_COLOR[1],
                           FOREGROUND_COLOR[2]);
        lineProp->SetLineWidth(2.0);  // ParaView default line width
        lineProp->SetAmbient(1.0);    // ParaView sets ambient to 1.0
    }

    // Configure distance label text properties for better readability
    if (auto* axis = rep->GetAxisActor()) {
        // Configure the Title text property (this displays the distance value)
        if (auto* titleProp = axis->GetTitleTextProperty()) {
            titleProp->SetFontSize(
                    6);               // Default font size for distance display
            titleProp->SetBold(0);    // Not bold for better readability
            titleProp->SetShadow(1);  // Add shadow for better visibility
            titleProp->SetColor(1.0, 1.0, 1.0);  // White text
        }

        // Configure the Label text property (for ruler tick labels)
        if (auto* labelProp = axis->GetLabelTextProperty()) {
            labelProp->SetFontSize(
                    5);  // Slightly smaller font size for tick labels
            labelProp->SetBold(0);
            labelProp->SetShadow(1);
            labelProp->SetColor(1.0, 1.0, 1.0);
        }
    }

    // Configure distance display and ruler features (custom functionality)
    rep->SetShowLabel(1);            // Show distance label by default
    rep->SetLabelFormat("%-#6.3g");  // Distance format (ParaView default)
    rep->SetRulerMode(0);            // Ruler mode off by default
    rep->SetRulerDistance(1.0);      // Default tick spacing
    rep->SetNumberOfRulerTicks(5);   // Default number of ticks
    rep->SetScale(1.0);              // Default scale factor
}

}  // anonymous namespace

cvDistanceTool::cvDistanceTool(QWidget* parent)
    : cvGenericMeasurementTool(parent), m_configUi(nullptr) {
    setWindowTitle(tr("Distance Measurement Tool"));
}

cvDistanceTool::~cvDistanceTool() {
    // CRITICAL: Explicitly hide and cleanup widget/representation before
    // destruction
    if (m_widget) {
        m_widget->Off();          // Turn off widget
        m_widget->SetEnabled(0);  // Disable widget
    }

    // Explicitly hide all representation elements
    if (m_rep) {
        m_rep->SetVisibility(0);  // Hide everything

        // Force immediate render to clear visual elements
        if (m_interactor && m_interactor->GetRenderWindow()) {
            m_interactor->GetRenderWindow()->Render();
        }
    }

    if (m_configUi) {
        delete m_configUi;
        m_configUi = nullptr;
    }
}

void cvDistanceTool::initTool() {
    VtkUtils::vtkInitOnce(m_rep);

    // Use constrained widget - automatically supports XYZL shortcuts
    m_widget = vtkSmartPointer<cvConstrainedDistanceWidget>::New();

    // Set representation BEFORE calling SetInteractor/SetRenderer
    m_widget->SetRepresentation(m_rep);

    if (m_interactor) {
        m_widget->SetInteractor(m_interactor);
    }
    if (m_renderer) {
        m_rep->SetRenderer(m_renderer);
    }

    // Following ParaView's approach:
    // 1. InstantiateHandleRepresentation is already called in
    // vtkLineRepresentation constructor
    // 2. Replace the default handles with custom axis handles for full XYZL
    // support Use template version for type-safe creation
    m_rep->ReplaceHandleRepresentationsTyped<
            cvCustomAxisHandleRepresentation>();

    // 2. Configure appearance AFTER instantiation but BEFORE enabling
    configureLineRepresentation(m_rep);  // 3D mode only

    // 3. Apply default green color (override configure defaults)
    if (auto* lineProp = m_rep->GetLineProperty()) {
        lineProp->SetColor(m_currentColor[0], m_currentColor[1],
                           m_currentColor[2]);
    }
    if (auto* selectedLineProp = m_rep->GetSelectedLineProperty()) {
        selectedLineProp->SetColor(m_currentColor[0], m_currentColor[1],
                                   m_currentColor[2]);
    }

    // 4. Set initial positions before building representation
    // This ensures the widget is visible outside the object from the start
    double defaultPos1[3] = {0.0, 0.0, 0.0};
    double defaultPos2[3] = {1.0, 0.0, 0.0};

    if (m_entity && m_entity->getBB_recursive().isValid()) {
        const ccBBox& bbox = m_entity->getBB_recursive();
        CCVector3 center = bbox.getCenter();
        CCVector3 diag = bbox.getDiagVec();

        // Calculate offset based on the bounding box diagonal length
        // Use a larger offset (30%) to ensure visibility outside any object
        // orientation
        double diagLength = diag.norm();
        double offset = diagLength * 0.3;

        // Ensure minimum offset for very small objects
        if (offset < 0.5) {
            offset = 0.5;
        }

        // Place the ruler above and in front of the object for better
        // visibility Use offset in Y and Z directions to avoid objects in any
        // orientation
        defaultPos1[0] = center.x - diag.x * 0.25;
        defaultPos1[1] = center.y + offset;  // Offset in Y direction
        defaultPos1[2] = center.z + offset;  // Offset in Z direction

        defaultPos2[0] = center.x + diag.x * 0.25;
        defaultPos2[1] = center.y + offset;  // Offset in Y direction
        defaultPos2[2] = center.z + offset;  // Offset in Z direction
    }

    m_rep->SetPoint1WorldPosition(defaultPos1);
    m_rep->SetPoint2WorldPosition(defaultPos2);

    // 5. Build representation (before updating UI)
    m_rep->BuildRepresentation();

    // 6. Apply font properties to override configureLineRepresentation defaults
    // This ensures user-configured font properties (size, bold, italic, etc.)
    // are applied
    applyFontProperties();

    // 7. Update UI controls with initial positions (if UI is already created)
    if (m_configUi) {
        VtkUtils::SignalBlocker blocker1(m_configUi->point1XSpinBox);
        VtkUtils::SignalBlocker blocker2(m_configUi->point1YSpinBox);
        VtkUtils::SignalBlocker blocker3(m_configUi->point1ZSpinBox);
        VtkUtils::SignalBlocker blocker4(m_configUi->point2XSpinBox);
        VtkUtils::SignalBlocker blocker5(m_configUi->point2YSpinBox);
        VtkUtils::SignalBlocker blocker6(m_configUi->point2ZSpinBox);

        m_configUi->point1XSpinBox->setValue(defaultPos1[0]);
        m_configUi->point1YSpinBox->setValue(defaultPos1[1]);
        m_configUi->point1ZSpinBox->setValue(defaultPos1[2]);
        m_configUi->point2XSpinBox->setValue(defaultPos2[0]);
        m_configUi->point2YSpinBox->setValue(defaultPos2[1]);
        m_configUi->point2ZSpinBox->setValue(defaultPos2[2]);

        // Update distance display to show initial distance
        updateDistanceDisplay();
    }

    // 7. Enable widget
    m_widget->On();

    hookWidget(m_widget);
}

void cvDistanceTool::createUi() {
    // CRITICAL: Only setup base UI once to avoid resetting configLayout
    // Each tool instance has its own m_ui, but setupUi clears all children
    // so we must ensure it's only called once per tool instance
    // Check if base UI is already set up by checking if widget has a layout
    // NOTE: Cannot check m_ui->configLayout directly as it's uninitialized
    // before setupUi()
    if (!m_ui) {
        CVLog::Error("[cvDistanceTool::createUi] m_ui is null!");
        return;
    }
    if (!layout()) {
        m_ui->setupUi(this);
    }

    // CRITICAL: Always clean up existing config UI before creating new one
    // This prevents UI interference when createUi() is called multiple times
    // (e.g., when tool is restarted or switched)
    if (m_configUi && m_ui->configLayout) {
        // Remove all existing widgets from configLayout
        QLayoutItem* item;
        while ((item = m_ui->configLayout->takeAt(0)) != nullptr) {
            if (item->widget()) {
                item->widget()->setParent(nullptr);
                item->widget()->deleteLater();
            }
            delete item;
        }
        delete m_configUi;
        m_configUi = nullptr;
    }

    // Create fresh config UI for this tool instance
    m_configUi = new Ui::DistanceToolDlg;
    QWidget* configWidget = new QWidget(this);
    configWidget->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    m_configUi->setupUi(configWidget);
    m_ui->configLayout->addWidget(configWidget);
    m_ui->groupBox->setTitle(tr("Distance Parameters"));

    m_configUi->distanceSpinBox->setValue(1.0);

#ifdef Q_OS_MAC
    m_configUi->instructionLabel->setText(
            m_configUi->instructionLabel->text().replace("Ctrl", "Cmd"));
#endif

    // Ensure Tips label wraps text properly
    if (m_configUi->instructionLabel) {
        m_configUi->instructionLabel->setSizePolicy(QSizePolicy::Preferred,
                                                    QSizePolicy::Minimum);
        m_configUi->instructionLabel->setWordWrap(true);
    }

    // Collapsible Tips: toggle label visibility via group box checkbox
    if (m_configUi->shortcutsTipsGroupBox) {
        m_configUi->instructionLabel->setVisible(
                m_configUi->shortcutsTipsGroupBox->isChecked());
        connect(m_configUi->shortcutsTipsGroupBox, &QGroupBox::toggled,
                m_configUi->instructionLabel, &QWidget::setVisible);
    }

    // Let parent dialog handle sizing via scroll area — no local adjustSize
    this->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);

    connect(m_configUi->point1XSpinBox,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &cvDistanceTool::on_point1XSpinBox_valueChanged);
    connect(m_configUi->point1YSpinBox,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &cvDistanceTool::on_point1YSpinBox_valueChanged);
    connect(m_configUi->point1ZSpinBox,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &cvDistanceTool::on_point1ZSpinBox_valueChanged);
    connect(m_configUi->point2XSpinBox,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &cvDistanceTool::on_point2XSpinBox_valueChanged);
    connect(m_configUi->point2YSpinBox,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &cvDistanceTool::on_point2YSpinBox_valueChanged);
    connect(m_configUi->point2ZSpinBox,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &cvDistanceTool::on_point2ZSpinBox_valueChanged);

    // Connect point picking buttons
    connect(m_configUi->pickPoint1ToolButton, &QToolButton::toggled, this,
            &cvDistanceTool::on_pickPoint1_toggled);
    connect(m_configUi->pickPoint2ToolButton, &QToolButton::toggled, this,
            &cvDistanceTool::on_pickPoint2_toggled);

    // Connect ruler mode controls
    connect(m_configUi->rulerModeCheckBox, &QCheckBox::toggled, this,
            &cvDistanceTool::on_rulerModeCheckBox_toggled);
    connect(m_configUi->rulerDistanceSpinBox,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &cvDistanceTool::on_rulerDistanceSpinBox_valueChanged);
    connect(m_configUi->numberOfTicksSpinBox,
            QOverload<int>::of(&QSpinBox::valueChanged), this,
            &cvDistanceTool::on_numberOfTicksSpinBox_valueChanged);
    connect(m_configUi->scaleSpinBox,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &cvDistanceTool::on_scaleSpinBox_valueChanged);
    connect(m_configUi->labelFormatLineEdit, &QLineEdit::textChanged, this,
            &cvDistanceTool::on_labelFormatLineEdit_textChanged);

    // Connect display options
    connect(m_configUi->widgetVisibilityCheckBox, &QCheckBox::toggled, this,
            &cvDistanceTool::on_widgetVisibilityCheckBox_toggled);
    connect(m_configUi->labelVisibilityCheckBox, &QCheckBox::toggled, this,
            &cvDistanceTool::on_labelVisibilityCheckBox_toggled);
    connect(m_configUi->lineWidthSpinBox,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &cvDistanceTool::on_lineWidthSpinBox_valueChanged);
}

void cvDistanceTool::start() { cvGenericMeasurementTool::start(); }

void cvDistanceTool::reset() {
    // Reset points to default positions (above the bounding box for better
    // visibility and accessibility)
    double defaultPos1[3] = {0.0, 0.0, 0.0};
    double defaultPos2[3] = {1.0, 0.0, 0.0};

    if (m_entity && m_entity->getBB_recursive().isValid()) {
        const ccBBox& bbox = m_entity->getBB_recursive();
        CCVector3 center = bbox.getCenter();
        CCVector3 diag = bbox.getDiagVec();

        // Calculate offset based on the bounding box diagonal length
        // Use a larger offset (30%) to ensure visibility outside any object
        // orientation
        double diagLength = diag.norm();
        double offset = diagLength * 0.3;

        // Ensure minimum offset for very small objects
        if (offset < 0.5) {
            offset = 0.5;
        }

        // Place the ruler above and in front of the object for better
        // visibility Use offset in Y and Z directions to avoid objects in any
        // orientation
        defaultPos1[0] = center.x - diag.x * 0.25;
        defaultPos1[1] = center.y + offset;  // Offset in Y direction
        defaultPos1[2] = center.z + offset;  // Offset in Z direction

        defaultPos2[0] = center.x + diag.x * 0.25;
        defaultPos2[1] = center.y + offset;  // Offset in Y direction
        defaultPos2[2] = center.z + offset;  // Offset in Z direction
    }

    // Reset widget points
    if (m_widget && m_widget->GetEnabled()) {
        m_rep->SetPoint1WorldPosition(defaultPos1);
        m_rep->SetPoint2WorldPosition(defaultPos2);
        m_rep->BuildRepresentation();
        m_widget->Modified();
    }

    // Update UI
    if (m_configUi) {
        VtkUtils::SignalBlocker blocker1(m_configUi->point1XSpinBox);
        VtkUtils::SignalBlocker blocker2(m_configUi->point1YSpinBox);
        VtkUtils::SignalBlocker blocker3(m_configUi->point1ZSpinBox);
        VtkUtils::SignalBlocker blocker4(m_configUi->point2XSpinBox);
        VtkUtils::SignalBlocker blocker5(m_configUi->point2YSpinBox);
        VtkUtils::SignalBlocker blocker6(m_configUi->point2ZSpinBox);

        m_configUi->point1XSpinBox->setValue(defaultPos1[0]);
        m_configUi->point1YSpinBox->setValue(defaultPos1[1]);
        m_configUi->point1ZSpinBox->setValue(defaultPos1[2]);
        m_configUi->point2XSpinBox->setValue(defaultPos2[0]);
        m_configUi->point2YSpinBox->setValue(defaultPos2[1]);
        m_configUi->point2ZSpinBox->setValue(defaultPos2[2]);
    }

    update();
    emit measurementValueChanged();
}

void cvDistanceTool::showWidget(bool state) {
    if (m_widget) {
        if (state) {
            m_widget->On();
        } else {
            m_widget->Off();
        }
    }
    update();
}

ccHObject* cvDistanceTool::getOutput() {
    // Export distance measurement as cc2DLabel with 2 points
    // Returns a new cc2DLabel that can be added to the DB tree

    if (!m_entity) {
        CVLog::Warning(
                "[cvDistanceTool::getOutput] No entity associated with this "
                "measurement");
        return nullptr;
    }

    // Get the point coordinates
    double p1[3], p2[3];
    getPoint1(p1);
    getPoint2(p2);

    // Try to get the associated point cloud
    ccGenericPointCloud* cloud = nullptr;
    if (m_entity->isKindOf(CV_TYPES::POINT_CLOUD)) {
        cloud = static_cast<ccGenericPointCloud*>(m_entity);
    } else if (m_entity->isKindOf(CV_TYPES::MESH)) {
        ccGenericMesh* mesh = static_cast<ccGenericMesh*>(m_entity);
        if (mesh) {
            cloud = mesh->getAssociatedCloud();
        }
    }

    if (!cloud || cloud->size() == 0) {
        CVLog::Warning(
                "[cvDistanceTool::getOutput] Could not find associated point "
                "cloud or cloud is empty");
        return nullptr;
    }

    // Convert distance tool's exact 3D coordinates to CCVector3
    CCVector3 point1(static_cast<PointCoordinateType>(p1[0]),
                     static_cast<PointCoordinateType>(p1[1]),
                     static_cast<PointCoordinateType>(p1[2]));
    CCVector3 point2(static_cast<PointCoordinateType>(p2[0]),
                     static_cast<PointCoordinateType>(p2[1]),
                     static_cast<PointCoordinateType>(p2[2]));

    // Find the nearest points in the cloud for both measurement endpoints
    // CRITICAL: We need to use the exact distance tool coordinates, not just
    // the nearest points in the cloud. If the nearest point is too far away, we
    // should add the exact point to the cloud to ensure the exported label
    // matches the distance tool exactly.
    unsigned nearestIndex1 = 0;
    unsigned nearestIndex2 = 0;
    double minDist1 = std::numeric_limits<double>::max();
    double minDist2 = std::numeric_limits<double>::max();

    // Threshold for considering a point "close enough" (1mm in world units)
    // If the nearest point is farther than this, we'll add the exact point
    const double DISTANCE_THRESHOLD = 0.001;

    for (unsigned i = 0; i < cloud->size(); ++i) {
        const CCVector3* P = cloud->getPoint(i);
        if (!P) continue;

        double d1 = (*P - point1).norm();
        if (d1 < minDist1) {
            minDist1 = d1;
            nearestIndex1 = i;
        }

        double d2 = (*P - point2).norm();
        if (d2 < minDist2) {
            minDist2 = d2;
            nearestIndex2 = i;
        }
    }

    // CRITICAL: If the nearest points are too far from the exact distance tool
    // coordinates, add the exact points to the cloud to ensure perfect
    // alignment This ensures the exported label's line matches the distance
    // tool's line exactly
    ccPointCloud* pointCloud = ccHObjectCaster::ToPointCloud(cloud);
    if (pointCloud) {
        // Calculate how many new points we need to add
        unsigned pointsToAdd = 0;
        if (minDist1 > DISTANCE_THRESHOLD) pointsToAdd++;
        if (minDist2 > DISTANCE_THRESHOLD) pointsToAdd++;

        // Reserve memory for all new points at once (more efficient)
        if (pointsToAdd > 0) {
            unsigned currentSize = pointCloud->size();
            if (pointCloud->reserve(currentSize + pointsToAdd)) {
                // Check and add point1 if needed
                if (minDist1 > DISTANCE_THRESHOLD) {
                    pointCloud->addPoint(point1);
                    nearestIndex1 = pointCloud->size() - 1;
                }

                // Check and add point2 if needed
                if (minDist2 > DISTANCE_THRESHOLD) {
                    pointCloud->addPoint(point2);
                    nearestIndex2 = pointCloud->size() - 1;
                }
            }
        }
    }

    // Create a new 2D label with the two points
    cc2DLabel* label = new cc2DLabel(
            QString("Distance: %1").arg(getMeasurementValue(), 0, 'f', 6));

    // Add the two picked points to the label
    if (!label->addPickedPoint(cloud, nearestIndex1)) {
        CVLog::Warning(
                "[cvDistanceTool::getOutput] Failed to add first point to "
                "label");
        delete label;
        return nullptr;
    }

    if (!label->addPickedPoint(cloud, nearestIndex2)) {
        CVLog::Warning(
                "[cvDistanceTool::getOutput] Failed to add second point to "
                "label");
        delete label;
        return nullptr;
    }

    // Configure the label display settings
    label->setVisible(true);
    label->setEnabled(true);
    label->setDisplayedIn2D(true);
    label->displayPointLegend(true);
    label->setCollapsed(false);

    // Set a position for the label (relative to screen)
    label->setPosition(label->getPosition()[0], label->getPosition()[1]);

    return label;
}

double cvDistanceTool::getMeasurementValue() const {
    if (m_configUi) {
        return m_configUi->distanceSpinBox->value();
    }
    return 0.0;
}

void cvDistanceTool::getPoint1(double pos[3]) const {
    if (m_configUi && pos) {
        pos[0] = m_configUi->point1XSpinBox->value();
        pos[1] = m_configUi->point1YSpinBox->value();
        pos[2] = m_configUi->point1ZSpinBox->value();
    } else if (pos) {
        pos[0] = pos[1] = pos[2] = 0.0;
    }
}

void cvDistanceTool::getPoint2(double pos[3]) const {
    if (m_configUi && pos) {
        pos[0] = m_configUi->point2XSpinBox->value();
        pos[1] = m_configUi->point2YSpinBox->value();
        pos[2] = m_configUi->point2ZSpinBox->value();
    } else if (pos) {
        pos[0] = pos[1] = pos[2] = 0.0;
    }
}

void cvDistanceTool::setPoint1(double pos[3]) {
    if (!m_configUi) {
        CVLog::Warning("[cvDistanceTool] setPoint1: m_configUi or pos is null");
        return;
    }

    // Uncheck the pick button
    if (m_configUi->pickPoint1ToolButton->isChecked()) {
        m_configUi->pickPoint1ToolButton->setChecked(false);
    }

    // Update spinboxes without triggering signals
    VtkUtils::SignalBlocker blocker1(m_configUi->point1XSpinBox);
    VtkUtils::SignalBlocker blocker2(m_configUi->point1YSpinBox);
    VtkUtils::SignalBlocker blocker3(m_configUi->point1ZSpinBox);
    m_configUi->point1XSpinBox->setValue(pos[0]);
    m_configUi->point1YSpinBox->setValue(pos[1]);
    m_configUi->point1ZSpinBox->setValue(pos[2]);

    // Update widget directly
    if (m_widget && m_widget->GetEnabled()) {
        m_rep->SetPoint1WorldPosition(pos);
        m_rep->BuildRepresentation();
        m_widget->Modified();
        m_widget->Render();
    }

    // Update distance display
    updateDistanceDisplay();
    update();
}

void cvDistanceTool::setPoint2(double pos[3]) {
    if (!m_configUi) {
        CVLog::Warning("[cvDistanceTool] setPoint2: m_configUi or pos is null");
        return;
    }

    // Uncheck the pick button
    if (m_configUi->pickPoint2ToolButton->isChecked()) {
        m_configUi->pickPoint2ToolButton->setChecked(false);
    }

    // Update spinboxes without triggering signals
    VtkUtils::SignalBlocker blocker1(m_configUi->point2XSpinBox);
    VtkUtils::SignalBlocker blocker2(m_configUi->point2YSpinBox);
    VtkUtils::SignalBlocker blocker3(m_configUi->point2ZSpinBox);
    m_configUi->point2XSpinBox->setValue(pos[0]);
    m_configUi->point2YSpinBox->setValue(pos[1]);
    m_configUi->point2ZSpinBox->setValue(pos[2]);

    // Update widget directly
    if (m_widget && m_widget->GetEnabled()) {
        m_rep->SetPoint2WorldPosition(pos);
        m_rep->BuildRepresentation();
        m_widget->Modified();
        m_widget->Render();
    }

    // Update distance display
    updateDistanceDisplay();
    update();
}

void cvDistanceTool::setColor(double r, double g, double b) {
    // Store current color
    m_currentColor[0] = r;
    m_currentColor[1] = g;
    m_currentColor[2] = b;

    // Set color for line representation
    if (m_rep) {
        if (auto* lineProp = m_rep->GetLineProperty()) {
            lineProp->SetColor(r, g, b);
        }
        if (auto* selectedLineProp = m_rep->GetSelectedLineProperty()) {
            selectedLineProp->SetColor(r, g, b);
        }
        m_rep->BuildRepresentation();
        if (m_widget) {
            m_widget->Modified();
        }
    }
    update();
}

bool cvDistanceTool::getColor(double& r, double& g, double& b) const {
    r = m_currentColor[0];
    g = m_currentColor[1];
    b = m_currentColor[2];
    return true;
}

void cvDistanceTool::lockInteraction() {
    CVLog::PrintDebug(QString("[cvDistanceTool::lockInteraction] Tool=%1, "
                              "m_pickingHelpers.size()=%2")
                              .arg((quintptr)this, 0, 16)
                              .arg(m_pickingHelpers.size()));

    // Disable VTK widget interaction (handles cannot be moved)
    if (m_widget) {
        m_widget->SetProcessEvents(0);  // Disable event processing
    }

    // Change widget color to indicate locked state (very dimmed, 10%
    // brightness)
    if (m_rep) {
        // Use a very dimmed color to indicate locked state (10% brightness, 50%
        // opacity)
        if (auto* lineProp = m_rep->GetLineProperty()) {
            lineProp->SetColor(m_currentColor[0] * 0.1, m_currentColor[1] * 0.1,
                               m_currentColor[2] * 0.1);
            lineProp->SetOpacity(0.5);  // Make it semi-transparent
        }
        if (auto* selectedLineProp = m_rep->GetSelectedLineProperty()) {
            selectedLineProp->SetColor(m_currentColor[0] * 0.1,
                                       m_currentColor[1] * 0.1,
                                       m_currentColor[2] * 0.1);
            selectedLineProp->SetOpacity(0.5);
        }

        // Dim handles (points) by accessing them through Point1Representation
        // and Point2Representation
        if (auto* point1Rep = dynamic_cast<vtkPointHandleRepresentation3D*>(
                    m_rep->GetPoint1Representation())) {
            if (auto* prop = point1Rep->GetProperty()) {
                prop->SetOpacity(0.5);
            }
            if (auto* selectedProp = point1Rep->GetSelectedProperty()) {
                selectedProp->SetOpacity(0.5);
            }
        }
        if (auto* point2Rep = dynamic_cast<vtkPointHandleRepresentation3D*>(
                    m_rep->GetPoint2Representation())) {
            if (auto* prop = point2Rep->GetProperty()) {
                prop->SetOpacity(0.5);
            }
            if (auto* selectedProp = point2Rep->GetSelectedProperty()) {
                selectedProp->SetOpacity(0.5);
            }
        }

        m_rep->BuildRepresentation();

        // Set distance text (title) and axis properties AFTER
        // BuildRepresentation
        if (auto* axis = m_rep->GetAxisActor()) {
            // Dim the axis line
            if (auto* axisProp = axis->GetProperty()) {
                axisProp->SetOpacity(0.5);
            }

            // Dim the distance text (title)
            if (auto* titleProp = axis->GetTitleTextProperty()) {
                titleProp->SetOpacity(0.5);
                titleProp->SetColor(0.5, 0.5,
                                    0.5);  // Dark gray for locked state
            }

            // Dim axis labels (if visible)
            if (auto* labelProp = axis->GetLabelTextProperty()) {
                labelProp->SetOpacity(0.5);
            }
        }

        if (m_widget) {
            m_widget->Modified();
            m_widget->Render();
        }
    }

    // Force render window update
    if (m_interactor && m_interactor->GetRenderWindow()) {
        m_interactor->GetRenderWindow()->Render();
    }

    // Disable UI controls
    if (m_configUi) {
        m_configUi->point1XSpinBox->setEnabled(false);
        m_configUi->point1YSpinBox->setEnabled(false);
        m_configUi->point1ZSpinBox->setEnabled(false);
        m_configUi->point2XSpinBox->setEnabled(false);
        m_configUi->point2YSpinBox->setEnabled(false);
        m_configUi->point2ZSpinBox->setEnabled(false);
        m_configUi->pickPoint1ToolButton->setEnabled(false);
        m_configUi->pickPoint2ToolButton->setEnabled(false);
        m_configUi->rulerModeCheckBox->setEnabled(false);
        m_configUi->rulerDistanceSpinBox->setEnabled(false);
        m_configUi->numberOfTicksSpinBox->setEnabled(false);
        m_configUi->scaleSpinBox->setEnabled(false);
        m_configUi->labelFormatLineEdit->setEnabled(false);
        m_configUi->widgetVisibilityCheckBox->setEnabled(false);
        m_configUi->labelVisibilityCheckBox->setEnabled(false);
        m_configUi->lineWidthSpinBox->setEnabled(false);
    }

    // Disable keyboard shortcuts
    disableShortcuts();
}

void cvDistanceTool::unlockInteraction() {
    // Enable VTK widget interaction
    if (m_widget) {
        m_widget->SetProcessEvents(1);  // Enable event processing
    }

    // Restore widget color to indicate active/unlocked state
    if (m_rep) {
        // Restore original color (full brightness and opacity)
        if (auto* lineProp = m_rep->GetLineProperty()) {
            lineProp->SetColor(m_currentColor[0], m_currentColor[1],
                               m_currentColor[2]);
            lineProp->SetOpacity(1.0);  // Fully opaque
        }
        if (auto* selectedLineProp = m_rep->GetSelectedLineProperty()) {
            selectedLineProp->SetColor(m_currentColor[0], m_currentColor[1],
                                       m_currentColor[2]);
            selectedLineProp->SetOpacity(1.0);
        }

        // Restore handles (points) by accessing them through
        // Point1Representation and Point2Representation
        if (auto* point1Rep = dynamic_cast<vtkPointHandleRepresentation3D*>(
                    m_rep->GetPoint1Representation())) {
            if (auto* prop = point1Rep->GetProperty()) {
                prop->SetOpacity(1.0);
            }
            if (auto* selectedProp = point1Rep->GetSelectedProperty()) {
                selectedProp->SetOpacity(1.0);
            }
        }
        if (auto* point2Rep = dynamic_cast<vtkPointHandleRepresentation3D*>(
                    m_rep->GetPoint2Representation())) {
            if (auto* prop = point2Rep->GetProperty()) {
                prop->SetOpacity(1.0);
            }
            if (auto* selectedProp = point2Rep->GetSelectedProperty()) {
                selectedProp->SetOpacity(1.0);
            }
        }

        m_rep->BuildRepresentation();

        // Restore distance text (title) and axis properties AFTER
        // BuildRepresentation
        if (auto* axis = m_rep->GetAxisActor()) {
            // Restore axis line
            if (auto* axisProp = axis->GetProperty()) {
                axisProp->SetOpacity(1.0);
            }

            // Restore distance text (title) to user-configured settings
            if (auto* titleProp = axis->GetTitleTextProperty()) {
                titleProp->SetOpacity(m_fontOpacity);
                titleProp->SetColor(m_fontColor[0], m_fontColor[1],
                                    m_fontColor[2]);
            }

            // Restore axis labels (if visible) to user-configured settings
            if (auto* labelProp = axis->GetLabelTextProperty()) {
                labelProp->SetOpacity(m_fontOpacity);
                labelProp->SetColor(m_fontColor[0], m_fontColor[1],
                                    m_fontColor[2]);
            }
        }

        if (m_widget) {
            m_widget->Modified();
            m_widget->Render();
        }
    }

    // Force render window update
    if (m_interactor && m_interactor->GetRenderWindow()) {
        m_interactor->GetRenderWindow()->Render();
    }

    // Enable UI controls
    if (m_configUi) {
        m_configUi->point1XSpinBox->setEnabled(true);
        m_configUi->point1YSpinBox->setEnabled(true);
        m_configUi->point1ZSpinBox->setEnabled(true);
        m_configUi->point2XSpinBox->setEnabled(true);
        m_configUi->point2YSpinBox->setEnabled(true);
        m_configUi->point2ZSpinBox->setEnabled(true);
        m_configUi->pickPoint1ToolButton->setEnabled(true);
        m_configUi->pickPoint2ToolButton->setEnabled(true);
        m_configUi->rulerModeCheckBox->setEnabled(true);

        // Ruler distance is only enabled when ruler mode is on
        bool rulerMode = m_configUi->rulerModeCheckBox->isChecked();
        m_configUi->rulerDistanceSpinBox->setEnabled(rulerMode);
        m_configUi->numberOfTicksSpinBox->setEnabled(!rulerMode);

        m_configUi->scaleSpinBox->setEnabled(true);
        m_configUi->labelFormatLineEdit->setEnabled(true);
        m_configUi->widgetVisibilityCheckBox->setEnabled(true);
        m_configUi->labelVisibilityCheckBox->setEnabled(true);
        m_configUi->lineWidthSpinBox->setEnabled(true);
    }

    if (m_pickingHelpers.isEmpty()) {
        // Shortcuts haven't been created yet - create them now
        if (m_vtkWidget) {
            CVLog::PrintDebug(
                    QString("[cvDistanceTool::unlockInteraction] Creating "
                            "shortcuts for tool=%1, using saved vtkWidget=%2")
                            .arg((quintptr)this, 0, 16)
                            .arg((quintptr)m_vtkWidget, 0, 16));
            setupShortcuts(m_vtkWidget);
            CVLog::PrintDebug(
                    QString("[cvDistanceTool::unlockInteraction] After "
                            "setupShortcuts, m_pickingHelpers.size()=%1")
                            .arg(m_pickingHelpers.size()));
        } else {
            CVLog::PrintDebug(
                    QString("[cvDistanceTool::unlockInteraction] m_vtkWidget "
                            "is null for tool=%1, cannot create shortcuts")
                            .arg((quintptr)this, 0, 16));
        }
    } else {
        // Shortcuts already exist - just enable them
        CVLog::PrintDebug(QString("[cvDistanceTool::unlockInteraction] "
                                  "Enabling %1 existing shortcuts for tool=%2")
                                  .arg(m_pickingHelpers.size())
                                  .arg((quintptr)this, 0, 16));
        for (cvPointPickingHelper* helper : m_pickingHelpers) {
            if (helper) {
                helper->setEnabled(true,
                                   false);  // Enable without setting focus
            }
        }
    }
}

void cvDistanceTool::setInstanceLabel(const QString& label) {
    // Store the instance label
    m_instanceLabel = label;

    // Update the VTK representation's label suffix
    if (m_rep) {
        m_rep->SetLabelSuffix(m_instanceLabel.toUtf8().constData());
        m_rep->BuildRepresentation();
        if (m_widget) {
            m_widget->Modified();
        }
        update();
    }
}

void cvDistanceTool::on_point1XSpinBox_valueChanged(double arg1) {
    if (!m_configUi) return;
    double pos[3];
    double newPos[3];
    newPos[0] = arg1;

    VtkUtils::SignalBlocker blocker(m_configUi->point1XSpinBox);

    if (m_widget && m_widget->GetEnabled()) {
        m_rep->GetPoint1WorldPosition(pos);
        newPos[1] = pos[1];
        newPos[2] = pos[2];
        m_rep->SetPoint1WorldPosition(newPos);
        m_rep->BuildRepresentation();
        m_widget->Modified();
        updateDistanceDisplay();
    }
    update();
}

void cvDistanceTool::on_point1YSpinBox_valueChanged(double arg1) {
    if (!m_configUi) return;
    double pos[3];
    double newPos[3];
    newPos[1] = arg1;

    VtkUtils::SignalBlocker blocker(m_configUi->point1YSpinBox);

    if (m_widget && m_widget->GetEnabled()) {
        m_rep->GetPoint1WorldPosition(pos);
        newPos[0] = pos[0];
        newPos[2] = pos[2];
        m_rep->SetPoint1WorldPosition(newPos);
        m_rep->BuildRepresentation();
        m_widget->Modified();
        updateDistanceDisplay();
    }
    update();
}

void cvDistanceTool::on_point1ZSpinBox_valueChanged(double arg1) {
    if (!m_configUi) return;
    double pos[3];
    double newPos[3];
    newPos[2] = arg1;

    VtkUtils::SignalBlocker blocker(m_configUi->point1ZSpinBox);

    if (m_widget && m_widget->GetEnabled()) {
        m_rep->GetPoint1WorldPosition(pos);
        newPos[0] = pos[0];
        newPos[1] = pos[1];
        m_rep->SetPoint1WorldPosition(newPos);
        m_rep->BuildRepresentation();
        m_widget->Modified();
        updateDistanceDisplay();
    }
    update();
}

void cvDistanceTool::on_point2XSpinBox_valueChanged(double arg1) {
    if (!m_configUi) return;
    double pos[3];
    double newPos[3];
    newPos[0] = arg1;

    VtkUtils::SignalBlocker blocker(m_configUi->point2XSpinBox);

    if (m_widget && m_widget->GetEnabled()) {
        m_rep->GetPoint2WorldPosition(pos);
        newPos[1] = pos[1];
        newPos[2] = pos[2];
        m_rep->SetPoint2WorldPosition(newPos);
        m_rep->BuildRepresentation();
        m_widget->Modified();
        updateDistanceDisplay();
    }
    update();
}

void cvDistanceTool::on_point2YSpinBox_valueChanged(double arg1) {
    if (!m_configUi) return;
    double pos[3];
    double newPos[3];
    newPos[1] = arg1;

    VtkUtils::SignalBlocker blocker(m_configUi->point2YSpinBox);

    if (m_widget && m_widget->GetEnabled()) {
        m_rep->GetPoint2WorldPosition(pos);
        newPos[0] = pos[0];
        newPos[2] = pos[2];
        m_rep->SetPoint2WorldPosition(newPos);
        m_rep->BuildRepresentation();
        m_widget->Modified();
        updateDistanceDisplay();
    }
    update();
}

void cvDistanceTool::on_point2ZSpinBox_valueChanged(double arg1) {
    if (!m_configUi) return;
    double pos[3];
    double newPos[3];
    newPos[2] = arg1;

    VtkUtils::SignalBlocker blocker(m_configUi->point2ZSpinBox);

    if (m_widget && m_widget->GetEnabled()) {
        m_rep->GetPoint2WorldPosition(pos);
        newPos[0] = pos[0];
        newPos[1] = pos[1];
        m_rep->SetPoint2WorldPosition(newPos);
        m_rep->BuildRepresentation();
        m_widget->Modified();
        updateDistanceDisplay();
    }
    update();
}

void cvDistanceTool::onDistanceChanged(double dist) {
    if (!m_configUi) return;

    // Apply scale factor (following ParaView - scale is applied to display
    // value)
    double scale = m_configUi->scaleSpinBox->value();
    double scaledDistance = dist * scale;

    VtkUtils::SignalBlocker blocker(m_configUi->distanceSpinBox);
    m_configUi->distanceSpinBox->setValue(scaledDistance);
    emit measurementValueChanged();
}

void cvDistanceTool::onWorldPoint1Changed(double* pos) {
    if (!m_configUi || !pos) return;
    VtkUtils::SignalBlocker blocker1(m_configUi->point1XSpinBox);
    VtkUtils::SignalBlocker blocker2(m_configUi->point1YSpinBox);
    VtkUtils::SignalBlocker blocker3(m_configUi->point1ZSpinBox);
    m_configUi->point1XSpinBox->setValue(pos[0]);
    m_configUi->point1YSpinBox->setValue(pos[1]);
    m_configUi->point1ZSpinBox->setValue(pos[2]);
    emit measurementValueChanged();
}

void cvDistanceTool::onWorldPoint2Changed(double* pos) {
    if (!m_configUi || !pos) return;
    VtkUtils::SignalBlocker blocker1(m_configUi->point2XSpinBox);
    VtkUtils::SignalBlocker blocker2(m_configUi->point2YSpinBox);
    VtkUtils::SignalBlocker blocker3(m_configUi->point2ZSpinBox);
    m_configUi->point2XSpinBox->setValue(pos[0]);
    m_configUi->point2YSpinBox->setValue(pos[1]);
    m_configUi->point2ZSpinBox->setValue(pos[2]);
    emit measurementValueChanged();
}

void cvDistanceTool::hookWidget(
        const vtkSmartPointer<cvConstrainedDistanceWidget>& widget) {
    VtkUtils::DistanceWidgetObserver* observer =
            new VtkUtils::DistanceWidgetObserver(this);
    observer->attach(widget.Get());
    connect(observer, &VtkUtils::DistanceWidgetObserver::distanceChanged, this,
            &cvDistanceTool::onDistanceChanged);
    connect(observer, &VtkUtils::DistanceWidgetObserver::worldPoint1Changed,
            this, &cvDistanceTool::onWorldPoint1Changed);
    connect(observer, &VtkUtils::DistanceWidgetObserver::worldPoint2Changed,
            this, &cvDistanceTool::onWorldPoint2Changed);
}

void cvDistanceTool::updateDistanceDisplay() {
    if (!m_configUi) return;

    double distance = 0.0;
    double scale = m_configUi->scaleSpinBox->value();

    if (m_rep) {
        // Get unscaled distance and apply scale (following ParaView)
        distance = m_rep->GetDistance() * scale;
    }

    VtkUtils::SignalBlocker blocker(m_configUi->distanceSpinBox);
    m_configUi->distanceSpinBox->setValue(distance);
    emit measurementValueChanged();
}

void cvDistanceTool::on_pickPoint1_toggled(bool checked) {
    if (checked) {
        // Uncheck other pick button
        if (m_configUi->pickPoint2ToolButton->isChecked()) {
            m_configUi->pickPoint2ToolButton->setChecked(false);
        }
        // Enable point picking mode for point 1
        emit pointPickingRequested(1);
    } else {
        // Disable point picking
        emit pointPickingCancelled();
    }
}

void cvDistanceTool::on_pickPoint2_toggled(bool checked) {
    if (checked) {
        // Uncheck other pick button
        if (m_configUi->pickPoint1ToolButton->isChecked()) {
            m_configUi->pickPoint1ToolButton->setChecked(false);
        }
        // Enable point picking mode for point 2
        emit pointPickingRequested(2);
    } else {
        // Disable point picking
        emit pointPickingCancelled();
    }
}

void cvDistanceTool::on_rulerModeCheckBox_toggled(bool checked) {
    if (!m_configUi) return;

    // Use cvConstrainedLineRepresentation's ruler mode functionality
    if (m_rep) {
        m_rep->SetRulerMode(checked ? 1 : 0);
        m_rep->BuildRepresentation();
    }

    // Enable/disable appropriate controls
    m_configUi->rulerDistanceSpinBox->setEnabled(checked);
    m_configUi->numberOfTicksSpinBox->setEnabled(!checked);

    update();
}

void cvDistanceTool::on_rulerDistanceSpinBox_valueChanged(double value) {
    if (!m_configUi) return;

    // Use cvConstrainedLineRepresentation's ruler distance functionality
    if (m_rep) {
        m_rep->SetRulerDistance(value);
        m_rep->BuildRepresentation();
    }
    update();
}

void cvDistanceTool::on_numberOfTicksSpinBox_valueChanged(int value) {
    if (!m_configUi) return;

    // Use cvConstrainedLineRepresentation's ruler ticks functionality
    if (m_rep) {
        m_rep->SetNumberOfRulerTicks(value);
        m_rep->BuildRepresentation();
    }
    update();
}

void cvDistanceTool::on_scaleSpinBox_valueChanged(double value) {
    if (!m_configUi) return;

    // Use cvConstrainedLineRepresentation's scale functionality
    if (m_rep) {
        m_rep->SetScale(value);
        m_rep->BuildRepresentation();
    }

    // Update distance display to reflect the scaled value
    updateDistanceDisplay();
    update();
}

void cvDistanceTool::on_labelFormatLineEdit_textChanged(const QString& text) {
    if (!m_configUi) return;

    // Use cvConstrainedLineRepresentation's label format functionality
    std::string formatStr = text.toStdString();

    // Basic validation: check if format specifier is present
    if (formatStr.find('%') == std::string::npos) {
        CVLog::Warning(
                "[cvDistanceTool] Invalid label format: missing '%' specifier");
        return;
    }

    if (m_rep) {
        m_rep->SetLabelFormat(formatStr.c_str());
        m_rep->BuildRepresentation();
    }

    update();
}

void cvDistanceTool::on_widgetVisibilityCheckBox_toggled(bool checked) {
    if (!m_configUi) return;

    // Show or hide the widget
    if (m_widget) {
        if (checked) {
            m_widget->On();
        } else {
            m_widget->Off();
        }
    }
    update();
}

void cvDistanceTool::on_labelVisibilityCheckBox_toggled(bool checked) {
    if (!m_configUi) return;

    // Use cvConstrainedLineRepresentation's ShowLabel functionality
    if (m_rep) {
        m_rep->SetShowLabel(checked ? 1 : 0);
        m_rep->BuildRepresentation();
    }
    update();
}

void cvDistanceTool::on_lineWidthSpinBox_valueChanged(double value) {
    if (!m_configUi) return;

    // vtkLineRepresentation 使用 GetLineProperty (与 ParaView vtkLineWidget2
    // 一致)
    if (m_rep) {
        if (auto* prop = m_rep->GetLineProperty()) {
            prop->SetLineWidth(value);
        }
        m_rep->BuildRepresentation();
    }

    update();
}

void cvDistanceTool::setupPointPickingShortcuts(QWidget* vtkWidget) {
    if (!vtkWidget) return;

    // 'P' - Pick alternating points on surface cell
    cvPointPickingHelper* pickHelper =
            new cvPointPickingHelper(QKeySequence(tr("P")), false, vtkWidget);
    pickHelper->setInteractor(m_interactor);
    pickHelper->setRenderer(m_renderer);
    pickHelper->setContextWidget(this);
    connect(pickHelper, &cvPointPickingHelper::pick, this,
            &cvDistanceTool::pickAlternatingPoint);
    m_pickingHelpers.append(pickHelper);

    // 'Ctrl+P' - Pick alternating points, snap to mesh points
    cvPointPickingHelper* pickHelper2 = new cvPointPickingHelper(
            QKeySequence(tr("Ctrl+P")), true, vtkWidget);
    pickHelper2->setInteractor(m_interactor);
    pickHelper2->setRenderer(m_renderer);
    pickHelper2->setContextWidget(this);
    connect(pickHelper2, &cvPointPickingHelper::pick, this,
            &cvDistanceTool::pickAlternatingPoint);
    m_pickingHelpers.append(pickHelper2);

    // '1' - Pick point 1 on surface cell
    cvPointPickingHelper* pickHelper3 =
            new cvPointPickingHelper(QKeySequence(tr("1")), false, vtkWidget);
    pickHelper3->setInteractor(m_interactor);
    pickHelper3->setRenderer(m_renderer);
    pickHelper3->setContextWidget(this);
    connect(pickHelper3, &cvPointPickingHelper::pick, this,
            &cvDistanceTool::pickKeyboardPoint1);
    m_pickingHelpers.append(pickHelper3);

    // 'Ctrl+1' - Pick point 1, snap to mesh points
    cvPointPickingHelper* pickHelper4 = new cvPointPickingHelper(
            QKeySequence(tr("Ctrl+1")), true, vtkWidget);
    pickHelper4->setInteractor(m_interactor);
    pickHelper4->setRenderer(m_renderer);
    pickHelper4->setContextWidget(this);
    connect(pickHelper4, &cvPointPickingHelper::pick, this,
            &cvDistanceTool::pickKeyboardPoint1);
    m_pickingHelpers.append(pickHelper4);

    // '2' - Pick point 2 on surface cell
    cvPointPickingHelper* pickHelper5 =
            new cvPointPickingHelper(QKeySequence(tr("2")), false, vtkWidget);
    pickHelper5->setInteractor(m_interactor);
    pickHelper5->setRenderer(m_renderer);
    pickHelper5->setContextWidget(this);
    connect(pickHelper5, &cvPointPickingHelper::pick, this,
            &cvDistanceTool::pickKeyboardPoint2);
    m_pickingHelpers.append(pickHelper5);

    // 'Ctrl+2' - Pick point 2, snap to mesh points
    cvPointPickingHelper* pickHelper6 = new cvPointPickingHelper(
            QKeySequence(tr("Ctrl+2")), true, vtkWidget);
    pickHelper6->setInteractor(m_interactor);
    pickHelper6->setRenderer(m_renderer);
    pickHelper6->setContextWidget(this);
    connect(pickHelper6, &cvPointPickingHelper::pick, this,
            &cvDistanceTool::pickKeyboardPoint2);
    m_pickingHelpers.append(pickHelper6);

    // 'N' - Pick point and set normal direction
    cvPointPickingHelper* pickHelperNormal = new cvPointPickingHelper(
            QKeySequence(tr("N")), false, vtkWidget,
            cvPointPickingHelper::CoordinatesAndNormal);
    pickHelperNormal->setInteractor(m_interactor);
    pickHelperNormal->setRenderer(m_renderer);
    pickHelperNormal->setContextWidget(this);
    connect(pickHelperNormal, &cvPointPickingHelper::pickNormal, this,
            &cvDistanceTool::pickNormalDirection);
    m_pickingHelpers.append(pickHelperNormal);
}

void cvDistanceTool::pickAlternatingPoint(double x, double y, double z) {
    if (m_pickPoint1Next) {
        pickKeyboardPoint1(x, y, z);
    } else {
        pickKeyboardPoint2(x, y, z);
    }
    m_pickPoint1Next = !m_pickPoint1Next;
}

void cvDistanceTool::pickKeyboardPoint1(double x, double y, double z) {
    double pos[3] = {x, y, z};
    setPoint1(pos);
}

void cvDistanceTool::pickKeyboardPoint2(double x, double y, double z) {
    double pos[3] = {x, y, z};
    setPoint2(pos);
}

void cvDistanceTool::pickNormalDirection(
        double px, double py, double pz, double nx, double ny, double nz) {
    // Set point 1 at the picked position
    double p1[3] = {px, py, pz};
    setPoint1(p1);

    // Set point 2 along the normal direction
    double p2[3] = {px + nx, py + ny, pz + nz};
    setPoint2(p2);
}

void cvDistanceTool::applyFontProperties() {
    if (!m_rep) return;

    // Apply font properties to the axis actor's text properties
    if (auto* axis = m_rep->GetAxisActor()) {
        // CRITICAL: vtkAxisActor2D uses automatic font scaling based on
        // FontFactor We need to disable automatic adjustment and set properties
        // correctly

        // Disable automatic label adjustment to prevent font size override
        axis->AdjustLabelsOff();

        // Calculate font factor based on desired font size
        // vtkAxisActor2D internally scales fonts, so we adjust the factor
        // accordingly Base reference size is typically around 12-14 in VTK
        double baseFontSize = 12.0;
        double fontFactor = static_cast<double>(m_fontSize) / baseFontSize;
        if (fontFactor < 0.1) fontFactor = 0.1;    // Prevent too small values
        if (fontFactor > 10.0) fontFactor = 10.0;  // Prevent too large values

        // Set font factors for both title and labels
        axis->SetFontFactor(fontFactor);
        axis->SetLabelFactor(fontFactor * 0.8);  // Labels slightly smaller

        // Apply to title text property (distance display)
        // The Title is what shows the distance value
        if (auto* titleProp = axis->GetTitleTextProperty()) {
            titleProp->SetFontFamilyAsString(m_fontFamily.toUtf8().constData());
            titleProp->SetFontSize(m_fontSize);
            titleProp->SetColor(m_fontColor[0], m_fontColor[1], m_fontColor[2]);
            titleProp->SetBold(m_fontBold ? 1 : 0);
            titleProp->SetItalic(m_fontItalic ? 1 : 0);
            titleProp->SetShadow(m_fontShadow ? 1 : 0);
            titleProp->SetOpacity(m_fontOpacity);

            // Apply justification
            if (m_horizontalJustification == "Left") {
                titleProp->SetJustificationToLeft();
            } else if (m_horizontalJustification == "Center") {
                titleProp->SetJustificationToCentered();
            } else if (m_horizontalJustification == "Right") {
                titleProp->SetJustificationToRight();
            }

            if (m_verticalJustification == "Top") {
                titleProp->SetVerticalJustificationToTop();
            } else if (m_verticalJustification == "Center") {
                titleProp->SetVerticalJustificationToCentered();
            } else if (m_verticalJustification == "Bottom") {
                titleProp->SetVerticalJustificationToBottom();
            }

            titleProp->Modified();  // Mark as modified to ensure VTK updates
        }

        // Apply to label text property (tick labels for ruler mode)
        if (auto* labelProp = axis->GetLabelTextProperty()) {
            labelProp->SetFontFamilyAsString(m_fontFamily.toUtf8().constData());
            labelProp->SetFontSize(m_fontSize);
            labelProp->SetColor(m_fontColor[0], m_fontColor[1], m_fontColor[2]);
            labelProp->SetBold(m_fontBold ? 1 : 0);
            labelProp->SetItalic(m_fontItalic ? 1 : 0);
            labelProp->SetShadow(m_fontShadow ? 1 : 0);
            labelProp->SetOpacity(m_fontOpacity);

            // Apply justification
            if (m_horizontalJustification == "Left") {
                labelProp->SetJustificationToLeft();
            } else if (m_horizontalJustification == "Center") {
                labelProp->SetJustificationToCentered();
            } else if (m_horizontalJustification == "Right") {
                labelProp->SetJustificationToRight();
            }

            if (m_verticalJustification == "Top") {
                labelProp->SetVerticalJustificationToTop();
            } else if (m_verticalJustification == "Center") {
                labelProp->SetVerticalJustificationToCentered();
            } else if (m_verticalJustification == "Bottom") {
                labelProp->SetVerticalJustificationToBottom();
            }

            labelProp->Modified();  // Mark as modified to ensure VTK updates
        }

        // Mark axis actor as modified to trigger re-render
        axis->Modified();
    }

    // Rebuild representation and update display
    m_rep->BuildRepresentation();
    if (m_widget) {
        m_widget->Modified();
        m_widget->Render();
    }

    // Force render window update
    if (m_interactor && m_interactor->GetRenderWindow()) {
        m_interactor->GetRenderWindow()->Render();
    }

    update();
}
