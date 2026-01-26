// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvProtractorTool.h"

#include <VtkUtils/anglewidgetobserver.h>
#include <VtkUtils/signalblocker.h>
#include <VtkUtils/vtkutils.h>
#include <vtkActor2D.h>
#include <vtkCallbackCommand.h>
#include <vtkCommand.h>
#include <vtkHandleRepresentation.h>
#include <vtkMath.h>
#include <vtkPointHandleRepresentation3D.h>
#include <vtkPolyLineRepresentation.h>
#include <vtkProperty.h>
#include <vtkProperty2D.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>

#include <QApplication>
#include <QLayout>
#include <QLayoutItem>
#include <QShortcut>
#include <QSizePolicy>
#include <algorithm>
#include <cmath>

#include "Tools/PickingTools/cvPointPickingHelper.h"
#include "VTKExtensions/ConstrainedWidgets/cvConstrainedPolyLineRepresentation.h"
#include "VTKExtensions/ConstrainedWidgets/cvConstrainedPolyLineWidget.h"
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

//! Configure PolyLine representation for angle measurement with ParaView-style
//! properties
void configurePolyLineRepresentation(cvConstrainedPolyLineRepresentation* rep,
                                     bool use3DHandles = true) {
    if (!rep) return;

    // Set number of handles to 3 for angle measurement (Point1, Center, Point2)
    rep->SetNumberOfHandles(3);

    // Configure line properties (matching ParaView's LineProperty)
    if (auto* lineProp = rep->GetLineProperty()) {
        lineProp->SetColor(RAY_COLOR[0], RAY_COLOR[1],
                           RAY_COLOR[2]);  // Red lines
        lineProp->SetLineWidth(2.0);       // ParaView default line width
        lineProp->SetAmbient(1.0);         // ParaView sets ambient to 1.0
    }

    // Configure handle properties (matching ParaView)
    if (auto* handleProp = rep->GetHandleProperty()) {
        handleProp->SetColor(FOREGROUND_COLOR[0], FOREGROUND_COLOR[1],
                             FOREGROUND_COLOR[2]);
    }
    if (auto* selectedHandleProp = rep->GetSelectedHandleProperty()) {
        selectedHandleProp->SetColor(INTERACTION_COLOR[0], INTERACTION_COLOR[1],
                                     INTERACTION_COLOR[2]);
    }

    // Configure angle label text properties for better readability
    if (auto* labelActor = rep->GetAngleLabelActor()) {
        if (auto* textProp = labelActor->GetTextProperty()) {
            textProp->SetFontSize(20);  // Default font size for angle display
            textProp->SetBold(0);       // Not bold for better readability
            textProp->SetShadow(1);     // Add shadow for better visibility
            textProp->SetColor(1.0, 1.0, 1.0);  // White text
        }
    }

    // Configure angle display features
    rep->SetShowAngleLabel(1);  // Show angle label by default
    rep->SetShowAngleArc(1);    // Show angle arc by default
    rep->SetArcRadius(1.0);     // Default arc radius
}

}  // anonymous namespace

cvProtractorTool::cvProtractorTool(QWidget* parent)
    : cvGenericMeasurementTool(parent), m_configUi(nullptr) {
    setWindowTitle(tr("Protractor Measurement Tool"));

    // Override base class font size default for angle measurements
    // Angles need larger font for better readability
    m_fontSize = 20;
}

cvProtractorTool::~cvProtractorTool() {
    // CRITICAL: Explicitly hide and cleanup widget/representation before
    // destruction
    if (m_widget) {
        m_widget->Off();          // Turn off widget
        m_widget->SetEnabled(0);  // Disable widget
    }

    // Explicitly hide all representation elements
    if (m_rep) {
        m_rep->SetVisibility(0);  // Hide everything
        if (auto* labelActor = m_rep->GetAngleLabelActor()) {
            labelActor->SetVisibility(0);
        }
        if (auto* arcActor = m_rep->GetAngleArcActor()) {
            arcActor->SetVisibility(0);
        }

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

void cvProtractorTool::initTool() {
    // Initialize 3D widget only (simplified - no 2D/3D switching)
    VtkUtils::vtkInitOnce(m_rep);

    // Use constrained PolyLine widget - automatically supports XYZ shortcuts
    // (ParaView way)
    m_widget = vtkSmartPointer<cvConstrainedPolyLineWidget>::New();

    // Set representation BEFORE calling SetInteractor/SetRenderer
    m_widget->SetRepresentation(m_rep);

    if (m_interactor) {
        m_widget->SetInteractor(m_interactor);
    }
    if (m_renderer) {
        m_rep->SetRenderer(m_renderer);
    }

    // Following ParaView's approach:
    // 1. Configure appearance (set 3 handles for angle measurement)
    configurePolyLineRepresentation(m_rep, true);  // 3D mode

    // 2. Apply default green color (override configure defaults)
    if (auto* lineProp = m_rep->GetLineProperty()) {
        lineProp->SetColor(m_currentColor[0], m_currentColor[1],
                           m_currentColor[2]);
    }
    if (auto* selectedLineProp = m_rep->GetSelectedLineProperty()) {
        selectedLineProp->SetColor(m_currentColor[0], m_currentColor[1],
                                   m_currentColor[2]);
    }

    // 3. Initialize handle positions based on bounding box
    // This ensures the protractor is visible outside the object from the start
    double defaultPos1[3] = {0.0, 0.0, 0.0};
    double defaultCenter[3] = {1.0, 0.0, 0.0};
    double defaultPos2[3] = {1.0, 1.0, 0.0};

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

        // Place the protractor above and in front of the object for better
        // visibility Use offset in Y and Z directions to avoid objects in any
        // orientation
        defaultCenter[0] = center.x;
        defaultCenter[1] = center.y + offset;  // Offset in Y direction
        defaultCenter[2] = center.z + offset;  // Offset in Z direction

        defaultPos1[0] = center.x - diag.x * 0.25;
        defaultPos1[1] = center.y + offset;  // Offset in Y direction
        defaultPos1[2] = center.z + offset;  // Offset in Z direction

        defaultPos2[0] = center.x + diag.x * 0.25;
        defaultPos2[1] = center.y + offset;  // Offset in Y direction
        defaultPos2[2] = center.z + offset;  // Offset in Z direction
    }

    m_rep->SetHandlePosition(0, defaultPos1);
    m_rep->SetHandlePosition(1, defaultCenter);
    m_rep->SetHandlePosition(2, defaultPos2);

    // 4. Build representation (before updating UI)
    m_rep->BuildRepresentation();

    // 5. Apply font properties to ensure user-configured font properties are
    // applied
    applyFontProperties();

    // 6. Update UI controls with initial positions (if UI is already created)
    if (m_configUi) {
        VtkUtils::SignalBlocker blocker1(m_configUi->centerXSpinBox);
        VtkUtils::SignalBlocker blocker2(m_configUi->centerYSpinBox);
        VtkUtils::SignalBlocker blocker3(m_configUi->centerZSpinBox);
        VtkUtils::SignalBlocker blocker4(m_configUi->point1XSpinBox);
        VtkUtils::SignalBlocker blocker5(m_configUi->point1YSpinBox);
        VtkUtils::SignalBlocker blocker6(m_configUi->point1ZSpinBox);
        VtkUtils::SignalBlocker blocker7(m_configUi->point2XSpinBox);
        VtkUtils::SignalBlocker blocker8(m_configUi->point2YSpinBox);
        VtkUtils::SignalBlocker blocker9(m_configUi->point2ZSpinBox);

        m_configUi->centerXSpinBox->setValue(defaultCenter[0]);
        m_configUi->centerYSpinBox->setValue(defaultCenter[1]);
        m_configUi->centerZSpinBox->setValue(defaultCenter[2]);
        m_configUi->point1XSpinBox->setValue(defaultPos1[0]);
        m_configUi->point1YSpinBox->setValue(defaultPos1[1]);
        m_configUi->point1ZSpinBox->setValue(defaultPos1[2]);
        m_configUi->point2XSpinBox->setValue(defaultPos2[0]);
        m_configUi->point2YSpinBox->setValue(defaultPos2[1]);
        m_configUi->point2ZSpinBox->setValue(defaultPos2[2]);

        // Update angle display to show initial angle
        updateAngleDisplay();
    }

    // 6. Enable widget
    m_widget->On();

    hookWidget(m_widget);
}

void cvProtractorTool::createUi() {
    // CRITICAL: Only setup base UI once to avoid resetting configLayout
    // Each tool instance has its own m_ui, but setupUi clears all children
    // so we must ensure it's only called once per tool instance
    // Check if base UI is already set up by checking if widget has a layout
    // NOTE: Cannot check m_ui->configLayout directly as it's uninitialized
    // before setupUi()
    if (!m_ui) {
        CVLog::Error("[cvProtractorTool::createUi] m_ui is null!");
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
    m_configUi = new Ui::ProtractorToolDlg;
    QWidget* configWidget = new QWidget(this);
    // CRITICAL: Set size policy to Minimum to prevent horizontal expansion
    // This ensures the widget only takes the space it needs
    configWidget->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Preferred);
    m_configUi->setupUi(configWidget);
    // CRITICAL: Set layout size constraint to ensure minimum size calculation
    // This prevents extra whitespace on the right
    if (configWidget->layout()) {
        configWidget->layout()->setSizeConstraint(QLayout::SetMinimumSize);
    }
    m_ui->configLayout->addWidget(configWidget);
    m_ui->groupBox->setTitle(tr("Protractor Parameters"));

#ifdef Q_OS_MAC
    m_configUi->instructionLabel->setText(
            m_configUi->instructionLabel->text().replace("Ctrl", "Cmd"));
#endif

    // CRITICAL: Ensure Tips label can display full text with ParaView-style
    // compact layout ParaView uses Minimum sizePolicy to prevent horizontal
    // expansion This must be done AFTER text is set (including macOS text
    // replacement)
    if (m_configUi->instructionLabel) {
        // ParaView-style: Use Minimum sizePolicy to prevent horizontal
        // expansion The label will wrap text based on its natural width, not a
        // fixed maximum
        m_configUi->instructionLabel->setSizePolicy(QSizePolicy::Minimum,
                                                    QSizePolicy::Minimum);
        // CRITICAL: Remove any maximum height constraint to allow full text
        // display
        m_configUi->instructionLabel->setMaximumHeight(
                16777215);  // QWIDGETSIZE_MAX equivalent
        // Remove maximum width constraint - let it wrap naturally based on
        // parent width
        m_configUi->instructionLabel->setMaximumWidth(
                16777215);  // QWIDGETSIZE_MAX equivalent
        m_configUi->instructionLabel->setWordWrap(true);
        // Force the label to update its size based on wrapped text
        // This ensures the label expands vertically to show all text
        m_configUi->instructionLabel->adjustSize();
        // CRITICAL: Update geometry to ensure layout recalculates
        m_configUi->instructionLabel->updateGeometry();
    }

    // CRITICAL: Use Qt's automatic sizing based on sizeHint
    // This ensures each tool adapts to its own content without interference
    // Reset size constraints to allow Qt's layout system to work properly
    // ParaView-style: use Minimum (horizontal) to prevent unnecessary expansion
    this->setMinimumSize(0, 0);
    this->setMaximumSize(16777215, 16777215);  // QWIDGETSIZE_MAX equivalent
    this->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Preferred);

    // Let Qt calculate the optimal size based on content
    // Order matters: adjust configWidget first, then the main widget
    configWidget->adjustSize();
    this->adjustSize();
    // Force layout update to apply size changes
    this->updateGeometry();
    // CRITICAL: Process events to ensure layout is fully updated
    QApplication::processEvents();

    connect(m_configUi->point1XSpinBox,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &cvProtractorTool::on_point1XSpinBox_valueChanged);
    connect(m_configUi->point1YSpinBox,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &cvProtractorTool::on_point1YSpinBox_valueChanged);
    connect(m_configUi->point1ZSpinBox,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &cvProtractorTool::on_point1ZSpinBox_valueChanged);
    connect(m_configUi->centerXSpinBox,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &cvProtractorTool::on_centerXSpinBox_valueChanged);
    connect(m_configUi->centerYSpinBox,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &cvProtractorTool::on_centerYSpinBox_valueChanged);
    connect(m_configUi->centerZSpinBox,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &cvProtractorTool::on_centerZSpinBox_valueChanged);
    connect(m_configUi->point2XSpinBox,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &cvProtractorTool::on_point2XSpinBox_valueChanged);
    connect(m_configUi->point2YSpinBox,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &cvProtractorTool::on_point2YSpinBox_valueChanged);
    connect(m_configUi->point2ZSpinBox,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &cvProtractorTool::on_point2ZSpinBox_valueChanged);

    // Connect point picking buttons
    connect(m_configUi->pickPoint1ToolButton, &QToolButton::toggled, this,
            &cvProtractorTool::on_pickPoint1_toggled);
    connect(m_configUi->pickCenterToolButton, &QToolButton::toggled, this,
            &cvProtractorTool::on_pickCenter_toggled);
    connect(m_configUi->pickPoint2ToolButton, &QToolButton::toggled, this,
            &cvProtractorTool::on_pickPoint2_toggled);

    // Connect display options
    connect(m_configUi->widgetVisibilityCheckBox, &QCheckBox::toggled, this,
            &cvProtractorTool::on_widgetVisibilityCheckBox_toggled);
    connect(m_configUi->arcVisibilityCheckBox, &QCheckBox::toggled, this,
            &cvProtractorTool::on_arcVisibilityCheckBox_toggled);
}

void cvProtractorTool::start() {
    cvGenericMeasurementTool::start();
    // if (m_renderer) {
    //     m_renderer->ResetCamera();
    // }
}

void cvProtractorTool::reset() {
    // Reset points to default positions (above the bounding box for better
    // visibility and accessibility)
    double defaultCenter[3] = {0.0, 0.0, 0.0};
    double defaultPos1[3] = {-0.5, 0.0, 0.0};
    double defaultPos2[3] = {0.5, 0.0, 0.0};

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

        // Place the protractor above and in front of the object for better
        // visibility Use offset in Y and Z directions to avoid objects in any
        // orientation
        defaultCenter[0] = center.x;
        defaultCenter[1] = center.y + offset;  // Offset in Y direction
        defaultCenter[2] = center.z + offset;  // Offset in Z direction

        defaultPos1[0] = center.x - diag.x * 0.25;
        defaultPos1[1] = center.y + offset;  // Offset in Y direction
        defaultPos1[2] = center.z + offset;  // Offset in Z direction

        defaultPos2[0] = center.x + diag.x * 0.25;
        defaultPos2[1] = center.y + offset;  // Offset in Y direction
        defaultPos2[2] = center.z + offset;  // Offset in Z direction
    }

    // Reset widget points
    if (m_widget && m_widget->GetEnabled()) {
        m_rep->SetCenterWorldPosition(defaultCenter);
        m_rep->SetPoint1WorldPosition(defaultPos1);
        m_rep->SetPoint2WorldPosition(defaultPos2);
        m_rep->BuildRepresentation();

        // Reapply text properties after BuildRepresentation
        applyTextPropertiesToLabel();

        m_widget->Modified();
    }

    // Update UI
    if (m_configUi) {
        VtkUtils::SignalBlocker blocker1(m_configUi->centerXSpinBox);
        VtkUtils::SignalBlocker blocker2(m_configUi->centerYSpinBox);
        VtkUtils::SignalBlocker blocker3(m_configUi->centerZSpinBox);
        VtkUtils::SignalBlocker blocker4(m_configUi->point1XSpinBox);
        VtkUtils::SignalBlocker blocker5(m_configUi->point1YSpinBox);
        VtkUtils::SignalBlocker blocker6(m_configUi->point1ZSpinBox);
        VtkUtils::SignalBlocker blocker7(m_configUi->point2XSpinBox);
        VtkUtils::SignalBlocker blocker8(m_configUi->point2YSpinBox);
        VtkUtils::SignalBlocker blocker9(m_configUi->point2ZSpinBox);

        m_configUi->centerXSpinBox->setValue(defaultCenter[0]);
        m_configUi->centerYSpinBox->setValue(defaultCenter[1]);
        m_configUi->centerZSpinBox->setValue(defaultCenter[2]);
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

void cvProtractorTool::setupPointPickingShortcuts(QWidget* vtkWidget) {
    if (!vtkWidget) return;

    // '1' - Pick point 1 on surface
    cvPointPickingHelper* pickHelper1 =
            new cvPointPickingHelper(QKeySequence(tr("1")), false, vtkWidget);
    pickHelper1->setInteractor(m_interactor);
    pickHelper1->setRenderer(m_renderer);
    pickHelper1->setContextWidget(this);
    connect(pickHelper1, &cvPointPickingHelper::pick, this,
            &cvProtractorTool::pickKeyboardPoint1);
    m_pickingHelpers.append(pickHelper1);

    // 'Ctrl+1' - Pick point 1, snap to mesh points
    cvPointPickingHelper* pickHelper1Snap = new cvPointPickingHelper(
            QKeySequence(tr("Ctrl+1")), true, vtkWidget);
    pickHelper1Snap->setInteractor(m_interactor);
    pickHelper1Snap->setRenderer(m_renderer);
    pickHelper1Snap->setContextWidget(this);
    connect(pickHelper1Snap, &cvPointPickingHelper::pick, this,
            &cvProtractorTool::pickKeyboardPoint1);
    m_pickingHelpers.append(pickHelper1Snap);

    // 'C' - Pick center on surface
    cvPointPickingHelper* pickHelperCenter =
            new cvPointPickingHelper(QKeySequence(tr("C")), false, vtkWidget);
    pickHelperCenter->setInteractor(m_interactor);
    pickHelperCenter->setRenderer(m_renderer);
    pickHelperCenter->setContextWidget(this);
    connect(pickHelperCenter, &cvPointPickingHelper::pick, this,
            &cvProtractorTool::pickKeyboardCenter);
    m_pickingHelpers.append(pickHelperCenter);

    // 'Ctrl+C' - Pick center, snap to mesh points
    cvPointPickingHelper* pickHelperCenterSnap = new cvPointPickingHelper(
            QKeySequence(tr("Ctrl+C")), true, vtkWidget);
    pickHelperCenterSnap->setInteractor(m_interactor);
    pickHelperCenterSnap->setRenderer(m_renderer);
    pickHelperCenterSnap->setContextWidget(this);
    connect(pickHelperCenterSnap, &cvPointPickingHelper::pick, this,
            &cvProtractorTool::pickKeyboardCenter);
    m_pickingHelpers.append(pickHelperCenterSnap);

    // '2' - Pick point 2 on surface
    cvPointPickingHelper* pickHelper2 =
            new cvPointPickingHelper(QKeySequence(tr("2")), false, vtkWidget);
    pickHelper2->setInteractor(m_interactor);
    pickHelper2->setRenderer(m_renderer);
    pickHelper2->setContextWidget(this);
    connect(pickHelper2, &cvPointPickingHelper::pick, this,
            &cvProtractorTool::pickKeyboardPoint2);
    m_pickingHelpers.append(pickHelper2);

    // 'Ctrl+2' - Pick point 2, snap to mesh points
    cvPointPickingHelper* pickHelper2Snap = new cvPointPickingHelper(
            QKeySequence(tr("Ctrl+2")), true, vtkWidget);
    pickHelper2Snap->setInteractor(m_interactor);
    pickHelper2Snap->setRenderer(m_renderer);
    pickHelper2Snap->setContextWidget(this);
    connect(pickHelper2Snap, &cvPointPickingHelper::pick, this,
            &cvProtractorTool::pickKeyboardPoint2);
    m_pickingHelpers.append(pickHelper2Snap);
}

void cvProtractorTool::showWidget(bool state) {
    if (m_widget) {
        if (state) {
            m_widget->On();
        } else {
            m_widget->Off();
        }
    }

    // Explicitly control representation visibility to ensure arc and label are
    // hidden/shown
    if (m_rep) {
        m_rep->SetVisibility(state ? 1 : 0);
        m_rep->BuildRepresentation();

        // Reapply text properties after BuildRepresentation
        if (state) {
            applyTextPropertiesToLabel();
        }

        if (m_widget) {
            m_widget->Modified();
            m_widget->Render();
        }
    }

    update();
}

ccHObject* cvProtractorTool::getOutput() {
    // Export angle measurement as cc2DLabel with 3 points (triangle/plane)
    // Returns a new cc2DLabel that can be added to the DB tree

    if (!m_entity) {
        CVLog::Warning(
                "[cvProtractorTool::getOutput] No entity associated with this "
                "measurement");
        return nullptr;
    }

    // Get the point coordinates
    double p1[3], center[3], p2[3];
    getPoint1(p1);
    getCenter(center);
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
                "[cvProtractorTool::getOutput] Could not find associated point "
                "cloud or cloud is empty");
        return nullptr;
    }

    // Convert protractor's exact 3D coordinates to CCVector3
    CCVector3 point1(static_cast<PointCoordinateType>(p1[0]),
                     static_cast<PointCoordinateType>(p1[1]),
                     static_cast<PointCoordinateType>(p1[2]));
    CCVector3 pointC(static_cast<PointCoordinateType>(center[0]),
                     static_cast<PointCoordinateType>(center[1]),
                     static_cast<PointCoordinateType>(center[2]));
    CCVector3 point2(static_cast<PointCoordinateType>(p2[0]),
                     static_cast<PointCoordinateType>(p2[1]),
                     static_cast<PointCoordinateType>(p2[2]));

    // Find the nearest points in the cloud for all three measurement endpoints
    // CRITICAL: We need to use the exact protractor coordinates, not just the
    // nearest points in the cloud. If the nearest point is too far away, we
    // should add the exact point to the cloud to ensure the exported label
    // matches the protractor exactly.
    unsigned nearestIndex1 = 0;
    unsigned nearestIndexC = 0;  // center
    unsigned nearestIndex2 = 0;
    double minDist1 = std::numeric_limits<double>::max();
    double minDistC = std::numeric_limits<double>::max();
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

        double dC = (*P - pointC).norm();
        if (dC < minDistC) {
            minDistC = dC;
            nearestIndexC = i;
        }

        double d2 = (*P - point2).norm();
        if (d2 < minDist2) {
            minDist2 = d2;
            nearestIndex2 = i;
        }
    }

    // CRITICAL: If the nearest points are too far from the exact protractor
    // coordinates, add the exact points to the cloud to ensure perfect
    // alignment This ensures the exported label's edges match the protractor's
    // rays exactly
    ccPointCloud* pointCloud = ccHObjectCaster::ToPointCloud(cloud);
    if (pointCloud) {
        // Calculate how many new points we need to add
        unsigned pointsToAdd = 0;
        if (minDist1 > DISTANCE_THRESHOLD) pointsToAdd++;
        if (minDistC > DISTANCE_THRESHOLD) pointsToAdd++;
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

                // Check and add center if needed
                if (minDistC > DISTANCE_THRESHOLD) {
                    pointCloud->addPoint(pointC);
                    nearestIndexC = pointCloud->size() - 1;
                }

                // Check and add point2 if needed
                if (minDist2 > DISTANCE_THRESHOLD) {
                    pointCloud->addPoint(point2);
                    nearestIndex2 = pointCloud->size() - 1;
                }
            }
        }
    }

    // Create a new 2D label with the three points
    // For angle/protractor measurements, the label displays as a triangle with
    // angle info
    cc2DLabel* label = new cc2DLabel(
            QString("Angle: %1°").arg(getMeasurementValue(), 0, 'f', 2));

    // Add the three picked points to the label
    // Order: A (point1), B (center/vertex), C (point2) - center is the vertex
    // of the angle
    if (!label->addPickedPoint(cloud, nearestIndex1)) {
        CVLog::Warning(
                "[cvProtractorTool::getOutput] Failed to add first point (A) "
                "to label");
        delete label;
        return nullptr;
    }

    if (!label->addPickedPoint(cloud, nearestIndexC)) {
        CVLog::Warning(
                "[cvProtractorTool::getOutput] Failed to add center point (B) "
                "to label");
        delete label;
        return nullptr;
    }

    if (!label->addPickedPoint(cloud, nearestIndex2)) {
        CVLog::Warning(
                "[cvProtractorTool::getOutput] Failed to add second point (C) "
                "to label");
        delete label;
        return nullptr;
    }

    // Configure the label display settings
    label->setVisible(true);
    label->setEnabled(true);
    label->setDisplayedIn2D(true);
    label->displayPointLegend(true);
    label->setCollapsed(false);

    // Get the angle label's screen position from VTK representation
    // This ensures the exported label appears at the same location as the 3D
    // angle label
    float labelPosX = 0.05f;  // Default fallback position
    float labelPosY = 0.90f;  // Default fallback position

    if (m_rep && m_renderer) {
        // Get the angle label actor position (in display/pixel coordinates)
        if (auto* labelActor = m_rep->GetAngleLabelActor()) {
            double* vtkPos = labelActor->GetPosition();  // Returns [x, y] in
                                                         // display coordinates
            if (vtkPos && m_interactor && m_interactor->GetRenderWindow()) {
                int* windowSize = m_interactor->GetRenderWindow()->GetSize();
                if (windowSize && windowSize[0] > 0 && windowSize[1] > 0) {
                    // Convert from VTK display coordinates (pixels, bottom-left
                    // origin) to cc2DLabel relative coordinates (0.0-1.0,
                    // top-left origin)
                    float normalizedX = static_cast<float>(vtkPos[0]) /
                                        static_cast<float>(windowSize[0]);
                    float normalizedY = static_cast<float>(vtkPos[1]) /
                                        static_cast<float>(windowSize[1]);

                    // VTK Y=0 is at bottom, cc2DLabel Y=0 is at top
                    // So invert Y: labelY = 1.0 - vtkY
                    labelPosX = normalizedX;
                    labelPosY = 1.0f - normalizedY;

                    CVLog::PrintVerbose(
                            QString("[cvProtractorTool::getOutput] Retrieved "
                                    "angle label "
                                    "position from VTK: display=(%1, %2), "
                                    "normalized=(%3, %4)")
                                    .arg(vtkPos[0])
                                    .arg(vtkPos[1])
                                    .arg(labelPosX)
                                    .arg(labelPosY));
                }
            }
        }
    }

    // Set the position for the exported label (relative to screen, 0.0-1.0)
    label->setPosition(labelPosX, labelPosY);

    CVLog::Print(QString("[cvProtractorTool] Exported angle measurement: %1° "
                         "at position (%2, %3)")
                         .arg(getMeasurementValue(), 0, 'f', 2)
                         .arg(labelPosX)
                         .arg(labelPosY));

    return label;
}

double cvProtractorTool::getMeasurementValue() const {
    if (m_configUi) {
        return m_configUi->angleSpinBox->value();
    }
    return 0.0;
}

void cvProtractorTool::getPoint1(double pos[3]) const {
    if (m_configUi && pos) {
        pos[0] = m_configUi->point1XSpinBox->value();
        pos[1] = m_configUi->point1YSpinBox->value();
        pos[2] = m_configUi->point1ZSpinBox->value();
    } else if (pos) {
        pos[0] = pos[1] = pos[2] = 0.0;
    }
}

void cvProtractorTool::getPoint2(double pos[3]) const {
    if (m_configUi && pos) {
        pos[0] = m_configUi->point2XSpinBox->value();
        pos[1] = m_configUi->point2YSpinBox->value();
        pos[2] = m_configUi->point2ZSpinBox->value();
    } else if (pos) {
        pos[0] = pos[1] = pos[2] = 0.0;
    }
}

void cvProtractorTool::getCenter(double pos[3]) const {
    if (m_configUi && pos) {
        pos[0] = m_configUi->centerXSpinBox->value();
        pos[1] = m_configUi->centerYSpinBox->value();
        pos[2] = m_configUi->centerZSpinBox->value();
    } else if (pos) {
        pos[0] = pos[1] = pos[2] = 0.0;
    }
}

void cvProtractorTool::setPoint1(double pos[3]) {
    if (!m_configUi || !pos) return;

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

    // Update 3D widget directly
    if (m_widget && m_widget->GetEnabled()) {
        m_rep->SetPoint1WorldPosition(pos);
        m_rep->BuildRepresentation();

        // Reapply text properties after BuildRepresentation
        applyTextPropertiesToLabel();

        m_widget->Modified();
        m_widget->Render();
    }

    // Update angle display
    updateAngleDisplay();
    update();
}

void cvProtractorTool::setPoint2(double pos[3]) {
    if (!m_configUi || !pos) return;

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

    // Update 3D widget directly
    if (m_widget && m_widget->GetEnabled()) {
        m_rep->SetPoint2WorldPosition(pos);
        m_rep->BuildRepresentation();

        // Reapply text properties after BuildRepresentation
        applyTextPropertiesToLabel();

        m_widget->Modified();
        m_widget->Render();
    }

    // Update angle display
    updateAngleDisplay();
    update();
}

void cvProtractorTool::setCenter(double pos[3]) {
    if (!m_configUi || !pos) return;

    // Uncheck the pick button
    if (m_configUi->pickCenterToolButton->isChecked()) {
        m_configUi->pickCenterToolButton->setChecked(false);
    }

    // Update spinboxes without triggering signals
    VtkUtils::SignalBlocker blocker1(m_configUi->centerXSpinBox);
    VtkUtils::SignalBlocker blocker2(m_configUi->centerYSpinBox);
    VtkUtils::SignalBlocker blocker3(m_configUi->centerZSpinBox);
    m_configUi->centerXSpinBox->setValue(pos[0]);
    m_configUi->centerYSpinBox->setValue(pos[1]);
    m_configUi->centerZSpinBox->setValue(pos[2]);

    // Update 3D widget directly
    if (m_widget && m_widget->GetEnabled()) {
        m_rep->SetCenterWorldPosition(pos);
        m_rep->BuildRepresentation();

        // Reapply text properties after BuildRepresentation
        applyTextPropertiesToLabel();

        m_widget->Modified();
        m_widget->Render();
    }

    // Update angle display
    updateAngleDisplay();
    update();
}

void cvProtractorTool::setColor(double r, double g, double b) {
    // Store current color
    m_currentColor[0] = r;
    m_currentColor[1] = g;
    m_currentColor[2] = b;

    // Set color for poly line representation (rays)
    if (m_rep) {
        if (auto* lineProp = m_rep->GetLineProperty()) {
            lineProp->SetColor(r, g, b);
        }
        if (auto* selectedLineProp = m_rep->GetSelectedLineProperty()) {
            selectedLineProp->SetColor(r, g, b);
        }
        m_rep->BuildRepresentation();

        // Reapply text properties after BuildRepresentation
        applyTextPropertiesToLabel();

        if (m_widget) {
            m_widget->Modified();
        }
    }
    update();
}

bool cvProtractorTool::getColor(double& r, double& g, double& b) const {
    r = m_currentColor[0];
    g = m_currentColor[1];
    b = m_currentColor[2];
    return true;
}

void cvProtractorTool::lockInteraction() {
    // Disable VTK widget interaction (handles cannot be moved)
    if (m_widget) {
        m_widget->SetProcessEvents(0);  // Disable event processing
    }

    // Change all widget elements to indicate locked state (very dimmed, 10%
    // brightness)
    if (m_rep) {
        // 1. Dim rays (10% brightness, 50% opacity for very obvious locked
        // effect)
        if (auto* lineProp = m_rep->GetLineProperty()) {
            lineProp->SetColor(m_currentColor[0] * 0.1, m_currentColor[1] * 0.1,
                               m_currentColor[2] * 0.1);
            lineProp->SetOpacity(0.5);
        }
        if (auto* selectedLineProp = m_rep->GetSelectedLineProperty()) {
            selectedLineProp->SetColor(m_currentColor[0] * 0.1,
                                       m_currentColor[1] * 0.1,
                                       m_currentColor[2] * 0.1);
            selectedLineProp->SetOpacity(0.5);
        }

        // 2. Dim handles (points) - 50% opacity
        if (auto* handleProp = m_rep->GetHandleProperty()) {
            handleProp->SetOpacity(0.5);
        }
        if (auto* selectedHandleProp = m_rep->GetSelectedHandleProperty()) {
            selectedHandleProp->SetOpacity(0.5);
        }

        // First build representation to update geometry
        m_rep->BuildRepresentation();

        // Then set locked state properties AFTER BuildRepresentation
        // (BuildRepresentation may reset some properties, so we override them
        // here)

        // 3. Dim angle label - 50% opacity with dark gray
        if (auto* labelActor = m_rep->GetAngleLabelActor()) {
            if (auto* textProp = labelActor->GetTextProperty()) {
                textProp->SetOpacity(0.5);
                textProp->SetColor(0.5, 0.5,
                                   0.5);  // Very dark gray for locked state
                textProp->Modified();     // Mark as modified
            }
            labelActor->Modified();  // Mark actor as modified
        }

        // 4. Dim angle arc - 50% opacity with very dark yellow
        if (auto* arcActor = m_rep->GetAngleArcActor()) {
            if (auto* arcProp = arcActor->GetProperty()) {
                arcProp->SetOpacity(0.5);
                arcProp->SetColor(0.5, 0.5,
                                  0.0);  // Very dark yellow for locked state
                arcProp->Modified();     // Mark as modified
            }
            arcActor->Modified();  // Mark actor as modified
        }

        if (m_widget) {
            m_widget->Modified();
            m_widget->Render();  // Force render to apply visual changes
        }
    }

    // Force render window update to show locked state
    if (m_interactor && m_interactor->GetRenderWindow()) {
        m_interactor->GetRenderWindow()->Render();
    }

    // Disable UI controls
    if (m_configUi) {
        m_configUi->point1XSpinBox->setEnabled(false);
        m_configUi->point1YSpinBox->setEnabled(false);
        m_configUi->point1ZSpinBox->setEnabled(false);
        m_configUi->centerXSpinBox->setEnabled(false);
        m_configUi->centerYSpinBox->setEnabled(false);
        m_configUi->centerZSpinBox->setEnabled(false);
        m_configUi->point2XSpinBox->setEnabled(false);
        m_configUi->point2YSpinBox->setEnabled(false);
        m_configUi->point2ZSpinBox->setEnabled(false);
        m_configUi->pickPoint1ToolButton->setEnabled(false);
        m_configUi->pickCenterToolButton->setEnabled(false);
        m_configUi->pickPoint2ToolButton->setEnabled(false);
        m_configUi->widgetVisibilityCheckBox->setEnabled(false);
        m_configUi->arcVisibilityCheckBox->setEnabled(false);
    }

    // Disable keyboard shortcuts
    disableShortcuts();
}

void cvProtractorTool::unlockInteraction() {
    // Enable VTK widget interaction
    if (m_widget) {
        m_widget->SetProcessEvents(1);  // Enable event processing
    }

    // Restore all widget elements to active/unlocked state (full color and
    // opacity)
    if (m_rep) {
        // 1. Restore rays (full brightness and opacity)
        if (auto* lineProp = m_rep->GetLineProperty()) {
            lineProp->SetColor(m_currentColor[0], m_currentColor[1],
                               m_currentColor[2]);
            lineProp->SetOpacity(1.0);
        }
        if (auto* selectedLineProp = m_rep->GetSelectedLineProperty()) {
            selectedLineProp->SetColor(m_currentColor[0], m_currentColor[1],
                                       m_currentColor[2]);
            selectedLineProp->SetOpacity(1.0);
        }

        // 2. Restore handles (points)
        if (auto* handleProp = m_rep->GetHandleProperty()) {
            handleProp->SetOpacity(1.0);
        }
        if (auto* selectedHandleProp = m_rep->GetSelectedHandleProperty()) {
            selectedHandleProp->SetOpacity(1.0);
        }

        // First build representation to update geometry
        m_rep->BuildRepresentation();

        // Then set unlocked state properties AFTER BuildRepresentation
        // (BuildRepresentation may reset some properties, so we override them
        // here)

        // 3. Restore angle label to user-configured settings
        if (auto* labelActor = m_rep->GetAngleLabelActor()) {
            if (auto* textProp = labelActor->GetTextProperty()) {
                textProp->SetColor(m_fontColor[0], m_fontColor[1],
                                   m_fontColor[2]);
                textProp->SetOpacity(m_fontOpacity);
                textProp->Modified();  // Mark as modified
            }
            labelActor->Modified();  // Mark actor as modified
        }

        // 4. Restore angle arc (full opacity and yellow color)
        if (auto* arcActor = m_rep->GetAngleArcActor()) {
            if (auto* arcProp = arcActor->GetProperty()) {
                arcProp->SetOpacity(1.0);
                arcProp->SetColor(1.0, 1.0, 0.0);  // Yellow color
                arcProp->Modified();               // Mark as modified
            }
            arcActor->Modified();  // Mark actor as modified
        }

        if (m_widget) {
            m_widget->Modified();
            m_widget->Render();  // Force render to apply visual changes
        }
    }

    // Force render window update to show unlocked state
    if (m_interactor && m_interactor->GetRenderWindow()) {
        m_interactor->GetRenderWindow()->Render();
    }

    // Enable UI controls
    if (m_configUi) {
        m_configUi->point1XSpinBox->setEnabled(true);
        m_configUi->point1YSpinBox->setEnabled(true);
        m_configUi->point1ZSpinBox->setEnabled(true);
        m_configUi->centerXSpinBox->setEnabled(true);
        m_configUi->centerYSpinBox->setEnabled(true);
        m_configUi->centerZSpinBox->setEnabled(true);
        m_configUi->point2XSpinBox->setEnabled(true);
        m_configUi->point2YSpinBox->setEnabled(true);
        m_configUi->point2ZSpinBox->setEnabled(true);
        m_configUi->pickPoint1ToolButton->setEnabled(true);
        m_configUi->pickCenterToolButton->setEnabled(true);
        m_configUi->pickPoint2ToolButton->setEnabled(true);
        m_configUi->widgetVisibilityCheckBox->setEnabled(true);
        m_configUi->arcVisibilityCheckBox->setEnabled(true);
    }

    // Re-enable keyboard shortcuts
    if (m_pickingHelpers.isEmpty()) {
        // Shortcuts haven't been created yet - create them now
        if (m_vtkWidget) {
            CVLog::PrintDebug(
                    QString("[cvProtractorTool::unlockInteraction] Creating "
                            "shortcuts for tool=%1, using saved vtkWidget=%2")
                            .arg((quintptr)this, 0, 16)
                            .arg((quintptr)m_vtkWidget, 0, 16));
            setupShortcuts(m_vtkWidget);
            CVLog::PrintDebug(
                    QString("[cvProtractorTool::unlockInteraction] After "
                            "setupShortcuts, m_pickingHelpers.size()=%1")
                            .arg(m_pickingHelpers.size()));
        } else {
            CVLog::PrintDebug(
                    QString("[cvProtractorTool::unlockInteraction] m_vtkWidget "
                            "is null for tool=%1, cannot create shortcuts")
                            .arg((quintptr)this, 0, 16));
        }
    } else {
        // Shortcuts already exist - update interactor/renderer and enable them
        // CRITICAL: Update interactor/renderer in case they weren't set when
        // shortcuts were created (e.g., in addInstance before tool is fully
        // initialized)
        CVLog::PrintDebug(QString("[cvProtractorTool::unlockInteraction] "
                                  "Updating %1 existing shortcuts for tool=%2")
                                  .arg(m_pickingHelpers.size())
                                  .arg((quintptr)this, 0, 16));
        updatePickingHelpers();

        // CRITICAL: Enable all shortcuts. Now using ecvModalShortcut
        // (ParaView-style) which automatically handles conflicts via
        // ecvKeySequences. When a shortcut is enabled,
        // ecvKeySequences::disableSiblings() automatically disables all other
        // shortcuts with the same key sequence, ensuring only the active
        // (unlocked) tool instance's shortcuts respond.
        //
        // IMPORTANT: However, QAction shortcuts (like MainWindow's
        // actionContourWidget with "Ctrl+C") are NOT managed by ecvKeySequences
        // and may still conflict. The MainWindow action's shortcut is handled
        // by ecvMeasurementTool when tools are activated/deactivated.
        //
        // The pickPoint() method in cvPointPickingHelper already checks for
        // visibility and enabled state to ensure only the active tool instance
        // responds.
        for (cvPointPickingHelper* helper : m_pickingHelpers) {
            if (helper) {
                // Simply enable - ecvModalShortcut will handle disabling
                // siblings
                helper->setEnabled(true,
                                   false);  // Enable without setting focus
            }
        }
    }
}

void cvProtractorTool::setInstanceLabel(const QString& label) {
    // Store the instance label
    m_instanceLabel = label;

    // Update the VTK representation's label suffix
    if (m_rep) {
        m_rep->SetLabelSuffix(m_instanceLabel.toUtf8().constData());

        // Call applyFontProperties() which will rebuild representation
        // and reapply font properties correctly
        applyFontProperties();
    }
}

void cvProtractorTool::on_point1XSpinBox_valueChanged(double arg1) {
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
        m_widget->Render();
    }
    updateAngleDisplay();
    update();
}

void cvProtractorTool::on_point1YSpinBox_valueChanged(double arg1) {
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
        m_widget->Render();
    }
    updateAngleDisplay();
    update();
}

void cvProtractorTool::on_point1ZSpinBox_valueChanged(double arg1) {
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
        m_widget->Render();
    }
    updateAngleDisplay();
    update();
}

void cvProtractorTool::on_centerXSpinBox_valueChanged(double arg1) {
    if (!m_configUi) return;
    double pos[3];
    double newPos[3];
    newPos[0] = arg1;

    VtkUtils::SignalBlocker blocker(m_configUi->centerXSpinBox);

    if (m_widget && m_widget->GetEnabled()) {
        m_rep->GetCenterWorldPosition(pos);
        newPos[1] = pos[1];
        newPos[2] = pos[2];
        m_rep->SetCenterWorldPosition(newPos);
        m_rep->BuildRepresentation();
        m_widget->Modified();
        m_widget->Render();
    }
    updateAngleDisplay();
    update();
}

void cvProtractorTool::on_centerYSpinBox_valueChanged(double arg1) {
    if (!m_configUi) return;
    double pos[3];
    double newPos[3];
    newPos[1] = arg1;

    VtkUtils::SignalBlocker blocker(m_configUi->centerYSpinBox);

    if (m_widget && m_widget->GetEnabled()) {
        m_rep->GetCenterWorldPosition(pos);
        newPos[0] = pos[0];
        newPos[2] = pos[2];
        m_rep->SetCenterWorldPosition(newPos);
        m_rep->BuildRepresentation();
        m_widget->Modified();
        m_widget->Render();
    }
    updateAngleDisplay();
    update();
}

void cvProtractorTool::on_centerZSpinBox_valueChanged(double arg1) {
    if (!m_configUi) return;
    double pos[3];
    double newPos[3];
    newPos[2] = arg1;

    VtkUtils::SignalBlocker blocker(m_configUi->centerZSpinBox);

    if (m_widget && m_widget->GetEnabled()) {
        m_rep->GetCenterWorldPosition(pos);
        newPos[0] = pos[0];
        newPos[1] = pos[1];
        m_rep->SetCenterWorldPosition(newPos);
        m_rep->BuildRepresentation();
        m_widget->Modified();
        m_widget->Render();
    }
    updateAngleDisplay();
    update();
}

void cvProtractorTool::on_point2XSpinBox_valueChanged(double arg1) {
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
        m_widget->Render();
    }
    updateAngleDisplay();
    update();
}

void cvProtractorTool::on_point2YSpinBox_valueChanged(double arg1) {
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
        m_widget->Render();
    }
    updateAngleDisplay();
    update();
}

void cvProtractorTool::on_point2ZSpinBox_valueChanged(double arg1) {
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
        m_widget->Render();
    }
    updateAngleDisplay();
    update();
}

void cvProtractorTool::onAngleChanged(double angle) {
    if (!m_configUi) return;
    VtkUtils::SignalBlocker blocker(m_configUi->angleSpinBox);
    m_configUi->angleSpinBox->setValue(angle);
    emit measurementValueChanged();
}

void cvProtractorTool::onWorldPoint1Changed(double* pos) {
    if (!m_configUi || !pos) return;
    VtkUtils::SignalBlocker blocker1(m_configUi->point1XSpinBox);
    VtkUtils::SignalBlocker blocker2(m_configUi->point1YSpinBox);
    VtkUtils::SignalBlocker blocker3(m_configUi->point1ZSpinBox);
    m_configUi->point1XSpinBox->setValue(pos[0]);
    m_configUi->point1YSpinBox->setValue(pos[1]);
    m_configUi->point1ZSpinBox->setValue(pos[2]);
    emit measurementValueChanged();
}

void cvProtractorTool::onWorldPoint2Changed(double* pos) {
    if (!m_configUi || !pos) return;
    VtkUtils::SignalBlocker blocker1(m_configUi->point2XSpinBox);
    VtkUtils::SignalBlocker blocker2(m_configUi->point2YSpinBox);
    VtkUtils::SignalBlocker blocker3(m_configUi->point2ZSpinBox);
    m_configUi->point2XSpinBox->setValue(pos[0]);
    m_configUi->point2YSpinBox->setValue(pos[1]);
    m_configUi->point2ZSpinBox->setValue(pos[2]);
    emit measurementValueChanged();
}

void cvProtractorTool::onWorldCenterChanged(double* pos) {
    if (!m_configUi || !pos) return;
    VtkUtils::SignalBlocker blocker1(m_configUi->centerXSpinBox);
    VtkUtils::SignalBlocker blocker2(m_configUi->centerYSpinBox);
    VtkUtils::SignalBlocker blocker3(m_configUi->centerZSpinBox);
    m_configUi->centerXSpinBox->setValue(pos[0]);
    m_configUi->centerYSpinBox->setValue(pos[1]);
    m_configUi->centerZSpinBox->setValue(pos[2]);
    emit measurementValueChanged();
}

void cvProtractorTool::hookWidget(
        const vtkSmartPointer<cvConstrainedPolyLineWidget>& widget) {
    // TODO: Create a PolyLineWidgetObserver similar to AngleWidgetObserver
    // For now, we'll use direct callbacks via vtkCommand::InteractionEvent

    vtkNew<vtkCallbackCommand> callback;
    callback->SetCallback([](vtkObject* caller, unsigned long /*eid*/,
                             void* clientData, void* /*callData*/) {
        cvProtractorTool* self = static_cast<cvProtractorTool*>(clientData);
        self->updateAngleDisplay();
    });
    callback->SetClientData(this);

    widget->AddObserver(vtkCommand::InteractionEvent, callback);
}

void cvProtractorTool::updateAngleDisplay() {
    if (!m_configUi) {
        CVLog::Warning(
                "[cvProtractorTool] updateAngleDisplay: m_configUi is null!");
        return;
    }

    // cvConstrainedPolyLineRepresentation::GetAngle() returns DEGREES
    double angleDegrees = 0.0;

    if (m_rep) {
        angleDegrees = m_rep->GetAngle();

        // Also update the point coordinates in the UI
        double p1[3], center[3], p2[3];
        m_rep->GetHandlePosition(0, p1);
        m_rep->GetHandlePosition(1, center);
        m_rep->GetHandlePosition(2, p2);

        VtkUtils::SignalBlocker blocker1(m_configUi->point1XSpinBox);
        VtkUtils::SignalBlocker blocker2(m_configUi->point1YSpinBox);
        VtkUtils::SignalBlocker blocker3(m_configUi->point1ZSpinBox);
        VtkUtils::SignalBlocker blocker4(m_configUi->centerXSpinBox);
        VtkUtils::SignalBlocker blocker5(m_configUi->centerYSpinBox);
        VtkUtils::SignalBlocker blocker6(m_configUi->centerZSpinBox);
        VtkUtils::SignalBlocker blocker7(m_configUi->point2XSpinBox);
        VtkUtils::SignalBlocker blocker8(m_configUi->point2YSpinBox);
        VtkUtils::SignalBlocker blocker9(m_configUi->point2ZSpinBox);

        m_configUi->point1XSpinBox->setValue(p1[0]);
        m_configUi->point1YSpinBox->setValue(p1[1]);
        m_configUi->point1ZSpinBox->setValue(p1[2]);
        m_configUi->centerXSpinBox->setValue(center[0]);
        m_configUi->centerYSpinBox->setValue(center[1]);
        m_configUi->centerZSpinBox->setValue(center[2]);
        m_configUi->point2XSpinBox->setValue(p2[0]);
        m_configUi->point2YSpinBox->setValue(p2[1]);
        m_configUi->point2ZSpinBox->setValue(p2[2]);
    }

    VtkUtils::SignalBlocker blocker(m_configUi->angleSpinBox);
    m_configUi->angleSpinBox->setValue(angleDegrees);
    emit measurementValueChanged();
}

void cvProtractorTool::on_pickPoint1_toggled(bool checked) {
    if (checked) {
        // Uncheck other pick buttons
        if (m_configUi->pickCenterToolButton->isChecked()) {
            m_configUi->pickCenterToolButton->setChecked(false);
        }
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

void cvProtractorTool::on_pickCenter_toggled(bool checked) {
    if (checked) {
        // Uncheck other pick buttons
        if (m_configUi->pickPoint1ToolButton->isChecked()) {
            m_configUi->pickPoint1ToolButton->setChecked(false);
        }
        if (m_configUi->pickPoint2ToolButton->isChecked()) {
            m_configUi->pickPoint2ToolButton->setChecked(false);
        }
        // Enable point picking mode for center
        emit pointPickingRequested(3);
    } else {
        // Disable point picking
        emit pointPickingCancelled();
    }
}

void cvProtractorTool::on_pickPoint2_toggled(bool checked) {
    if (checked) {
        // Uncheck other pick buttons
        if (m_configUi->pickPoint1ToolButton->isChecked()) {
            m_configUi->pickPoint1ToolButton->setChecked(false);
        }
        if (m_configUi->pickCenterToolButton->isChecked()) {
            m_configUi->pickCenterToolButton->setChecked(false);
        }
        // Enable point picking mode for point 2
        emit pointPickingRequested(2);
    } else {
        // Disable point picking
        emit pointPickingCancelled();
    }
}

void cvProtractorTool::on_widgetVisibilityCheckBox_toggled(bool checked) {
    if (!m_configUi) return;

    // Show or hide the widget
    if (m_widget) {
        if (checked) {
            m_widget->On();
        } else {
            m_widget->Off();
        }
    }

    // Explicitly control representation visibility to ensure arc and label are
    // hidden/shown
    if (m_rep) {
        m_rep->SetVisibility(checked ? 1 : 0);
        m_rep->BuildRepresentation();

        // Reapply text properties after BuildRepresentation
        if (checked) {
            applyTextPropertiesToLabel();
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

    update();
}

void cvProtractorTool::on_arcVisibilityCheckBox_toggled(bool checked) {
    if (!m_configUi) return;

    // Show or hide the arc
    if (m_rep) {
        m_rep->SetShowAngleArc(checked ? 1 : 0);
        m_rep->BuildRepresentation();

        // Reapply text properties after BuildRepresentation
        applyTextPropertiesToLabel();

        // Explicitly control arc actor visibility
        if (auto* arcActor = m_rep->GetAngleArcActor()) {
            arcActor->SetVisibility(checked);
        }

        if (m_widget) {
            m_widget->Modified();
            m_widget->Render();
        }
    }

    update();
}

void cvProtractorTool::pickKeyboardPoint1(double x, double y, double z) {
    double pos[3] = {x, y, z};
    setPoint1(pos);
}

void cvProtractorTool::pickKeyboardCenter(double x, double y, double z) {
    double pos[3] = {x, y, z};
    setCenter(pos);
}

void cvProtractorTool::pickKeyboardPoint2(double x, double y, double z) {
    double pos[3] = {x, y, z};
    setPoint2(pos);
}

void cvProtractorTool::applyTextPropertiesToLabel() {
    if (!m_rep) return;

    // Apply font properties to the angle label actor
    // This does NOT call BuildRepresentation()
    if (auto* labelActor = m_rep->GetAngleLabelActor()) {
        if (auto* textProp = labelActor->GetTextProperty()) {
            // Set font family and size
            textProp->SetFontFamilyAsString(m_fontFamily.toUtf8().constData());
            textProp->SetFontSize(m_fontSize);

            // Set color and opacity
            textProp->SetColor(m_fontColor[0], m_fontColor[1], m_fontColor[2]);
            textProp->SetOpacity(m_fontOpacity);

            // Set style properties
            textProp->SetBold(m_fontBold ? 1 : 0);
            textProp->SetItalic(m_fontItalic ? 1 : 0);
            textProp->SetShadow(m_fontShadow ? 1 : 0);

            // Apply horizontal justification
            if (m_horizontalJustification == "Left") {
                textProp->SetJustificationToLeft();
            } else if (m_horizontalJustification == "Center") {
                textProp->SetJustificationToCentered();
            } else if (m_horizontalJustification == "Right") {
                textProp->SetJustificationToRight();
            }

            // Apply vertical justification
            if (m_verticalJustification == "Top") {
                textProp->SetVerticalJustificationToTop();
            } else if (m_verticalJustification == "Center") {
                textProp->SetVerticalJustificationToCentered();
            } else if (m_verticalJustification == "Bottom") {
                textProp->SetVerticalJustificationToBottom();
            }

            // Mark text property as modified to ensure VTK updates
            textProp->Modified();
        }

        // Mark label actor as modified to trigger re-render
        labelActor->Modified();
    }
}

void cvProtractorTool::applyFontProperties() {
    if (!m_rep) return;

    // CRITICAL: First rebuild representation to update geometry
    // Then set text properties AFTER BuildRepresentation()
    // BuildRepresentation() may reset some text properties to defaults,
    // so we must set them AFTER, not before (like cvDistanceTool does)
    m_rep->BuildRepresentation();

    // Apply text properties AFTER BuildRepresentation
    applyTextPropertiesToLabel();

    // Mark widget as modified and trigger render
    if (m_widget) {
        m_widget->Modified();
        m_widget->Render();
    }

    // Force render window update to apply changes immediately
    if (m_interactor && m_interactor->GetRenderWindow()) {
        m_interactor->GetRenderWindow()->Render();
    }

    // Update Qt widget
    update();
}
