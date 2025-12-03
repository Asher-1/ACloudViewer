// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvDistanceTool.h"

#include "Tools/PickingTools/cvPointPickingHelper.h"

#include <QShortcut>

#include <algorithm>

#include <VtkUtils/distancewidgetobserver.h>
#include <VtkUtils/signalblocker.h>
#include <VtkUtils/vtkutils.h>
#include <vtkAxisActor2D.h>
#include <vtkCommand.h>
#include <vtkDistanceRepresentation2D.h>
#include <vtkDistanceRepresentation3D.h>
#include <vtkDistanceWidget.h>
#include <vtkHandleRepresentation.h>
#include <vtkMath.h>
#include <vtkPointHandleRepresentation2D.h>
#include <vtkPointHandleRepresentation3D.h>
#include <vtkProperty.h>
#include <vtkProperty2D.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>

// ECV_DB_LIB
#include <ecvBBox.h>
#include <ecvHObject.h>

namespace {

// ParaView-style colors
constexpr double FOREGROUND_COLOR[3] = {1.0, 1.0, 1.0};  // White for normal state
constexpr double INTERACTION_COLOR[3] = {0.0, 1.0, 0.0}; // Green for selected/interactive state

//! Configure 2D handle representation with ParaView-style properties
void configureHandle2D(vtkPointHandleRepresentation2D* handle) {
    if (!handle) return;
    
    // Set normal (foreground) color - white
    if (auto* prop = handle->GetProperty()) {
        prop->SetColor(FOREGROUND_COLOR[0], FOREGROUND_COLOR[1], FOREGROUND_COLOR[2]);
    }
    
    // Set selected (interaction) color - green
    if (auto* selectedProp = handle->GetSelectedProperty()) {
        selectedProp->SetColor(INTERACTION_COLOR[0], INTERACTION_COLOR[1], INTERACTION_COLOR[2]);
    }
}

//! Configure 3D handle representation with ParaView-style properties
void configureHandle3D(vtkPointHandleRepresentation3D* handle) {
    if (!handle) return;
    
    // Set normal (foreground) color - white
    if (auto* prop = handle->GetProperty()) {
        prop->SetColor(FOREGROUND_COLOR[0], FOREGROUND_COLOR[1], FOREGROUND_COLOR[2]);
    }
    
    // Set selected (interaction) color - green
    if (auto* selectedProp = handle->GetSelectedProperty()) {
        selectedProp->SetColor(INTERACTION_COLOR[0], INTERACTION_COLOR[1], INTERACTION_COLOR[2]);
    }
    
    // Configure cursor appearance - show only the crosshair axes, no outline/shadows
    handle->AllOff();  // Turn off outline and all shadows
    
    // Enable smooth motion and translation mode for better handle movement
    handle->SmoothMotionOn();
    handle->TranslationModeOn();
}

//! Configure distance representation 2D with ParaView-style properties
void configureDistanceRep2D(vtkDistanceRepresentation2D* rep) {
    if (!rep) return;
    
    // Configure handles
    auto* h1 = vtkPointHandleRepresentation2D::SafeDownCast(rep->GetPoint1Representation());
    auto* h2 = vtkPointHandleRepresentation2D::SafeDownCast(rep->GetPoint2Representation());
    configureHandle2D(h1);
    configureHandle2D(h2);
    
    // The axis (line) color is already set to green (0,1,0) by default in VTK
    // which matches ParaView's style
}

//! Configure distance representation 3D with ParaView-style properties
void configureDistanceRep3D(vtkDistanceRepresentation3D* rep) {
    if (!rep) return;
    
    // Configure handles
    auto* h1 = vtkPointHandleRepresentation3D::SafeDownCast(rep->GetPoint1Representation());
    auto* h2 = vtkPointHandleRepresentation3D::SafeDownCast(rep->GetPoint2Representation());
    configureHandle3D(h1);
    configureHandle3D(h2);
    
    // Configure line properties (matching ParaView's LineProperty)
    if (auto* lineProp = rep->GetLineProperty()) {
        lineProp->SetColor(FOREGROUND_COLOR[0], FOREGROUND_COLOR[1], FOREGROUND_COLOR[2]);
        lineProp->SetLineWidth(2.0);  // ParaView default line width
        lineProp->SetAmbient(1.0);     // ParaView sets ambient to 1.0
    }
}

} // anonymous namespace

cvDistanceTool::cvDistanceTool(QWidget* parent)
    : cvGenericMeasurementTool(parent), m_configUi(nullptr) {
    setWindowTitle(tr("Distance Measurement Tool"));
}

cvDistanceTool::~cvDistanceTool() {
    if (m_configUi) {
        delete m_configUi;
        m_configUi = nullptr;
    }
}

void cvDistanceTool::initTool() {
    // Initialize only 2D widget as default (matching UI currentIndex=0)
    // 3D widget will be lazily initialized when user switches to it
    
    VtkUtils::vtkInitOnce(m_2dRep);
    VtkUtils::vtkInitOnce(m_2dWidget);
    
    // Set representation BEFORE calling SetInteractor/SetRenderer
    m_2dWidget->SetRepresentation(m_2dRep);
    
    if (m_interactor) {
        m_2dWidget->SetInteractor(m_interactor);
    }
    if (m_renderer) {
        m_2dRep->SetRenderer(m_renderer);
    }
    
    // Initialize ruler mode settings from UI
    if (m_configUi) {
        m_2dRep->SetRulerMode(m_configUi->rulerModeCheckBox->isChecked() ? 1 : 0);
        m_2dRep->SetRulerDistance(m_configUi->rulerDistanceSpinBox->value());
        m_2dRep->SetNumberOfRulerTicks(m_configUi->numberOfTicksSpinBox->value());
        m_2dRep->SetScale(m_configUi->scaleSpinBox->value());
        
        // Enable/disable appropriate controls based on ruler mode
        bool rulerMode = m_configUi->rulerModeCheckBox->isChecked();
        m_configUi->rulerDistanceSpinBox->setEnabled(rulerMode);
        m_configUi->numberOfTicksSpinBox->setEnabled(!rulerMode);
    }
    
    // Following ParaView's approach:
    // 1. InstantiateHandleRepresentation (done in CreateDefaultRepresentation)
    m_2dRep->InstantiateHandleRepresentation();
    
    // 2. Configure appearance AFTER instantiation but BEFORE enabling
    configureDistanceRep2D(m_2dRep);
    
    // 3. Build representation
    m_2dRep->BuildRepresentation();
    
    // 4. Enable widget - this will internally handle the handle widgets
    m_2dWidget->On();
    
    hookWidget(m_2dWidget); // Hook observer for 2D widget
}


void cvDistanceTool::createUi() {
    m_configUi = new Ui::DistanceToolDlg;
    QWidget* configWidget = new QWidget(this);
    m_configUi->setupUi(configWidget);
    m_ui->setupUi(this);
    m_ui->configLayout->addWidget(configWidget);
    m_ui->groupBox->setTitle(tr("Distance Parameters"));

#ifdef Q_OS_MAC
    m_configUi->instructionLabel->setText(
        m_configUi->instructionLabel->text().replace("Ctrl", "Cmd"));
#endif

    connect(m_configUi->typeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &cvDistanceTool::on_typeCombo_currentIndexChanged);
    connect(m_configUi->point1XSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &cvDistanceTool::on_point1XSpinBox_valueChanged);
    connect(m_configUi->point1YSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &cvDistanceTool::on_point1YSpinBox_valueChanged);
    connect(m_configUi->point1ZSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &cvDistanceTool::on_point1ZSpinBox_valueChanged);
    connect(m_configUi->point2XSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &cvDistanceTool::on_point2XSpinBox_valueChanged);
    connect(m_configUi->point2YSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &cvDistanceTool::on_point2YSpinBox_valueChanged);
    connect(m_configUi->point2ZSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &cvDistanceTool::on_point2ZSpinBox_valueChanged);
    
    // Connect point picking buttons
    connect(m_configUi->pickPoint1ToolButton, &QToolButton::toggled,
            this, &cvDistanceTool::on_pickPoint1_toggled);
    connect(m_configUi->pickPoint2ToolButton, &QToolButton::toggled,
            this, &cvDistanceTool::on_pickPoint2_toggled);
    
    // Connect ruler mode controls
    connect(m_configUi->rulerModeCheckBox, &QCheckBox::toggled,
            this, &cvDistanceTool::on_rulerModeCheckBox_toggled);
    connect(m_configUi->rulerDistanceSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &cvDistanceTool::on_rulerDistanceSpinBox_valueChanged);
    connect(m_configUi->numberOfTicksSpinBox, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &cvDistanceTool::on_numberOfTicksSpinBox_valueChanged);
    connect(m_configUi->scaleSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &cvDistanceTool::on_scaleSpinBox_valueChanged);
    
    // Connect display options
    connect(m_configUi->widgetVisibilityCheckBox, &QCheckBox::toggled,
            this, &cvDistanceTool::on_widgetVisibilityCheckBox_toggled);
    connect(m_configUi->labelVisibilityCheckBox, &QCheckBox::toggled,
            this, &cvDistanceTool::on_labelVisibilityCheckBox_toggled);
    connect(m_configUi->lineWidthSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &cvDistanceTool::on_lineWidthSpinBox_valueChanged);
}

void cvDistanceTool::start() {
    cvGenericMeasurementTool::start();
    // if (m_renderer) {
    //     m_renderer->ResetCamera();
    // }
}

void cvDistanceTool::reset() {
    // Reset points to default positions (center of bounding box if available)
    double defaultPos1[3] = {0.0, 0.0, 0.0};
    double defaultPos2[3] = {1.0, 0.0, 0.0};
    
    if (m_entity && m_entity->getBB_recursive().isValid()) {
        const ccBBox& bbox = m_entity->getBB_recursive();
        CCVector3 center = bbox.getCenter();
        CCVector3 diag = bbox.getDiagVec();
        
        defaultPos1[0] = center.x - diag.x * 0.25;
        defaultPos1[1] = center.y;
        defaultPos1[2] = center.z;
        
        defaultPos2[0] = center.x + diag.x * 0.25;
        defaultPos2[1] = center.y;
        defaultPos2[2] = center.z;
    }
    
    // Reset widget points
    if (m_2dWidget && m_2dWidget->GetEnabled()) {
        m_2dRep->SetPoint1WorldPosition(defaultPos1);
        m_2dRep->SetPoint2WorldPosition(defaultPos2);
        m_2dRep->BuildRepresentation();
        m_2dWidget->Modified();
    } else if (m_3dWidget && m_3dWidget->GetEnabled()) {
        m_3dRep->SetPoint1WorldPosition(defaultPos1);
        m_3dRep->SetPoint2WorldPosition(defaultPos2);
        m_3dRep->BuildRepresentation();
        m_3dWidget->Modified();
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
    if (m_2dWidget) {
        if (state) {
            m_2dWidget->On();
        } else {
            m_2dWidget->Off();
        }
    }
    if (m_3dWidget) {
        if (state) {
            m_3dWidget->On();
        } else {
            m_3dWidget->Off();
        }
    }
    update();
}

ccHObject* cvDistanceTool::getOutput() {
    return nullptr;
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
    if (m_2dWidget && m_2dWidget->GetEnabled()) {
        m_2dRep->SetPoint1WorldPosition(pos);
        m_2dRep->BuildRepresentation();
        m_2dWidget->Modified();
        m_2dWidget->Render(); // Ensure render
    } else if (m_3dWidget && m_3dWidget->GetEnabled()) {
        m_3dRep->SetPoint1WorldPosition(pos);
        m_3dRep->BuildRepresentation();
        m_3dWidget->Modified();
        m_3dWidget->Render(); // Ensure render
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
    if (m_2dWidget && m_2dWidget->GetEnabled()) {
        m_2dRep->SetPoint2WorldPosition(pos);
        m_2dRep->BuildRepresentation();
        m_2dWidget->Modified();
        m_2dWidget->Render(); // Ensure render
    } else if (m_3dWidget && m_3dWidget->GetEnabled()) {
        m_3dRep->SetPoint2WorldPosition(pos);
        m_3dRep->BuildRepresentation();
        m_3dWidget->Modified();
        m_3dWidget->Render(); // Ensure render
    }
    
    // Update distance display
    updateDistanceDisplay();
    update();
}

void cvDistanceTool::on_typeCombo_currentIndexChanged(int index) {
    if (index == 0) {
        // Switch to 2D widget
        if (!m_2dWidget) {
            // Lazy initialization of 2D widget (shouldn't happen as it's default)
            VtkUtils::vtkInitOnce(m_2dRep);
            VtkUtils::vtkInitOnce(m_2dWidget);
            
            // Set representation BEFORE SetInteractor/SetRenderer
            m_2dWidget->SetRepresentation(m_2dRep);
            
            if (m_interactor) {
                m_2dWidget->SetInteractor(m_interactor);
            }
            if (m_renderer) {
                m_2dRep->SetRenderer(m_renderer);
            }
            
            // Initialize ruler mode settings for 2D widget from UI
            if (m_configUi) {
                m_2dRep->SetRulerMode(m_configUi->rulerModeCheckBox->isChecked() ? 1 : 0);
                m_2dRep->SetRulerDistance(m_configUi->rulerDistanceSpinBox->value());
                m_2dRep->SetNumberOfRulerTicks(m_configUi->numberOfTicksSpinBox->value());
                m_2dRep->SetScale(m_configUi->scaleSpinBox->value());
            }
            
            // Following ParaView's approach:
            // 1. InstantiateHandleRepresentation
            m_2dRep->InstantiateHandleRepresentation();
            
            // 2. Configure appearance AFTER instantiation but BEFORE enabling
            configureDistanceRep2D(m_2dRep);
            
            // 3. Build representation
            m_2dRep->BuildRepresentation();
            
            hookWidget(m_2dWidget); // Hook observer for 2D widget
        }
        if (m_2dWidget) {
            m_2dWidget->On();
        }
        if (m_3dWidget) {
            m_3dWidget->Off();
        }
    } else {
        // Switch to 3D widget
        if (!m_3dWidget) {
            // Lazy initialization of 3D widget
            VtkUtils::vtkInitOnce(m_3dRep);
            VtkUtils::vtkInitOnce(m_3dWidget);
            
            // Set representation BEFORE SetInteractor/SetRenderer
            m_3dWidget->SetRepresentation(m_3dRep);
            
            if (m_interactor) {
                m_3dWidget->SetInteractor(m_interactor);
            }
            if (m_renderer) {
                m_3dRep->SetRenderer(m_renderer);
            }
            
            // Initialize ruler mode settings for 3D widget from UI
            if (m_configUi) {
                m_3dRep->SetRulerMode(m_configUi->rulerModeCheckBox->isChecked() ? 1 : 0);
                m_3dRep->SetRulerDistance(m_configUi->rulerDistanceSpinBox->value());
                m_3dRep->SetNumberOfRulerTicks(m_configUi->numberOfTicksSpinBox->value());
                m_3dRep->SetScale(m_configUi->scaleSpinBox->value());
            }
            
            // Following ParaView's approach:
            // 1. InstantiateHandleRepresentation
            m_3dRep->InstantiateHandleRepresentation();
            
            // 2. Configure appearance AFTER instantiation but BEFORE enabling
            configureDistanceRep3D(m_3dRep);
            
            // 3. Build representation
            m_3dRep->BuildRepresentation();
    
            hookWidget(m_3dWidget); // Hook observer for 3D widget
        }
        if (m_3dWidget) {
            m_3dWidget->On();
        }
        if (m_2dWidget) {
            m_2dWidget->Off();
        }
    }
    update();
}

void cvDistanceTool::on_point1XSpinBox_valueChanged(double arg1) {
    if (!m_configUi) return;
    double pos[3];
    double newPos[3];
    newPos[0] = arg1;
    
    VtkUtils::SignalBlocker blocker(m_configUi->point1XSpinBox);
    
    if (m_2dWidget && m_2dWidget->GetEnabled()) {
        m_2dRep->GetPoint1WorldPosition(pos);
        newPos[1] = pos[1];
        newPos[2] = pos[2];
        m_2dRep->SetPoint1WorldPosition(newPos);
        m_2dRep->BuildRepresentation();
        m_2dWidget->Modified();
        // Update distance display
        updateDistanceDisplay();
    } else if (m_3dWidget && m_3dWidget->GetEnabled()) {
        m_3dRep->GetPoint1WorldPosition(pos);
        newPos[1] = pos[1];
        newPos[2] = pos[2];
        m_3dRep->SetPoint1WorldPosition(newPos);
        m_3dRep->BuildRepresentation();
        m_3dWidget->Modified();
        // Update distance display
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
    
    if (m_2dWidget && m_2dWidget->GetEnabled()) {
        m_2dRep->GetPoint1WorldPosition(pos);
        newPos[0] = pos[0];
        newPos[2] = pos[2];
        m_2dRep->SetPoint1WorldPosition(newPos);
        m_2dRep->BuildRepresentation();
        m_2dWidget->Modified();
        updateDistanceDisplay();
    } else if (m_3dWidget && m_3dWidget->GetEnabled()) {
        m_3dRep->GetPoint1WorldPosition(pos);
        newPos[0] = pos[0];
        newPos[2] = pos[2];
        m_3dRep->SetPoint1WorldPosition(newPos);
        m_3dRep->BuildRepresentation();
        m_3dWidget->Modified();
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
    
    if (m_2dWidget && m_2dWidget->GetEnabled()) {
        m_2dRep->GetPoint1WorldPosition(pos);
        newPos[0] = pos[0];
        newPos[1] = pos[1];
        m_2dRep->SetPoint1WorldPosition(newPos);
        m_2dRep->BuildRepresentation();
        m_2dWidget->Modified();
        updateDistanceDisplay();
    } else if (m_3dWidget && m_3dWidget->GetEnabled()) {
        m_3dRep->GetPoint1WorldPosition(pos);
        newPos[0] = pos[0];
        newPos[1] = pos[1];
        m_3dRep->SetPoint1WorldPosition(newPos);
        m_3dRep->BuildRepresentation();
        m_3dWidget->Modified();
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
    
    if (m_2dWidget && m_2dWidget->GetEnabled()) {
        m_2dRep->GetPoint2WorldPosition(pos);
        newPos[1] = pos[1];
        newPos[2] = pos[2];
        m_2dRep->SetPoint2WorldPosition(newPos);
        m_2dRep->BuildRepresentation();
        m_2dWidget->Modified();
        updateDistanceDisplay();
    } else if (m_3dWidget && m_3dWidget->GetEnabled()) {
        m_3dRep->GetPoint2WorldPosition(pos);
        newPos[1] = pos[1];
        newPos[2] = pos[2];
        m_3dRep->SetPoint2WorldPosition(newPos);
        m_3dRep->BuildRepresentation();
        m_3dWidget->Modified();
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
    
    if (m_2dWidget && m_2dWidget->GetEnabled()) {
        m_2dRep->GetPoint2WorldPosition(pos);
        newPos[0] = pos[0];
        newPos[2] = pos[2];
        m_2dRep->SetPoint2WorldPosition(newPos);
        m_2dRep->BuildRepresentation();
        m_2dWidget->Modified();
        updateDistanceDisplay();
    } else if (m_3dWidget && m_3dWidget->GetEnabled()) {
        m_3dRep->GetPoint2WorldPosition(pos);
        newPos[0] = pos[0];
        newPos[2] = pos[2];
        m_3dRep->SetPoint2WorldPosition(newPos);
        m_3dRep->BuildRepresentation();
        m_3dWidget->Modified();
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
    
    if (m_2dWidget && m_2dWidget->GetEnabled()) {
        m_2dRep->GetPoint2WorldPosition(pos);
        newPos[0] = pos[0];
        newPos[1] = pos[1];
        m_2dRep->SetPoint2WorldPosition(newPos);
        m_2dRep->BuildRepresentation();
        m_2dWidget->Modified();
        updateDistanceDisplay();
    } else if (m_3dWidget && m_3dWidget->GetEnabled()) {
        m_3dRep->GetPoint2WorldPosition(pos);
        newPos[0] = pos[0];
        newPos[1] = pos[1];
        m_3dRep->SetPoint2WorldPosition(newPos);
        m_3dRep->BuildRepresentation();
        m_3dWidget->Modified();
        updateDistanceDisplay();
    }
    update();
}

void cvDistanceTool::onDistanceChanged(double dist) {
    if (!m_configUi) return;
    VtkUtils::SignalBlocker blocker(m_configUi->distanceSpinBox);
    m_configUi->distanceSpinBox->setValue(dist);
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

void cvDistanceTool::hookWidget(const vtkSmartPointer<vtkDistanceWidget>& widget) {
    VtkUtils::DistanceWidgetObserver* observer = new VtkUtils::DistanceWidgetObserver(this);
    observer->attach(widget);
    connect(observer, &VtkUtils::DistanceWidgetObserver::distanceChanged,
            this, &cvDistanceTool::onDistanceChanged);
    connect(observer, &VtkUtils::DistanceWidgetObserver::worldPoint1Changed,
            this, &cvDistanceTool::onWorldPoint1Changed);
    connect(observer, &VtkUtils::DistanceWidgetObserver::worldPoint2Changed,
            this, &cvDistanceTool::onWorldPoint2Changed);
}

void cvDistanceTool::updateDistanceDisplay() {
    if (!m_configUi) return;
    
    double distance = 0.0;
    if (m_2dWidget && m_2dWidget->GetEnabled()) {
        distance = m_2dRep->GetDistance();
    } else if (m_3dWidget && m_3dWidget->GetEnabled()) {
        distance = m_3dRep->GetDistance();
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
    
    // Update both 2D and 3D representations
    if (m_2dRep) {
        m_2dRep->SetRulerMode(checked ? 1 : 0);
        m_2dRep->BuildRepresentation();
    }
    if (m_3dRep) {
        m_3dRep->SetRulerMode(checked ? 1 : 0);
        m_3dRep->BuildRepresentation();
    }
    
    // Enable/disable appropriate controls
    m_configUi->rulerDistanceSpinBox->setEnabled(checked);
    m_configUi->numberOfTicksSpinBox->setEnabled(!checked);
    
    update();
}

void cvDistanceTool::on_rulerDistanceSpinBox_valueChanged(double value) {
    if (!m_configUi) return;
    
    // Update both 2D and 3D representations
    if (m_2dRep) {
        m_2dRep->SetRulerDistance(value);
        m_2dRep->BuildRepresentation();
    }
    if (m_3dRep) {
        m_3dRep->SetRulerDistance(value);
        m_3dRep->BuildRepresentation();
    }
    
    update();
}

void cvDistanceTool::on_numberOfTicksSpinBox_valueChanged(int value) {
    if (!m_configUi) return;
    
    // Update both 2D and 3D representations
    if (m_2dRep) {
        m_2dRep->SetNumberOfRulerTicks(value);
        m_2dRep->BuildRepresentation();
    }
    if (m_3dRep) {
        m_3dRep->SetNumberOfRulerTicks(value);
        m_3dRep->BuildRepresentation();
    }
    
    update();
}

void cvDistanceTool::on_scaleSpinBox_valueChanged(double value) {
    if (!m_configUi) return;
    
    // Update both 2D and 3D representations
    if (m_2dRep) {
        m_2dRep->SetScale(value);
        m_2dRep->BuildRepresentation();
    }
    if (m_3dRep) {
        m_3dRep->SetScale(value);
        m_3dRep->BuildRepresentation();
    }
    
    // Update distance display to reflect the scaled value
    updateDistanceDisplay();
    update();
}

void cvDistanceTool::on_widgetVisibilityCheckBox_toggled(bool checked) {
    if (!m_configUi) return;
    
    // Show or hide the widget
    if (m_2dWidget) {
        if (checked) {
            m_2dWidget->On();
        } else {
            m_2dWidget->Off();
        }
    }
    if (m_3dWidget) {
        if (checked) {
            m_3dWidget->On();
        } else {
            m_3dWidget->Off();
        }
    }
    
    update();
}

void cvDistanceTool::on_labelVisibilityCheckBox_toggled(bool checked) {
    if (!m_configUi) return;
    
    // Show or hide the distance label
    if (m_2dRep) {
        vtkAxisActor2D* axis = m_2dRep->GetAxis();
        if (axis) {
            axis->SetLabelVisibility(checked);
            // axis->SetTitleVisibility(checked);
        }
        m_2dRep->BuildRepresentation();
    }
    if (m_3dRep) {
        // vtkDistanceRepresentation3D doesn't have GetAxis(), use SetLabelVisibility directly
        // Note: 3D representation may not support label visibility control the same way
        m_3dRep->BuildRepresentation();
    }
    
    update();
}

void cvDistanceTool::on_lineWidthSpinBox_valueChanged(double value) {
    if (!m_configUi) return;
    
    // Update line width for both 2D and 3D representations
    if (m_2dRep) {
        vtkProperty2D* prop = m_2dRep->GetAxisProperty();
        if (prop) {
            prop->SetLineWidth(value);
        }
        m_2dRep->BuildRepresentation();
    }
    if (m_3dRep) {
        // vtkDistanceRepresentation3D uses GetLineProperty() instead of GetAxisProperty()
        vtkProperty* prop = m_3dRep->GetLineProperty();
        if (prop) {
            prop->SetLineWidth(value);
        }
        m_3dRep->BuildRepresentation();
    }
    
    update();
}

void cvDistanceTool::setupPointPickingShortcuts(QWidget* vtkWidget) {
    if (!vtkWidget) return;
    
    // 'P' - Pick alternating points on surface cell
    cvPointPickingHelper* pickHelper = new cvPointPickingHelper(
        QKeySequence(tr("P")), false, vtkWidget);
    pickHelper->setInteractor(m_interactor);
    pickHelper->setRenderer(m_renderer);
    pickHelper->setContextWidget(this);
    connect(pickHelper, &cvPointPickingHelper::pick,
            this, &cvDistanceTool::pickAlternatingPoint);
    m_pickingHelpers.append(pickHelper);

    // 'Ctrl+P' - Pick alternating points, snap to mesh points
    cvPointPickingHelper* pickHelper2 = new cvPointPickingHelper(
        QKeySequence(tr("Ctrl+P")), true, vtkWidget);
    pickHelper2->setInteractor(m_interactor);
    pickHelper2->setRenderer(m_renderer);
    pickHelper2->setContextWidget(this);
    connect(pickHelper2, &cvPointPickingHelper::pick,
            this, &cvDistanceTool::pickAlternatingPoint);
    m_pickingHelpers.append(pickHelper2);

    // '1' - Pick point 1 on surface cell
    cvPointPickingHelper* pickHelper3 = new cvPointPickingHelper(
        QKeySequence(tr("1")), false, vtkWidget);
    pickHelper3->setInteractor(m_interactor);
    pickHelper3->setRenderer(m_renderer);
    pickHelper3->setContextWidget(this);
    connect(pickHelper3, &cvPointPickingHelper::pick,
            this, &cvDistanceTool::pickKeyboardPoint1);
    m_pickingHelpers.append(pickHelper3);

    // 'Ctrl+1' - Pick point 1, snap to mesh points
    cvPointPickingHelper* pickHelper4 = new cvPointPickingHelper(
        QKeySequence(tr("Ctrl+1")), true, vtkWidget);
    pickHelper4->setInteractor(m_interactor);
    pickHelper4->setRenderer(m_renderer);
    pickHelper4->setContextWidget(this);
    connect(pickHelper4, &cvPointPickingHelper::pick,
            this, &cvDistanceTool::pickKeyboardPoint1);
    m_pickingHelpers.append(pickHelper4);

    // '2' - Pick point 2 on surface cell
    cvPointPickingHelper* pickHelper5 = new cvPointPickingHelper(
        QKeySequence(tr("2")), false, vtkWidget);
    pickHelper5->setInteractor(m_interactor);
    pickHelper5->setRenderer(m_renderer);
    pickHelper5->setContextWidget(this);
    connect(pickHelper5, &cvPointPickingHelper::pick,
            this, &cvDistanceTool::pickKeyboardPoint2);
    m_pickingHelpers.append(pickHelper5);

    // 'Ctrl+2' - Pick point 2, snap to mesh points
    cvPointPickingHelper* pickHelper6 = new cvPointPickingHelper(
        QKeySequence(tr("Ctrl+2")), true, vtkWidget);
    pickHelper6->setInteractor(m_interactor);
    pickHelper6->setRenderer(m_renderer);
    pickHelper6->setContextWidget(this);
    connect(pickHelper6, &cvPointPickingHelper::pick,
            this, &cvDistanceTool::pickKeyboardPoint2);
    m_pickingHelpers.append(pickHelper6);

    // 'N' - Pick point and set normal direction
    cvPointPickingHelper* pickHelperNormal = new cvPointPickingHelper(
        QKeySequence(tr("N")), false, vtkWidget,
        cvPointPickingHelper::CoordinatesAndNormal);
    pickHelperNormal->setInteractor(m_interactor);
    pickHelperNormal->setRenderer(m_renderer);
    pickHelperNormal->setContextWidget(this);
    connect(pickHelperNormal, &cvPointPickingHelper::pickNormal,
            this, &cvDistanceTool::pickNormalDirection);
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

void cvDistanceTool::pickNormalDirection(double px, double py, double pz,
                                          double nx, double ny, double nz) {
    // Set point 1 at the picked position
    double p1[3] = {px, py, pz};
    setPoint1(p1);
    
    // Set point 2 along the normal direction
    double p2[3] = {px + nx, py + ny, pz + nz};
    setPoint2(p2);
}

