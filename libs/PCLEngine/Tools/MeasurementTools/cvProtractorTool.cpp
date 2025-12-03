// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvProtractorTool.h"

#include "Tools/PickingTools/cvPointPickingHelper.h"

#include <QShortcut>

#include <VtkUtils/anglewidgetobserver.h>
#include <VtkUtils/signalblocker.h>
#include <VtkUtils/vtkutils.h>
#include <vtkAngleRepresentation2D.h>
#include <vtkAngleRepresentation3D.h>
#include <vtkAngleWidget.h>
#include <vtkCommand.h>
#include <vtkFollower.h>
#include <vtkHandleRepresentation.h>
#include <vtkMath.h>
#include <vtkPointHandleRepresentation2D.h>
#include <vtkPointHandleRepresentation3D.h>
#include <vtkProperty.h>
#include <vtkProperty2D.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>

#include <algorithm>
#include <cmath>

// ECV_DB_LIB
#include <ecvBBox.h>
#include <ecvHObject.h>

namespace {

// ParaView-style colors
constexpr double FOREGROUND_COLOR[3] = {1.0, 1.0, 1.0};  // White for normal state
constexpr double INTERACTION_COLOR[3] = {0.0, 1.0, 0.0}; // Green for selected/interactive state
constexpr double RAY_COLOR[3] = {1.0, 0.0, 0.0};         // Red for angle rays (ParaView default)
constexpr double ARC_COLOR[3] = {1.0, 0.1, 0.0};         // Orange-red for arc and text (ParaView default)

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

//! Configure angle representation 2D with ParaView-style properties
void configureAngleRep2D(vtkAngleRepresentation2D* rep) {
    if (!rep) return;
    
    // Configure handles
    auto* h1 = vtkPointHandleRepresentation2D::SafeDownCast(rep->GetPoint1Representation());
    auto* hc = vtkPointHandleRepresentation2D::SafeDownCast(rep->GetCenterRepresentation());
    auto* h2 = vtkPointHandleRepresentation2D::SafeDownCast(rep->GetPoint2Representation());
    configureHandle2D(h1);
    configureHandle2D(hc);
    configureHandle2D(h2);
    
    // 2D representation uses vtkLeaderActor2D for rays and arc
    // The default colors are already reasonable for 2D
}

//! Configure angle representation 3D with ParaView-style properties
void configureAngleRep3D(vtkAngleRepresentation3D* rep) {
    if (!rep) return;
    
    // Configure handles
    auto* h1 = vtkPointHandleRepresentation3D::SafeDownCast(rep->GetPoint1Representation());
    auto* hc = vtkPointHandleRepresentation3D::SafeDownCast(rep->GetCenterRepresentation());
    auto* h2 = vtkPointHandleRepresentation3D::SafeDownCast(rep->GetPoint2Representation());
    configureHandle3D(h1);
    configureHandle3D(hc);
    configureHandle3D(h2);
    
    // Configure ray actors (matching ParaView's red color)
    if (auto* ray1 = rep->GetRay1()) {
        if (auto* prop = ray1->GetProperty()) {
            prop->SetColor(RAY_COLOR[0], RAY_COLOR[1], RAY_COLOR[2]);
            prop->SetLineWidth(2.0);
        }
    }
    if (auto* ray2 = rep->GetRay2()) {
        if (auto* prop = ray2->GetProperty()) {
            prop->SetColor(RAY_COLOR[0], RAY_COLOR[1], RAY_COLOR[2]);
            prop->SetLineWidth(2.0);
        }
    }
    
    // Configure arc actor (matching ParaView's orange-red color)
    if (auto* arc = rep->GetArc()) {
        if (auto* prop = arc->GetProperty()) {
            prop->SetColor(ARC_COLOR[0], ARC_COLOR[1], ARC_COLOR[2]);
            prop->SetLineWidth(2.0);
        }
    }
    
    // Configure text actor (matching ParaView's orange-red color)
    if (auto* textActor = rep->GetTextActor()) {
        if (auto* prop = textActor->GetProperty()) {
            prop->SetColor(ARC_COLOR[0], ARC_COLOR[1], ARC_COLOR[2]);
        }
    }
}

} // anonymous namespace

cvProtractorTool::cvProtractorTool(QWidget* parent)
    : cvGenericMeasurementTool(parent), m_configUi(nullptr) {
    setWindowTitle(tr("Protractor Measurement Tool"));
}

cvProtractorTool::~cvProtractorTool() {
    if (m_configUi) {
        delete m_configUi;
        m_configUi = nullptr;
    }
}

void cvProtractorTool::initTool() {
    // Initialize only 3D widget as default (matching UI currentIndex=1)
    // 2D widget will be lazily initialized when user switches to it
    
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
    
    // Following ParaView's approach:
    // 1. InstantiateHandleRepresentation (done in CreateDefaultRepresentation)
    m_3dRep->InstantiateHandleRepresentation();
    
    // 2. Configure appearance AFTER instantiation but BEFORE enabling
    configureAngleRep3D(m_3dRep);
    
    // 3. Build representation
    m_3dRep->BuildRepresentation();
    
    // 4. Enable widget - this will internally handle the handle widgets
    m_3dWidget->On();
    
    hookWidget(m_3dWidget);  // Hook observer for 3D widget
}

void cvProtractorTool::createUi() {
    m_configUi = new Ui::ProtractorToolDlg;
    QWidget* configWidget = new QWidget(this);
    m_configUi->setupUi(configWidget);
    m_ui->setupUi(this);
    m_ui->configLayout->addWidget(configWidget);
    m_ui->groupBox->setTitle(tr("Protractor Parameters"));

#ifdef Q_OS_MAC
    m_configUi->instructionLabel->setText(
        m_configUi->instructionLabel->text().replace("Ctrl", "Cmd"));
#endif

    connect(m_configUi->typeCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &cvProtractorTool::on_typeCombo_currentIndexChanged);
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
    connect(m_configUi->ray1VisibilityCheckBox, &QCheckBox::toggled, this,
            &cvProtractorTool::on_ray1VisibilityCheckBox_toggled);
    connect(m_configUi->ray2VisibilityCheckBox, &QCheckBox::toggled, this,
            &cvProtractorTool::on_ray2VisibilityCheckBox_toggled);
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
    // Reset points to default positions (center of bounding box if available)
    double defaultCenter[3] = {0.0, 0.0, 0.0};
    double defaultPos1[3] = {-0.5, 0.0, 0.0};
    double defaultPos2[3] = {0.5, 0.0, 0.0};

    if (m_entity && m_entity->getBB_recursive().isValid()) {
        const ccBBox& bbox = m_entity->getBB_recursive();
        CCVector3 center = bbox.getCenter();
        CCVector3 diag = bbox.getDiagVec();

        defaultCenter[0] = center.x;
        defaultCenter[1] = center.y;
        defaultCenter[2] = center.z;

        defaultPos1[0] = center.x - diag.x * 0.25;
        defaultPos1[1] = center.y;
        defaultPos1[2] = center.z;

        defaultPos2[0] = center.x + diag.x * 0.25;
        defaultPos2[1] = center.y;
        defaultPos2[2] = center.z;
    }

    // Reset widget points
    if (m_2dWidget && m_2dWidget->GetEnabled() && m_renderer) {
        double displayCenter[3], displayPos1[3], displayPos2[3];

        m_renderer->SetWorldPoint(defaultCenter[0], defaultCenter[1],
                                  defaultCenter[2], 1.0);
        m_renderer->WorldToDisplay();
        const double* display = m_renderer->GetDisplayPoint();
        displayCenter[0] = display[0];
        displayCenter[1] = display[1];
        displayCenter[2] = display[2];

        m_renderer->SetWorldPoint(defaultPos1[0], defaultPos1[1],
                                  defaultPos1[2], 1.0);
        m_renderer->WorldToDisplay();
        display = m_renderer->GetDisplayPoint();
        displayPos1[0] = display[0];
        displayPos1[1] = display[1];
        displayPos1[2] = display[2];

        m_renderer->SetWorldPoint(defaultPos2[0], defaultPos2[1],
                                  defaultPos2[2], 1.0);
        m_renderer->WorldToDisplay();
        display = m_renderer->GetDisplayPoint();
        displayPos2[0] = display[0];
        displayPos2[1] = display[1];
        displayPos2[2] = display[2];

        m_2dRep->SetCenterDisplayPosition(displayCenter);
        m_2dRep->SetPoint1DisplayPosition(displayPos1);
        m_2dRep->SetPoint2DisplayPosition(displayPos2);
        m_2dRep->BuildRepresentation();
        m_2dWidget->Modified();
    } else if (m_3dWidget && m_3dWidget->GetEnabled()) {
        m_3dRep->SetCenterWorldPosition(defaultCenter);
        m_3dRep->SetPoint1WorldPosition(defaultPos1);
        m_3dRep->SetPoint2WorldPosition(defaultPos2);
        m_3dRep->BuildRepresentation();
        m_3dWidget->Modified();
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

void cvProtractorTool::showWidget(bool state) {
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

ccHObject* cvProtractorTool::getOutput() { return nullptr; }

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

    // Update widget directly
    if (m_2dWidget && m_2dWidget->GetEnabled()) {
        // For 2D representation, convert world to display coordinates
        if (m_renderer) {
            double displayPos[3];
            m_renderer->SetWorldPoint(pos[0], pos[1], pos[2], 1.0);
            m_renderer->WorldToDisplay();
            const double* display = m_renderer->GetDisplayPoint();
            displayPos[0] = display[0];
            displayPos[1] = display[1];
            displayPos[2] = display[2];
            m_2dRep->SetPoint1DisplayPosition(displayPos);
            m_2dRep->BuildRepresentation();
            m_2dWidget->Modified();
            m_2dWidget->Render();
        }
    } else if (m_3dWidget && m_3dWidget->GetEnabled()) {
        m_3dRep->SetPoint1WorldPosition(pos);
        m_3dRep->BuildRepresentation();
        m_3dWidget->Modified();
        m_3dWidget->Render();  // Ensure render
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

    // Update widget directly
    if (m_2dWidget && m_2dWidget->GetEnabled()) {
        // For 2D representation, convert world to display coordinates
        if (m_renderer) {
            double displayPos[3];
            m_renderer->SetWorldPoint(pos[0], pos[1], pos[2], 1.0);
            m_renderer->WorldToDisplay();
            const double* display = m_renderer->GetDisplayPoint();
            displayPos[0] = display[0];
            displayPos[1] = display[1];
            displayPos[2] = display[2];
            m_2dRep->SetPoint2DisplayPosition(displayPos);
            m_2dRep->BuildRepresentation();
            m_2dWidget->Modified();
            m_2dWidget->Render();
        }
    } else if (m_3dWidget && m_3dWidget->GetEnabled()) {
        m_3dRep->SetPoint2WorldPosition(pos);
        m_3dRep->BuildRepresentation();
        m_3dWidget->Modified();
        m_3dWidget->Render();  // Ensure render
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

    // Update widget directly
    if (m_2dWidget && m_2dWidget->GetEnabled()) {
        // For 2D representation, convert world to display coordinates
        if (m_renderer) {
            double displayPos[3];
            m_renderer->SetWorldPoint(pos[0], pos[1], pos[2], 1.0);
            m_renderer->WorldToDisplay();
            const double* display = m_renderer->GetDisplayPoint();
            displayPos[0] = display[0];
            displayPos[1] = display[1];
            displayPos[2] = display[2];
            m_2dRep->SetCenterDisplayPosition(displayPos);
            m_2dRep->BuildRepresentation();
            m_2dWidget->Modified();
            m_2dWidget->Render();
        }
    } else if (m_3dWidget && m_3dWidget->GetEnabled()) {
        m_3dRep->SetCenterWorldPosition(pos);
        m_3dRep->BuildRepresentation();
        m_3dWidget->Modified();
        m_3dWidget->Render();  // Ensure render
    }

    // Update angle display
    updateAngleDisplay();
    update();
}

void cvProtractorTool::on_typeCombo_currentIndexChanged(int index) {
    if (index == 0) {
        // Switch to 2D widget
        if (!m_2dWidget) {
            // Lazy initialization of 2D widget
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
            
            // Following ParaView's approach:
            // 1. InstantiateHandleRepresentation
            m_2dRep->InstantiateHandleRepresentation();
            
            // 2. Configure appearance AFTER instantiation but BEFORE enabling
            configureAngleRep2D(m_2dRep);
            
            // 3. Build representation
            m_2dRep->BuildRepresentation();
            
            hookWidget(m_2dWidget);  // Hook observer for 2D widget
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
            // Lazy initialization of 3D widget (shouldn't happen as it's default)
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
            
            // Following ParaView's approach:
            // 1. InstantiateHandleRepresentation
            m_3dRep->InstantiateHandleRepresentation();
            
            // 2. Configure appearance AFTER instantiation but BEFORE enabling
            configureAngleRep3D(m_3dRep);
            
            // 3. Build representation
            m_3dRep->BuildRepresentation();
            
            hookWidget(m_3dWidget);  // Hook observer for 3D widget
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

void cvProtractorTool::on_point1XSpinBox_valueChanged(double arg1) {
    if (!m_configUi) return;
    double pos[3];
    double newPos[3];
    newPos[0] = arg1;

    VtkUtils::SignalBlocker blocker(m_configUi->point1XSpinBox);

    if (m_2dWidget && m_2dWidget->GetEnabled()) {
        m_2dRep->GetPoint1WorldPosition(pos);
        newPos[1] = pos[1];
        newPos[2] = pos[2];
        // For 2D representation, convert world to display coordinates
        if (m_renderer) {
            double displayPos[3];
            m_renderer->SetWorldPoint(newPos[0], newPos[1], newPos[2], 1.0);
            m_renderer->WorldToDisplay();
            const double* display = m_renderer->GetDisplayPoint();
            displayPos[0] = display[0];
            displayPos[1] = display[1];
            displayPos[2] = display[2];
            m_2dRep->SetPoint1DisplayPosition(displayPos);
            m_2dRep->BuildRepresentation();
            m_2dWidget->Modified();
        }
    } else if (m_3dWidget && m_3dWidget->GetEnabled()) {
        m_3dRep->GetPoint1WorldPosition(pos);
        newPos[1] = pos[1];
        newPos[2] = pos[2];
        m_3dRep->SetPoint1WorldPosition(newPos);
        m_3dRep->BuildRepresentation();
        m_3dWidget->Modified();
        m_3dWidget->Render();
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

    if (m_2dWidget && m_2dWidget->GetEnabled()) {
        m_2dRep->GetPoint1WorldPosition(pos);
        newPos[0] = pos[0];
        newPos[2] = pos[2];
        // For 2D representation, convert world to display coordinates
        if (m_renderer) {
            double displayPos[3];
            m_renderer->SetWorldPoint(newPos[0], newPos[1], newPos[2], 1.0);
            m_renderer->WorldToDisplay();
            const double* display = m_renderer->GetDisplayPoint();
            displayPos[0] = display[0];
            displayPos[1] = display[1];
            displayPos[2] = display[2];
            m_2dRep->SetPoint1DisplayPosition(displayPos);
            m_2dRep->BuildRepresentation();
            m_2dWidget->Modified();
        }
    } else if (m_3dWidget && m_3dWidget->GetEnabled()) {
        m_3dRep->GetPoint1WorldPosition(pos);
        newPos[0] = pos[0];
        newPos[2] = pos[2];
        m_3dRep->SetPoint1WorldPosition(newPos);
        m_3dRep->BuildRepresentation();
        m_3dWidget->Modified();
        m_3dWidget->Render();
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

    if (m_2dWidget && m_2dWidget->GetEnabled()) {
        m_2dRep->GetPoint1WorldPosition(pos);
        newPos[0] = pos[0];
        newPos[1] = pos[1];
        // For 2D representation, convert world to display coordinates
        if (m_renderer) {
            double displayPos[3];
            m_renderer->SetWorldPoint(newPos[0], newPos[1], newPos[2], 1.0);
            m_renderer->WorldToDisplay();
            const double* display = m_renderer->GetDisplayPoint();
            displayPos[0] = display[0];
            displayPos[1] = display[1];
            displayPos[2] = display[2];
            m_2dRep->SetPoint1DisplayPosition(displayPos);
            m_2dRep->BuildRepresentation();
            m_2dWidget->Modified();
        }
    } else if (m_3dWidget && m_3dWidget->GetEnabled()) {
        m_3dRep->GetPoint1WorldPosition(pos);
        newPos[0] = pos[0];
        newPos[1] = pos[1];
        m_3dRep->SetPoint1WorldPosition(newPos);
        m_3dRep->BuildRepresentation();
        m_3dWidget->Modified();
        m_3dWidget->Render();
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

    if (m_2dWidget && m_2dWidget->GetEnabled()) {
        m_2dRep->GetCenterWorldPosition(pos);
        newPos[1] = pos[1];
        newPos[2] = pos[2];
        // For 2D representation, convert world to display coordinates
        if (m_renderer) {
            double displayPos[3];
            m_renderer->SetWorldPoint(newPos[0], newPos[1], newPos[2], 1.0);
            m_renderer->WorldToDisplay();
            const double* display = m_renderer->GetDisplayPoint();
            displayPos[0] = display[0];
            displayPos[1] = display[1];
            displayPos[2] = display[2];
            m_2dRep->SetCenterDisplayPosition(displayPos);
            m_2dRep->BuildRepresentation();
            m_2dWidget->Modified();
        }
    } else if (m_3dWidget && m_3dWidget->GetEnabled()) {
        m_3dRep->GetCenterWorldPosition(pos);
        newPos[1] = pos[1];
        newPos[2] = pos[2];
        m_3dRep->SetCenterWorldPosition(newPos);
        m_3dRep->BuildRepresentation();
        m_3dWidget->Modified();
        m_3dWidget->Render();
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

    if (m_2dWidget && m_2dWidget->GetEnabled()) {
        m_2dRep->GetCenterWorldPosition(pos);
        newPos[0] = pos[0];
        newPos[2] = pos[2];
        // For 2D representation, convert world to display coordinates
        if (m_renderer) {
            double displayPos[3];
            m_renderer->SetWorldPoint(newPos[0], newPos[1], newPos[2], 1.0);
            m_renderer->WorldToDisplay();
            const double* display = m_renderer->GetDisplayPoint();
            displayPos[0] = display[0];
            displayPos[1] = display[1];
            displayPos[2] = display[2];
            m_2dRep->SetCenterDisplayPosition(displayPos);
            m_2dRep->BuildRepresentation();
            m_2dWidget->Modified();
        }
    } else if (m_3dWidget && m_3dWidget->GetEnabled()) {
        m_3dRep->GetCenterWorldPosition(pos);
        newPos[0] = pos[0];
        newPos[2] = pos[2];
        m_3dRep->SetCenterWorldPosition(newPos);
        m_3dRep->BuildRepresentation();
        m_3dWidget->Modified();
        m_3dWidget->Render();
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

    if (m_2dWidget && m_2dWidget->GetEnabled()) {
        m_2dRep->GetCenterWorldPosition(pos);
        newPos[0] = pos[0];
        newPos[1] = pos[1];
        // For 2D representation, convert world to display coordinates
        if (m_renderer) {
            double displayPos[3];
            m_renderer->SetWorldPoint(newPos[0], newPos[1], newPos[2], 1.0);
            m_renderer->WorldToDisplay();
            const double* display = m_renderer->GetDisplayPoint();
            displayPos[0] = display[0];
            displayPos[1] = display[1];
            displayPos[2] = display[2];
            m_2dRep->SetCenterDisplayPosition(displayPos);
            m_2dRep->BuildRepresentation();
            m_2dWidget->Modified();
        }
    } else if (m_3dWidget && m_3dWidget->GetEnabled()) {
        m_3dRep->GetCenterWorldPosition(pos);
        newPos[0] = pos[0];
        newPos[1] = pos[1];
        m_3dRep->SetCenterWorldPosition(newPos);
        m_3dRep->BuildRepresentation();
        m_3dWidget->Modified();
        m_3dWidget->Render();
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

    if (m_2dWidget && m_2dWidget->GetEnabled()) {
        m_2dRep->GetPoint2WorldPosition(pos);
        newPos[1] = pos[1];
        newPos[2] = pos[2];
        // For 2D representation, convert world to display coordinates
        if (m_renderer) {
            double displayPos[3];
            m_renderer->SetWorldPoint(newPos[0], newPos[1], newPos[2], 1.0);
            m_renderer->WorldToDisplay();
            const double* display = m_renderer->GetDisplayPoint();
            displayPos[0] = display[0];
            displayPos[1] = display[1];
            displayPos[2] = display[2];
            m_2dRep->SetPoint2DisplayPosition(displayPos);
            m_2dRep->BuildRepresentation();
            m_2dWidget->Modified();
        }
    } else if (m_3dWidget && m_3dWidget->GetEnabled()) {
        m_3dRep->GetPoint2WorldPosition(pos);
        newPos[1] = pos[1];
        newPos[2] = pos[2];
        m_3dRep->SetPoint2WorldPosition(newPos);
        m_3dRep->BuildRepresentation();
        m_3dWidget->Modified();
        m_3dWidget->Render();
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

    if (m_2dWidget && m_2dWidget->GetEnabled()) {
        m_2dRep->GetPoint2WorldPosition(pos);
        newPos[0] = pos[0];
        newPos[2] = pos[2];
        // For 2D representation, convert world to display coordinates
        if (m_renderer) {
            double displayPos[3];
            m_renderer->SetWorldPoint(newPos[0], newPos[1], newPos[2], 1.0);
            m_renderer->WorldToDisplay();
            const double* display = m_renderer->GetDisplayPoint();
            displayPos[0] = display[0];
            displayPos[1] = display[1];
            displayPos[2] = display[2];
            m_2dRep->SetPoint2DisplayPosition(displayPos);
            m_2dRep->BuildRepresentation();
            m_2dWidget->Modified();
        }
    } else if (m_3dWidget && m_3dWidget->GetEnabled()) {
        m_3dRep->GetPoint2WorldPosition(pos);
        newPos[0] = pos[0];
        newPos[2] = pos[2];
        m_3dRep->SetPoint2WorldPosition(newPos);
        m_3dRep->BuildRepresentation();
        m_3dWidget->Modified();
        m_3dWidget->Render();
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

    if (m_2dWidget && m_2dWidget->GetEnabled()) {
        m_2dRep->GetPoint2WorldPosition(pos);
        newPos[0] = pos[0];
        newPos[1] = pos[1];
        // For 2D representation, convert world to display coordinates
        if (m_renderer) {
            double displayPos[3];
            m_renderer->SetWorldPoint(newPos[0], newPos[1], newPos[2], 1.0);
            m_renderer->WorldToDisplay();
            const double* display = m_renderer->GetDisplayPoint();
            displayPos[0] = display[0];
            displayPos[1] = display[1];
            displayPos[2] = display[2];
            m_2dRep->SetPoint2DisplayPosition(displayPos);
            m_2dRep->BuildRepresentation();
            m_2dWidget->Modified();
        }
    } else if (m_3dWidget && m_3dWidget->GetEnabled()) {
        m_3dRep->GetPoint2WorldPosition(pos);
        newPos[0] = pos[0];
        newPos[1] = pos[1];
        m_3dRep->SetPoint2WorldPosition(newPos);
        m_3dRep->BuildRepresentation();
        m_3dWidget->Modified();
        m_3dWidget->Render();
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
        const vtkSmartPointer<vtkAngleWidget>& widget) {
    VtkUtils::AngleWidgetObserver* observer =
            new VtkUtils::AngleWidgetObserver(this);
    observer->attach(widget);
    connect(observer, &VtkUtils::AngleWidgetObserver::angleChanged, this,
            &cvProtractorTool::onAngleChanged);
    connect(observer, &VtkUtils::AngleWidgetObserver::worldPoint1Changed, this,
            &cvProtractorTool::onWorldPoint1Changed);
    connect(observer, &VtkUtils::AngleWidgetObserver::worldPoint2Changed, this,
            &cvProtractorTool::onWorldPoint2Changed);
    connect(observer, &VtkUtils::AngleWidgetObserver::worldCenterChanged, this,
            &cvProtractorTool::onWorldCenterChanged);
}

void cvProtractorTool::updateAngleDisplay() {
    if (!m_configUi) {
        CVLog::Warning("[cvProtractorTool] updateAngleDisplay: m_configUi is null!");
        return;
    }
    
    // IMPORTANT: vtkAngleRepresentation2D::GetAngle() returns DEGREES
    //            vtkAngleRepresentation3D::GetAngle() returns RADIANS
    double angleDegrees = 0.0;
    bool is2D = m_2dWidget && m_2dWidget->GetEnabled();
    bool is3D = m_3dWidget && m_3dWidget->GetEnabled();
    
    if (is2D && m_2dRep) {
        // 2D representation returns degrees directly
        angleDegrees = m_2dRep->GetAngle();
    } else if (is3D && m_3dRep) {
        // 3D representation returns radians, convert to degrees
        double angleRadians = m_3dRep->GetAngle();
        angleDegrees = vtkMath::DegreesFromRadians(angleRadians);
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

void cvProtractorTool::on_ray1VisibilityCheckBox_toggled(bool checked) {
    if (!m_configUi) return;

    // Show or hide ray 1
    if (m_2dRep) {
        m_2dRep->SetRay1Visibility(checked);
        m_2dRep->BuildRepresentation();
    }
    if (m_3dRep) {
        m_3dRep->SetRay1Visibility(checked);
        m_3dRep->BuildRepresentation();
    }

    update();
}

void cvProtractorTool::on_ray2VisibilityCheckBox_toggled(bool checked) {
    if (!m_configUi) return;

    // Show or hide ray 2
    if (m_2dRep) {
        m_2dRep->SetRay2Visibility(checked);
        m_2dRep->BuildRepresentation();
    }
    if (m_3dRep) {
        m_3dRep->SetRay2Visibility(checked);
        m_3dRep->BuildRepresentation();
    }

    update();
}

void cvProtractorTool::on_arcVisibilityCheckBox_toggled(bool checked) {
    if (!m_configUi) return;

    // Show or hide the arc
    if (m_2dRep) {
        m_2dRep->SetArcVisibility(checked);
        m_2dRep->BuildRepresentation();
    }
    if (m_3dRep) {
        m_3dRep->SetArcVisibility(checked);
        m_3dRep->BuildRepresentation();
    }

    update();
}

void cvProtractorTool::setupPointPickingShortcuts(QWidget* vtkWidget) {
    if (!vtkWidget) return;
    
    // '1' - Pick point 1 on surface
    cvPointPickingHelper* pickHelper1 = new cvPointPickingHelper(
        QKeySequence(tr("1")), false, vtkWidget);
    pickHelper1->setInteractor(m_interactor);
    pickHelper1->setRenderer(m_renderer);
    pickHelper1->setContextWidget(this);
    connect(pickHelper1, &cvPointPickingHelper::pick,
            this, &cvProtractorTool::pickKeyboardPoint1);
    m_pickingHelpers.append(pickHelper1);

    // 'Ctrl+1' - Pick point 1, snap to mesh points
    cvPointPickingHelper* pickHelper1Snap = new cvPointPickingHelper(
        QKeySequence(tr("Ctrl+1")), true, vtkWidget);
    pickHelper1Snap->setInteractor(m_interactor);
    pickHelper1Snap->setRenderer(m_renderer);
    pickHelper1Snap->setContextWidget(this);
    connect(pickHelper1Snap, &cvPointPickingHelper::pick,
            this, &cvProtractorTool::pickKeyboardPoint1);
    m_pickingHelpers.append(pickHelper1Snap);

    // 'C' - Pick center on surface
    cvPointPickingHelper* pickHelperCenter = new cvPointPickingHelper(
        QKeySequence(tr("C")), false, vtkWidget);
    pickHelperCenter->setInteractor(m_interactor);
    pickHelperCenter->setRenderer(m_renderer);
    pickHelperCenter->setContextWidget(this);
    connect(pickHelperCenter, &cvPointPickingHelper::pick,
            this, &cvProtractorTool::pickKeyboardCenter);
    m_pickingHelpers.append(pickHelperCenter);

    // 'Ctrl+C' - Pick center, snap to mesh points
    cvPointPickingHelper* pickHelperCenterSnap = new cvPointPickingHelper(
        QKeySequence(tr("Ctrl+C")), true, vtkWidget);
    pickHelperCenterSnap->setInteractor(m_interactor);
    pickHelperCenterSnap->setRenderer(m_renderer);
    pickHelperCenterSnap->setContextWidget(this);
    connect(pickHelperCenterSnap, &cvPointPickingHelper::pick,
            this, &cvProtractorTool::pickKeyboardCenter);
    m_pickingHelpers.append(pickHelperCenterSnap);

    // '2' - Pick point 2 on surface
    cvPointPickingHelper* pickHelper2 = new cvPointPickingHelper(
        QKeySequence(tr("2")), false, vtkWidget);
    pickHelper2->setInteractor(m_interactor);
    pickHelper2->setRenderer(m_renderer);
    pickHelper2->setContextWidget(this);
    connect(pickHelper2, &cvPointPickingHelper::pick,
            this, &cvProtractorTool::pickKeyboardPoint2);
    m_pickingHelpers.append(pickHelper2);

    // 'Ctrl+2' - Pick point 2, snap to mesh points
    cvPointPickingHelper* pickHelper2Snap = new cvPointPickingHelper(
        QKeySequence(tr("Ctrl+2")), true, vtkWidget);
    pickHelper2Snap->setInteractor(m_interactor);
    pickHelper2Snap->setRenderer(m_renderer);
    pickHelper2Snap->setContextWidget(this);
    connect(pickHelper2Snap, &cvPointPickingHelper::pick,
            this, &cvProtractorTool::pickKeyboardPoint2);
    m_pickingHelpers.append(pickHelper2Snap);
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

