// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvContourTool.h"

#include <VtkUtils/vtkutils.h>
#include <vtkContourRepresentation.h>
#include <vtkContourWidget.h>
#include <vtkOrientedGlyphContourRepresentation.h>
#include <vtkPolyData.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>

// LOCAL
#include <Utils/vtk2cc.h>

#include "Tools/PickingTools/cvPointPickingHelper.h"
#include "VTKExtensions/ConstrainedWidgets/cvConstrainedContourRepresentation.h"

// QT
#include <QApplication>
#include <QLayout>
#include <QLayoutItem>
#include <QSizePolicy>

// CV_DB_LIB
#include <CVLog.h>
#include <ecvColorTypes.h>
#include <ecvPolyline.h>

// Static counter initialization
int cvContourTool::s_contourIdCounter = 0;

cvContourTool::cvContourTool(QWidget* parent)
    : cvGenericMeasurementTool(parent),
      m_configUi(nullptr),
      m_toolId(++s_contourIdCounter),
      m_exportCounter(0) {
    setWindowTitle(tr("Contour Measurement Tool"));
}

cvContourTool::~cvContourTool() {
    // CRITICAL: Explicitly cleanup all contour widgets and their
    // representations
    for (auto& pair : m_contours) {
        if (pair.second) {
            // Hide representation first
            if (auto* rep = vtkContourRepresentation::SafeDownCast(
                        pair.second->GetRepresentation())) {
                rep->SetVisibility(0);

                // If using custom representation with label, hide it too
                if (auto* customRep =
                            cvConstrainedContourRepresentation::SafeDownCast(
                                    rep)) {
                    if (auto* labelActor = customRep->GetLabelActor()) {
                        labelActor->SetVisibility(0);
                    }
                }
            }

            pair.second->SetInteractor(nullptr);
            pair.second->Off();
        }
    }
    m_contours.clear();

    // Force immediate render to clear visual elements
    if (m_interactor && m_interactor->GetRenderWindow()) {
        m_interactor->GetRenderWindow()->Render();
    }

    if (m_configUi) {
        delete m_configUi;
        m_configUi = nullptr;
    }
}

void cvContourTool::initTool() { createNewContour(); }

void cvContourTool::createNewContour() {
    // If we have a current contour, ensure it's in a state that allows a new
    // one vtkContourWidget doesn't auto-close, but if we create a new one, the
    // old one will just sit there (hopefully in Manipulate state).

    vtkSmartPointer<vtkContourWidget> newContour =
            vtkSmartPointer<vtkContourWidget>::New();
    VtkUtils::vtkInitOnce(newContour);

    // Create and set the custom representation with label support
    vtkSmartPointer<cvConstrainedContourRepresentation> rep =
            vtkSmartPointer<cvConstrainedContourRepresentation>::New();
    newContour->SetRepresentation(rep);

    // Set default node visibility
    rep->SetShowSelectedNodes(true);

    // Set default color
    if (auto* linesProp = rep->GetLinesProperty()) {
        linesProp->SetColor(m_currentColor[0], m_currentColor[1],
                            m_currentColor[2]);
    }
    if (auto* nodesProp = rep->GetProperty()) {
        nodesProp->SetColor(m_currentColor[0], m_currentColor[1],
                            m_currentColor[2]);
    }
    if (auto* activeNodesProp = rep->GetActiveProperty()) {
        activeNodesProp->SetColor(m_currentColor[0], m_currentColor[1],
                                  m_currentColor[2]);
    }

    if (m_interactor) {
        newContour->SetInteractor(m_interactor);
    }
    if (m_renderer) {
        rep->SetRenderer(m_renderer);
        newContour->SetCurrentRenderer(m_renderer);
    }
    newContour->On();

    m_currentContourId++;
    m_contours[m_currentContourId] = newContour;

    // Apply font properties to the newly created contour
    applyFontProperties();
}

void cvContourTool::createUi() {
    // CRITICAL: Only setup base UI once to avoid resetting configLayout
    // Each tool instance has its own m_ui, but setupUi clears all children
    // so we must ensure it's only called once per tool instance
    // Check if base UI is already set up by checking if widget has a layout
    // NOTE: Cannot check m_ui->configLayout directly as it's uninitialized before setupUi()
    if (!m_ui) {
        CVLog::Error("[cvContourTool::createUi] m_ui is null!");
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
    m_configUi = new Ui::ContourToolDlg;
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
    m_ui->groupBox->setTitle(tr("Contour Parameters"));

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

    // Connect display options
    connect(m_configUi->widgetVisibilityCheckBox, &QCheckBox::toggled, this,
            &cvContourTool::on_widgetVisibilityCheckBox_toggled);
    connect(m_configUi->showNodesCheckBox, &QCheckBox::toggled, this,
            &cvContourTool::on_showNodesCheckBox_toggled);
    connect(m_configUi->closedLoopCheckBox, &QCheckBox::toggled, this,
            &cvContourTool::on_closedLoopCheckBox_toggled);
    connect(m_configUi->lineWidthSpinBox,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &cvContourTool::on_lineWidthSpinBox_valueChanged);
}

void cvContourTool::start() { cvGenericMeasurementTool::start(); }

void cvContourTool::reset() {
    // Clear only the current contour
    if (m_contours.find(m_currentContourId) != m_contours.end()) {
        auto widget = m_contours[m_currentContourId];
        if (widget) {
            widget->Initialize(nullptr);
            update();
            emit measurementValueChanged();
        }
    }
}

void cvContourTool::showWidget(bool state) {
    for (auto& pair : m_contours) {
        if (pair.second) {
            if (state) {
                pair.second->On();
            } else {
                pair.second->Off();
            }
        }
    }
    update();
}

void cvContourTool::setColor(double r, double g, double b) {
    // Store current color
    m_currentColor[0] = r;
    m_currentColor[1] = g;
    m_currentColor[2] = b;

    // Set color for all contour widgets
    for (auto& pair : m_contours) {
        if (pair.second) {
            vtkContourRepresentation* rep =
                    vtkContourRepresentation::SafeDownCast(
                            pair.second->GetRepresentation());
            if (rep) {
                vtkOrientedGlyphContourRepresentation* orientedRep =
                        vtkOrientedGlyphContourRepresentation::SafeDownCast(
                                rep);
                if (orientedRep) {
                    if (auto* prop = orientedRep->GetLinesProperty()) {
                        prop->SetColor(r, g, b);
                    }
                    if (auto* activeNodeProp =
                                orientedRep->GetActiveProperty()) {
                        activeNodeProp->SetColor(r, g, b);
                    }
                    rep->BuildRepresentation();
                }
            }
        }
    }
    update();
}

bool cvContourTool::getColor(double& r, double& g, double& b) const {
    r = m_currentColor[0];
    g = m_currentColor[1];
    b = m_currentColor[2];
    return true;
}

void cvContourTool::lockInteraction() {
    CVLog::PrintDebug(QString("[cvContourTool::lockInteraction] Tool=%1, "
                              "m_contours.size()=%2")
                              .arg((quintptr)this, 0, 16)
                              .arg(m_contours.size()));

    // Disable all contour widgets' interaction
    for (auto& pair : m_contours) {
        if (pair.second) {
            pair.second->SetProcessEvents(0);  // Disable event processing
        }
    }

    // Change widget color to indicate locked state (very dimmed, 10%
    // brightness)
    for (auto& pair : m_contours) {
        if (pair.second) {
            vtkContourRepresentation* rep =
                    vtkContourRepresentation::SafeDownCast(
                            pair.second->GetRepresentation());
            if (rep) {
                vtkOrientedGlyphContourRepresentation* orientedRep =
                        vtkOrientedGlyphContourRepresentation::SafeDownCast(
                                rep);
                if (orientedRep) {
                    // Use a very dimmed color to indicate locked state (10%
                    // brightness, 50% opacity)
                    if (auto* prop = orientedRep->GetLinesProperty()) {
                        prop->SetColor(m_currentColor[0] * 0.1,
                                       m_currentColor[1] * 0.1,
                                       m_currentColor[2] * 0.1);
                        prop->SetOpacity(0.5);
                    }
                    if (auto* activeNodeProp =
                                orientedRep->GetActiveProperty()) {
                        activeNodeProp->SetColor(m_currentColor[0] * 0.1,
                                                 m_currentColor[1] * 0.1,
                                                 m_currentColor[2] * 0.1);
                        activeNodeProp->SetOpacity(0.5);
                    }
                    if (auto* nodeProp = orientedRep->GetProperty()) {
                        nodeProp->SetOpacity(0.5);
                    }

                    // Dim the label if using custom representation
                    cvConstrainedContourRepresentation* customRep =
                            cvConstrainedContourRepresentation::SafeDownCast(
                                    orientedRep);
                    if (customRep) {
                        if (auto* labelActor = customRep->GetLabelActor()) {
                            if (auto* textProp =
                                        labelActor->GetTextProperty()) {
                                textProp->SetOpacity(0.5);
                                textProp->SetColor(
                                        0.5, 0.5,
                                        0.5);  // Dark gray for locked state
                            }
                        }
                    }

                    rep->BuildRepresentation();
                }
            }
        }
    }

    // Disable UI controls
    if (m_configUi) {
        m_configUi->widgetVisibilityCheckBox->setEnabled(false);
        m_configUi->showNodesCheckBox->setEnabled(false);
        m_configUi->closedLoopCheckBox->setEnabled(false);
        m_configUi->lineWidthSpinBox->setEnabled(false);
    }

    // Disable keyboard shortcuts
    disableShortcuts();

    // Force render window update
    if (m_interactor && m_interactor->GetRenderWindow()) {
        m_interactor->GetRenderWindow()->Render();
    }

    update();
}

void cvContourTool::unlockInteraction() {
    CVLog::PrintDebug(QString("[cvContourTool::unlockInteraction] Tool=%1, "
                              "m_contours.size()=%2")
                              .arg((quintptr)this, 0, 16)
                              .arg(m_contours.size()));

    // Enable all contour widgets' interaction
    for (auto& pair : m_contours) {
        if (pair.second) {
            pair.second->SetProcessEvents(1);  // Enable event processing
        }
    }

    // Restore widget color to indicate active/unlocked state
    for (auto& pair : m_contours) {
        if (pair.second) {
            vtkContourRepresentation* rep =
                    vtkContourRepresentation::SafeDownCast(
                            pair.second->GetRepresentation());
            if (rep) {
                vtkOrientedGlyphContourRepresentation* orientedRep =
                        vtkOrientedGlyphContourRepresentation::SafeDownCast(
                                rep);
                if (orientedRep) {
                    // Restore original color (full brightness and opacity)
                    if (auto* prop = orientedRep->GetLinesProperty()) {
                        prop->SetColor(m_currentColor[0], m_currentColor[1],
                                       m_currentColor[2]);
                        prop->SetOpacity(1.0);
                    }
                    if (auto* activeNodeProp =
                                orientedRep->GetActiveProperty()) {
                        activeNodeProp->SetColor(m_currentColor[0],
                                                 m_currentColor[1],
                                                 m_currentColor[2]);
                        activeNodeProp->SetOpacity(1.0);
                    }
                    if (auto* nodeProp = orientedRep->GetProperty()) {
                        nodeProp->SetOpacity(1.0);
                    }

                    // Restore the label to user-configured settings
                    cvConstrainedContourRepresentation* customRep =
                            cvConstrainedContourRepresentation::SafeDownCast(
                                    orientedRep);
                    if (customRep) {
                        if (auto* labelActor = customRep->GetLabelActor()) {
                            if (auto* textProp =
                                        labelActor->GetTextProperty()) {
                                textProp->SetOpacity(m_fontOpacity);
                                textProp->SetColor(m_fontColor[0],
                                                   m_fontColor[1],
                                                   m_fontColor[2]);
                            }
                        }
                    }

                    rep->BuildRepresentation();
                }
            }
        }
    }

    // Re-enable keyboard shortcuts
    if (m_pickingHelpers.isEmpty()) {
        // Shortcuts haven't been created yet - create them now
        if (m_vtkWidget) {
            CVLog::PrintDebug(
                    QString("[cvContourTool::unlockInteraction] Creating "
                            "shortcuts for tool=%1, using saved vtkWidget=%2")
                            .arg((quintptr)this, 0, 16)
                            .arg((quintptr)m_vtkWidget, 0, 16));
            setupShortcuts(m_vtkWidget);
            CVLog::PrintDebug(
                    QString("[cvContourTool::unlockInteraction] After "
                            "setupShortcuts, m_pickingHelpers.size()=%1")
                            .arg(m_pickingHelpers.size()));
        } else {
            CVLog::PrintDebug(
                    QString("[cvContourTool::unlockInteraction] m_vtkWidget is "
                            "null for tool=%1, cannot create shortcuts")
                            .arg((quintptr)this, 0, 16));
        }
    } else {
        // Shortcuts already exist - just enable them
        CVLog::PrintDebug(QString("[cvContourTool::unlockInteraction] Enabling "
                                  "%1 existing shortcuts for tool=%2")
                                  .arg(m_pickingHelpers.size())
                                  .arg((quintptr)this, 0, 16));
        for (cvPointPickingHelper* helper : m_pickingHelpers) {
            if (helper) {
                helper->setEnabled(true,
                                   false);  // Enable without setting focus
            }
        }
    }

    // Enable UI controls
    if (m_configUi) {
        m_configUi->widgetVisibilityCheckBox->setEnabled(true);
        m_configUi->showNodesCheckBox->setEnabled(true);
        m_configUi->closedLoopCheckBox->setEnabled(true);
        m_configUi->lineWidthSpinBox->setEnabled(true);
    }

    // Force render window update
    if (m_interactor && m_interactor->GetRenderWindow()) {
        m_interactor->GetRenderWindow()->Render();
    }

    update();
}

void cvContourTool::setInstanceLabel(const QString& label) {
    // Store the instance label
    m_instanceLabel = label;

    // Update window title to show the instance label
    QString title = tr("Contour Measurement Tool");
    if (!m_instanceLabel.isEmpty()) {
        title += QString(" %1").arg(m_instanceLabel);
    }
    setWindowTitle(title);

    // Update VTK representation's label suffix for all contours
    for (auto& pair : m_contours) {
        if (pair.second) {
            vtkContourRepresentation* rep =
                    vtkContourRepresentation::SafeDownCast(
                            pair.second->GetRepresentation());
            if (rep) {
                cvConstrainedContourRepresentation* customRep =
                        cvConstrainedContourRepresentation::SafeDownCast(rep);
                if (customRep) {
                    customRep->SetLabelSuffix(
                            m_instanceLabel.toUtf8().constData());
                    customRep->BuildRepresentation();
                }
            }
        }
    }

    update();
}

ccHObject* cvContourTool::getOutput() {
    // Export only the current contour (as requested)
    auto it = m_contours.find(m_currentContourId);
    if (it == m_contours.end() || !it->second) {
        CVLog::Warning("[cvContourTool] No current contour active");
        return nullptr;
    }

    vtkContourWidget* widget = it->second;

    // Check if contour widget is enabled and has valid data
    if (!widget->GetEnabled()) {
        CVLog::Warning("[cvContourTool] Current contour widget is not enabled");
        return nullptr;
    }

    // Get contour representation
    vtkContourRepresentation* rep =
            vtkContourRepresentation::SafeDownCast(widget->GetRepresentation());
    if (!rep) {
        CVLog::Warning("[cvContourTool] Failed to get contour representation");
        return nullptr;
    }

    // Force update of the representation to ensure polydata is up-to-date
    rep->BuildRepresentation();

    // Get polyData from contour representation
    vtkPolyData* polyData = rep->GetContourRepresentationAsPolyData();
    if (!polyData) {
        CVLog::Warning(
                "[cvContourTool] Failed to get polyData from contour "
                "representation");
        return nullptr;
    }

    // Check if contour has valid points
    if (polyData->GetNumberOfPoints() == 0) {
        CVLog::Warning("[cvContourTool] Contour has no points");
        return nullptr;
    }

    // Get properties from the contour representation
    bool isClosed = rep->GetClosedLoop();
    double lineWidth = 2.0;  // default value

    // Try to get line width from the representation
    vtkOrientedGlyphContourRepresentation* orientedRep =
            vtkOrientedGlyphContourRepresentation::SafeDownCast(rep);
    if (orientedRep) {
        vtkProperty* prop = orientedRep->GetLinesProperty();
        if (prop) {
            lineWidth = prop->GetLineWidth();
        }
    }

    // Create a deep copy of polyData to avoid modifying the original contour
    vtkSmartPointer<vtkPolyData> polyDataCopy =
            vtkSmartPointer<vtkPolyData>::New();
    polyDataCopy->DeepCopy(polyData);

    // Convert vtkPolyData to ccPolyline
    ccPolyline* polyline = vtk2cc::ConvertToPolyline(polyDataCopy, true);
    if (polyline) {
        // Increment export counter for this contour instance
        m_exportCounter++;

        // Set name with contour ID and export counter
        // Use m_toolId to distinguish tool instances, m_currentContourId for
        // contours within tool
        polyline->setName(QString("Contour_%1_%2_%3")
                                  .arg(m_toolId)
                                  .arg(m_currentContourId)
                                  .arg(m_exportCounter));
        polyline->setColor(ecvColor::green);

        // Apply properties from vtkContourWidget to ccPolyline
        polyline->setWidth(static_cast<PointCoordinateType>(lineWidth));
        polyline->setClosed(isClosed);
        polyline->setVisible(true);

        CVLog::Print(QString("[cvContourTool] Exported contour %1 with %2 "
                             "points (closed: %3, width: %4)")
                             .arg(polyline->getName())
                             .arg(polyData->GetNumberOfPoints())
                             .arg(isClosed ? "true" : "false")
                             .arg(lineWidth));

        return polyline;
    } else {
        CVLog::Error("[cvContourTool] Failed to convert polyData to polyline");
        return nullptr;
    }
}

void cvContourTool::onDataChanged(vtkPolyData* pd) {
    // Handle contour data changes
    Q_UNUSED(pd);
    update();
}

void cvContourTool::on_widgetVisibilityCheckBox_toggled(bool checked) {
    if (!m_configUi) return;

    // Show or hide all contour widgets
    for (auto& pair : m_contours) {
        if (pair.second) {
            if (checked) {
                pair.second->On();
            } else {
                pair.second->Off();
            }
        }
    }

    update();
}

void cvContourTool::on_showNodesCheckBox_toggled(bool checked) {
    if (!m_configUi) return;

    // Show or hide nodes for all contours
    for (auto& pair : m_contours) {
        if (pair.second) {
            vtkContourRepresentation* rep =
                    vtkContourRepresentation::SafeDownCast(
                            pair.second->GetRepresentation());
            if (rep) {
                vtkOrientedGlyphContourRepresentation* orientedRep =
                        vtkOrientedGlyphContourRepresentation::SafeDownCast(
                                rep);
                if (orientedRep) {
                    orientedRep->SetShowSelectedNodes(checked);
                }
            }
        }
    }

    update();
}

void cvContourTool::on_closedLoopCheckBox_toggled(bool checked) {
    if (!m_configUi) return;

    // Set closed loop for all contours
    for (auto& pair : m_contours) {
        if (pair.second) {
            vtkContourRepresentation* rep =
                    vtkContourRepresentation::SafeDownCast(
                            pair.second->GetRepresentation());
            if (rep) {
                rep->SetClosedLoop(checked);
                rep->BuildRepresentation();
            }
        }
    }

    update();
}

void cvContourTool::on_lineWidthSpinBox_valueChanged(double value) {
    if (!m_configUi) return;

    // Update line width for all contours
    for (auto& pair : m_contours) {
        if (pair.second) {
            vtkContourRepresentation* rep =
                    vtkContourRepresentation::SafeDownCast(
                            pair.second->GetRepresentation());
            if (rep) {
                // Try to get the property through
                // vtkOrientedGlyphContourRepresentation
                vtkOrientedGlyphContourRepresentation* orientedRep =
                        vtkOrientedGlyphContourRepresentation::SafeDownCast(
                                rep);
                if (orientedRep) {
                    // Use GetLinesProperty() to set the line width for the
                    // contour lines
                    vtkProperty* linesProp = orientedRep->GetLinesProperty();
                    if (linesProp) {
                        linesProp->SetLineWidth(value);
                    }

                    // Also update the property for control points and active
                    // state
                    vtkProperty* prop = orientedRep->GetProperty();
                    if (prop) {
                        prop->SetLineWidth(value);
                    }
                    vtkProperty* activeProp = orientedRep->GetActiveProperty();
                    if (activeProp) {
                        activeProp->SetLineWidth(value);
                    }
                }
                rep->BuildRepresentation();
            }
        }
    }

    update();
}

void cvContourTool::applyFontProperties() {
    // Apply font properties to all contour label actors
    for (auto& pair : m_contours) {
        if (pair.second) {
            vtkContourRepresentation* rep =
                    vtkContourRepresentation::SafeDownCast(
                            pair.second->GetRepresentation());
            if (rep) {
                // Check if using custom representation with label support
                cvConstrainedContourRepresentation* customRep =
                        cvConstrainedContourRepresentation::SafeDownCast(rep);
                if (customRep) {
                    if (auto* labelActor = customRep->GetLabelActor()) {
                        if (auto* textProp = labelActor->GetTextProperty()) {
                            textProp->SetFontFamilyAsString(
                                    m_fontFamily.toUtf8().constData());
                            textProp->SetFontSize(m_fontSize);
                            textProp->SetColor(m_fontColor[0], m_fontColor[1],
                                               m_fontColor[2]);
                            textProp->SetBold(m_fontBold ? 1 : 0);
                            textProp->SetItalic(m_fontItalic ? 1 : 0);
                            textProp->SetShadow(m_fontShadow ? 1 : 0);
                            textProp->SetOpacity(m_fontOpacity);

                            // Apply justification
                            if (m_horizontalJustification == "Left") {
                                textProp->SetJustificationToLeft();
                            } else if (m_horizontalJustification == "Center") {
                                textProp->SetJustificationToCentered();
                            } else if (m_horizontalJustification == "Right") {
                                textProp->SetJustificationToRight();
                            }

                            if (m_verticalJustification == "Top") {
                                textProp->SetVerticalJustificationToTop();
                            } else if (m_verticalJustification == "Center") {
                                textProp->SetVerticalJustificationToCentered();
                            } else if (m_verticalJustification == "Bottom") {
                                textProp->SetVerticalJustificationToBottom();
                            }

                            textProp->Modified();  // Mark as modified to ensure
                                                   // VTK updates
                        }
                        labelActor->Modified();  // Mark actor as modified to
                                                 // trigger re-render
                    }
                }
                rep->BuildRepresentation();
            }
        }
    }

    // Force render window update
    if (m_interactor && m_interactor->GetRenderWindow()) {
        m_interactor->GetRenderWindow()->Render();
    }

    update();
}
