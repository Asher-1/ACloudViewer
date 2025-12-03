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
#include <vtkRenderer.h>

// LOCAL
#include "PclUtils/vtk2cc.h"

// ECV_DB_LIB
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
    for (auto& pair : m_contours) {
        if (pair.second) {
            pair.second->SetInteractor(nullptr);
            pair.second->Off();
        }
    }
    m_contours.clear();

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

    // Create and set the representation
    vtkSmartPointer<vtkOrientedGlyphContourRepresentation> rep =
            vtkSmartPointer<vtkOrientedGlyphContourRepresentation>::New();
    newContour->SetRepresentation(rep);

    // Set default node visibility
    rep->SetShowSelectedNodes(true);

    if (m_interactor) {
        newContour->SetInteractor(m_interactor);
    }
    if (m_renderer) {
        newContour->SetCurrentRenderer(m_renderer);
    }
    newContour->On();

    m_currentContourId++;
    m_contours[m_currentContourId] = newContour;
}

void cvContourTool::createUi() {
    m_configUi = new Ui::ContourToolDlg;
    QWidget* configWidget = new QWidget(this);
    m_configUi->setupUi(configWidget);
    m_ui->setupUi(this);
    m_ui->configLayout->addWidget(configWidget);
    m_ui->groupBox->setTitle(tr("Contour Parameters"));

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

void cvContourTool::start() {
    cvGenericMeasurementTool::start();
}

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
