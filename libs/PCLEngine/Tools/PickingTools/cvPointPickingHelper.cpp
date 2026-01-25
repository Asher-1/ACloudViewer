// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvPointPickingHelper.h"

// CV_CORE_LIB
#include <CVLog.h>

// VTK
#include <vtkActor.h>
#include <vtkCell.h>
#include <vtkCellData.h>
#include <vtkCellPicker.h>
#include <vtkDataArray.h>
#include <vtkDataSet.h>
#include <vtkHardwareSelector.h>
#include <vtkIdTypeArray.h>
#include <vtkInformation.h>
#include <vtkMapper.h>
#include <vtkPointData.h>
#include <vtkPointPicker.h>
#include <vtkPoints.h>
#include <vtkProp.h>
#include <vtkProp3D.h>
#include <vtkPropPicker.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSelection.h>
#include <vtkSelectionNode.h>

// QT
#include <QApplication>
#include <QCursor>
#include <QShortcut>
#include <QWidget>
#include <cmath>

//-----------------------------------------------------------------------------
cvPointPickingHelper::cvPointPickingHelper(const QKeySequence& keySequence,
                                           bool pickOnPoint,
                                           QWidget* parent,
                                           PickOption pickOpt)
    : QObject(parent), m_pickOnPoint(pickOnPoint), m_pickOption(pickOpt) {
    if (!parent) {
        CVLog::Warning(
                "[cvPointPickingHelper] Parent widget is null, shortcut may "
                "not work properly");
        return;
    }

    // Use ApplicationShortcut context for global shortcuts
    // We rely on m_contextWidget visibility check in pickPoint() to ensure
    // only the active tool responds to the shortcut
    m_shortcut = new QShortcut(keySequence, parent);
    m_shortcut->setContext(Qt::ApplicationShortcut);
    connect(m_shortcut, &QShortcut::activated, this,
            &cvPointPickingHelper::pickPoint);

    // Set focus policy so the widget can receive focus when needed
    parent->setFocusPolicy(Qt::StrongFocus);
}

//-----------------------------------------------------------------------------
cvPointPickingHelper::~cvPointPickingHelper() {
    if (m_shortcut) {
        delete m_shortcut;
    }
}

//-----------------------------------------------------------------------------
void cvPointPickingHelper::setInteractor(
        vtkRenderWindowInteractor* interactor) {
    m_interactor = interactor;
}

//-----------------------------------------------------------------------------
void cvPointPickingHelper::setRenderer(vtkRenderer* renderer) {
    m_renderer = renderer;
}

//-----------------------------------------------------------------------------
void cvPointPickingHelper::setContextWidget(QWidget* widget) {
    m_contextWidget = widget;
    // Note: We don't change the shortcut's parent here because it should
    // remain bound to the VTK render window widget for proper focus handling
}

//-----------------------------------------------------------------------------
void cvPointPickingHelper::setEnabled(bool enabled, bool setFocus) {
    if (m_shortcut) {
        m_shortcut->setEnabled(enabled);

        // If enabling and setFocus is true, give focus to the shortcut's parent
        // widget This is similar to ParaView's pqModalShortcut::setEnabled
        // approach
        if (enabled && setFocus) {
            QWidget* parent = qobject_cast<QWidget*>(m_shortcut->parent());
            if (parent) {
                parent->setFocus(Qt::OtherFocusReason);
            }
        }
    }
}

//-----------------------------------------------------------------------------
bool cvPointPickingHelper::isEnabled() const {
    return m_shortcut ? m_shortcut->isEnabled() : false;
}

//-----------------------------------------------------------------------------
void cvPointPickingHelper::getCellNormal(vtkDataSet* dataset,
                                         vtkIdType cellId,
                                         vtkCell* cell,
                                         double normal[3]) {
    // Try to get normal from cell data first
    vtkDataArray* cellNormals = dataset->GetCellData()->GetNormals();
    if (cellNormals && cellId < cellNormals->GetNumberOfTuples()) {
        cellNormals->GetTuple(cellId, normal);
        return;
    }

    // Fallback: compute normal from cell geometry
    int cellType = cell->GetCellType();
    if (cellType == VTK_TRIANGLE || cellType == VTK_QUAD ||
        cellType == VTK_POLYGON) {
        // For polygonal cells, compute normal from first 3 points
        double p0[3], p1[3], p2[3];
        cell->GetPoints()->GetPoint(0, p0);
        cell->GetPoints()->GetPoint(1, p1);
        cell->GetPoints()->GetPoint(2, p2);

        // Compute two edge vectors
        double v1[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
        double v2[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};

        // Cross product
        normal[0] = v1[1] * v2[2] - v1[2] * v2[1];
        normal[1] = v1[2] * v2[0] - v1[0] * v2[2];
        normal[2] = v1[0] * v2[1] - v1[1] * v2[0];

        // Normalize
        double len = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] +
                               normal[2] * normal[2]);
        if (len > 0) {
            normal[0] /= len;
            normal[1] /= len;
            normal[2] /= len;
        }
    }
}

//-----------------------------------------------------------------------------
void cvPointPickingHelper::pickPoint() {
    CVLog::PrintDebug(
            QString("[cvPointPickingHelper::pickPoint] Helper=%1, shortcut "
                    "enabled=%2, contextWidget=%3")
                    .arg((quintptr)this, 0, 16)
                    .arg(m_shortcut && m_shortcut->isEnabled() ? "yes" : "no")
                    .arg((quintptr)m_contextWidget.data(), 0, 16));

    // CRITICAL: Check if shortcut is enabled
    // When using ApplicationShortcut context, shortcuts from all tool instances
    // can be triggered. We must check if this specific instance's shortcut is
    // enabled to ensure only the active (unlocked) tool responds.
    if (!m_shortcut || !m_shortcut->isEnabled()) {
        // This shortcut is disabled (tool is locked), don't process picking
        CVLog::PrintDebug(
                "[cvPointPickingHelper::pickPoint] Shortcut disabled, "
                "skipping");
        return;
    }

    // CRITICAL: Check if context widget is valid and visible
    // This prevents crashes when multiple tool instances exist or tool is being
    // deleted
    if (!m_contextWidget) {
        CVLog::Warning("[cvPointPickingHelper] Context widget is null");
        return;
    }

    if (!m_contextWidget->isVisible()) {
        // Tool dialog is hidden, don't process picking
        CVLog::PrintDebug(
                "[cvPointPickingHelper::pickPoint] Context widget not visible, "
                "skipping");
        return;
    }

    CVLog::PrintDebug("[cvPointPickingHelper::pickPoint] Processing pick...");

    if (!m_interactor || !m_renderer) {
        CVLog::Warning("[cvPointPickingHelper] Interactor or renderer not set");
        return;
    }

    // Get the render window widget
    QWidget* renderWidget = nullptr;
    if (m_interactor->GetRenderWindow()) {
        renderWidget = static_cast<QWidget*>(
                m_interactor->GetRenderWindow()->GetGenericWindowId());
    }

    // Fallback to shortcut's parent widget
    if (!renderWidget) {
        renderWidget = qobject_cast<QWidget*>(m_shortcut ? m_shortcut->parent()
                                                         : nullptr);
    }

    if (!renderWidget) {
        CVLog::Warning("[cvPointPickingHelper] Cannot determine render widget");
        return;
    }

    // ParaView approach: Check if the keypress event actually happened in the
    // window This matches ParaView's pqPointPickingHelper::pickPoint()
    // implementation
    QPointF pos = renderWidget->mapFromGlobal(QCursor::pos());
    QSize sz = renderWidget->size();
    bool outside = pos.x() < 0 || pos.x() > sz.width() || pos.y() < 0 ||
                   pos.y() > sz.height();
    if (outside) {
        CVLog::PrintVerbose(
                "[cvPointPickingHelper] Cursor is outside the render widget");
        return;
    }

    // ParaView approach: Use GetEventPosition() from the interactor
    // This matches ParaView's pqPointPickingHelper::pickPoint() implementation
    // exactly GetEventPosition() returns the position of the last event
    // (keyboard shortcut activation) which is more reliable than QCursor::pos()
    // for keyboard-triggered picking, especially on macOS
    int displayX = 0;
    int displayY = 0;
    const int* eventpos = m_interactor->GetEventPosition();
    if (eventpos && (eventpos[0] >= 0 && eventpos[1] >= 0)) {
        // Use event position from interactor (ParaView style)
        // GetEventPosition() returns coordinates in VTK display space (origin
        // at bottom-left)
        displayX = eventpos[0];
        displayY = eventpos[1];

        // Validate coordinates against render window size
        int* renderSize = m_interactor->GetRenderWindow()->GetSize();
        if (displayX < 0 || displayX >= renderSize[0] || displayY < 0 ||
            displayY >= renderSize[1]) {
            // Event position is out of bounds, fallback to cursor position
            CVLog::PrintDebug(
                    QString("[cvPointPickingHelper] GetEventPosition() out of "
                            "bounds (%1,%2), "
                            "falling back to QCursor::pos()")
                            .arg(displayX)
                            .arg(displayY));
            displayX = static_cast<int>(pos.x());
            displayY = sz.height() - static_cast<int>(pos.y()) - 1;
        }
    } else {
        // Fallback to cursor position if GetEventPosition() is invalid
        // Convert to VTK display coordinates (origin at bottom-left)
        displayX = static_cast<int>(pos.x());
        displayY = sz.height() - static_cast<int>(pos.y()) - 1;
    }

    double position[3] = {0.0, 0.0, 0.0};
    double normal[3] = {0.0, 0.0, 1.0};  // Default normal
    bool pickSuccess = false;

    // ParaView-style hybrid picking strategy:
    // 1. Point mode: Use HardwareSelector (GPU-accelerated, fastest for points)
    // 2. Cell mode: Use PropPicker + CellPicker (accurate surface intersection)
    // 3. Cache results to avoid redundant picks

    // Check cache first
    if (m_selectionCache.valid && m_selectionCache.displayX == displayX &&
        m_selectionCache.displayY == displayY &&
        m_selectionCache.pickOnPoint == m_pickOnPoint) {
        // Use cached result
        std::copy(m_selectionCache.position, m_selectionCache.position + 3,
                  position);
        std::copy(m_selectionCache.normal, m_selectionCache.normal + 3, normal);
        pickSuccess = true;
    } else {
        // Clear cache
        m_selectionCache.valid = false;

        if (m_pickOnPoint) {
            // ===== Point Picking Mode =====
            // Hybrid strategy: HardwareSelector for point clouds (GPU fast),
            //                  PropPicker + PointPicker for meshes (accurate)

            // Try HardwareSelector first (fastest for point clouds)
            if (!m_hardwareSelector) {
                m_hardwareSelector =
                        vtkSmartPointer<vtkHardwareSelector>::New();
            }

            m_hardwareSelector->SetRenderer(m_renderer);
            m_hardwareSelector->SetArea(displayX, displayY, displayX, displayY);
            m_hardwareSelector->SetFieldAssociation(
                    vtkDataObject::FIELD_ASSOCIATION_POINTS);

            vtkSelection* selection = m_hardwareSelector->Select();

            if (selection && selection->GetNumberOfNodes() > 0) {
                vtkSelectionNode* node = selection->GetNode(0);
                if (node) {
                    vtkProp* prop =
                            vtkProp::SafeDownCast(node->GetProperties()->Get(
                                    vtkSelectionNode::PROP()));
                    vtkActor* actor = vtkActor::SafeDownCast(prop);

                    if (actor && actor->GetMapper()) {
                        vtkIdTypeArray* selectionIds =
                                vtkIdTypeArray::SafeDownCast(
                                        node->GetSelectionList());

                        if (selectionIds &&
                            selectionIds->GetNumberOfTuples() > 0) {
                            vtkIdType pointId = selectionIds->GetValue(0);
                            vtkDataSet* dataset =
                                    actor->GetMapper()->GetInput();

                            if (dataset && pointId >= 0 &&
                                pointId < dataset->GetNumberOfPoints()) {
                                dataset->GetPoint(pointId, position);
                                pickSuccess = true;

                                if (m_pickOption != Coordinates) {
                                    vtkDataArray* normals =
                                            dataset->GetPointData()
                                                    ->GetNormals();
                                    if (normals &&
                                        pointId <
                                                normals->GetNumberOfTuples()) {
                                        normals->GetTuple(pointId, normal);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (selection) {
                selection->Delete();
            }

            // Fallback: If HardwareSelector failed (e.g., mesh with no visible
            // vertex at cursor), use PointPicker with PickList for accurate
            // mesh point picking
            if (!pickSuccess) {
                if (!m_propPicker) {
                    m_propPicker = vtkSmartPointer<vtkPropPicker>::New();
                }

                if (m_propPicker->Pick(displayX, displayY, 0, m_renderer)) {
                    vtkProp3D* prop = m_propPicker->GetProp3D();
                    vtkActor* actor = vtkActor::SafeDownCast(prop);

                    if (actor && actor->GetMapper()) {
                        if (!m_pointPicker) {
                            m_pointPicker =
                                    vtkSmartPointer<vtkPointPicker>::New();
                            m_pointPicker->SetTolerance(0.01);
                        }

                        // Limit to this actor only (huge performance boost)
                        m_pointPicker->AddPickList(actor);
                        m_pointPicker->PickFromListOn();

                        if (m_pointPicker->Pick(displayX, displayY, 0,
                                                m_renderer)) {
                            vtkIdType pointId = m_pointPicker->GetPointId();
                            vtkDataSet* dataset =
                                    actor->GetMapper()->GetInput();

                            if (dataset && pointId >= 0 &&
                                pointId < dataset->GetNumberOfPoints()) {
                                dataset->GetPoint(pointId, position);
                                pickSuccess = true;

                                if (m_pickOption != Coordinates) {
                                    vtkDataArray* normals =
                                            dataset->GetPointData()
                                                    ->GetNormals();
                                    if (normals &&
                                        pointId <
                                                normals->GetNumberOfTuples()) {
                                        normals->GetTuple(pointId, normal);
                                    }
                                }
                            }
                        }

                        // Clean up
                        m_pointPicker->InitializePickList();
                        m_pointPicker->PickFromListOff();
                    }
                }
            }

        } else {
            // ===== Surface/Cell Picking Mode =====
            // ParaView strategy: Use CellPicker with PropPicker pre-filter for
            // accuracy

            // Step 1: Fast PropPicker to identify actor
            if (!m_propPicker) {
                m_propPicker = vtkSmartPointer<vtkPropPicker>::New();
            }

            if (m_propPicker->Pick(displayX, displayY, 0, m_renderer)) {
                vtkProp3D* prop = m_propPicker->GetProp3D();
                vtkActor* actor = vtkActor::SafeDownCast(prop);

                if (actor && actor->GetMapper()) {
                    // Step 2: Precise CellPicker with PickList (ParaView
                    // approach)
                    if (!m_cellPicker) {
                        m_cellPicker = vtkSmartPointer<vtkCellPicker>::New();
                        m_cellPicker->SetTolerance(0.005);
                    }

                    // Limit to this actor only (huge performance boost)
                    m_cellPicker->AddPickList(actor);
                    m_cellPicker->PickFromListOn();

                    if (m_cellPicker->Pick(displayX, displayY, 0, m_renderer)) {
                        m_cellPicker->GetPickPosition(position);
                        pickSuccess = true;

                        // Get accurate normal from CellPicker
                        if (m_pickOption != Coordinates) {
                            double* pickedNormal =
                                    m_cellPicker->GetPickNormal();
                            if (pickedNormal) {
                                normal[0] = pickedNormal[0];
                                normal[1] = pickedNormal[1];
                                normal[2] = pickedNormal[2];
                            }
                        }
                    }

                    // Clean up
                    m_cellPicker->InitializePickList();
                    m_cellPicker->PickFromListOff();
                }
            }
        }

        // Update cache
        if (pickSuccess) {
            m_selectionCache.displayX = displayX;
            m_selectionCache.displayY = displayY;
            m_selectionCache.pickOnPoint = m_pickOnPoint;
            std::copy(position, position + 3, m_selectionCache.position);
            std::copy(normal, normal + 3, m_selectionCache.normal);
            m_selectionCache.valid = true;
        }
    }

    if (!pickSuccess) {
        // Pick failed - cursor might not be over any geometry
        return;
    }

    // Validate picked position
    auto isValidVector = [](const double x[3]) {
        return !std::isnan(x[0]) && !std::isnan(x[1]) && !std::isnan(x[2]) &&
               !std::isinf(x[0]) && !std::isinf(x[1]) && !std::isinf(x[2]);
    };

    switch (m_pickOption) {
        case Coordinates:
            if (isValidVector(position)) {
                Q_EMIT pick(position[0], position[1], position[2]);
            } else {
                CVLog::Warning(
                        "[cvPointPickingHelper] Invalid position picked");
            }
            break;

        case Normal:
            if (isValidVector(normal)) {
                Q_EMIT pick(normal[0], normal[1], normal[2]);
            } else {
                CVLog::Warning(
                        "[cvPointPickingHelper] Normal was not available");
            }
            break;

        case CoordinatesAndNormal:
            if (isValidVector(position) && isValidVector(normal)) {
                CVLog::PrintDebug(
                        QString("[cvPointPickingHelper] Picked point: "
                                "(%1, %2, %3), normal: (%4, %5, %6)")
                                .arg(position[0])
                                .arg(position[1])
                                .arg(position[2])
                                .arg(normal[0])
                                .arg(normal[1])
                                .arg(normal[2]));
                Q_EMIT pickNormal(position[0], position[1], position[2],
                                  normal[0], normal[1], normal[2]);
            } else {
                CVLog::Warning(
                        "[cvPointPickingHelper] Position or normal was not "
                        "available");
            }
            break;
    }
}

//-----------------------------------------------------------------------------
void cvPointPickingHelper::clearSelectionCache() {
    m_selectionCache.valid = false;
    m_selectionCache.displayX = -1;
    m_selectionCache.displayY = -1;
    m_selectionCache.id = -1;
}
