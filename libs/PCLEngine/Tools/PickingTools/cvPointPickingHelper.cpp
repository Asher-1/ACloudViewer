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
#include <vtkCellPicker.h>
#include <vtkDataArray.h>
#include <vtkDataSet.h>
#include <vtkPointData.h>
#include <vtkPointPicker.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>

// QT
#include <QApplication>
#include <QCursor>
#include <QShortcut>
#include <QWidget>
#include <cmath>

//-----------------------------------------------------------------------------
cvPointPickingHelper::cvPointPickingHelper(const QKeySequence& keySequence,
                                           bool pickOnMesh,
                                           QWidget* parent,
                                           PickOption pickOpt)
    : QObject(parent), m_pickOnMesh(pickOnMesh), m_pickOption(pickOpt) {
    if (!parent) {
        CVLog::Warning(
                "[cvPointPickingHelper] Parent widget is null, shortcut may "
                "not work properly");
        return;
    }

    // Use WindowShortcut context so it works when focus is anywhere in the
    // window This is similar to ParaView's pqModalShortcut approach
    m_shortcut = new QShortcut(keySequence, parent);
    m_shortcut->setContext(Qt::WindowShortcut);
    connect(m_shortcut, &QShortcut::activated, this,
            &cvPointPickingHelper::pickPoint);

    // Give focus to the parent widget so shortcuts can be activated
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
void cvPointPickingHelper::pickPoint() {
    // Check if the context widget (measurement tool dialog) is visible
    // This prevents the shortcut from triggering when the tool is closed
    if (m_contextWidget && !m_contextWidget->isVisible()) {
        CVLog::Print("[cvPointPickingHelper] Context widget is not visible");
        return;
    }

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

    // Get cursor position in widget coordinates
    QPoint globalPos = QCursor::pos();
    QPoint localPos = renderWidget->mapFromGlobal(globalPos);

    // Check if cursor is within the widget
    QSize widgetSize = renderWidget->size();
    if (localPos.x() < 0 || localPos.x() > widgetSize.width() ||
        localPos.y() < 0 || localPos.y() > widgetSize.height()) {
        CVLog::Print(
                "[cvPointPickingHelper] Cursor is outside the render widget");
        return;
    }

    // Convert to VTK display coordinates (origin at bottom-left)
    int displayX = localPos.x();
    int displayY = widgetSize.height() - localPos.y() - 1;

    double position[3] = {0.0, 0.0, 0.0};
    double normal[3] = {0.0, 0.0, 1.0};  // Default normal
    bool pickSuccess = false;

    if (m_pickOnMesh) {
        // Use point picker to snap to mesh points (reuse picker)
        if (!m_pointPicker) {
            m_pointPicker = vtkSmartPointer<vtkPointPicker>::New();
            m_pointPicker->SetTolerance(
                    0.01);  // Increased tolerance for easier picking
        }

        if (m_pointPicker->Pick(displayX, displayY, 0, m_renderer)) {
            m_pointPicker->GetPickPosition(position);
            pickSuccess = true;

            // Try to get normal if needed
            if (m_pickOption == Normal ||
                m_pickOption == CoordinatesAndNormal) {
                vtkIdType pointId = m_pointPicker->GetPointId();
                if (pointId >= 0 && m_pointPicker->GetDataSet()) {
                    vtkDataArray* normals = m_pointPicker->GetDataSet()
                                                    ->GetPointData()
                                                    ->GetNormals();
                    if (normals) {
                        normals->GetTuple(pointId, normal);
                    }
                }
            }
        }
    } else {
        // Use cell picker to pick on surface (reuse picker)
        if (!m_cellPicker) {
            m_cellPicker = vtkSmartPointer<vtkCellPicker>::New();
            m_cellPicker->SetTolerance(
                    0.005);  // Increased tolerance for easier picking
        }

        if (m_cellPicker->Pick(displayX, displayY, 0, m_renderer)) {
            m_cellPicker->GetPickPosition(position);
            pickSuccess = true;

            // Get normal from the picked cell
            if (m_pickOption == Normal ||
                m_pickOption == CoordinatesAndNormal) {
                double* pickedNormal = m_cellPicker->GetPickNormal();
                if (pickedNormal) {
                    normal[0] = pickedNormal[0];
                    normal[1] = pickedNormal[1];
                    normal[2] = pickedNormal[2];
                }
            }
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
