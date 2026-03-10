// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "anglewidgetobserver.h"

#include <vtkAngleRepresentation2D.h>
#include <vtkAngleRepresentation3D.h>
#include <vtkAngleWidget.h>
#include <vtkMath.h>

// CV_DB_LIB
#include <CVLog.h>

namespace VtkUtils {

AngleWidgetObserver::AngleWidgetObserver(QObject* parent)
    : AbstractWidgetObserver(parent) {}

void AngleWidgetObserver::Execute(vtkObject* caller,
                                  unsigned long eventId,
                                  void* callData) {
    Q_UNUSED(eventId)
    Q_UNUSED(callData)

    vtkAngleWidget* widget = reinterpret_cast<vtkAngleWidget*>(caller);
    if (widget) {
        vtkAngleRepresentation* angleRep = vtkAngleRepresentation::SafeDownCast(
                widget->GetRepresentation());

        if (!angleRep) {
            CVLog::Warning("[AngleWidgetObserver] Execute: angleRep is null");
            return;
        }

        // Check if it's 2D or 3D representation
        vtkAngleRepresentation2D* rep2D =
                vtkAngleRepresentation2D::SafeDownCast(angleRep);

        double worldPot1[3];
        double worldPot2[3];
        double worldCenter[3];
        double displayPot1[3];
        double displayPot2[3];
        double displayCenter[3];

        angleRep->GetPoint1WorldPosition(worldPot1);
        angleRep->GetPoint2WorldPosition(worldPot2);
        angleRep->GetCenterWorldPosition(worldCenter);
        angleRep->GetPoint1DisplayPosition(displayPot1);
        angleRep->GetPoint2DisplayPosition(displayPot2);
        angleRep->GetCenterDisplayPosition(displayCenter);

        // IMPORTANT: vtkAngleRepresentation2D::GetAngle() returns DEGREES
        //            vtkAngleRepresentation3D::GetAngle() returns RADIANS
        double angleDegrees = 0.0;
        double rawAngle = angleRep->GetAngle();
        if (rep2D) {
            // 2D representation returns degrees directly
            angleDegrees = rawAngle;
        } else {
            // 3D representation returns radians, convert to degrees
            angleDegrees = vtkMath::DegreesFromRadians(rawAngle);
        }

        emit angleChanged(angleDegrees);
        emit worldPoint1Changed(worldPot1);
        emit worldPoint2Changed(worldPot2);
        emit worldCenterChanged(worldCenter);
        emit displayPoint1Changed(displayPot1);
        emit displayPoint2Changed(displayPot2);
        emit displayCenterChanged(displayCenter);
    }
}

}  // namespace VtkUtils
