// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "distancewidgetobserver.h"

#include <vtkDistanceRepresentation.h>
#include <vtkDistanceWidget.h>
#include <vtkLineRepresentation.h>
#include <vtkLineWidget2.h>
#include <vtkMath.h>

namespace VtkUtils {

DistanceWidgetObserver::DistanceWidgetObserver(QObject* parent)
    : AbstractWidgetObserver(parent) {}

void DistanceWidgetObserver::Execute(vtkObject* caller,
                                     unsigned long eventId,
                                     void* callData) {
    Q_UNUSED(eventId)
    Q_UNUSED(callData)

    // Try vtkLineWidget2 first (for cvConstrainedDistanceWidget)
    vtkLineWidget2* lineWidget = vtkLineWidget2::SafeDownCast(caller);
    if (lineWidget) {
        vtkWidgetRepresentation* rep = lineWidget->GetRepresentation();
        vtkLineRepresentation* lineRep =
                vtkLineRepresentation::SafeDownCast(rep);
        if (lineRep) {
            double worldPot1[3];
            double worldPot2[3];
            double displayPot1[3];
            double displayPot2[3];

            lineRep->GetPoint1WorldPosition(worldPot1);
            lineRep->GetPoint2WorldPosition(worldPot2);
            lineRep->GetPoint1DisplayPosition(displayPot1);
            lineRep->GetPoint2DisplayPosition(displayPot2);

            // Calculate distance (following ParaView)
            double distance =
                    sqrt(vtkMath::Distance2BetweenPoints(worldPot1, worldPot2));

            emit distanceChanged(distance);
            emit worldPoint1Changed(worldPot1);
            emit worldPoint2Changed(worldPot2);
            emit displayPoint1Changed(displayPot1);
            emit displayPoint2Changed(displayPot2);
        }
        return;
    }

    // Fallback to vtkDistanceWidget (for backward compatibility)
    vtkDistanceWidget* widget = vtkDistanceWidget::SafeDownCast(caller);
    if (widget) {
        vtkWidgetRepresentation* rep = widget->GetRepresentation();
        vtkDistanceRepresentation* distRep =
                vtkDistanceRepresentation::SafeDownCast(rep);
        if (distRep) {
            double worldPot1[3];
            double worldPot2[3];
            double displayPot1[3];
            double displayPot2[3];

            distRep->GetPoint1WorldPosition(worldPot1);
            distRep->GetPoint2WorldPosition(worldPot2);
            distRep->GetPoint1DisplayPosition(displayPot1);
            distRep->GetPoint2DisplayPosition(displayPot2);

            emit distanceChanged(distRep->GetDistance());
            emit worldPoint1Changed(worldPot1);
            emit worldPoint2Changed(worldPot2);
            emit displayPoint1Changed(displayPot1);
            emit displayPoint2Changed(displayPot2);
        }
    }
}

}  // namespace VtkUtils
