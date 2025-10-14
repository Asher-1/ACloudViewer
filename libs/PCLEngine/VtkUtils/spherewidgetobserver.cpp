// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "spherewidgetobserver.h"

#include <vtkSphereWidget.h>

namespace VtkUtils {

SphereWidgetObserver::SphereWidgetObserver(QObject *parent)
    : AbstractWidgetObserver(parent) {}

void SphereWidgetObserver::Execute(vtkObject *caller,
                                   unsigned long eventId,
                                   void *) {
    vtkSphereWidget *widget = reinterpret_cast<vtkSphereWidget *>(caller);
    if (widget) {
        emit centerChanged(widget->GetCenter());
        emit radiusChanged(widget->GetRadius());
    }
}

}  // namespace VtkUtils
