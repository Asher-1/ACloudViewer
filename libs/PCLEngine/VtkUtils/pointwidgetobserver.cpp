// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pointwidgetobserver.h"

#include <vtkPointWidget.h>

namespace VtkUtils {

PointWidgetObserver::PointWidgetObserver(QObject* parent)
    : AbstractWidgetObserver(parent) {}

void PointWidgetObserver::Execute(vtkObject* caller,
                                  unsigned long eventId,
                                  void* callData) {
    Q_UNUSED(eventId)
    Q_UNUSED(callData)

    vtkPointWidget* widget = reinterpret_cast<vtkPointWidget*>(caller);
    if (widget) emit positionChanged(widget->GetPosition());
}

}  // namespace VtkUtils
