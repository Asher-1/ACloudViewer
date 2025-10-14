// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "linewidgetobserver.h"

#include <vtkLineWidget.h>

namespace VtkUtils {

LineWidgetObserver::LineWidgetObserver(QObject *parent)
    : AbstractWidgetObserver(parent) {}

void LineWidgetObserver::Execute(vtkObject *caller,
                                 unsigned long eventId,
                                 void *) {
    vtkLineWidget *widget = reinterpret_cast<vtkLineWidget *>(caller);
    if (widget) emit pointsChanged(widget->GetPoint1(), widget->GetPoint2());
}

}  // namespace VtkUtils
