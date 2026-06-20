// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "abstractwidgetobserver.h"

#include <vtkInteractorObserver.h>

namespace VtkUtils {

AbstractWidgetObserver::AbstractWidgetObserver(QObject *parent)
    : QObject(parent) {}

AbstractWidgetObserver::~AbstractWidgetObserver() {}

void AbstractWidgetObserver::attach(vtkInteractorObserver *widget) {
    if (widget && widget != m_widget) {
        m_widget = widget;

        // Keep tools responsive while handles are dragged, and also emit the
        // final state when the interaction ends.
        m_widget->AddObserver(vtkCommand::InteractionEvent, this);
        m_widget->AddObserver(vtkCommand::EndInteractionEvent, this);
    }
}

}  // namespace VtkUtils
