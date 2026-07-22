// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "vtkBoxWidgetCustomCallback.h"

#include <vtkActor.h>
#include <vtkBoxRepresentation.h>
#include <vtkBoxWidget.h>
#include <vtkTransform.h>

vtkBoxWidgetCustomCallback *vtkBoxWidgetCustomCallback::New() {
    return new vtkBoxWidgetCustomCallback;
}

void vtkBoxWidgetCustomCallback::SetActor(vtkSmartPointer<vtkActor> actor) {
    m_actor = actor;
}

void vtkBoxWidgetCustomCallback::Execute(vtkObject *caller,
                                         unsigned long,
                                         void *) {
    if (m_preview) {
        // Cast the callback caller pointer to vtkBoxWidget2
        vtkSmartPointer<vtkBoxWidget> boxWidget =
                vtkBoxWidget::SafeDownCast(caller);
        // vtkSmartPointer<vtkBoxWidget2>
        // boxWidget=reinterpret_cast<vtkBoxWidget2>(caller); this cast is
        // invalid; vtkBoxWidget works
        vtkSmartPointer<vtkTransform> t = vtkSmartPointer<vtkTransform>::New();
        // Store the box widget transform matrix in t
        //  vtkBoxRepresentation::SafeDownCast(boxWidget->GetRepresentation())->GetTransform(t);
        boxWidget->GetTransform(t);
        // this->m_actor->SetUserTransform(t);
    }
}
