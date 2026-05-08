// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "actorexporter.h"

#include <vtkActor.h>
#include <vtkDataSetMapper.h>
#include <vtkMapper.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkVRMLExporter.h>

#include "vtkutils.h"

namespace VtkUtils {

ActorExporter::ActorExporter(vtkActor* actor, const QString& file)
    : m_exportFile(file) {
    setAutoDelete(true);

    // Deep-copy geometry on the calling (GUI) thread so that the pool
    // thread never touches the original actor's pipeline.  VTK objects
    // are NOT thread-safe — sharing a vtkActor across threads without
    // external synchronization causes data races.
    if (actor && actor->GetMapper()) {
        actor->GetMapper()->Update();
        auto* input = actor->GetMapper()->GetInput();
        if (input) {
            m_polyData = vtkSmartPointer<vtkPolyData>::New();
            m_polyData->DeepCopy(input);
        }
        if (actor->GetProperty()) {
            m_property = vtkSmartPointer<vtkProperty>::New();
            m_property->DeepCopy(actor->GetProperty());
        }
    }
}

void ActorExporter::run() {
    if (!m_polyData || m_exportFile.isEmpty()) return;

    auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputData(m_polyData);

    auto localActor = vtkSmartPointer<vtkActor>::New();
    localActor->SetMapper(mapper);
    if (m_property) {
        localActor->SetProperty(m_property);
    }

    VTK_CREATE(vtkRenderWindow, renderWindow);
    renderWindow->SetOffScreenRendering(1);
    VTK_CREATE(vtkRenderer, renderer);
    renderer->AddActor(localActor);
    renderWindow->AddRenderer(renderer);

    VTK_CREATE(vtkVRMLExporter, exporter);
    exporter->SetFileName(m_exportFile.toUtf8().data());
    exporter->SetRenderWindow(renderWindow);
    exporter->Write();
}

}  // namespace VtkUtils
