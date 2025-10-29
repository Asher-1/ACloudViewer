// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "clipwindow.h"

#include <VtkUtils/vtkutils.h>
#include <VtkUtils/vtkwidget.h>
#include <vtkBox.h>
#include <vtkClipDataSet.h>
#include <vtkClipPolyData.h>
#include <vtkDataSetMapper.h>
#include <vtkLODActor.h>
#include <vtkLookupTable.h>
#include <vtkPlane.h>
#include <vtkPlanes.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderer.h>
#include <vtkSphere.h>

#include <QDebug>

#include "VtkUtils/utils.h"
#include "ui_cutconfig.h"
#include "ui_generalfilterwindow.h"

ClipWindow::ClipWindow(QWidget *parent) : CutWindow(parent) {
    setWindowTitle(tr("Clip"));
}

ClipWindow::~ClipWindow() {}

void ClipWindow::apply() {
    if (!m_dataObject) return;

    if (isValidPolyData()) {
        VTK_CREATE(vtkClipPolyData, clip);
        clip->SetInputData(m_dataObject);

        switch (cutType()) {
            case CutWindow::Plane: {
                VTK_CREATE(vtkPlane, plane);
                plane->SetOrigin(m_origin);
                plane->SetNormal(m_normal);
                clip->SetClipFunction(plane);
            } break;

            case CutWindow::Box: {
                clip->SetClipFunction(m_planes);
            } break;

            case CutWindow::Sphere: {
                VTK_CREATE(vtkSphere, sphere);
                sphere->SetCenter(m_center);
                sphere->SetRadius(m_radius);
                clip->SetClipFunction(sphere);
            } break;
        }
        clip->SetInsideOut(-1);
        clip->Update();

        setResultData(clip->GetOutput());
        VtkUtils::vtkInitOnce(m_filterActor);

        VTK_CREATE(vtkPolyDataMapper, mapper);
        mapper->SetInputConnection(clip->GetOutputPort());
        m_filterActor->SetMapper(mapper);
    } else if (isValidDataSet()) {
        VTK_CREATE(vtkClipDataSet, clip);
        clip->SetInputData(m_dataObject);

        switch (cutType()) {
            case CutWindow::Plane: {
                VTK_CREATE(vtkPlane, plane);
                plane->SetOrigin(m_origin);
                plane->SetNormal(m_normal);
                clip->SetClipFunction(plane);
            } break;

            case CutWindow::Box: {
                VTK_CREATE(vtkBox, box);
            } break;

            case CutWindow::Sphere: {
                VTK_CREATE(vtkSphere, sphere);
                sphere->SetCenter(m_center);
                sphere->SetRadius(m_radius);
                clip->SetClipFunction(sphere);
            } break;
        }

        VtkUtils::vtkInitOnce(m_filterActor);
        setResultData((vtkDataObject *)clip->GetOutput());

        VTK_CREATE(vtkDataSetMapper, mapper);
        mapper->SetInputConnection(clip->GetOutputPort());
        m_filterActor->SetMapper(mapper);
    }

    vtkSmartPointer<vtkLookupTable> lut =
            createLookupTable(scalarMin(), scalarMax());
    m_filterActor->GetMapper()->SetLookupTable(lut);

    m_vtkWidget->defaultRenderer()->AddActor(m_filterActor);
    m_vtkWidget->update();
    applyDisplayEffect();
}
