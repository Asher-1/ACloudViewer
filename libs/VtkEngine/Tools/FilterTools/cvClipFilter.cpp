// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvClipFilter.h"

#include <VtkUtils/vtkutils.h>
#include <vtkBox.h>
#include <vtkClipDataSet.h>
#include <vtkClipPolyData.h>
#include <vtkDataSetMapper.h>
#include <VTKExtensions/Views/vtkPVLODActor.h>
#include <vtkLookupTable.h>
#include <vtkPlane.h>
#include <vtkPlanes.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderer.h>
#include <vtkSphere.h>

#include "VtkUtils/utils.h"

cvClipFilter::cvClipFilter(QWidget *parent) : cvCutFilter(parent) {
    setWindowTitle(tr("Clip"));
}

cvClipFilter::~cvClipFilter() {}

void cvClipFilter::clearAllActor() { cvCutFilter::clearAllActor(); }

void cvClipFilter::apply() {
    if (!m_dataObject || m_keepMode) return;

    if (!m_preview) {
        if (isValidPolyData()) {
            if (!m_filterActor) {
                VtkUtils::vtkInitOnce(m_filterActor);
                addActor(m_filterActor);
            }
            vtkPolyDataMapper* mapper =
                    vtkPolyDataMapper::SafeDownCast(m_filterActor->GetMapper());
            if (!mapper) {
                auto newMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
                m_filterActor->SetMapper(newMapper);
                mapper = newMapper;
            }
            mapper->SetInputData(vtkPolyData::SafeDownCast(m_dataObject));
        } else if (isValidDataSet()) {
            if (!m_filterActor) {
                VtkUtils::vtkInitOnce(m_filterActor);
                addActor(m_filterActor);
            }
            vtkDataSetMapper* mapper =
                    vtkDataSetMapper::SafeDownCast(m_filterActor->GetMapper());
            if (!mapper) {
                auto newMapper = vtkSmartPointer<vtkDataSetMapper>::New();
                m_filterActor->SetMapper(newMapper);
                mapper = newMapper;
            }
            mapper->SetInputData(vtkDataSet::SafeDownCast(m_dataObject));
        }
        applyDisplayEffect();

        return;
    }

    if (isValidPolyData()) {
        VtkUtils::vtkInitOnce(m_PolyClip);
        m_PolyClip->SetInputData(m_dataObject);

        switch (cutType()) {
            case cvCutFilter::Plane: {
                VTK_CREATE(vtkPlane, plane);
                plane->SetOrigin(m_origin);
                plane->SetNormal(m_normal);
                m_PolyClip->SetClipFunction(plane);
            } break;

            case cvCutFilter::Box: {
                m_PolyClip->SetClipFunction(m_planes);
            } break;

            case cvCutFilter::Sphere: {
                VTK_CREATE(vtkSphere, sphere);
                sphere->SetCenter(m_center);
                sphere->SetRadius(m_radius);
                m_PolyClip->SetClipFunction(sphere);
            } break;
        }

        m_negative ? m_PolyClip->InsideOutOn() : m_PolyClip->InsideOutOff();
        m_PolyClip->Update();

        const bool addFilterActor = !m_filterActor;
        if (!m_filterActor) {
            VtkUtils::vtkInitOnce(m_filterActor);
        }
        vtkPolyDataMapper* mapper =
                vtkPolyDataMapper::SafeDownCast(m_filterActor->GetMapper());
        if (!mapper) {
            auto newMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
            m_filterActor->SetMapper(newMapper);
            mapper = newMapper;
        }
        if (mapper->GetInputConnection(0, 0) != m_PolyClip->GetOutputPort()) {
            mapper->SetInputConnection(m_PolyClip->GetOutputPort());
        }
        if (addFilterActor) {
            addActor(m_filterActor);
        }
    } else if (isValidDataSet()) {
        VtkUtils::vtkInitOnce(m_DataSetClip);
        m_DataSetClip->SetInputData(m_dataObject);

        switch (cutType()) {
            case cvCutFilter::Plane: {
                VTK_CREATE(vtkPlane, plane);
                plane->SetOrigin(m_origin);
                plane->SetNormal(m_normal);
                m_DataSetClip->SetClipFunction(plane);
            } break;

            case cvCutFilter::Box: {
                m_DataSetClip->SetClipFunction(m_planes);
            } break;

            case cvCutFilter::Sphere: {
                VTK_CREATE(vtkSphere, sphere);
                sphere->SetCenter(m_center);
                sphere->SetRadius(m_radius);
                m_DataSetClip->SetClipFunction(sphere);
            } break;
        }

        m_negative ? m_DataSetClip->InsideOutOn()
                   : m_DataSetClip->InsideOutOff();
        m_DataSetClip->Update();

        const bool addFilterActor = !m_filterActor;
        if (!m_filterActor) {
            VtkUtils::vtkInitOnce(m_filterActor);
        }
        vtkDataSetMapper* mapper =
                vtkDataSetMapper::SafeDownCast(m_filterActor->GetMapper());
        if (!mapper) {
            auto newMapper = vtkSmartPointer<vtkDataSetMapper>::New();
            m_filterActor->SetMapper(newMapper);
            mapper = newMapper;
        }
        if (mapper->GetInputConnection(0, 0) != m_DataSetClip->GetOutputPort()) {
            mapper->SetInputConnection(m_DataSetClip->GetOutputPort());
        }
        if (addFilterActor) {
            addActor(m_filterActor);
        }
    }

    applyDisplayEffect();
}

ccHObject *cvClipFilter::getOutput() {
    bool old_preview = m_preview;
    m_preview = true;
    apply();
    m_preview = old_preview;

    if (isValidPolyData()) {
        if (!m_PolyClip) {
            return nullptr;
        }

        // set exported polydata
        setResultData(m_PolyClip->GetOutput());

        // enable Clipped Output
        m_PolyClip->GenerateClippedOutputOn();

        // update remaining part
        m_negative ? m_PolyClip->InsideOutOn() : m_PolyClip->InsideOutOff();
        m_PolyClip->Update();
        m_dataObject->DeepCopy(m_PolyClip->GetClippedOutput());

        // disable Clipped Output
        m_PolyClip->GenerateClippedOutputOff();

    } else if (isValidDataSet()) {
        if (!m_DataSetClip) {
            return nullptr;
        }

        // set exported DataSet
        setResultData((vtkDataObject *)m_DataSetClip->GetOutput());

        // enable Clipped Output
        m_DataSetClip->GenerateClippedOutputOn();

        // update remaining part
        m_negative ? m_DataSetClip->InsideOutOn()
                   : m_DataSetClip->InsideOutOff();
        m_DataSetClip->Update();
        m_dataObject->DeepCopy(
                (vtkDataObject *)m_DataSetClip->GetClippedOutput());

        // disable Clipped Output
        m_DataSetClip->GenerateClippedOutputOff();
    }

    // export clipping data
    return cvGenericFilter::getOutput();
}
