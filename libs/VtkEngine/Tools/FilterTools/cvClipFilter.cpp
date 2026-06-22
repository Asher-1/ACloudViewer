// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvClipFilter.h"

#include <VTKExtensions/Views/vtkPVLODActor.h>
#include <VtkUtils/vtkutils.h>
#include <vtkBox.h>
#include <vtkClipDataSet.h>
#include <vtkClipPolyData.h>
#include <vtkDataSetMapper.h>
#include <vtkLookupTable.h>
#include <vtkMapper.h>
#include <vtkNonMergingPointLocator.h>
#include <vtkPlane.h>
#include <vtkPlanes.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkSphere.h>

#include "VtkUtils/utils.h"

cvClipFilter::cvClipFilter(QWidget* parent) : cvCutFilter(parent) {
    setWindowTitle(tr("Clip"));
}

cvClipFilter::~cvClipFilter() {}

void cvClipFilter::clearAllActor() {
    if (m_filterActor && m_filterActor->GetMapper()) {
        clearMapperClipping(m_filterActor->GetMapper());
    }
    cvCutFilter::clearAllActor();
}

void cvClipFilter::clearMapperClipping(vtkMapper* mapper) {
    if (mapper) {
        mapper->RemoveAllClippingPlanes();
    }
}

bool cvClipFilter::useMapperClippingPreview() const {
    // Always prefer GPU-based mapper clipping planes — they are persistent
    // across renders and immune to actor property resets from the entity
    // redraw pipeline. Geometry clip is only needed for Sphere (no plane
    // equivalent) and as fallback when mapper clipping is not applicable.
    return true;
}

void cvClipFilter::updateFilterActorMapperClipping(vtkMapper* mapper) {
    if (!mapper) return;

    clearMapperClipping(mapper);

    switch (cutType()) {
        case cvCutFilter::Plane: {
            VTK_CREATE(vtkPlane, plane);
            plane->SetOrigin(m_origin);
            if (m_negative) {
                plane->SetNormal(-m_normal[0], -m_normal[1], -m_normal[2]);
            } else {
                plane->SetNormal(m_normal);
            }
            mapper->AddClippingPlane(plane);
        } break;

        case cvCutFilter::Box: {
            if (!m_planes) break;
            const int numPlanes = m_planes->GetNumberOfPlanes();
            for (int i = 0; i < numPlanes; ++i) {
                VTK_CREATE(vtkPlane, plane);
                m_planes->GetPlane(i, plane);
                if (m_negative) {
                    double normal[3];
                    plane->GetNormal(normal);
                    plane->SetNormal(-normal[0], -normal[1], -normal[2]);
                }
                mapper->AddClippingPlane(plane);
            }
        } break;

        case cvCutFilter::Sphere:
        default:
            break;
    }
}

void cvClipFilter::applyMapperClippingPreview() {
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

    vtkPolyData* polydata = vtkPolyData::SafeDownCast(m_dataObject);
    if (!polydata) return;

    mapper->SetInputData(polydata);

    if (m_modelActor) {
        vtkMapper* modelMapper = m_modelActor->GetMapper();
        if (modelMapper) {
            mapper->SetLookupTable(modelMapper->GetLookupTable());
            double range[2];
            modelMapper->GetScalarRange(range);
            mapper->SetScalarRange(range);
            mapper->SetScalarMode(modelMapper->GetScalarMode());
            mapper->SetColorMode(modelMapper->GetColorMode());
            if (modelMapper->GetScalarVisibility()) {
                mapper->ScalarVisibilityOn();
            } else {
                mapper->ScalarVisibilityOff();
            }
            if (auto* dsMapper = vtkDataSetMapper::SafeDownCast(modelMapper)) {
                mapper->SetInterpolateScalarsBeforeMapping(
                        dsMapper->GetInterpolateScalarsBeforeMapping());
            }
        }
        m_filterActor->GetProperty()->SetPointSize(
                m_modelActor->GetProperty()->GetPointSize());
    }

    updateFilterActorMapperClipping(mapper);
    mapper->Modified();
}

void cvClipFilter::runGeometryClip() {
    if (!m_dataObject) return;

    if (isValidPolyData()) {
        VtkUtils::vtkInitOnce(m_PolyClip);
        m_PolyClip->SetInputData(m_dataObject);

        // Prevent point merging at clip boundaries: Cc2Vtk creates unique
        // vertices per triangle, so adjacent triangles with different materials
        // must not share interpolated clip-boundary points (their UVs differ).
        auto locator = vtkSmartPointer<vtkNonMergingPointLocator>::New();
        m_PolyClip->SetLocator(locator);

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

        auto dsLocator = vtkSmartPointer<vtkNonMergingPointLocator>::New();
        m_DataSetClip->SetLocator(dsLocator);

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
        if (mapper->GetInputConnection(0, 0) !=
            m_DataSetClip->GetOutputPort()) {
            mapper->SetInputConnection(m_DataSetClip->GetOutputPort());
        }
        if (addFilterActor) {
            addActor(m_filterActor);
        }
    }
}

void cvClipFilter::apply() {
    if (!m_dataObject || m_keepMode) return;

    const bool mapperClipPreview =
            useMapperClippingPreview() && cutType() != Sphere;

    if (mapperClipPreview) {
        applyMapperClippingPreview();
        applyDisplayEffect();
        return;
    }

    if (m_filterActor && m_filterActor->GetMapper()) {
        clearMapperClipping(m_filterActor->GetMapper());
    }

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

    runGeometryClip();
    applyDisplayEffect();
}

ccHObject* cvClipFilter::getOutput() {
    runGeometryClip();

    if (isValidPolyData()) {
        if (!m_PolyClip) {
            return nullptr;
        }

        // Deep-copy the clipped result BEFORE re-running the filter with
        // GenerateClippedOutput. The second Update() can modify the output
        // data object in-place, corrupting the result we want to export.
        auto exportedData = vtkSmartPointer<vtkPolyData>::New();
        exportedData->DeepCopy(m_PolyClip->GetOutput());
        setResultData(exportedData);

        // enable Clipped Output to obtain the remaining part
        m_PolyClip->GenerateClippedOutputOn();
        m_negative ? m_PolyClip->InsideOutOn() : m_PolyClip->InsideOutOff();
        m_PolyClip->Update();
        m_dataObject->DeepCopy(m_PolyClip->GetClippedOutput());
        m_PolyClip->GenerateClippedOutputOff();

    } else if (isValidDataSet()) {
        if (!m_DataSetClip) {
            return nullptr;
        }

        // For dataset path, deep-copy the output before re-running the
        // filter. vtkClipDataSet outputs vtkUnstructuredGrid; the base
        // setResultData expects vtkPolyData, so we pass through directly
        // (pre-existing limitation for non-polydata datasets).
        setResultData((vtkDataObject*)m_DataSetClip->GetOutput());

        m_DataSetClip->GenerateClippedOutputOn();
        m_negative ? m_DataSetClip->InsideOutOn()
                   : m_DataSetClip->InsideOutOff();
        m_DataSetClip->Update();
        m_dataObject->DeepCopy(
                (vtkDataObject*)m_DataSetClip->GetClippedOutput());
        m_DataSetClip->GenerateClippedOutputOff();
    }

    // export clipping data
    return cvGenericFilter::getOutput();
}
