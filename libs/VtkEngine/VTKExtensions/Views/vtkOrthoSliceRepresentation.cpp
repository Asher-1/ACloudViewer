// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Orthographic slice representation
// ----------------------------------------------------------------------------

#include "vtkOrthoSliceRepresentation.h"

#include <vtkActor.h>
#include <vtkAppendPolyData.h>
#include <vtkCutter.h>
#include <vtkDataSet.h>
#include <vtkDataSetSurfaceFilter.h>
#include <vtkMapper.h>
#include <vtkNew.h>
#include <vtkOutlineSource.h>
#include <vtkPlane.h>
#include <vtkPlaneCollection.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyDataPlaneClipper.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkScalarsToColors.h>
#include <vtkTriangleFilter.h>
#include <vtkWeakPointer.h>

namespace {

static bool hasValidSliceGeometry(vtkPolyData* pd) {
    return pd && pd->GetNumberOfPoints() > 0 &&
           (pd->GetNumberOfPolys() > 0 || pd->GetNumberOfLines() > 0 ||
            pd->GetNumberOfVerts() > 0);
}

static vtkPolyData* emptySlicePolyData() {
    static vtkNew<vtkPolyData> empty;
    return empty.GetPointer();
}

static vtkSmartPointer<vtkPolyDataPlaneClipper> createSliceClipper(
        vtkPolyData* input, vtkPlane* plane) {
    vtkNew<vtkTriangleFilter> tri;
    tri->SetInputData(input);
    tri->Update();
    auto clipper = vtkSmartPointer<vtkPolyDataPlaneClipper>::New();
    clipper->SetInputConnection(tri->GetOutputPort());
    clipper->SetPlane(plane);
    clipper->SetCapping(true);
    clipper->SetClippingLoops(true);
    clipper->Update();
    return clipper;
}

static vtkPolyData* getSliceCapSurface(vtkPolyDataPlaneClipper* clipper) {
    if (!clipper) return nullptr;
    vtkPolyData* cap = clipper->GetCap();
    if (cap && cap->GetNumberOfPoints() > 0) return cap;
    return nullptr;
}

static vtkPolyData* getSliceSurface(vtkPolyDataPlaneClipper* clipper,
                                    vtkCutter* cutter) {
    if (auto* cap = getSliceCapSurface(clipper)) return cap;
    if (cutter && cutter->GetOutput() &&
        cutter->GetOutput()->GetNumberOfPoints() > 0)
        return cutter->GetOutput();
    return nullptr;
}

static void assignSliceMapper(vtkPolyDataMapper* mapper, vtkPolyData* slice) {
    if (!mapper) return;
    if (hasValidSliceGeometry(slice))
        mapper->SetInputData(slice);
    else
        mapper->SetInputData(emptySlicePolyData());
}

static void copyAppearance(vtkActor* target, vtkActor* source) {
    if (!target || !source || !source->GetMapper()) return;
    auto* mapper = vtkPolyDataMapper::SafeDownCast(target->GetMapper());
    if (!mapper) return;
    vtkSmartPointer<vtkPolyData> input = mapper->GetInput();
    mapper->ShallowCopy(source->GetMapper());
    mapper->SetInputData(input);
    if (auto* lut = source->GetMapper()->GetLookupTable()) {
        vtkSmartPointer<vtkScalarsToColors> owned;
        owned.TakeReference(lut->NewInstance());
        owned->DeepCopy(lut);
        mapper->SetLookupTable(owned);
    }
    vtkNew<vtkPlaneCollection> planes;
    if (auto* sourcePlanes = source->GetMapper()->GetClippingPlanes()) {
        for (int i = 0; i < sourcePlanes->GetNumberOfItems(); ++i) {
            vtkNew<vtkPlane> plane;
            plane->SetOrigin(sourcePlanes->GetItem(i)->GetOrigin());
            plane->SetNormal(sourcePlanes->GetItem(i)->GetNormal());
            planes->AddItem(plane);
        }
    }
    mapper->SetClippingPlanes(planes);
    target->GetProperty()->DeepCopy(source->GetProperty());
    target->SetPickable(source->GetPickable());
}

}  // namespace

struct vtkOrthoSliceRepresentation::Impl {
    vtkWeakPointer<vtkRenderer> orthoRenderers[3];
    vtkWeakPointer<vtkRenderer> perspectiveRenderer;
    vtkSmartPointer<vtkDataSet> input;
    vtkSmartPointer<vtkActor> sourceAppearance;
    vtkSmartPointer<vtkPlane> planes[3];
    vtkSmartPointer<vtkPolyDataPlaneClipper> clippers[3];
    vtkSmartPointer<vtkCutter> cutters[3];
    vtkSmartPointer<vtkActor> actors2D[3];
    vtkSmartPointer<vtkActor> actors3D[3];
    vtkSmartPointer<vtkActor> surfaceActor3D;
    vtkSmartPointer<vtkActor> outlineActor3D;
    vtkSmartPointer<vtkOutlineSource> outlineSource;
    double sliceOrigin[3] = {0, 0, 0};

    static const double kNormals[3][3];
};

const double vtkOrthoSliceRepresentation::Impl::kNormals[3][3] = {
        {0, 1, 0}, {1, 0, 0}, {0, 0, 1}};

vtkOrthoSliceRepresentation::vtkOrthoSliceRepresentation() : d(new Impl) {}

vtkOrthoSliceRepresentation::~vtkOrthoSliceRepresentation() {
    vtkRenderer* renderers[3] = {d->orthoRenderers[0], d->orthoRenderers[1],
                                 d->orthoRenderers[2]};
    removeFromRenderers(renderers, d->perspectiveRenderer);
    delete d;
}

void vtkOrthoSliceRepresentation::setInputPolyData(vtkPolyData* input) {
    setInputData(input);
}

void vtkOrthoSliceRepresentation::setInputData(vtkDataSet* input,
                                               vtkActor* sourceAppearance) {
    vtkRenderer* renderers[3] = {d->orthoRenderers[0], d->orthoRenderers[1],
                                 d->orthoRenderers[2]};
    vtkRenderer* perspective = d->perspectiveRenderer;
    removeFromRenderers(renderers, perspective);
    d->input = input;
    d->sourceAppearance = sourceAppearance;
    for (int axis = 0; axis < 3; ++axis) {
        d->clippers[axis] = nullptr;
        d->cutters[axis] = nullptr;
        d->actors2D[axis] = nullptr;
        d->actors3D[axis] = nullptr;
    }
    d->surfaceActor3D = nullptr;
    d->outlineActor3D = nullptr;
    d->outlineSource = nullptr;
    if (perspective || renderers[0] || renderers[1] || renderers[2])
        addToRenderers(renderers, perspective);
}

void vtkOrthoSliceRepresentation::ensureSurfaceActor() {
    if (!d->input || d->input->GetNumberOfPoints() == 0) return;
    if (d->surfaceActor3D) return;

    vtkNew<vtkPolyDataMapper> mapper;
    vtkSmartPointer<vtkPolyData> surface = vtkPolyData::SafeDownCast(d->input);
    if (!surface) {
        vtkNew<vtkDataSetSurfaceFilter> geometry;
        geometry->SetInputData(d->input);
        geometry->PassThroughPointIdsOn();
        geometry->PassThroughCellIdsOn();
        geometry->Update();
        surface = vtkSmartPointer<vtkPolyData>::New();
        surface->ShallowCopy(geometry->GetOutput());
    }
    mapper->SetInputData(surface);
    if (d->input->GetPointData() && d->input->GetPointData()->GetScalars())
        mapper->ScalarVisibilityOn();
    else
        mapper->ScalarVisibilityOff();

    d->surfaceActor3D = vtkSmartPointer<vtkActor>::New();
    d->surfaceActor3D->SetMapper(mapper);
    const bool hasMesh = surface->GetNumberOfPolys() > 0;
    if (hasMesh) {
        d->surfaceActor3D->GetProperty()->SetRepresentationToSurface();
        d->surfaceActor3D->GetProperty()->SetAmbient(0.2);
        d->surfaceActor3D->GetProperty()->SetDiffuse(0.8);
        d->surfaceActor3D->GetProperty()->LightingOn();
        d->surfaceActor3D->GetProperty()->SetInterpolationToPhong();
        if (!d->input->GetPointData() ||
            !d->input->GetPointData()->GetScalars()) {
            d->surfaceActor3D->GetProperty()->SetColor(0.8, 0.8, 0.8);
        }
    } else {
        d->surfaceActor3D->GetProperty()->SetRepresentationToPoints();
        d->surfaceActor3D->GetProperty()->SetPointSize(2.0);
        d->surfaceActor3D->GetProperty()->SetAmbient(1.0);
        d->surfaceActor3D->GetProperty()->SetDiffuse(0.0);
        d->surfaceActor3D->GetProperty()->LightingOff();
        d->surfaceActor3D->GetProperty()->SetColor(0.8, 0.8, 0.8);
    }
    copyAppearance(d->surfaceActor3D, d->sourceAppearance);
    d->surfaceActor3D->SetVisibility(0);
}

void vtkOrthoSliceRepresentation::ensureOutlineActor() {
    if (!d->input || d->input->GetNumberOfPoints() == 0) return;
    if (d->outlineActor3D) return;

    double bounds[6];
    d->input->GetBounds(bounds);

    d->outlineSource = vtkSmartPointer<vtkOutlineSource>::New();
    d->outlineSource->SetBounds(bounds);

    vtkNew<vtkPolyDataMapper> mapper;
    mapper->SetInputConnection(d->outlineSource->GetOutputPort());
    d->outlineActor3D = vtkSmartPointer<vtkActor>::New();
    d->outlineActor3D->SetMapper(mapper);
    d->outlineActor3D->SetUseBounds(false);
    d->outlineActor3D->GetProperty()->SetColor(1.0, 1.0, 1.0);
    d->outlineActor3D->GetProperty()->SetLineWidth(1.0);
    d->outlineActor3D->GetProperty()->SetRepresentationToWireframe();
    d->outlineActor3D->GetProperty()->LightingOff();
    d->outlineActor3D->GetProperty()->SetAmbient(1.0);
    d->outlineActor3D->GetProperty()->SetDiffuse(0.0);
    d->outlineActor3D->SetVisibility(0);
}

vtkPolyData* vtkOrthoSliceRepresentation::inputPolyData() const {
    return vtkPolyData::SafeDownCast(d->input);
}

vtkDataSet* vtkOrthoSliceRepresentation::inputData() const { return d->input; }

void vtkOrthoSliceRepresentation::restoreSourceAppearance() {
    copyAppearance(d->surfaceActor3D, d->sourceAppearance);
    for (int axis = 0; axis < 3; ++axis) {
        copyAppearance(d->actors2D[axis], d->sourceAppearance);
        copyAppearance(d->actors3D[axis], d->sourceAppearance);
    }
}

void vtkOrthoSliceRepresentation::setSliceOrigin(double x, double y, double z) {
    d->sliceOrigin[0] = x;
    d->sliceOrigin[1] = y;
    d->sliceOrigin[2] = z;
}

void vtkOrthoSliceRepresentation::update() {
    if (!d->input || d->input->GetNumberOfPoints() == 0) return;

    for (int i = 0; i < 3; ++i) {
        if (!d->planes[i]) {
            d->planes[i] = vtkSmartPointer<vtkPlane>::New();
            d->planes[i]->SetNormal(Impl::kNormals[i]);
        }
        d->planes[i]->SetOrigin(d->sliceOrigin);

        auto* poly = vtkPolyData::SafeDownCast(d->input);
        d->clippers[i] = poly && poly->GetNumberOfPolys() > 0
                                 ? createSliceClipper(poly, d->planes[i])
                                 : nullptr;
        if (!d->cutters[i]) {
            d->cutters[i] = vtkSmartPointer<vtkCutter>::New();
            d->cutters[i]->SetInputData(d->input);
            d->cutters[i]->SetCutFunction(d->planes[i]);
        } else {
            d->cutters[i]->SetCutFunction(d->planes[i]);
            d->cutters[i]->Modified();
        }
        d->cutters[i]->Update();

        vtkPolyData* slice = getSliceSurface(d->clippers[i], d->cutters[i]);
        const bool valid = hasValidSliceGeometry(slice);

        if (!d->actors2D[i]) {
            vtkNew<vtkPolyDataMapper> mapper;
            assignSliceMapper(mapper, slice);
            mapper->ScalarVisibilityOn();
            d->actors2D[i] = vtkSmartPointer<vtkActor>::New();
            d->actors2D[i]->SetMapper(mapper);
            d->actors2D[i]->GetProperty()->SetAmbient(1.0);
            d->actors2D[i]->GetProperty()->SetDiffuse(0.0);
            d->actors2D[i]->GetProperty()->LightingOff();
            copyAppearance(d->actors2D[i], d->sourceAppearance);
        } else if (auto* mapper = vtkPolyDataMapper::SafeDownCast(
                           d->actors2D[i]->GetMapper())) {
            assignSliceMapper(mapper, slice);
            mapper->Modified();
        }
        d->actors2D[i]->SetVisibility(valid ? 1 : 0);

        if (!d->actors3D[i]) {
            vtkNew<vtkPolyDataMapper> mapper3D;
            assignSliceMapper(mapper3D, slice);
            mapper3D->ScalarVisibilityOn();
            d->actors3D[i] = vtkSmartPointer<vtkActor>::New();
            d->actors3D[i]->SetMapper(mapper3D);
            d->actors3D[i]->GetProperty()->SetAmbient(1.0);
            d->actors3D[i]->GetProperty()->SetDiffuse(0.0);
            d->actors3D[i]->GetProperty()->LightingOff();
            copyAppearance(d->actors3D[i], d->sourceAppearance);
        } else if (auto* mapper = vtkPolyDataMapper::SafeDownCast(
                           d->actors3D[i]->GetMapper())) {
            assignSliceMapper(mapper, slice);
            mapper->Modified();
        }
        d->actors3D[i]->SetVisibility(valid ? 1 : 0);
    }
}

void vtkOrthoSliceRepresentation::addToRenderers(
        vtkRenderer* orthoRenderers[3], vtkRenderer* perspectiveRenderer) {
    for (int axis = 0; axis < 3; ++axis)
        d->orthoRenderers[axis] = orthoRenderers[axis];
    d->perspectiveRenderer = perspectiveRenderer;
    update();
    ensureSurfaceActor();
    ensureOutlineActor();
    for (int i = 0; i < 3; ++i) {
        if (d->actors2D[i] && orthoRenderers[i])
            orthoRenderers[i]->AddActor(d->actors2D[i]);
        if (d->actors3D[i] && perspectiveRenderer)
            perspectiveRenderer->AddActor(d->actors3D[i]);
    }
    if (d->surfaceActor3D && perspectiveRenderer)
        perspectiveRenderer->AddActor(d->surfaceActor3D);
    if (d->outlineActor3D && perspectiveRenderer)
        perspectiveRenderer->AddActor(d->outlineActor3D);
}

void vtkOrthoSliceRepresentation::removeFromRenderers(
        vtkRenderer* orthoRenderers[3], vtkRenderer* perspectiveRenderer) {
    for (int i = 0; i < 3; ++i) {
        if (d->orthoRenderers[i] == orthoRenderers[i])
            d->orthoRenderers[i] = nullptr;
        if (d->actors2D[i] && orthoRenderers[i])
            orthoRenderers[i]->RemoveActor(d->actors2D[i]);
        if (d->actors3D[i] && perspectiveRenderer)
            perspectiveRenderer->RemoveActor(d->actors3D[i]);
    }
    if (d->perspectiveRenderer == perspectiveRenderer)
        d->perspectiveRenderer = nullptr;
    if (d->surfaceActor3D && perspectiveRenderer)
        perspectiveRenderer->RemoveActor(d->surfaceActor3D);
    if (d->outlineActor3D && perspectiveRenderer)
        perspectiveRenderer->RemoveActor(d->outlineActor3D);
}

vtkActor* vtkOrthoSliceRepresentation::sliceActor2D(PlaneIndex plane) const {
    const int i = static_cast<int>(plane);
    return (i >= 0 && i < 3) ? d->actors2D[i] : nullptr;
}

vtkActor* vtkOrthoSliceRepresentation::sliceActor3D(PlaneIndex plane) const {
    const int i = static_cast<int>(plane);
    return (i >= 0 && i < 3) ? d->actors3D[i] : nullptr;
}

vtkActor* vtkOrthoSliceRepresentation::surfaceActor3D() const {
    return d->surfaceActor3D;
}

vtkActor* vtkOrthoSliceRepresentation::outlineActor3D() const {
    return d->outlineActor3D;
}

bool vtkOrthoSliceRepresentation::hasValidSlice(PlaneIndex plane) const {
    const int i = static_cast<int>(plane);
    if (i < 0 || i >= 3) return false;
    return d->actors2D[i] && d->actors2D[i]->GetVisibility() != 0;
}
